"""JAX-accelerated backend for batch model fitting.

Wraps the autodiff Newton–Raphson logistic solver and ``jnp.linalg``
OLS path behind the :class:`~._backends.BackendProtocol` interface.

Architecture
~~~~~~~~~~~~
The module is structured in two layers:

1. **Solver helper functions** (module-level, inside ``if _CAN_IMPORT_JAX``):
   Pure-JAX JIT-compiled functions for NLL, gradient, Hessian, and
   Newton–Raphson solvers for each GLM family.  These are the
   performance-critical kernels compiled to XLA.

2. **JaxBackend class** (``BackendProtocol`` implementation):
   Thin wrapper that converts NumPy → JAX arrays at the boundary,
   dispatches to the solver helpers via ``jax.vmap``, and converts
   JAX arrays back to NumPy on return.  Callers never touch JAX
   types directly.

NumPy ↔ JAX boundary convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All public methods accept NumPy arrays and return NumPy arrays.
JAX arrays are materialised at the method boundary:

* **Inbound:** ``jnp.array(X, dtype=jnp.float64)`` — creates a
  float64 JAX DeviceArray.  The float64 cast is explicit because
  JAX defaults to float32 (even with ``jax_enable_x64``, the
  input dtype matters for type promotion).
* **Outbound:** ``np.asarray(result)`` — zero-copy when backend is
  CPU, triggers device-to-host transfer on GPU.

Float64 rationale
~~~~~~~~~~~~~~~~~
All arithmetic uses **float64** for numerical reliability on
ill-conditioned Hessians, matching statsmodels and scipy.  On Apple
Silicon this routes through CPU Accelerate BLAS rather than Metal
(which lacks float64 support), with no measurable overhead for the
small matrices typical of GLM regression.

The Newton–Raphson solver requires float64 because the gradient
noise floor is κ(H) × ε, where κ(H) is the Hessian condition
number and ε is machine epsilon.  In float32 (~ 6e-8) with
κ ≈ 10 000, the noise floor is ~ 6e-4 — above any reasonable
tolerance.  Float64 (ε ~ 1e-16) lowers the floor to ~ 1e-12.

Graceful degradation
~~~~~~~~~~~~~~~~~~~~
If JAX is not installed, the :class:`JaxBackend` can still be
instantiated (for introspection) but ``is_available`` returns
``False`` and :func:`resolve_backend` will raise ``ImportError``
when this backend is explicitly requested.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..families import _augment_intercept

if TYPE_CHECKING:
    import jax

# ------------------------------------------------------------------ #
# Solver defaults
# ------------------------------------------------------------------ #

_DEFAULT_TOL: float = 1e-8
"""Default convergence tolerance for Newton–Raphson.

With float64 arithmetic the gradient noise floor is
``κ(H) × ε_f64 ≈ κ(H) × 1e-16``, so tolerances down to ~1e-12
are achievable even on ill-conditioned Hessians (κ ≈ 10,000).
The value 1e-8 matches statsmodels' IRLS default and provides a
comfortable margin above machine epsilon.
"""

_MIN_DAMPING: float = 1e-8
"""Singularity guard added to the Hessian diagonal.

Prevents ``jnp.linalg.solve`` from producing NaN / Inf steps when
the Hessian ``X'WX`` is exactly singular (e.g. a constant column,
perfect separation, or rank-deficient design).  The value is small
enough to leave well-conditioned solves unaffected (typical Hessian
eigenvalues are O(n)) while still stabilising singular directions.

This is *not* ridge regularisation — it does not modify the
objective function, so the solver still targets the unregularised
MLE.  Only the linear-algebra solve is protected.
"""

# ------------------------------------------------------------------ #
# Optional JAX import
# ------------------------------------------------------------------ #
#
# The try/except here mirrors the pattern from the v0.2.0 _jax.py
# module.  If JAX is absent, the module loads successfully but the
# JIT-compiled helpers are not defined; ``is_available`` returns
# ``False`` and resolve_backend() gates on that.

try:
    import jax

    # Enable 64-bit floating point before any array creation.
    # The Newton–Raphson solver requires float64 to converge
    # reliably on ill-conditioned Hessians (κ > 1000).  In float32
    # the gradient noise floor is κ(H) × ε ≈ 6e-4, which exceeds
    # any reasonable tolerance.  Float64 lowers the floor to ~1e-12.
    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    from jax import grad, hessian, jit, vmap

    _CAN_IMPORT_JAX = True
except ImportError:
    _CAN_IMPORT_JAX = False


# ------------------------------------------------------------------ #
# JAX helper functions (defined only when JAX is importable)
# ------------------------------------------------------------------ #
#
# All solver helpers below live inside the ``if _CAN_IMPORT_JAX``
# guard so the module can be imported (and introspected) even when
# JAX is absent.  Each family follows the same pattern:
#
#   1. NLL function (→ scalar) — the objective to minimise.
#   2. Gradient function (→ vector) — ∇_β NLL.
#   3. Hessian function (→ matrix) — H = ∇²_β NLL.
#   4. Newton–Raphson solver — iterates β ← β − H⁻¹g using
#      ``jax.lax.while_loop`` for dynamic early exit.
#
# For families with a canonical link (logistic, Poisson) the
# gradient and Hessian are coded analytically for clarity and
# to avoid double-compilation through ``jax.grad(jax.grad(...))``.
# For ordinal and multinomial, ``jax.grad`` and ``jax.hessian``
# are used because the closed-form derivatives are unwieldy.
# ------------------------------------------------------------------ #

if _CAN_IMPORT_JAX:
    # ============================================================== #
    # Generic static-damping Newton solver (shared by all GLM families)
    # ============================================================== #
    #
    # Every GLM family (logistic, Poisson, NB, ordinal, multinomial)
    # uses the identical ``jax.lax.while_loop`` body: damped Newton
    # step, triple convergence check (gradient / step / relative NLL),
    # gated on finite parameters.  The only differences are:
    #
    #   - Which NLL / gradient / Hessian functions to call
    #   - How to initialise β₀
    #
    # ``_newton_solve_static`` factors out this shared logic.  Each
    # family's ``_make_*_solver`` becomes a thin wrapper that
    # constructs the closures and β₀, then delegates here.
    #
    # This solver uses **static** damping (``min_damping * I``) and
    # is correct for convex objectives (all GLM families with
    # canonical or standard links).  For the **non-convex** REML /
    # Laplace NLL, see ``_reml_newton_solve`` which uses adaptive
    # Levenberg–Marquardt–Nielsen damping instead.
    # -------------------------------------------------------------- #

    def _newton_solve_static(
        nll_fn: Callable[[jnp.ndarray], jnp.ndarray],
        grad_fn: Callable[[jnp.ndarray], jnp.ndarray],
        hess_fn: Callable[[jnp.ndarray], jnp.ndarray],
        beta_init: jnp.ndarray,
        max_iter: int,
        tol: float,
        min_damping: float = _MIN_DAMPING,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r"""Damped Newton–Raphson solver via ``jax.lax.while_loop``.

        Shared by all GLM families.  Uses a static damping
        ``min_damping * I`` added to the Hessian as a singularity
        guard — this does *not* modify the objective, only the
        linear-algebra solve.

        Convergence is checked via three criteria (OR):

        * **Gradient criterion** — :math:`\|g\|_\infty < \tau`
        * **Step criterion** — :math:`\|\Delta\beta\|_\infty < \tau`
        * **Relative NLL change** —
          :math:`|\Delta f| / \max(|f|, 1) < \tau`

        All criteria are gated on finite parameters.

        Args:
            nll_fn: ``β → scalar`` NLL (JIT'd or JIT-compatible).
            grad_fn: ``β → (p,)`` gradient (JIT'd or JIT-compatible).
            hess_fn: ``β → (p, p)`` Hessian (JIT'd or JIT-compatible).
            beta_init: Initial parameter vector ``(p,)``.
            max_iter: Maximum Newton iterations.
            tol: Convergence threshold.
            min_damping: Diagonal singularity guard.

        Returns:
            ``(beta, nll, converged)`` — final parameters, NLL,
            and scalar bool convergence flag.
        """
        n_params = beta_init.shape[0]
        damping_matrix = min_damping * jnp.eye(n_params, dtype=jnp.float64)

        # State: (iteration, beta, nll_prev, converged)
        init_state = (
            jnp.array(0),
            beta_init,
            jnp.array(jnp.inf, dtype=jnp.float64),
            jnp.array(False),
        )

        def cond(
            state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> jax.Array:
            i, _beta, _nll, converged = state
            return (i < max_iter) & (~converged)

        def body(
            state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            i, beta, nll_prev, _ = state

            g = grad_fn(beta)
            H = hess_fn(beta)
            H_damped = H + damping_matrix
            beta_new = beta - jnp.linalg.solve(H_damped, g)

            nll_new = nll_fn(beta_new)

            # Three convergence criteria (OR), gated on finite beta.
            grad_small = jnp.max(jnp.abs(g)) < tol
            step_small = jnp.max(jnp.abs(beta_new - beta)) < tol
            nll_rel = jnp.abs(nll_new - nll_prev) / jnp.maximum(
                jnp.maximum(jnp.abs(nll_prev), jnp.abs(nll_new)),
                1.0,
            )
            func_small = nll_rel < tol

            converged = (grad_small | step_small | func_small) & jnp.all(
                jnp.isfinite(beta_new)
            )
            return (i + 1, beta_new, nll_new, converged)

        _, beta_final, nll_final, converged = jax.lax.while_loop(cond, body, init_state)
        return beta_final, nll_final, converged

    # ============================================================== #
    # Fisher information / standard-error helpers
    # ============================================================== #
    #
    # After the Newton solver converges, the **observed Fisher
    # information** I(β̂) = H(β̂) (the Hessian of the NLL evaluated
    # at the MLE) gives Cov(β̂) ≈ I(β̂)⁻¹.  Standard errors are
    # SE(β̂_j) = √[I(β̂)⁻¹]_jj.
    #
    # These helpers are used by ``classical_p_values()`` on each
    # family to avoid refitting via statsmodels.  The batch path
    # (``_batch_shared_X`` etc.) does NOT use these — it discards
    # the Hessian to keep the ``(B, p, p)`` tensor from
    # materialising.
    #
    # Ref: Efron & Hinkley (1978), "Assessing the accuracy of the
    # maximum likelihood estimator: observed versus expected
    # Fisher information", Biometrika 65(3), 457–487.
    # -------------------------------------------------------------- #

    def _fisher_information_se(
        hess_fn: Callable[[jnp.ndarray], jnp.ndarray],
        beta: jnp.ndarray,
    ) -> np.ndarray:
        r"""Standard errors from the observed Fisher information.

        Evaluates the Hessian of the NLL at the converged β̂ and
        inverts it:

        .. math::
            \text{SE}(\hat\beta_j) = \sqrt{[H(\hat\beta)^{-1}]_{jj}}

        Args:
            hess_fn: ``β → (p, p)`` Hessian of the NLL.
            beta: Converged parameter vector ``(p,)``.

        Returns:
            Standard errors ``(p,)`` as a NumPy array.
        """
        H = hess_fn(beta)
        # Hessian may be singular at convergence boundary (e.g. perfect
        # separation); pinv returns finite but inflated SEs.
        # JAX's linalg.solve does not raise on singular matrices — it
        # silently returns NaN/Inf — so we check finiteness explicitly.
        cov = jnp.linalg.solve(H, jnp.eye(H.shape[0]))
        if not jnp.all(jnp.isfinite(cov)):
            cov = jnp.linalg.pinv(H)
        return np.asarray(jnp.sqrt(jnp.abs(jnp.diag(cov))))

    # ============================================================== #
    # Per-observation NLL functions (for sandwich SEs & Cook's D)
    # ============================================================== #
    #
    # Each ``_*_nll_per_obs`` function returns an ``(n,)`` vector of
    # per-observation contributions to the NLL, i.e. the same as the
    # ``_*_nll`` functions but **without** the final ``jnp.sum()``.
    #
    # Used by:
    #   - ``_sandwich_se()``: score matrix via ``vmap(grad(nll_i))``
    #   - ``_autodiff_cooks_d()``: influence function IF_i = H⁻¹ s_i
    # -------------------------------------------------------------- #

    @jit
    def _logistic_nll_per_obs(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Per-observation logistic NLL: log(1 + exp(-s * η))."""
        signs = 2.0 * y - 1.0
        return jax.nn.softplus(-signs * (X @ beta))

    @jit
    def _poisson_nll_per_obs(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Per-observation Poisson NLL: μ − y·η."""
        eta = X @ beta
        mu = jnp.exp(eta)
        return mu - y * eta

    def _make_negbin_nll_per_obs(
        alpha: float,
    ) -> Callable[..., Any]:
        """Return per-observation NB2 NLL with fixed α."""

        @jit
        def _negbin_nll_per_obs(
            beta: jnp.ndarray,
            X: jnp.ndarray,
            y: jnp.ndarray,
        ) -> jnp.ndarray:
            eta = X @ beta
            mu = jnp.exp(eta)
            inv_a = 1.0 / alpha
            return (y + inv_a) * jnp.log(1.0 + alpha * mu) - y * eta

        return _negbin_nll_per_obs

    def _make_ordinal_nll_per_obs(K: int) -> Callable[..., Any]:
        """Return per-observation ordinal NLL with fixed K."""
        _cat_probs = _make_ordinal_category_probs(K)

        @jit
        def _ordinal_nll_per_obs(
            params: jnp.ndarray,
            X: jnp.ndarray,
            y: jnp.ndarray,
        ) -> jnp.ndarray:
            n_features = X.shape[1]
            beta = params[:n_features]
            thresholds = params[n_features:]
            eta = X @ beta
            probs = _cat_probs(thresholds, eta)
            y_int = y.astype(jnp.int32)
            log_probs = jnp.log(probs[jnp.arange(X.shape[0]), y_int])
            return -log_probs

        return _ordinal_nll_per_obs

    def _make_multinomial_nll_per_obs(K: int) -> Callable[..., Any]:
        """Return per-observation multinomial NLL with fixed K."""
        Km1 = K - 1

        @jit
        def _multinomial_nll_per_obs(
            params_flat: jnp.ndarray,
            X: jnp.ndarray,
            y: jnp.ndarray,
        ) -> jnp.ndarray:
            p_aug = X.shape[1]
            B_mat = params_flat.reshape(Km1, p_aug)
            logits = jnp.concatenate(
                [jnp.zeros((X.shape[0], 1), dtype=jnp.float64), X @ B_mat.T],
                axis=1,
            )
            log_probs = jax.nn.log_softmax(logits, axis=1)
            y_int = y.astype(jnp.int32)
            return -log_probs[jnp.arange(X.shape[0]), y_int]

        return _multinomial_nll_per_obs

    # ============================================================== #
    # Sandwich (robust) standard errors
    # ============================================================== #
    #
    # The Eicker–Huber–White sandwich estimator provides
    # heteroscedasticity-robust SEs:
    #
    #   V = H⁻¹ (Σ sᵢ sᵢ') H⁻¹ = H⁻¹ S'S H⁻¹
    #
    # where H is the Hessian and sᵢ = ∂ℓᵢ/∂β is the per-observation
    # score vector.  The score matrix S = (s₁, …, sₙ)' is computed
    # via ``jax.vmap(jax.grad(nll_per_obs_i))`` where nll_per_obs_i
    # indexes a single observation.
    #
    # Ref: White, H. (1980). "A heteroskedasticity-consistent
    # covariance matrix estimator and a direct test for
    # heteroskedasticity." Econometrica, 48(4), 817–838.
    # -------------------------------------------------------------- #

    def _sandwich_se(
        nll_per_obs_fn: Callable[..., jnp.ndarray],
        hess_fn: Callable[[jnp.ndarray], jnp.ndarray],
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> np.ndarray:
        r"""Sandwich (robust) standard errors.

        Args:
            nll_per_obs_fn: ``(β, X, y) → (n,)`` per-observation NLL.
            hess_fn: ``β → (p, p)`` Hessian of the (summed) NLL.
            beta: Converged parameter vector ``(p,)``.
            X: Augmented design matrix ``(n, p)``.
            y: Response ``(n,)``.

        Returns:
            Robust standard errors ``(p,)`` as a NumPy array.
        """
        H = hess_fn(beta)
        H_inv = jnp.linalg.inv(H)

        # Score matrix: each row is the gradient of the i-th obs NLL.
        # We define a scalar-valued function nll_i(β) for observation i,
        # then vmap its gradient over i.
        def _nll_i(beta_: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
            """NLL for a single observation (used in vmap)."""
            return nll_per_obs_fn(beta_, X, y)[i]

        n = X.shape[0]
        # vmap over observation index, keeping beta fixed
        score_fn = jit(vmap(grad(_nll_i), in_axes=(None, 0)))
        S = score_fn(beta, jnp.arange(n))  # (n, p) score matrix

        # Meat: S'S
        meat = S.T @ S  # (p, p)

        # Sandwich: H⁻¹ S'S H⁻¹
        V = H_inv @ meat @ H_inv
        return np.asarray(jnp.sqrt(jnp.abs(jnp.diag(V))))

    # ============================================================== #
    # Autodiff Cook's distance
    # ============================================================== #
    #
    # Cook's D via the influence function:
    #   IF_i = H⁻¹ sᵢ
    #   D_i = (IF_i)' H (IF_i) / p
    #       = sᵢ' H⁻¹ sᵢ / p
    #
    # This extends Cook's D to families where statsmodels lacks
    # influence diagnostics (ordinal, multinomial, NB, Poisson).
    #
    # Ref: Cook, R. D. (1977). "Detection of influential observation
    # in linear regression." Technometrics, 19(1), 15–18.
    # -------------------------------------------------------------- #

    def _autodiff_cooks_d(
        nll_per_obs_fn: Callable[..., jnp.ndarray],
        hess_fn: Callable[[jnp.ndarray], jnp.ndarray],
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> np.ndarray:
        r"""Cook's distance via autodiff influence function.

        For each observation i:

        .. math::
            D_i = \frac{1}{p} s_i^\top H^{-1} s_i

        where ``s_i = ∂ℓ_i/∂β`` and ``H = Σ ∂²ℓ_i/∂β∂β'``.

        Args:
            nll_per_obs_fn: ``(β, X, y) → (n,)`` per-observation NLL.
            hess_fn: ``β → (p, p)`` Hessian of the (summed) NLL.
            beta: Converged parameter vector ``(p,)``.
            X: Augmented design matrix ``(n, p)``.
            y: Response ``(n,)``.

        Returns:
            Cook's distances ``(n,)`` as a NumPy array.
        """
        H = hess_fn(beta)
        H_inv = jnp.linalg.inv(H)
        p = beta.shape[0]

        def _nll_i(beta_: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
            return nll_per_obs_fn(beta_, X, y)[i]

        n = X.shape[0]
        score_fn = jit(vmap(grad(_nll_i), in_axes=(None, 0)))
        S = score_fn(beta, jnp.arange(n))  # (n, p)

        # D_i = (1/p) sᵢ' H⁻¹ sᵢ
        # Vectorise: IF = S @ H⁻¹, then D = rowwise dot(IF, S) / p
        IF = S @ H_inv  # (n, p)
        cooks_d = jnp.sum(IF * S, axis=1) / p  # (n,)
        return np.asarray(cooks_d)

    # ============================================================== #
    # Profile likelihood confidence intervals
    # ============================================================== #
    #
    # A profile likelihood CI for β_j is the set of values where the
    # profile deviance does not exceed the χ²₁(α) critical value:
    #
    #   {β_j : 2[ℓ_P(β̂) − ℓ_P(β_j)] ≤ χ²₁(α)}
    #
    # Here ℓ_P(β_j) = max_{β_{-j}} ℓ(β) is the profile log-
    # likelihood, obtained by re-optimising all parameters except β_j
    # at each candidate value.  Since our NLL functions store
    # *negative* log-likelihood, the deviance becomes
    # 2[NLL_profile − NLL_full].
    #
    # Two helpers:
    #   _profile_nll_at  — evaluates the profile NLL at a single
    #                      fixed value of β_j (inner Newton loop).
    #   _profile_ci_coefficients — bisection over _profile_nll_at
    #                              for each requested coefficient.
    #
    # Ref: Venzon, D. J. & Moolgavkar, S. H. (1988). "A method for
    # computing profile-likelihood-based confidence intervals."
    # *Applied Statistics*, 37(1), 87–94.
    # -------------------------------------------------------------- #

    def _profile_nll_at(
        nll_fn: Callable[..., jnp.ndarray],
        grad_fn: Callable[..., jnp.ndarray],
        hess_fn: Callable[..., jnp.ndarray],
        beta_hat: jnp.ndarray,
        j: int,
        beta_j_star: float,
        X: jnp.ndarray,
        y: jnp.ndarray,
        max_iter: int = 30,
        tol: float = 1e-8,
    ) -> float:
        """Profile NLL at β_j = *beta_j_star*, optimising over β_{-j}.

        Starts from β̂ with β_j replaced by *beta_j_star* and runs
        Newton–Raphson on the free parameters (everything except j)
        until convergence.

        Args:
            nll_fn: ``(β, X, y) → scalar`` negative log-likelihood.
            grad_fn: ``(β, X, y) → (p,)`` JIT-compiled gradient.
            hess_fn: ``(β, X, y) → (p, p)`` JIT-compiled Hessian.
            beta_hat: Full MLE parameter vector ``(p,)``.
            j: Index of the parameter to fix.
            beta_j_star: Value to fix β_j at.
            X: Design matrix ``(n, p_aug)``.
            y: Response vector ``(n,)``.
            max_iter: Maximum Newton iterations.
            tol: Convergence tolerance on the step norm.

        Returns:
            Scalar profile NLL (Python float).
        """
        p = beta_hat.shape[0]
        beta = jnp.array(beta_hat).at[j].set(beta_j_star)

        # Indices of the p-1 free (non-fixed) parameters.
        free = jnp.concatenate([jnp.arange(j), jnp.arange(j + 1, p)])

        for _ in range(max_iter):
            g = grad_fn(beta, X, y)
            H = hess_fn(beta, X, y)

            # Reduce to the free sub-system.
            g_free = g[free]
            H_free = H[free][:, free]

            # Newton step.  Use Cholesky for speed / stability.
            try:
                L = jnp.linalg.cholesky(H_free)
                delta = jax.scipy.linalg.cho_solve((L, True), g_free)
            except Exception:
                # Fall back to direct solve if Cholesky fails
                # (can happen near a boundary).
                delta = jnp.linalg.solve(
                    H_free + 1e-8 * jnp.eye(H_free.shape[0]),
                    g_free,
                )

            beta = beta.at[free].add(-delta)
            beta = beta.at[j].set(beta_j_star)  # keep j fixed

            if float(jnp.max(jnp.abs(delta))) < tol:
                break

        return float(nll_fn(beta, X, y))

    def _profile_ci_coefficients(
        nll_fn: Callable[..., jnp.ndarray],
        beta_hat: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
        feature_indices: list[int],
        alpha: float = 0.05,
        max_newton: int = 30,
        max_bisect: int = 60,
        tol: float = 1e-6,
    ) -> np.ndarray:
        r"""Profile likelihood CIs for selected parameters.

        For each index *j* in *feature_indices*, uses bisection to
        find the lower and upper values of β_j where:

        .. math::
            2\bigl[\mathrm{NLL}_{\mathrm{profile}}(\beta_j)
                  - \mathrm{NLL}(\hat\beta)\bigr]
            = \chi^2_1(\alpha)

        The initial bracket for the bisection is set to ±5 SE (from
        the observed Fisher information) and expanded geometrically
        if the deviance at the bracket endpoint is insufficient.

        Args:
            nll_fn: ``(β, X, y) → scalar`` NLL — NOT jitted (we
                jit the grad / Hessian closures internally).
            beta_hat: MLE parameter vector ``(p,)``.
            X: Design matrix ``(n, p_aug)``.
            y: Response vector ``(n,)``.
            feature_indices: Which parameter indices to compute
                profile CIs for (typically the slope indices,
                excluding intercept/thresholds).
            alpha: 1 − confidence_level.
            max_newton: Maximum Newton iterations per profile
                evaluation.
            max_bisect: Maximum bisection iterations per bound.
            tol: Bisection stops when the bracket width falls
                below ``tol * SE_j``.

        Returns:
            ``(len(feature_indices), 2)`` NumPy array of
            ``[lower, upper]`` profile CI bounds.
        """
        from scipy.stats import chi2 as chi2_dist

        chi2_crit = float(chi2_dist.ppf(1.0 - alpha, df=1))
        nll_full = float(nll_fn(beta_hat, X, y))

        # Pre-compile grad / Hessian closures (shared across all j).
        _grad_fn = jit(grad(nll_fn))
        _hess_fn = jit(jax.hessian(nll_fn))

        # Approximate SEs from the Hessian at the MLE — used only
        # to scale the initial bisection bracket.
        H_full = _hess_fn(beta_hat, X, y)
        se_approx = jnp.sqrt(jnp.abs(jnp.diag(jnp.linalg.inv(H_full))))

        def _deviance(j: int, beta_j_star: float) -> float:
            """Profile deviance at β_j = beta_j_star."""
            nll_prof = _profile_nll_at(
                nll_fn,
                _grad_fn,
                _hess_fn,
                beta_hat,
                j,
                beta_j_star,
                X,
                y,
                max_iter=max_newton,
                tol=1e-8,
            )
            return 2.0 * (nll_prof - nll_full)

        results = np.full((len(feature_indices), 2), np.nan)

        for idx, j in enumerate(feature_indices):
            se_j = float(se_approx[j])
            b_hat_j = float(beta_hat[j])

            if se_j <= 0.0 or not np.isfinite(se_j):
                continue

            # ── Lower bound ──────────────────────────────────── #
            lo = b_hat_j - 5.0 * se_j
            # Expand bracket until deviance exceeds the critical
            # value (max 5 doublings → 160 SE total).
            for _ in range(5):
                if _deviance(j, lo) >= chi2_crit:
                    break
                lo = b_hat_j - 2.0 * (b_hat_j - lo)

            hi = b_hat_j
            for _ in range(max_bisect):
                mid = 0.5 * (lo + hi)
                if _deviance(j, mid) > chi2_crit:
                    lo = mid
                else:
                    hi = mid
                if abs(hi - lo) < tol * se_j:
                    break
            results[idx, 0] = 0.5 * (lo + hi)

            # ── Upper bound ──────────────────────────────────── #
            hi = b_hat_j + 5.0 * se_j
            for _ in range(5):
                if _deviance(j, hi) >= chi2_crit:
                    break
                hi = b_hat_j + 2.0 * (hi - b_hat_j)

            lo = b_hat_j
            for _ in range(max_bisect):
                mid = 0.5 * (lo + hi)
                if _deviance(j, mid) > chi2_crit:
                    hi = mid
                else:
                    lo = mid
                if abs(hi - lo) < tol * se_j:
                    break
            results[idx, 1] = 0.5 * (lo + hi)

        return results

    # ============================================================== #
    # Single-observation fit with SEs (for classical_p_values)
    # ============================================================== #

    def _classical_p_values_logistic(
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        *,
        robust_se: bool = False,
    ) -> np.ndarray:
        """Wald z-test p-values for logistic regression via JAX.

        Fits a single logistic model, computes SEs from the observed
        Fisher information (or sandwich estimator if robust), and
        returns two-sided Wald p-values for slope coefficients.
        """
        from scipy.stats import norm

        X_aug = _augment_intercept(X, fit_intercept)
        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)

        beta, _, _ = _make_newton_solver(X_j, y_j, max_iter=100, tol=_DEFAULT_TOL)

        def hess_fn(b: jnp.ndarray) -> jnp.ndarray:
            return _logistic_hessian(b, X_j, y_j)  # type: ignore[no-any-return]

        if robust_se:
            se = _sandwich_se(_logistic_nll_per_obs, hess_fn, beta, X_j, y_j)
        else:
            se = _fisher_information_se(hess_fn, beta)

        z = np.asarray(beta) / se
        pvals = 2.0 * norm.sf(np.abs(z))
        return pvals[1:] if fit_intercept else pvals  # type: ignore[no-any-return]

    def _classical_p_values_poisson(
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        *,
        robust_se: bool = False,
    ) -> np.ndarray:
        """Wald z-test p-values for Poisson regression via JAX."""
        from scipy.stats import norm

        X_aug = _augment_intercept(X, fit_intercept)
        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)

        beta, _, _ = _make_poisson_solver(X_j, y_j, max_iter=100, tol=_DEFAULT_TOL)

        def hess_fn(b: jnp.ndarray) -> jnp.ndarray:
            return _poisson_hessian(b, X_j, y_j)  # type: ignore[no-any-return]

        if robust_se:
            se = _sandwich_se(_poisson_nll_per_obs, hess_fn, beta, X_j, y_j)
        else:
            se = _fisher_information_se(hess_fn, beta)

        z = np.asarray(beta) / se
        pvals = 2.0 * norm.sf(np.abs(z))
        return pvals[1:] if fit_intercept else pvals  # type: ignore[no-any-return]

    def _classical_p_values_negbin(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        fit_intercept: bool,
        *,
        robust_se: bool = False,
    ) -> np.ndarray:
        """Wald z-test p-values for NB2 regression via JAX."""
        from scipy.stats import norm

        X_aug = _augment_intercept(X, fit_intercept)
        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)

        beta, _, _ = _make_negbin_solver(
            X_j, y_j, alpha, max_iter=100, tol=_DEFAULT_TOL
        )
        _nb_hess = _make_negbin_hessian(alpha)

        def hess_fn(b: jnp.ndarray) -> jnp.ndarray:
            return _nb_hess(b, X_j, y_j)  # type: ignore[no-any-return]

        if robust_se:
            nll_per_obs_fn = _make_negbin_nll_per_obs(alpha)
            se = _sandwich_se(nll_per_obs_fn, hess_fn, beta, X_j, y_j)
        else:
            se = _fisher_information_se(hess_fn, beta)

        z = np.asarray(beta) / se
        pvals = 2.0 * norm.sf(np.abs(z))
        return pvals[1:] if fit_intercept else pvals  # type: ignore[no-any-return]

    def _classical_p_values_ordinal(
        X: np.ndarray,
        y: np.ndarray,
        K: int,
        n_features: int,
        fit_intercept: bool,  # noqa: ARG001 — ignored; thresholds are intercepts
        *,
        robust_se: bool = False,
    ) -> np.ndarray:
        """Wald z-test p-values for ordinal regression via JAX.

        ``fit_intercept`` is accepted for protocol compatibility but
        ignored — ordinal thresholds serve as category-specific
        intercepts. The design matrix is used WITHOUT an intercept
        column.
        """
        from scipy.stats import norm

        # No intercept augmentation — thresholds serve as intercepts.
        X_j = jnp.array(np.asarray(X, dtype=np.float64), dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)

        params, _, _ = _make_ordinal_solver(X_j, y_j, K, max_iter=100, tol=_DEFAULT_TOL)
        _nll = _make_ordinal_nll(K)
        _ord_hess = jit(jax.hessian(_nll))

        def hess_fn(p: jnp.ndarray) -> jnp.ndarray:
            return _ord_hess(p, X_j, y_j)  # type: ignore[no-any-return]

        if robust_se:
            nll_per_obs_fn = _make_ordinal_nll_per_obs(K)
            se = _sandwich_se(nll_per_obs_fn, hess_fn, params, X_j, y_j)
        else:
            se = _fisher_information_se(hess_fn, params)

        # Slopes are params[0:n_features]; thresholds follow.
        z = np.asarray(params[:n_features]) / se[:n_features]
        pvals = 2.0 * norm.sf(np.abs(z))
        return pvals  # type: ignore[no-any-return]

    def _classical_p_values_multinomial(
        X: np.ndarray,
        y: np.ndarray,
        K: int,
        n_features: int,
        fit_intercept: bool,
        *,
        robust_se: bool = False,
    ) -> np.ndarray:
        """Wald χ² p-values for multinomial regression via JAX.

        For each slope predictor j, extracts the (K-1) coefficients
        across categories and computes a joint Wald χ² test —
        matching the statsmodels MNLogit convention.
        """
        from scipy.stats import chi2

        X_aug = _augment_intercept(X, fit_intercept)
        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)
        p_aug = X_aug.shape[1]

        params, _, _ = _make_multinomial_solver(
            X_j, y_j, K, max_iter=100, tol=_DEFAULT_TOL
        )

        _nll = _make_multinomial_nll(K)
        H = jit(jax.hessian(_nll))(params, X_j, y_j)
        cov = jnp.linalg.inv(H)

        # Wald χ² per slope predictor via the existing helper
        _wald = _make_multinomial_wald_chi2(K, p_aug, fit_intercept)
        wald_chi2 = np.asarray(_wald(params, cov))  # (n_features,)
        pvals = chi2.sf(wald_chi2, df=K - 1)
        return pvals  # type: ignore[no-any-return]

    # ============================================================== #
    # GLM score projection helpers (for non-linear families)
    # ============================================================== #
    #
    # Score projection computes the permuted test statistic via a
    # single Fisher-information-weighted score projection:
    #
    #   β̂*_j ≈ (I⁻¹ U)_j
    #
    # where I = X' W X is the Fisher information at the reduced-model
    # MLE and U = X' (y* − μ̂_red) is the score evaluated at the
    # reduced-model parameters with the full design matrix.
    #
    # For GLM families with canonical link:
    #   U_j = Σ x_{ij} (y*_i − μ̂_i)
    #
    # The one-step Newton update from the reduced-model MLE is
    #
    #   Δβ = (X'WX)⁻¹ X'(y − μ̂_red)
    #
    # so the projection row for feature j is the j-th row of
    # (X'WX)⁻¹ X' (WITHOUT W on the right — the W appears only in
    # the Fisher information, not in the score vector for canonical
    # links).  Permuted scores are E_π @ projection_row.
    #
    # Ref: Rao, C. R. (1948). "Large sample tests of composite
    # hypotheses." Sankhyā, 9(1), 44–56.
    # -------------------------------------------------------------- #

    def _glm_score_projection_row(
        X_aug: np.ndarray,
        W_diag: np.ndarray,
        feature_idx: int,
    ) -> np.ndarray:
        r"""Compute the j-th row of the score test projection matrix.

        For a GLM with canonical link and working weights W, the
        one-step Newton update is:

        .. math::
            \Delta\hat\beta = (X^\top W X)^{-1} X^\top (y - \hat\mu)

        The projection row for feature j is:

        .. math::
            A_j = [(X^\top W X)^{-1} X^\top]_j

        Note: W appears only in the Fisher information (X'WX), NOT
        in the score projection (X'e).  This matches the score test
        definition where U = X'(y - μ̂).

        Args:
            X_aug: Augmented design matrix ``(n, p_aug)``.
            W_diag: Working weights ``(n,)`` — ``Var(μ̂)`` for
                canonical link families (e.g. μ̂(1−μ̂) for logistic).
            feature_idx: Index into the *augmented* parameter vector.

        Returns:
            Projection row ``(n,)`` as a NumPy array.
        """
        XtW = X_aug.T * W_diag[None, :]  # (p_aug, n)
        fisher = XtW @ X_aug  # (p_aug, p_aug) — Fisher information
        # solve(Fisher, e_j) extracts row j of I⁻¹ without full inversion;
        # pinv fallback for singular/near-singular Fisher (e.g. perfect
        # separation).
        e_j = np.zeros(fisher.shape[0])
        e_j[feature_idx] = 1.0
        try:
            A_row = np.linalg.solve(fisher, e_j) @ X_aug.T  # (n,)
        except np.linalg.LinAlgError:
            A_row = np.linalg.pinv(fisher)[feature_idx] @ X_aug.T
        return A_row  # type: ignore[no-any-return]

    # ============================================================== #
    # Logistic regression helpers
    # ============================================================== #
    #
    # Binary logistic regression with the canonical logit link:
    #
    #   P(y=1 | X) = σ(Xβ)  where σ(z) = 1/(1 + e^{−z})
    #
    # NLL uses the numerically stable ``logaddexp`` form to avoid
    # overflow when |Xβ| is large (common with well-separated data):
    #
    #   NLL = Σ_i log(1 + exp(−s_i · X_iβ))  where s_i = 2y_i − 1
    #
    # The gradient and Hessian are coded analytically (not via
    # ``jax.grad``) because the closed forms are simple and avoid
    # a second tracing pass through the XLA compiler.
    # -------------------------------------------------------------- #

    @jit
    def _logistic_nll(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Negative log-likelihood for logistic regression.

        Uses the numerically stable logaddexp form:

            NLL = sum_i log(1 + exp(-s_i * X_i @ beta))

        where s_i = 2*y_i - 1.
        """
        logits = X @ beta
        return jnp.sum(jnp.logaddexp(0.0, -logits * (2.0 * y - 1.0)))

    _logistic_grad = jit(grad(_logistic_nll))

    @jit
    def _logistic_hessian(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Full Hessian of the logistic NLL: H = X' diag(p*(1-p)) X."""
        p = jax.nn.sigmoid(X @ beta)
        W = p * (1.0 - p)
        return (X.T * W[None, :]) @ X

    def _make_newton_solver(
        X: jnp.ndarray,
        y: jnp.ndarray,
        max_iter: int,
        tol: float,
        min_damping: float = _MIN_DAMPING,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Newton–Raphson logistic solve with early stopping.

        Delegates to :func:`_newton_solve_static` with analytical
        gradient and Hessian closures.  β₀ = 0 (safe for convex
        logistic NLL).

        Args:
            X: Augmented design matrix ``(n, p+1)``.
            y: Binary response ``(n,)``.
            max_iter: Maximum Newton iterations.
            tol: Convergence threshold.
            min_damping: Singularity guard.

        Returns:
            ``(beta, nll, converged)``.
        """
        beta_init = jnp.zeros(X.shape[1], dtype=jnp.float64)
        return _newton_solve_static(
            nll_fn=lambda b: _logistic_nll(b, X, y),
            grad_fn=lambda b: _logistic_grad(b, X, y),
            hess_fn=lambda b: _logistic_hessian(b, X, y),
            beta_init=beta_init,
            max_iter=max_iter,
            tol=tol,
            min_damping=min_damping,
        )

    # ============================================================== #
    # Poisson regression helpers
    # ============================================================== #
    #
    # Poisson regression with the canonical log link:
    #
    #   E(y) = μ = exp(Xβ)
    #   P(y | μ) = μ^y e^{−μ} / y!
    #
    # NLL (dropping the constant log(y!) term):
    #
    #   NLL = Σ [exp(Xβ) − y · Xβ]
    #
    # The gradient X'(μ − y) and Hessian X' diag(μ) X are coded
    # analytically.  Warm start uses OLS on the working response
    # log(y + 0.5) — the standard IRLS initialisation.
    # -------------------------------------------------------------- #

    @jit
    def _poisson_nll(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Negative log-likelihood for Poisson regression (log link).

        NLL = Σ [exp(Xβ) − y·(Xβ)]   (constant log(y!) dropped).
        """
        eta = X @ beta
        # Clip to prevent exp overflow (|η| > 709 → inf); matches GLMM Poisson bound.
        eta = jnp.clip(eta, -20.0, 20.0)
        mu = jnp.exp(eta)
        return jnp.sum(mu - y * eta)

    @jit
    def _poisson_grad(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Gradient of Poisson NLL: X'(μ − y) where μ = exp(Xβ)."""
        # Clip to prevent exp overflow; matches GLMM Poisson bound.
        eta = jnp.clip(X @ beta, -20.0, 20.0)
        mu = jnp.exp(eta)
        return X.T @ (mu - y)

    @jit
    def _poisson_hessian(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,  # noqa: ARG001
    ) -> jnp.ndarray:
        """Hessian of Poisson NLL: X' diag(μ) X where μ = exp(Xβ)."""
        # Clip to prevent exp overflow; matches GLMM Poisson bound.
        mu = jnp.exp(jnp.clip(X @ beta, -20.0, 20.0))
        return (X.T * mu[None, :]) @ X

    def _make_poisson_solver(
        X: jnp.ndarray,
        y: jnp.ndarray,
        max_iter: int,
        tol: float,
        min_damping: float = _MIN_DAMPING,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Newton–Raphson Poisson solve with early stopping.

        Delegates to :func:`_newton_solve_static` with analytical
        gradient and Hessian closures.  β₀ is OLS on the log-link
        working response ``log(y + 0.5)`` — the standard IRLS
        warm start to avoid overflow when the true intercept is
        far from zero.
        """
        n_params = X.shape[1]
        damping_matrix = min_damping * jnp.eye(n_params, dtype=jnp.float64)

        # Log-link warm start: β₀ = (X'X + λI)⁻¹ X' log(y + 0.5)
        eta_init = jnp.log(y + 0.5)
        XtX = X.T @ X + damping_matrix
        Xty = X.T @ eta_init
        beta_init = jnp.linalg.solve(XtX, Xty)

        return _newton_solve_static(
            nll_fn=lambda b: _poisson_nll(b, X, y),
            grad_fn=lambda b: _poisson_grad(b, X, y),
            hess_fn=lambda b: _poisson_hessian(b, X, y),
            beta_init=beta_init,
            max_iter=max_iter,
            tol=tol,
            min_damping=min_damping,
        )

    # ============================================================== #
    # Negative Binomial (NB2) regression helpers
    # ============================================================== #
    #
    # NB2 extends Poisson by adding an overdispersion parameter α:
    #
    #   E(y) = μ = exp(Xβ)
    #   Var(y) = μ + α·μ²
    #
    # The α parameter is calibrated **once** on the observed data
    # and **held fixed** across all B permutations (see the NumPy
    # backend's NB2 section header for the statistical rationale).
    #
    # Because α is a Python float (not a JAX tracer), the NLL,
    # gradient, and Hessian are constructed via **closure factories**
    # (``_make_negbin_nll(alpha)``, etc.) that capture α as a
    # compile-time constant.  This lets XLA fold the constant into
    # the compiled kernel and avoids recompilation when α changes
    # across different datasets.
    # -------------------------------------------------------------- #

    def _make_negbin_nll(alpha: float) -> Callable[..., Any]:
        """Return a JIT-compiled NB2 NLL function with fixed α.

        NB2 NLL (up to constants not depending on β):
            Σ[(y + 1/α)·log(1 + α·μ) − y·log(α·μ)]
        where μ = exp(Xβ).
        """

        @jit
        def _negbin_nll(
            beta: jnp.ndarray,
            X: jnp.ndarray,
            y: jnp.ndarray,
        ) -> jnp.ndarray:
            eta = X @ beta
            mu = jnp.exp(eta)
            inv_a = 1.0 / alpha
            # NB2 NLL = -Σ[y·log(α·μ/(1+α·μ)) + (1/α)·log(1/(1+α·μ))]
            # = Σ[(y + 1/α)·log(1 + α·μ) − y·(log(α) + eta)]
            return jnp.sum((y + inv_a) * jnp.log(1.0 + alpha * mu) - y * eta)

        return _negbin_nll

    def _make_negbin_grad(alpha: float) -> Callable[..., Any]:
        """Return a JIT-compiled NB2 gradient via autodiff.

        Uses ``jax.grad`` on the NB2 NLL closure to compute the
        exact gradient ∂NLL/∂β automatically.  This replaces a
        hand-derived gradient and ensures consistency with the NLL
        definition in :func:`_make_negbin_nll`.
        """
        return jit(grad(_make_negbin_nll(alpha)))

    def _make_negbin_hessian(alpha: float) -> Callable[..., Any]:
        """Return a JIT-compiled NB2 Hessian via autodiff.

        Uses ``jax.hessian`` on the NB2 NLL closure to compute the
        exact Hessian ∂²NLL/∂β∂β' automatically.  This replaces a
        hand-derived Hessian and ensures consistency with the NLL
        definition in :func:`_make_negbin_nll`.
        """
        return jit(hessian(_make_negbin_nll(alpha)))

    def _make_negbin_solver(
        X: jnp.ndarray,
        y: jnp.ndarray,
        alpha: float,
        max_iter: int,
        tol: float,
        min_damping: float = _MIN_DAMPING,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Newton–Raphson NB2 solve with early stopping and fixed α.

        Delegates to :func:`_newton_solve_static` with closure-based
        NLL / gradient / Hessian for the fixed overdispersion α.
        β₀ is OLS on the log-link working response ``log(y + 0.5)``.
        """
        n_params = X.shape[1]
        damping_matrix = min_damping * jnp.eye(n_params, dtype=jnp.float64)

        _nll = _make_negbin_nll(alpha)
        _grad = _make_negbin_grad(alpha)
        _hess = _make_negbin_hessian(alpha)

        # Log-link warm start: β₀ = (X'X + λI)⁻¹ X' log(y + 0.5)
        eta_init = jnp.log(y + 0.5)
        XtX = X.T @ X + damping_matrix
        Xty = X.T @ eta_init
        beta_init = jnp.linalg.solve(XtX, Xty)

        return _newton_solve_static(
            nll_fn=lambda b: _nll(b, X, y),
            grad_fn=lambda b: _grad(b, X, y),
            hess_fn=lambda b: _hess(b, X, y),
            beta_init=beta_init,
            max_iter=max_iter,
            tol=tol,
            min_damping=min_damping,
        )

    # ============================================================== #
    # Ordinal (proportional-odds logistic) regression helpers
    # ============================================================== #
    #
    # The ordered logistic model parameterises cumulative probabilities:
    #
    #   P(y ≤ k | X) = σ(α_k − Xβ)    for k = 0, …, K-2
    #
    # Parameters are packed as [β₁, …, β_p, α₀, …, α_{K-2}] —
    # slopes first, thresholds last.  K is a compile-time constant
    # captured in closures to avoid JAX tracing issues with
    # dynamic shapes.
    #
    # The gradient and Hessian are obtained via ``jax.grad`` and
    # ``jax.hessian`` (autodiff) because the closed-form derivatives
    # involve sums over categories with threshold-dependent sigmoid
    # chains that are error-prone to hand-code.
    # -------------------------------------------------------------- #

    @jit
    def _ordinal_cumulative_probs(
        thresholds: jnp.ndarray,
        eta: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute cumulative probabilities P(Y ≤ k | X).

        Args:
            thresholds: Sorted cutpoints (K-1,).
            eta: Linear predictor Xβ (n,).

        Returns:
            Cumulative probs (n, K-1).
        """
        # P(Y ≤ k) = σ(α_k − η) for k = 0, …, K-2
        return jax.nn.sigmoid(thresholds[None, :] - eta[:, None])

    def _make_ordinal_category_probs(K: int) -> Callable[..., Any]:
        """Return a function that computes category probs for K categories.

        K is captured as a compile-time constant in the closure,
        avoiding JAX tracing issues with dynamic shapes.
        """

        @jit
        def _category_probs(
            thresholds: jnp.ndarray,
            eta: jnp.ndarray,
        ) -> jnp.ndarray:
            cum_probs = _ordinal_cumulative_probs(thresholds, eta)
            # Build probabilities using padding + diff approach
            # Pad with 0 on left, 1 on right: [0, cum[0], ..., cum[K-2], 1]
            zeros = jnp.zeros((eta.shape[0], 1), dtype=jnp.float64)
            ones = jnp.ones((eta.shape[0], 1), dtype=jnp.float64)
            padded = jnp.concatenate([zeros, cum_probs, ones], axis=1)  # (n, K+1)
            # P(Y=k) = padded[:, k+1] - padded[:, k]
            probs = padded[:, 1:] - padded[:, :-1]  # (n, K)
            return jnp.clip(probs, 1e-12, 1.0 - 1e-12)

        return _category_probs

    def _make_ordinal_nll(K: int) -> Callable[..., Any]:
        """Return JIT-compiled ordinal NLL.

        Parameters are packed as [β_1, …, β_p, α_0, …, α_{K-2}].
        K is a compile-time constant captured in the closure.
        """
        _category_probs = _make_ordinal_category_probs(K)

        @jit
        def _ordinal_nll(
            params: jnp.ndarray,
            X: jnp.ndarray,
            y: jnp.ndarray,
        ) -> jnp.ndarray:
            n_features = X.shape[1]
            beta = params[:n_features]
            thresholds = params[n_features:]
            eta = X @ beta
            probs = _category_probs(thresholds, eta)
            # Index into category probs: NLL = -Σ log P(Y=y_i)
            y_int = y.astype(jnp.int32)
            log_probs = jnp.log(probs[jnp.arange(X.shape[0]), y_int])
            return -jnp.sum(log_probs)

        return _ordinal_nll

    def _make_ordinal_solver(
        X: jnp.ndarray,
        y: jnp.ndarray,
        K: int,
        max_iter: int,
        tol: float,
        min_damping: float = _MIN_DAMPING,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Newton–Raphson ordinal solve using autodiff.

        Delegates to :func:`_newton_solve_static` with autodiff
        gradient and Hessian closures.  Parameters are packed as
        ``[β_1, …, β_p, α_0, …, α_{K-2}]``.

        Returns ``(params, nll, converged)`` where params has
        length ``p + K - 1``.
        """
        n_features = X.shape[1]

        _nll = _make_ordinal_nll(K)
        _grad_fn = jit(grad(_nll))
        _hess_fn = jit(jax.hessian(_nll))

        # Initialize: β=0, thresholds evenly spaced
        init_thresholds = jnp.linspace(-1.0, 1.0, K - 1)
        init_params = jnp.concatenate(
            [
                jnp.zeros(n_features, dtype=jnp.float64),
                init_thresholds,
            ]
        )

        return _newton_solve_static(
            nll_fn=lambda p: _nll(p, X, y),
            grad_fn=lambda p: _grad_fn(p, X, y),
            hess_fn=lambda p: _hess_fn(p, X, y),
            beta_init=init_params,
            max_iter=max_iter,
            tol=tol,
            min_damping=min_damping,
        )

    # ============================================================== #
    # Multinomial (softmax) regression helpers
    # ============================================================== #
    #
    # Multinomial logistic (softmax) regression for a nominal outcome
    # y ∈ {0, 1, …, K-1}.  Category 0 is the reference class;
    # K-1 sets of coefficients β_k (k = 1, …, K-1) model the
    # log-odds relative to category 0:
    #
    #   log P(y=k|X) / P(y=0|X) = Xβ_k
    #
    # Parameters are stored as a flat vector of length (K-1) × p_aug
    # and reshaped internally to (K-1, p_aug).  K and p_aug are
    # compile-time constants captured in closures.
    #
    # NLL uses ``jax.nn.log_softmax`` for numerical stability
    # (log-sum-exp trick avoids overflow in exp(η)).
    #
    # The Wald χ² extractor follows the same logic as the NumPy
    # backend's ``_wald_chi2_from_mnlogit`` but is JIT-compiled
    # and uses ``jax.vmap`` over predictors.
    # -------------------------------------------------------------- #

    def _make_multinomial_nll(K: int) -> Callable[..., Any]:
        """Return JIT-compiled multinomial (softmax) NLL.

        Parameters are packed as a flat vector of length
        ``(K-1) × p_aug``, reshaped internally to ``(K-1, p_aug)``.
        Each row corresponds to class k = 1, …, K-1 (vs reference
        class 0).  K is a compile-time constant captured in the
        closure.

        The NLL uses ``jax.nn.log_softmax`` for numerical stability
        (log-sum-exp trick avoids overflow in exp(η)).
        """
        Km1 = K - 1

        @jit
        def _multinomial_nll(
            params_flat: jnp.ndarray,
            X: jnp.ndarray,
            y: jnp.ndarray,
        ) -> jnp.ndarray:
            p_aug = X.shape[1]
            # Reshape flat params → (K-1, p_aug), one row per non-ref class
            B_mat = params_flat.reshape(Km1, p_aug)  # (K-1, p_aug)
            # Linear predictors: η_k = X @ β_k for k=1..K-1
            # Prepend a zero column for reference class 0
            logits = jnp.concatenate(
                [jnp.zeros((X.shape[0], 1), dtype=jnp.float64), X @ B_mat.T],
                axis=1,
            )  # (n, K)
            log_probs = jax.nn.log_softmax(logits, axis=1)  # (n, K)
            y_int = y.astype(jnp.int32)
            return -jnp.sum(log_probs[jnp.arange(X.shape[0]), y_int])

        return _multinomial_nll

    def _make_multinomial_solver(
        X: jnp.ndarray,
        y: jnp.ndarray,
        K: int,
        max_iter: int,
        tol: float,
        min_damping: float = _MIN_DAMPING,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Newton–Raphson multinomial solve using autodiff.

        Delegates to :func:`_newton_solve_static` with autodiff
        gradient and Hessian closures.  Parameters are a flat vector
        of length ``(K-1) × p_aug``.

        Returns ``(params_flat, nll, converged)``.
        """
        p_aug = X.shape[1]
        n_params = (K - 1) * p_aug

        _nll = _make_multinomial_nll(K)
        _grad_fn = jit(grad(_nll))
        _hess_fn = jit(jax.hessian(_nll))

        # Zero initialisation — safe for softmax (uniform probs 1/K)
        init_params = jnp.zeros(n_params, dtype=jnp.float64)

        return _newton_solve_static(
            nll_fn=lambda p: _nll(p, X, y),
            grad_fn=lambda p: _grad_fn(p, X, y),
            hess_fn=lambda p: _hess_fn(p, X, y),
            beta_init=init_params,
            max_iter=max_iter,
            tol=tol,
            min_damping=min_damping,
        )

    def _make_multinomial_wald_chi2(
        K: int,
        p_aug: int,
        has_intercept: bool,
    ) -> Callable[..., Any]:
        """Return a JIT-compiled Wald χ² extractor for multinomial params.

        K, p_aug, and has_intercept are captured as compile-time
        constants in the closure, avoiding JAX tracing issues with
        ``jnp.arange`` requiring concrete values.

        The returned function signature is:
            ``(params_flat, cov) -> wald_chi2``

        For each slope predictor j (intercept excluded), extracts the
        (K-1)-vector of coefficients across categories and the
        corresponding (K-1, K-1) covariance sub-block, then computes:

            χ²_j = β_j^T [Var(β_j)]^{-1} β_j

        Returns:
            A JIT-compiled function ``(params_flat, cov) -> (p_slopes,)``.
        """
        Km1 = K - 1
        start = 1 if has_intercept else 0
        p_slopes = p_aug - start

        @jit
        def _wald_chi2(
            params_flat: jnp.ndarray,
            cov: jnp.ndarray,
        ) -> jnp.ndarray:
            def _wald_one(j_slope: jnp.ndarray) -> jnp.ndarray:
                j = j_slope + start  # column index in X_aug
                # Flat indices for this predictor across all K-1 categories
                idx = jnp.arange(Km1) * p_aug + j
                beta_j = params_flat[idx]  # (K-1,)
                cov_j = cov[jnp.ix_(idx, idx)]  # (K-1, K-1)
                # Wald = β' V^{-1} β
                return beta_j @ jnp.linalg.solve(  # type: ignore[no-any-return]
                    cov_j + _MIN_DAMPING * jnp.eye(Km1, dtype=jnp.float64),
                    beta_j,
                )

            return vmap(_wald_one)(jnp.arange(p_slopes))  # type: ignore[no-any-return]

        return _wald_chi2

    # ============================================================== #
    # Henderson REML helpers (linear mixed models)
    # ============================================================== #
    #
    # REML estimation for linear mixed models:
    #
    #   y = Xβ + Zu + ε,  u ~ N(0, σ²Γ),  ε ~ N(0, σ²I)
    #
    # Γ is block-diagonal:
    #   Γ = block_diag( kron(I_{G_1}, Σ_1), …, kron(I_{G_K}, Σ_K) )
    #
    # where Σ_k is the (d_k × d_k) covariance matrix for one group
    # in factor k (intercept + slopes), parameterised via the
    # log-Cholesky decomposition: Σ_k = L_k L_k' with L_k lower-
    # triangular and exp'd diagonal for positivity.
    #
    # θ = [vech(L_1), …, vech(L_K)] has Σ_k d_k(d_k+1)/2 entries.
    # σ² is profiled out analytically.  Special case d_k=1 recovers
    # the scalar variance-ratio parameterisation γ_k = exp(2θ_k).
    #
    # The solver operates on the Henderson mixed-model equations
    # — flat (p+q)-dimensional arrays with *no per-cluster loops*.
    # The same code path handles balanced, unbalanced, nested,
    # crossed, and correlated-slope random-effect structures by
    # varying only *Z* and *re_struct*.
    #
    # The projection matrix A = S⁻¹(X' − X'Z C₂₂⁻¹ Z') is σ²-free
    # and enables batch permutation via the single matmul A @ E_π,
    # structurally identical to batch_ols.
    #
    # Mathematical reference: plan-v040Series.prompt.md Appendix A.
    # Validated implementation: research/test_henderson_reml.py.
    # -------------------------------------------------------------- #

    def _fill_lower_triangular_jax(
        params: jnp.ndarray,
        d: int,
    ) -> jnp.ndarray:
        """Build a d×d lower-triangular Cholesky factor from flat params.

        The diagonal entries are exponentiated to enforce positivity.
        Off-diagonal entries are unconstrained.  Row-major vech order:
        ``[L[0,0], L[1,0], L[1,1], L[2,0], …]``.

        JAX-compatible: loops are unrolled during tracing since *d*
        is a Python constant.

        Args:
            params: Flat parameter vector of length ``d(d+1)/2``.
            d: Matrix dimension.

        Returns:
            Lower-triangular matrix ``(d, d)`` with positive diagonal.
        """
        L = jnp.zeros((d, d))
        idx = 0
        for i in range(d):
            for j in range(i):
                L = L.at[i, j].set(params[idx])
                idx += 1
            # Diagonal: exp for positivity
            L = L.at[i, i].set(jnp.exp(params[idx]))
            idx += 1
        return L

    def _fill_lower_triangular_np(
        params: np.ndarray,
        d: int,
    ) -> np.ndarray:
        """NumPy version of :func:`_fill_lower_triangular_jax`.

        Used for post-convergence recovery (outside JAX tracing).
        """
        L = np.zeros((d, d))
        idx = 0
        for i in range(d):
            for j in range(i):
                L[i, j] = params[idx]
                idx += 1
            L[i, i] = np.exp(params[idx])
            idx += 1
        return L

    # -------------------------------------------------------------- #
    # GLMM: Per-family conditional NLL and IRLS working quantities
    # -------------------------------------------------------------- #
    #
    # These four pure JAX functions are the ONLY lines in the entire
    # GLMM stack that differ between families.  Everything else —
    # Laplace NLL, IRLS inner loop, Henderson algebra, Newton outer
    # solver — is generic.
    #
    # Convention:
    #   eta = X @ beta + Z @ u   (linear predictor)
    #   mu  = g^{-1}(eta)        (conditional mean)
    #   w   = 1 / [g'(mu)^2 · Var(y|u)]   (IRLS weight)
    #   z~  = eta + (y - mu) / w           (working response)
    #
    # Pre-scaling by sqrt(W) converts the weighted Henderson system
    # into the unweighted system from Plan A — same algebra,
    # different inputs.

    def _logistic_conditional_nll(
        y: jnp.ndarray,
        eta: jnp.ndarray,
    ) -> jnp.ndarray:
        """Negative conditional log-likelihood for Bernoulli(logit).

        .. math::
            -\\ell(y|\\eta) = -\\sum_i [y_i \\eta_i - \\log(1 + e^{\\eta_i})]

        Uses ``jnp.logaddexp(0, eta)`` for numerical stability
        (avoids overflow in ``exp(eta)`` for large positive eta).
        """
        return -jnp.sum(y * eta - jnp.logaddexp(0.0, eta))

    def _logistic_working_response_and_weights(
        y: jnp.ndarray,
        eta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """IRLS working response and weights for logistic GLMM.

        Returns ``(z_tilde, w)`` where:

        * ``mu = sigmoid(eta)``
        * ``w = clip(mu * (1 - mu), eps)`` — canonical variance
        * ``z_tilde = eta + (y - mu) / w`` — working response

        Pre-scaling X̃ = √W ⊙ X, Z̃ = √W ⊙ Z, ỹ* = √W ⊙ z̃
        converts the weighted Henderson system into the unweighted
        Henderson system from Plan A.
        """
        mu = jax.nn.sigmoid(eta)
        w = jnp.clip(mu * (1.0 - mu), 1e-10)
        z_tilde = eta + (y - mu) / w
        return z_tilde, w

    def _poisson_conditional_nll(
        y: jnp.ndarray,
        eta: jnp.ndarray,
    ) -> jnp.ndarray:
        """Negative conditional log-likelihood for Poisson(log).

        .. math::
            -\\ell(y|\\eta) = -\\sum_i [y_i \\eta_i - e^{\\eta_i}
                              - \\log\\Gamma(y_i + 1)]

        The ``gammaln(y + 1)`` term is constant w.r.t. parameters
        but included for correct NLL value (needed by Laplace).
        """
        return -jnp.sum(y * eta - jnp.exp(eta) - jax.scipy.special.gammaln(y + 1.0))

    def _poisson_working_response_and_weights(
        y: jnp.ndarray,
        eta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """IRLS working response and weights for Poisson GLMM.

        Returns ``(z_tilde, w)`` where:

        * ``mu = exp(eta)``
        * ``w = clip(mu, eps)`` — canonical variance (Poisson: Var = mu)
        * ``z_tilde = eta + (y - mu) / w`` — working response
        """
        mu = jnp.exp(eta)
        w = jnp.clip(mu, 1e-10)
        z_tilde = eta + (y - mu) / w
        return z_tilde, w

    def _build_reml_nll(
        X: jnp.ndarray,
        Z: jnp.ndarray,
        y: jnp.ndarray,
        re_struct: list[tuple[int, int]],
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Build a pure REML NLL function for JAX autodiff.

        Uses a block-diagonal Cholesky parameterisation for the
        random-effect covariance Γ.  Each factor k has a d_k × d_k
        covariance Σ_k = L_k L_k' replicated across G_k groups.
        σ² is profiled out analytically.

        The profile REML deviance is:

            d_R(θ) = ½[(n−p)·log Q + log|Ṽ| + log|S|]

        where all terms are computed via the Henderson system
        without building the dense n×n matrix Ṽ.

        Args:
            X: Fixed-effect design matrix (n, p), WITH intercept.
            Z: Random-effect design matrix (n, q).
            y: Response vector (n,).
            re_struct: ``[(G_1, d_1), …, (G_K, d_K)]`` — number of
                groups and RE dimension per factor.  d_k=1 for
                random intercepts only, d_k>1 for intercept+slopes.

        Returns:
            Pure function f(log_chol_params) → scalar REML NLL.
        """
        n, p = X.shape
        q = Z.shape[1]
        n_components = len(re_struct)

        # Pre-compute sufficient statistics (O(n) BLAS-3 ops, done once)
        XtX = X.T @ X  # (p, p)
        XtZ = X.T @ Z  # (p, q)
        ZtZ = Z.T @ Z  # (q, q)
        Xty = X.T @ y  # (p,)
        Zty = Z.T @ y  # (q,)
        yty = y @ y  # scalar

        def reml_nll(log_chol_params: jnp.ndarray) -> jnp.ndarray:
            """Profile REML NLL — pure function of θ."""
            # Build block-diagonal Γ⁻¹ and compute log|Γ|
            Gamma_inv = jnp.zeros((q, q))
            log_det_Gamma = jnp.zeros(())

            factor_offset = 0
            theta_offset = 0
            for k in range(n_components):
                G_k, d_k = re_struct[k]
                n_chol_k = d_k * (d_k + 1) // 2
                theta_k = log_chol_params[theta_offset : theta_offset + n_chol_k]

                L_k = _fill_lower_triangular_jax(theta_k, d_k)
                Sigma_k_inv = jax.scipy.linalg.cho_solve((L_k, True), jnp.eye(d_k))

                # Fill G_k identical d_k×d_k blocks via Kronecker product
                block_k = jnp.kron(jnp.eye(G_k), Sigma_k_inv)
                size_k = G_k * d_k
                Gamma_inv = Gamma_inv.at[
                    factor_offset : factor_offset + size_k,
                    factor_offset : factor_offset + size_k,
                ].set(block_k)

                # log|Γ_k| = G_k · log|Σ_k| = G_k · 2 · Σᵢ log(Lₖ[i,i])
                log_det_Sigma_k = 2.0 * jnp.sum(jnp.log(jnp.diag(L_k)))
                log_det_Gamma = log_det_Gamma + G_k * log_det_Sigma_k

                factor_offset += size_k
                theta_offset += n_chol_k

            # Henderson C₂₂ = Z'Z + Γ⁻¹
            C22 = ZtZ + Gamma_inv  # (q, q)
            C22_chol = jnp.linalg.cholesky(C22)  # (q, q)

            # Schur complement S = X'X − X'Z C₂₂⁻¹ Z'X  (= X'Ṽ⁻¹X)
            C22_inv_ZtX = jax.scipy.linalg.cho_solve((C22_chol, True), XtZ.T)
            S = XtX - XtZ @ C22_inv_ZtX  # (p, p)

            # β̂ = S⁻¹(X'y − X'Z C₂₂⁻¹ Z'y)
            C22_inv_Zty = jax.scipy.linalg.cho_solve((C22_chol, True), Zty)
            rhs = Xty - XtZ @ C22_inv_Zty  # (p,)
            beta_hat = jnp.linalg.solve(S, rhs)  # (p,)

            # Quadratic form Q = r'Ṽ⁻¹r via Woodbury:
            #   Q = ‖r‖² − (Z'r)' C₂₂⁻¹ (Z'r)
            w = Zty - XtZ.T @ beta_hat  # (q,)
            r_norm_sq = yty - 2.0 * Xty @ beta_hat + beta_hat @ XtX @ beta_hat
            C22_inv_w = jax.scipy.linalg.cho_solve((C22_chol, True), w)
            Q = r_norm_sq - w @ C22_inv_w  # scalar

            # log|Ṽ| = log|C₂₂| + log|Γ|
            log_det_C22 = 2.0 * jnp.sum(jnp.log(jnp.diag(C22_chol)))
            log_det_V_tilde = log_det_C22 + log_det_Gamma

            # log|S| = log|X'Ṽ⁻¹X|
            _, log_det_S = jnp.linalg.slogdet(S)

            # Profile REML NLL
            nll = 0.5 * ((n - p) * jnp.log(Q) + log_det_V_tilde + log_det_S)
            # NaN sentinel: failed Cholesky or negative Q → large
            # but finite value so the LM-Nielsen solver rejects the
            # step and increases λ rather than stalling on NaN.
            nll = jnp.where(jnp.isfinite(nll), nll, 1e30)
            return jnp.asarray(nll)

        return reml_nll

    @dataclass(frozen=True)
    class REMLResult:
        """Container for Henderson REML estimation results.

        All arrays are plain numpy (not JAX) — the solver converts
        back after optimisation so downstream code never needs JAX.
        """

        beta: np.ndarray  # (p,) fixed-effect coefficients [intercept first]
        sigma2: float  # residual variance σ̂²
        re_covariances: tuple[np.ndarray, ...]  # σ̂²·Σ_k per factor (d_k, d_k)
        log_chol: np.ndarray  # optimised log-Cholesky params θ̂
        projection: np.ndarray  # (p, n) projection matrix A, σ²-free
        C22: np.ndarray  # (q, q) Henderson C₂₂ = Z'Z + Γ⁻¹
        converged: bool
        n_iter: int
        nll: float  # final REML NLL value

    def _reml_newton_solve(
        nll_fn: Callable[[jnp.ndarray], jnp.ndarray],
        total_chol_params: int,
        max_iter: int,
        tol: float,
    ) -> tuple[jnp.ndarray, float, bool, int]:
        r"""Levenberg–Marquardt–Nielsen solver for REML / Laplace NLL.

        Unlike the GLM solvers (which use static damping on convex
        objectives via :func:`_newton_solve_static`), this solver
        handles the **non-convex** REML and Laplace marginal NLL
        where the Hessian can be indefinite at θ = 0 when d_k > 1
        (random slopes).

        Algorithm (Nielsen 1999)
        ~~~~~~~~~~~~~~~~~~~~~~~~
        Each iteration solves the damped system:

        .. math::
            (H + \lambda I)\,\delta = g

        and evaluates the **gain ratio**:

        .. math::
            \rho = \frac{f(\theta) - f(\theta - \delta)}
                        {\tfrac{1}{2}\,\delta^\top(\lambda\,\delta + g)}

        * :math:`\rho > 0` → step accepted;
          :math:`\lambda \leftarrow \lambda \cdot
          \max\!\bigl(\tfrac{1}{3},\;1 - (2\rho - 1)^3\bigr)`,
          :math:`\nu \leftarrow 2`.
        * :math:`\rho \le 0` → step rejected (NLL did not decrease);
          :math:`\lambda \leftarrow \lambda \cdot \nu`,
          :math:`\nu \leftarrow 2\nu`.

        All branching uses ``jnp.where`` so the solver stays inside
        ``jax.lax.while_loop`` (XLA-traceable, no Python control
        flow).

        Initial damping from the Hessian spectrum
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Rather than hard-coding :math:`\lambda_0`, the solver
        evaluates the Hessian at :math:`\theta_0 = 0` and derives
        all tuning constants from the eigenvalues (Gill–Murray
        modification):

        * :math:`\lambda_0 = |\lambda_{\min}(H_0)|
          + \varepsilon\,\|H_0\|_2` if :math:`H_0` is indefinite,
          else :math:`\varepsilon\,\|H_0\|_2` (near-Newton).
        * :math:`\lambda_{\min} =
          \varepsilon_{f64} \cdot \|H_0\|_2` (machine-precision
          floor — below this, damping is numerically zero).
        * :math:`\lambda_{\max} = 10^{16}\,\|H_0\|_2`
          (overflow guard — standard LM has no upper bound;
          this prevents floating-point overflow while never
          constraining the algorithm in practice).

        NaN sentinel
        ~~~~~~~~~~~~
        If the NLL evaluates to NaN or ±Inf (e.g. from a failed
        Cholesky inside the profile REML closure), the value is
        replaced with ``1e30``.  This produces :math:`\rho < 0`
        (reject), so the solver increases :math:`\lambda` and
        retries — no special-case handling needed.

        Args:
            nll_fn: Pure JAX function ``f(θ) → scalar`` — profile
                REML NLL from :func:`_build_reml_nll` or Laplace
                NLL from :func:`_build_laplace_nll`.
            total_chol_params: Number of log-Cholesky parameters.
            max_iter: Maximum outer iterations.
            tol: Convergence threshold applied to gradient, step,
                and relative NLL change (gated on acceptance).

        Returns:
            ``(params, nll, converged, n_iter)`` tuple.
        """
        _nll_jit = jit(nll_fn)
        _grad_fn = jit(grad(nll_fn))
        _hess_fn = jit(hessian(nll_fn))

        # ---- Derive λ₀, λ_min, λ_max from H(θ₀) eigenvalues ----
        theta_0 = jnp.zeros(total_chol_params, dtype=jnp.float64)
        H_0 = _hess_fn(theta_0)
        eigs_0 = jnp.linalg.eigvalsh(H_0)
        min_eig = jnp.min(eigs_0)
        spectral_norm = jnp.max(jnp.abs(eigs_0))

        # Guard against degenerate case where spectral_norm ≈ 0
        spectral_norm = jnp.maximum(spectral_norm, 1.0)
        eps_f64 = jnp.finfo(jnp.float64).eps  # ≈ 2.2e-16

        # Gill–Murray: if H₀ is PD, near-zero damping; else shift
        lambda_init = jnp.where(
            min_eig > 0.0,
            1e-6 * spectral_norm,  # PD → near-Newton
            jnp.abs(min_eig) + 1e-6 * spectral_norm,  # indefinite → shift
        )
        lambda_min = eps_f64 * spectral_norm  # machine-precision floor
        lambda_max = 1e16 * spectral_norm  # overflow guard (no algorithmic cap)

        # Evaluate NLL at the starting point
        nll_0 = _nll_jit(theta_0)
        nll_0 = jnp.where(jnp.isfinite(nll_0), nll_0, 1e30)

        identity = jnp.eye(total_chol_params, dtype=jnp.float64)

        # State: (i, params, nll_prev, converged, lambda_, nu)
        init_state = (
            jnp.array(0),
            theta_0,
            nll_0,
            jnp.array(False),
            lambda_init,
            jnp.array(2.0, dtype=jnp.float64),  # Nielsen ν
        )

        def cond(state: tuple) -> jax.Array:
            i, _params, _nll, converged, _lam, _nu = state
            result: jax.Array = (i < max_iter) & (~converged)
            return result

        def body(state: tuple) -> tuple:
            i, params, nll_prev, _, lambda_, nu = state

            g = _grad_fn(params)
            H = _hess_fn(params)
            H_damped = H + lambda_ * identity
            step = jnp.linalg.solve(H_damped, g)
            params_new = params - step

            nll_new = _nll_jit(params_new)
            # NaN sentinel: failed Cholesky → reject step
            nll_new = jnp.where(jnp.isfinite(nll_new), nll_new, 1e30)

            # ---- Gain ratio (Nielsen 1999) ----
            # predicted = 0.5·δᵀg + 0.5·λ·‖δ‖²  where δ = step
            # Derivation: model reduction m(0)−m(δ) with (H+λI)δ = g
            # gives δᵀg − 0.5·δᵀHδ = 0.5·δᵀg + 0.5·λ·‖δ‖²
            predicted = 0.5 * step @ (lambda_ * step + g)
            actual = nll_prev - nll_new
            rho = jnp.where(
                jnp.abs(predicted) > 1e-30,
                actual / predicted,
                jnp.array(0.0, dtype=jnp.float64),
            )

            accept = rho > 0.0

            # ---- Nielsen λ update ----
            # Accept: λ *= max(1/3, 1 − (2ρ − 1)³), ν = 2
            # Reject: λ *= ν, ν *= 2
            shrink = jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            lambda_accept = lambda_ * shrink
            lambda_reject = lambda_ * nu

            lambda_new = jnp.where(accept, lambda_accept, lambda_reject)
            lambda_new = jnp.clip(lambda_new, lambda_min, lambda_max)
            nu_new = jnp.where(accept, 2.0, nu * 2.0)

            # ---- Accept / reject step (jnp.where, no branching) ----
            params_out = jnp.where(accept, params_new, params)
            nll_out = jnp.where(accept, nll_new, nll_prev)

            # ---- Convergence (only meaningful on accepted steps) ----
            grad_small = jnp.max(jnp.abs(g)) < tol
            step_small = jnp.max(jnp.abs(step)) < tol
            nll_rel = jnp.abs(actual) / jnp.maximum(jnp.abs(nll_prev), 1.0)
            func_small = nll_rel < tol

            converged = (
                accept
                & (grad_small | step_small | func_small)
                & jnp.all(jnp.isfinite(params_new))
            )

            return (i + 1, params_out, nll_out, converged, lambda_new, nu_new)

        final_state = jax.lax.while_loop(cond, body, init_state)
        n_iter_final, params_final, nll_final, converged, _, _ = final_state

        return params_final, float(nll_final), bool(converged), int(n_iter_final)

    def _reml_solve(
        X_raw: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        re_struct: list[tuple[int, int]],
        *,
        fit_intercept: bool = True,
        max_iter: int = 500,
        tol: float = _DEFAULT_TOL,
        solver: str = "newton",
    ) -> REMLResult:
        """Henderson-based REML solver with JAX autodiff.

        Supports two solver backends:

        * ``"newton"`` (default) — Levenberg–Marquardt–Nielsen with
          adaptive damping via ``jax.lax.while_loop``.  Uses
          ``jax.grad`` for the exact gradient, ``jax.hessian`` for
          the exact Hessian, and the gain-ratio λ update (Nielsen
          1999) for robust convergence even when the Hessian is
          indefinite at the starting point (e.g. random slopes with
          d_k > 1).  See :func:`_reml_newton_solve`.

        * ``"lbfgsb"`` — L-BFGS-B via ``scipy.optimize.minimize``
          with exact gradients from ``jax.grad``.  Uses a quasi-
          Newton Hessian approximation (no ``jax.hessian`` needed).
          More robust to indefinite curvature at the starting point
          but slower convergence for well-conditioned problems.

        The log-Cholesky parameterisation makes the problem
        unconstrained, which is suitable for both solvers.

        Args:
            X_raw: Fixed-effect design matrix (n, p), WITHOUT intercept.
            Z: Random-effect design matrix (n, q).
            y: Response vector (n,).
            re_struct: ``[(G_1, d_1), …, (G_K, d_K)]`` — groups and
                RE dimension per factor.
            fit_intercept: Whether to prepend an intercept column.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            solver: ``"newton"`` or ``"lbfgsb"``.

        Returns:
            ``REMLResult`` with β̂, σ̂², per-factor covariance
            matrices, C₂₂, projection A, and convergence diagnostics.
        """
        X = _augment_intercept(X_raw, fit_intercept)

        n, p = X.shape
        q = Z.shape[1]
        n_components = len(re_struct)

        # Total Cholesky parameters: Σ_k d_k(d_k+1)/2
        total_chol_params = sum(d_k * (d_k + 1) // 2 for _, d_k in re_struct)

        # Move to JAX float64 arrays
        X_j = jnp.array(X, dtype=jnp.float64)
        Z_j = jnp.array(Z, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)

        # Build REML NLL — a pure function of log_chol_params only
        nll_fn = _build_reml_nll(X_j, Z_j, y_j, re_struct)

        if solver == "newton":
            # ── Levenberg–Marquardt–Nielsen with adaptive damping ──
            params_jax, nll_final, converged_flag, n_iter_final = _reml_newton_solve(
                nll_fn, total_chol_params, max_iter, tol
            )
            params_np = np.asarray(params_jax)
        elif solver == "lbfgsb":
            # ── L-BFGS-B via scipy ──
            from scipy.optimize import minimize as sp_minimize

            _grad_fn = jit(grad(nll_fn))
            _nll_jit = jit(nll_fn)

            x0 = np.zeros(total_chol_params, dtype=np.float64)

            def objective(params_np: np.ndarray) -> tuple[float, np.ndarray]:
                """Return (NLL, gradient) for scipy L-BFGS-B."""
                p_jax = jnp.array(params_np, dtype=jnp.float64)
                val = float(_nll_jit(p_jax))
                g = np.asarray(_grad_fn(p_jax), dtype=np.float64)
                return val, g

            result = sp_minimize(
                objective,
                x0,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
            )

            params_np = result.x
            converged_flag = result.success
            n_iter_final = result.nit
            nll_final = result.fun
        else:
            msg = f"Unknown REML solver: {solver!r}. Use 'newton' or 'lbfgsb'."
            raise ValueError(msg)

        # Rebuild Γ⁻¹ from optimal Cholesky factors (numpy)
        Gamma_inv = np.zeros((q, q))
        L_factors: list[np.ndarray] = []  # save for covariance extraction
        factor_offset = 0
        theta_offset = 0
        for k in range(n_components):
            G_k, d_k = re_struct[k]
            n_chol_k = d_k * (d_k + 1) // 2
            theta_k = params_np[theta_offset : theta_offset + n_chol_k]

            L_k = _fill_lower_triangular_np(theta_k, d_k)
            L_factors.append(L_k)
            Sigma_k = L_k @ L_k.T
            Sigma_k_inv = np.linalg.solve(Sigma_k, np.eye(d_k))

            block_k = np.kron(np.eye(G_k), Sigma_k_inv)
            size_k = G_k * d_k
            Gamma_inv[
                factor_offset : factor_offset + size_k,
                factor_offset : factor_offset + size_k,
            ] = block_k

            factor_offset += size_k
            theta_offset += n_chol_k

        # Sufficient statistics (numpy — for post-convergence recovery)
        XtX_np = X.T @ X
        XtZ_np = X.T @ Z
        ZtZ_np = Z.T @ Z
        Xty_np = X.T @ y
        Zty_np = Z.T @ y

        # Henderson C₂₂ = Z'Z + Γ⁻¹
        C22 = ZtZ_np + Gamma_inv

        # Schur complement S = X'Ṽ⁻¹X
        C22_inv_ZtX = np.linalg.solve(C22, XtZ_np.T)  # (q, p)
        S = XtX_np - XtZ_np @ C22_inv_ZtX  # (p, p)

        # β̂
        C22_inv_Zty = np.linalg.solve(C22, Zty_np)
        rhs = Xty_np - XtZ_np @ C22_inv_Zty
        beta_hat = np.linalg.solve(S, rhs)

        # Quadratic form Q → profiled σ̂²
        w = Zty_np - XtZ_np.T @ beta_hat
        r_norm_sq = (
            float(y @ y) - 2.0 * Xty_np @ beta_hat + beta_hat @ XtX_np @ beta_hat
        )
        C22_inv_w = np.linalg.solve(C22, w)
        Q = float(r_norm_sq - w @ C22_inv_w)
        sigma2_hat = Q / (n - p)

        # Per-factor covariance matrices: σ̂² · Σ̂_k
        re_covariances: list[np.ndarray] = []
        for L_k in L_factors:
            Sigma_k_ratio = L_k @ L_k.T
            re_covariances.append(sigma2_hat * Sigma_k_ratio)

        # Projection matrix A = S⁻¹ X'Ṽ⁻¹  (σ²-free)
        C22_inv_Zt = np.linalg.solve(C22, Z.T)  # (q, n)
        Xt_Vtilde_inv = X.T - XtZ_np @ C22_inv_Zt  # (p, n)
        A = np.linalg.solve(S, Xt_Vtilde_inv)  # (p, n)

        return REMLResult(
            beta=beta_hat,
            sigma2=float(sigma2_hat),
            re_covariances=tuple(re_covariances),
            log_chol=params_np,
            projection=A,
            C22=C22,
            converged=bool(converged_flag),
            n_iter=int(n_iter_final),
            nll=float(nll_final),
        )

    # -------------------------------------------------------------- #
    # GLMM: Robust fixed-effects GLM via IRLS (pure NumPy)
    # -------------------------------------------------------------- #

    @dataclass(frozen=True)
    class _GLMIRLSResult:
        """Result from a pure-NumPy fixed-effects GLM fit via IRLS.

        Stores everything needed by both the Laplace warm-start
        path (``_laplace_solve`` captures ``beta`` as a JAX
        constant) and the reduced-model path in
        ``LogisticMixedFamily.fit()`` / ``PoissonMixedFamily.fit()``
        (which needs ``eta``, ``mu``, ``beta``).
        """

        beta: np.ndarray  # (p,) coefficients [intercept first if present]
        eta: np.ndarray  # (n,) linear predictor at convergence
        mu: np.ndarray  # (n,) fitted values on response scale
        converged: bool
        n_iter: int

    def _logistic_nll_np(
        y: np.ndarray,
        eta: np.ndarray,
    ) -> float:
        """Pure-NumPy logistic NLL for step-halving deviance check.

        Uses the numerically stable logaddexp form:
        ``NLL = Σ log(1 + exp(-s_i · η_i))``,  s_i = 2y_i − 1.
        """
        return float(np.sum(np.logaddexp(0.0, -eta * (2.0 * y - 1.0))))

    def _poisson_nll_np(
        y: np.ndarray,
        eta: np.ndarray,
    ) -> float:
        """Pure-NumPy Poisson NLL for step-halving deviance check.

        ``NLL = Σ [exp(η) − y·η]`` (constant log(y!) dropped).
        """
        return float(np.sum(np.exp(eta) - y * eta))

    def _fit_glm_irls(
        X: np.ndarray,
        y: np.ndarray,
        working_fn: Callable[
            [jnp.ndarray, jnp.ndarray],
            tuple[jnp.ndarray, jnp.ndarray],
        ],
        nll_fn: Callable[[np.ndarray, np.ndarray], float],
        family: str,
        *,
        max_iter: int = 25,
        tol: float = 1e-8,
        max_step_halve: int = 5,
    ) -> _GLMIRLSResult:
        r"""Fit a fixed-effects GLM via IRLS with step-halving.

        IRLS (Iteratively Reweighted Least Squares) is the canonical
        algorithm for maximum-likelihood estimation in GLMs.
        Ref: McCullagh & Nelder (1989), *Generalized Linear Models*,
        2nd ed., Chapman & Hall, §2.5.

        Robust replacement for the earlier ``_glm_warmstart``.
        Serves two purposes:

        1. **Laplace warm-start** — provides :math:`\hat\beta_{GLM}`
           for :func:`_build_laplace_nll` (lme4 §2.3 strategy).
        2. **Reduced-model fit** — the ``fit()`` method on GLMM
           family classes needs a fixed-effects GLM when X has fewer
           columns than calibration.  By returning a full
           ``_GLMIRLSResult`` (with ``eta``, ``mu``), callers avoid
           a second pass.

        **Not traced by JAX** — runs in NumPy outside the
        ``jax.grad`` / ``jax.hessian`` computation graph.

        Improvements over the old ``_glm_warmstart``
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        * **Step-halving** (Marschner 2011, ``glm2``): if the NLL
          increases after an IRLS step, the step direction
          :math:`\Delta\beta = \beta_{new} - \beta_{old}` is halved
          up to ``max_step_halve`` times.  This is R's ``glm.fit()``
          approach and prevents IRLS oscillation on Poisson data
          with large counts or logistic data with sparse events.
        * **Dual convergence**: both :math:`\|\Delta\beta\|_\infty`
          AND relative deviance change :math:`|\Delta d|/(|d|+0.1)`
          are checked.  Either triggers convergence.
        * **Returns full result** — ``_GLMIRLSResult(beta, eta, mu,
          converged, n_iter)`` so callers never need to recompute
          ``mu`` from ``eta``.

        Args:
            X: Design matrix ``(n, p)`` WITH intercept column.
            y: Response vector ``(n,)``.
            working_fn: ``(y, η) → (z̃, w)`` — family-specific IRLS
                working response and weights (JAX-traced function,
                called with JAX arrays, output converted to NumPy).
            nll_fn: ``(y, η) → float`` — pure-NumPy NLL for deviance
                monitoring.  One of ``_logistic_nll_np`` or
                ``_poisson_nll_np``.
            family: ``"logistic"`` or ``"poisson"`` — used to compute
                ``mu`` via the correct inverse link at convergence.
            max_iter: Maximum IRLS iterations (default 25).
            tol: Convergence threshold for ``|Δβ|∞`` and relative
                deviance change (default 1e-8).
            max_step_halve: Maximum step halvings per iteration
                (default 5).  0 disables step control.

        Returns:
            ``_GLMIRLSResult`` with ``beta``, ``eta``, ``mu``,
            convergence flag, and iteration count.
        """
        n, p = X.shape

        # ---- Initialisation: OLS on log(y + 0.5) ----
        # Poisson: log(y + 0.5) ≈ log(y) — right scale for log-link.
        # Logistic (y ∈ {0,1}): log(0.5) ≈ −0.69, log(1.5) ≈ 0.41
        #   — reasonable logit-scale approximation.
        y_safe = np.maximum(y, 0.0)
        eta_init = np.log(y_safe + 0.5)
        XtX = X.T @ X + 1e-8 * np.eye(p)
        Xty = X.T @ eta_init
        beta: np.ndarray = np.linalg.solve(XtX, Xty)

        eta = np.clip(X @ beta, -20.0, 20.0)
        dev = nll_fn(y, eta)
        converged = False

        for _it in range(max_iter):
            # Working response and weights (via JAX fns, converted)
            z_j, w_j = working_fn(
                jnp.array(y, dtype=jnp.float64),
                jnp.array(eta, dtype=jnp.float64),
            )
            z_np: np.ndarray = np.asarray(z_j)
            w_np: np.ndarray = np.clip(np.asarray(w_j), 1e-10, 1e6)

            # Weighted least squares
            sqrt_w = np.sqrt(w_np)
            X_w = X * sqrt_w[:, None]
            z_w = z_np * sqrt_w
            beta_new: np.ndarray
            beta_new, _, _, _ = np.linalg.lstsq(X_w, z_w, rcond=None)

            if not np.all(np.isfinite(beta_new)):
                break  # return last good beta

            # ---- Step-halving (Marschner 2011, glm2) ----
            # If NLL increased, halve the step direction.
            eta_new = np.clip(X @ beta_new, -20.0, 20.0)
            dev_new = nll_fn(y, eta_new)

            for _halve in range(max_step_halve):
                if dev_new <= dev + 1e-12:
                    break  # step is acceptable
                # Halve: β ← β + 0.5·(β_new − β)
                beta_new = 0.5 * (beta + beta_new)
                eta_new = np.clip(X @ beta_new, -20.0, 20.0)
                dev_new = nll_fn(y, eta_new)

            # ---- Convergence check ----
            delta_beta = float(np.max(np.abs(beta_new - beta)))
            delta_dev = abs(dev_new - dev) / (abs(dev) + 0.1)

            beta = beta_new
            eta = eta_new
            dev = dev_new

            if delta_beta < tol or delta_dev < tol:
                converged = True
                break

        # ---- Compute mu via the correct inverse link ----
        if family == "logistic":
            mu = 1.0 / (1.0 + np.exp(-eta))
        elif family == "poisson":
            mu = np.exp(np.clip(eta, -20.0, 20.0))
        else:
            msg = f"Unknown family: {family!r}"
            raise ValueError(msg)

        return _GLMIRLSResult(
            beta=beta,
            eta=eta,
            mu=mu,
            converged=converged,
            n_iter=_it + 1 if "_it" in dir() else 0,
        )

    @dataclass(frozen=True)
    class LaplaceResult:
        """Container for Laplace GLMM estimation results.

        All arrays are plain numpy (not JAX) — the solver converts
        back after optimisation so downstream code never needs JAX.

        Compared to ``REMLResult``, this stores IRLS quantities
        (W, mu, V_inv_diag, fisher_info) needed for the one-step
        corrector in the permutation loop, plus the BLUPs u.
        There is no ``sigma2`` (GLMM has no residual variance
        parameter) and no ``projection`` (the score projection
        uses ``V_inv_diag`` and ``fisher_info`` instead).
        """

        beta: np.ndarray  # (p,) fixed-effect coefficients [intercept first]
        u: np.ndarray  # (q,) random-effect BLUPs
        re_covariances: tuple[np.ndarray, ...]  # Σ̂_k per factor (d_k, d_k)
        log_chol: np.ndarray  # optimised log-Cholesky params θ̂
        W: np.ndarray  # (n,) IRLS weights at convergence
        mu: np.ndarray  # (n,) fitted values (conditional mean) at convergence
        V_inv_diag: np.ndarray  # (n,) diagonal of approx V⁻¹ for score projection
        fisher_info: np.ndarray  # (p, p) Fisher information for one-step corrector
        C22: np.ndarray  # (q, q) weighted Henderson C₂₂ = Z̃'Z̃ + Γ⁻¹
        Z: np.ndarray  # (n, q) random-effect design matrix (stored for rebuild)
        converged: bool
        n_iter_outer: int
        n_iter_inner_total: int
        nll: float  # final Laplace NLL value

    def _pql_fixed_irls_vmap(
        X: np.ndarray,
        Z: np.ndarray,
        y_perm: np.ndarray,
        Gamma_inv: np.ndarray,
        beta_init: np.ndarray,
        working_fn: Callable[
            [jnp.ndarray, jnp.ndarray],
            tuple[jnp.ndarray, jnp.ndarray],
        ],
        *,
        max_inner: int = 20,
    ) -> np.ndarray:
        r"""PQL-fixed IRLS at fixed θ̂, vmapped across permutations.

        For each permutation π, runs the full IRLS inner loop to
        convergence on `y_π` while holding variance components
        (Γ⁻¹) fixed from the null model.  Produces exact
        β̂_π(θ̂_fixed) — not a one-step approximation.

        The inner loop is the **same Henderson algebra** as
        :func:`_build_laplace_nll` — pre-scale by √W, solve the
        unweighted Henderson system, iterate.  Unrolled (not
        ``lax.while_loop``) for ``jax.vmap`` compatibility.

        Warm-start at ``beta_init`` (from the null-model GLM fit)
        ensures convergence in 3–8 iterations.  η-clipping to
        ``[-20, 20]`` and weight capping to ``[1e-10, 1e6]``
        match the guards in :func:`_build_laplace_nll`.

        Args:
            X: Fixed-effect design ``(n, p)``, WITH intercept.
            Z: Random-effect design ``(n, q)``.
            y_perm: Permuted responses ``(B, n)``.
            Gamma_inv: Block-diagonal RE precision ``(q, q)`` from
                the null model (fixed).
            beta_init: ``(p,)`` warm-start coefficients.
            working_fn: Family-specific IRLS ``(y, eta) → (z̃, w)``.
            max_inner: Unrolled IRLS iterations (default 20).

        Returns:
            ``(B, p)`` — full coefficient vectors per permutation.
        """
        X_j = jnp.array(X, dtype=jnp.float64)
        Z_j = jnp.array(Z, dtype=jnp.float64)
        Gamma_inv_j = jnp.array(Gamma_inv, dtype=jnp.float64)
        beta_init_j = jnp.array(beta_init, dtype=jnp.float64)
        q = Z_j.shape[1]

        def _single_perm_irls(y_pi: jnp.ndarray) -> jnp.ndarray:
            """IRLS at fixed Γ⁻¹ for one permuted y — pure JAX."""
            beta = beta_init_j
            u = jnp.zeros(q)

            for _ in range(max_inner):
                eta = jnp.clip(X_j @ beta + Z_j @ u, -20.0, 20.0)
                z_tilde, w = working_fn(y_pi, eta)
                w = jnp.clip(w, 1e-10, 1e6)

                sqrt_w = jnp.sqrt(w)
                X_t = X_j * sqrt_w[:, None]
                Z_t = Z_j * sqrt_w[:, None]
                y_star = z_tilde * sqrt_w

                XtX = X_t.T @ X_t
                XtZ = X_t.T @ Z_t
                Xty = X_t.T @ y_star
                ZtZ = Z_t.T @ Z_t
                Zty = Z_t.T @ y_star

                C22 = ZtZ + Gamma_inv_j
                C22_chol = jnp.linalg.cholesky(C22)

                C22_inv_ZtX = jax.scipy.linalg.cho_solve((C22_chol, True), XtZ.T)
                S = XtX - XtZ @ C22_inv_ZtX

                C22_inv_Zty = jax.scipy.linalg.cho_solve((C22_chol, True), Zty)
                rhs_beta = Xty - XtZ @ C22_inv_Zty
                beta = jnp.linalg.solve(S, rhs_beta)

                rhs_u = Zty - XtZ.T @ beta
                u = jax.scipy.linalg.cho_solve((C22_chol, True), rhs_u)

            return beta

        # vmap across the batch dimension (B permutations)
        beta_perm_j = jax.vmap(_single_perm_irls)(jnp.array(y_perm, dtype=jnp.float64))
        return np.asarray(beta_perm_j)  # (B, p)

    def _build_laplace_nll(
        X: jnp.ndarray,
        Z: jnp.ndarray,
        y: jnp.ndarray,
        re_struct: list[tuple[int, int]],
        working_fn: Callable[
            [jnp.ndarray, jnp.ndarray],
            tuple[jnp.ndarray, jnp.ndarray],
        ],
        cond_nll_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        max_inner: int = 10,
        beta_init: jnp.ndarray | None = None,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Build a pure Laplace marginal NLL function for JAX autodiff.

        The Laplace approximation integrates out the random effects u
        analytically at the mode of the penalised log-likelihood:

        .. math::
            \\ell_L(\\theta) = \\ell(y|\\hat\\beta, \\hat u; \\theta)
                - \\tfrac{1}{2}\\hat u^\\top \\Gamma^{-1} \\hat u
                - \\tfrac{1}{2}\\log|\\Gamma|
                - \\tfrac{1}{2}\\log|H_u|

        where :math:`(\\hat\\beta, \\hat u)` are the joint mode from
        the inner IRLS loop and :math:`H_u` is the Hessian of the
        penalised negative log-likelihood w.r.t. u at the mode.

        The inner IRLS loop is **unrolled** (not ``lax.while_loop``)
        so that ``jax.grad`` / ``jax.hessian`` can differentiate
        through it.  This is standard practice for Laplace — the
        inner loop converges in 3–8 iterations for well-conditioned
        problems, and extra iterations at convergence are identity
        operations.

        Warm-start initialisation
        ~~~~~~~~~~~~~~~~~~~~~~~~~
        The inner IRLS starts at ``beta_init`` (from
        :func:`_fit_glm_irls`) rather than :math:`\\beta = 0`.
        This is critical for Poisson data where :math:`\\exp(0) = 1`
        creates catastrophic working responses when the true counts
        are large.  From a warm start the unrolled IRLS converges
        in 2–4 iterations.  ``beta_init`` is a **captured constant**
        (not a function of the optimised :math:`\\theta`) so it does
        not affect the JAX computation graph.

        Numerical guards
        ~~~~~~~~~~~~~~~~
        * **η-clipping** to ``[-20, 20]`` before computing working
          responses — :math:`\\exp(20) \\approx 5 \\times 10^8` is
          large but finite, preventing overflow in the Poisson
          link.
        * **Weight capping** to ``[1e-10, 1e6]`` — prevents
          degenerate IRLS weights from making the Henderson system
          ill-conditioned.

        Args:
            X: Fixed-effect design matrix ``(n, p)``, WITH intercept.
            Z: Random-effect design matrix ``(n, q)``.
            y: Response vector ``(n,)``.
            re_struct: ``[(G_1, d_1), …]`` — groups and RE dimension
                per factor.
            working_fn: ``(y, eta) → (z_tilde, w)`` — IRLS working
                response and weights (family-specific).
            cond_nll_fn: ``(y, eta) → scalar`` — negative conditional
                log-likelihood (family-specific).
            max_inner: Number of unrolled IRLS iterations.
            beta_init: ``(p,)`` warm-start coefficients from
                :func:`_fit_glm_irls`.  If ``None``, falls back
                to ``jnp.zeros(p)`` (appropriate for logistic but
                unstable for Poisson).

        Returns:
            Pure function ``f(log_chol_params) → scalar`` Laplace NLL.
        """
        n, p = X.shape
        q = Z.shape[1]
        n_components = len(re_struct)

        # Capture warm-start as a JAX constant
        _beta_init = beta_init if beta_init is not None else jnp.zeros(p)

        def laplace_nll(log_chol_params: jnp.ndarray) -> jnp.ndarray:
            """Laplace marginal NLL — pure function of θ."""
            # ---- Build block-diagonal Γ⁻¹ and log|Γ| ----
            Gamma_inv = jnp.zeros((q, q))
            log_det_Gamma = jnp.zeros(())

            factor_offset = 0
            theta_offset = 0
            for k in range(n_components):
                G_k, d_k = re_struct[k]
                n_chol_k = d_k * (d_k + 1) // 2
                theta_k = log_chol_params[theta_offset : theta_offset + n_chol_k]

                L_k = _fill_lower_triangular_jax(theta_k, d_k)
                Sigma_k_inv = jax.scipy.linalg.cho_solve((L_k, True), jnp.eye(d_k))

                block_k = jnp.kron(jnp.eye(G_k), Sigma_k_inv)
                size_k = G_k * d_k
                Gamma_inv = Gamma_inv.at[
                    factor_offset : factor_offset + size_k,
                    factor_offset : factor_offset + size_k,
                ].set(block_k)

                log_det_Sigma_k = 2.0 * jnp.sum(jnp.log(jnp.diag(L_k)))
                log_det_Gamma = log_det_Gamma + G_k * log_det_Sigma_k

                factor_offset += size_k
                theta_offset += n_chol_k

            # ---- Inner IRLS loop (unrolled) ----
            # Initialise β at GLM warm-start, u at zero
            beta = _beta_init
            u = jnp.zeros(q)

            for _irls_iter in range(max_inner):
                # Linear predictor (clipped for numerical safety)
                eta = jnp.clip(X @ beta + Z @ u, -20.0, 20.0)  # (n,)

                # Working response and weights (family-specific)
                z_tilde, w = working_fn(y, eta)  # (n,), (n,)
                w = jnp.clip(w, 1e-10, 1e6)  # Cap extreme weights

                # Pre-scale by √W → unweighted Henderson system
                sqrt_w = jnp.sqrt(w)  # (n,)
                X_tilde = X * sqrt_w[:, None]  # (n, p)
                Z_tilde = Z * sqrt_w[:, None]  # (n, q)
                y_star = z_tilde * sqrt_w  # (n,)

                # Sufficient statistics for unweighted Henderson
                XtX = X_tilde.T @ X_tilde  # (p, p)
                XtZ = X_tilde.T @ Z_tilde  # (p, q)
                ZtZ = Z_tilde.T @ Z_tilde  # (q, q)
                Xty = X_tilde.T @ y_star  # (p,)
                Zty = Z_tilde.T @ y_star  # (q,)

                # Henderson C₂₂ = Z̃'Z̃ + Γ⁻¹
                C22 = ZtZ + Gamma_inv  # (q, q)
                C22_chol = jnp.linalg.cholesky(C22)

                # Schur complement S = X̃'X̃ − X̃'Z̃ C₂₂⁻¹ Z̃'X̃
                C22_inv_ZtX = jax.scipy.linalg.cho_solve((C22_chol, True), XtZ.T)
                S = XtX - XtZ @ C22_inv_ZtX  # (p, p)

                # β̂ = S⁻¹(X̃'ỹ* − X̃'Z̃ C₂₂⁻¹ Z̃'ỹ*)
                C22_inv_Zty = jax.scipy.linalg.cho_solve((C22_chol, True), Zty)
                rhs_beta = Xty - XtZ @ C22_inv_Zty  # (p,)
                beta = jnp.linalg.solve(S, rhs_beta)

                # û = C₂₂⁻¹(Z̃'ỹ* − Z̃'X̃ β̂)
                rhs_u = Zty - XtZ.T @ beta  # (q,)
                u = jax.scipy.linalg.cho_solve((C22_chol, True), rhs_u)

            # ---- Laplace NLL at the mode ----
            # Final linear predictor
            eta_final = X @ beta + Z @ u

            # Conditional NLL: -ℓ(y|β̂, û)
            neg_loglik = cond_nll_fn(y, eta_final)

            # Penalisation: ½ û'Γ⁻¹û
            penalty = 0.5 * u @ (Gamma_inv @ u)

            # log|H_u| via the last Henderson C₂₂:
            # H_u = Z'WZ + Γ⁻¹ = Z̃'Z̃ + Γ⁻¹ = C₂₂
            # log|C₂₂| = 2 Σ log(diag(Chol(C₂₂)))
            log_det_Hu = 2.0 * jnp.sum(jnp.log(jnp.diag(C22_chol)))

            # log|Γ| penalises large RE variance — without it the
            # optimizer drives σ²_b → ∞ (random effects unpenalised).
            nll = neg_loglik + penalty + 0.5 * log_det_Gamma + 0.5 * log_det_Hu
            # NaN sentinel: see _build_reml_nll for rationale.
            nll = jnp.where(jnp.isfinite(nll), nll, 1e30)
            return jnp.asarray(nll)

        return laplace_nll

    def _laplace_solve(
        X_raw: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        re_struct: list[tuple[int, int]],
        *,
        working_fn: Callable[
            [jnp.ndarray, jnp.ndarray],
            tuple[jnp.ndarray, jnp.ndarray],
        ],
        cond_nll_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        family: str,
        fit_intercept: bool = True,
        max_iter: int = 500,
        max_inner: int = 10,
        tol: float = _DEFAULT_TOL,
    ) -> LaplaceResult:
        """Laplace GLMM solver via outer Newton on the marginal NLL.

        Architecture
        ~~~~~~~~~~~~
        1. **GLM warm-start** — fit a fixed-effects GLM via IRLS
           (:func:`_fit_glm_irls`) with step-halving to obtain
           :math:`\\hat\\beta_{GLM}`.  This is the ``lme4::glmer()``
           initialisation strategy (Bates et al. 2015, §2.3).
        2. **Build** ``laplace_nll(θ)`` via
           :func:`_build_laplace_nll` — a pure JAX function of the
           log-Cholesky parameters, with the inner IRLS warm-started
           at :math:`\\hat\\beta_{GLM}`.
        3. **Optimise** θ via Levenberg–Marquardt–Nielsen
           (:func:`_reml_newton_solve`).
        4. **Post-convergence** — recover β̂, û, W, μ̂, V⁻¹_diag,
           Fisher information from the optimal θ̂.

        Args:
            X_raw: Fixed-effect design ``(n, p)``, WITHOUT intercept.
            Z: Random-effect design ``(n, q)``.
            y: Response vector ``(n,)``.
            re_struct: ``[(G_k, d_k), …]`` per factor.
            working_fn: Family-specific IRLS function.
            cond_nll_fn: Family-specific conditional NLL.
            family: ``"logistic"`` or ``"poisson"`` — selects the
                NLL function for step-halving and the inverse link
                for computing ``mu`` in the warm-start.
            fit_intercept: Whether to prepend an intercept column.
            max_iter: Maximum outer Newton iterations.
            max_inner: Inner IRLS iterations per outer step.
            tol: Convergence tolerance.

        Returns:
            ``LaplaceResult`` with β̂, û, W, μ̂, V⁻¹_diag,
            Fisher information, and convergence diagnostics.
        """
        X = _augment_intercept(X_raw, fit_intercept)

        n, p = X.shape
        q = Z.shape[1]
        n_components = len(re_struct)

        # Total Cholesky parameters
        total_chol_params = sum(d_k * (d_k + 1) // 2 for _, d_k in re_struct)

        # ---- GLM warm-start (lme4 approach, with step-halving) ----
        _nll_np = _logistic_nll_np if family == "logistic" else _poisson_nll_np
        glm_result = _fit_glm_irls(X, y, working_fn, _nll_np, family=family)
        beta_init_np = glm_result.beta

        # Move to JAX float64 arrays
        X_j = jnp.array(X, dtype=jnp.float64)
        Z_j = jnp.array(Z, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)
        beta_init_j = jnp.array(beta_init_np, dtype=jnp.float64)

        # Build Laplace NLL — warm-started at β̂_GLM
        nll_fn = _build_laplace_nll(
            X_j,
            Z_j,
            y_j,
            re_struct,
            working_fn,
            cond_nll_fn,
            max_inner,
            beta_init=beta_init_j,
        )

        # ── Outer Levenberg–Marquardt–Nielsen on θ ──
        params_jax, nll_final, converged_flag, n_iter_final = _reml_newton_solve(
            nll_fn, total_chol_params, max_iter, tol
        )
        params_np = np.asarray(params_jax)

        # ---- Post-convergence recovery (numpy) ----
        # Rebuild Γ⁻¹ and per-factor covariance matrices
        Gamma_inv = np.zeros((q, q))
        re_covariances_list: list[np.ndarray] = []
        factor_offset = 0
        theta_offset = 0
        for k in range(n_components):
            G_k, d_k = re_struct[k]
            n_chol_k = d_k * (d_k + 1) // 2
            theta_k = params_np[theta_offset : theta_offset + n_chol_k]

            L_k = _fill_lower_triangular_np(theta_k, d_k)
            Sigma_k = L_k @ L_k.T
            re_covariances_list.append(Sigma_k)

            Sigma_k_inv = np.linalg.solve(Sigma_k, np.eye(d_k))
            block_k = np.kron(np.eye(G_k), Sigma_k_inv)
            size_k = G_k * d_k
            Gamma_inv[
                factor_offset : factor_offset + size_k,
                factor_offset : factor_offset + size_k,
            ] = block_k
            factor_offset += size_k
            theta_offset += n_chol_k

        # Run final IRLS pass to recover β̂, û, W, μ̂
        # Start from warm-start β, matching the traced inner loop
        beta = beta_init_np.copy()
        u = np.zeros(q)

        for _irls in range(max_inner):
            eta = np.clip(X @ beta + Z @ u, -20.0, 20.0)
            # Use numpy-compatible evaluation of working response
            eta_j = jnp.array(eta, dtype=jnp.float64)
            y_j_local = jnp.array(y, dtype=jnp.float64)
            z_tilde_j, w_j = working_fn(y_j_local, eta_j)
            z_tilde_np = np.asarray(z_tilde_j)
            w_np: np.ndarray = np.clip(np.asarray(w_j), 1e-10, 1e6)

            # Pre-scale
            sqrt_w = np.sqrt(w_np)
            X_tilde = X * sqrt_w[:, None]
            Z_tilde = Z * sqrt_w[:, None]
            y_star = z_tilde_np * sqrt_w

            # Unweighted Henderson
            XtX = X_tilde.T @ X_tilde
            XtZ_np = X_tilde.T @ Z_tilde
            ZtZ_np = Z_tilde.T @ Z_tilde
            Xty_np = X_tilde.T @ y_star
            Zty_np = Z_tilde.T @ y_star

            C22 = ZtZ_np + Gamma_inv
            C22_inv_ZtX = np.linalg.solve(C22, XtZ_np.T)
            S = XtX - XtZ_np @ C22_inv_ZtX

            C22_inv_Zty = np.linalg.solve(C22, Zty_np)
            rhs_beta = Xty_np - XtZ_np @ C22_inv_Zty
            beta = np.linalg.solve(S, rhs_beta)

            rhs_u = Zty_np - XtZ_np.T @ beta
            u = np.linalg.solve(C22, rhs_u)

        # Final IRLS quantities
        eta_final = X @ beta + Z @ u
        eta_j_final = jnp.array(eta_final, dtype=jnp.float64)
        y_j_final = jnp.array(y, dtype=jnp.float64)
        _, w_final_j = working_fn(y_j_final, eta_j_final)
        W = np.asarray(w_final_j)  # (n,)

        # Fitted values (conditional mean on the response scale)
        # For logistic: mu = sigmoid(eta), for Poisson: mu = exp(eta)
        # We recompute from working_fn to stay generic:
        z_final_j, _ = working_fn(y_j_final, eta_j_final)
        # mu = y - w * (z_tilde - eta)  [from z_tilde = eta + (y-mu)/w]
        mu = np.asarray(y_j_final) - W * (np.asarray(z_final_j) - eta_final)

        # V⁻¹_diag for score projection
        # V⁻¹ ≈ W − W·Z·C₂₂⁻¹·Z'·W  (Woodbury on the last IRLS iterate)
        # For individual score projection we only need the diagonal:
        # V_inv_diag[i] = W[i] - W[i] * (Z[i,:] @ C₂₂⁻¹ @ Z[i,:]') * W[i]
        # More efficiently: V_inv_diag = W - (W * Z @ C₂₂⁻¹ @ Z' * W).diag
        C22_inv_Zt_W = np.linalg.solve(C22, (Z * W[:, None]).T)  # (q, n)
        WZ_C22inv_ZtW_diag = np.sum((Z * W[:, None]) * C22_inv_Zt_W.T, axis=1)  # (n,)
        V_inv_diag = W - WZ_C22inv_ZtW_diag  # (n,)

        # Fisher information: I = X'V⁻¹X (Schur complement)
        # = X'WX - X'WZ C₂₂⁻¹ Z'WX  (from last IRLS iterate)
        # Equivalently: S from the last Henderson solve (already computed)
        fisher_info = S  # (p, p)

        return LaplaceResult(
            beta=beta,
            u=u,
            re_covariances=tuple(re_covariances_list),
            log_chol=params_np,
            W=W,
            mu=mu,
            V_inv_diag=V_inv_diag,
            fisher_info=fisher_info,
            C22=C22,
            Z=Z,
            converged=bool(converged_flag),
            n_iter_outer=int(n_iter_final),
            n_iter_inner_total=int(n_iter_final) * max_inner,
            nll=float(nll_final),
        )

    def _batch_mixed_project(
        projection_A: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Project permuted responses through the GLS projection.

        LMM analogue of the OLS ``pinv(X) @ Y.T`` matmul.  A single
        BLAS-3 multiply produces all B coefficient vectors at once.

        Args:
            projection_A: ``(p, n)`` GLS projection from
                :func:`_reml_solve`.  Includes intercept row when
                ``fit_intercept`` was ``True`` during calibration.
            Y_matrix: ``(B, n)`` permuted response vectors.
            fit_intercept: If ``True``, drop the intercept row
                (index 0) from the output, matching the ``batch_ols``
                convention.

        Returns:
            Slope coefficients ``(B, p−1)`` when
            ``fit_intercept=True``, else ``(B, p)``.
        """
        all_coefs = np.asarray(Y_matrix @ projection_A.T)  # (B, p)
        return all_coefs[:, 1:] if fit_intercept else all_coefs


def _check_convergence(converged: np.ndarray, max_iter: int) -> None:
    """Emit a single summary warning if any solves did not converge.

    Non-converged permutations are **retained** in the null
    distribution — discarding them would bias the p-value
    anti-conservatively (the null would appear tighter than it
    actually is, inflating the false-positive rate).  The warning
    is informational only — the caller should check VIF for
    multicollinearity or inspect the data for quasi-complete
    separation if a large fraction fails.
    """
    n_failed = int(np.sum(~converged))
    if n_failed > 0:
        total = converged.shape[0]
        pct = 100.0 * n_failed / total
        warnings.warn(
            f"{n_failed} of {total} Newton–Raphson solves "
            f"({pct:.1f}%) did not converge within {max_iter} "
            f"iterations. Non-converged permutations are retained "
            f"in the null distribution (conservative). If this "
            f"fraction is large, check for multicollinearity (VIF) "
            f"or quasi-complete separation.",
            RuntimeWarning,
            stacklevel=3,
        )


# ------------------------------------------------------------------ #
# Intercept augmentation / stripping helpers
# ------------------------------------------------------------------ #
#
# Factored out of the 30+ batch_* methods to eliminate per-method
# boilerplate.  These are NumPy-level operations (pre-conversion to
# JAX), so they live outside the ``if _CAN_IMPORT_JAX`` guard.
# ------------------------------------------------------------------ #


def _augment_intercept_2d(X: np.ndarray, fit_intercept: bool) -> np.ndarray:
    """Prepend a ones column to a 2-D design matrix.

    Args:
        X: Design matrix ``(n, p)``.
        fit_intercept: If ``False``, return *X* unchanged.

    Returns:
        ``(n, p+1)`` with intercept column when ``fit_intercept``
        is ``True``, else the original ``(n, p)`` array.
    """
    if fit_intercept:
        return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])
    return X


def _augment_intercept_3d(X_batch: np.ndarray, fit_intercept: bool) -> np.ndarray:
    """Prepend a ones column to each matrix in a 3-D batch.

    Args:
        X_batch: Design matrices ``(B, n, p)``.
        fit_intercept: If ``False``, return *X_batch* unchanged.

    Returns:
        ``(B, n, p+1)`` with intercept column when ``fit_intercept``
        is ``True``, else the original ``(B, n, p)`` array.
    """
    if fit_intercept:
        B, n, _ = X_batch.shape
        return np.concatenate(
            [np.ones((B, n, 1), dtype=X_batch.dtype), X_batch], axis=2
        )
    return X_batch


def _strip_intercept(coefs: np.ndarray, fit_intercept: bool) -> np.ndarray:
    """Remove the intercept column from a batch coefficient matrix.

    Args:
        coefs: Coefficients ``(B, p+1)`` (including intercept) or
            ``(B, p)`` (without).
        fit_intercept: Whether an intercept column is present.

    Returns:
        ``(B, p)`` slope coefficients.
    """
    return coefs[:, 1:] if fit_intercept else coefs


# ------------------------------------------------------------------ #
# JaxBackend
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class JaxBackend:
    """JAX-accelerated compute backend.

    Implements :class:`BackendProtocol` using JAX’s XLA compiler,
    automatic differentiation, and ``vmap`` vectorisation.

    Performance model
    ~~~~~~~~~~~~~~~~~
    * **OLS:** Single ``jnp.linalg.pinv`` + BLAS-3 matmul (same
      algorithmic approach as the NumPy backend, but JIT-compiled).
    * **GLMs (logistic, Poisson, NB2, ordinal, multinomial):**
      ``jax.vmap`` maps the Newton–Raphson solver across all B
      permutations, fusing them into a single XLA kernel launch.
      This eliminates Python-loop overhead and enables SIMD
      parallelism across permutations.

    The frozen dataclass has no mutable state — all per-call data
    flows through method arguments, making instances trivially
    thread-safe and picklable.

    NumPy ↔ JAX conversion boundary
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Every public method follows the same pattern:

    1. Convert inputs: ``X_j = jnp.array(X, dtype=jnp.float64)``
    2. Run JIT-compiled computation
    3. Convert outputs: ``return np.asarray(result)``

    Callers never see JAX types.
    """

    @property
    def name(self) -> str:
        return "jax"

    @property
    def is_available(self) -> bool:  # noqa: PLR6301
        return _CAN_IMPORT_JAX

    # ================================================================ #
    # Generic batch helpers (dispatch layer)
    # ================================================================ #
    #
    # Three private methods — one per vmap pattern — that factor out
    # the boilerplate shared by all GLM ``batch_*`` methods:
    #
    #   _batch_shared_X   — shared X, many Y   (ter Braak)
    #   _batch_varying_X  — many X, shared y   (Kennedy)
    #   _batch_paired     — both X and Y vary   (bootstrap/jackknife)
    #
    # Each accepts a **normalised solver function** with signature:
    #
    #   (X_jax, y_jax, max_iter, tol, min_damping) → (params, nll, converged)
    #
    # Family-specific parameters (alpha, K, …) must be bound in the
    # solver closure *before* passing it here.  The generic helpers
    # handle intercept augmentation, NumPy↔JAX conversion, vmap/jit
    # dispatch, convergence checking, intercept stripping, and the
    # optional ``return_nll`` flag.
    #
    # ``coef_slice`` overrides the default intercept-stripping logic
    # for families that pack extra parameters (e.g. ordinal packs
    # [slopes, thresholds] and needs ``slice(0, n_features)``).
    # ---------------------------------------------------------------- #

    def _batch_shared_X(
        self,
        solver_fn: Callable[..., Any],
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        return_nll: bool,
        coef_slice: slice | None = None,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Shared X, many Y — ter Braak / Freedman–Lane pattern.

        ``vmap`` maps *solver_fn* across the B dimension of
        *Y_matrix*.  The shared design matrix X is captured in the
        closure.

        Args:
            solver_fn: Normalised solver ``(X, y, max_iter, tol,
                min_damping) → (params, nll, converged)``.
            X: Design matrix ``(n, p)`` — no intercept column.
            Y_matrix: Permuted responses ``(B, n)``.
            fit_intercept: Prepend intercept column.
            return_nll: If ``True``, also return ``2·NLL``.
            coef_slice: Optional slice applied to the parameter
                vector *instead of* intercept stripping.
            **kwargs: Forwarded; ``max_iter``, ``tol``,
                ``min_damping`` are read with defaults.

        Returns:
            Slope coefficients ``(B, p)``, or
            ``(coefs, 2·NLL)`` when ``return_nll`` is ``True``.
        """
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        X_aug = _augment_intercept_2d(X, fit_intercept)
        X_j = jnp.array(X_aug, dtype=jnp.float64)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float64)

        def _solve_one(
            y_vec: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return solver_fn(X_j, y_vec, max_iter, tol, min_damping)  # type: ignore[no-any-return]

        all_params, all_nll, all_converged = jit(vmap(_solve_one))(Y_j)
        _check_convergence(np.asarray(all_converged), max_iter)

        coefs = np.asarray(all_params)
        if coef_slice is not None:
            coefs = coefs[:, coef_slice]
        else:
            coefs = _strip_intercept(coefs, fit_intercept)

        if return_nll:
            # 2 × NLL = deviance (up to a data-only constant that
            # cancels in improvement = score_reduced − score_full).
            return coefs, 2.0 * np.asarray(all_nll)
        return coefs

    def _batch_varying_X(
        self,
        solver_fn: Callable[..., Any],
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        return_nll: bool,
        coef_slice: slice | None = None,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Many X, shared y — Kennedy pattern.

        ``vmap`` maps *solver_fn* across the B dimension of
        *X_batch*.  The shared response *y* is captured in the
        closure.

        Args:
            solver_fn: Normalised solver ``(X, y, max_iter, tol,
                min_damping) → (params, nll, converged)``.
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared response ``(n,)``.
            fit_intercept: Prepend intercept column.
            return_nll: If ``True``, also return ``2·NLL``.
            coef_slice: Optional slice applied to the parameter
                vector *instead of* intercept stripping.
            **kwargs: Forwarded; ``max_iter``, ``tol``,
                ``min_damping`` are read with defaults.

        Returns:
            Slope coefficients ``(B, p)``, or
            ``(coefs, 2·NLL)`` when ``return_nll`` is ``True``.
        """
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        X_aug = _augment_intercept_3d(X_batch, fit_intercept)
        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)

        def _solve_one(
            X_single: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return solver_fn(X_single, y_j, max_iter, tol, min_damping)  # type: ignore[no-any-return]

        all_params, all_nll, all_converged = jit(vmap(_solve_one))(X_j)
        _check_convergence(np.asarray(all_converged), max_iter)

        coefs = np.asarray(all_params)
        if coef_slice is not None:
            coefs = coefs[:, coef_slice]
        else:
            coefs = _strip_intercept(coefs, fit_intercept)

        if return_nll:
            return coefs, 2.0 * np.asarray(all_nll)
        return coefs

    def _batch_paired(
        self,
        solver_fn: Callable[..., Any],
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool,
        return_nll: bool,
        coef_slice: slice | None = None,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Both X and Y vary — bootstrap / jackknife pattern.

        ``vmap`` maps *solver_fn* across both *X_batch* and
        *Y_batch* simultaneously.

        Args:
            solver_fn: Normalised solver ``(X, y, max_iter, tol,
                min_damping) → (params, nll, converged)``.
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            Y_batch: Response vectors ``(B, n)``.
            fit_intercept: Prepend intercept column.
            return_nll: If ``True``, also return ``2·NLL``.
            coef_slice: Optional slice applied to the parameter
                vector *instead of* intercept stripping.
            **kwargs: Forwarded; ``max_iter``, ``tol``,
                ``min_damping`` are read with defaults.

        Returns:
            Slope coefficients ``(B, p)``, or
            ``(coefs, 2·NLL)`` when ``return_nll`` is ``True``.
        """
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        X_aug = _augment_intercept_3d(X_batch, fit_intercept)
        X_j = jnp.array(X_aug, dtype=jnp.float64)
        Y_j = jnp.array(Y_batch, dtype=jnp.float64)

        def _solve_one(
            X_single: jax.Array,
            y_vec: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return solver_fn(X_single, y_vec, max_iter, tol, min_damping)  # type: ignore[no-any-return]

        all_params, all_nll, all_converged = jit(vmap(_solve_one))(X_j, Y_j)
        _check_convergence(np.asarray(all_converged), max_iter)

        coefs = np.asarray(all_params)
        if coef_slice is not None:
            coefs = coefs[:, coef_slice]
        else:
            coefs = _strip_intercept(coefs, fit_intercept)

        if return_nll:
            return coefs, 2.0 * np.asarray(all_nll)
        return coefs

    # ================================================================ #
    # OLS — shared X, many Y
    # ================================================================ #
    #
    # JIT-compiled pseudoinverse approach — identical algorithm to the
    # NumPy backend's batch_ols, but compiled to XLA for potential
    # fusion with downstream operations.
    # ---------------------------------------------------------------- #

    def batch_ols(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Batch OLS via JIT-compiled pseudoinverse multiply.

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            Y_matrix: Permuted responses ``(B, n)``.
            fit_intercept: Prepend intercept column.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        X_aug = _augment_intercept_2d(X, fit_intercept)

        X_j = jnp.array(X_aug, dtype=jnp.float64)  # NumPy → JAX
        Y_j = jnp.array(Y_matrix, dtype=jnp.float64)

        @jit
        def _batch_solve(X_mat: jax.Array, Y_mat: jax.Array) -> jax.Array:
            # Single SVD-based pseudoinverse, then BLAS-3 matmul
            # for all B right-hand sides simultaneously.
            pinv = jnp.linalg.pinv(X_mat)  # (p+1, n)
            return (pinv @ Y_mat.T).T  # (B, p+1)

        result = np.asarray(_batch_solve(X_j, Y_j))  # JAX → NumPy
        return _strip_intercept(result, fit_intercept)

    # ================================================================ #
    # Logistic — delegates to generic batch helpers
    # ================================================================ #

    def batch_logistic(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch logistic via vmap'd Newton–Raphson.

        Args:
            X: Design matrix ``(n, p)``.
            Y_matrix: Permuted binary responses ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``max_iter``, ``tol``, ``min_damping``.

        Returns:
            Slope coefficients ``(B, p)``.
        """
        return self._batch_shared_X(  # type: ignore[return-value]
            _make_newton_solver, X, Y_matrix, fit_intercept, False, **kwargs
        )

    def batch_logistic_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch logistic with per-permutation design matrices.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            y: Shared binary response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``max_iter``, ``tol``, ``min_damping``.

        Returns:
            Slope coefficients ``(B, p)``.
        """
        return self._batch_varying_X(  # type: ignore[return-value]
            _make_newton_solver, X_batch, y, fit_intercept, False, **kwargs
        )

    # ================================================================ #
    # OLS — many X, shared y (Kennedy path)
    # ================================================================ #
    #
    # vmap over ``jnp.linalg.lstsq`` — each permutation gets its own
    # design matrix.  Unlike the shared-X OLS path (which uses a
    # single pseudoinverse), each X_b requires its own decomposition.
    # ---------------------------------------------------------------- #

    def batch_ols_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Batch OLS with per-permutation design matrices via vmap.

        Kennedy individual linear path — each permutation has its
        own design matrix (column *j* replaced with permuted exposure
        residuals).  Uses ``jax.vmap`` over ``jnp.linalg.lstsq`` to
        solve all *B* systems in a single XLA kernel launch.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared continuous response ``(n,)``.
            fit_intercept: Prepend intercept column.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        X_aug = _augment_intercept_3d(X_batch, fit_intercept)

        X_j = jnp.array(X_aug, dtype=jnp.float64)  # NumPy → JAX
        y_j = jnp.array(y, dtype=jnp.float64)

        @jit
        def _batch_lstsq(X_all: jax.Array, y_vec: jax.Array) -> jax.Array:
            def _solve_one(X_single: jax.Array) -> jax.Array:
                coefs, _, _, _ = jnp.linalg.lstsq(X_single, y_vec, rcond=None)
                return coefs

            # vmap maps the least-squares solve across B design matrices.
            return vmap(_solve_one)(X_all)

        all_coefs = np.asarray(_batch_lstsq(X_j, y_j))  # JAX → NumPy
        return _strip_intercept(all_coefs, fit_intercept)

    # ================================================================ #
    # Poisson — delegates to generic batch helpers
    # ================================================================ #

    def batch_poisson(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson via vmap'd Newton–Raphson.

        Args:
            X: Design matrix ``(n, p)``.
            Y_matrix: Permuted count responses ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``max_iter``, ``tol``, ``min_damping``.

        Returns:
            Slope coefficients ``(B, p)``.
        """
        return self._batch_shared_X(  # type: ignore[return-value]
            _make_poisson_solver, X, Y_matrix, fit_intercept, False, **kwargs
        )

    def batch_poisson_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson with per-permutation design matrices.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            y: Shared count response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``max_iter``, ``tol``, ``min_damping``.

        Returns:
            Slope coefficients ``(B, p)``.
        """
        return self._batch_varying_X(  # type: ignore[return-value]
            _make_poisson_solver, X_batch, y, fit_intercept, False, **kwargs
        )

    # ================================================================ #
    # NB2 — delegates to generic batch helpers
    # ================================================================ #
    #
    # NB2 requires an ``alpha`` (dispersion) kwarg bound into the
    # solver closure before passing to the generic helpers.
    # ---------------------------------------------------------------- #

    def batch_negbin(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB2 via vmap'd Newton–Raphson with fixed α.

        Args:
            X: Design matrix ``(n, p)``.
            Y_matrix: Permuted count responses ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``alpha`` (required), ``max_iter``, ``tol``,
                ``min_damping``.

        Returns:
            Slope coefficients ``(B, p)``.
        """
        alpha: float = kwargs["alpha"]

        def solver(X: Any, y: Any, mi: Any, t: Any, md: Any) -> Any:
            return _make_negbin_solver(X, y, alpha, mi, t, md)

        return self._batch_shared_X(  # type: ignore[return-value]
            solver, X, Y_matrix, fit_intercept, False, **kwargs
        )

    def batch_negbin_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB2 with per-permutation design matrices.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            y: Shared count response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``alpha`` (required), ``max_iter``, ``tol``,
                ``min_damping``.

        Returns:
            Slope coefficients ``(B, p)``.
        """
        alpha: float = kwargs["alpha"]

        def solver(X: Any, y: Any, mi: Any, t: Any, md: Any) -> Any:
            return _make_negbin_solver(X, y, alpha, mi, t, md)

        return self._batch_varying_X(  # type: ignore[return-value]
            solver, X_batch, y, fit_intercept, False, **kwargs
        )

    # ================================================================ #
    # Ordinal — delegates to generic batch helpers
    # ================================================================ #
    #
    # Ordinal packs parameters as [slopes, thresholds].  We pass
    # ``fit_intercept=False`` (thresholds serve as intercepts) and
    # ``coef_slice=slice(0, n_features)`` to extract only slopes.
    # ---------------------------------------------------------------- #

    def batch_ordinal(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch ordinal via vmap'd Newton–Raphson with autodiff.

        Args:
            X: Design matrix ``(n, p)``.
            Y_matrix: Permuted ordinal responses ``(B, n)``.
            fit_intercept: Ignored (thresholds serve as intercepts).
            **kwargs: ``K`` (required), ``max_iter``, ``tol``,
                ``min_damping``.

        Returns:
            Slope coefficients ``(B, p)``.
        """
        K: int = kwargs["K"]
        n_features = X.shape[1]

        def solver(X: Any, y: Any, mi: Any, t: Any, md: Any) -> Any:
            return _make_ordinal_solver(X, y, K, mi, t, md)

        return self._batch_shared_X(  # type: ignore[return-value]
            solver,
            X,
            Y_matrix,
            False,
            False,
            coef_slice=slice(0, n_features),
            **kwargs,
        )

    def batch_ordinal_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch ordinal with per-permutation design matrices.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            y: Shared ordinal response ``(n,)``.
            fit_intercept: Ignored (thresholds serve as intercepts).
            **kwargs: ``K`` (required), ``max_iter``, ``tol``,
                ``min_damping``.

        Returns:
            Slope coefficients ``(B, p)``.
        """
        K: int = kwargs["K"]
        n_features = X_batch.shape[2]

        def solver(X: Any, y: Any, mi: Any, t: Any, md: Any) -> Any:
            return _make_ordinal_solver(X, y, K, mi, t, md)

        return self._batch_varying_X(  # type: ignore[return-value]
            solver,
            X_batch,
            y,
            False,
            False,
            coef_slice=slice(0, n_features),
            **kwargs,
        )

    # ================================================================ #
    # Multinomial (Softmax) — shared X, many Y
    # ================================================================ #
    #
    # Multinomial returns per-predictor Wald χ² rather than raw
    # coefficients (matching ``MultinomialFamily.coefs()`` semantics).
    # After the Newton–Raphson solve, each permutation’s Hessian is
    # evaluated at the MLE to compute the covariance matrix, then
    # the JIT-compiled Wald extractor produces one scalar per slope.
    #
    # The Hessian re-evaluation per permutation is necessary because
    # the NLL Hessian depends on the response y (through the
    # predicted probabilities at the MLE), so each permutation’s
    # null-distribution response produces a different covariance.
    # ---------------------------------------------------------------- #

    def batch_multinomial(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch multinomial via vmap'd Newton–Raphson with autodiff.

        Shared design matrix *X*, multiple nominal response vectors
        *Y* (ter Braak direct-permutation path).  Returns per-predictor
        Wald χ² test statistics rather than raw coefficients.

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            Y_matrix: Permuted nominal responses ``(B, n)``,
                integer-coded 0 … K-1.
            fit_intercept: Prepend intercept column.
            **kwargs: ``K`` (number of categories, required),
                ``max_iter`` (default 100), ``tol`` (default
                :data:`_DEFAULT_TOL`), ``min_damping`` (default
                :data:`_MIN_DAMPING`).

        Returns:
            Wald χ² statistics ``(B, p)`` — one scalar per slope
            predictor per permutation (intercept excluded).
        """
        K: int = kwargs["K"]
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        X_aug = _augment_intercept_2d(X, fit_intercept)

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float64)
        p_aug = X_aug.shape[1]
        n_params = (K - 1) * p_aug

        _nll = _make_multinomial_nll(K)
        _hess_fn = jit(jax.hessian(_nll))
        _wald_fn = _make_multinomial_wald_chi2(K, p_aug, fit_intercept)

        def _solve_one(
            y_vec: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _make_multinomial_solver(X_j, y_vec, K, max_iter, tol, min_damping)

        all_params, _all_nll, all_converged = jit(vmap(_solve_one))(Y_j)
        _check_convergence(np.asarray(all_converged), max_iter)

        # Compute Wald χ² per predictor for each permutation.
        # Each permutation needs its own Hessian (evaluated at its MLE
        # with its own y vector) to compute the covariance matrix.
        # vmap parallelises the Hessian inversion + Wald extraction
        # across all B permutations in a single XLA kernel.
        def _wald_with_y(params_flat: jax.Array, y_vec: jax.Array) -> jax.Array:
            H = _hess_fn(params_flat, X_j, y_vec)
            cov = jnp.linalg.inv(
                H + _MIN_DAMPING * jnp.eye(n_params, dtype=jnp.float64)
            )
            return _wald_fn(params_flat, cov)  # type: ignore[no-any-return]

        all_wald = jit(vmap(_wald_with_y))(all_params, Y_j)
        return np.asarray(all_wald)  # JAX → NumPy

    # ================================================================ #
    # Multinomial — many X, shared y (Kennedy path)
    # ================================================================ #
    #
    # Kennedy strategy: each permutation has its own X.  The Hessian
    # must be evaluated at each permutation’s (X, MLE) pair because
    # the Hessian depends on X through the predicted probabilities.
    # ---------------------------------------------------------------- #

    def batch_multinomial_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch multinomial with per-permutation design matrices.

        Kennedy individual multinomial path — each permutation replaces
        one column of *X* with permuted exposure residuals.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared nominal response ``(n,)``, integer-coded
                0 … K-1.
            fit_intercept: Prepend intercept column.
            **kwargs: ``K`` (number of categories, required),
                ``max_iter`` (default 100), ``tol`` (default
                :data:`_DEFAULT_TOL`), ``min_damping`` (default
                :data:`_MIN_DAMPING`).

        Returns:
            Wald χ² statistics ``(B, p)`` — one scalar per slope
            predictor per permutation (intercept excluded).
        """
        K: int = kwargs["K"]
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        X_aug = _augment_intercept_3d(X_batch, fit_intercept)

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)
        p_aug = X_aug.shape[2]
        n_params = (K - 1) * p_aug

        _nll = _make_multinomial_nll(K)
        _hess_fn = jit(jax.hessian(_nll))
        _wald_fn = _make_multinomial_wald_chi2(K, p_aug, fit_intercept)

        def _solve_one(
            X_single: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _make_multinomial_solver(
                X_single, y_j, K, max_iter, tol, min_damping
            )

        all_params, _all_nll, all_converged = jit(vmap(_solve_one))(X_j)
        _check_convergence(np.asarray(all_converged), max_iter)

        # Compute Wald χ² per predictor — each permutation has its own X,
        # so the Hessian depends on both the MLE and the design matrix.
        def _wald_with_X(params_flat: jax.Array, X_single: jax.Array) -> jax.Array:
            H = _hess_fn(params_flat, X_single, y_j)
            cov = jnp.linalg.inv(
                H + _MIN_DAMPING * jnp.eye(n_params, dtype=jnp.float64)
            )
            return _wald_fn(params_flat, cov)  # type: ignore[no-any-return]

        all_wald = jit(vmap(_wald_with_X))(all_params, X_j)
        return np.asarray(all_wald)  # JAX → NumPy

    # ================================================================ #
    # batch_fit_and_score — shared X, varying Y
    # ================================================================ #
    #
    # Returns ``(coefs, scores)`` where ``scores = 2·NLL`` (deviance).
    # The 2× factor makes the score directly comparable to the
    # classical deviance statistic.  For improvement computations
    # (Δ = S_reduced − S_full), the constant terms in the NLL that
    # depend only on the data (not the parameters) cancel.
    #
    # Score conventions match the NumPy backend:
    #   OLS        → RSS (residual sum of squares)
    #   Logistic   → 2·NLL  (deviance)
    #   Poisson    → 2·NLL  (deviance)
    #   NB2        → 2·NLL  (deviance)
    #   Ordinal    → 2·NLL
    #   Multinomial→ 2·NLL  (with Wald χ² as "coefs")
    # ---------------------------------------------------------------- #

    def batch_ols_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch OLS returning ``(coefs, RSS)``."""
        X_aug = _augment_intercept_2d(X, fit_intercept)

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float64)

        @jit
        def _batch_solve_with_rss(
            X_mat: jax.Array,
            Y_mat: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            pinv = jnp.linalg.pinv(X_mat)
            coefs_all = (pinv @ Y_mat.T).T  # (B, p_aug)

            def _rss_one(coefs: jax.Array, y_vec: jax.Array) -> jax.Array:
                resid = y_vec - X_mat @ coefs
                return jnp.sum(resid**2)

            rss = vmap(_rss_one)(coefs_all, Y_mat)
            return coefs_all, rss

        all_coefs, all_rss = _batch_solve_with_rss(X_j, Y_j)
        all_coefs = np.asarray(all_coefs)
        coefs = _strip_intercept(all_coefs, fit_intercept)
        return coefs, np.asarray(all_rss)

    def batch_logistic_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch logistic returning ``(coefs, 2·NLL)``."""
        return self._batch_shared_X(  # type: ignore[return-value]
            _make_newton_solver, X, Y_matrix, fit_intercept, True, **kwargs
        )

    def batch_poisson_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch Poisson returning ``(coefs, 2·NLL)``."""
        return self._batch_shared_X(  # type: ignore[return-value]
            _make_poisson_solver, X, Y_matrix, fit_intercept, True, **kwargs
        )

    def batch_negbin_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch NB2 returning ``(coefs, 2·NLL)``."""
        alpha: float = kwargs["alpha"]

        def solver(X: Any, y: Any, mi: Any, t: Any, md: Any) -> Any:
            return _make_negbin_solver(X, y, alpha, mi, t, md)

        return self._batch_shared_X(  # type: ignore[return-value]
            solver, X, Y_matrix, fit_intercept, True, **kwargs
        )

    def batch_ordinal_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch ordinal returning ``(coefs, 2·NLL)``."""
        K: int = kwargs["K"]
        n_features = X.shape[1]

        def solver(X: Any, y: Any, mi: Any, t: Any, md: Any) -> Any:
            return _make_ordinal_solver(X, y, K, mi, t, md)

        return self._batch_shared_X(  # type: ignore[return-value]
            solver,
            X,
            Y_matrix,
            False,
            True,
            coef_slice=slice(0, n_features),
            **kwargs,
        )

    def batch_multinomial_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch multinomial returning ``(wald_chi2, 2·NLL)``."""
        K: int = kwargs["K"]
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        X_aug = _augment_intercept_2d(X, fit_intercept)

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float64)
        p_aug = X_aug.shape[1]
        n_params = (K - 1) * p_aug

        _nll = _make_multinomial_nll(K)
        _hess_fn = jit(jax.hessian(_nll))
        _wald_fn = _make_multinomial_wald_chi2(K, p_aug, fit_intercept)

        def _solve_one(
            y_vec: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _make_multinomial_solver(X_j, y_vec, K, max_iter, tol, min_damping)

        all_params, all_nll, all_converged = jit(vmap(_solve_one))(Y_j)
        _check_convergence(np.asarray(all_converged), max_iter)

        def _wald_with_y(
            params_flat: jax.Array,
            y_vec: jax.Array,
        ) -> jax.Array:
            H = _hess_fn(params_flat, X_j, y_vec)
            cov = jnp.linalg.inv(
                H + _MIN_DAMPING * jnp.eye(n_params, dtype=jnp.float64)
            )
            return _wald_fn(params_flat, cov)  # type: ignore[no-any-return]

        all_wald = jit(vmap(_wald_with_y))(all_params, Y_j)
        return np.asarray(all_wald), 2.0 * np.asarray(all_nll)

    # ================================================================ #
    # batch_fit_and_score_varying_X — varying X, shared Y
    # ================================================================ #
    #
    # Mirror of the ``fit_and_score`` group above, but for Kennedy
    # strategies where X changes per permutation while y is fixed.
    # Score conventions are identical.
    # ---------------------------------------------------------------- #

    def batch_ols_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch OLS (varying X) returning ``(coefs, RSS)``."""
        X_aug = _augment_intercept_3d(X_batch, fit_intercept)

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)

        @jit
        def _batch_lstsq_with_rss(
            X_all: jax.Array,
            y_vec: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            def _solve_one(
                X_single: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                coefs, _, _, _ = jnp.linalg.lstsq(X_single, y_vec, rcond=None)
                resid = y_vec - X_single @ coefs
                return coefs, jnp.sum(resid**2)

            return vmap(_solve_one)(X_all)

        all_coefs, all_rss = _batch_lstsq_with_rss(X_j, y_j)
        all_coefs = np.asarray(all_coefs)
        coefs = _strip_intercept(all_coefs, fit_intercept)
        return coefs, np.asarray(all_rss)

    def batch_logistic_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch logistic (varying X) returning ``(coefs, 2·NLL)``."""
        return self._batch_varying_X(  # type: ignore[return-value]
            _make_newton_solver, X_batch, y, fit_intercept, True, **kwargs
        )

    def batch_poisson_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch Poisson (varying X) returning ``(coefs, 2·NLL)``."""
        return self._batch_varying_X(  # type: ignore[return-value]
            _make_poisson_solver, X_batch, y, fit_intercept, True, **kwargs
        )

    def batch_negbin_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch NB2 (varying X) returning ``(coefs, 2·NLL)``."""
        alpha: float = kwargs["alpha"]

        def solver(X: Any, y: Any, mi: Any, t: Any, md: Any) -> Any:
            return _make_negbin_solver(X, y, alpha, mi, t, md)

        return self._batch_varying_X(  # type: ignore[return-value]
            solver, X_batch, y, fit_intercept, True, **kwargs
        )

    def batch_ordinal_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch ordinal (varying X) returning ``(coefs, 2·NLL)``."""
        K: int = kwargs["K"]
        n_features = X_batch.shape[2]

        def solver(X: Any, y: Any, mi: Any, t: Any, md: Any) -> Any:
            return _make_ordinal_solver(X, y, K, mi, t, md)

        return self._batch_varying_X(  # type: ignore[return-value]
            solver,
            X_batch,
            y,
            False,
            True,
            coef_slice=slice(0, n_features),
            **kwargs,
        )

    def batch_multinomial_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch multinomial (varying X) returning ``(wald_chi2, 2·NLL)``."""
        K: int = kwargs["K"]
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        X_aug = _augment_intercept_3d(X_batch, fit_intercept)

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)
        p_aug = X_aug.shape[2]
        n_params = (K - 1) * p_aug

        _nll = _make_multinomial_nll(K)
        _hess_fn = jit(jax.hessian(_nll))
        _wald_fn = _make_multinomial_wald_chi2(K, p_aug, fit_intercept)

        def _solve_one(
            X_single: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _make_multinomial_solver(
                X_single, y_j, K, max_iter, tol, min_damping
            )

        all_params, all_nll, all_converged = jit(vmap(_solve_one))(X_j)
        _check_convergence(np.asarray(all_converged), max_iter)

        def _wald_with_X(
            params_flat: jax.Array,
            X_single: jax.Array,
        ) -> jax.Array:
            H = _hess_fn(params_flat, X_single, y_j)
            cov = jnp.linalg.inv(
                H + _MIN_DAMPING * jnp.eye(n_params, dtype=jnp.float64)
            )
            return _wald_fn(params_flat, cov)  # type: ignore[no-any-return]

        all_wald = jit(vmap(_wald_with_X))(all_params, X_j)
        return np.asarray(all_wald), 2.0 * np.asarray(all_nll)

    # ================================================================ #
    # batch_*_paired — both X and Y vary per replicate
    # ================================================================ #
    #
    # These methods support **bootstrap** and **jackknife** loops where
    # each replicate resamples (or leaves-one-out) *rows*, so both the
    # design matrix X and the response y change simultaneously.
    #
    # Shape convention:
    #   X_batch : (B, n, p) — B design matrices, no intercept column
    #   Y_batch : (B, n)    — B response vectors
    #
    # Returns only slope coefficients (no scores) — bootstrap /
    # jackknife resampling is used for **confidence-interval**
    # construction, not hypothesis testing.
    #
    # ``vmap`` maps over **both** X and Y simultaneously:
    # ``vmap(_solve_one)(X_j, Y_j)`` — each replicate gets its own
    # (X, y) pair.
    #
    # For multinomial, coefs are Wald χ² per predictor (matching
    # ``MultinomialFamily.coefs()`` semantics).
    # ---------------------------------------------------------------- #

    def batch_ols_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Batch OLS where both X and Y vary per replicate.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            Y_batch: Response vectors ``(B, n)``.
            fit_intercept: Prepend intercept column.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        X_aug = _augment_intercept_3d(X_batch, fit_intercept)

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        Y_j = jnp.array(Y_batch, dtype=jnp.float64)

        @jit
        def _batch_solve(
            X_all: jax.Array,
            Y_all: jax.Array,
        ) -> jax.Array:
            def _solve_one(
                X_single: jax.Array,
                y_vec: jax.Array,
            ) -> jax.Array:
                coefs, _, _, _ = jnp.linalg.lstsq(X_single, y_vec, rcond=None)
                return coefs

            return vmap(_solve_one)(X_all, Y_all)

        all_coefs = np.asarray(_batch_solve(X_j, Y_j))
        return _strip_intercept(all_coefs, fit_intercept)

    def batch_logistic_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch logistic where both X and Y vary per replicate.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            Y_batch: Binary response vectors ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``max_iter``, ``tol``, ``min_damping``.

        Returns:
            Slope coefficients ``(B, p)``.
        """
        return self._batch_paired(  # type: ignore[return-value]
            _make_newton_solver, X_batch, Y_batch, fit_intercept, False, **kwargs
        )

    def batch_poisson_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson where both X and Y vary per replicate.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            Y_batch: Count response vectors ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``max_iter``, ``tol``, ``min_damping``.

        Returns:
            Slope coefficients ``(B, p)``.
        """
        return self._batch_paired(  # type: ignore[return-value]
            _make_poisson_solver, X_batch, Y_batch, fit_intercept, False, **kwargs
        )

    def batch_negbin_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB2 where both X and Y vary per replicate.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            Y_batch: Count response vectors ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``alpha`` (required), ``max_iter``, ``tol``,
                ``min_damping``.

        Returns:
            Slope coefficients ``(B, p)``.
        """
        alpha: float = kwargs["alpha"]

        def solver(X: Any, y: Any, mi: Any, t: Any, md: Any) -> Any:
            return _make_negbin_solver(X, y, alpha, mi, t, md)

        return self._batch_paired(  # type: ignore[return-value]
            solver, X_batch, Y_batch, fit_intercept, False, **kwargs
        )

    def batch_ordinal_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch ordinal where both X and Y vary per replicate.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            Y_batch: Ordinal response vectors ``(B, n)``.
            fit_intercept: Ignored (thresholds serve as intercepts).
            **kwargs: ``K`` (required), ``max_iter``, ``tol``,
                ``min_damping``.

        Returns:
            Slope coefficients ``(B, p)``.
        """
        K: int = kwargs["K"]
        n_features = X_batch.shape[2]

        def solver(X: Any, y: Any, mi: Any, t: Any, md: Any) -> Any:
            return _make_ordinal_solver(X, y, K, mi, t, md)

        return self._batch_paired(  # type: ignore[return-value]
            solver,
            X_batch,
            Y_batch,
            False,
            False,
            coef_slice=slice(0, n_features),
            **kwargs,
        )

    def batch_multinomial_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch multinomial where both X and Y vary per replicate.

        Returns per-predictor Wald χ² statistics (matching
        ``MultinomialFamily.coefs()`` semantics).

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            Y_batch: Nominal response vectors ``(B, n)``,
                integer-coded 0 … K-1.
            fit_intercept: Prepend intercept column.
            **kwargs: ``K`` (required), ``max_iter``, ``tol``,
                ``min_damping``.

        Returns:
            Wald χ² statistics ``(B, p)`` — one scalar per slope
            predictor per replicate (intercept excluded).
        """
        K: int = kwargs["K"]
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        X_aug = _augment_intercept_3d(X_batch, fit_intercept)

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        Y_j = jnp.array(Y_batch, dtype=jnp.float64)
        p_aug = X_aug.shape[2]
        n_params = (K - 1) * p_aug

        _nll = _make_multinomial_nll(K)
        _hess_fn = jit(jax.hessian(_nll))
        _wald_fn = _make_multinomial_wald_chi2(K, p_aug, fit_intercept)

        def _solve_one(
            X_single: jax.Array,
            y_vec: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _make_multinomial_solver(
                X_single, y_vec, K, max_iter, tol, min_damping
            )

        all_params, _all_nll, all_converged = jit(vmap(_solve_one))(X_j, Y_j)
        _check_convergence(np.asarray(all_converged), max_iter)

        def _wald_paired(
            params_flat: jax.Array,
            X_single: jax.Array,
            y_vec: jax.Array,
        ) -> jax.Array:
            H = _hess_fn(params_flat, X_single, y_vec)
            cov = jnp.linalg.inv(
                H + _MIN_DAMPING * jnp.eye(n_params, dtype=jnp.float64)
            )
            return _wald_fn(params_flat, cov)  # type: ignore[no-any-return]

        all_wald = jit(vmap(_wald_paired))(all_params, X_j, Y_j)
        return np.asarray(all_wald)

    # ================================================================ #
    # Linear Mixed Model — shared X, many Y (ter Braak path)
    # ================================================================ #
    #
    # Uses the pre-computed GLS projection A from REML calibration.
    # A single BLAS-3 matmul (projection_A @ Y.T) produces all B
    # coefficient vectors — structurally identical to batch_ols's
    # pinv(X) @ Y.T.
    # ---------------------------------------------------------------- #

    def batch_mixed_lm(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch LMM via pre-computed GLS projection.

        Shared design matrix *X* (encoded in projection), multiple
        response vectors *Y* (ter Braak / Freedman–Lane path).

        Args:
            X: Design matrix ``(n, p)`` — accepted for protocol
                compatibility but not used directly (the projection
                already encodes X).
            Y_matrix: Permuted responses ``(B, n)``.
            fit_intercept: Whether the projection includes an
                intercept row.
            **kwargs: ``projection_A`` (required) — ``(p, n)``
                GLS projection from REML calibration.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        projection_A: np.ndarray = kwargs["projection_A"]
        return _batch_mixed_project(projection_A, Y_matrix, fit_intercept)

    # ================================================================ #
    # Linear Mixed Model — many X, shared y (Kennedy path)
    # ================================================================ #
    #
    # Kennedy strategy: each permutation has its own design matrix
    # (column j replaced with permuted exposure residuals).  The GLS
    # projection A must be rebuilt for each X_b because A depends on
    # X, but the variance components (C₂₂ = Z'Z + Γ⁻¹) are fixed
    # from REML calibration.
    #
    # Pre-computes C₂₂⁻¹Z' once (invariant across permutations),
    # then vmaps the per-X projection rebuild.
    # ---------------------------------------------------------------- #

    def batch_mixed_lm_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch LMM with per-permutation design matrices via vmap.

        Kennedy individual linear mixed path — each permutation
        replaces one column of *X* with permuted exposure residuals.
        Variance components are fixed from REML calibration; only
        the GLS projection changes per permutation.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared continuous response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``Z`` (required) — ``(n, q)`` RE design
                matrix.  ``C22`` (required) — ``(q, q)``
                Henderson C₂₂ = Z'Z + Γ⁻¹ from calibration.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        Z: np.ndarray = kwargs["Z"]
        C22: np.ndarray = kwargs["C22"]

        X_aug = _augment_intercept_3d(X_batch, fit_intercept)

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)
        Z_j = jnp.array(Z, dtype=jnp.float64)
        C22_j = jnp.array(C22, dtype=jnp.float64)

        # Pre-compute invariant: C₂₂⁻¹ Z'  (q, n)
        C22_inv_Zt = jnp.linalg.solve(C22_j, Z_j.T)

        def _solve_one(X_single: jax.Array) -> jax.Array:
            XtZ = X_single.T @ Z_j  # (p, q)
            C22_inv_ZtX = C22_inv_Zt @ X_single  # (q, p)
            S = X_single.T @ X_single - XtZ @ C22_inv_ZtX  # (p, p)
            Xt_Vtilde_inv = X_single.T - XtZ @ C22_inv_Zt  # (p, n)
            A = jnp.linalg.solve(S, Xt_Vtilde_inv)  # (p, n)
            return jnp.asarray(A @ y_j)  # (p,)

        all_coefs: np.ndarray = np.asarray(jit(vmap(_solve_one))(X_j))  # (B, p)
        return _strip_intercept(all_coefs, fit_intercept)

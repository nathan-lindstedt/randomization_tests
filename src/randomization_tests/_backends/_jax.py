"""JAX-accelerated backend for batch model fitting.

Wraps the autodiff Newton–Raphson logistic solver and ``jnp.linalg``
OLS path behind the :class:`~._backends.BackendProtocol` interface.

All public methods accept NumPy arrays and return NumPy arrays —
JAX arrays are materialised at the boundary via ``np.asarray()`` so
callers never touch JAX types directly.

All arithmetic uses **float64** for numerical reliability on
ill-conditioned Hessians, matching statsmodels and scipy.  On Apple
Silicon this routes through CPU Accelerate BLAS rather than Metal
(which lacks float64 support), with no measurable overhead for the
small matrices typical of GLM regression.

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
    from jax import grad, jit, vmap

    _CAN_IMPORT_JAX = True
except ImportError:
    _CAN_IMPORT_JAX = False


# ------------------------------------------------------------------ #
# JAX helper functions (defined only when JAX is importable)
# ------------------------------------------------------------------ #

if _CAN_IMPORT_JAX:

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
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Newton–Raphson logistic solve with early stopping.

        Uses ``jax.lax.while_loop`` for dynamic early exit — the
        solver stops as soon as convergence is detected (typically
        3–4 iterations).  Float64 arithmetic ensures the gradient
        noise floor (``κ(H) × ε_f64``) is far below any reasonable
        tolerance, making the convergence check reliable even on
        ill-conditioned Hessians.

        All arithmetic is **float64** for reliable convergence on
        ill-conditioned Hessians, matching statsmodels / scipy.

        A small constant ``min_damping * I`` is added to the Hessian
        before each linear solve as a singularity guard.  This does
        *not* modify the objective function — the solver still
        targets the unregularised MLE.

        Convergence is checked via three criteria (OR):

        * **Gradient criterion** — ``|g|_∞ < tol``: classical
          first-order optimality.
        * **Parameter-change criterion** — ``|Δβ|_∞ < tol``: detects
          convergence when the Newton step has effectively vanished.
          Mirrors scipy.optimize's xtol.
        * **Relative NLL change** — ``|Δf| / max(|f|, 1) < tol``:
          detects convergence when the objective has plateaued.
          Mirrors scipy.optimize's ftol.

        OR is safe because logistic regression is strictly convex —
        no saddle points exist where one criterion could be satisfied
        without genuine convergence.

        All criteria are gated on finite coefficients: a beta
        containing Inf / NaN (e.g. from separation-induced overflow)
        is never marked as converged.

        Args:
            X: Augmented design matrix ``(n, p+1)``.
            y: Binary response ``(n,)``.
            max_iter: Maximum Newton iterations.
            tol: Convergence threshold applied to gradient, step,
                and relative function change.  Default
                :data:`_DEFAULT_TOL`.
            min_damping: Singularity guard added to the Hessian
                diagonal.  Default :data:`_MIN_DAMPING`.

        Returns:
            ``(beta, converged)`` where *converged* is a scalar bool.
        """
        n_params = X.shape[1]
        damping_matrix = min_damping * jnp.eye(n_params, dtype=jnp.float64)

        # State: (iteration, beta, nll_prev, converged)
        init_state = (
            jnp.array(0),
            jnp.zeros(n_params, dtype=jnp.float64),
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

            g = _logistic_grad(beta, X, y)
            H = _logistic_hessian(beta, X, y)
            H_damped = H + damping_matrix
            beta_new = beta - jnp.linalg.solve(H_damped, g)

            nll_new = _logistic_nll(beta_new, X, y)

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

        _, beta_final, _nll_final, converged = jax.lax.while_loop(
            cond, body, init_state
        )
        return beta_final, converged

    # ---- Poisson helpers -------------------------------------------

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
        mu = jnp.exp(eta)
        return jnp.sum(mu - y * eta)

    @jit
    def _poisson_grad(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Gradient of Poisson NLL: X'(μ − y) where μ = exp(Xβ)."""
        mu = jnp.exp(X @ beta)
        return X.T @ (mu - y)

    @jit
    def _poisson_hessian(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,  # noqa: ARG001
    ) -> jnp.ndarray:
        """Hessian of Poisson NLL: X' diag(μ) X where μ = exp(Xβ)."""
        mu = jnp.exp(X @ beta)
        return (X.T * mu[None, :]) @ X

    def _make_poisson_solver(
        X: jnp.ndarray,
        y: jnp.ndarray,
        max_iter: int,
        tol: float,
        min_damping: float = _MIN_DAMPING,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Newton–Raphson Poisson solve with early stopping.

        Uses ``jax.lax.while_loop`` for dynamic early exit — the
        solver stops as soon as convergence is detected (typically
        4–6 iterations for well-conditioned Poisson data).

        Convergence criteria mirror the logistic solver: gradient
        norm, step size, and relative NLL change (OR), gated on
        finite coefficients.

        Initialises β via OLS on the log-link working response
        ``log(y + 0.5)`` — the standard IRLS warm start — to avoid
        overflow when the true intercept is far from zero.
        """
        n_params = X.shape[1]
        damping_matrix = min_damping * jnp.eye(n_params, dtype=jnp.float64)

        # Log-link warm start: β₀ = (X'X + λI)⁻¹ X' log(y + 0.5)
        eta_init = jnp.log(y + 0.5)
        XtX = X.T @ X + damping_matrix
        Xty = X.T @ eta_init
        beta_init = jnp.linalg.solve(XtX, Xty)

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

            g = _poisson_grad(beta, X, y)
            H = _poisson_hessian(beta, X, y)
            H_damped = H + damping_matrix
            beta_new = beta - jnp.linalg.solve(H_damped, g)

            nll_new = _poisson_nll(beta_new, X, y)

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

        _, beta_final, _nll_final, converged = jax.lax.while_loop(
            cond, body, init_state
        )
        return beta_final, converged

    # ---- Negative binomial helpers ---------------------------------

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
        """Return a JIT-compiled NB2 gradient function with fixed α.

        Gradient: X'(y − μ·(y + 1/α)/(μ + 1/α))
                = X'((μ − y)/(1 + α·μ))  [simplified]
        Wait, let's derive correctly:
        d/dβ NLL = d/dβ Σ[(y+1/α)·log(1+α·μ) − y·η]
                 = Σ[(y+1/α)·α·μ/(1+α·μ) − y] · (dη/dβ_j)
                 = X' · [(y+1/α)·α·μ/(1+α·μ) − y]
                 = X' · [μ·(y+1/α)·α/(1+α·μ) − y]

        Let w = α·μ/(1+α·μ), then gradient = X' · [(y+1/α)·w − y]
        """

        @jit
        def _negbin_grad(
            beta: jnp.ndarray,
            X: jnp.ndarray,
            y: jnp.ndarray,
        ) -> jnp.ndarray:
            mu = jnp.exp(X @ beta)
            inv_a = 1.0 / alpha
            # derivative of NLL wrt eta: (y+1/α)·α·μ/(1+α·μ) − y
            denom = 1.0 + alpha * mu
            d_eta = (y + inv_a) * alpha * mu / denom - y
            return X.T @ d_eta

        return _negbin_grad

    def _make_negbin_hessian(alpha: float) -> Callable[..., Any]:
        """Return a JIT-compiled NB2 Hessian function with fixed α.

        Hessian: X' diag(W) X  where
        W_i = μ_i · (y_i + 1/α) · α / (1 + α·μ_i)²
            = μ_i · (1/α + y_i) / (1/α + μ_i)²

        For the expected (Fisher) information, replace y with E[y]=μ:
        W_i = μ_i / (1/α + μ_i)
        We use the observed Hessian for faster convergence.
        """

        @jit
        def _negbin_hessian(
            beta: jnp.ndarray,
            X: jnp.ndarray,
            y: jnp.ndarray,
        ) -> jnp.ndarray:
            mu = jnp.exp(X @ beta)
            inv_a = 1.0 / alpha
            denom = 1.0 + alpha * mu
            # Second derivative of NLL wrt eta:
            # d²/dη² = (y+1/α)·α·μ/(1+α·μ)² · (1+α·μ − α·μ)
            #         = (y+1/α)·α·μ/(1+α·μ)²
            W = (y + inv_a) * alpha * mu / (denom * denom)
            return (X.T * W[None, :]) @ X

        return _negbin_hessian

    def _make_negbin_solver(
        X: jnp.ndarray,
        y: jnp.ndarray,
        alpha: float,
        max_iter: int,
        tol: float,
        min_damping: float = _MIN_DAMPING,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Newton–Raphson NB2 solve with early stopping and fixed α.

        Initialises β via OLS on the log-link working response
        ``log(y + 0.5)`` — the standard IRLS warm start.
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

        init_state = (
            jnp.array(0),
            beta_init,
            jnp.array(jnp.inf, dtype=jnp.float64),
            jnp.array(False),
        )

        def cond(
            state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> jax.Array:
            i, _beta, _nll_val, converged = state
            return (i < max_iter) & (~converged)

        def body(
            state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            i, beta, nll_prev, _ = state

            g = _grad(beta, X, y)
            H = _hess(beta, X, y)
            H_damped = H + damping_matrix
            beta_new = beta - jnp.linalg.solve(H_damped, g)

            nll_new = _nll(beta_new, X, y)

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

        _, beta_final, _nll_final, converged = jax.lax.while_loop(
            cond, body, init_state
        )
        return beta_final, converged

    # ---- Ordinal (proportional-odds) helpers -----------------------

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
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Newton–Raphson ordinal solve using autodiff.

        Uses ``jax.grad`` and ``jax.hessian`` for exact derivatives
        of the ordinal NLL.  Parameters are packed as
        [β_1, …, β_p, α_0, …, α_{K-2}].

        Returns (params, converged) where params has length p + K - 1.
        """
        n_features = X.shape[1]
        n_params = n_features + K - 1

        _nll = _make_ordinal_nll(K)
        _grad_fn = jit(grad(_nll))

        # Use Gauss-Newton with autodiff gradient + BFGS-style Hessian approx
        # Actually, for ordinal we use full autodiff Hessian for reliability
        _hess_fn = jit(jax.hessian(_nll))

        damping_matrix = min_damping * jnp.eye(n_params, dtype=jnp.float64)

        # Initialize: β=0, thresholds evenly spaced
        init_thresholds = jnp.linspace(-1.0, 1.0, K - 1)
        init_params = jnp.concatenate(
            [
                jnp.zeros(n_features, dtype=jnp.float64),
                init_thresholds,
            ]
        )

        init_state = (
            jnp.array(0),
            init_params,
            jnp.array(jnp.inf, dtype=jnp.float64),
            jnp.array(False),
        )

        def cond(
            state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> jax.Array:
            i, _params, _nll_val, converged = state
            return (i < max_iter) & (~converged)

        def body(
            state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            i, params, nll_prev, _ = state

            g = _grad_fn(params, X, y)
            H = _hess_fn(params, X, y)
            H_damped = H + damping_matrix
            params_new = params - jnp.linalg.solve(H_damped, g)

            nll_new = _nll(params_new, X, y)

            grad_small = jnp.max(jnp.abs(g)) < tol
            step_small = jnp.max(jnp.abs(params_new - params)) < tol
            nll_rel = jnp.abs(nll_new - nll_prev) / jnp.maximum(
                jnp.maximum(jnp.abs(nll_prev), jnp.abs(nll_new)),
                1.0,
            )
            func_small = nll_rel < tol

            converged = (grad_small | step_small | func_small) & jnp.all(
                jnp.isfinite(params_new)
            )
            return (i + 1, params_new, nll_new, converged)

        _, params_final, _nll_final, converged = jax.lax.while_loop(
            cond, body, init_state
        )
        return params_final, converged


def _check_convergence(converged: np.ndarray, max_iter: int) -> None:
    """Emit a single summary warning if any solves did not converge.

    Non-converged permutations are **retained** in the null
    distribution — discarding them would bias the p-value
    anti-conservatively.  The warning is informational only.
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
# JaxBackend
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class JaxBackend:
    """JAX-accelerated compute backend.

    All arithmetic uses float64 for numerical reliability.  Uses
    ``jax.vmap`` to vectorise Newton–Raphson logistic solves across
    all *B* permutations in a single XLA kernel launch.  OLS uses
    ``jnp.linalg.pinv`` for a JIT-compiled batch multiply.
    """

    @property
    def name(self) -> str:
        return "jax"

    @property
    def is_available(self) -> bool:  # noqa: PLR6301
        return _CAN_IMPORT_JAX

    # ---- OLS -----------------------------------------------------------

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
        if fit_intercept:
            X_aug = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_aug = X

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float64)

        @jit
        def _batch_solve(X_mat: jax.Array, Y_mat: jax.Array) -> jax.Array:
            pinv = jnp.linalg.pinv(X_mat)  # (p+1, n)
            return (pinv @ Y_mat.T).T  # (B, p+1)

        result = np.asarray(_batch_solve(X_j, Y_j))
        return result[:, 1:] if fit_intercept else result

    # ---- Logistic (shared X, many Y) ----------------------------------

    def batch_logistic(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch logistic via vmap'd Newton–Raphson.

        Shared design matrix *X*, multiple binary response vectors
        *Y* (ter Braak logistic path).

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            Y_matrix: Permuted binary responses ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``max_iter`` (default 100), ``tol`` (default
                :data:`_DEFAULT_TOL`), ``min_damping`` (default
                :data:`_MIN_DAMPING`).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        if fit_intercept:
            ones = np.ones((X.shape[0], 1), dtype=X.dtype)
            X_aug = np.hstack([ones, X])
        else:
            X_aug = X

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float64)

        def _solve_one(
            y_vec: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return _make_newton_solver(X_j, y_vec, max_iter, tol, min_damping)

        all_betas, all_converged = jit(vmap(_solve_one))(Y_j)
        _check_convergence(np.asarray(all_converged), max_iter)
        all_coefs = np.asarray(all_betas)
        return all_coefs[:, 1:] if fit_intercept else all_coefs

    # ---- Logistic (many X, shared y) -----------------------------------

    def batch_logistic_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch logistic with per-permutation design matrices.

        Kennedy individual logistic path — each permutation replaces
        one column of *X* with permuted exposure residuals.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared binary response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``max_iter`` (default 100), ``tol`` (default
                :data:`_DEFAULT_TOL`), ``min_damping`` (default
                :data:`_MIN_DAMPING`).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        if fit_intercept:
            B, n, _ = X_batch.shape
            ones = np.ones((B, n, 1), dtype=X_batch.dtype)
            X_aug = np.concatenate([ones, X_batch], axis=2)
        else:
            X_aug = X_batch

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)

        def _solve_one(
            X_single: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return _make_newton_solver(X_single, y_j, max_iter, tol, min_damping)

        all_betas, all_converged = jit(vmap(_solve_one))(X_j)
        _check_convergence(np.asarray(all_converged), max_iter)
        all_coefs = np.asarray(all_betas)
        return all_coefs[:, 1:] if fit_intercept else all_coefs

    # ---- OLS (many X, shared y) ----------------------------------------

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
        if fit_intercept:
            B, n, _ = X_batch.shape
            ones = np.ones((B, n, 1), dtype=X_batch.dtype)
            X_aug = np.concatenate([ones, X_batch], axis=2)
        else:
            X_aug = X_batch

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)

        @jit
        def _batch_lstsq(X_all: jax.Array, y_vec: jax.Array) -> jax.Array:
            def _solve_one(X_single: jax.Array) -> jax.Array:
                coefs, _, _, _ = jnp.linalg.lstsq(X_single, y_vec, rcond=None)
                return coefs

            return vmap(_solve_one)(X_all)

        all_coefs = np.asarray(_batch_lstsq(X_j, y_j))
        return all_coefs[:, 1:] if fit_intercept else all_coefs

    # ---- Poisson (shared X, many Y) -----------------------------------

    def batch_poisson(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson via vmap'd Newton–Raphson.

        Shared design matrix *X*, multiple count response vectors
        *Y* (ter Braak Poisson path).

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            Y_matrix: Permuted count responses ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``max_iter`` (default 100), ``tol`` (default
                :data:`_DEFAULT_TOL`), ``min_damping`` (default
                :data:`_MIN_DAMPING`).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        if fit_intercept:
            ones = np.ones((X.shape[0], 1), dtype=X.dtype)
            X_aug = np.hstack([ones, X])
        else:
            X_aug = X

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float64)

        def _solve_one(
            y_vec: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return _make_poisson_solver(X_j, y_vec, max_iter, tol, min_damping)

        all_betas, all_converged = jit(vmap(_solve_one))(Y_j)
        _check_convergence(np.asarray(all_converged), max_iter)
        all_coefs = np.asarray(all_betas)
        return all_coefs[:, 1:] if fit_intercept else all_coefs

    # ---- Poisson (many X, shared y) -----------------------------------

    def batch_poisson_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson with per-permutation design matrices.

        Kennedy individual Poisson path — each permutation replaces
        one column of *X* with permuted exposure residuals.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared count response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``max_iter`` (default 100), ``tol`` (default
                :data:`_DEFAULT_TOL`), ``min_damping`` (default
                :data:`_MIN_DAMPING`).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        if fit_intercept:
            B, n, _ = X_batch.shape
            ones = np.ones((B, n, 1), dtype=X_batch.dtype)
            X_aug = np.concatenate([ones, X_batch], axis=2)
        else:
            X_aug = X_batch

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)

        def _solve_one(
            X_single: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return _make_poisson_solver(X_single, y_j, max_iter, tol, min_damping)

        all_betas, all_converged = jit(vmap(_solve_one))(X_j)
        _check_convergence(np.asarray(all_converged), max_iter)
        all_coefs = np.asarray(all_betas)
        return all_coefs[:, 1:] if fit_intercept else all_coefs

    # ---- Negative Binomial (shared X, many Y) -------------------------

    def batch_negbin(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB2 via vmap'd Newton–Raphson with fixed α.

        Shared design matrix *X*, multiple count response vectors
        *Y*.  The dispersion parameter α must be supplied via
        kwargs (estimated once on the observed data).

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            Y_matrix: Permuted count responses ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``alpha`` (required), ``max_iter`` (default
                100), ``tol`` (default :data:`_DEFAULT_TOL`),
                ``min_damping`` (default :data:`_MIN_DAMPING`).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        alpha: float = kwargs["alpha"]
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        if fit_intercept:
            ones = np.ones((X.shape[0], 1), dtype=X.dtype)
            X_aug = np.hstack([ones, X])
        else:
            X_aug = X

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float64)

        def _solve_one(
            y_vec: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return _make_negbin_solver(X_j, y_vec, alpha, max_iter, tol, min_damping)

        all_betas, all_converged = jit(vmap(_solve_one))(Y_j)
        _check_convergence(np.asarray(all_converged), max_iter)
        all_coefs = np.asarray(all_betas)
        return all_coefs[:, 1:] if fit_intercept else all_coefs

    # ---- Negative Binomial (many X, shared y) -------------------------

    def batch_negbin_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB2 with per-permutation design matrices and fixed α.

        Kennedy individual NB path.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared count response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``alpha`` (required), ``max_iter`` (default
                100), ``tol`` (default :data:`_DEFAULT_TOL`),
                ``min_damping`` (default :data:`_MIN_DAMPING`).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        alpha: float = kwargs["alpha"]
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        if fit_intercept:
            B, n, _ = X_batch.shape
            ones = np.ones((B, n, 1), dtype=X_batch.dtype)
            X_aug = np.concatenate([ones, X_batch], axis=2)
        else:
            X_aug = X_batch

        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)

        def _solve_one(
            X_single: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return _make_negbin_solver(X_single, y_j, alpha, max_iter, tol, min_damping)

        all_betas, all_converged = jit(vmap(_solve_one))(X_j)
        _check_convergence(np.asarray(all_converged), max_iter)
        all_coefs = np.asarray(all_betas)
        return all_coefs[:, 1:] if fit_intercept else all_coefs

    # ---- Ordinal (shared X, many Y) -----------------------------------

    def batch_ordinal(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch ordinal via vmap'd Newton–Raphson with autodiff.

        Shared design matrix *X*, multiple ordinal response vectors
        *Y* (ter Braak direct-permutation path).

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
                ``fit_intercept`` is ignored (thresholds serve as
                intercepts in ordinal models).
            Y_matrix: Permuted ordinal responses ``(B, n)``,
                integer-coded 0 … K-1.
            fit_intercept: Ignored (protocol compatibility).
            **kwargs: ``K`` (number of categories, required),
                ``max_iter`` (default 100), ``tol`` (default
                :data:`_DEFAULT_TOL`), ``min_damping`` (default
                :data:`_MIN_DAMPING`).

        Returns:
            Slope coefficients ``(B, p)`` (thresholds excluded).
        """
        K: int = kwargs["K"]
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        X_j = jnp.array(X, dtype=jnp.float64)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float64)
        n_features = X.shape[1]

        def _solve_one(
            y_vec: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return _make_ordinal_solver(X_j, y_vec, K, max_iter, tol, min_damping)

        all_params, all_converged = jit(vmap(_solve_one))(Y_j)
        _check_convergence(np.asarray(all_converged), max_iter)
        # Extract only slope coefficients (first p columns)
        all_coefs = np.asarray(all_params)[:, :n_features]
        return all_coefs

    # ---- Ordinal (many X, shared y) -----------------------------------

    def batch_ordinal_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch ordinal with per-permutation design matrices.

        Kennedy individual ordinal path.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared ordinal response ``(n,)``, integer-coded
                0 … K-1.
            fit_intercept: Ignored (protocol compatibility).
            **kwargs: ``K`` (number of categories, required),
                ``max_iter`` (default 100), ``tol`` (default
                :data:`_DEFAULT_TOL`), ``min_damping`` (default
                :data:`_MIN_DAMPING`).

        Returns:
            Slope coefficients ``(B, p)`` (thresholds excluded).
        """
        K: int = kwargs["K"]
        max_iter: int = kwargs.get("max_iter", 100)
        tol: float = kwargs.get("tol", _DEFAULT_TOL)
        min_damping: float = kwargs.get("min_damping", _MIN_DAMPING)

        X_j = jnp.array(X_batch, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)
        n_features = X_batch.shape[2]

        def _solve_one(
            X_single: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return _make_ordinal_solver(X_single, y_j, K, max_iter, tol, min_damping)

        all_params, all_converged = jit(vmap(_solve_one))(X_j)
        _check_convergence(np.asarray(all_converged), max_iter)
        all_coefs = np.asarray(all_params)[:, :n_features]
        return all_coefs

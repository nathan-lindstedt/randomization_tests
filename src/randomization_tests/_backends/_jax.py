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

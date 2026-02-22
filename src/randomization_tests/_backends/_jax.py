"""JAX-accelerated backend for batch model fitting.

Wraps the autodiff Newton–Raphson logistic solver and ``jnp.linalg``
OLS path behind the :class:`~._backends.BackendProtocol` interface.

All public methods accept NumPy arrays and return NumPy arrays —
JAX arrays are materialised at the boundary via ``np.asarray()`` so
callers never touch JAX types directly.

If JAX is not installed, the :class:`JaxBackend` can still be
instantiated (for introspection) but ``is_available`` returns
``False`` and :func:`resolve_backend` will raise ``ImportError``
when this backend is explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import jax

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


# ------------------------------------------------------------------ #
# JaxBackend
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class JaxBackend:
    """JAX-accelerated compute backend.

    Uses ``jax.vmap`` to vectorise Newton–Raphson logistic solves
    across all *B* permutations in a single XLA kernel launch.
    OLS uses ``jnp.linalg.pinv`` for a JIT-compiled batch multiply.
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

        X_j = jnp.array(X_aug, dtype=jnp.float32)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float32)

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
            **kwargs: ``max_iter`` (default 100).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        max_iter: int = kwargs.get("max_iter", 100)

        if fit_intercept:
            ones = np.ones((X.shape[0], 1), dtype=X.dtype)
            X_aug = np.hstack([ones, X])
        else:
            X_aug = X

        X_j = jnp.array(X_aug, dtype=jnp.float32)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float32)

        def _solve_one(y_vec: jax.Array) -> jax.Array:
            beta = jnp.zeros(X_j.shape[1], dtype=jnp.float32)
            for _ in range(max_iter):
                g = _logistic_grad(beta, X_j, y_vec)
                H = _logistic_hessian(beta, X_j, y_vec)
                beta = beta - jnp.linalg.solve(H, g)
            return beta

        all_coefs = np.asarray(jit(vmap(_solve_one))(Y_j))
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
            **kwargs: ``max_iter`` (default 100).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        max_iter: int = kwargs.get("max_iter", 100)

        if fit_intercept:
            B, n, _ = X_batch.shape
            ones = np.ones((B, n, 1), dtype=X_batch.dtype)
            X_aug = np.concatenate([ones, X_batch], axis=2)
        else:
            X_aug = X_batch

        X_j = jnp.array(X_aug, dtype=jnp.float32)
        y_j = jnp.array(y, dtype=jnp.float32)

        def _solve_one(X_single: jax.Array, _y: jax.Array = y_j) -> jax.Array:
            beta = jnp.zeros(X_single.shape[1], dtype=jnp.float32)
            for _ in range(max_iter):
                g = _logistic_grad(beta, X_single, _y)
                H = _logistic_hessian(beta, X_single, _y)
                beta = beta - jnp.linalg.solve(H, g)
            return beta

        all_coefs = np.asarray(jit(vmap(_solve_one))(X_j))
        return all_coefs[:, 1:] if fit_intercept else all_coefs

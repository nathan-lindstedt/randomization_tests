"""Backward-compatibility shim for the v0.2.0 JAX module.

As of v0.3.0, all JAX code lives in ``_backends/_jax.py`` behind the
:class:`~._backends.BackendProtocol` interface.  This shim re-exports
the legacy function names so that ``core.py`` imports continue to work
until Phase 2 replaces them with backend-dispatched calls.

.. deprecated:: 0.3.0
    Import from ``_backends`` instead.
"""

from __future__ import annotations

import numpy as np

from ._backends._jax import _CAN_IMPORT_JAX, _DEFAULT_TOL, JaxBackend
from ._config import get_backend

# ------------------------------------------------------------------ #
# Legacy public API (consumed by core.py until Phase 2 refactor)
# ------------------------------------------------------------------ #


def jax_is_available() -> bool:
    """Return ``True`` if JAX can be imported."""
    return _CAN_IMPORT_JAX


def use_jax() -> bool:
    """Return ``True`` if JAX should be used for the current call."""
    return _CAN_IMPORT_JAX and get_backend() == "jax"


# Singleton backend instance — created lazily on first call.
_jax_backend: JaxBackend | None = None


def _get_jax_backend() -> JaxBackend:
    global _jax_backend
    if _jax_backend is None:
        _jax_backend = JaxBackend()
    return _jax_backend


def fit_logistic_batch_jax(
    X_base: np.ndarray,
    Y_matrix: np.ndarray,
    max_iter: int = 100,
    fit_intercept: bool = True,
    tol: float = _DEFAULT_TOL,
) -> np.ndarray:
    """Shim — delegates to ``JaxBackend.batch_logistic``."""
    return _get_jax_backend().batch_logistic(
        X_base, Y_matrix, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol
    )


def fit_logistic_varying_X_jax(
    X_batch: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    fit_intercept: bool = True,
    tol: float = _DEFAULT_TOL,
) -> np.ndarray:
    """Shim — delegates to ``JaxBackend.batch_logistic_varying_X``."""
    return _get_jax_backend().batch_logistic_varying_X(
        X_batch, y, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol
    )


def fit_ols_varying_X_jax(
    X_batch: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = True,
) -> np.ndarray:
    """Shim — delegates to ``JaxBackend.batch_ols_varying_X``."""
    return _get_jax_backend().batch_ols_varying_X(
        X_batch, y, fit_intercept=fit_intercept
    )

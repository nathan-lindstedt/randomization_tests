"""Backend abstraction layer for batch model fitting.

Each backend implements the :class:`BackendProtocol` interface, which
defines the contract for batch-fitting models across many permuted
response (or design) matrices.  The core engine and family classes
dispatch to the active backend via :func:`resolve_backend` rather than
testing ``if _use_jax()`` at every call site.

Resolution follows the policy set by :mod:`._config`:

1. Programmatic override via :func:`~randomization_tests.set_backend`.
2. ``RANDOMIZATION_TESTS_BACKEND`` environment variable.
3. Auto-detection: ``"jax"`` if importable, else ``"numpy"``.

:func:`resolve_backend` translates the policy string into a concrete
backend instance.  When ``"jax"`` is explicitly requested but JAX is
not installed, an :class:`ImportError` is raised — explicit requests
are never silently degraded.  The ``"auto"`` policy is the only mode
that falls back from JAX to NumPy, and that is expected behaviour
rather than a surprise.

Adding a new backend (e.g. CuPy) requires:

1. A new module ``_backends/_cupy.py`` with a class implementing
   :class:`BackendProtocol`.
2. A branch in :func:`resolve_backend` mapping ``"cupy"`` to the new
   class.
3. Adding ``"cupy"`` to ``_VALID_BACKENDS`` in :mod:`._config`.

No changes to ``families.py`` or ``core.py`` are needed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .._config import get_backend

# ------------------------------------------------------------------ #
# BackendProtocol
# ------------------------------------------------------------------ #


@runtime_checkable
class BackendProtocol(Protocol):
    """Interface that every compute backend must implement.

    Backends provide batch-fitting primitives that the
    :class:`~randomization_tests.families.ModelFamily` classes
    delegate to.  Each method accepts the design matrix (or matrices),
    a batch of response vectors, and fitting options, and returns a
    coefficient matrix of shape ``(B, p)``.

    Attributes:
        name: Short identifier (e.g. ``"numpy"``, ``"jax"``).
    """

    @property
    def name(self) -> str: ...

    @property
    def is_available(self) -> bool:
        """Whether the backend's dependencies are importable."""
        ...

    def batch_ols(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Batch OLS: shared *X*, many *Y* vectors.

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            Y_matrix: Permuted responses ``(B, n)``.
            fit_intercept: Prepend intercept column.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        ...

    def batch_logistic(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch logistic: shared *X*, many binary *Y* vectors.

        Used by the ter Braak logistic path where Y is permuted.

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            Y_matrix: Permuted binary responses ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: Solver options (``max_iter``, ``tol``).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        ...

    def batch_logistic_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch logistic: many *X* matrices, shared *y*.

        Used by the Kennedy individual logistic path where column *j*
        of *X* is replaced with permuted exposure residuals.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared binary response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: Solver options (``max_iter``, ``tol``).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        ...

    def batch_ols_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Batch OLS: many *X* matrices, shared *y*.

        Used by the Kennedy individual linear path where column *j*
        of *X* is replaced with permuted exposure residuals.
        Each permutation has its own design matrix, so the single-
        pseudoinverse trick cannot be used.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared continuous response ``(n,)``.
            fit_intercept: Prepend intercept column.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        ...

    def batch_poisson(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson GLM: shared *X*, many count *Y* vectors.

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            Y_matrix: Permuted count responses ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: Solver options (``max_iter``, ``tol``).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        ...

    def batch_poisson_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson GLM: many *X* matrices, shared count *y*.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared count response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: Solver options.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        ...

    def batch_negbin(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB2 GLM: shared *X*, many count *Y* vectors.

        Requires ``alpha`` in kwargs (dispersion, estimated once
        from observed data).

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            Y_matrix: Permuted count responses ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``alpha`` (required), solver options.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        ...

    def batch_negbin_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB2 GLM: many *X* matrices, shared count *y*.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared count response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``alpha`` (required), solver options.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        ...

    def batch_ordinal(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch ordinal: shared *X*, many ordinal *Y* vectors.

        Requires ``K`` in kwargs (number of categories).
        ``fit_intercept`` is accepted but ignored (thresholds
        serve as intercepts).

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            Y_matrix: Permuted ordinal responses ``(B, n)``.
            fit_intercept: Accepted but ignored.
            **kwargs: ``K`` (required), solver options.

        Returns:
            Slope coefficients ``(B, p)`` (thresholds excluded).
        """
        ...

    def batch_ordinal_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch ordinal: many *X* matrices, shared ordinal *y*.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared ordinal response ``(n,)``.
            fit_intercept: Accepted but ignored.
            **kwargs: ``K`` (required), solver options.

        Returns:
            Slope coefficients ``(B, p)`` (thresholds excluded).
        """
        ...


# ------------------------------------------------------------------ #
# Backend resolution
# ------------------------------------------------------------------ #

# Singleton cache — instantiated once per backend name.
_BACKEND_CACHE: dict[str, BackendProtocol] = {}


def resolve_backend(name: str | None = None) -> BackendProtocol:
    """Return a :class:`BackendProtocol` instance for *name*.

    When *name* is ``None`` (the default), the policy from
    :func:`~randomization_tests._config.get_backend` is used.

    Args:
        name: ``"numpy"``, ``"jax"``, or ``None`` for policy default.

    Returns:
        A backend instance ready for batch fitting.

    Raises:
        ImportError: If ``"jax"`` is explicitly requested but JAX
            is not installed.
        ValueError: If *name* is not a recognised backend.
    """
    if name is None:
        name = get_backend()

    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]

    if name == "numpy":
        from ._numpy import NumpyBackend

        backend: BackendProtocol = NumpyBackend()

    elif name == "jax":
        from ._jax import JaxBackend

        jax_backend = JaxBackend()
        if not jax_backend.is_available:
            msg = (
                "Backend 'jax' was explicitly requested but JAX is "
                "not installed.  Install JAX (`pip install jax`) or "
                "use set_backend('numpy')."
            )
            raise ImportError(msg)
        backend = jax_backend

    else:
        msg = f"Unknown backend {name!r}.  Choose 'numpy' or 'jax'."
        raise ValueError(msg)

    _BACKEND_CACHE[name] = backend
    return backend

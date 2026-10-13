r"""Global and thread-local computational backend configuration management.

Controls global and thread-local computational backend resolution across all
resampling loops, numerical GLM solvers, and mixed-model optimization engines:
- JAX backend (:mod:`._backends._jax`): XLA-compiled vectorised batch fitting,
  automatic differentiation, Henderson REML, and two-loop Laplace GLMM engines.
- NumPy backend (:mod:`._backends._numpy`): CPU fallback using standard NumPy
  BLAS-3 operations, statsmodels IRLS, and scikit-learn estimators.

Resolution Policy & Precedence
------------------------------
Resolution order (first match wins):
1. Context-isolated override via :func:`backend` or programmatic :func:`set_backend`.
2. The ``RANDOMIZATION_TESTS_BACKEND`` environment variable.
3. Auto-detection: ``"jax"`` if JAX is importable, else ``"numpy"``.

Valid backend names are ``"jax"`` and ``"numpy"`` (case-insensitive).

Context Isolation & Thread Safety
---------------------------------
Thread-local and coroutine-safe backend overrides are managed via Python's
:mod:`contextvars` module. Context managers modify state strictly within the
calling thread or task, ensuring that concurrent executions in multi-threaded
or asynchronous environments remain completely isolated.

Usage Examples
--------------
Temporarily override backend within a block::

    with rt.backend("numpy"):
        res = rt.randomization_test_regression(X, y)

Disable JAX globally from the shell::

    export RANDOMIZATION_TESTS_BACKEND=numpy

Disable JAX programmatically::

    import randomization_tests
    randomization_tests.set_backend("numpy")

Re-enable auto-detection::

    randomization_tests.set_backend("auto")

Cross-Module Integration
------------------------
* Inbound: Public API functions (:func:`set_backend`, :func:`get_backend`, :func:`backend`).
* Outbound: Consumed by :func:`~._backends.resolve_backend` during engine initialization.
"""

from __future__ import annotations

import contextvars
import os
from collections.abc import Generator
from contextlib import contextmanager

_VALID_BACKENDS = {"jax", "numpy", "auto"}

# Context-isolated override — thread-safe and coroutine-safe across parallel workers.
_backend_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_backend_override", default=None
)


def _jax_is_available() -> bool:
    """Return ``True`` if JAX can be imported."""
    try:
        # Side-effect import to test availability; value unused.
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


def get_backend() -> str:
    """Return the active backend name (``"jax"`` or ``"numpy"``).

    Resolution order:
        1. ContextVar override set by :func:`set_backend` or :func:`backend` (unless ``"auto"`` or ``None``).
        2. ``RANDOMIZATION_TESTS_BACKEND`` environment variable.
        3. ``"jax"`` if importable, otherwise ``"numpy"``.

    Returns:
        ``"jax"`` or ``"numpy"``.
    """
    # 1. Programmatic override (context-isolated)
    override = _backend_override.get()
    if override is not None and override != "auto":
        return override

    # 2. Environment variable
    env = os.environ.get("RANDOMIZATION_TESTS_BACKEND", "").strip().lower()
    if env in ("jax", "numpy"):
        return env

    # 3. Auto-detect
    return "jax" if _jax_is_available() else "numpy"


def set_backend(name: str | None) -> None:
    """Override the backend selection in the current execution context.

    Args:
        name: One of ``"jax"``, ``"numpy"``, ``"auto"``, or ``None``
            (case-insensitive).  ``"auto"`` or ``None`` restores the default
            resolution order.

    Raises:
        ValueError: If *name* is not a recognised backend.
    """
    if name is None or (isinstance(name, str) and name.strip().lower() == "auto"):
        _backend_override.set(None)
        return

    normalised = name.strip().lower()
    if normalised not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Choose from: {sorted(_VALID_BACKENDS)}"
        )
    _backend_override.set(normalised)


@contextmanager
def backend(name: str) -> Generator[str, None, None]:
    """Context manager for temporarily overriding the backend in the current context/thread.

    Args:
        name: One of ``"jax"``, ``"numpy"``, or ``"auto"`` (case-insensitive).

    Yields:
        The active backend name within the context block.
    """
    normalised = name.strip().lower()
    if normalised not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Choose from: {sorted(_VALID_BACKENDS)}"
        )
    val = None if normalised == "auto" else normalised
    token = _backend_override.set(val)
    try:
        yield get_backend()
    finally:
        _backend_override.reset(token)


__all__ = ["backend", "get_backend", "set_backend"]

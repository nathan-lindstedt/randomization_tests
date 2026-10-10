"""Backend configuration for the randomization_tests package.

Controls whether the JAX-accelerated logistic regression path is used
or whether the package falls back to the sklearn/numpy implementation.

Resolution order (first match wins):
    1. Programmatic override via :func:`set_backend`.
    2. The ``RANDOMIZATION_TESTS_BACKEND`` environment variable.
    3. Auto-detection: ``"jax"`` if JAX is importable, else ``"numpy"``.

Valid backend names are ``"jax"`` and ``"numpy"`` (case-insensitive).

Examples:
    Disable JAX globally from the shell::

        export RANDOMIZATION_TESTS_BACKEND=numpy

    Disable JAX programmatically::

        import randomization_tests
        randomization_tests.set_backend("numpy")

    Re-enable auto-detection::

        randomization_tests.set_backend("auto")
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

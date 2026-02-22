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

import os

_VALID_BACKENDS = {"jax", "numpy", "auto"}

# Sentinel indicating "no programmatic override has been set".
_backend_override: str | None = None


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
        1. Value set by :func:`set_backend` (unless ``"auto"``).
        2. ``RANDOMIZATION_TESTS_BACKEND`` environment variable.
        3. ``"jax"`` if importable, otherwise ``"numpy"``.

    Returns:
        ``"jax"`` or ``"numpy"``.
    """
    # 1. Programmatic override
    if _backend_override is not None and _backend_override != "auto":
        return _backend_override

    # 2. Environment variable
    env = os.environ.get("RANDOMIZATION_TESTS_BACKEND", "").strip().lower()
    if env in ("jax", "numpy"):
        return env

    # 3. Auto-detect
    return "jax" if _jax_is_available() else "numpy"


def set_backend(name: str) -> None:
    """Override the backend selection.

    Args:
        name: One of ``"jax"``, ``"numpy"``, or ``"auto"``
            (case-insensitive).  ``"auto"`` restores the default
            resolution order.

    Raises:
        ValueError: If *name* is not a recognised backend.
    """
    global _backend_override
    normalised = name.strip().lower()
    if normalised not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Choose from: {sorted(_VALID_BACKENDS)}"
        )
    _backend_override = normalised

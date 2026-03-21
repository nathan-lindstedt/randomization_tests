"""Permutation strategy registry and protocol.

Each strategy encapsulates a single permutation testing algorithm
(ter Braak, Kennedy, Freedman–Lane) and exposes a uniform
``execute()`` interface that the :class:`~randomization_tests.engine.PermutationEngine`
calls after resolving family, backend, and permutation indices.

Two result shapes exist:

* **Individual strategies** return ``np.ndarray`` of shape
  ``(B, n_features)`` — one permuted coefficient vector per
  permutation.
* **Joint strategies** return a
  ``(obs_improvement, perm_improvements, metric_type, features_tested)``
  tuple describing the collective fit-improvement test.

Adding a new strategy
~~~~~~~~~~~~~~~~~~~~~
1. Create a module under ``strategies/`` with a class that satisfies
   the :class:`PermutationStrategy` protocol.
2. Register it in the :data:`_STRATEGY_REGISTRY` mapping below.
3. The engine and ``core.randomization_test_regression`` will pick it
   up automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..families import ModelFamily

# ------------------------------------------------------------------ #
# Strategy protocol
# ------------------------------------------------------------------ #


@runtime_checkable
class PermutationStrategy(Protocol):
    """Interface that every permutation strategy must satisfy.

    A strategy receives the data, a resolved ``ModelFamily``, and
    pre-generated permutation indices, then returns either:

    * ``np.ndarray`` of shape ``(B, n_features)`` for individual tests, or
    * ``tuple[float, np.ndarray, str, list[str]]`` for joint tests.

    The ``is_joint`` class attribute distinguishes the two shapes so
    that the caller can route result packaging accordingly.
    """

    is_joint: bool
    """``True`` for joint strategies, ``False`` for individual."""

    def execute(
        self,
        X: pd.DataFrame,
        y_values: np.ndarray,
        family: ModelFamily,
        perm_indices: np.ndarray,
        *,
        confounders: list[str] | None = None,
        model_coefs: np.ndarray | None = None,
        fit_intercept: bool = True,
        n_jobs: int = 1,
        randomization: str = "permute",
    ) -> np.ndarray | tuple[float, np.ndarray, str, list[str]]:
        """Run the permutation algorithm.

        Args:
            X: Feature matrix as a pandas DataFrame.
            y_values: Response vector of shape ``(n,)``.
            family: Resolved ``ModelFamily`` instance.
            perm_indices: Pre-generated permutation indices ``(B, n)``.
                When ``randomization="permute"``, contains integer
                indices in ``[0, n)`` (dtype int64).  When
                ``randomization="sign_flip"``, contains ±1 values
                (dtype int8).
            confounders: Confounder column names (Kennedy / FL only).
            model_coefs: Observed coefficients ``(p,)`` — used by
                Kennedy / FL individual to fill confounder slots.
            fit_intercept: Whether to include an intercept.
            n_jobs: Parallelism level for the batch-fit step.
            randomization: Randomization mode — ``"permute"`` for index-
                based permutation (default) or ``"sign_flip"`` for
                Rademacher sign-flip multiplication.

        Returns:
            Strategy-specific result (see class docstring).
        """
        ...


# ------------------------------------------------------------------ #
# Registry
# ------------------------------------------------------------------ #

# Lazy imports to avoid circular dependencies at module load time.
# Each entry maps a method string → callable that returns a strategy
# instance.  The callables are parameter-less factories.

_STRATEGY_REGISTRY: dict[str, type[PermutationStrategy]] = {}


def _ensure_registry() -> None:
    """Populate the registry on first access."""
    if _STRATEGY_REGISTRY:
        return

    from .freedman_lane import FreedmanLaneIndividualStrategy, FreedmanLaneJointStrategy
    from .kennedy import KennedyIndividualStrategy, KennedyJointStrategy
    from .manly import ManlyJointStrategy, ManlyStrategy
    from .score import ScoreExactStrategy, ScoreIndividualStrategy, ScoreJointStrategy
    from .ter_braak import TerBraakStrategy

    _STRATEGY_REGISTRY.update(
        {
            "ter_braak": TerBraakStrategy,
            "kennedy": KennedyIndividualStrategy,
            "kennedy_joint": KennedyJointStrategy,
            "freedman_lane": FreedmanLaneIndividualStrategy,
            "freedman_lane_joint": FreedmanLaneJointStrategy,
            "manly": ManlyStrategy,
            "manly_joint": ManlyJointStrategy,
            "score": ScoreIndividualStrategy,
            "score_joint": ScoreJointStrategy,
            "score_exact": ScoreExactStrategy,
        }
    )


def resolve_strategy(method: str) -> PermutationStrategy:
    """Return a strategy instance for the given method string.

    Args:
        method: One of ``"ter_braak"``, ``"kennedy"``,
            ``"kennedy_joint"``, ``"freedman_lane"``,
            ``"freedman_lane_joint"``, ``"manly"``,
            ``"manly_joint"``, ``"score"``,
            ``"score_joint"``, ``"score_exact"``.

    Raises:
        ValueError: If *method* is not recognised.
    """
    _ensure_registry()
    cls = _STRATEGY_REGISTRY.get(method)
    if cls is None:
        valid = ", ".join(sorted(_STRATEGY_REGISTRY))
        raise ValueError(f"Invalid method '{method}'. Choose from: {valid}.")
    return cls()


def _apply_randomization(
    residuals: np.ndarray,
    randomization_matrix: np.ndarray,
    randomization: str,
) -> np.ndarray:
    """Apply randomization to a residual vector (or matrix).

    Centralises the permutation-vs-sign-flip dispatch so that
    every strategy and ``score_project()`` implementation shares
    the same two-line branch.

    Args:
        residuals: Residual array — 1-D ``(n,)`` for per-feature
            strategies or 2-D ``(n, q)`` for row-wise Kennedy joint.
        randomization_matrix: Index array ``(B, n)``.  When
            ``randomization="permute"``, contains integer indices in
            ``[0, n)`` (dtype int64).  When ``randomization="sign_flip"``,
            contains ±1 values (dtype int8).
        randomization: ``"permute"`` or ``"sign_flip"``.

    Returns:
        Randomized residuals — ``(B, n)`` when *residuals* is 1-D,
        ``(B, n, q)`` when *residuals* is 2-D.
    """
    if randomization == "sign_flip":
        if residuals.ndim == 1:
            # (B, n) * (1, n) → (B, n)
            return randomization_matrix * residuals[np.newaxis, :]  # type: ignore[no-any-return]
        # 2-D: (B, n, 1) * (1, n, q) → (B, n, q)
        return randomization_matrix[:, :, np.newaxis] * residuals[np.newaxis, :, :]  # type: ignore[no-any-return]
    # Default: integer-index permutation.
    return residuals[randomization_matrix]  # type: ignore[no-any-return]


__all__ = [
    "PermutationStrategy",
    "_apply_randomization",
    "resolve_strategy",
]

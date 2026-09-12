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

from ..families import fit_reduced

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


def _residual_joint_statistic(
    family: ModelFamily,
    Z: np.ndarray,
    X_np: np.ndarray,
    y_values: np.ndarray,
    perm_indices: np.ndarray,
    *,
    fit_intercept: bool,
    randomization: str,
    n_jobs: int,
) -> tuple[float, np.ndarray]:
    """Canonical Freedman–Lane (1983) joint reconstruction, refit, and RSS reduction.

    Shared by ``FreedmanLaneJointStrategy`` and ``ScoreJointStrategy``'s residual-based
    branch so the two "equivalent" methods cannot independently drift the way
    ``ScoreJointStrategy`` previously did (M13: it permuted full-model residuals instead
    of reduced-model residuals, silently understating the null's noise scale).

    Permutes REDUCED-model residuals ``e_Z = y − ŷ_Z`` (never full-model residuals —
    those have already had any real signal in the tested features regressed out, which
    mechanically shrinks their variance under nested least squares and narrows the null).
    Both the observed statistic and the null draws go through ``family.
    residual_permutation_refit()`` — an identity permutation for the observed value, the
    real ``perm_indices`` for the null — so a family that whitens (LMM) evaluates both on
    the same scale; a no-op for families that don't.
    """

    reduced_model, preds_reduced = fit_reduced(family, Z, y_values, fit_intercept)
    if reduced_model is not None:
        reduced_resids = family.residuals(reduced_model, Z, y_values)
    else:
        reduced_resids = y_values - preds_reduced

    n = len(y_values)
    # Identity uses "permute" mode explicitly: under "sign_flip" the index array
    # holds ±1 values, not positions, so arange(n) would not mean "unchanged".
    identity = np.arange(n, dtype=perm_indices.dtype).reshape(1, -1)
    _, base_obs = family.residual_permutation_refit(
        Z,
        preds_reduced,
        reduced_resids,
        identity,
        fit_intercept=fit_intercept,
        randomization="permute",
        n_jobs=n_jobs,
    )
    _, full_obs = family.residual_permutation_refit(
        X_np,
        preds_reduced,
        reduced_resids,
        identity,
        fit_intercept=fit_intercept,
        randomization="permute",
        n_jobs=n_jobs,
    )
    obs_improvement = float(base_obs[0] - full_obs[0])

    _, reduced_scores = family.residual_permutation_refit(
        Z,
        preds_reduced,
        reduced_resids,
        perm_indices,
        fit_intercept=fit_intercept,
        randomization=randomization,
        n_jobs=n_jobs,
    )
    _, full_scores = family.residual_permutation_refit(
        X_np,
        preds_reduced,
        reduced_resids,
        perm_indices,
        fit_intercept=fit_intercept,
        randomization=randomization,
        n_jobs=n_jobs,
    )
    perm_improvements = reduced_scores - full_scores

    return obs_improvement, perm_improvements


__all__ = [
    "PermutationStrategy",
    "_apply_randomization",
    "resolve_strategy",
]

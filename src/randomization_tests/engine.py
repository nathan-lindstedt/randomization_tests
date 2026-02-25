"""Permutation engine — Builder for resolution, calibration, and shared primitives.

The :class:`PermutationEngine` centralises everything that happens
*before* a strategy executes:

1. **Family resolution** — map ``"auto"`` / ``"linear"`` / ``"logistic"``
   to a ``ModelFamily`` instance.
2. **Calibration** — estimate nuisance parameters (e.g. negative
   binomial α) from the observed data.
3. **Y validation** — reject mismatched family / response combos early.
4. **Backend resolution** — determine NumPy vs JAX and apply n_jobs
   overrides.
5. **Observed model fit** — fit once to get β̂ and diagnostics.
6. **Permutation index generation** — single call to
   ``generate_unique_permutations`` shared by all strategies.

The engine also exposes :meth:`permute_indices` as the single shared
primitive that v0.4.0 exchangeability cells will hook into — all
strategies receive their indices from this method.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .families import ModelFamily, resolve_family
from .permutations import (
    generate_between_cell_permutations,
    generate_two_stage_permutations,
    generate_unique_permutations,
    generate_within_cell_permutations,
)


class PermutationEngine:
    """Builder that resolves family, backend, and shared state.

    Construct an engine, then call :meth:`run` with a strategy to
    execute the permutation test.  The engine is immutable after
    construction — it captures a snapshot of the resolved state.

    Attributes:
        family: The resolved ``ModelFamily`` instance.
        backend_name: Active backend identifier (``"numpy"`` or
            ``"jax"``).
        model_coefs: Observed (unpermuted) slope coefficients.
        diagnostics: Model-level diagnostics dict from statsmodels.
        perm_indices: Pre-generated permutation index array of shape
            ``(B, n_samples)``.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y_values: np.ndarray,
        *,
        family: str | ModelFamily = "auto",
        fit_intercept: bool = True,
        n_permutations: int = 5_000,
        random_state: int | None = None,
        n_jobs: int = 1,
        method: str = "ter_braak",
        backend: str | None = None,
        groups: np.ndarray | None = None,
        permutation_strategy: str | None = None,
        permutation_constraints: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        # ---- Family resolution ------------------------------------
        self.family: ModelFamily = resolve_family(family, y_values)

        # Calibrate nuisance parameters if the family supports it.
        if hasattr(self.family, "calibrate"):
            self.family = self.family.calibrate(
                X.to_numpy().astype(float), y_values, fit_intercept
            )

        # Validate Y against the family's constraints.
        # Skip only for "auto" — auto-detection is mechanically
        # guaranteed to pick a compatible family.  Explicit choices
        # (strings or ModelFamily instances) can be wrong and deserve
        # a clear validate_y error rather than an opaque sklearn crash.
        if family != "auto":
            self.family.validate_y(y_values)

        # Reject Freedman-Lane for direct-permutation families.
        if self.family.direct_permutation and method in (
            "freedman_lane",
            "freedman_lane_joint",
        ):
            msg = (
                f"Freedman-Lane method is not supported for "
                f"family='{self.family.name}' because residuals are not "
                f"well-defined for this model type.  The ter Braak method "
                f"uses direct Y permutation (equivalent to Manly 1997), "
                f"and the Kennedy methods permute exposure-model residuals "
                f"(always linear OLS).  Supported methods: 'ter_braak', "
                f"'kennedy', 'kennedy_joint'."
            )
            raise ValueError(msg)

        # ---- Backend resolution -----------------------------------
        from ._backends import resolve_backend

        _backend = resolve_backend(backend)
        self.backend_name: str = _backend.name
        self._n_jobs = n_jobs

        # JAX override — vmap replaces joblib parallelism.
        if n_jobs != 1 and self.backend_name == "jax":
            warnings.warn(
                "n_jobs is ignored when the JAX backend is active because "
                "JAX uses vmap vectorisation for batch fits.  Falling back "
                "to n_jobs=1.",
                UserWarning,
                stacklevel=3,
            )
            self._n_jobs = 1

        # OLS vectorised paths don't benefit from joblib.
        if (
            n_jobs != 1
            and self.backend_name == "numpy"
            and self.family.name == "linear"
            and method in ("ter_braak", "freedman_lane")
        ):
            warnings.warn(
                "n_jobs has no effect for linear ter_braak/freedman_lane "
                "because OLS batch fitting is already a single vectorised "
                "BLAS operation (pinv @ Y.T).  Falling back to n_jobs=1.  "
                "n_jobs provides genuine parallelism for logistic families, "
                "Kennedy individual, and joint methods.",
                UserWarning,
                stacklevel=3,
            )
            self._n_jobs = 1

        # ---- Observed model fit -----------------------------------
        X_np = X.values.astype(float)
        observed_model = self.family.fit(X_np, y_values, fit_intercept)
        self.model_coefs: np.ndarray = self.family.coefs(observed_model)

        # Model diagnostics — wrapped in try/except for degenerate data.
        # Each family's diagnostics() method handles its own warning
        # suppression internally (Step 7a).
        try:
            self.diagnostics: dict[str, Any] = self.family.diagnostics(
                X_np, y_values, fit_intercept
            )
        except Exception:
            self.diagnostics = {
                "n_observations": len(y_values),
                "n_features": X_np.shape[1],
            }

        # ---- Permutation indices ----------------------------------
        #
        # Store groups/strategy/constraints before calling
        # permute_indices() because _permute_hook reads them.
        self.groups: np.ndarray | None = groups
        self.permutation_strategy: str | None = permutation_strategy
        self._permutation_constraints = permutation_constraints

        self.perm_indices: np.ndarray = self.permute_indices(
            n_samples=len(y_values),
            n_permutations=n_permutations,
            random_state=random_state,
        )

    # ---- Shared primitive -----------------------------------------

    def permute_indices(
        self,
        n_samples: int,
        n_permutations: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Generate unique permutation indices.

        This is the single point of permutation generation that all
        strategies consume.  Delegates to :meth:`_permute_hook`, which
        subclasses can override to implement exchangeability-constrained
        permutations (v0.4.1).

        Args:
            n_samples: Number of observations.
            n_permutations: Number of unique permutations to generate.
            random_state: Seed for reproducibility.

        Returns:
            Array of shape ``(B, n_samples)`` with permutation indices.
        """
        return self._permute_hook(n_samples, n_permutations, random_state)

    def _permute_hook(
        self,
        n_samples: int,
        n_permutations: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Extension point for exchangeability-constrained permutations.

        Dispatches permutation generation based on the resolved
        ``permutation_strategy`` and any family-suggested
        exchangeability cells:

        1. If ``groups`` is set: use the requested strategy
           (``"within"``, ``"between"``, or ``"two-stage"``).
        2. If ``groups`` is ``None``: check
           ``self.family.exchangeability_cells(X, y)``.  If
           non-``None``, use within-cell permutations.
        3. Otherwise: global permutation via
           :func:`generate_unique_permutations`.

        If a ``permutation_constraints`` callback is set, it is
        applied as a post-filter.  The engine back-fills gaps by
        generating more permutations and re-filtering until
        *n_permutations* are collected (with a max-iterations safety
        cap).

        Args:
            n_samples: Number of observations.
            n_permutations: Number of unique permutations to generate.
            random_state: Seed for reproducibility.

        Returns:
            Array of shape ``(B, n_samples)`` with permutation indices.
        """
        # ---- Determine cells and generator -----------------------
        cells = self.groups
        strategy = self.permutation_strategy

        # If no explicit groups, consult the family.
        if cells is None:
            family_cells = self.family.exchangeability_cells(
                np.zeros((n_samples, 1)),  # placeholder X
                np.zeros(n_samples),  # placeholder y
            )
            if family_cells is not None:
                cells = family_cells
                strategy = "within"

        # ---- Generate indices ------------------------------------
        indices = self._generate_for_strategy(
            cells, strategy, n_samples, n_permutations, random_state
        )

        # ---- Apply callback post-filter --------------------------
        if self._permutation_constraints is not None:
            indices = self._apply_constraints(
                indices, cells, strategy, n_samples, n_permutations, random_state
            )

        return indices

    def _generate_for_strategy(
        self,
        cells: np.ndarray | None,
        strategy: str | None,
        n_samples: int,
        n_permutations: int,
        random_state: int | None,
    ) -> np.ndarray:
        """Route to the correct permutation generator."""
        if cells is None or strategy is None:
            return generate_unique_permutations(
                n_samples=n_samples,
                n_permutations=n_permutations,
                random_state=random_state,
                exclude_identity=True,
            )

        if strategy == "within":
            return generate_within_cell_permutations(
                n_samples=n_samples,
                n_permutations=n_permutations,
                cells=cells,
                random_state=random_state,
                exclude_identity=True,
            )

        if strategy == "between":
            return generate_between_cell_permutations(
                n_samples=n_samples,
                n_permutations=n_permutations,
                cells=cells,
                random_state=random_state,
                exclude_identity=True,
            )

        if strategy == "two-stage":
            return generate_two_stage_permutations(
                n_samples=n_samples,
                n_permutations=n_permutations,
                cells=cells,
                random_state=random_state,
                exclude_identity=True,
            )

        # Unreachable — strategy is validated upstream.
        return generate_unique_permutations(  # pragma: no cover
            n_samples=n_samples,
            n_permutations=n_permutations,
            random_state=random_state,
            exclude_identity=True,
        )

    def _apply_constraints(
        self,
        indices: np.ndarray,
        cells: np.ndarray | None,
        strategy: str | None,
        n_samples: int,
        n_permutations: int,
        random_state: int | None,
    ) -> np.ndarray:
        """Apply callback post-filter and back-fill gaps.

        The callback removes rows that violate domain constraints.
        We regenerate and re-filter until we have enough rows, with
        a max-iterations safety cap to prevent infinite loops.
        """
        assert self._permutation_constraints is not None  # noqa: S101
        rng_offset = 0
        max_rounds = 50
        round_count = 0

        filtered = self._permutation_constraints(indices)

        while filtered.shape[0] < n_permutations and round_count < max_rounds:
            deficit = n_permutations - filtered.shape[0]
            # Generate extras with an offset seed for variety.
            rng_offset += 1
            extra_seed = (
                (random_state + rng_offset) if random_state is not None else None
            )
            extras = self._generate_for_strategy(
                cells, strategy, n_samples, deficit * 2, extra_seed
            )
            extras_filtered = self._permutation_constraints(extras)
            filtered = np.concatenate([filtered, extras_filtered], axis=0)
            round_count += 1

        return filtered[:n_permutations]


__all__ = ["PermutationEngine"]

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
from typing import Any

import numpy as np
import pandas as pd

from .families import ModelFamily, resolve_family
from .permutations import generate_unique_permutations


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

        _backend = resolve_backend()
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

        Default implementation generates globally-exchangeable
        permutations via :func:`generate_unique_permutations`.
        Subclasses override this to restrict permutations within
        exchangeability cells (v0.4.1) or apply user-supplied
        constraints.

        Args:
            n_samples: Number of observations.
            n_permutations: Number of unique permutations to generate.
            random_state: Seed for reproducibility.

        Returns:
            Array of shape ``(B, n_samples)`` with permutation indices.
        """
        return generate_unique_permutations(
            n_samples=n_samples,
            n_permutations=n_permutations,
            random_state=random_state,
            exclude_identity=True,
        )


__all__ = ["PermutationEngine"]

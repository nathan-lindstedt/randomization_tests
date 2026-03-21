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

from ._context import FitContext
from .exchangeability import ExchangeabilityTree
from .families import ModelFamily, resolve_family
from .permutations import (
    generate_between_cell_permutations,
    generate_nested_permutations,
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
        n_randomizations: int = 5_000,
        random_state: int | None = None,
        n_jobs: int = 1,
        method: str = "ter_braak",
        backend: str | None = None,
        groups: np.ndarray | ExchangeabilityTree | None = None,
        permutation_strategy: str | None = None,
        permutation_constraints: Callable[[np.ndarray], np.ndarray] | None = None,
        random_slopes: list[int] | dict[str, list[int]] | None = None,
        ctx: FitContext | None = None,
        ar_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # ---- Context accumulator ----------------------------------
        self.ctx: FitContext = ctx if ctx is not None else FitContext()

        # Populate input artifacts.
        self.ctx.X = X.values.astype(float)
        self.ctx.y = y_values.copy()
        self.ctx.feature_names = list(X.columns)
        self.ctx.target_name = str(X.columns[0]) if hasattr(X, "columns") else None

        # ---- Family resolution ------------------------------------
        self.family: ModelFamily = resolve_family(family, y_values)

        # Calibrate nuisance parameters (protocol method, no-op default).
        calibrate_kw: dict[str, Any] = {}
        if groups is not None:
            # Mixed-model calibration expects a flat 1-D array or dict,
            # not a tree.  Flatten if necessary.
            calibrate_kw["groups"] = (
                groups.to_flat_cells()
                if isinstance(groups, ExchangeabilityTree)
                else groups
            )
        if random_slopes is not None:
            calibrate_kw["random_slopes"] = random_slopes
        if ar_kwargs:
            calibrate_kw.update(ar_kwargs)
        self.family = self.family.calibrate(
            X.to_numpy().astype(float),
            y_values,
            fit_intercept,
            **calibrate_kw,
        )

        # Populate family properties on context.
        self.ctx.family = self.family
        self.ctx.family_name = self.family.name
        self.ctx.residual_type = self.family.residual_type
        self.ctx.direct_permutation = self.family.direct_permutation
        self.ctx.metric_label = self.family.metric_label

        # Populate LMM-specific context fields from calibrated family.
        if hasattr(self.family, "_groups_arr") and self.family._groups_arr is not None:
            self.ctx.groups_arr = self.family._groups_arr
        if hasattr(self.family, "_exog_re_kw"):
            self.ctx.exog_re_kw = self.family._exog_re_kw
        if hasattr(self.family, "_sm_model"):
            self.ctx.sm_mixed_model = self.family._sm_model
        if hasattr(self.family, "_raw_groups"):
            self.ctx.raw_groups = self.family._raw_groups

        # Validate Y against the family's constraints.
        # Skip only for "auto" — auto-detection is mechanically
        # guaranteed to pick a compatible family.  Explicit choices
        # (strings or ModelFamily instances) can be wrong and deserve
        # a clear validate_y error rather than an opaque sklearn crash.
        if family != "auto":
            self.family.validate_y(y_values)

        # Reject residual-based methods for direct-permutation families.
        # score / score_joint / score_exact are excluded here because Guard 3
        # (the score_project() probe below) independently checks their
        # compatibility: once score_project() is implemented for a family,
        # those methods become valid even for direct-permutation families.
        _RESIDUAL_ONLY_METHODS = frozenset(
            {
                "ter_braak",
                "freedman_lane",
                "freedman_lane_joint",
            }
        )
        if self.family.direct_permutation and method in _RESIDUAL_ONLY_METHODS:
            raise ValueError(
                f"method='{method}' requires well-defined residuals but "
                f"family='{self.family.name}' uses direct Y permutation.  "
                f"Supported methods for this family: 'manly', "
                f"'manly_joint', 'kennedy', 'kennedy_joint', "
                f"'score', 'score_joint'."
            )

        # Reject non-score methods for GLMM families (logistic_mixed,
        # poisson_mixed).  Re-estimating variance components per permutation
        # is computationally prohibitive AND statistically incorrect — the
        # null hypothesis holds the random-effects structure fixed exactly
        # as calibrated.  Only score-based methods (which hold θ̂ fixed) are
        # valid.  score_joint is also blocked: it uses batch_fit_and_score
        # which hits the GLMMBatchStubMixin NotImplementedError.
        #
        # This guard comes before the Manly UserWarning so that GLMM
        # families get a clean ValueError rather than a warning-then-error.
        _GLMM_BLOCKED_METHODS = frozenset(
            {
                "ter_braak",
                "kennedy",
                "kennedy_joint",
                "freedman_lane",
                "freedman_lane_joint",
                "manly",
                "manly_joint",
                "score_joint",
            }
        )
        _is_glmm = self.family.name in ("logistic_mixed", "poisson_mixed")
        if _is_glmm and method in _GLMM_BLOCKED_METHODS:
            raise ValueError(
                f"method={method!r} requires per-permutation model refitting "
                f"which is not supported for GLMM family {self.family.name!r}. "
                "Re-estimating variance components per permutation is "
                "computationally prohibitive and statistically incorrect — "
                "the null hypothesis holds the random-effects structure fixed. "
                "Use method='score' or method='score_exact' instead."
            )

        # Warn when Manly is used on a family that supports residuals.
        if method in ("manly", "manly_joint") and not self.family.direct_permutation:
            warnings.warn(
                f"method='{method}' uses direct Y permutation (Manly "
                f"1997), which tests marginal rather than partial "
                f"association.  For family='{self.family.name}' (which "
                f"supports residuals), residual-based methods like "
                f"'ter_braak', 'freedman_lane', or 'score' are more "
                f"powerful.",
                UserWarning,
                stacklevel=3,
            )

        # Warn when score_exact is used on a non-mixed family.
        # score_exact holds variance components fixed and runs full IRLS
        # per permutation via JAX vmap — designed for GLMM; wasteful for
        # standard GLMs where method='score' is exact and much faster.
        if method == "score_exact" and not self.family.name.endswith("_mixed"):
            warnings.warn(
                f"method='score_exact' is designed for GLMM families but "
                f"family={self.family.name!r} is not a mixed-effects model. "
                "score_exact refits the full model per permutation (expensive). "
                "Consider method='score' for non-mixed families.",
                UserWarning,
                stacklevel=3,
            )

        # Reject score methods for families without score_project().
        #
        # Duck-typing probe: we call score_project() with synthetic
        # inputs.  If the family raises NotImplementedError, the
        # method is genuinely unsupported.  Any *other* exception
        # (e.g. "not calibrated") is benign — it means the method
        # exists but can't run yet, which is fine at construction
        # time (calibration happens later in core.py).
        if method in ("score", "score_joint", "score_exact"):
            try:
                self.family.score_project(
                    np.zeros((2, 1)),
                    0,
                    np.zeros(2),
                    np.arange(2, dtype=np.intp).reshape(1, -1),
                )
            except NotImplementedError:
                raise ValueError(
                    f"method='{method}' requires family='{self.family.name}' "
                    f"to implement score_project().  Use method='ter_braak' or "
                    f"method='freedman_lane' instead."
                ) from None
            except Exception:
                pass  # probe raised for non-fatal reasons — OK

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
            and method in ("ter_braak", "freedman_lane", "score", "score_joint")
        ):
            warnings.warn(
                "n_jobs has no effect for linear ter_braak/freedman_lane/"
                "score because OLS batch fitting is already a single "
                "vectorised BLAS operation (pinv @ Y.T).  Falling back "
                "to n_jobs=1.  n_jobs provides genuine parallelism for "
                "logistic families, Kennedy individual, and joint methods.",
                UserWarning,
                stacklevel=3,
            )
            self._n_jobs = 1

        # ---- Observed model fit -----------------------------------
        X_np = X.values.astype(float)
        observed_model = self.family.fit(X_np, y_values, fit_intercept)
        self.model_coefs: np.ndarray = self.family.coefs(observed_model)

        # Populate observed-fit artifacts on context.
        self.ctx.observed_model = observed_model
        self.ctx.coefficients = self.model_coefs.copy()
        self.ctx.predictions = self.family.predict(observed_model, X_np)

        # Residuals — not available for direct-permutation families.
        if not self.family.direct_permutation:
            try:
                self.ctx.residuals = self.family.residuals(
                    observed_model, X_np, y_values
                )
            except (NotImplementedError, Exception):
                self.ctx.residuals = None
        else:
            self.ctx.residuals = None

        # Fit metric — not available for all families.
        if self.ctx.predictions is not None:
            try:
                self.ctx.fit_metric_value = float(
                    self.family.fit_metric(y_values, self.ctx.predictions)
                )
            except (NotImplementedError, Exception):
                self.ctx.fit_metric_value = None

        # Exchangeability cells.
        self.ctx.exchangeability_cells = self.family.exchangeability_cells(
            X_np, y_values
        )

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

        self.ctx.diagnostics = self.diagnostics

        # ---- Permutation indices ----------------------------------
        #
        # Store groups/strategy/constraints before calling
        # permute_indices() because _permute_hook reads them.
        self.groups: np.ndarray | ExchangeabilityTree | None = groups
        self.permutation_strategy: str | None = permutation_strategy
        self._permutation_constraints = permutation_constraints

        # Populate permutation metadata on context.
        self.ctx.backend = self.backend_name
        self.ctx.groups = groups  # type: ignore[assignment]
        self.ctx.permutation_strategy = permutation_strategy

        self.perm_indices: np.ndarray = self.permute_indices(
            n_samples=len(y_values),
            n_randomizations=n_randomizations,
            random_state=random_state,
        )

    # ---- Shared primitive -----------------------------------------

    def permute_indices(
        self,
        n_samples: int,
        n_randomizations: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Generate unique permutation indices.

        This is the single point of permutation generation that all
        strategies consume.  Delegates to :meth:`_permute_hook`, which
        subclasses can override to implement exchangeability-constrained
        permutations (v0.4.1).

        Args:
            n_samples: Number of observations.
            n_randomizations: Number of unique permutations to generate.
            random_state: Seed for reproducibility.

        Returns:
            Array of shape ``(B, n_samples)`` with permutation indices.
        """
        return self._permute_hook(n_samples, n_randomizations, random_state)

    def _permute_hook(
        self,
        n_samples: int,
        n_randomizations: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Extension point for exchangeability-constrained permutations.

        Dispatches permutation generation based on the resolved
        ``permutation_strategy`` and any family-suggested
        exchangeability cells:

        1. If ``groups`` is set: use the requested strategy
           (``"within"``, ``"between"``, or ``"two-stage"``).
        2. If ``groups`` is ``None``: check
           ``self.ctx.exchangeability_cells`` (pre-computed from
           real data during the observed-fit phase).  If
           non-``None``, use within-cell permutations.
        3. Otherwise: global permutation via
           :func:`generate_unique_permutations`.

        If a ``permutation_constraints`` callback is set, it is
        applied as a post-filter.  The engine back-fills gaps by
        generating more permutations and re-filtering until
        *n_randomizations* are collected (with a max-iterations safety
        cap).

        Args:
            n_samples: Number of observations.
            n_randomizations: Number of unique permutations to generate.
            random_state: Seed for reproducibility.

        Returns:
            Array of shape ``(B, n_samples)`` with permutation indices.
        """
        # ---- Determine cells and generator -----------------------
        cells = self.groups
        strategy = self.permutation_strategy

        # Tree-based dispatch: ExchangeabilityTree encodes per-level
        # strategies, so we go straight to the nested generator.
        if isinstance(cells, ExchangeabilityTree):
            indices = generate_nested_permutations(
                n_samples,
                n_randomizations,
                cells,
                random_state,
                exclude_identity=True,
            )
        else:
            # If no explicit groups, use the exchangeability cells that
            # were already computed from real data during the observed-
            # fit phase (stored on the context).  This avoids the
            # previous approach of calling exchangeability_cells() with
            # placeholder zeros — which only worked by accident for
            # families that ignore X and y in that method.
            if cells is None:
                family_cells = self.ctx.exchangeability_cells
                if family_cells is not None:
                    cells = family_cells
                    if not isinstance(cells, ExchangeabilityTree):
                        strategy = "within"

            # ---- Generate indices --------------------------------
            indices = self._generate_for_strategy(
                cells, strategy, n_samples, n_randomizations, random_state
            )

        # ---- Apply callback post-filter --------------------------
        if self._permutation_constraints is not None:
            indices = self._apply_constraints(
                indices, cells, strategy, n_samples, n_randomizations, random_state
            )

        return indices

    def _generate_for_strategy(
        self,
        cells: np.ndarray | ExchangeabilityTree | None,
        strategy: str | None,
        n_samples: int,
        n_randomizations: int,
        random_state: int | None,
    ) -> np.ndarray:
        """Route to the correct permutation generator.

        Three cell-level strategies (Anderson & Robinson, 2001):

        * **within** — shuffle only within cells, preserving cell
          membership.  Appropriate when exchangeability holds
          within groups but not between them.
        * **between** — permute entire cells as units, keeping
          within-cell ordering fixed.  Tests group-level effects.
        * **two-stage** — compose within-cell and between-cell
          permutations for maximal null coverage.

        Each generator uses birthday-paradox collision bounds to
        decide whether post-hoc deduplication is needed (see
        ``generate_unique_permutations()`` in ``permutations.py``
        for the bound derivation).
        """
        # Tree-based dispatch (e.g. from family-suggested cells).
        if isinstance(cells, ExchangeabilityTree):
            return generate_nested_permutations(
                n_samples,
                n_randomizations,
                cells,
                random_state,
                exclude_identity=True,
            )

        if cells is None or strategy is None:
            return generate_unique_permutations(
                n_samples=n_samples,
                n_randomizations=n_randomizations,
                random_state=random_state,
                exclude_identity=True,
            )

        if strategy == "within":
            return generate_within_cell_permutations(
                n_samples=n_samples,
                n_randomizations=n_randomizations,
                cells=cells,
                random_state=random_state,
                exclude_identity=True,
            )

        if strategy == "between":
            return generate_between_cell_permutations(
                n_samples=n_samples,
                n_randomizations=n_randomizations,
                cells=cells,
                random_state=random_state,
                exclude_identity=True,
            )

        if strategy == "two-stage":
            return generate_two_stage_permutations(
                n_samples=n_samples,
                n_randomizations=n_randomizations,
                cells=cells,
                random_state=random_state,
                exclude_identity=True,
            )

        # Unreachable — strategy is validated upstream.
        return generate_unique_permutations(  # pragma: no cover
            n_samples=n_samples,
            n_randomizations=n_randomizations,
            random_state=random_state,
            exclude_identity=True,
        )

    def _apply_constraints(
        self,
        indices: np.ndarray,
        cells: np.ndarray | ExchangeabilityTree | None,
        strategy: str | None,
        n_samples: int,
        n_randomizations: int,
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

        while filtered.shape[0] < n_randomizations and round_count < max_rounds:
            deficit = n_randomizations - filtered.shape[0]
            # Over-generate by 2× to amortise the callback cost:
            # if the filter keeps ≥ 50 % of candidates, one round
            # suffices; otherwise we iterate with offset seeds.
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

        return filtered[:n_randomizations]


__all__ = ["PermutationEngine"]

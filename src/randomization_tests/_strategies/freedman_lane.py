"""Freedman–Lane (1983) — reduced-model residual permutation.

Canonical algorithm (Freedman & Lane 1983; Anderson & Legendre 1999;
Winkler et al. 2014, Table 2).  For each H₀(j): β_j = 0, the nuisance
block is everything except the tested regressor — X_{−j} — and:

1. Fit the **reduced model** Y ~ X_{−j} to get fitted values ŷ₋ⱼ and
   residuals e₋ⱼ = Y − ŷ₋ⱼ.
2. **Permute** the reduced-model residuals: π(e₋ⱼ).
3. **Reconstruct** Y* = ŷ₋ⱼ + π(e₋ⱼ) — the nuisance signal is
   preserved, any real contribution of X_j is destroyed.
4. **Refit** the full model on (X, Y*) and extract β*_j.

The null draws β*_j are centred at zero — the property the package's
``|β*| ≥ |β̂|`` p-value count and shift-inversion CIs require.

Guarantee: reduced-model residuals are only approximately
exchangeable under H₀ (correlated through the reduced-model hat
matrix), so the test is **asymptotically exact**, not finite-sample
exact (Anderson & Robinson 2001).  Anderson & Legendre (1999) found
the Freedman–Lane scheme to have among the best small-sample Type I
and power properties of the residual-permutation methods.

Two strategies:

* **FreedmanLaneIndividualStrategy** — per-coefficient test.  For
  each tested feature j, permutes the X_{−j} reduced-model residuals
  and refits the full model on Y*.  Confounder columns are never
  tested; their slots carry the observed coefficients.

* **FreedmanLaneJointStrategy** — group-level improvement test.  The
  nuisance block is the confounder set Z; permutes the Y ~ Z
  reduced-model residuals, and both reduced and full models are refit
  per permutation with the improvement in fit as the test statistic.

References:
    Freedman, D. & Lane, D. (1983). A nonstochastic interpretation of
    reported significance levels. *J. Business & Economic Statistics*,
    1(4), 292–298.

    Anderson, M. J. & Legendre, P. (1999). An empirical comparison of
    permutation methods for tests of partial regression coefficients
    in a linear model. *J. Statistical Computation and Simulation*,
    62(3), 271–303.

    Anderson, M. J. & Robinson, J. (2001). Permutation tests for
    linear models. *Australian & New Zealand J. Statistics*, 43(1),
    75–88.

    Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., &
    Nichols, T. E. (2014). Permutation inference for the general
    linear model. *NeuroImage*, 92, 381–397.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..families import fit_reduced
from . import _apply_randomization

if TYPE_CHECKING:
    from ..families import ModelFamily


# ------------------------------------------------------------------ #
# Freedman–Lane individual
# ------------------------------------------------------------------ #


class FreedmanLaneIndividualStrategy:
    """Freedman–Lane (1983) per-coefficient reduced-model residual permutation."""

    is_joint: bool = False

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
    ) -> np.ndarray:
        """Run the Freedman–Lane individual permutation algorithm.

        Args:
            X: Feature matrix as a pandas DataFrame.
            y_values: Response vector of shape ``(n,)``.
            family: Resolved ``ModelFamily`` instance.
            perm_indices: Pre-generated permutation indices ``(B, n)``.
            confounders: Confounder column names — never tested; their
                columns carry the observed coefficients.
            model_coefs: Observed coefficients ``(p,)`` — used to
                fill confounder slots.
            fit_intercept: Whether to include an intercept.
            n_jobs: Parallelism level for the batch-fit step.
            randomization: ``"permute"`` (default) or ``"sign_flip"``.

        Returns:
            Array of shape ``(B, n_features)`` with permuted
            coefficients (zero-centred null draws).  Confounder
            columns are filled with the observed coefficient value.
        """
        if confounders is None:
            confounders = []
        if model_coefs is None:
            raise ValueError("model_coefs is required for Freedman-Lane individual.")

        X_np = X.values.astype(float)  # (n, p) full design matrix
        n_perm, n = perm_indices.shape  # B permutations, n observations
        n_features = X_np.shape[1]  # p = number of predictors

        result = np.zeros((n_perm, n_features))  # (B, p) permuted coefficients

        # Deterministic RNG seeded from the first row of perm_indices.
        # Ensures stochastic reconstruction steps (e.g. Bernoulli
        # sampling for logistic) are reproducible for both permutation
        # (0..n-1) and sign-flip (±1) matrices; shift makes all values
        # positive so they are valid as seed entries.
        rng = np.random.default_rng(
            (perm_indices[0].astype(np.int64) + perm_indices.shape[1]).astype(np.uint64)
        )

        # Loop over tested features j.  Each iteration tests
        # H₀(j): β_j = 0 with nuisance block X_{−j} (all other
        # columns, confounders included — canonical Freedman–Lane).
        for j in range(n_features):
            # Confounders keep their observed coefficient — not tested.
            if X.columns[j] in confounders:
                result[:, j] = model_coefs[j]
                continue

            # Step 1: Fit the reduced model Y ~ X_{−j}.
            # np.delete removes column j, yielding (n, p−1) design.
            X_red = np.delete(X_np, j, axis=1)  # (n, p−1)
            # fit_reduced returns (model_or_None, predicted_values).
            # model is None when X_red has zero columns (single-
            # feature design).
            reduced_model, preds_red = fit_reduced(
                family, X_red, y_values, fit_intercept
            )  # preds_red = ŷ₋ⱼ, shape (n,)

            # Reduced-model residuals: e₋ⱼ = Y − ŷ₋ⱼ.
            # For GLM families, family.residuals() computes the
            # appropriate residual type.
            if reduced_model is not None:
                resids_red = family.residuals(reduced_model, X_red, y_values)
            else:
                # Zero-column edge case: intercept-only reduced model;
                # raw residuals e = Y − ŷ.
                resids_red = y_values - preds_red  # (n,)

            # Step 2: Resample the reduced-model residual vector.
            permuted_resids = _apply_randomization(
                resids_red, perm_indices, randomization
            )  # (B, n)

            # Step 3: Reconstruct Y* = ŷ₋ⱼ + π(e₋ⱼ).
            # Nuisance signal preserved; X_j's contribution destroyed.
            Y_perm = family.reconstruct_y(
                preds_red[np.newaxis, :],  # (1, n) → broadcast to (B, n)
                permuted_resids,  # (B, n)
                rng,
            )  # (B, n) synthetic response vectors

            # Step 4: Batch-refit the full model; extract column j.
            all_coefs = np.array(
                family.batch_fit(X_np, Y_perm, fit_intercept, n_jobs=n_jobs)
            )  # (B, p)
            result[:, j] = all_coefs[:, j]

        return result


# ------------------------------------------------------------------ #
# Freedman–Lane joint
# ------------------------------------------------------------------ #


class FreedmanLaneJointStrategy:
    """Freedman–Lane (1983) joint collective-improvement test.

    Tests whether all non-confounder features *collectively* improve
    model fit beyond confounders alone.  The test statistic is:

        Δ = M(Y, ŷ_Z) − M(Y, ŷ_full)

    where M(Y, ŷ) is a prediction-based fit metric (e.g. RSS for
    linear, deviance for GLMs) computed via ``family.fit_metric()``.

    **Why fit_metric instead of score?**  The joint strategy re-fits
    both the reduced and full models on *each* permuted Y*, so there
    is no single pre-fitted model object to pass to ``score()``.
    Instead, we compute predictions from each refit and evaluate
    fit quality via ``fit_metric(y*, ŷ*)`` which needs only the
    response and predicted values.  This is equivalent for
    prediction-based families (linear, logistic, Poisson, NB).

    **Why re-fit both models per permutation?**  Under H₀, the
    permuted Y* has a different realisation for each permutation.
    The reduced model’s fit to Y* therefore changes across
    permutations — the reduced metric is NOT constant (unlike
    Kennedy joint, where the reduced model is fixed and only the
    full model changes).  Both must be refit to get the correct
    improvement Δ*_b.
    """

    is_joint: bool = True

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
    ) -> tuple[float, np.ndarray, str, list[str]]:
        """Run the Freedman–Lane joint permutation algorithm.

        Returns:
            A ``(obs_improvement, perm_improvements, metric_type,
            features_tested)`` tuple.
        """
        if confounders is None:
            confounders = []

        features_to_test = [c for c in X.columns if c not in confounders]
        metric_type = family.metric_label  # e.g. "RSS", "deviance"

        X_np = X.values.astype(float)  # (n, p) full design matrix
        n_perm, n = perm_indices.shape  # B permutations, n observations

        # Deterministic RNG seeded from the first row of perm_indices.
        # Shift by row length so sign-flip values (±1) become positive;
        # permutation values (0..n-1) remain distinct and non-negative.
        rng = np.random.default_rng(
            (perm_indices[0].astype(np.int64) + perm_indices.shape[1]).astype(np.uint64)
        )

        # Z = confounder design matrix (n, q_z).
        if confounders:
            conf_idx = [X.columns.get_loc(c) for c in confounders]  # column positions
            Z = X_np[:, conf_idx]  # (n, q_z)
        else:
            Z = np.zeros((n, 0))  # (n, 0) — no confounders

        # --- Observed reduced model (confounders only) ---
        # ŷ_Z = predictions from Y ~ Z.  fit_metric compares Y to
        # ŷ_Z to get the baseline metric with only confounders.
        reduced_model, preds_reduced = fit_reduced(family, Z, y_values, fit_intercept)
        # preds_reduced = ŷ_Z, shape (n,)

        # Baseline metric: M(Y, ŷ_Z) = how well confounders alone
        # explain Y.  Higher is worse ("lower is better" convention).
        base_metric = family.fit_metric(y_values, preds_reduced)

        # --- Observed full model (all features) ---
        # Fit Y ~ X_full and compute M(Y, ŷ_full).
        full_model = family.fit(X_np, y_values, fit_intercept)
        preds_full = family.predict(full_model, X_np)  # ŷ_full, (n,)
        # Δ_obs = M(Y, ŷ_Z) − M(Y, ŷ_full).  Positive means the
        # tested features improve fit.
        obs_improvement = base_metric - family.fit_metric(y_values, preds_full)

        # --- Reduced-model residuals (canonical Freedman–Lane) ---
        # e_Z = Y − ŷ_Z: residuals of the CONFOUNDER-ONLY reduced
        # model.  Permuting these preserves the nuisance (confounder)
        # signal exactly while destroying any contribution of the
        # tested features (Freedman & Lane 1983; Winkler et al. 2014).
        if reduced_model is not None:
            reduced_resids = family.residuals(reduced_model, Z, y_values)
        else:
            # Zero-column edge case (no confounders): intercept-only
            # reduced model; raw residuals.
            reduced_resids = y_values - preds_reduced  # (n,)

        # --- Permutation loop (vectorised via batch backend) ---
        # Build Y*_batch in one vectorised call, then use
        # batch_fit_and_score() to refit BOTH reduced and full
        # models across all permutations simultaneously.
        #
        # For each permutation b:
        #   1. Permute reduced-model residuals: e*_b = e_Z[perm_b]
        #   2. Reconstruct: Y*_b = preds_reduced + e*_b
        #   3. batch-fit reduced (Z, Y*_batch) -> reduced_scores
        #   4. batch-fit full  (X, Y*_batch) -> full_scores
        #   5. perm_improvements = reduced_scores - full_scores
        #
        # The score values (2*NLL / deviance / RSS) differ from
        # fit_metric by a y-dependent constant that cancels in
        # the improvement delta = S_reduced - S_full.

        # Vectorised reconstruction: (B, n)
        perm_resids_batch = _apply_randomization(
            reduced_resids, perm_indices, randomization
        )  # (B, n)
        preds_reduced_tiled = np.broadcast_to(preds_reduced[np.newaxis, :], (n_perm, n))
        Y_star_batch = family.reconstruct_y(
            preds_reduced_tiled, perm_resids_batch, rng
        )  # (B, n)

        # Batch-fit reduced model (Z, Y*) for all permutations.
        _, reduced_scores = family.batch_fit_and_score(
            Z, Y_star_batch, fit_intercept, n_jobs=n_jobs
        )

        # Batch-fit full model (X_np, Y*) for all permutations.
        _, full_scores = family.batch_fit_and_score(
            X_np, Y_star_batch, fit_intercept, n_jobs=n_jobs
        )

        perm_improvements = reduced_scores - full_scores

        return (
            obs_improvement,
            perm_improvements,
            metric_type,
            features_to_test,
        )

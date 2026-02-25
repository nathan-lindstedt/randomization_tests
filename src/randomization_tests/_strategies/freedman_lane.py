"""Freedman–Lane (1983) — full-model residual permutation.

The Freedman–Lane procedure is arguably the most widely recommended
residual-permutation strategy in the literature (Anderson & Robinson
2001; Winkler et al. 2014).  It differs from ter Braak (1992) in a
crucial way:

* **ter Braak** fits a *reduced* model (without X_j) and permutes
  those residuals.  This tests H₀: β_j = 0 but can be sensitive to
  model misspecification of the reduced model.

* **Freedman–Lane** fits the *full* model (including X_j) and permutes
  those residuals.  The permuted residuals are then added back to the
  *reduced* (confounders-only) fitted values.  This has better finite-
  sample properties because the full-model residuals are closer to
  the true error distribution when the full model is correctly specified.

Algorithm (individual test for H₀: β_j = 0):

1. Fit the **full model** Y ~ X and compute residuals e = Y − ŷ.
2. Fit the **reduced model** Y ~ Z (confounders only) to get ŷ_Z.
3. For each permutation b:
   a. π_b(e) = permuted full-model residuals.
   b. Y*_b = ŷ_Z + π_b(e)  [reduced fitted values + permuted noise].
   c. Refit Y*_b ~ X_full and extract β*_j(b).
4. Compare observed β_j to the distribution {β*_j(1), …, β*_j(B)}.

Two strategies:

* **FreedmanLaneIndividualStrategy** — per-coefficient test.  Permutes
  full-model residuals, adds them to reduced-model (confounders-only)
  fitted values, and refits the full model on Y*.

* **FreedmanLaneJointStrategy** — group-level improvement test.  Same
  Y* construction, but both reduced and full models are refit per
  permutation and the improvement in fit is the test statistic.

References:
    Freedman, D. & Lane, D. (1983). A nonstochastic interpretation of
    reported significance levels. *J. Business & Economic Statistics*,
    1(4), 292–298.

    Anderson, M. J. & Robinson, J. (2001). Permutation tests for
    linear models. *Australian & New Zealand J. Statistics*, 43(1),
    75–88.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..families import fit_reduced

if TYPE_CHECKING:
    from ..families import ModelFamily


# ------------------------------------------------------------------ #
# Freedman–Lane individual
# ------------------------------------------------------------------ #


class FreedmanLaneIndividualStrategy:
    """Freedman–Lane (1983) per-coefficient full-model residual permutation."""

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
    ) -> np.ndarray:
        """Run the Freedman–Lane individual permutation algorithm.

        Args:
            X: Feature matrix as a pandas DataFrame.
            y_values: Response vector of shape ``(n,)``.
            family: Resolved ``ModelFamily`` instance.
            perm_indices: Pre-generated permutation indices ``(B, n)``.
            confounders: Confounder column names.
            model_coefs: Observed coefficients ``(p,)`` — used to
                fill confounder slots.
            fit_intercept: Whether to include an intercept.
            n_jobs: Parallelism level for the batch-fit step.

        Returns:
            Array of shape ``(B, n_features)`` with permuted
            coefficients.  Confounder columns are filled with the
            observed coefficient value.
        """
        if confounders is None:
            confounders = []
        if model_coefs is None:
            raise ValueError("model_coefs is required for Freedman-Lane individual.")

        X_np = X.values.astype(float)  # (n, p) full design matrix
        n_perm, n = perm_indices.shape  # B permutations, n observations

        # Deterministic RNG seeded from the first permutation index.
        # Ensures stochastic reconstruction steps (e.g. Bernoulli
        # sampling for logistic) are reproducible.
        rng = np.random.default_rng(int(perm_indices[0, 0]))

        # Step 1: Fit the FULL model Y ~ X and get residuals.
        # Unlike ter Braak, we use full-model residuals because they
        # are closer to the true error distribution when the model
        # is correctly specified (Freedman & Lane 1983, §3).
        full_model = family.fit(X_np, y_values, fit_intercept)
        full_resids = family.residuals(full_model, X_np, y_values)  # e = Y − ŷ, (n,)

        # Step 2: Fit the REDUCED model Y ~ Z (confounders only).
        # The reduced predictions ŷ_Z become the "signal" component to
        # which permuted residuals are added.
        if confounders:
            conf_idx = [X.columns.get_loc(c) for c in confounders]  # column positions
            Z = X_np[:, conf_idx]  # (n, q_z) confounder design
        else:
            Z = np.zeros((n, 0))  # (n, 0) — no confounders
        _, preds_reduced = fit_reduced(family, Z, y_values, fit_intercept)
        # preds_reduced = ŷ_Z, shape (n,)

        # Step 3: Permute full-model residuals.
        # Fancy-indexing with (B, n) index array broadcasts the 1-D
        # residual vector into B shuffled copies in one shot.
        permuted_resids = full_resids[perm_indices]  # (B, n)

        # Step 4: Reconstruct Y* = ŷ_Z + π(e).
        # For linear: direct addition.  For logistic: clamp to [0,1]
        # and draw Bernoulli.  The np.newaxis broadcast lets (1, n)
        # predictions combine with (B, n) permuted residuals.
        Y_perm = family.reconstruct_y(
            preds_reduced[np.newaxis, :],  # (1, n) → broadcast to (B, n)
            permuted_resids,  # (B, n)
            rng,
        )  # (B, n) synthetic response vectors

        # Step 5: Batch-refit the full model on all B synthetic
        # responses.  Returns (B, p) coefficient matrix.
        all_coefs = np.array(
            family.batch_fit(X_np, Y_perm, fit_intercept, n_jobs=n_jobs)
        )  # (B, p)

        # Confounder columns keep their observed coefficients —
        # they are not being tested and should not contribute
        # to the permutation distribution.
        if confounders:
            for i, col in enumerate(X.columns):
                if col in confounders:
                    all_coefs[:, i] = model_coefs[i]

        return all_coefs


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

        # Deterministic RNG for stochastic reconstruction.
        rng = np.random.default_rng(int(perm_indices[0, 0]))

        # Z = confounder design matrix (n, q_z).
        if confounders:
            conf_idx = [X.columns.get_loc(c) for c in confounders]  # column positions
            Z = X_np[:, conf_idx]  # (n, q_z)
        else:
            Z = np.zeros((n, 0))  # (n, 0) — no confounders

        # --- Observed reduced model (confounders only) ---
        # ŷ_Z = predictions from Y ~ Z.  fit_metric compares Y to
        # ŷ_Z to get the baseline metric with only confounders.
        _, preds_reduced = fit_reduced(family, Z, y_values, fit_intercept)
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

        # --- Full-model residuals ---
        # e = Y − ŷ_full (or appropriate GLM residuals).  These are
        # the residuals that will be permuted.
        full_resids = family.residuals(full_model, X_np, y_values)  # (n,)

        # --- Permutation loop (vectorised via batch backend) ---
        # Build Y*_batch in one vectorised call, then use
        # batch_fit_and_score() to refit BOTH reduced and full
        # models across all permutations simultaneously.
        #
        # For each permutation b:
        #   1. Permute residuals: e*_b = e[perm_b]
        #   2. Reconstruct: Y*_b = preds_reduced + e*_b
        #   3. batch-fit reduced (Z, Y*_batch) -> reduced_scores
        #   4. batch-fit full  (X, Y*_batch) -> full_scores
        #   5. perm_improvements = reduced_scores - full_scores
        #
        # The score values (2*NLL / deviance / RSS) differ from
        # fit_metric by a y-dependent constant that cancels in
        # the improvement delta = S_reduced - S_full.

        # Vectorised reconstruction: (B, n)
        perm_resids_batch = full_resids[perm_indices]  # (B, n)
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

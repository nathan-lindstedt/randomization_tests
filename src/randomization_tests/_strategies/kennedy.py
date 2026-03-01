"""Kennedy (1995) — exposure-model residual permutation.

The Kennedy method separates the effect of confounders from the effect
of the features of interest by permuting *exposure-model residuals*
rather than the response variable Y.  This idea is rooted in the
Frisch–Waugh–Lovell (FWL) theorem:

    In a multiple regression Y ~ X + Z, the coefficient on X is
    identical whether we regress Y on (X, Z) jointly, or first
    partial out Z from X (regress X on Z, take residuals e_X) and
    then regress Y on e_X.

Kennedy extends this to permutation testing: permuting the exposure
residuals e_X breaks any association between X and Y while preserving
the confounders’ relationship to Y intact.  This is valid under the
null hypothesis that X has no effect on Y after controlling for Z.

Two strategies:

* **KennedyIndividualStrategy** — per-coefficient test.  For each
  non-confounder predictor X_j:
  1. Regress X_j on confounders Z to get exposure residuals e_{X_j}.
  2. Permute e_{X_j} to get π(e_{X_j}).
  3. Reconstruct X*_j = X̂_j + π(e_{X_j}) where X̂_j is the
     confounder-predicted part.
  4. Refit Y ~ (X*_1, …, X*_j, …, X_p) and record β*_j.

* **KennedyJointStrategy** — group-level improvement test.  All
  non-confounder exposure residuals are permuted **row-wise** (same
  shuffle for all columns) to preserve inter-predictor correlations.
  The test statistic is the improvement in fit:
  Δ = S(reduced) − S(full), where S is the family’s score.

Reference:
    Kennedy, P. E. (1995). Randomization tests in econometrics.
    *J. Business & Economic Statistics*, 13(1), 85–94.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..families import _augment_intercept, fit_reduced

if TYPE_CHECKING:
    from ..families import ModelFamily


# ------------------------------------------------------------------ #
# Kennedy individual
# ------------------------------------------------------------------ #


class KennedyIndividualStrategy:
    """Kennedy (1995) per-coefficient exposure-residual permutation."""

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
        """Run the Kennedy individual permutation algorithm.

        For each non-confounder feature X_j:

        1. **Exposure model**: regress X_j on Z using OLS to get
           X̂_j (predicted) and e_{X_j} = X_j − X̂_j (residuals).
           The exposure residuals represent the variation in X_j
           that is *not* explained by confounders.

        2. **Permute**: shuffle e_{X_j} using the pre-generated
           permutation indices.  Under H₀ (X_j has no effect on Y
           after controlling for Z), the mapping from e_{X_j} to Y
           is arbitrary — permuting it is valid.

        3. **Reconstruct**: X*_j = X̂_j + π(e_{X_j}).  This produces
           a feature vector with the same confounder structure but
           a disrupted X_j–Y association.

        4. **Refit**: fit Y ~ (X_1, …, X*_j, …, X_p) and extract
           β*_j.  Confounder coefficients are held at observed values.

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
            coefficients.
        """
        if confounders is None:
            confounders = []
        if model_coefs is None:
            raise ValueError("model_coefs is required for Kennedy individual.")

        X_np = X.values.astype(float)  # (n, p) full design matrix
        n_perm, n = perm_indices.shape  # B permutations, n observations
        n_features = X_np.shape[1]  # p = total predictors (tested + confounders)
        result = np.zeros((n_perm, n_features))  # (B, p) permuted coefficients

        # Separate features into those being tested and confounders.
        features_to_test = [c for c in X.columns if c not in confounders]

        # Z = confounder design matrix (n, q).  When q = 0 we use a
        # zero-column placeholder so downstream algebra still works.
        if confounders:
            Z = X[confounders].values  # (n, q)
        else:
            Z = np.zeros((n, 0))  # (n, 0) — no confounders

        # Confounder coefficients are NOT tested — they keep their
        # observed values in every permutation row.  Only the
        # non-confounder features receive permuted coefficients.
        for i, col in enumerate(X.columns):
            if col in confounders:
                result[:, i] = model_coefs[i]

        for feature in features_to_test:
            feat_idx = X.columns.get_loc(feature)
            x_target = X[[feature]].values  # (n, 1)

            # Step 1: Exposure model — regress X_j on confounders Z.
            #
            # This is the FWL partial-out step.  We compute:
            #   X̂_j = Z · (Z⁺ · X_j)   [OLS projection onto Z]
            #   e_{X_j} = X_j − X̂_j     [exposure residuals]
            #
            # When Z is empty, X̂_j = mean(X_j) (intercept-only) or
            # 0 (no intercept).
            if Z.shape[1] > 0:
                if fit_intercept:
                    # Augment Z with a column of ones for the intercept:
                    # Z_aug = [1 | Z], shape (n, q+1).
                    Z_aug = _augment_intercept(Z)
                else:
                    Z_aug = Z  # (n, q)
                # Moore-Penrose pseudoinverse — handles rank-deficient Z.
                pinv_z = np.linalg.pinv(Z_aug)  # (q+1, n) or (q, n)
                # OLS projection: X̂_j = Z_aug · (Z_aug⁺ · X_j).
                # This is the "hat matrix" multiplication P_Z · X_j.
                x_hat = Z_aug @ (pinv_z @ x_target)  # (n, 1)
            else:
                # No confounders: the exposure model is intercept-only
                # (or zero if no intercept).
                if fit_intercept:
                    x_hat = np.full_like(x_target, x_target.mean())  # (n, 1)
                else:
                    x_hat = np.zeros_like(x_target)  # (n, 1)
            # Flatten from (n, 1) → (n,) for permutation indexing.
            x_resids = (x_target - x_hat).ravel()  # e_{X_j}, shape (n,)

            # Step 2: Permute exposure residuals.
            # Fancy-indexing with (B, n) index array broadcasts the
            # 1-D residual vector into B shuffled copies in one shot.
            shuffled = x_resids[perm_indices]  # (B, n)

            # Step 3: Reconstruct X*_j and build batch design matrices.
            # Start from a (B, n, p) copy of the original design.
            # np.broadcast_to is read-only, so .copy() is required
            # before we can write into the j-th column.
            X_perm_all = np.broadcast_to(X_np, (n_perm, n, n_features)).copy()
            # X*_j = X̂_j + π(e_{X_j}) — confounder-predicted part
            # plus permuted residuals.  Other columns stay unchanged.
            X_perm_all[:, :, feat_idx] = x_hat.ravel()[np.newaxis, :] + shuffled

            # Step 4: Batch-refit the outcome model across all B
            # permutations simultaneously.  Returns (B, p) coefficients.
            all_coefs = family.batch_fit_varying_X(
                X_perm_all, y_values, fit_intercept, n_jobs=n_jobs
            )
            # Only the j-th coefficient is the test statistic for
            # feature j; confounder slots were filled earlier.
            result[:, feat_idx] = all_coefs[:, feat_idx]

        return result


# ------------------------------------------------------------------ #
# Kennedy joint
# ------------------------------------------------------------------ #


class KennedyJointStrategy:
    """Kennedy (1995) joint collective-improvement test.

    Tests whether all non-confounder features *collectively* improve
    model fit beyond confounders alone.  The test statistic is:

        Δ = S(reduced) − S(full)

    where S(·) is the family’s score (lower is better).  When the
    tested features genuinely improve fit, S(full) < S(reduced)
    and Δ > 0.  The permutation distribution of Δ under H₀ is
    obtained by permuting the exposure residuals row-wise (same
    shuffle for all non-confounder columns) to break the joint
    association while preserving inter-predictor correlations.
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
        """Run the Kennedy joint permutation algorithm.

        Returns:
            A ``(obs_improvement, perm_improvements, metric_type,
            features_tested)`` tuple.
        """
        if confounders is None:
            confounders = []

        features_to_test = [c for c in X.columns if c not in confounders]
        metric_type = family.metric_label  # e.g. "RSS", "deviance"

        n_perm, n = perm_indices.shape  # B permutations, n observations
        # X_target = non-confounder feature matrix (n, q) where q is
        # the number of features whose joint contribution is tested.
        X_target = X[features_to_test].values

        # Z = confounder matrix (n, q_z); zero-column when no confounders.
        if confounders:
            Z = X[confounders].values  # (n, q_z)
        else:
            Z = np.zeros((n, 0))  # (n, 0)

        # --- Reduced model (confounders only) ---
        # Fit Y ~ Z to get the baseline score.  If Z has zero columns,
        # null_score() provides the intercept-only baseline (see
        # families.py for the mathematical derivation).  If Z has
        # columns, score() evaluates the fitted model.
        reduced_model, _ = fit_reduced(family, Z, y_values, fit_intercept)
        if reduced_model is not None:
            base_metric = family.score(reduced_model, Z, y_values)
        else:
            base_metric = family.null_score(y_values, fit_intercept)

        # --- Full model (all features) ---
        # Fit Y ~ (X_target, Z) to get the full-model score.
        # The observed test statistic is:
        #   Δ_obs = S(reduced) − S(full)
        # Positive Δ means the tested features improve fit.
        full_features = np.hstack([X_target, Z]) if Z.shape[1] > 0 else X_target
        full_model = family.fit(full_features, y_values, fit_intercept)
        obs_improvement = base_metric - family.score(
            full_model, full_features, y_values
        )

        # --- Exposure model residuals ---
        # Regress all non-confounder features on Z simultaneously
        # to get exposure residuals.  These residuals capture the
        # variation in (X_1, …, X_q) not explained by confounders.
        # Permuting rows of this residual matrix preserves the
        # correlation structure among the tested features.
        if Z.shape[1] > 0:
            exp_model = LinearRegression(fit_intercept=fit_intercept).fit(Z, X_target)
            x_hat = exp_model.predict(Z)
        else:
            if fit_intercept:
                x_hat = np.full_like(X_target, X_target.mean(axis=0))
            else:
                x_hat = np.zeros_like(X_target)
        x_resids = X_target - x_hat

        # --- Permutation loop (vectorised via batch backend) ---
        # Build the full (B, n, p_full) batch of permuted designs in
        # one vectorised operation, then delegate to the family's
        # batch_fit_and_score_varying_X() which dispatches to the
        # JAX vmap or NumPy parallel backend.
        #
        # For each permutation b:
        #   X*_b = x_hat + e_X[perm_b]   (reconstructed features)
        #   full_b = [X*_b | Z]          (appending confounders)
        #
        # The batch call returns (coefs, scores) where
        # scores[b] = S(full model fitted on (full_b, y))
        # and perm_improvements = base_metric - scores.

        # Vectorised reconstruction: (B, n, q) = (1,n,q) + (B,n,q)
        x_star_batch = x_hat[np.newaxis, :, :] + x_resids[perm_indices]

        # Append confounders to each permuted design matrix.
        if Z.shape[1] > 0:
            Z_tiled = np.broadcast_to(
                Z[np.newaxis, :, :], (n_perm, n, Z.shape[1])
            ).copy()
            X_batch = np.concatenate([x_star_batch, Z_tiled], axis=2)  # (B, n, q + q_z)
        else:
            X_batch = x_star_batch  # (B, n, q)

        _, perm_scores = family.batch_fit_and_score_varying_X(
            X_batch, y_values, fit_intercept, n_jobs=n_jobs
        )
        perm_improvements = base_metric - perm_scores

        return (
            obs_improvement,
            perm_improvements,
            metric_type,
            features_to_test,
        )

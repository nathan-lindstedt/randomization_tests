"""Kennedy (1995) — exposure-model residual permutation.

Two strategies:

* **KennedyIndividualStrategy** — per-coefficient test.  For each
  non-confounder predictor X_j, regress X_j on confounders Z to get
  exposure residuals eₓⱼ, permute eₓⱼ, reconstruct X*_j, and refit
  the full model.

* **KennedyJointStrategy** — group-level improvement test.  All
  non-confounder exposure residuals are permuted **row-wise** (same
  shuffle for all columns) to preserve inter-predictor correlations.
  The test statistic is the improvement in fit compared to a
  confounders-only model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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

        X_np = X.values.astype(float)
        n_perm, n = perm_indices.shape
        n_features = X_np.shape[1]
        result = np.zeros((n_perm, n_features))

        features_to_test = [c for c in X.columns if c not in confounders]

        if confounders:
            Z = X[confounders].values
        else:
            Z = np.zeros((n, 0))

        # Confounder columns keep observed coefficients.
        for i, col in enumerate(X.columns):
            if col in confounders:
                result[:, i] = model_coefs[i]

        for feature in features_to_test:
            feat_idx = X.columns.get_loc(feature)
            x_target = X[[feature]].values  # (n, 1)

            # Step 1: Exposure model — regress X_j on confounders Z.
            if Z.shape[1] > 0:
                if fit_intercept:
                    Z_aug = np.column_stack([np.ones(n), Z])
                else:
                    Z_aug = Z
                pinv_z = np.linalg.pinv(Z_aug)
                x_hat = Z_aug @ (pinv_z @ x_target)
            else:
                if fit_intercept:
                    x_hat = np.full_like(x_target, x_target.mean())
                else:
                    x_hat = np.zeros_like(x_target)
            x_resids = (x_target - x_hat).ravel()  # (n,)

            # Step 2: Permute exposure residuals.
            shuffled = x_resids[perm_indices]  # (B, n)

            # Step 3: Reconstruct X*_j and build batch design matrices.
            X_perm_all = np.broadcast_to(X_np, (n_perm, n, n_features)).copy()
            X_perm_all[:, :, feat_idx] = x_hat.ravel()[np.newaxis, :] + shuffled

            # Step 4: Batch-refit the outcome model.
            all_coefs = family.batch_fit_varying_X(
                X_perm_all, y_values, fit_intercept, n_jobs=n_jobs
            )
            result[:, feat_idx] = all_coefs[:, feat_idx]

        return result


# ------------------------------------------------------------------ #
# Kennedy joint
# ------------------------------------------------------------------ #


class KennedyJointStrategy:
    """Kennedy (1995) joint collective-improvement test."""

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
        metric_type = family.metric_label

        n_perm, n = perm_indices.shape
        X_target = X[features_to_test].values

        if confounders:
            Z = X[confounders].values
        else:
            Z = np.zeros((n, 0))

        # Duck-typed model-metric dispatch for ordinal/multinomial.
        _uses_model_metric = hasattr(family, "model_fit_metric")

        # --- Reduced model (confounders only) ---
        if Z.shape[1] > 0:
            reduced_model = family.fit(Z, y_values, fit_intercept)
            if _uses_model_metric:
                base_metric = family.model_fit_metric(reduced_model)  # type: ignore[attr-defined]
            else:
                preds_reduced = family.predict(reduced_model, Z)
                base_metric = family.fit_metric(y_values, preds_reduced)
        else:
            if not _uses_model_metric:
                if fit_intercept:
                    preds_reduced = np.full(n, np.mean(y_values), dtype=float)
                else:
                    preds_reduced = np.zeros(n, dtype=float)
                base_metric = family.fit_metric(y_values, preds_reduced)

        # --- Full model (all features) ---
        full_features = np.hstack([X_target, Z]) if Z.shape[1] > 0 else X_target
        full_model = family.fit(full_features, y_values, fit_intercept)

        if _uses_model_metric:
            if Z.shape[1] == 0:
                base_metric = family.null_fit_metric(full_model)  # type: ignore[attr-defined]
            obs_improvement = (
                base_metric - family.model_fit_metric(full_model)  # type: ignore[attr-defined]
            )
        else:
            preds_full = family.predict(full_model, full_features)
            obs_improvement = base_metric - family.fit_metric(y_values, preds_full)

        # --- Exposure model residuals ---
        if Z.shape[1] > 0:
            exp_model = LinearRegression(fit_intercept=fit_intercept).fit(Z, X_target)
            x_hat = exp_model.predict(Z)
        else:
            if fit_intercept:
                x_hat = np.full_like(X_target, X_target.mean(axis=0))
            else:
                x_hat = np.zeros_like(X_target)
        x_resids = X_target - x_hat

        # --- Permutation loop ---
        def _joint_one_perm(idx: np.ndarray) -> float:
            x_star = x_hat + x_resids[idx]
            perm_features = np.hstack([x_star, Z]) if Z.shape[1] > 0 else x_star
            perm_model = family.fit(perm_features, y_values, fit_intercept)
            if _uses_model_metric:
                return float(
                    base_metric - family.model_fit_metric(perm_model)  # type: ignore[attr-defined]
                )
            perm_preds = family.predict(perm_model, perm_features)
            return float(base_metric - family.fit_metric(y_values, perm_preds))

        if n_jobs == 1:
            perm_improvements = np.zeros(n_perm)
            for i in range(n_perm):
                perm_improvements[i] = _joint_one_perm(perm_indices[i])
        else:
            from joblib import Parallel, delayed

            perm_improvements = np.array(
                Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_joint_one_perm)(perm_indices[i]) for i in range(n_perm)
                )
            )

        return (
            obs_improvement,
            perm_improvements,
            metric_type,
            features_to_test,
        )

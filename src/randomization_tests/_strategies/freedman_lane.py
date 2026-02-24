"""Freedman–Lane (1983) — full-model residual permutation.

Two strategies:

* **FreedmanLaneIndividualStrategy** — per-coefficient test.  Permutes
  full-model residuals, adds them to reduced-model (confounders-only)
  fitted values, and refits the full model on Y*.

* **FreedmanLaneJointStrategy** — group-level improvement test.  Same
  Y* construction, but both reduced and full models are refit per
  permutation and the improvement in fit is the test statistic.
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

        X_np = X.values.astype(float)
        n_perm, n = perm_indices.shape

        # Deterministic RNG for stochastic reconstruction.
        rng = np.random.default_rng(int(perm_indices[0, 0]))

        # Step 1: Fit the FULL model and get residuals.
        full_model = family.fit(X_np, y_values, fit_intercept)
        full_resids = family.residuals(full_model, X_np, y_values)

        # Step 2: Fit the REDUCED model (confounders only).
        if confounders:
            conf_idx = [X.columns.get_loc(c) for c in confounders]
            Z = X_np[:, conf_idx]
        else:
            Z = np.zeros((n, 0))
        _, preds_reduced = fit_reduced(family, Z, y_values, fit_intercept)

        # Step 3: Permute full-model residuals.
        permuted_resids = full_resids[perm_indices]  # (B, n)

        # Step 4: Reconstruct Y*.
        Y_perm = family.reconstruct_y(
            preds_reduced[np.newaxis, :],
            permuted_resids,
            rng,
        )  # (B, n)

        # Step 5: Batch-refit the full model on all B permuted Y.
        all_coefs = np.array(
            family.batch_fit(X_np, Y_perm, fit_intercept, n_jobs=n_jobs)
        )  # (B, n_features)

        # Fill confounder columns with observed coefficient.
        if confounders:
            for i, col in enumerate(X.columns):
                if col in confounders:
                    all_coefs[:, i] = model_coefs[i]

        return all_coefs


# ------------------------------------------------------------------ #
# Freedman–Lane joint
# ------------------------------------------------------------------ #


class FreedmanLaneJointStrategy:
    """Freedman–Lane (1983) joint collective-improvement test."""

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
        metric_type = family.metric_label

        X_np = X.values.astype(float)
        n_perm, n = perm_indices.shape

        # Deterministic RNG for stochastic reconstruction.
        rng = np.random.default_rng(int(perm_indices[0, 0]))

        if confounders:
            conf_idx = [X.columns.get_loc(c) for c in confounders]
            Z = X_np[:, conf_idx]
        else:
            Z = np.zeros((n, 0))

        # --- Observed reduced model ---
        _, preds_reduced = fit_reduced(family, Z, y_values, fit_intercept)

        base_metric = family.fit_metric(y_values, preds_reduced)

        # --- Observed full model ---
        full_model = family.fit(X_np, y_values, fit_intercept)
        preds_full = family.predict(full_model, X_np)
        obs_improvement = base_metric - family.fit_metric(y_values, preds_full)

        # --- Full-model residuals ---
        full_resids = family.residuals(full_model, X_np, y_values)

        # --- Permutation loop ---
        def _fl_joint_one_perm(idx: np.ndarray) -> float:
            perm_resids = full_resids[idx]
            y_star = family.reconstruct_y(
                preds_reduced[np.newaxis, :],
                perm_resids[np.newaxis, :],
                rng,
            ).ravel()

            # Refit reduced model on Y*
            _, red_preds_star = fit_reduced(family, Z, y_star, fit_intercept)
            metric_red = family.fit_metric(y_star, red_preds_star)

            # Refit full model on Y*
            full_model_star = family.fit(X_np, y_star, fit_intercept)
            full_preds_star = family.predict(full_model_star, X_np)
            metric_full = family.fit_metric(y_star, full_preds_star)

            return float(metric_red - metric_full)

        if n_jobs == 1:
            perm_improvements = np.zeros(n_perm)
            for i in range(n_perm):
                perm_improvements[i] = _fl_joint_one_perm(perm_indices[i])
        else:
            from joblib import Parallel, delayed

            perm_improvements = np.array(
                Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_fl_joint_one_perm)(perm_indices[i]) for i in range(n_perm)
                )
            )

        return (
            obs_improvement,
            perm_improvements,
            metric_type,
            features_to_test,
        )

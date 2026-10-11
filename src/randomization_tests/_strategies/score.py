"""Score projection strategy — batch permutation via a single matmul.

Instead of refitting the full model B times (ter Braak, Freedman–Lane),
the score strategy computes the permutation test statistic for each
feature via a single matrix–vector product:

    β̂*_j = A_j @ (ŷ_red + e_π)  =  (A_j @ ŷ_red) + (A_j @ e_π)

where A_j is the j-th row of the projection matrix (pseudoinverse for
OLS, GLS projection for LMM) and e_π are the permuted reduced-model
residuals.

The first term ``A_j @ ŷ_red`` is constant across all permutations —
only the second term varies.  ``family.score_project()`` computes the
varying part ``A_j @ e_π`` for all B permutations at once via a single
matmul ``(B, n) @ (n,) → (B,)``.  The strategy adds the constant
offset so the returned values are on the coefficient scale, compatible
with the existing p-value computation.

**Equivalence guarantees:**

* **Linear / LMM:** Score p-values = Freedman–Lane p-values
  (bit-for-bit identical: for each tested feature j both permute the
  X_{−j} reduced-model residuals and evaluate A_j @ Y*, the score
  strategy via one matmul, Freedman–Lane via the full batch refit).
* **GLMM:** The one-step corrector upgrades score accuracy
  to second-order (Le Cam estimator).  Handled inside
  ``family.score_project()``, transparent to this strategy.

**Guarantee**: the constant offset
``A_j @ ŷ_red`` vanishes identically for full-column-rank designs
(``pinv(X) @ X = I`` and ŷ_red lies in the span of the non-j columns),
so the score permutation null is centred at ZERO — exactly like the
Freedman–Lane null it reproduces.  Like all residual-based schemes,
Type I control is **asymptotically exact**, not finite-sample exact
(Anderson & Robinson 2001).

Two strategies:

* **ScoreIndividualStrategy** — per-coefficient test (``is_joint=False``).
* **ScoreJointStrategy** — collective improvement test (``is_joint=True``).

References:
    Rao, C. R. (1948). Large sample tests of composite hypotheses.
    Sankhyā, 9(1), 44–56.

    Le Cam, L. (1960). Locally asymptotically normal families of
    distributions. *University of California Publications in
    Statistics*, 3, 37–98.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, final

import numpy as np
import pandas as pd

from ..families import _augment_intercept, fit_reduced
from . import _apply_randomization, _residual_joint_statistic

if TYPE_CHECKING:
    from ..families import ModelFamily


# ------------------------------------------------------------------ #
# Score individual
# ------------------------------------------------------------------ #


@final
class ScoreIndividualStrategy:
    """Score projection — per-coefficient batch permutation test.

    For each feature j, computes the permuted test statistic via
    ``family.score_project()`` — a single matmul per feature.  The
    resulting permuted coefficients are on the same scale as the
    observed coefficients, so the existing p-value calculation works
    unchanged.
    """

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
        **kwargs: Any,
    ) -> np.ndarray:
        """Run the score projection permutation algorithm.

        Args:
            X: Feature matrix as a pandas DataFrame.
            y_values: Response vector of shape ``(n,)``.
            family: Resolved ``ModelFamily`` instance.
            perm_indices: Pre-generated permutation indices ``(B, n)``.
            confounders: Confounder column names (used to identify
                which features are being tested vs controlled).
            model_coefs: Observed coefficients ``(p,)`` — needed for
                the constant offset that puts permuted scores on the
                coefficient scale.
            fit_intercept: Whether to include an intercept.
            n_jobs: Unused (score is fully vectorised).
            randomization: ``"permute"`` (default) or ``"sign_flip"``.

        Returns:
            Array of shape ``(B, n_features)`` with permuted
            coefficients.  Confounder columns are filled with the
            observed coefficient value (same convention as
            Freedman–Lane individual).
        """
        if confounders is None:
            confounders = []
        if model_coefs is None:
            raise ValueError("model_coefs is required for score individual.")

        X_np = X.values.astype(float)  # (n, p)
        n_perm = perm_indices.shape[0]  # B
        n = len(y_values)
        n_features = X_np.shape[1]  # p

        result = np.zeros((n_perm, n_features))  # (B, p)

        for j in range(n_features):
            # Confounders keep their observed coefficient — not tested.
            col_name = X.columns[j]
            if col_name in confounders:
                result[:, j] = model_coefs[j]
                continue

            # Step 1: Reduced design matrix — drop feature j.
            # Same pattern as ter Braak / Freedman–Lane.
            X_red = np.delete(X_np, j, axis=1)  # (n, p−1)

            # Step 2: Fit reduced model, get predictions.
            # Reuses the same fit_reduced() as other strategies.
            _, preds_reduced = fit_reduced(
                family, X_red, y_values, fit_intercept
            )  # ŷ₋ⱼ, shape (n,)

            # Step 3: Reduced-model residuals.
            e = y_values - preds_reduced  # (n,)

            # Step 4: Score projection — batch matmul.
            # family.score_project() returns A_j @ e_π for each
            # permutation — shape (B,).  This is the varying part.
            raw_scores = family.score_project(
                X_np,
                y_values,
                j,
                e,
                perm_indices,
                fit_intercept=fit_intercept,
                randomization=randomization,
            )  # (B,) — raw score (sans constant)

            # Step 5: Add constant offset to put on coefficient scale.
            # observed_raw = A_j @ e (identity permutation)
            # constant = model_coefs[j] - observed_raw
            # result[:, j] = constant + raw_scores
            #
            # Equivalently:
            #   result[:, j] = model_coefs[j] + (raw_scores - observed_raw)
            identity = np.arange(n, dtype=perm_indices.dtype).reshape(1, -1)
            observed_raw = family.score_project(
                X_np,
                y_values,
                j,
                e,
                identity,
                fit_intercept=fit_intercept,
                randomization="permute",  # identity uses index mode
            )[0]  # scalar — A_j @ e (unpermuted)
            result[:, j] = model_coefs[j] + (raw_scores - observed_raw)

        return result


# ------------------------------------------------------------------ #
# Score joint
# ------------------------------------------------------------------ #


@final
class ScoreJointStrategy:
    """Joint (omnibus) permutation test — canonical Freedman–Lane reconstruction.

    Tests whether all non-confounder features collectively improve model fit
    beyond confounders alone.  For direct-permutation families (ordinal,
    multinomial), permutes Y directly — the reason this class exists
    separately from ``FreedmanLaneJointStrategy``, which is blocked for those
    families entirely (no well-defined residuals to permute).  For every
    other family, delegates to the same canonical reduced-residual
    reconstruction Freedman–Lane joint uses (``_residual_joint_statistic``),
    so the two are identical by construction, not by name only — this class
    does not use ``score_project()`` or any one-step/Fisher-information
    machinery; "score" here is a naming convention (every individual
    strategy has a joint counterpart), not the Rao score-test principle.
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
        **kwargs: Any,
    ) -> tuple[float, np.ndarray, str, list[str]]:
        """Run the joint permutation algorithm.

        Args:
            X: Feature matrix as a pandas DataFrame.
            y_values: Response vector of shape ``(n,)``.
            family: Resolved ``ModelFamily`` instance.
            perm_indices: Pre-generated permutation indices ``(B, n)``.
            confounders: Confounder column names.
            model_coefs: Unused.
            fit_intercept: Whether to include an intercept.
            n_jobs: Parallelism level for the batch-fit step.
            randomization: ``"permute"`` (default) or ``"sign_flip"``.

        Returns:
            ``(obs_improvement, perm_improvements, metric_type,
            features_tested)`` tuple.
        """
        if confounders is None:
            confounders = []

        features_to_test = [c for c in X.columns if c not in confounders]
        metric_type = family.metric_label

        X_np = X.values.astype(float)  # (n, p)
        n = perm_indices.shape[1]

        # Confounder design matrix — same as Freedman–Lane joint.
        if confounders:
            conf_idx = [X.columns.get_loc(c) for c in confounders]
            Z = X_np[:, conf_idx]  # (n, q_z)
        else:
            Z = np.zeros((n, 0))  # (n, 0)

        # GLMM branch: closed-form Rao score quadratic form on the whitened
        # tangent space (no iterative model refitting per permutation).
        if family.name in ("logistic_mixed", "poisson_mixed"):
            if not features_to_test:
                return (0.0, np.zeros(perm_indices.shape[0]), metric_type, [])

            assert hasattr(family, "z_tilde") and family.z_tilde is not None

            X_full = _augment_intercept(X_np, fit_intercept)
            conf_aug_idx = (
                [0] + [X.columns.get_loc(c) + 1 for c in confounders]
                if fit_intercept
                else [X.columns.get_loc(c) for c in confounders]
            )
            test_aug_idx = (
                [X.columns.get_loc(c) + 1 for c in features_to_test]
                if fit_intercept
                else [X.columns.get_loc(c) for c in features_to_test]
            )

            Z_aug = X_full[:, conf_aug_idx]
            X_test = X_full[:, test_aug_idx]

            Z_w = family.whiten(Z_aug)
            z_tilde_w = family.whiten(family.z_tilde)
            X_test_w = family.whiten(X_test)

            if Z_aug.shape[1] > 0:
                Z_pinv = np.linalg.pinv(Z_w)
                beta_red = Z_pinv @ z_tilde_w
                r_w = z_tilde_w - Z_w @ beta_red
                P_Z = Z_w @ Z_pinv
                X_test_w_star = X_test_w - P_Z @ X_test_w
            else:
                r_w = z_tilde_w
                X_test_w_star = X_test_w

            Q, _ = np.linalg.qr(X_test_w_star, mode="reduced")
            obs_improvement = float(np.sum((Q.T @ r_w) ** 2))

            E_pi = _apply_randomization(r_w, perm_indices, randomization)
            A = E_pi @ Q
            perm_improvements = np.sum(A**2, axis=1)

            return (
                obs_improvement,
                perm_improvements,
                "Score Reduction",
                features_to_test,
            )

        if not family.direct_permutation:
            obs_improvement, perm_improvements = _residual_joint_statistic(
                family,
                Z,
                X_np,
                y_values,
                perm_indices,
                fit_intercept=fit_intercept,
                randomization=randomization,
                n_jobs=n_jobs,
            )
            return (obs_improvement, perm_improvements, metric_type, features_to_test)

        # Direct-permutation families (ordinal, multinomial): no residual
        # pipeline, so permute Y directly instead — same validity argument
        # as Manly (full exchangeability under H0), the only branch where
        # this class's behaviour differs from Freedman–Lane joint.
        reduced_model, _ = fit_reduced(family, Z, y_values, fit_intercept)
        if reduced_model is not None:
            base_metric = family.score(reduced_model, Z, y_values)
        else:
            base_metric = family.null_score(y_values, fit_intercept)
        full_model = family.fit(X_np, y_values, fit_intercept)
        obs_improvement = base_metric - family.score(full_model, X_np, y_values)

        Y_perm = y_values[perm_indices]  # (B, n)
        _, reduced_scores = family.batch_fit_and_score(
            Z, Y_perm, fit_intercept, n_jobs=n_jobs
        )
        _, full_scores = family.batch_fit_and_score(
            X_np, Y_perm, fit_intercept, n_jobs=n_jobs
        )
        perm_improvements = reduced_scores - full_scores

        return (
            obs_improvement,
            perm_improvements,
            metric_type,
            features_to_test,
        )

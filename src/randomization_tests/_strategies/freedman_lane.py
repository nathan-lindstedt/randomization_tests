r"""Freedman–Lane reduced-model residual permutation and Double Machine Learning.

Implements canonical reduced-model residual permutation testing (Freedman & Lane 1983)
and Double/Debiased Machine Learning (DML; Chernozhukov et al. 2018) for partial
regression inference in linear and generalized linear models.

Primary Components
------------------
* :class:`FreedmanLaneIndividualStrategy`: Per-feature partial regression test.
* :class:`FreedmanLaneJointStrategy`: Group-level omnibus fit improvement test.

Mathematical Foundations & Algorithm Flow
-----------------------------------------
For each tested feature $j$, the nuisance block comprises all remaining predictors
$X_{-j}$. The 4-step canonical workflow operates as follows:

1. **Reduced-Model Fit**: Fit $Y \sim X_{-j}$ to obtain predictions $\hat y_{-j}$
   and response-scale residuals $e_{-j} = Y - \hat y_{-j}$.
2. **Residual Randomization**: Randomize $e_{-j}$ via permutation $\pi(e_{-j})$ or
   Rademacher sign-flipping $s \odot e_{-j}$ (when ``randomization="sign_flip"``).
3. **Synthetic Outcome Reconstruction**: Form $Y^* = \hat y_{-j} + \pi(e_{-j})$.
   The nuisance relationship between $X_{-j}$ and $Y$ is preserved, while any
   true contribution from $X_j$ is destroyed under the null.
4. **Full-Model Batch Refit**: Refit the full model on $(X, Y^*)$ and extract
   the null slope draws $\beta^*_j$.

Double/Debiased Machine Learning (DML) Integration
--------------------------------------------------
When `reduced_model` is supplied, this module executes Double/Debiased Machine Learning
(DML; Chernozhukov et al. 2018) out-of-fold:
- Cross-fits non-linear nuisance regressions $g(Z) = \mathbb{E}[Y \mid Z]$ and
  $m(Z) = \mathbb{E}[X \mid Z]$ using scikit-learn estimators.
- Achieves **Neyman orthogonality**, preventing first-order bias from non-linear
  confounder leakage and protecting Type I error control against power collapse.
- Employs `GroupKFold` when cluster labels are present to eliminate cross-fold leakage.

Inferential Guarantees
----------------------
* **Linear / GLM Freedman–Lane**: [ Guarantee: Asymptotically Exact (Canonical Freedman–Lane) ]
  Reduced-model residuals are correlated through the hat matrix, providing asymptotic
  size control under exchangeability (Anderson & Robinson 2001).
* **DML Freedman–Lane**: [ Guarantee: Asymptotically Exact (DML) ]
  Asymptotically exact under root-n rate conditions on nuisance learners.
* **Rademacher Sign-Flipping**: [ Guarantee: Exact Under Symmetry ]
  Exact under conditional residual symmetry about zero.

Cross-Module Integration
------------------------
* Inbound: Instantiated by :func:`~._strategies.resolve_strategy` when
  ``method="freedman_lane"`` or ``"freedman_lane_joint"``.
* Outbound: Dispatches batch refits to :meth:`~._backends.BackendProtocol.batch_ols`
  or GLM batch solvers; calls :func:`~.families.fit_reduced` for nuisance fitting.

References
----------
Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported
    significance levels. *Journal of Business & Economic Statistics*, 1(4), 292–298.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.,
    & Robins, J. (2018). Double/debiased machine learning for treatment and
    structural parameters. *The Econometrics Journal*, 21(1), C1–C68.

Anderson, M. J., & Legendre, P. (1999). An empirical comparison of permutation
    methods for tests of partial regression coefficients in a linear model.
    *Journal of Statistical Computation and Simulation*, 62(3), 271–303.

Anderson, M. J., & Robinson, J. (2001). Permutation tests for linear models.
    *Australian & New Zealand Journal of Statistics*, 43(1), 75–88.

Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014).
    Permutation inference for the general linear model. *NeuroImage*, 92, 381–397.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..families import _cross_fit_residualize, fit_reduced
from . import _apply_randomization, _residual_joint_statistic

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
        reduced_model: Any | None = None,
        groups: np.ndarray | None = None,
        random_state: int | None = None,
        **kwargs: Any,
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
            reduced_model: Optional flexible machine learning estimator
                implementing scikit-learn's fit/predict interface for
                Double/Debiased Machine Learning (DML).
            groups: Optional cluster group labels for GroupKFold cross-fitting.
            random_state: Optional seed for reproducible cross-fitting.
            **kwargs: Additional keyword arguments for protocol parity.

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

        if reduced_model is not None:
            # ---------------------------------------------------------- #
            # Double/Debiased Machine Learning (DML-DR) Pathway
            # (Chernozhukov et al. 2018; Robinson 1988)
            # ---------------------------------------------------------- #
            # When true confounding is nonlinear, single-residualized
            # Freedman–Lane suffers severe omitted variable bias because
            # linear projection cannot absorb nonlinear confounder signals
            # present in X_test. Double-Residualized DML cross-fits both Y
            # and X_test against confounders Z out-of-fold, achieving Neyman
            # orthogonality and restoring nominal Type I control and unbiased
            # coefficient estimation.
            test_cols = [c for c in X.columns if c not in confounders]
            if not test_cols:
                for c in confounders:
                    c_idx = X.columns.get_loc(c)
                    result[:, c_idx] = model_coefs[c_idx]
                return result

            test_idx = [X.columns.get_loc(c) for c in test_cols]
            conf_idx = [X.columns.get_loc(c) for c in confounders]
            Z = X_np[:, conf_idx]  # (n, q) confounder design
            X_test = X_np[:, test_idx]  # (n, p_t) tested features

            # Step 1: Out-of-fold double cross-fitting
            # Purges nonlinear nuisance functions g(Z) = E[Y|Z] and
            # m(Z) = E[X|Z] to obtain out-of-fold residuals:
            # y_tilde = Y - g_hat(Z), X_test_tilde = X_test - m_hat(Z).
            y_tilde, X_test_tilde = _cross_fit_residualize(
                Z,
                y_values,
                X=X_test,
                reduced_model=reduced_model,
                groups=groups,
                random_state=random_state,
            )
            if X_test_tilde.ndim == 1:
                X_test_tilde = X_test_tilde.reshape(-1, 1)

            p_t = len(test_cols)
            # Step 2: Loop over tested features. Under Frisch–Waugh–Lovell (FWL),
            # partial out other tested features in the double-residualized space.
            for k in range(p_t):
                full_col = test_idx[k]
                x_k = X_test_tilde[:, k]  # (n,) target tested predictor

                if p_t > 1:
                    # Other tested features to partial out linearly
                    other_k = np.delete(X_test_tilde, k, axis=1)  # (n, p_t - 1)
                    if fit_intercept:
                        W = np.column_stack([np.ones(n), other_k])  # (n, p_t)
                    else:
                        W = other_k  # (n, p_t - 1)
                    # FWL projection matrix M = I - W (W^T W)^(-1) W^T
                    W_pinv = np.linalg.pinv(W)
                    v_k = x_k - W @ (W_pinv @ x_k)  # residualized tested feature
                    e_k = y_tilde - W @ (
                        W_pinv @ y_tilde
                    )  # residualized response under H0(k)
                else:
                    if fit_intercept:
                        v_k = x_k - np.mean(x_k)
                        e_k = y_tilde - np.mean(y_tilde)
                    else:
                        v_k = x_k
                        e_k = y_tilde

                denom = float(np.dot(v_k, v_k))
                if denom > 1e-14:
                    beta_dml = float(np.dot(v_k, e_k) / denom)
                    w_k = v_k / denom  # (n,) projection vector
                else:
                    beta_dml = 0.0
                    w_k = np.zeros_like(v_k)

                # Overwrite observed coefficient for tested feature with debiased DML point estimate
                model_coefs[full_col] = beta_dml

                # Step 3: Permute reduced residuals under H0(k)
                # By orthogonality of v_k to W, the refit on Y* = W*beta + pi(e_k)
                # simplifies to: beta*_b = w_k^T pi_b(e_k). Evaluated across all B
                # permutations in a single vectorised BLAS matrix multiplication.
                E_pi = _apply_randomization(e_k, perm_indices, randomization)  # (B, n)
                result[:, full_col] = np.asarray(E_pi @ w_k)

            # Confounder columns keep observed coefficients
            for c in confounders:
                c_idx = X.columns.get_loc(c)
                result[:, c_idx] = model_coefs[c_idx]

            return result

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

            # Steps 2-4: whiten (if applicable), permute, reconstruct, refit —
            # owned by the family so it cannot diverge from method="score",
            # which computes the same estimator through the same seam.
            all_coefs, _ = family.residual_permutation_refit(
                X_np,
                preds_red,
                resids_red,
                perm_indices,
                fit_intercept=fit_intercept,
                randomization=randomization,
                n_jobs=n_jobs,
            )  # (B, p)
            result[:, j] = np.asarray(all_coefs)[:, j]

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
        reduced_model: Any | None = None,
        groups: np.ndarray | None = None,
        random_state: int | None = None,
        **kwargs: Any,
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
        n = perm_indices.shape[1]  # n observations

        if reduced_model is not None:
            # ---------------------------------------------------------- #
            # Double-Residualized DML Joint Improvement Test
            # ---------------------------------------------------------- #
            if not features_to_test:
                return 0.0, np.zeros(perm_indices.shape[0]), metric_type, []

            test_idx = [X.columns.get_loc(c) for c in features_to_test]
            conf_idx = [X.columns.get_loc(c) for c in confounders]
            Z = X_np[:, conf_idx]  # (n, q)
            X_test = X_np[:, test_idx]  # (n, p_t)

            # Step 1: Out-of-fold double cross-fitting
            y_tilde, X_test_tilde = _cross_fit_residualize(
                Z,
                y_values,
                X=X_test,
                reduced_model=reduced_model,
                groups=groups,
                random_state=random_state,
            )
            if X_test_tilde.ndim == 1:
                X_test_tilde = X_test_tilde.reshape(-1, 1)

            if fit_intercept:
                y_tilde_c = y_tilde - np.mean(y_tilde)
                X_test_tilde_c = X_test_tilde - np.mean(X_test_tilde, axis=0)
            else:
                y_tilde_c = y_tilde
                X_test_tilde_c = X_test_tilde

            # Step 2: Orthonormal basis for tested subspace via thin QR:
            # P_X = Q Q^T where Q has orthonormal columns (n, p_t).
            Q, _ = np.linalg.qr(X_test_tilde_c, mode="reduced")

            # Step 3: Observed improvement = ||P_X y_tilde||^2 = ||Q^T y_tilde||^2
            obs_proj = Q.T @ y_tilde_c  # (p_t,)
            obs_improvement = float(np.sum(obs_proj**2))

            # Step 4: Permuted improvements = ||Q^T pi(y_tilde)||^2
            E_pi = _apply_randomization(
                y_tilde_c, perm_indices, randomization
            )  # (B, n)
            A = E_pi @ Q  # (B, p_t)
            perm_improvements = np.sum(A**2, axis=1)  # (B,)

            return (
                obs_improvement,
                perm_improvements,
                metric_type,
                features_to_test,
            )

        # Z = confounder design matrix (n, q_z).
        if confounders:
            conf_idx = [X.columns.get_loc(c) for c in confounders]  # column positions
            Z = X_np[:, conf_idx]  # (n, q_z)
        else:
            Z = np.zeros((n, 0))  # (n, 0) — no confounders

        # --- Canonical Freedman–Lane reconstruction + refit, shared with
        # ScoreJointStrategy's residual-based branch so equivalent methods
        # evaluate the same joint statistic. ---
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

        return (
            obs_improvement,
            perm_improvements,
            metric_type,
            features_to_test,
        )

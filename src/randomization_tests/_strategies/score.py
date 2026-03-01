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

* **LMM:** Score p-values = Freedman–Lane p-values (bit-for-bit
  identical, since both compute ``A @ Y*``).
* **Linear (no confounders):** Score p-values = ter Braak p-values
  (A_j @ e_π ≡ pinv(X)[j] @ e_π).
* **GLMM (Plan C):** The one-step corrector upgrades score accuracy
  to second-order (Le Cam estimator).  Handled inside
  ``family.score_project()``, transparent to this strategy.

Two strategies:

* **ScoreIndividualStrategy** — per-coefficient test (``is_joint=False``).
* **ScoreJointStrategy** — collective improvement test (``is_joint=True``).

And:

* **ScoreExactStrategy** — PQL-fixed vmap for GLMM families.  Holds
  θ̂ fixed and runs full IRLS per permutation via ``jax.vmap``.
  ~500× the one-step corrector but gives exact β̂(θ̂_fixed).

References:
    Rao, C. R. (1948). Large sample tests of composite hypotheses.
    Sankhyā, 9(1), 44–56.

    Le Cam, L. (1960). Locally asymptotically normal families of
    distributions. *University of California Publications in
    Statistics*, 3, 37–98.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

import numpy as np
import pandas as pd

from ..families import _augment_intercept, fit_reduced

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
                j,
                e,
                perm_indices,
                fit_intercept=fit_intercept,
                y=y_values,
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
                j,
                e,
                identity,
                fit_intercept=fit_intercept,
                y=y_values,
            )[0]  # scalar — A_j @ e (unpermuted)
            result[:, j] = model_coefs[j] + (raw_scores - observed_raw)

        return result


# ------------------------------------------------------------------ #
# Score joint
# ------------------------------------------------------------------ #


@final
class ScoreJointStrategy:
    """Score projection — collective improvement test.

    Tests whether all non-confounder features collectively improve
    model fit beyond confounders alone.  Uses the same
    ``family.score_project()`` mechanism as the individual strategy
    but aggregates across features via RSS reduction (same metric
    as Freedman–Lane joint).
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
        """Run the joint score projection permutation algorithm.

        The test statistic is the RSS reduction from adding the tested
        features to the confounder-only model.  For each permutation,
        the full-model predictions are reconstructed via score
        projection (one matmul per feature), then the RSS is computed.
        This matches the Freedman–Lane joint metric exactly for LMM
        and linear families.

        Args:
            X: Feature matrix as a pandas DataFrame.
            y_values: Response vector of shape ``(n,)``.
            family: Resolved ``ModelFamily`` instance.
            perm_indices: Pre-generated permutation indices ``(B, n)``.
            confounders: Confounder column names.
            model_coefs: Observed coefficients ``(p,)``.
            fit_intercept: Whether to include an intercept.
            n_jobs: Unused (score is fully vectorised).

        Returns:
            ``(obs_improvement, perm_improvements, metric_type,
            features_tested)`` tuple.
        """
        if confounders is None:
            confounders = []

        features_to_test = [c for c in X.columns if c not in confounders]
        metric_type = family.metric_label

        X_np = X.values.astype(float)  # (n, p)
        n_perm, n = perm_indices.shape  # B, n

        # Confounder design matrix — same as Freedman–Lane joint.
        if confounders:
            conf_idx = [X.columns.get_loc(c) for c in confounders]
            Z = X_np[:, conf_idx]  # (n, q_z)
        else:
            Z = np.zeros((n, 0))  # (n, 0)

        # Reduced model (confounders only).
        _, preds_reduced = fit_reduced(family, Z, y_values, fit_intercept)

        # Observed improvement: reduced metric - full metric.
        base_metric = family.fit_metric(y_values, preds_reduced)
        full_model = family.fit(X_np, y_values, fit_intercept)
        preds_full = family.predict(full_model, X_np)
        obs_improvement = base_metric - family.fit_metric(y_values, preds_full)

        # Full-model residuals — permuted to build Y*.
        full_resids = family.residuals(full_model, X_np, y_values)  # (n,)

        # Build permuted Y*: ŷ_reduced + e_π[full-model].
        perm_resids = full_resids[perm_indices]  # (B, n)
        rng = np.random.default_rng(int(perm_indices[0, 0]))
        Y_perm = family.reconstruct_y(
            preds_reduced[np.newaxis, :], perm_resids, rng
        )  # (B, n)

        # For each permutation, compute full-model scores and RSS.
        # Reuse batch_fit_and_score — same as Freedman–Lane joint.
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


# ------------------------------------------------------------------ #
# Score exact (PQL-fixed) — GLMM per-permutation IRLS
# ------------------------------------------------------------------ #


@final
class ScoreExactStrategy:
    """PQL-fixed exact permutation for GLMM families.

    Holds θ̂ (variance components) from the null model fixed and
    runs the full IRLS inner loop per permutation, vectorized via
    ``jax.vmap``.  Each vmap lane converges in 3–8 iterations from
    warm start; cost is ~500× the one-step corrector but gives
    exact β̂(θ̂_fixed) per permutation.

    Only supported for GLMM families (``LogisticMixedFamily``,
    ``PoissonMixedFamily``).  Non-GLMM families raise ValueError.
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
    ) -> np.ndarray:
        """PQL-fixed exact permutation via vmapped IRLS.

        Fits the full model on each permuted y_π at fixed Γ⁻¹ via
        a single ``jax.vmap`` call, then extracts per-feature β̂.
        Returns ``(B, p)`` on the coefficient scale, matching
        ``ScoreIndividualStrategy``.

        Args:
            X: Feature matrix as a pandas DataFrame.
            y_values: Response vector ``(n,)``.
            family: Calibrated GLMM ``ModelFamily`` instance.
            perm_indices: Pre-generated permutation indices ``(B, n)``.
            confounders: Confounder column names.
            model_coefs: Observed coefficients ``(p,)``.
            fit_intercept: Whether to include an intercept.
            n_jobs: Unused (vmap handles vectorisation).

        Returns:
            ``(B, n_features)`` with permuted coefficients.

        Raises:
            ValueError: If the family is not a GLMM (no ``log_chol``).
        """
        # ---- Guard: GLMM only ------------------------------------
        if not hasattr(family, "log_chol") or family.log_chol is None:
            raise ValueError(
                f"method='score_exact' requires a calibrated GLMM family "
                f"(LogisticMixedFamily or PoissonMixedFamily).  "
                f"Got family='{family.name}'.  Use method='score' for "
                f"one-step corrector or method='ter_braak' for non-GLMM."
            )

        if confounders is None:
            confounders = []
        if model_coefs is None:
            raise ValueError("model_coefs is required for score_exact.")

        # ---- Resolve family-specific IRLS function ---------------
        from ..families_mixed import LogisticMixedFamily, PoissonMixedFamily

        if isinstance(family, LogisticMixedFamily):
            from .._backends._jax import (
                _logistic_nll_np,
            )
            from .._backends._jax import (
                _logistic_working_response_and_weights as wfn,
            )

            nll_np = _logistic_nll_np
            glm_family = "logistic"
        elif isinstance(family, PoissonMixedFamily):
            from .._backends._jax import (
                _poisson_nll_np,
            )
            from .._backends._jax import (
                _poisson_working_response_and_weights as wfn,
            )

            nll_np = _poisson_nll_np
            glm_family = "poisson"
        else:
            raise ValueError(
                f"method='score_exact' is not supported for family='{family.name}'."
            )

        from .._backends._jax import (
            _fill_lower_triangular_np,
            _fit_glm_irls,
            _pql_fixed_irls_vmap,
        )

        # ---- Build Γ⁻¹ from stored log-Cholesky params ----------
        #
        # θ is a packed 1-D vector of log-Cholesky parameters
        # produced by the JAX REML solver.  For each random-effects
        # factor k (with G_k groups, each d_k-dimensional):
        #
        #   1. Extract the d_k(d_k+1)/2 elements for factor k.
        #   2. Unpack into L_k — a lower-triangular matrix whose
        #      diagonal entries are exp(θ_diag) (log-Cholesky
        #      parameterisation ensures positive definiteness).
        #   3. Σ_k = L_k L_k'  — the d_k × d_k random-effects
        #      covariance for one group in factor k.
        #   4. Σ_k⁻¹ — the per-group precision matrix.
        #   5. Kronecker-replicate across G_k groups:
        #      block_k = I_{G_k} ⊗ Σ_k⁻¹   (block-diagonal).
        #   6. Insert block_k into the global q × q precision
        #      matrix Γ⁻¹ at the rows/cols for factor k.
        #
        assert family.re_struct is not None
        assert family.Z is not None
        re_struct = family.re_struct
        log_chol = family.log_chol
        q = family.Z.shape[1]

        Gamma_inv = np.zeros((q, q))
        factor_offset = 0
        theta_offset = 0
        for k in range(len(re_struct)):
            G_k, d_k = re_struct[k]
            n_chol_k = d_k * (d_k + 1) // 2  # vech length for L_k
            theta_k = log_chol[theta_offset : theta_offset + n_chol_k]
            L_k = _fill_lower_triangular_np(theta_k, d_k)  # θ → L_k
            Sigma_k = L_k @ L_k.T  # L_k L_k' = Σ_k
            Sigma_k_inv = np.linalg.solve(Sigma_k, np.eye(d_k))  # Σ_k⁻¹
            block_k = np.kron(np.eye(G_k), Sigma_k_inv)  # I_G ⊗ Σ_k⁻¹
            size_k = G_k * d_k
            Gamma_inv[
                factor_offset : factor_offset + size_k,
                factor_offset : factor_offset + size_k,
            ] = block_k
            factor_offset += size_k
            theta_offset += n_chol_k

        # ---- Full-model PQL-fixed IRLS via vmap ------------------
        X_np = X.values.astype(float)
        Z_np = family.Z
        n_features = X_np.shape[1]

        if fit_intercept:
            X_aug = _augment_intercept(X_np)
        else:
            X_aug = np.asarray(X_np)

        # GLM warm-start for the full model
        glm_res = _fit_glm_irls(X_aug, y_values, wfn, nll_np, family=glm_family)

        # Permuted y — all permutations at once
        y_perm = y_values[perm_indices]  # (B, n)

        # Single vmap call: fit full model on each permuted y
        beta_perm = _pql_fixed_irls_vmap(
            X_aug,
            Z_np,
            y_perm,
            Gamma_inv,
            glm_res.beta,
            wfn,
        )  # (B, p_aug)

        # Extract slope coefficients (skip intercept if present)
        if fit_intercept:
            beta_slopes = beta_perm[:, 1:]  # (B, n_features)
        else:
            beta_slopes = beta_perm  # (B, n_features)

        # Mask confounders with observed coefficients
        for j in range(n_features):
            col_name = X.columns[j]
            if col_name in confounders:
                beta_slopes[:, j] = model_coefs[j]

        return beta_slopes

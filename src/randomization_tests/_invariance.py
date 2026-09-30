"""Multi-environment causal invariance testing module (Phase 7, Steps 18-20).

Tests conditional distribution invariance across environments:
    H₀: P(Y | X) is identical across environments E ∈ {1, ..., K}
equivalently formulated as the conditional independence hypothesis:
    H₀: Y ⟂ E | X

Supports three rigorous tiers:
- Tier 0: Exact discrete cell stratification (method="stratified") via ExchangeabilityTree
- Tier 1: Residualized joint E-block testing (method="freedman_lane", "kennedy", "score")
- Tier 2: Conditional Permutation Test (method="conditional") via CPT MCMC (BWBS 2020)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._compat import DataFrameLike, _ensure_pandas_df
from ._cpt import cpt_sample
from ._results import InvarianceResult, JointTestResult
from .core import randomization_test_regression
from .exchangeability import ExchangeabilityTree
from .families import ModelFamily, resolve_family

if TYPE_CHECKING:
    pass


def invariance_test(
    X: DataFrameLike,
    y: DataFrameLike | pd.Series | np.ndarray,
    predictor: str,
    environments: np.ndarray | pd.Series | str,
    family: str | ModelFamily = "auto",
    method: str = "auto",
    reduced_model: Any | None = None,
    n_randomizations: int = 5000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> InvarianceResult:
    """Test conditional distribution invariance P(Y | X) across heterogeneous environments.

    Tests whether the mechanism generating response Y from predictor of interest
    and conditioning covariates X is invariant across environments E, formalized as
    the conditional independence null:

    .. math::
        H_0: Y \\perp\\!\\!\\perp E \\mid X

    Harmonizes with canonical permutation frameworks across three rigorous tiers:
    - **Tier 0 (Exact Discrete Conditioning)**: When conditioning covariates in X
      are discrete, permutes E strictly within discrete X-cells via
      :class:`ExchangeabilityTree`, guaranteeing finite-sample exactness
      (Shah & Peters 2020; Axioms D22, D26).
    - **Tier 1 (Residualized Joint E-block)**: Constructs environment dummy variables
      and slope interactions {E_k, E_k × X_pred}, and evaluates the omnibus joint
      invariance null using residualized permutation testing. Supports:
      * ``"freedman_lane"`` (default for linear models & DML): permutes reduced-model
        residuals e_X = (I - P_X) Y holding design [X, E] fixed (Freedman & Lane 1983;
        Anderson & Robinson 2001). Supports Double/Debiased Machine Learning (DML) via
        ``reduced_model=`` to absorb non-linear confounding.
      * ``"kennedy"``: partials out X from E-block to form exposure residuals
        E_tilde = (I - P_X) E, leaving Y untouched (Kennedy 1995). Ideal for discrete GLMs.
      * ``"score"``: evaluates the Rao (1948) score vector at the reduced model beta_hat_X.
    - **Tier 2 (Conditional Permutation Test)**: Uses the MCMC pairwise-swap sampler
      of Berrett, Wang, Barber & Samworth (2020, JRSS-B §5) to evaluate the Peters et al.
      (2016) max-coefficient deviation statistic under the conditional law P(pi | X).
      Type I error is guaranteed to satisfy:
      .. math::
          \\mathbb{P}(\\text{Type I Error}) \\le \\alpha + \\text{TV}(P_{E|X}, \\hat{P}_{E|X})

    Args:
        X: Covariate feature matrix as a pandas or Polars DataFrame.
        y: Response vector or single-column DataFrame/Series.
        predictor: Column name in X representing the primary predictor of interest.
        environments: 1D array/Series of environment labels, or the column name in X
            containing environment identifiers.
        family: Regression family name (e.g. ``"linear"``, ``"logistic"``) or
            :class:`ModelFamily` instance. ``"auto"`` detects outcome type.
        method: Permutation strategy: ``"auto"`` (default, selects Tier 0 if X is
            discrete, else Tier 1), ``"freedman_lane"``, ``"kennedy"``, ``"score"``,
            ``"conditional"``, or ``"stratified"``.
        reduced_model: Optional flexible machine learning estimator implementing
            scikit-learn's fit/predict interface for Double/Debiased Machine Learning (DML).
        n_randomizations: Number of randomization draws (default 5000).
        confidence_level: Confidence level 1 - alpha for invariance decision (default 0.95).
        random_state: Optional seed for reproducible permutations.

    Returns:
        :class:`~randomization_tests._results.InvarianceResult` containing decision,
        p-value, per-environment slopes, pooled slope, max deviation, and tier metadata.

    Raises:
        ValueError: If inputs are invalid, predictor/environment is missing, or
            fewer than 2 environments exist.
    """
    valid_methods = (
        "auto",
        "freedman_lane",
        "kennedy",
        "score",
        "conditional",
        "stratified",
    )
    if method not in valid_methods:
        msg = f"Unknown method {method!r}. Choose from: {valid_methods}."
        raise ValueError(msg)

    if not 0.0 < confidence_level < 1.0:
        msg = f"confidence_level must be in (0, 1), got {confidence_level}."
        raise ValueError(msg)

    alpha = 1.0 - confidence_level

    # Normalize X to pandas DataFrame
    X_df = _ensure_pandas_df(X, name="X").copy()

    # Extract or validate environments
    if isinstance(environments, str):
        if environments not in X_df.columns:
            msg = f"environments column {environments!r} not found in X."
            raise ValueError(msg)
        env_series = X_df[environments].astype(str)
        X_covariates = X_df.drop(columns=[environments]).copy()
    else:
        env_series = pd.Series(np.asarray(environments).astype(str), index=X_df.index)
        X_covariates = X_df.copy()

    if predictor not in X_covariates.columns:
        msg = (
            f"predictor {predictor!r} not found in X columns: "
            f"{list(X_covariates.columns)}."
        )
        raise ValueError(msg)

    n_samples = len(X_covariates)
    if len(env_series) != n_samples:
        msg = f"Length mismatch: X has {n_samples} rows, but environments has {len(env_series)} items."
        raise ValueError(msg)

    # Normalize y to pandas DataFrame
    if isinstance(y, pd.DataFrame):
        y_df = y.copy()
    elif isinstance(y, pd.Series):
        y_df = y.to_frame()
    else:
        y_df = pd.DataFrame(
            {"target": np.asarray(y, dtype=np.float64).ravel()},
            index=X_covariates.index,
        )

    if len(y_df) != n_samples:
        msg = f"Length mismatch: X has {n_samples} rows, but y has {len(y_df)} rows."
        raise ValueError(msg)

    unique_envs = np.unique(env_series)
    K = len(unique_envs)
    if K < 2:
        msg = f"invariance_test requires at least 2 distinct environments, got {K}."
        raise ValueError(msg)

    # Resolve model family
    y_vec = y_df.iloc[:, 0].to_numpy(dtype=np.float64)
    resolved_family = resolve_family(family, y_vec)

    # Extract design matrix and target predictor index
    X_np = X_covariates.to_numpy(dtype=np.float64)
    pred_idx = list(X_covariates.columns).index(predictor)

    # Step 1: Compute pooled and per-environment slope estimates
    model_pool = resolved_family.fit(X_np, y_vec, fit_intercept=True)
    coefs_pool = resolved_family.coefs(model_pool)
    pooled_coef = float(coefs_pool[pred_idx])

    env_coefs: dict[str, float] = {}
    env_arr = env_series.to_numpy()

    for e in unique_envs:
        mask_e = env_arr == e
        n_e = int(np.sum(mask_e))
        if n_e >= X_np.shape[1] + 1:
            try:
                m_e = resolved_family.fit(
                    X_np[mask_e], y_vec[mask_e], fit_intercept=True
                )
                c_e = resolved_family.coefs(m_e)
                env_coefs[str(e)] = float(c_e[pred_idx])
            except Exception:
                env_coefs[str(e)] = np.nan
        else:
            env_coefs[str(e)] = np.nan

    valid_deviations = [
        abs(b - pooled_coef) for b in env_coefs.values() if not np.isnan(b)
    ]
    max_dev = float(max(valid_deviations)) if valid_deviations else 0.0

    # Step 2: Determine strategy tier
    def _is_discrete_series(s: pd.Series) -> bool:
        if (
            s.dtype == object
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_integer_dtype(s)
        ):
            return True
        n_unq = len(np.unique(s.dropna()))
        return n_unq <= min(10, max(2, len(s) // 5))

    all_discrete = all(
        _is_discrete_series(X_covariates[c]) for c in X_covariates.columns
    )

    if method == "auto":
        if all_discrete:
            chosen_method = "stratified"
        elif resolved_family.name == "linear" or reduced_model is not None:
            chosen_method = "freedman_lane"
        else:
            chosen_method = "kennedy"
    else:
        chosen_method = method

    diag: dict[str, Any] = {
        "n_environments": K,
        "environment_counts": {str(e): int(np.sum(env_arr == e)) for e in unique_envs},
        "target_predictor": predictor,
        "conditioning_covariates": [c for c in X_covariates.columns if c != predictor],
    }
    if reduced_model is not None:
        diag["reduced_model"] = type(reduced_model).__name__

    # Step 3: Execute selected tier

    # -------------------------------------------------------------------------
    # Tier 0: Exact Discrete Stratification via ExchangeabilityTree
    # -------------------------------------------------------------------------
    if chosen_method == "stratified":
        strata_cols = [c for c in X_covariates.columns if c != predictor]
        if not strata_cols:
            strata_cols = [predictor]
        strata_labels = (
            X_covariates[strata_cols].astype(str).agg("_".join, axis=1).values
        )
        tree = ExchangeabilityTree.from_labels([strata_labels])

        # Construct augmented E-block
        augmented_df = X_covariates.copy()
        other_envs = unique_envs[1:]
        tested_cols: list[str] = []

        for idx_e, e_val in enumerate(other_envs):
            d_col = f"__env_{idx_e}__"
            i_col = f"__env_x_{predictor}_{idx_e}__"
            mask_val = (env_arr == e_val).astype(float)
            augmented_df[d_col] = mask_val
            augmented_df[i_col] = mask_val * X_covariates[predictor].values
            tested_cols.extend([d_col, i_col])

        joint_res = randomization_test_regression(
            augmented_df,
            y_df,
            confounders=list(X_covariates.columns),
            method="kennedy_joint",
            groups=tree,
            permutation_strategy="within",
            n_randomizations=n_randomizations,
            family=resolved_family,
            fit_intercept=True,
            random_state=random_state,
        )
        assert isinstance(joint_res, JointTestResult)

        p_val = float(joint_res.p_value)
        null_dist = np.asarray(joint_res.permuted_improvements)
        diag["observed_joint_improvement"] = float(joint_res.observed_improvement)
        diag["joint_metric_type"] = joint_res.metric_type

        return InvarianceResult(
            is_invariant=bool(p_val >= alpha),
            p_value=p_val,
            environment_coefficients=env_coefs,
            pooled_coefficient=pooled_coef,
            max_deviation=max_dev,
            null_distribution=null_dist,
            tier="tier_0_stratified",
            guarantee_tier="finite_sample_exact",
            method="stratified",
            diagnostics=diag,
        )

    # -------------------------------------------------------------------------
    # Tier 2: Conditional Permutation Test (CPT MCMC)
    # -------------------------------------------------------------------------
    if chosen_method == "conditional":
        draws = cpt_sample(
            env_arr,
            X_np,
            n_draws=n_randomizations,
            random_state=random_state,
        )
        T_perm = np.zeros(n_randomizations, dtype=np.float64)

        for d in range(n_randomizations):
            env_d = draws[d]
            devs_d: list[float] = []
            for e in unique_envs:
                mask_de = env_d == e
                n_de = int(np.sum(mask_de))
                if n_de >= X_np.shape[1] + 1:
                    try:
                        m_de = resolved_family.fit(
                            X_np[mask_de], y_vec[mask_de], fit_intercept=True
                        )
                        c_de = resolved_family.coefs(m_de)
                        devs_d.append(abs(float(c_de[pred_idx]) - pooled_coef))
                    except Exception:
                        pass
            T_perm[d] = max(devs_d) if devs_d else 0.0

        p_val = float((1.0 + np.sum(T_perm >= max_dev)) / (n_randomizations + 1.0))
        diag["cpt_mcmc_draws"] = n_randomizations

        return InvarianceResult(
            is_invariant=bool(p_val >= alpha),
            p_value=p_val,
            environment_coefficients=env_coefs,
            pooled_coefficient=pooled_coef,
            max_deviation=max_dev,
            null_distribution=T_perm,
            tier="tier_2_cpt",
            guarantee_tier="exact_up_to_tv_error",
            method="conditional",
            diagnostics=diag,
        )

    # -------------------------------------------------------------------------
    # Tier 1: Continuous Residualized Joint E-block Test
    # -------------------------------------------------------------------------
    augmented_df = X_covariates.copy()
    other_envs = unique_envs[1:]
    tested_cols = []

    for idx_e, e_val in enumerate(other_envs):
        d_col = f"__env_{idx_e}__"
        i_col = f"__env_x_{predictor}_{idx_e}__"
        mask_val = (env_arr == e_val).astype(float)
        augmented_df[d_col] = mask_val
        augmented_df[i_col] = mask_val * X_covariates[predictor].values
        tested_cols.extend([d_col, i_col])

    strategy_method = f"{chosen_method}_joint"

    joint_res = randomization_test_regression(
        augmented_df,
        y_df,
        confounders=list(X_covariates.columns),
        method=strategy_method,
        reduced_model=reduced_model,
        n_randomizations=n_randomizations,
        family=resolved_family,
        fit_intercept=True,
        random_state=random_state,
    )
    assert isinstance(joint_res, JointTestResult)

    p_val = float(joint_res.p_value)
    null_dist = np.asarray(joint_res.permuted_improvements)
    diag["observed_joint_improvement"] = float(joint_res.observed_improvement)
    diag["joint_metric_type"] = joint_res.metric_type

    return InvarianceResult(
        is_invariant=bool(p_val >= alpha),
        p_value=p_val,
        environment_coefficients=env_coefs,
        pooled_coefficient=pooled_coef,
        max_deviation=max_dev,
        null_distribution=null_dist,
        tier="tier_1_residualized",
        guarantee_tier="asymptotically_exact",
        method=chosen_method,
        diagnostics=diag,
    )

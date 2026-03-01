"""Extended model diagnostics for permutation test results.

Provides per-predictor and model-level diagnostics that help users
assess whether the assumptions underlying permutation tests are met
and interpret coefficient importance.

Per-predictor diagnostics:

* **Standardized coefficients** — rescale coefficients so that
  predictors measured on different scales are comparable.  For linear
  models: β* = β · SD(X_j) / SD(Y).  For logistic models:
  β* = β · SD(X_j) (log-odds change per SD of X).

* **Variance Inflation Factor (VIF)** — quantifies multicollinearity.
  VIF_j = 1 / (1 − R²_j) where R²_j is the R² from regressing X_j on
  all other predictors.  VIF > 5 indicates problematic collinearity;
  VIF > 10 is severe.  High VIF makes coefficient estimates unstable
  regardless of whether you use permutation or asymptotic inference.

* **Monte Carlo standard error** — the SE of the empirical p-value
  itself, due to the finite number of permutations B:
  SE = √[p̂(1 − p̂) / (B + 1)].  Reports precision of the p-value
  estimate so users know when to increase B.

* **Permutation-vs-asymptotic divergence** — flags predictors where the
  empirical (permutation) and classical (asymptotic) p-values lead to
  different conclusions at the chosen significance level.  Divergence
  highlights exactly where the permutation test adds value over
  classical inference.

Model-level diagnostics:

* **Breusch-Pagan test** (linear only) — tests the null hypothesis of
  homoscedasticity (constant error variance).  Significant results
  indicate heteroscedastic residuals, which violates the
  exchangeability assumption underlying the ter Braak method.

  Reference: Breusch, T. S. & Pagan, A. R. (1979). A simple test for
  heteroscedasticity and random coefficient variation.
  *Econometrica*, 47(5), 1287–1294.

* **Deviance residual diagnostics** (logistic only) — summarises the
  deviance residuals of the fitted model.  Under a correctly specified
  model, deviance residuals should have mean ≈ 0, variance ≈ 1, and
  no systematic patterns.  A high count of |d_i| > 2 flags poorly fit
  observations.  A runs test on the sorted residuals detects
  non-exchangeability patterns.

* **Cook's distance** — identifies influential observations.  Both linear
  and logistic models delegate to the statsmodels influence API
  (``OLSInfluence`` / ``GLMInfluence``), which computes
  D_i = r*²_i · h_i / (p · (1 − h_i)).  Observations with D_i > 4/n
  are flagged.

  Reference: Cook, R. D. (1977). Detection of influential observation
  in linear regression. *Technometrics*, 19(1), 15–18.

* **Effective permutation coverage** — the ratio B / n!, representing
  what fraction of the total permutation space was sampled.  For small
  n this is informative (e.g., 5000/40320 = 12.4% for n=8); for large
  n it is essentially zero and reported as such.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as _sp_stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning as SmConvergenceWarning,
)
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationWarning,
)

from .families import _augment_intercept

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .families import ModelFamily

# ------------------------------------------------------------------ #
# Per-predictor diagnostics
# ------------------------------------------------------------------ #
#
# These diagnostics are computed once per predictor and help users
# assess three things:
#
# 1. **Relative importance** — Standardized coefficients rescale
#    each β_j so that predictors measured on different scales (e.g.,
#    income in dollars vs. age in years) become comparable.  Without
#    standardization, the magnitude of a coefficient is confounded
#    with the unit of measurement.
#
# 2. **Estimate stability** — VIF quantifies how much the variance
#    of β̂_j is inflated by collinearity.  A VIF of 10 means the
#    standard error is √10 ≈ 3.2× larger than it would be if X_j
#    were uncorrelated with all other predictors.  This matters
#    equally for permutation and asymptotic inference.
#
# 3. **p-value precision** — Monte Carlo SE tells users how much
#    noise is in the empirical p-value itself.  If MC SE is large
#    relative to the distance between the p-value and the threshold,
#    the significance conclusion is uncertain and B should be
#    increased.


def _bca_percentile(
    boot_dist: np.ndarray,
    observed_stat: float,
    jackknife_stats: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) percentile interval.

    Statistic-agnostic implementation of the BCa adjustment.
    The three pieces — bias correction z₀, acceleration â, and
    adjusted percentiles — are generic; the caller is responsible
    for computing the observed statistic and jackknife replicates.

    Args:
        boot_dist: Bootstrap (or permutation) replicates of the
            statistic, shape ``(B,)``.
        observed_stat: Point estimate of the statistic.
        jackknife_stats: Leave-one-out estimates, shape ``(n,)``.
        alpha: ``1 - confidence_level`` (e.g. 0.05 for 95% CI).

    Returns:
        ``(ci_lower, ci_upper)`` tuple.
    """
    # --- Bias correction constant z₀ ---
    prop_less = np.mean(boot_dist < observed_stat)
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = _sp_stats.norm.ppf(prop_less)

    # --- Acceleration constant â via jackknife ---
    theta_bar = np.mean(jackknife_stats)
    diffs = theta_bar - jackknife_stats
    a_hat = np.sum(diffs**3) / (6.0 * (np.sum(diffs**2)) ** 1.5 + 1e-10)

    # --- Adjusted percentiles ---
    z_alpha_lower = _sp_stats.norm.ppf(alpha / 2)
    z_alpha_upper = _sp_stats.norm.ppf(1 - alpha / 2)

    denom_lower = 1 - a_hat * (z0 + z_alpha_lower)
    denom_upper = 1 - a_hat * (z0 + z_alpha_upper)

    # Guard against division by zero / near-zero denominators which
    # produce NaN or extreme percentiles.
    if abs(denom_lower) < 1e-10 or abs(denom_upper) < 1e-10:
        # Fall back to simple percentile CI.
        ci_lower = float(np.percentile(boot_dist, (alpha / 2) * 100))
        ci_upper = float(np.percentile(boot_dist, (1 - alpha / 2) * 100))
        return ci_lower, ci_upper

    p_lower = _sp_stats.norm.cdf(z0 + (z0 + z_alpha_lower) / denom_lower)
    p_upper = _sp_stats.norm.cdf(z0 + (z0 + z_alpha_upper) / denom_upper)

    # Clip to valid percentile range.
    B = len(boot_dist)
    p_lower = np.clip(p_lower, 0.5 / B, 1 - 0.5 / B)
    p_upper = np.clip(p_upper, 0.5 / B, 1 - 0.5 / B)

    # Guard against NaN from degenerate distributions.
    if np.isnan(p_lower) or np.isnan(p_upper):
        ci_lower = float(np.percentile(boot_dist, (alpha / 2) * 100))
        ci_upper = float(np.percentile(boot_dist, (1 - alpha / 2) * 100))
        return ci_lower, ci_upper

    ci_lower = float(np.percentile(boot_dist, p_lower * 100))
    ci_upper = float(np.percentile(boot_dist, p_upper * 100))

    return ci_lower, ci_upper


def compute_jackknife_coefs(
    family: ModelFamily,
    X: np.ndarray,
    y_values: np.ndarray,
    fit_intercept: bool,
) -> np.ndarray | None:
    """Leave-one-out coefficient estimates for BCa acceleration.

    Computes ``n`` jackknife replicates by fitting the model with each
    observation removed in turn.  Uses ``batch_fit_paired()`` for
    vectorised LOO when the family supports it, falling back to a
    sequential loop otherwise.

    Args:
        family: Model family instance.
        X: Design matrix, shape ``(n, p)``.
        y_values: Response vector, shape ``(n,)``.
        fit_intercept: Whether the model includes an intercept.

    Returns:
        Jackknife coefficient matrix ``(n, p)`` or ``None`` when
        ``n > 500`` (BCa's advantage over percentile CIs is
        negligible at large *n*, and *n* refits is too expensive).
    """
    n, p = X.shape
    if n > 500:
        return None

    # Build (n, n-1) leave-one-out index array.
    loo_idx = np.empty((n, n - 1), dtype=int)
    for i in range(n):
        loo_idx[i] = np.concatenate([np.arange(i), np.arange(i + 1, n)])

    # Vectorised path via batch_fit_paired (3-D X, 2-D Y).
    if hasattr(family, "batch_fit_paired"):
        X_loo = X[loo_idx]  # (n, n-1, p)
        Y_loo = y_values[loo_idx].astype(float)  # (n, n-1)
        try:
            jack_coefs: np.ndarray = family.batch_fit_paired(
                X_loo, Y_loo, fit_intercept=fit_intercept
            )
            # Ensure shape is (n, p) — multinomial returns (n, p) of
            # Wald χ² values via coefs().
            if jack_coefs.shape == (n, p):
                return jack_coefs
        except Exception:
            logger.debug(
                "batch_fit_paired failed for jackknife; falling back to "
                "sequential loop",
                exc_info=True,
            )

    # Sequential fallback (ordinal, multinomial, LMM, or batch failure).
    jack_coefs_seq = np.empty((n, p))
    for i in range(n):
        X_loo_i = X[loo_idx[i]]
        y_loo_i = y_values[loo_idx[i]].astype(float)
        try:
            model_i = family.fit(X_loo_i, y_loo_i, fit_intercept)
            jack_coefs_seq[i] = family.coefs(model_i)[:p]
        except Exception:
            jack_coefs_seq[i] = np.nan
    return jack_coefs_seq


# ------------------------------------------------------------------ #
# Confidence interval computation
# ------------------------------------------------------------------ #


def compute_permutation_ci(
    permuted_coefs: np.ndarray,
    model_coefs: np.ndarray,
    method: str,
    alpha: float,
    jackknife_coefs: np.ndarray | None,
    confounders: list[str],
    feature_names: list[str],
) -> np.ndarray:
    """Confidence intervals for regression coefficients from the permutation distribution.

    Applies strategy-aware centering: ter Braak and Freedman–Lane null
    distributions are centred on zero and must be shifted by ``+β̂``
    before CI computation.  Score and Kennedy distributions are already
    centred on the observed coefficient.

    When jackknife coefficients are available, BCa intervals are
    computed via :func:`_bca_percentile`; otherwise simple shifted-
    percentile intervals are returned.

    Args:
        permuted_coefs: Permuted coefficient matrix ``(B, p)``.
        model_coefs: Observed coefficients ``(p,)``.
        method: Strategy name (``"ter_braak"``, ``"freedman_lane"``,
            ``"kennedy"``, ``"score"``).
        alpha: ``1 - confidence_level``.
        jackknife_coefs: Leave-one-out coefficients ``(n, p)`` or
            ``None``.
        confounders: Confounder column names.
        feature_names: Feature column names.

    Returns:
        CI array of shape ``(p, 2)``.
    """
    p = len(model_coefs)

    # Strategy-aware centering (vectorised).
    needs_shift = method in ("ter_braak", "freedman_lane")
    shifted = (
        permuted_coefs + model_coefs[np.newaxis, :] if needs_shift else permuted_coefs
    )

    # Confounder mask — these columns get NaN CIs.
    confounder_set = set(confounders)
    is_confounder = np.array([fn in confounder_set for fn in feature_names])

    if jackknife_coefs is not None:
        # ---- BCa: vectorise z₀ and â across all p columns ----
        # Bias correction z₀ = Φ⁻¹(mean(boot < θ̂))
        prop_less = np.mean(shifted < model_coefs[np.newaxis, :], axis=0)  # (p,)
        prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
        z0 = _sp_stats.norm.ppf(prop_less)  # (p,)

        # Acceleration â via jackknife
        theta_bar = np.mean(jackknife_coefs, axis=0)  # (p,)
        diffs = theta_bar[np.newaxis, :] - jackknife_coefs  # (n, p)
        sum_d3 = np.sum(diffs**3, axis=0)
        sum_d2_pow = np.sum(diffs**2, axis=0) ** 1.5
        a_hat = sum_d3 / (6.0 * sum_d2_pow + 1e-10)  # (p,)

        # Adjusted percentiles
        z_lo = _sp_stats.norm.ppf(alpha / 2)
        z_hi = _sp_stats.norm.ppf(1 - alpha / 2)

        denom_lo = 1.0 - a_hat * (z0 + z_lo)
        denom_hi = 1.0 - a_hat * (z0 + z_hi)

        # Degenerate columns → fall back to simple percentile
        degenerate = (np.abs(denom_lo) < 1e-10) | (np.abs(denom_hi) < 1e-10)
        safe_denom_lo = np.where(degenerate, 1.0, denom_lo)
        safe_denom_hi = np.where(degenerate, 1.0, denom_hi)

        p_lo = _sp_stats.norm.cdf(z0 + (z0 + z_lo) / safe_denom_lo)
        p_hi = _sp_stats.norm.cdf(z0 + (z0 + z_hi) / safe_denom_hi)

        B = shifted.shape[0]
        p_lo = np.clip(p_lo, 0.5 / B, 1 - 0.5 / B)
        p_hi = np.clip(p_hi, 0.5 / B, 1 - 0.5 / B)

        degenerate |= np.isnan(p_lo) | np.isnan(p_hi)

        # Merge BCa and fallback percentiles
        pct_lo = np.where(degenerate, alpha / 2, p_lo) * 100
        pct_hi = np.where(degenerate, 1 - alpha / 2, p_hi) * 100

        # Per-column percentile (different percentile per column)
        ci = np.empty((p, 2))
        for j in range(p):
            ci[j, 0] = np.percentile(shifted[:, j], pct_lo[j])
            ci[j, 1] = np.percentile(shifted[:, j], pct_hi[j])
    else:
        # Simple shifted-percentile CI
        lo_pct = alpha / 2 * 100
        hi_pct = (1 - alpha / 2) * 100
        ci = np.column_stack(
            [
                np.percentile(shifted, lo_pct, axis=0),
                np.percentile(shifted, hi_pct, axis=0),
            ]
        )

    # Mask confounder columns
    ci[is_confounder] = np.nan

    return ci


def compute_pvalue_ci(
    counts: np.ndarray,
    n_permutations: int,
    alpha: float,
) -> np.ndarray:
    """Clopper-Pearson exact binomial CIs for empirical p-values.

    Ref: Clopper, C. J. & Pearson, E. S. (1934), "The use of
    confidence or fiducial limits illustrated in the case of the
    binomial", *Biometrika*, 26(4), 404–413.

    Aligned with the Phipson & Smyth (2010) ``(b+1)/(B+1)`` estimator:
    ``successes = counts + 1``, ``trials = B + 1``.

    Args:
        counts: Number of permuted |β*| ≥ observed |β| per feature,
            shape ``(p,)``.
        n_permutations: Total permutation count *B*.
        alpha: ``1 - confidence_level``.

    Returns:
        CI array of shape ``(p, 2)``.
    """
    successes = counts + 1  # (p,)
    trials = n_permutations + 1

    # Lower bound: Beta.ppf(α/2, s, n-s+1), with s=0 → 0
    a_lo = np.maximum(successes, 1)  # guard against a=0
    b_lo = trials - successes + 1
    lower = np.where(
        successes == 0,
        0.0,
        _sp_stats.beta.ppf(alpha / 2, a_lo, b_lo),
    )

    # Upper bound: Beta.ppf(1−α/2, s+1, n-s), with s=n → 1
    a_hi = successes + 1
    b_hi = np.maximum(trials - successes, 1)  # guard against b=0
    upper = np.where(
        successes == trials,
        1.0,
        _sp_stats.beta.ppf(1 - alpha / 2, a_hi, b_hi),
    )

    return np.column_stack([lower, upper])  # (p, 2)


def compute_wald_ci(
    observed_model: Any,
    family: ModelFamily,
    n_features: int,
    alpha: float,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Classical Wald CIs from the fitted statsmodels model.

    For **linear** and **logistic** families the pipeline uses sklearn
    estimators that lack ``conf_int()``.  When the sklearn model is
    detected (no ``conf_int`` attribute) and *X* / *y* are supplied,
    a lightweight statsmodels OLS or Logit model is fitted on the fly
    to obtain asymptotic Wald intervals.

    Family dispatch:

    - **Linear / Logistic / Poisson / Negative Binomial**:
      ``model.conf_int(alpha)`` returns ``(p_aug, 2)`` with intercept.
      Intercept row stripped → ``(p, 2)``.
    - **Ordinal**: ``model.conf_int(alpha)`` returns
      ``(p + K-1, 2)`` (slopes then thresholds).
      Slice ``[:n_features]``.
    - **Multinomial**: ``model.conf_int(alpha)`` returns
      ``(K-1, p_aug, 2)`` (statsmodels ≥ 0.14) or
      ``((K-1)·p_aug, 2)`` (older).  Reshaped to ``(p, K-1, 2)``
      for per-category CIs.  Per-predictor Wald CI set to NaN.
    - **Mixed models**: ``conf_int()`` if available from statsmodels.

    Args:
        observed_model: Fitted model object (sklearn or statsmodels).
        family: Model family instance.
        n_features: Number of slope features.
        alpha: ``1 - confidence_level``.
        X: Design matrix ``(n, p)`` — needed for statsmodels refit
            when the primary model is an sklearn estimator.
        y: Response vector ``(n,)`` — same purpose as *X*.
        fit_intercept: Whether the model includes an intercept.

    Returns:
        ``(wald_ci, category_wald_ci)`` — ``wald_ci`` is ``(p, 2)``;
        ``category_wald_ci`` is ``(p, K-1, 2)`` for multinomial,
        ``None`` otherwise.
    """
    import statsmodels.api as sm

    nan_ci = np.full((n_features, 2), np.nan)
    name = family.name

    # ── sklearn refit for linear / logistic ────────────────────── #
    if name in ("linear", "logistic") and not hasattr(observed_model, "conf_int"):
        if X is None or y is None:
            return nan_ci, None
        try:
            X_aug = _augment_intercept(X, fit_intercept)
            if name == "linear":
                sm_model = sm.OLS(y, X_aug).fit()
            else:
                sm_model = sm.Logit(y, X_aug).fit(disp=0)
            raw_ci = np.asarray(sm_model.conf_int(alpha))
        except Exception:
            return nan_ci, None
    else:
        try:
            raw_ci = np.asarray(observed_model.conf_int(alpha))
        except Exception:
            return nan_ci, None

    if name in ("linear", "logistic", "poisson", "negative_binomial"):
        # Strip intercept row (row 0).
        if raw_ci.shape[0] == n_features + 1:
            return raw_ci[1 : n_features + 1], None
        # No intercept case.
        if raw_ci.shape[0] == n_features:
            return raw_ci[:n_features], None
        return nan_ci, None

    if name in ("linear_mixed", "logistic_mixed", "poisson_mixed"):
        # statsmodels MixedLMResults.conf_int() returns (p_aug, 2).
        if raw_ci.shape[0] >= n_features + 1:
            return raw_ci[1 : n_features + 1], None
        if raw_ci.shape[0] >= n_features:
            return raw_ci[:n_features], None
        return nan_ci, None

    if name == "ordinal":
        # OrderedModel: slopes first, then thresholds.
        if raw_ci.shape[0] >= n_features:
            return raw_ci[:n_features], None
        return nan_ci, None

    if name == "multinomial":
        # MNLogit conf_int() may return either:
        #   - 3-D array of shape (K-1, p_aug, 2)   [statsmodels ≥ 0.14]
        #   - 2-D array of shape ((K-1)·p_aug, 2)   [older releases]
        # In both cases, strip the intercept column and transpose to
        # (p, K-1, 2).
        p_aug = n_features + 1

        if raw_ci.ndim == 3:
            # Already (K-1, p_aug, 2).
            if raw_ci.shape[1] < p_aug:
                return nan_ci, None
            cat_ci = raw_ci[:, 1:, :].transpose(1, 0, 2)  # (p, K-1, 2)
            return nan_ci, cat_ci

        # 2-D fallback: ((K-1)·p_aug, 2).
        n_rows = raw_ci.shape[0]
        if n_rows < p_aug:
            return nan_ci, None
        k_minus_1 = n_rows // p_aug
        if k_minus_1 * p_aug != n_rows:
            return nan_ci, None
        reshaped = raw_ci.reshape(k_minus_1, p_aug, 2)
        cat_ci = reshaped[:, 1:, :].transpose(1, 0, 2)  # (p, K-1, 2)
        return nan_ci, cat_ci

    return nan_ci, None


def compute_standardized_ci(
    permutation_ci: np.ndarray,
    model_coefs: np.ndarray,
    X: pd.DataFrame,
    y_values: np.ndarray,
    family: ModelFamily,
) -> np.ndarray:
    """CIs for standardised (beta-weight) coefficients.

    Applies the same per-feature scaling factor used in
    :func:`compute_standardized_coefs` to the permutation CI endpoints.
    This is valid because standardisation is a linear transformation
    of the coefficient.

    Args:
        permutation_ci: Permutation CI array ``(p, 2)``.
        model_coefs: Observed coefficients ``(p,)``.
        X: Feature matrix.
        y_values: Response vector.
        family: Model family instance.

    Returns:
        Standardised CI array ``(p, 2)``.
    """
    p = len(model_coefs)
    sd_x = np.std(X.values, axis=0, ddof=1)

    if family.name == "multinomial":
        return np.full((p, 2), np.nan)

    if family.name in ("linear", "linear_mixed"):
        sd_y = np.std(y_values, ddof=1)
        if sd_y == 0:
            return np.zeros((p, 2))
        scale = sd_x / sd_y
    else:
        # Log-odds / log-link families.
        scale = sd_x

    result: np.ndarray = permutation_ci * scale[:, np.newaxis]
    return result


def compute_standardized_coefs(
    X: pd.DataFrame,
    y_values: np.ndarray,
    model_coefs: np.ndarray,
    family: ModelFamily,
) -> np.ndarray:
    """Compute standardized (beta-weight) coefficients.

    The formula depends on the family's link function:

    - **Identity link** (``"linear"``, ``"linear_mixed"``):
      ``β* = β · SD(X_j) / SD(Y)`` — the classic beta weight.
    - **Log-odds link** (``"logistic"``, ``"logistic_mixed"``,
      ``"ordinal"``):
      ``β* = β · SD(X_j)`` — log-odds change per one-SD increase.
    - **Log link** (``"poisson"``, ``"poisson_mixed"``,
      ``"negative_binomial"``):
      ``β* = β · SD(X_j)`` — log-rate change per one-SD increase.
    - **Multinomial** (``"multinomial"``):
      returns ``NaN`` — ``coefs()`` returns Wald χ² statistics, not
      regression coefficients, so standardisation is not meaningful.

    Args:
        X: Feature matrix.
        y_values: Response vector.
        model_coefs: Raw coefficients, shape ``(n_features,)``.
        family: :class:`ModelFamily` instance.  Controls the
            standardisation formula.

    Returns:
        Array of standardized coefficients, shape ``(n_features,)``.
    """
    # Sample standard deviations (ddof=1 for Bessel's correction)
    # of each column of X.  Shape: (n_features,).
    sd_x = np.std(X.values, axis=0, ddof=1)

    # Multinomial: coefs() returns Wald χ² statistics, not regression
    # coefficients — standardisation is meaningless.
    if family.name == "multinomial":
        return np.full_like(model_coefs, np.nan, dtype=float)

    # Identity-link families: classic beta weight β · SD(X) / SD(Y).
    if family.name in ("linear", "linear_mixed"):
        sd_y = np.std(y_values, ddof=1)
        if sd_y == 0:
            return np.zeros_like(model_coefs)
        result: np.ndarray = model_coefs * sd_x / sd_y
        return result

    # Log-odds link (logistic, ordinal) and log link (Poisson, NB):
    # β · SD(X_j).  There is no natural SD(Y) denominator on the
    # link scale.
    result = model_coefs * sd_x
    return result


def compute_vif(X: pd.DataFrame) -> np.ndarray:
    """Compute Variance Inflation Factors for each predictor.

    Uses the algebraic identity VIF_j = [(X'X)⁻¹]_jj · (X'X)_jj on
    centered predictors (centering absorbs the implicit intercept in
    each auxiliary regression X_j ~ X_{-j} + 1).  One matrix inverse,
    no loop.  Ref: Greene, *Econometric Analysis*, §4.9.

    Args:
        X: Feature matrix.

    Returns:
        Array of VIFs, shape ``(n_features,)``.
    """
    X_np = X.values.astype(float)
    n_features = X_np.shape[1]

    if n_features <= 1:
        # Only one predictor — no collinearity possible.
        return np.ones(n_features)

    # Center columns to absorb the implicit intercept in each
    # auxiliary regression X_j ~ X_{-j} + 1.
    X_c = X_np - X_np.mean(axis=0)
    C = X_c.T @ X_c  # (p, p)

    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        # Singular — at least one predictor is perfectly collinear.
        # Use pseudo-inverse; perfectly collinear directions get
        # near-zero eigenvalues → large diagonal entries → large VIF.
        C_inv = np.linalg.pinv(C)

    # VIF_j = (X'X)_jj · [(X'X)⁻¹]_jj
    vifs = np.diag(C) * np.diag(C_inv)
    return np.asarray(vifs)


def compute_monte_carlo_se(
    raw_p_values: np.ndarray,
    n_permutations: int,
) -> np.ndarray:
    """Compute Monte Carlo standard error of empirical p-values.

    SE = √[p̂(1 − p̂) / (B + 1)]

    This quantifies the precision of the p-value estimate due to the
    finite number of permutations.

    Args:
        raw_p_values: Numeric empirical p-values, shape ``(n_features,)``.
        n_permutations: Number of permutations (B).

    Returns:
        Array of standard errors, shape ``(n_features,)``.
    """
    # The empirical p-value p̂ is a binomial proportion estimated
    # from B+1 equally-likely arrangements (see Phipson & Smyth 2010
    # in pvalues.py).  Its standard error under the binomial model is
    # SE = √[p̂(1 − p̂) / (B + 1)].
    #
    # Example: with B = 5000 and p̂ = 0.05, SE ≈ 0.003 — a 95% CI
    # of roughly [0.044, 0.056].  This tells the user the p-value is
    # well-resolved.  Near p̂ = 0.5, SE is maximised at ~0.007.
    p = np.asarray(raw_p_values)
    result: np.ndarray = np.sqrt(p * (1.0 - p) / (n_permutations + 1))
    return result


def compute_divergence_flags(
    raw_empirical_p: np.ndarray,
    raw_classic_p: np.ndarray,
    threshold: float = 0.05,
) -> list[str]:
    """Flag predictors where empirical and asymptotic p-values diverge.

    Divergence is defined as the two p-values leading to different
    significance conclusions at the given *threshold*.

    Args:
        raw_empirical_p: Numeric empirical p-values.
        raw_classic_p: Numeric classical p-values.
        threshold: Significance level for comparison.

    Returns:
        List of flag strings: ``"DIVERGENT"`` or ``""``.
    """
    emp = np.asarray(raw_empirical_p)
    cls = np.asarray(raw_classic_p)
    flags: list[str] = []

    # For each predictor, check whether the two p-values fall on
    # opposite sides of the significance threshold.  A "DIVERGENT"
    # flag means one method rejects H₀ while the other does not —
    # precisely the situation where the permutation test adds value
    # over classical inference (e.g., non-normal residuals, small
    # samples, or influential observations distorting the Wald test).
    #
    # Confounders in Kennedy methods have NaN p-values; these are
    # silently skipped (flagged as "").
    for e, c in zip(emp, cls, strict=True):
        if np.isnan(e) or np.isnan(c):
            flags.append("")
        elif (e < threshold) != (c < threshold):
            flags.append("DIVERGENT")
        else:
            flags.append("")
    return flags


# ------------------------------------------------------------------ #
# Model-level diagnostics
# ------------------------------------------------------------------ #
#
# Model-level diagnostics assess whether the global assumptions needed
# for valid permutation inference are satisfied.
#
# **Exchangeability** is the key requirement.  Under H₀: β_j = 0,
# the residuals (ter Braak) or exposure residuals (Kennedy) must be
# exchangeable — i.e., their joint distribution is invariant to
# permutation.  For OLS, this follows from i.i.d. errors with
# constant variance.  Breusch-Pagan detects violations of constant
# variance (heteroscedasticity).  For logistic models, deviance
# residual patterns serve an analogous role.
#
# **Influential observations** can distort both the observed
# coefficient and the permutation null distribution.  A single high-
# leverage point may produce an artificially extreme β̂ that no
# permuted sample can match, yielding a spuriously small p-value.
# Cook's D identifies such points.
#
# **Permutation coverage** reports B / n!, the fraction of the total
# permutation space that was sampled.  This is primarily informative
# for small n (where the full space is enumerable) and reassures
# users that B is adequate.


def compute_breusch_pagan(
    X: pd.DataFrame,
    y_values: np.ndarray,
) -> dict:
    """Run the Breusch-Pagan test for heteroscedasticity.

    Tests H₀: the error variances are all equal (homoscedasticity).
    A small p-value indicates the residuals are heteroscedastic,
    which violates the exchangeability assumption for ter Braak.

    Args:
        X: Feature matrix.
        y_values: Response vector.

    Returns:
        Dictionary with ``lm_stat``, ``lm_p_value``, ``f_stat``,
        ``f_p_value``, and a ``warning`` string (empty if no issue).
    """
    # Fit OLS on the full model to obtain residuals.
    X_aug = _augment_intercept(X)
    ols_result = sm.OLS(y_values, X_aug).fit()

    # The Breusch-Pagan test regresses squared residuals on X and
    # tests whether the resulting R² is significantly non-zero.
    # Two versions are returned:
    #   LM (Lagrange Multiplier): n · R² ~ χ²(p)
    #   F:  (R² / p) / ((1 − R²) / (n − p − 1)) ~ F(p, n − p − 1)
    # The F variant is preferred for small samples; both are reported.
    lm_stat, lm_p, f_stat, f_p = het_breuschpagan(
        ols_result.resid,
        X_aug,
    )

    # Flag heteroscedasticity at α = 0.05.  When present, the
    # residuals under the reduced model are NOT exchangeable because
    # their variances depend on X — the ter Braak permutation scheme
    # is then anti-conservative.
    warning = ""
    if lm_p < 0.05:
        warning = (
            f"Breusch-Pagan p = {lm_p:.4f}: residuals may be "
            f"heteroscedastic — exchangeability assumption may be violated."
        )
    return {
        "lm_stat": np.round(lm_stat, 4),
        "lm_p_value": lm_p,
        "f_stat": np.round(f_stat, 4),
        "f_p_value": f_p,
        "warning": warning,
    }


def compute_deviance_residual_diagnostics(
    X: pd.DataFrame,
    y_values: np.ndarray,
) -> dict:
    """Compute deviance residual diagnostics for logistic regression.

    Deviance residuals under a correctly specified model should have
    mean ≈ 0 and variance ≈ 1.  Observations with |d_i| > 2 are
    poorly fit.  A Wald-Wolfowitz runs test on the signs of sorted
    residuals detects non-random patterns (non-exchangeability).

    Args:
        X: Feature matrix.
        y_values: Binary response vector.

    Returns:
        Dictionary with ``mean``, ``variance``, ``n_extreme``
        (count of |d_i| > 2), ``runs_test_z``, ``runs_test_p``,
        and a ``warning`` string.
    """
    # Fit the full logistic model via statsmodels.  The disp=0 flag
    # suppresses the iteration log.
    X_aug = _augment_intercept(X)
    sm_model = sm.Logit(y_values, X_aug).fit(disp=0)

    # --- Deviance residuals ---
    #
    # For observation i, the deviance residual is:
    #
    #   d_i = sign(y_i − p̂_i) · √[ −2 (y_i log p̂_i + (1−y_i) log(1−p̂_i)) ]
    #
    # Under a correctly specified model:
    #   • E[d_i] ≈ 0
    #   • Var(d_i) ≈ 1
    #   • |d_i| > 2 flags poorly fit observations (analogous to
    #     standardized residuals > 2 in OLS)
    #
    # Convert to plain numpy to avoid pandas index-alignment issues
    # in downstream arithmetic.
    dev_resid = np.asarray(sm_model.resid_dev)

    mean_d = np.mean(dev_resid)
    var_d = np.var(dev_resid, ddof=1)
    n_extreme = int(np.sum(np.abs(dev_resid) > 2))

    # --- Wald-Wolfowitz runs test ---
    #
    # Sort deviance residuals by fitted probability p̂_i (ascending)
    # and classify each as positive or negative.  Under
    # exchangeability, the signs should be randomly interspersed —
    # clusters of same-sign residuals at certain probability levels
    # indicate the model fits systematically poorly in those regions.
    #
    # A significant runs-test p-value (<0.05) warns that the logistic
    # model may be mis-specified, which undermines the exchangeability
    # assumption that permutation tests require.
    sorted_idx = np.argsort(np.asarray(sm_model.fittedvalues))
    signs = (dev_resid[sorted_idx] >= 0).astype(int)
    runs_z, runs_p = _runs_test(signs)

    warning = ""
    warnings_parts: list[str] = []
    if n_extreme > 0:
        warnings_parts.append(
            f"{n_extreme} observation(s) with |deviance residual| > 2"
        )
    if runs_p < 0.05:
        warnings_parts.append(
            f"runs test p = {runs_p:.4f}: residuals show non-random pattern"
        )
    if warnings_parts:
        warning = "; ".join(warnings_parts) + "."

    return {
        "mean": np.round(mean_d, 4),
        "variance": np.round(var_d, 4),
        "n_extreme": n_extreme,
        "runs_test_z": np.round(runs_z, 4),
        "runs_test_p": runs_p,
        "warning": warning,
    }


def _runs_test(binary_seq: np.ndarray) -> tuple[float, float]:
    """Wald-Wolfowitz runs test for randomness of a binary sequence.

    A "run" is a maximal consecutive subsequence of identical values.
    Under H₀ (random arrangement), the number of runs R has a known
    mean and variance:

        μ_R = 2·n₊·n₋ / n + 1
        σ²_R = 2·n₊·n₋·(2·n₊·n₋ − n) / (n²·(n − 1))

    The test statistic Z = (R − μ_R) / σ_R is approximately standard
    normal for n ≥ 20.

    Args:
        binary_seq: Array of 0s and 1s.

    Returns:
        ``(z_statistic, p_value)`` tuple.
    """
    from scipy import stats

    n = len(binary_seq)
    if n < 2:
        return 0.0, 1.0

    # Count positives (1s) and negatives (0s) in the sequence.
    n_pos = int(np.sum(binary_seq))
    n_neg = n - n_pos

    # Degenerate case: all values are the same — no information about
    # randomness.
    if n_pos == 0 or n_neg == 0:
        return 0.0, 1.0

    # Count runs — a "run" is a maximal consecutive block of identical
    # values.  For example, [0, 0, 1, 1, 0] has 3 runs: (0,0), (1,1),
    # (0).  We count transitions (where adjacent values differ) and
    # add 1.
    runs = 1 + int(np.sum(binary_seq[1:] != binary_seq[:-1]))

    # Under H₀ (random arrangement of n₊ ones and n₋ zeros), the
    # expected number of runs and its variance are:
    #
    #   μ_R  = 2·n₊·n₋ / n  +  1
    #   σ²_R = 2·n₊·n₋·(2·n₊·n₋ − n) / (n²·(n − 1))
    #
    # These follow from combinatorial arguments — see Bradley (1968)
    # "Distribution-Free Statistical Tests", Chapter 12.
    mu = 2.0 * n_pos * n_neg / n + 1.0
    var = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (n * n * (n - 1.0))

    if var <= 0:
        return 0.0, 1.0

    # Standardise to Z ~ N(0, 1) for n ≥ 20; for smaller n the
    # normal approximation is rough but still directionally useful.
    z = (runs - mu) / np.sqrt(var)
    p = 2.0 * stats.norm.sf(np.abs(z))  # two-sided

    return float(z), float(p)


def _autodiff_profile_ci(
    X: np.ndarray,
    y_values: np.ndarray,
    family: ModelFamily,
    alpha: float,
    fit_intercept: bool = True,
) -> np.ndarray:
    """Profile likelihood CIs via JAX autodiff.

    Dispatcher that selects the right NLL function and solver for the
    given family, fits the model, and delegates to
    ``_profile_ci_coefficients()`` in the JAX backend.

    Args:
        X: Feature matrix ``(n, p)`` — raw, without intercept.
        y_values: Response vector ``(n,)``.
        family: Resolved ``ModelFamily`` instance.
        alpha: ``1 - confidence_level``.
        fit_intercept: Whether the model includes an intercept.

    Returns:
        ``(n_features, 2)`` NumPy array of profile CI bounds.

    Raises:
        ImportError: If JAX is not available.
        ValueError: If the family is not supported.
    """
    import jax.numpy as jnp

    from ._backends._jax import (
        _DEFAULT_TOL,
        _logistic_nll,
        _make_negbin_nll,
        _make_negbin_solver,
        _make_newton_solver,
        _make_ordinal_nll,
        _make_ordinal_solver,
        _make_poisson_solver,
        _poisson_nll,
        _profile_ci_coefficients,
    )

    n_features = X.shape[1]
    name = family.name

    if name == "linear":
        # For linear models, the profile CI is identical to the Wald
        # CI because the log-likelihood is exactly quadratic.  No need
        # for the expensive bisection — fall back to Wald.
        raise ValueError("Linear profile CI is identical to Wald CI.")

    X_aug = _augment_intercept(X, fit_intercept)
    X_j = jnp.array(X_aug, dtype=jnp.float64)
    y_j = jnp.array(y_values, dtype=jnp.float64)

    # Intercept is column 0 when fit_intercept is True; slope
    # parameters start at index 1.
    idx_start = 1 if fit_intercept else 0
    feature_indices = list(range(idx_start, idx_start + n_features))

    if name == "logistic":
        beta, _, _ = _make_newton_solver(X_j, y_j, max_iter=100, tol=_DEFAULT_TOL)
        return _profile_ci_coefficients(
            _logistic_nll,
            beta,
            X_j,
            y_j,
            feature_indices,
            alpha,
        )

    if name == "poisson":
        beta, _, _ = _make_poisson_solver(X_j, y_j, max_iter=100, tol=_DEFAULT_TOL)
        return _profile_ci_coefficients(
            _poisson_nll,
            beta,
            X_j,
            y_j,
            feature_indices,
            alpha,
        )

    if name == "negative_binomial":
        a = family.alpha  # type: ignore[attr-defined]
        if a is None:
            raise RuntimeError(
                "Profile CI for NB requires calibrated α. Call calibrate() first."
            )
        beta, _, _ = _make_negbin_solver(
            X_j,
            y_j,
            a,
            max_iter=100,
            tol=_DEFAULT_TOL,
        )
        nll_fn = _make_negbin_nll(a)
        return _profile_ci_coefficients(
            nll_fn,
            beta,
            X_j,
            y_j,
            feature_indices,
            alpha,
        )

    if name == "ordinal":
        K = int(len(np.unique(y_values)))
        # Ordinal models DON'T use intercept augmentation — thresholds
        # serve as intercepts.  Use the raw feature matrix.
        X_bare = np.asarray(X)
        X_j_bare = jnp.array(X_bare, dtype=jnp.float64)
        params, _, _ = _make_ordinal_solver(
            X_j_bare,
            y_j,
            K,
            max_iter=100,
            tol=_DEFAULT_TOL,
        )
        nll_fn = _make_ordinal_nll(K)
        # Ordinal params: [slopes..., thresholds...].  Feature indices
        # are the first n_features elements (no intercept to skip).
        feature_indices_ord = list(range(n_features))
        return _profile_ci_coefficients(
            nll_fn,
            params,
            X_j_bare,
            y_j,
            feature_indices_ord,
            alpha,
        )

    if name == "multinomial":
        # Multinomial has (K-1) coefficients per predictor.  Profile
        # CIs per-predictor are not well-defined (joint profile over
        # K-1 params would be needed).  Return NaN.
        raise ValueError("Multinomial profile CIs not supported.")

    raise ValueError(f"Unsupported family for profile CI: {name}")


def compute_profile_ci(
    X: pd.DataFrame | np.ndarray,
    y_values: np.ndarray,
    family: ModelFamily,
    alpha: float = 0.05,
    fit_intercept: bool = True,
) -> np.ndarray:
    """Profile likelihood CIs for regression coefficients.

    Tries JAX autodiff first.  Falls back to a NaN array when JAX
    is unavailable or the family is not supported (e.g. linear models,
    where the profile CI is identical to the Wald CI, or multinomial
    models where per-predictor profile CIs are not well-defined).

    Profile CIs are based on inverting the likelihood-ratio test
    and are asymmetric — they respect the curvature of the
    likelihood surface.  For small samples they are more accurate
    than Wald CIs.

    Ref: Venzon & Moolgavkar (1988), *Applied Statistics*, 37(1).

    Args:
        X: Feature matrix ``(n, p)``.
        y_values: Response vector ``(n,)``.
        family: Resolved :class:`ModelFamily` instance.
        alpha: ``1 - confidence_level``.
        fit_intercept: Whether the model includes an intercept.

    Returns:
        ``(n_features, 2)`` NumPy array of ``[lower, upper]``
        profile CI bounds.  NaN where unavailable.
    """
    n_features = X.shape[1] if hasattr(X, "shape") else len(X.columns)  # type: ignore[union-attr]
    try:
        return _autodiff_profile_ci(
            np.asarray(X),
            np.asarray(y_values),
            family,
            alpha,
            fit_intercept,
        )
    except (ImportError, ValueError, RuntimeError):
        return np.full((n_features, 2), np.nan)


def _autodiff_cooks_distance(
    X_aug: np.ndarray,
    y_values: np.ndarray,
    family: ModelFamily,
) -> np.ndarray:
    """Compute Cook's D for any family via JAX autodiff influence function.

    Fits a single model using the family's JAX Newton solver, then
    computes D_i = (1/p) sᵢ' H⁻¹ sᵢ for each observation via
    ``_autodiff_cooks_d()`` in the JAX backend.

    This is the generalised Cook (1977) definition applicable to
    any smooth GLM family — it reduces to the standard formula for
    OLS and logistic (up to small numerical differences from the
    statsmodels closed-form implementation).

    Args:
        X_aug: Augmented design matrix ``(n, p_aug)`` (with intercept).
        y_values: Response vector ``(n,)``.
        family: Resolved ``ModelFamily`` instance.

    Returns:
        Cook's distances ``(n,)`` as a NumPy array.

    Raises:
        ImportError: If JAX is not available.
        ValueError: If the family name is not recognised.
    """
    import jax
    import jax.numpy as jnp
    from jax import jit

    from ._backends._jax import (
        _DEFAULT_TOL,
        _autodiff_cooks_d,
        _logistic_hessian,
        _logistic_nll_per_obs,
        _make_multinomial_nll,
        _make_multinomial_nll_per_obs,
        _make_multinomial_solver,
        _make_negbin_hessian,
        _make_negbin_nll_per_obs,
        _make_negbin_solver,
        _make_newton_solver,
        _make_ordinal_nll,
        _make_ordinal_nll_per_obs,
        _make_ordinal_solver,
        _make_poisson_solver,
        _poisson_hessian,
        _poisson_nll_per_obs,
    )

    X_j = jnp.array(X_aug, dtype=jnp.float64)
    y_j = jnp.array(y_values, dtype=jnp.float64)

    name = family.name

    if name == "linear":
        # OLS via logistic solver infrastructure is wasteful; use a
        # simple closed form instead:  D_i = ê²_i h_i / (p MSE (1−h_i)²)
        # But for consistency we use the autodiff path with a Gaussian
        # NLL:  nll_i = 0.5 (y_i - x_i'β)² (up to constant).
        # However, OLS doesn't have a Newton solver in _jax.py.
        # Fall back — caller should use statsmodels for linear.
        raise ValueError("Linear family should use statsmodels OLS path.")

    elif name == "logistic":
        beta, _, _ = _make_newton_solver(X_j, y_j, max_iter=100, tol=_DEFAULT_TOL)

        def hess_fn_logistic(b: jnp.ndarray) -> jnp.ndarray:
            return _logistic_hessian(b, X_j, y_j)  # type: ignore[no-any-return]

        return _autodiff_cooks_d(
            _logistic_nll_per_obs, hess_fn_logistic, beta, X_j, y_j
        )

    elif name == "poisson":
        beta, _, _ = _make_poisson_solver(X_j, y_j, max_iter=100, tol=_DEFAULT_TOL)

        def hess_fn_poisson(b: jnp.ndarray) -> jnp.ndarray:
            return _poisson_hessian(b, X_j, y_j)  # type: ignore[no-any-return]

        return _autodiff_cooks_d(_poisson_nll_per_obs, hess_fn_poisson, beta, X_j, y_j)

    elif name == "negative_binomial":
        alpha = family.alpha  # type: ignore[attr-defined]
        if alpha is None:
            raise RuntimeError(
                "NB Cook's D requires calibrated α. Call calibrate() first."
            )
        beta, _, _ = _make_negbin_solver(
            X_j, y_j, alpha, max_iter=100, tol=_DEFAULT_TOL
        )
        _nb_hess = _make_negbin_hessian(alpha)

        def hess_fn_nb(b: jnp.ndarray) -> jnp.ndarray:
            return _nb_hess(b, X_j, y_j)  # type: ignore[no-any-return]

        nll_per_obs_fn = _make_negbin_nll_per_obs(alpha)
        return _autodiff_cooks_d(nll_per_obs_fn, hess_fn_nb, beta, X_j, y_j)

    elif name == "ordinal":
        K = int(len(np.unique(y_values)))
        # Ordinal doesn't use intercept augmentation — but the caller
        # already passed X_aug.  For ordinal, we need to strip the
        # intercept column and use bare X.
        X_bare = X_aug[:, 1:]  # strip intercept
        X_j_bare = jnp.array(X_bare, dtype=jnp.float64)
        params, _, _ = _make_ordinal_solver(
            X_j_bare, y_j, K, max_iter=100, tol=_DEFAULT_TOL
        )
        _nll = _make_ordinal_nll(K)
        _ord_hess = jit(jax.hessian(_nll))

        def hess_fn_ord(p: jnp.ndarray) -> jnp.ndarray:
            return _ord_hess(p, X_j_bare, y_j)  # type: ignore[no-any-return]

        nll_per_obs_fn = _make_ordinal_nll_per_obs(K)
        return _autodiff_cooks_d(nll_per_obs_fn, hess_fn_ord, params, X_j_bare, y_j)

    elif name == "multinomial":
        K = int(len(np.unique(y_values)))
        params, _, _ = _make_multinomial_solver(
            X_j, y_j, K, max_iter=100, tol=_DEFAULT_TOL
        )
        _nll = _make_multinomial_nll(K)
        _mn_hess = jit(jax.hessian(_nll))

        def hess_fn_mn(p: jnp.ndarray) -> jnp.ndarray:
            return _mn_hess(p, X_j, y_j)  # type: ignore[no-any-return]

        nll_per_obs_fn = _make_multinomial_nll_per_obs(K)
        return _autodiff_cooks_d(nll_per_obs_fn, hess_fn_mn, params, X_j, y_j)

    else:
        raise ValueError(f"Unsupported family for autodiff Cook's D: {name}")


def compute_cooks_distance(
    X: pd.DataFrame,
    y_values: np.ndarray,
    family: ModelFamily,
) -> dict:
    """Compute Cook's distance and flag influential observations.

    Tries JAX autodiff influence function first (works for all
    families); falls back to statsmodels ``OLSInfluence`` /
    ``GLMInfluence`` for linear and logistic when JAX is unavailable.

    The autodiff definition generalises Cook (1977) to arbitrary
    smooth GLMs via the influence function:

        D_i = (1/p) sᵢ' H⁻¹ sᵢ

    where sᵢ = ∂ℓᵢ/∂β is the per-observation score and H is the
    Hessian of the NLL at β̂.

    Observations with D_i > 4/n are flagged as influential.

    Args:
        X: Feature matrix.
        y_values: Response vector.
        family: :class:`ModelFamily` instance.

    Returns:
        Dictionary with ``cooks_d`` (array), ``n_influential`` (count
        of D_i > 4/n), ``threshold`` (4/n), ``influential_indices``
        (list of flagged row indices), and a ``warning`` string.
    """
    n = len(y_values)

    # The conventional threshold for "influential" is 4/n.  Some
    # authors use 1 (exact Cook & Weisberg bound) but 4/n is more
    # suitable for the moderate-n settings typical of permutation
    # testing.
    threshold = 4.0 / n
    X_aug = _augment_intercept(X)

    # Ensure y_values is a plain numpy array so that arithmetic with
    # statsmodels fitted values (which may be pandas Series) does not
    # trigger index-alignment broadcasts.
    y_values = np.asarray(y_values)

    # ---- JAX path (preferred) ----------------------------------------
    try:
        cooks_d = _autodiff_cooks_distance(X_aug, y_values, family)
    except (ImportError, ValueError):
        # ImportError: JAX not installed.
        # ValueError:  Family not supported by autodiff (e.g. linear).
        # Fall through to statsmodels path.
        cooks_d = None

    # ---- statsmodels fallback ----------------------------------------
    if cooks_d is None:
        if family.name == "logistic":
            # GLM Cook's D via statsmodels GLMInfluence.
            # McCullagh & Nelder (1989, §12.5), Pregibon (1981).
            sm_model = sm.GLM(
                y_values,
                X_aug,
                family=sm.families.Binomial(),
            ).fit(disp=0)
            influence = sm_model.get_influence()
            cooks_d = np.asarray(influence.cooks_distance[0])
        else:
            # Default: exact OLS Cook's D via statsmodels.
            sm_model = sm.OLS(y_values, X_aug).fit()
            influence = sm_model.get_influence()
            cooks_d = np.asarray(influence.cooks_distance[0])

    # Flag observations exceeding the 4/n threshold.
    influential_mask = cooks_d > threshold
    n_influential = int(np.sum(influential_mask))
    influential_indices = list(np.where(influential_mask)[0])

    warning = ""
    if n_influential > 0:
        warning = (
            f"{n_influential} observation(s) with Cook's D > {threshold:.4f} "
            f"(4/n) — results may be driven by influential points."
        )

    return {
        "cooks_d": cooks_d,
        "n_influential": n_influential,
        "threshold": threshold,
        "influential_indices": influential_indices,
        "warning": warning,
    }


def compute_permutation_coverage(
    n_samples: int,
    n_permutations: int,
) -> dict:
    """Compute effective permutation coverage of the sample space.

    Reports B / n!, the fraction of all possible permutations that
    were actually sampled.

    Args:
        n_samples: Number of observations (n).
        n_permutations: Number of permutations drawn (B).

    Returns:
        Dictionary with ``coverage`` (float), ``n_factorial`` (int or
        ``"overflow"``), and ``coverage_str`` (human-readable string).
    """
    # Python's math.factorial handles arbitrarily large integers, but
    # for n > ~170 the result exceeds the float64 range, so we wrap
    # in a try/except.  In practice, for n > ~25 the coverage is
    # essentially zero (25! ≈ 1.6 × 10²⁵) and the string is purely
    # informational.
    n_factorial: int | str
    try:
        n_factorial = math.factorial(n_samples)
        coverage = n_permutations / n_factorial
        if coverage >= 0.001:
            coverage_str = f"{coverage:.1%} of {n_factorial:,} possible"
        else:
            coverage_str = f"< 0.1% of {n_factorial:.2e} possible"
    except (OverflowError, ValueError):
        n_factorial = "overflow"
        coverage = 0.0
        coverage_str = f"{n_permutations:,} of > 10^{n_samples} possible"

    return {
        "coverage": coverage,
        "n_factorial": n_factorial,
        "coverage_str": coverage_str,
    }


# ------------------------------------------------------------------ #
# Exposure R² (Kennedy method only)
# ------------------------------------------------------------------ #
#
# When the Kennedy method is used, each non-confounder predictor X_j
# is regressed on the confounders Z (the "exposure model").  The
# exposure R² quantifies how much of X_j's variance is explained by
# Z.  Values close to 1.0 mean virtually all of X_j's variation is
# redundant with the confounders, leaving near-zero residual variance
# available for the permutation test.  This makes the permuted
# coefficients extremely unstable and typically inflates the p-value
# toward 1.0 — not because of a test error, but because there is
# genuinely no independent signal left in X_j.


def compute_exposure_r_squared(
    X: pd.DataFrame,
    confounders: list[str],
    fit_intercept: bool = True,
) -> list[float | None]:
    """Compute exposure R² for each feature under the Kennedy method.

    For non-confounder features, this is the R² of ``X_j ~ Z``.  For
    confounder features the value is ``None`` (they are controls, not
    tested).

    Args:
        X: Feature matrix.
        confounders: Column names of confounders.
        fit_intercept: Whether an intercept is included in the
            exposure model (should match the main model).

    Returns:
        List of length ``n_features`` where each entry is a float
        R² ∈ [0, 1] for tested predictors or ``None`` for
        confounders.
    """
    result: list[float | None] = []

    if confounders:
        Z = X[confounders].values
    else:
        # No confounders: the exposure model reduces to a mean-only
        # (fit_intercept=True) or zero (fit_intercept=False) model.
        # R² will be 0 for every feature, indicating that no variance
        # is absorbed by confounders and the full signal is available.
        Z = np.zeros((len(X), 0))

    for col in X.columns:
        if col in confounders:
            # Confounders are controls, not hypotheses — they are not
            # run through the exposure model, so R² is meaningless.
            result.append(None)
            continue

        x_j = X[col].values.astype(float)

        # ----------------------------------------------------------
        # SS_total: total sum of squares of X_j.
        #
        # When fit_intercept is True, this is the centred form
        #   SS_total = Σ(xᵢ − x̄)²
        # which matches the denominator of the standard R² definition.
        #
        # When fit_intercept is False, the exposure model passes
        # through the origin, so the appropriate decomposition uses
        # uncentred sums of squares:
        #   SS_total = Σxᵢ²
        #
        # If SS_total ≈ 0, the predictor is effectively constant
        # (zero variance).  R² is undefined in that case — we report
        # 1.0 to signal "nothing left to test".
        # ----------------------------------------------------------
        if fit_intercept:
            ss_total = np.sum((x_j - x_j.mean()) ** 2)
        else:
            ss_total = np.sum(x_j**2)

        if ss_total < 1e-12:
            result.append(1.0)
            continue

        # ----------------------------------------------------------
        # Exposure-model predictions  X̂_j = Z · γ̂.
        #
        # This mirrors the exposure model used inside the Kennedy
        # permutation engine (_kennedy_individual_linear and
        # _kennedy_individual_logistic).  The R² computed here
        # therefore reflects exactly the same residual variance that
        # the permutation test operates on.
        #
        # When there are no confounders and fit_intercept is True,
        # X̂_j collapses to the column mean x̄, and the residuals are
        # mean-centred values.  When fit_intercept is False and there
        # are no confounders, X̂_j = 0, so the residuals equal the
        # raw feature values.
        # ----------------------------------------------------------
        if Z.shape[1] > 0:
            from sklearn.linear_model import LinearRegression

            exp_model = LinearRegression(
                fit_intercept=fit_intercept,
            ).fit(Z, x_j)
            x_hat = exp_model.predict(Z)
        else:
            if fit_intercept:
                x_hat = np.full_like(x_j, x_j.mean())
            else:
                x_hat = np.zeros_like(x_j)

        # R² = 1 − SS_resid / SS_total.
        # Clipped to [0, 1] as a guard against floating-point
        # arithmetic producing values marginally outside the
        # theoretical range (e.g., −1e-16 from near-perfect fits).
        ss_resid = np.sum((x_j - x_hat) ** 2)
        r2 = 1.0 - ss_resid / ss_total
        result.append(round(float(np.clip(r2, 0.0, 1.0)), 6))

    return result


# ------------------------------------------------------------------ #
# Proportional odds test (Brant-like)
# ------------------------------------------------------------------ #


def _proportional_odds_test(
    X: np.ndarray,
    y: np.ndarray,
    ord_model: Any,
) -> dict:
    """Test the proportional-odds assumption via a Brant-like χ² test.

    The test compares the pooled proportional-odds slopes to
    category-specific slopes obtained from separate binary logistic
    regressions at each cumulative threshold.  Under H₀ all slopes
    are equal; a significant χ² indicates violation.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Predictor matrix (no constant column).
    y : ndarray, shape (n,)
        Integer-coded ordinal outcome.
    ord_model : OrderedModel result
        Fitted proportional-odds model (used only to extract the
        pooled slope vector).

    Returns
    -------
    dict
        Keys ``prop_odds_chi2``, ``prop_odds_df``, ``prop_odds_p``.
        All values are ``float('nan')`` when the test cannot be
        computed (e.g. singular Hessian, too few categories).
    """
    from scipy import stats as sp_stats

    nan_result: dict = {
        "prop_odds_chi2": float("nan"),
        "prop_odds_df": float("nan"),
        "prop_odds_p": float("nan"),
    }

    try:
        categories = np.sort(np.unique(y))
        J = len(categories)  # number of categories
        if J < 3:
            return nan_result

        # pooled slopes from the proportional-odds model
        pooled_params = np.asarray(ord_model.params)
        # OrderedModel stores thresholds first, then slopes
        p = X.shape[1]
        pooled_slopes = pooled_params[-p:]  # last p entries = slopes

        # Fit J-1 separate binary logit models at each threshold
        n_thresholds = J - 1
        binary_slopes = np.empty((n_thresholds, p))
        binary_covs = []

        for j in range(n_thresholds):
            y_bin = (y > categories[j]).astype(float)
            # skip if degenerate
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                return nan_result
            X_c = _augment_intercept(X)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                logit_res = sm.Logit(y_bin, X_c).fit(disp=0, maxiter=200)
            # slopes are columns 1..p (skip intercept)
            binary_slopes[j, :] = logit_res.params[1:]
            # covariance of slope params (drop intercept row/col)
            cov_full = np.asarray(logit_res.cov_params())
            binary_covs.append(cov_full[1:, 1:])

        # Brant χ²: sum over thresholds of
        #   (β_j − β_pooled)ᵀ V_j⁻¹ (β_j − β_pooled)
        chi2_total = 0.0
        df_total = 0
        for j in range(n_thresholds):
            diff = binary_slopes[j] - pooled_slopes
            V_j = binary_covs[j]
            try:
                V_inv = np.linalg.inv(V_j)
            except np.linalg.LinAlgError:
                return nan_result
            chi2_total += float(diff @ V_inv @ diff)
            df_total += p

        p_value = float(sp_stats.chi2.sf(chi2_total, df_total))
        return {
            "prop_odds_chi2": round(chi2_total, 4),
            "prop_odds_df": df_total,
            "prop_odds_p": round(p_value, 6),
        }
    except Exception:
        return nan_result


# ------------------------------------------------------------------ #
# Aggregate helper
# ------------------------------------------------------------------ #
#
# compute_all_diagnostics() is the single entry point called by
# core.permutation_test_regression().  It orchestrates every per-
# ------------------------------------------------------------------ #
# E-value sensitivity analysis (VanderWeele & Ding, 2017)
# ------------------------------------------------------------------ #


def compute_e_value(
    coefficient: float,
    family: str,
    ci_bound: float | None = None,
    sd_x: float | None = None,
    sd_y: float | None = None,
    baseline_prevalence: float | None = None,
) -> dict:
    """Compute the E-value for unmeasured confounding sensitivity.

    The E-value is the minimum strength of association (on the RR
    scale) that an unmeasured confounder would need to have with both
    the treatment and the outcome to fully explain away the observed
    association.  Larger E-values indicate more robust findings.

    Family-dispatched conversion:

    * **linear** — Standardize β → Cohen's d, then
      RR = exp(0.91 × d).  Requires ``sd_x`` and ``sd_y``.
    * **logistic** / **ordinal** — OR = exp(β), then
      E = √OR + √(√OR × (√OR − 1)) (Cornfield inequality, conservative).
      When ``baseline_prevalence`` is provided, computes exact
      RR = OR / (1 − p₀ + p₀ × OR) for a tighter bound.
    * **poisson** / **negative_binomial** — RR = exp(β) directly.
    * **multinomial** — Returns NaN (Wald χ², not scalar log-odds).

    Mixed-family names (e.g. ``"linear_mixed"``) are automatically
    stripped to their base name.

    Args:
        coefficient: Estimated coefficient (log-OR for logistic/ordinal,
            log-RR for Poisson/NegBin, raw β for linear).
        family: Family name string (e.g. ``"linear"``, ``"logistic"``).
        ci_bound: Optional CI bound to compute E-value for.
        sd_x: Standard deviation of the predictor (linear only).
        sd_y: Standard deviation of the outcome (linear only).
        baseline_prevalence: Baseline outcome probability (logistic
            / ordinal) for exact RR conversion.

    Returns:
        Dictionary with keys ``e_value``, ``e_value_ci``, ``rr``,
        ``family``, and ``interpretation``.

    References:
        VanderWeele, T. J. & Ding, P. (2017). Sensitivity analysis in
        observational research: introducing the E-value.  *Annals of
        Internal Medicine*, 167(4), 268–274.
    """
    import math

    # Strip _mixed suffix — fixed-effect coefficients have the same
    # interpretation regardless of random-effects structure.
    _MIXED_STRIP = {
        "linear_mixed": "linear",
        "logistic_mixed": "logistic",
        "poisson_mixed": "poisson",
    }
    base_family = _MIXED_STRIP.get(family, family)

    def _e_from_rr(rr: float) -> float:
        """Standard E-value formula for RR."""
        if rr < 1:
            rr = 1 / rr
        return float(rr + math.sqrt(rr * (rr - 1)))

    def _e_from_or(or_val: float) -> float:
        """Direct E-value formula for OR via Cornfield inequality."""
        if or_val < 1:
            or_val = 1 / or_val
        sqrt_or = math.sqrt(or_val)
        return float(sqrt_or + math.sqrt(sqrt_or * (sqrt_or - 1)))

    # --- Multinomial guard ---
    if base_family == "multinomial":
        warnings.warn(
            "No clean RR conversion exists for multinomial families. "
            "Multinomial coefficients are Wald χ² statistics, not scalar "
            "log-odds ratios.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "e_value": float("nan"),
            "e_value_ci": float("nan"),
            "rr": float("nan"),
            "family": family,
            "interpretation": ("E-value is not computable for multinomial families."),
        }

    # --- Compute RR and E-value by family ---
    if base_family == "linear":
        if sd_x is None or sd_y is None:
            raise ValueError(
                "sd_x and sd_y are required for linear family E-value "
                "computation (Cohen's d conversion)."
            )
        d = coefficient * sd_x / sd_y  # Cohen's d
        # Cohen's d → RR via the probit/logistic bridge:
        #   RR ≈ exp(0.91 × |d|)
        # The constant 0.91 = π/√3 ≈ 1.814 / 2 comes from
        # equating the logistic and normal CDFs.
        # Ref: Chinn, S. (2000), *Statistics in Medicine*,
        # 19(22), 3127–3131; VanderWeele & Ding (2017), Appendix.
        rr = math.exp(0.91 * abs(d))
        e_val = _e_from_rr(rr)

        # CI bound E-value.
        if ci_bound is not None:
            d_ci = ci_bound * sd_x / sd_y
            rr_ci = math.exp(0.91 * abs(d_ci))
            e_val_ci = _e_from_rr(rr_ci)
        else:
            e_val_ci = float("nan")

    elif base_family in ("logistic", "ordinal"):
        or_val = math.exp(coefficient)
        if baseline_prevalence is not None:
            p0 = baseline_prevalence
            rr = or_val / (1 - p0 + p0 * or_val)
            e_val = _e_from_rr(rr)
        else:
            rr = or_val  # Report OR; use direct formula.
            e_val = _e_from_or(or_val)

        if ci_bound is not None:
            or_ci = math.exp(ci_bound)
            if baseline_prevalence is not None:
                rr_ci = or_ci / (1 - p0 + p0 * or_ci)
                e_val_ci = _e_from_rr(rr_ci)
            else:
                e_val_ci = _e_from_or(or_ci)
        else:
            e_val_ci = float("nan")

    elif base_family in ("poisson", "negative_binomial"):
        rr = math.exp(coefficient)
        e_val = _e_from_rr(rr)

        if ci_bound is not None:
            rr_ci = math.exp(ci_bound)
            e_val_ci = _e_from_rr(rr_ci)
        else:
            e_val_ci = float("nan")

    else:
        raise ValueError(f"Unsupported family for E-value: {family!r}")

    interpretation = (
        f"E-value = {e_val:.2f}: an unmeasured confounder would need "
        f"an association of at least RR = {e_val:.2f} with both the "
        f"treatment and the outcome to explain away the observed effect."
    )

    return {
        "e_value": round(e_val, 4),
        "e_value_ci": round(e_val_ci, 4) if not math.isnan(e_val_ci) else float("nan"),
        "rr": round(rr, 4),
        "family": family,
        "interpretation": interpretation,
    }


# ------------------------------------------------------------------ #
# Rosenbaum bounds (Ding & Miratrix 2015 adaptation)
# ------------------------------------------------------------------ #


def rosenbaum_bounds(
    result: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    gammas: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    predictor_index: int | None = None,
    alpha: float = 0.05,
) -> dict:
    """Rosenbaum sensitivity bounds for unmeasured confounding.

    Computes worst-case p-values under varying degrees of hidden bias Γ.
    At Γ = 1, there is no hidden bias and the p-value equals the
    observed p-value.  As Γ increases, the worst-case p-value grows.
    The critical Γ is the smallest value where the test stops being
    significant — a measure of robustness.

    **Restricted to LinearFamily with binary predictors**.

    * Non-linear families lack a linear decomposition for the contrast
      vector.  Use :func:`compute_e_value` instead.
    * LinearMixedFamily is rejected — REML estimation with
      variance-components weighting invalidates the OLS contrast
      vector $(X^TX)^{-1}X^T$.
    * Continuous-exposure permutation tests permute regression
      residuals, not binary treatment assignments, so the Rosenbaum
      framework does not apply.  Generalisation via influence-function
      sensitivity is deferred to v0.5.0+.

    Args:
        result: Permutation test result dict (must contain
            ``"p_values"`` and optionally ``"family"``).
        X: Design matrix used in the permutation test.
        y: Outcome vector.
        gammas: Sequence of Γ values to evaluate.
        predictor_index: Column index of the predictor of interest
            in *X*.  Defaults to 0.
        alpha: Significance level for critical Γ.

    Returns:
        Dictionary with ``predictor``, ``observed_p``,
        ``gamma_values``, ``worst_case_p``, ``critical_gamma``,
        and ``interpretation``.

    References:
        Ding, P. & Miratrix, L. W. (2015). To adjust or not to adjust?
        Sensitivity analysis of M-estimation and the role of the
        propensity score. *Statistica Sinica*, 25(2), 643–666.

        Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.).
        Springer.
    """
    # --- Family guard: linear only ---
    family_name = result.get("family", "linear")
    if isinstance(family_name, str):
        fam_name = family_name
    else:
        fam_name = getattr(family_name, "name", "linear")

    if fam_name != "linear":
        raise NotImplementedError(
            "Rosenbaum bounds require a linear test statistic. "
            "Use compute_e_value() for sensitivity analysis with "
            "non-linear families."
        )

    # --- Binary predictor guard ---
    if predictor_index is None:
        predictor_index = 0
    X_arr = np.asarray(X, dtype=float)
    x = X_arr[:, predictor_index]
    unique_vals = np.unique(x[~np.isnan(x)])
    if len(unique_vals) != 2:
        raise ValueError(
            "Rosenbaum bounds require a binary predictor "
            "(treatment/control). For continuous predictors, use "
            "compute_e_value() for sensitivity analysis."
        )

    n = len(y)
    y = np.asarray(y, dtype=float).ravel()

    # Observed p-value.
    p_values = result.get("p_values", {})
    if isinstance(p_values, dict):
        # Multi-predictor result — get the first or indexed predictor.
        keys = list(p_values.keys())
        if predictor_index < len(keys):
            observed_p = float(p_values[keys[predictor_index]])
        else:
            observed_p = float(list(p_values.values())[0])
    else:
        observed_p = float(np.asarray(p_values).ravel()[predictor_index])

    # Compute contrast vector c_i = [(X'X)^{-1}X']_{j,i}.
    # Add intercept column for OLS.
    X_design = _augment_intercept(X_arr)

    try:
        XtX_inv = np.linalg.inv(X_design.T @ X_design)
    except np.linalg.LinAlgError:
        return {
            "predictor": predictor_index,
            "observed_p": observed_p,
            "gamma_values": list(gammas),
            "worst_case_p": [float("nan")] * len(gammas),
            "critical_gamma": float("nan"),
            "interpretation": "Rosenbaum bounds unavailable: singular design matrix.",
        }

    # Contrast vector for predictor (predictor_index + 1 to skip intercept).
    contrast = (XtX_inv @ X_design.T)[predictor_index + 1, :]

    # Observed test statistic.
    beta_hat = XtX_inv @ X_design.T @ y
    t_obs = beta_hat[predictor_index + 1]

    # Residual standard error.
    p_design = X_design.shape[1]
    residuals = y - X_design @ beta_hat
    rse = float(np.sqrt(np.sum(residuals**2) / max(n - p_design, 1)))

    # For each Γ, compute worst-case p-value.
    worst_case_p: list[float] = []
    critical_gamma = float("nan")

    for gamma in gammas:
        if gamma == 1.0:
            worst_case_p.append(observed_p)
            if observed_p >= alpha and np.isnan(critical_gamma):
                critical_gamma = 1.0
            continue

        # Worst-case allocation: maximize the bias by flipping the
        # unmeasured confounder to align with the observed effect
        # direction.
        log_gamma = np.log(gamma)

        # Under worst-case Γ, the shifted null mean is:
        #   E_Γ[T] = Σ_i c_i * Γ^{u_i} / Σ_i Γ^{u_i}
        # where u_i ∈ {0, 1} is chosen to maximize deviation.
        # For the worst case, set u_i = 1 for units where c_i
        # has the sign of the observed statistic.
        sign_match = contrast * np.sign(t_obs) > 0
        u_worst = sign_match.astype(float)

        # Shifted null expectation and variance.
        #
        # Under worst-case Γ, each unit i is weighted by
        #   w_i = Γ^{u_i} / Σ_j Γ^{u_j}
        # and the linear contrast T = Σ_i c_i y_i has:
        #   E_Γ[T] = (Σ c_i w_i) × n   (rescaled by n because
        #             the contrast vector c is normalised to sum
        #             to 0 for n observations)
        #   Var_Γ[T] = RSE² × (Σ c_i² w_i) × n
        # See Rosenbaum (2002), *Observational Studies*, §4.3.
        weights = np.exp(log_gamma * u_worst)
        weights /= weights.sum()

        shifted_mean = np.sum(contrast * weights) * n
        shifted_var = rse**2 * np.sum(contrast**2 * weights) * n

        if shifted_var <= 0:
            worst_case_p.append(1.0)
        else:
            z_shifted = (t_obs - shifted_mean) / np.sqrt(shifted_var)
            p_gamma = float(2.0 * _sp_stats.norm.sf(abs(z_shifted)))
            worst_case_p.append(p_gamma)

        if worst_case_p[-1] >= alpha and np.isnan(critical_gamma):
            critical_gamma = gamma

    if np.isnan(critical_gamma) and all(p < alpha for p in worst_case_p):
        critical_gamma = float(gammas[-1])  # Robust beyond tested range.

    interpretation = (
        f"Critical Γ = {critical_gamma:.1f}: the result remains significant "
        f"at α={alpha} up to Γ = {critical_gamma:.1f}. An unmeasured "
        f"confounder would need to change treatment odds by a factor of "
        f"{critical_gamma:.1f} to alter the conclusion."
    )

    return {
        "predictor": predictor_index,
        "observed_p": round(observed_p, 6),
        "gamma_values": list(gammas),
        "worst_case_p": [round(p, 6) for p in worst_case_p],
        "critical_gamma": round(critical_gamma, 2)
        if not np.isnan(critical_gamma)
        else float("nan"),
        "interpretation": interpretation,
    }


# ---- Aggregate helper -------------------------------------------- #
#
# compute_all_diagnostics collects every per-predictor and model-level
# diagnostic in one pass and returns a flat dictionary suitable for
# inclusion in the results dict.  The caller does not need to know
# which individual functions exist — this keeps the public API
# surface small.


def compute_all_diagnostics(
    X: pd.DataFrame,
    y_values: np.ndarray,
    model_coefs: np.ndarray,
    family: ModelFamily,
    *,
    raw_empirical_p: np.ndarray,
    raw_classic_p: np.ndarray,
    n_permutations: int,
    p_value_threshold: float = 0.05,
    method: str = "ter_braak",
    confounders: list[str] | None = None,
    fit_intercept: bool = True,
    panel_id: np.ndarray | None = None,
) -> dict:
    """Compute all extended diagnostics in one call.

    Args:
        X: Feature matrix.
        y_values: Response vector.
        model_coefs: Raw coefficients, shape ``(n_features,)``.
        family: The ``ModelFamily`` instance for the active model.
            Used to dispatch family-specific diagnostics via
            ``compute_extended_diagnostics()`` and passed directly
            to helper functions.
        raw_empirical_p: Numeric empirical p-values.
        raw_classic_p: Numeric classical p-values.
        n_permutations: Number of permutations (B).
        p_value_threshold: Significance level for divergence flags.
        method: Permutation method (``'ter_braak'``, ``'kennedy'``,
            or ``'kennedy_joint'``).  Exposure R² is computed only
            for the ``'kennedy'`` method.
        confounders: Column names of confounders (Kennedy only).
        fit_intercept: Whether the model includes an intercept.
        panel_id: Panel/subject labels for longitudinal data.
            When provided, a ``panel_diagnostics`` sub-dict is added
            with panel count, obs-per-panel statistics, and a
            balanced flag.

    Returns:
        Dictionary with keys for each diagnostic component.
    """
    if confounders is None:
        confounders = []
    result: dict = {}

    # Store the permutation count so downstream display code can
    # report B alongside Monte Carlo SE without a separate lookup.
    result["n_permutations"] = n_permutations

    # ---- Per-predictor diagnostics ----
    # Each of these returns an array of length n_features, one value
    # per predictor.  They are converted to plain Python lists so
    # that the results dict is JSON-serialisable.
    result["standardized_coefs"] = compute_standardized_coefs(
        X,
        y_values,
        model_coefs,
        family,
    ).tolist()

    result["vif"] = compute_vif(X).tolist()

    result["monte_carlo_se"] = compute_monte_carlo_se(
        raw_empirical_p,
        n_permutations,
    ).tolist()

    result["divergence_flags"] = compute_divergence_flags(
        raw_empirical_p,
        raw_classic_p,
        threshold=p_value_threshold,
    )

    # ---- Model-level diagnostics ----
    # Family-specific diagnostics (Breusch-Pagan, deviance residuals,
    # Poisson/NB GoF, ordinal/multinomial GoF) are computed by the
    # family's ``compute_extended_diagnostics()`` method.  Each
    # implementation wraps its calculations in try/except so that
    # degenerate data returns NaN-filled sentinel dicts rather than
    # crashing the results pipeline.
    result.update(
        family.compute_extended_diagnostics(
            X.values.astype(float), y_values, fit_intercept
        )
    )

    # Influential observations
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                category=SmConvergenceWarning,
            )
            warnings.filterwarnings(
                "ignore",
                category=PerfectSeparationWarning,
            )
            result["cooks_distance"] = compute_cooks_distance(X, y_values, family)
    except Exception as exc:
        logger.debug("Cook's distance diagnostics failed: %s", exc)
        n = len(y_values)
        result["cooks_distance"] = {
            "cooks_d": np.full(n, float("nan")),
            "n_influential": 0,
            "threshold": 4.0 / n if n > 0 else float("nan"),
            "influential_indices": [],
            "warning": f"Diagnostics unavailable: {exc}",
        }

    # Permutation coverage
    result["permutation_coverage"] = compute_permutation_coverage(
        n_samples=len(y_values),
        n_permutations=n_permutations,
    )

    # Exposure R² — Kennedy individual method only, and only when
    # confounders are actually specified.  This tells the user how
    # much of each non-confounder predictor is explained by the
    # confounders.  Values near 1.0 indicate near-collinearity, which
    # makes the permutation null distribution degenerate (inflated
    # p-values).  When there are no confounders, R² is trivially 0
    # for every feature, so the column is suppressed to avoid a
    # misleading wall of 0.0000 values.
    if method == "kennedy" and confounders:
        result["exposure_r_squared"] = compute_exposure_r_squared(
            X,
            confounders,
            fit_intercept,
        )

    # ---- Panel diagnostics (Step 15) ----
    if panel_id is not None:
        _, panel_counts = np.unique(panel_id, return_counts=True)
        n_panels = len(panel_counts)
        result["panel_diagnostics"] = {
            "n_panels": n_panels,
            "obs_per_panel_min": int(panel_counts.min()),
            "obs_per_panel_max": int(panel_counts.max()),
            "obs_per_panel_mean": float(np.mean(panel_counts)),
            "balanced": bool(panel_counts.min() == panel_counts.max()),
        }

    return result

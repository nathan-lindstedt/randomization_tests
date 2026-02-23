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

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning as SmConvergenceWarning,
)
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationWarning,
)

logger = logging.getLogger(__name__)

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


def compute_standardized_coefs(
    X: pd.DataFrame,
    y_values: np.ndarray,
    model_coefs: np.ndarray,
    model_type: str = "linear",
) -> np.ndarray:
    """Compute standardized (beta-weight) coefficients.

    Linear:   β* = β · SD(X_j) / SD(Y)
    Logistic: β* = β · SD(X_j)

    Args:
        X: Feature matrix.
        y_values: Response vector.
        model_coefs: Raw coefficients, shape ``(n_features,)``.
        model_type: Model family name (e.g. ``"linear"``,
            ``"logistic"``).  Controls the standardisation formula.

    Returns:
        Array of standardized coefficients, shape ``(n_features,)``.
    """
    # Sample standard deviations (ddof=1 for Bessel's correction)
    # of each column of X.  Shape: (n_features,).
    sd_x = np.std(X.values, axis=0, ddof=1)

    if model_type == "logistic":
        # Logistic: β* = β · SD(X_j)
        # The coefficient β_j is in log-odds units; multiplying by
        # SD(X_j) gives the log-odds change per one-SD increase in
        # X_j.  There is no natural SD(Y) for a binary outcome, so
        # we omit the denominator.
        result: np.ndarray = model_coefs * sd_x
        return result
    else:
        # Linear: β* = β · SD(X_j) / SD(Y)
        # This is the classic "beta weight" — the expected change in
        # Y (in SD units) per one-SD change in X_j, holding all other
        # predictors constant.  Guard against constant Y (sd_y == 0).
        sd_y = np.std(y_values, ddof=1)
        if sd_y == 0:
            return np.zeros_like(model_coefs)
        result = model_coefs * sd_x / sd_y
        return result


def compute_vif(X: pd.DataFrame) -> np.ndarray:
    """Compute Variance Inflation Factors for each predictor.

    VIF_j = 1 / (1 − R²_j), where R²_j is the R² from regressing
    X_j on all other columns of X.

    Args:
        X: Feature matrix.

    Returns:
        Array of VIFs, shape ``(n_features,)``.
    """
    # Convert to plain numpy for the auxiliary regressions.
    X_np = X.values.astype(float)
    n_features = X_np.shape[1]
    vifs = np.zeros(n_features)

    # For each predictor j, regress X_j on all remaining predictors
    # X_{-j} and record R²_j.  This loop is unavoidable because each
    # auxiliary regression has a different response variable.  The
    # number of features p is typically small (< 20), so the cost is
    # dominated by the permutation loop in the core module.
    for j in range(n_features):
        y_j = X_np[:, j]  # shape: (n,)
        X_others = np.delete(X_np, j, axis=1)  # shape: (n, p-1)

        if X_others.shape[1] == 0:
            # Only one predictor — no collinearity possible.
            vifs[j] = 1.0
            continue

        # The auxiliary regression needs its own intercept because we
        # are regressing one predictor on the others, not on Y.
        X_aug = sm.add_constant(X_others)
        r_squared = sm.OLS(y_j, X_aug).fit().rsquared

        # VIF_j = 1 / (1 − R²_j).  If R²_j = 1 (perfect collinearity),
        # VIF is infinite — the coefficient is not identifiable.
        vifs[j] = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else np.inf

    return vifs


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
    X_aug = sm.add_constant(X)
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
    X_aug = sm.add_constant(X)
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


def compute_cooks_distance(
    X: pd.DataFrame,
    y_values: np.ndarray,
    model_type: str = "linear",
) -> dict:
    """Compute Cook's distance and flag influential observations.

    For linear models, delegates to ``OLSInfluence.cooks_distance``.
    For logistic models, delegates to ``GLMInfluence.cooks_distance``.
    Both use the same statsmodels influence API, which computes:

        D_i = (r*²_i · h_i) / (p · (1 − h_i))

    where r*_i is the internally-studentized Pearson residual
    r_P,i / √(1 − h_i), h_i is the hat-matrix leverage, and p is
    the number of parameters.  This is equivalent to:

        D_i = (r²_P,i · h_i) / (p · (1 − h_i)²)

    Observations with D_i > 4/n are flagged as influential.

    Args:
        X: Feature matrix.
        y_values: Response vector.
        model_type: Model family name (e.g. ``"linear"``,
            ``"logistic"``).  Controls whether OLS or GLM Cook's D
            is computed.

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
    X_aug = sm.add_constant(X)

    # Ensure y_values is a plain numpy array so that arithmetic with
    # statsmodels fitted values (which may be pandas Series) does not
    # trigger index-alignment broadcasts.
    y_values = np.asarray(y_values)

    if model_type == "logistic":
        # --- Logistic: GLM Cook's D via statsmodels ---
        #
        # We delegate to statsmodels GLMInfluence rather than computing
        # the hat matrix and residuals manually.  This avoids subtle
        # pitfalls — e.g., sm.Logit.fittedvalues returns log-odds
        # (the linear predictor η = Xβ), NOT probabilities, so using
        # it as p̂ in the variance function would produce garbage.
        #
        # statsmodels GLMInfluence internally:
        #   1. Computes the hat matrix H = W^½ X (X'WX)⁻¹ X' W^½
        #      where W = diag(p̂_i(1 − p̂_i))
        #   2. Forms studentized Pearson residuals r*_i = r_P,i / √(1 − h_i)
        #   3. Returns D_i = r*²_i · h_i / (k · (1 − h_i))
        #      which equals r²_P,i · h_i / (k · (1 − h_i)²)
        #
        # This matches the standard GLM Cook's D from McCullagh &
        # Nelder (1989, §12.5) and Pregibon (1981).
        sm_model = sm.GLM(
            y_values,
            X_aug,
            family=sm.families.Binomial(),
        ).fit(disp=0)
        influence = sm_model.get_influence()
        cooks_d = np.asarray(influence.cooks_distance[0])
    else:
        # --- Linear: exact OLS Cook's D via statsmodels ---
        #
        # For OLS the closed-form Cook's D is:
        #   D_i = (ê_i² · h_i) / (p · MSE · (1 − h_i)²)
        # statsmodels computes this via the OLSInfluence class.
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
# Aggregate helper
# ------------------------------------------------------------------ #
#
# compute_all_diagnostics() is the single entry point called by
# core.permutation_test_regression().  It orchestrates every per-
# predictor and model-level diagnostic in one pass and returns a
# flat dictionary suitable for inclusion in the results dict.  The
# caller does not need to know which individual functions exist —
# this keeps the public API surface small.


def compute_all_diagnostics(
    X: pd.DataFrame,
    y_values: np.ndarray,
    model_coefs: np.ndarray,
    model_type: str = "linear",
    *,
    raw_empirical_p: np.ndarray,
    raw_classic_p: np.ndarray,
    n_permutations: int,
    p_value_threshold: float = 0.05,
    method: str = "ter_braak",
    confounders: list[str] | None = None,
    fit_intercept: bool = True,
) -> dict:
    """Compute all extended diagnostics in one call.

    Args:
        X: Feature matrix.
        y_values: Response vector.
        model_coefs: Raw coefficients, shape ``(n_features,)``.
        model_type: Model family name (e.g. ``"linear"``,
            ``"logistic"``).  Replaces the former ``is_binary``
            flag to support arbitrary family extensions.
        raw_empirical_p: Numeric empirical p-values.
        raw_classic_p: Numeric classical p-values.
        n_permutations: Number of permutations (B).
        p_value_threshold: Significance level for divergence flags.
        method: Permutation method (``'ter_braak'``, ``'kennedy'``,
            or ``'kennedy_joint'``).  Exposure R² is computed only
            for the ``'kennedy'`` method.
        confounders: Column names of confounders (Kennedy only).
        fit_intercept: Whether the model includes an intercept.

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
        model_type,
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
    # These check global assumptions (exchangeability, influence) and
    # report a single summary per model.  The branch on model_type
    # selects Breusch-Pagan (linear) vs. deviance residuals (logistic).
    #
    # Each block is wrapped in try/except because statsmodels can fail
    # on degenerate data (perfect separation, rank-deficient designs,
    # near-saturated probabilities).  The permutation test itself
    # succeeds in these cases — only the post-hoc diagnostics break.
    # Graceful degradation returns NaN-filled sentinel dicts so the
    # rest of the results pipeline is unaffected.
    if model_type == "logistic":
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
                result["deviance_residuals"] = compute_deviance_residual_diagnostics(
                    X, y_values
                )
        except Exception as exc:
            logger.debug("Deviance residual diagnostics failed: %s", exc)
            result["deviance_residuals"] = {
                "mean": float("nan"),
                "variance": float("nan"),
                "n_extreme": 0,
                "runs_test_z": float("nan"),
                "runs_test_p": float("nan"),
                "warning": f"Diagnostics unavailable: {exc}",
            }
    elif model_type == "poisson":
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
                pois_model = sm.GLM(y_values, X_sm, family=sm.families.Poisson()).fit(
                    disp=0
                )
                pearson_chi2 = float(pois_model.pearson_chi2)
                df_resid = float(pois_model.df_resid)
                dispersion = pearson_chi2 / df_resid if df_resid > 0 else float("nan")
                result["poisson_gof"] = {
                    "deviance": float(pois_model.deviance),
                    "pearson_chi2": pearson_chi2,
                    "dispersion": dispersion,
                    "overdispersed": dispersion > 1.5,
                }
        except Exception as exc:
            logger.debug("Poisson GoF diagnostics failed: %s", exc)
            result["poisson_gof"] = {
                "deviance": float("nan"),
                "pearson_chi2": float("nan"),
                "dispersion": float("nan"),
                "overdispersed": False,
                "warning": f"Diagnostics unavailable: {exc}",
            }
    elif model_type == "negative_binomial":
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
                # Estimate α via MLE, then fit GLM with fixed α.
                nb_mle = sm.NegativeBinomial(y_values, X_sm).fit(disp=0, maxiter=200)
                alpha_hat = float(np.exp(nb_mle.lnalpha))
                nb_model = sm.GLM(
                    y_values,
                    X_sm,
                    family=sm.families.NegativeBinomial(alpha=alpha_hat),
                ).fit(disp=0)
                pearson_chi2 = float(nb_model.pearson_chi2)
                df_resid = float(nb_model.df_resid)
                dispersion = pearson_chi2 / df_resid if df_resid > 0 else float("nan")
                result["nb_gof"] = {
                    "deviance": float(nb_model.deviance),
                    "pearson_chi2": pearson_chi2,
                    "dispersion": dispersion,
                    "alpha": alpha_hat,
                    "overdispersed": dispersion > 1.5,
                }
        except Exception as exc:
            logger.debug("NB GoF diagnostics failed: %s", exc)
            result["nb_gof"] = {
                "deviance": float("nan"),
                "pearson_chi2": float("nan"),
                "dispersion": float("nan"),
                "alpha": float("nan"),
                "warning": f"Diagnostics unavailable: {exc}",
            }
    else:
        try:
            result["breusch_pagan"] = compute_breusch_pagan(X, y_values)
        except Exception as exc:
            logger.debug("Breusch-Pagan diagnostics failed: %s", exc)
            result["breusch_pagan"] = {
                "lm_stat": float("nan"),
                "lm_p_value": float("nan"),
                "f_stat": float("nan"),
                "f_p_value": float("nan"),
                "warning": f"Diagnostics unavailable: {exc}",
            }

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
            result["cooks_distance"] = compute_cooks_distance(X, y_values, model_type)
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

    return result

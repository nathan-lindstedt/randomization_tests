"""Confounder identification via correlation screening and mediation analysis.

Controlling for the right variables is critical in observational studies.
A **confounder** Z is a variable that causally influences both the
predictor X and the outcome Y, creating a spurious association between
X and Y that does not reflect a genuine causal effect.  Failing to
control for Z inflates (or deflates) the estimated effect of X.

A **mediator** M lies on the causal pathway from X to Y (X → M → Y).
Controlling for a mediator removes part of the *real* effect of X —
almost always an analytic mistake unless the research question
specifically asks about direct effects.

Workflow:
    1. :func:`screen_potential_confounders` – find variables correlated
       with both X and Y using Pearson *r*.  Any variable passing dual
       correlation thresholds is a *candidate* confounder.
    2. :func:`mediation_analysis` – apply the Preacher & Hayes (2004,
       2008) bootstrap test of the indirect effect to each candidate.
       If the indirect effect's BCa confidence interval excludes zero,
       the candidate is reclassified as a mediator.
    3. :func:`identify_confounders` – orchestrates Steps 1–2, returning
       separate lists of confounders and mediators along with a
       plain-language recommendation for the Kennedy method.

References:
    Baron, R. M. & Kenny, D. A. (1986). The moderator–mediator
    variable distinction. *Journal of Personality and Social
    Psychology*, 51(6), 1173–1182.

    Preacher, K. J. & Hayes, A. F. (2004). SPSS and SAS procedures for
    estimating indirect effects in simple mediation models.  *Behavior
    Research Methods*, 36(4), 717–731.

    Preacher, K. J. & Hayes, A. F. (2008). Asymptotic and resampling
    strategies for assessing and comparing indirect effects in multiple
    mediator models.  *Behavior Research Methods*, 40(3), 879–891.

    Efron, B. (1987). Better bootstrap confidence intervals.  *Journal
    of the American Statistical Association*, 82(397), 171–185.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from ._compat import DataFrameLike, _ensure_pandas_df

# ------------------------------------------------------------------ #
# Step 1 – Correlation screening (Pearson r)
# ------------------------------------------------------------------ #
#
# The Pearson correlation coefficient measures the linear association
# between two variables:
#
#   r(Z, X) = Σ_i (z_i - z̄)(x_i - x̄)
#             ─────────────────────────────────────
#             √[Σ_i (z_i - z̄)²] √[Σ_i (x_i - x̄)²]
#
# r ranges from -1 (perfect negative linear relationship) to +1
# (perfect positive linear relationship).  A candidate confounder Z
# must be linearly associated with BOTH the predictor X and the
# outcome Y beyond minimum thresholds of |r| and p-value.
#
# This is a necessary condition for confounding (Z must be associated
# with both X and Y), but not sufficient — correlation alone cannot
# distinguish a confounder (X ← Z → Y) from a mediator (X → Z → Y).
# Step 2 (mediation analysis) resolves this ambiguity.

def screen_potential_confounders(
    X: "DataFrameLike",
    y: "DataFrameLike",
    predictor: str,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
) -> dict:
    """Screen for variables correlated with both predictor and outcome.

    A potential confounder *Z* satisfies ``|r(Z, X)| >= threshold``
    **and** ``|r(Z, Y)| >= threshold``, both with ``p < p_value_threshold``.

    Args:
        X: Feature matrix.  Accepts pandas or Polars DataFrames.
        y: Target variable.  Accepts pandas or Polars DataFrames.
        predictor: Name of the predictor of interest.
        correlation_threshold: Minimum absolute Pearson *r* to flag a
            variable.
        p_value_threshold: Maximum p-value for significance.

    Returns:
        Dictionary with keys ``predictor``, ``potential_confounders``,
        ``correlations_with_predictor``, ``correlations_with_outcome``,
        and ``excluded_variables``.
    """
    X = _ensure_pandas_df(X, name="X")
    y = _ensure_pandas_df(y, name="y")

    y_values = np.ravel(y)
    other_features = [c for c in X.columns if c != predictor]
    predictor_values = X[predictor].values

    potential_confounders: list[str] = []
    correlations_with_predictor: dict = {}
    correlations_with_outcome: dict = {}
    excluded_variables: list[str] = []

    for feature in other_features:
        fv = X[feature].values

        # Compute Pearson r and its two-sided p-value (t-test on r)
        # for Z vs. X (predictor) and Z vs. Y (outcome).
        #
        # The p-value tests H0: ρ = 0 using t = r√(n-2) / √(1-r²),
        # which follows a t-distribution with n-2 degrees of freedom
        # under the null of zero population correlation.
        corr_pred, p_pred = stats.pearsonr(fv, predictor_values)
        corr_out, p_out = stats.pearsonr(fv, y_values)

        # Dual-threshold criterion: Z is flagged as a potential
        # confounder only if it is significantly correlated with
        # BOTH the predictor AND the outcome.  The magnitude
        # threshold (default |r| >= 0.1) prevents flagging trivially
        # small associations in large samples where everything can
        # be statistically significant.
        sig_pred = (abs(corr_pred) >= correlation_threshold) and (p_pred < p_value_threshold)
        sig_out = (abs(corr_out) >= correlation_threshold) and (p_out < p_value_threshold)

        if sig_pred and sig_out:
            potential_confounders.append(feature)
            correlations_with_predictor[feature] = {"r": corr_pred, "p": p_pred}
            correlations_with_outcome[feature] = {"r": corr_out, "p": p_out}
        else:
            excluded_variables.append(feature)

    return {
        "predictor": predictor,
        "potential_confounders": potential_confounders,
        "correlations_with_predictor": correlations_with_predictor,
        "correlations_with_outcome": correlations_with_outcome,
        "excluded_variables": excluded_variables,
    }


# ------------------------------------------------------------------ #
# Step 2 – Mediation analysis (Preacher & Hayes bootstrap)
# ------------------------------------------------------------------ #
#
# Mediation analysis decomposes the relationship between a predictor X
# and an outcome Y through a candidate mediator M into three pathways:
#
#   Total effect (c path):    Y = c·X + e₁
#     The unconditional effect of X on Y ignoring M.
#
#   a path:                   M = a·X + e₂
#     The effect of X on the candidate mediator.
#
#   b path + direct (c′):    Y = c′·X + b·M + e₃
#     The direct effect of X on Y (c′) after controlling for M, and
#     the effect of M on Y (b) after controlling for X.
#
# The **indirect effect** is the product a·b — the portion of X's
# influence on Y that is transmitted through M.  The total effect
# decomposes as:
#
#   c = c′ + a·b
#       ↑     ↑
#     direct  indirect
#
# Baron & Kenny (1986) proposed a four-step "causal steps" approach
# that required the total effect c to be significant as a prerequisite.
# Preacher & Hayes (2004, 2008) showed this is unnecessary — the
# bootstrap confidence interval of the indirect effect (a·b) is the
# sole criterion.  A significant total effect is neither necessary
# (inconsistent mediation can suppress c) nor sufficient for mediation.
#
# The bootstrap works as follows:
#   1. Resample the data with replacement B times (typically B >= 5000).
#   2. For each resample, compute a* and b* via OLS and form a*·b*.
#   3. Build the empirical distribution of the B indirect effects.
#   4. Construct a confidence interval.  If it excludes zero, the
#      indirect effect is statistically significant → mediation.
#
# We use bias-corrected and accelerated (BCa) intervals (Efron, 1987)
# rather than simple percentile intervals.  BCa adjusts for two
# problems:
#   • **Bias** – the bootstrap distribution's median may not equal the
#     point estimate (measured by z₀, the normal quantile of the
#     proportion of bootstrap values below the observed statistic).
#   • **Skewness** – the bootstrap distribution may be asymmetric
#     (measured by the acceleration â, estimated via jackknife).
#
# Together, z₀ and â shift the percentile cutoffs so the resulting
# interval has better coverage properties than unadjusted percentiles.

def mediation_analysis(
    X: "DataFrameLike",
    y: "DataFrameLike",
    predictor: str,
    mediator: str,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    precision: int = 4,
    random_state: int | None = None,
) -> dict:
    """Preacher & Hayes (2004, 2008) bootstrap test of the indirect effect.

    Decomposes the total effect of *predictor* on *y* into a direct
    effect (c′) and an indirect effect through *mediator* (a × b).
    The indirect effect is tested via bias-corrected and accelerated
    (BCa) bootstrap confidence intervals, which correct for both
    median bias and skewness in the bootstrap distribution (Efron,
    1987).  Unlike the Baron & Kenny (1986) causal-steps approach,
    this method does **not** require the total effect to be significant
    as a prerequisite — the bootstrap CI of the indirect effect is the
    sole criterion for mediation.

    The bootstrap is **vectorised**: all *n_bootstrap* index arrays are
    pre-generated and the a-path / b-path regressions are batched via
    :func:`numpy.linalg.lstsq`.

    Args:
        X: Feature matrix.  Accepts pandas or Polars DataFrames.
        y: Target variable.  Accepts pandas or Polars DataFrames.
        predictor: Predictor (X in X → M → Y).
        mediator: Potential mediator (M).
        n_bootstrap: Number of bootstrap samples.  Preacher & Hayes
            recommend ≥ 5 000 for BCa intervals.
        confidence_level: Confidence-interval level.
        precision: Decimal places for rounding.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary containing the mediation decomposition, BCa
        bootstrap CI, and a textual interpretation.

    References:
        Preacher, K. J. & Hayes, A. F. (2004). SPSS and SAS procedures
        for estimating indirect effects in simple mediation models.
        *Behavior Research Methods*, 36(4), 717–731.

        Preacher, K. J. & Hayes, A. F. (2008). Asymptotic and
        resampling strategies for assessing and comparing indirect
        effects in multiple mediator models. *Behavior Research
        Methods*, 40(3), 879–891.

        Efron, B. (1987). Better bootstrap confidence intervals.
        *Journal of the American Statistical Association*, 82(397),
        171–185.
    """
    X = _ensure_pandas_df(X, name="X")
    y = _ensure_pandas_df(y, name="y")

    y_values = np.ravel(y)
    x_vals = X[predictor].values.reshape(-1, 1)
    m_vals = X[mediator].values.reshape(-1, 1)
    n = len(y_values)

    # --- Observed paths (point estimates from the full sample) ---

    # Step 1: Total effect (c path) — regress Y on X alone.
    # This captures X's entire influence on Y, both direct and
    # any portion routed through M.
    #   Y = c·X + e₁
    model_total = LinearRegression().fit(x_vals, y_values)
    c_total = model_total.coef_[0]

    # Step 2: a path — regress M on X.
    # Measures how much a one-unit change in X shifts M.
    #   M = a·X + e₂
    model_a = LinearRegression().fit(x_vals, m_vals.ravel())
    a_path = model_a.coef_[0]

    # Step 3: b path + direct effect (c′) — regress Y on X and M jointly.
    # After partialling out M, the coefficient on X is the direct
    # effect c′.  The coefficient on M is the b path: the effect of M
    # on Y holding X constant.
    #   Y = c′·X + b·M + e₃
    xm = np.hstack([x_vals, m_vals])
    model_full = LinearRegression().fit(xm, y_values)
    c_prime = model_full.coef_[0]   # direct effect
    b_path = model_full.coef_[1]    # M → Y controlling for X

    # The indirect effect: a·b
    # This is the product of X→M and M→Y|X, representing the portion
    # of X's influence on Y that is channelled through M.
    indirect_effect = a_path * b_path

    # --- Vectorised bootstrap for the indirect effect (a*b) ---
    #
    # For each of B bootstrap iterations we:
    #   1. Resample n observations WITH replacement.
    #   2. Recompute the a path (M on X) and b path (Y on X+M).
    #   3. Record the product a*·b* as one draw from the bootstrap
    #      distribution of the indirect effect.
    #
    # Pre-generating all B index arrays up front allows us to avoid
    # Python-level RNG calls inside the loop.  The regressions
    # themselves are computed via np.linalg.lstsq (the normal equation),
    # which is faster than instantiating sklearn objects B times.
    rng = np.random.default_rng(random_state)
    boot_idx = rng.choice(n, size=(n_bootstrap, n), replace=True)

    # Build the design matrix with an explicit intercept column for lstsq.
    # lstsq solves  min‖Xβ - y‖²  directly, returning β = (X'X)⁻¹X'y
    # (or the Moore-Penrose pseudoinverse for rank-deficient X).
    ones = np.ones((n, 1))
    X_a_design = np.hstack([ones, x_vals])             # (n, 2)

    bootstrap_indirect = np.empty(n_bootstrap)

    for b_i in range(n_bootstrap):
        idx = boot_idx[b_i]
        x_b = X_a_design[idx]       # (n, 2) — resampled [1, X]
        m_b = m_vals[idx].ravel()    # (n,)   — resampled M
        y_b = y_values[idx]          # (n,)   — resampled Y

        # a path: M* = a₀ + a*·X* + ε
        # a_coef[0] is the intercept, a_coef[1] is the slope a*
        a_coef, _, _, _ = np.linalg.lstsq(x_b, m_b, rcond=None)
        a_boot = a_coef[1]  # slope (index 0 is intercept)

        # b path: Y* = b₀ + c′*·X* + b*·M* + ε
        # xm_b columns: [1, X*, M*] → b_coef = [intercept, c′*, b*]
        xm_b = np.hstack([x_b, m_vals[idx]])  # (n, 3)
        b_coef, _, _, _ = np.linalg.lstsq(xm_b, y_b, rcond=None)
        b_boot = b_coef[2]  # mediator slope

        bootstrap_indirect[b_i] = a_boot * b_boot

    # --- BCa confidence interval (Efron, 1987) ---
    # The BCa interval adjusts the percentile endpoints of the bootstrap
    # distribution to correct for bias and skewness, yielding better
    # coverage than simple percentile or normal-approximation intervals.
    # See the _bca_ci helper for the mathematical details.
    ci_lower, ci_upper = _bca_ci(
        bootstrap_indirect, indirect_effect, n, boot_idx,
        x_vals, m_vals, y_values, confidence_level,
    )

    # Decision criterion (Preacher & Hayes):
    # If the BCa CI for the indirect effect (a·b) does NOT contain
    # zero, the indirect effect is statistically significant at the
    # chosen confidence level → the candidate is a mediator.
    # Unlike Baron & Kenny's causal-steps approach, we do NOT require
    # the total effect (c) to be significant.
    is_mediator = (ci_lower > 0) or (ci_upper < 0)

    if abs(c_total) > 1e-10:
        proportion_mediated = indirect_effect / c_total
    else:
        proportion_mediated = np.nan

    if is_mediator:
        if abs(c_prime) < abs(c_total) * 0.1:
            interpretation = (
                f"'{mediator}' fully mediates the effect of '{predictor}' "
                f"on the outcome (BCa CI excludes zero; direct effect "
                f"< 10% of total). Do not control for it as a confounder."
            )
        else:
            interpretation = (
                f"'{mediator}' partially mediates the effect of "
                f"'{predictor}' on the outcome (BCa CI excludes zero). "
                f"Consider whether to control for it based on the "
                f"research question."
            )
    else:
        interpretation = (
            f"'{mediator}' is not a significant mediator (BCa CI "
            f"includes zero). It may be a confounder if correlated "
            f"with both '{predictor}' and the outcome. Consider "
            f"controlling for it."
        )

    return {
        "predictor": predictor,
        "mediator": mediator,
        "total_effect": np.round(c_total, precision),
        "direct_effect": np.round(c_prime, precision),
        "indirect_effect": np.round(indirect_effect, precision),
        "a_path": np.round(a_path, precision),
        "b_path": np.round(b_path, precision),
        "indirect_effect_ci": (np.round(ci_lower, precision), np.round(ci_upper, precision)),
        "ci_method": "BCa",
        "proportion_mediated": (
            np.round(proportion_mediated, precision)
            if not np.isnan(proportion_mediated)
            else np.nan
        ),
        "is_mediator": is_mediator,
        "interpretation": interpretation,
    }


def _bca_ci(
    bootstrap_dist: np.ndarray,
    observed_stat: float,
    n: int,
    boot_idx: np.ndarray,
    x_vals: np.ndarray,
    m_vals: np.ndarray,
    y_values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap CI.

    Args:
        bootstrap_dist: Array of bootstrap indirect-effect replicates.
        observed_stat: Observed indirect effect (a × b).
        n: Sample size.
        boot_idx: Pre-generated bootstrap indices, shape
            ``(n_bootstrap, n)``.
        x_vals: Predictor values, shape ``(n, 1)``.
        m_vals: Mediator values, shape ``(n, 1)``.
        y_values: Outcome values, shape ``(n,)``.
        confidence_level: Nominal coverage (e.g. 0.95).

    Returns:
        ``(ci_lower, ci_upper)`` tuple.
    """
    alpha = 1 - confidence_level

    # --- Bias correction constant (z₀) ---
    # z₀ measures the median bias of the bootstrap distribution relative
    # to the observed statistic.  It is defined as:
    #   z₀ = Φ⁻¹(#{θ̂* < θ̂} / B)
    # where Φ⁻¹ is the standard normal quantile function, θ̂ is the
    # observed indirect effect, and θ̂* are the bootstrap replicates.
    # If the bootstrap distribution is centred on θ̂, exactly half the
    # replicates fall below, giving z₀ = 0 (no bias).  A positive z₀
    # means the bootstrap distribution is shifted left relative to θ̂.
    prop_less = np.mean(bootstrap_dist < observed_stat)
    # Clip to avoid ±inf from ppf at exactly 0 or 1
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_less)

    # --- Acceleration constant (â) via jackknife ---
    # â captures the rate at which the standard error of θ̂ changes
    # with the true parameter value (i.e., the skewness of the
    # influence function).  It is estimated from jackknife replicates:
    #
    #   θ̂₍₋ᵢ₎ = indirect effect with observation i removed
    #   θ̄     = mean of all n jackknife replicates
    #   â      = Σᵢ (θ̄ - θ̂₍₋ᵢ₎)³ / [6 · (Σᵢ (θ̄ - θ̂₍₋ᵢ₎)²)^{3/2}]
    #
    # When the bootstrap distribution is symmetric, â ≈ 0 and BCa
    # reduces to ordinary bias-corrected (BC) intervals.
    ones = np.ones((n, 1))
    X_a_full = np.hstack([ones, x_vals])  # (n, 2)
    jackknife_indirect = np.empty(n)

    for i in range(n):
        # Leave-one-out: fit a- and b-path regressions on n-1 observations
        idx = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        x_j = X_a_full[idx]
        m_j = m_vals[idx].ravel()
        y_j = y_values[idx]

        a_coef, _, _, _ = np.linalg.lstsq(x_j, m_j, rcond=None)
        a_jack = a_coef[1]

        xm_j = np.hstack([x_j, m_vals[idx]])
        b_coef, _, _, _ = np.linalg.lstsq(xm_j, y_j, rcond=None)
        b_jack = b_coef[2]

        jackknife_indirect[i] = a_jack * b_jack

    theta_bar = np.mean(jackknife_indirect)
    diffs = theta_bar - jackknife_indirect
    # The 1e-10 in the denominator prevents division by zero when all
    # jackknife estimates are identical (perfectly symmetric case).
    a_hat = np.sum(diffs ** 3) / (6.0 * (np.sum(diffs ** 2)) ** 1.5 + 1e-10)

    # --- Adjusted percentiles ---
    # The BCa interval replaces the naïve (α/2, 1-α/2) percentiles
    # with adjusted percentiles that account for z₀ (bias) and â
    # (acceleration):
    #
    #   α₁ = Φ( z₀ + (z₀ + z_{α/2}) / (1 - â·(z₀ + z_{α/2})) )
    #   α₂ = Φ( z₀ + (z₀ + z_{1-α/2}) / (1 - â·(z₀ + z_{1-α/2})) )
    #
    # The CI endpoints are then the α₁-th and α₂-th percentiles of the
    # bootstrap distribution.  When z₀=0 and â=0, this reduces to the
    # simple percentile interval (α/2, 1-α/2).
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a_hat * (z0 + z_alpha_lower)))
    p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a_hat * (z0 + z_alpha_upper)))

    # Clip adjusted percentiles to valid range [0.5/B, 1-0.5/B] so that
    # np.percentile does not receive out-of-bounds values.  This can
    # happen when z₀ or â are large (e.g. very biased or skewed
    # bootstrap distributions).
    p_lower = np.clip(p_lower, 0.5 / len(bootstrap_dist), 1 - 0.5 / len(bootstrap_dist))
    p_upper = np.clip(p_upper, 0.5 / len(bootstrap_dist), 1 - 0.5 / len(bootstrap_dist))

    # Extract the corrected percentiles from the bootstrap distribution.
    # These are the BCa CI endpoints.
    ci_lower = float(np.percentile(bootstrap_dist, p_lower * 100))
    ci_upper = float(np.percentile(bootstrap_dist, p_upper * 100))

    return ci_lower, ci_upper


# ------------------------------------------------------------------ #
# Step 3 – Orchestrator
# ------------------------------------------------------------------ #
#
# The full workflow ties Steps 1 and 2 together:
#   1. Screen all features (except the predictor of interest) for dual
#      correlation with both X and Y.  Candidates that pass are
#      potential confounders.
#   2. For each candidate, run the Preacher & Hayes bootstrap mediation
#      test.  If the BCa CI for the indirect effect excludes zero, the
#      candidate is reclassified as a mediator.
#   3. Remaining candidates (not mediators) are reported as confounders
#      that should be controlled for in the Kennedy method.
#
# This two-stage pipeline prevents the common mistake of controlling
# for mediators (which removes part of the real causal effect) while
# still identifying genuine confounders (which introduce bias if
# uncontrolled).

def identify_confounders(
    X: "DataFrameLike",
    y: "DataFrameLike",
    predictor: str,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> dict:
    """Two-step confounder identification.

    1. Screen for variables correlated with both predictor and outcome.
    2. Use mediation analysis to filter out mediators.

    Variables that pass screening but are **not** identified as
    mediators are likely confounders.

    Args:
        X: Feature matrix.  Accepts pandas or Polars DataFrames.
        y: Target variable.  Accepts pandas or Polars DataFrames.
        predictor: Predictor of interest.
        correlation_threshold: Minimum absolute Pearson *r*.
        p_value_threshold: Significance cutoff.
        n_bootstrap: Bootstrap iterations for mediation analysis.
        confidence_level: Confidence-interval level.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with keys ``identified_confounders``,
        ``identified_mediators``, ``screening_results``,
        and ``mediation_results``.
    """
    X = _ensure_pandas_df(X, name="X")
    y = _ensure_pandas_df(y, name="y")

    screening = screen_potential_confounders(
        X, y, predictor,
        correlation_threshold=correlation_threshold,
        p_value_threshold=p_value_threshold,
    )
    candidates = screening["potential_confounders"]

    identified_confounders: list[str] = []
    identified_mediators: list[str] = []
    mediation_results: dict = {}

    for candidate in candidates:
        med = mediation_analysis(
            X, y, predictor, candidate,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        )
        mediation_results[candidate] = med
        if med["is_mediator"]:
            identified_mediators.append(candidate)
        else:
            identified_confounders.append(candidate)

    return {
        "predictor": predictor,
        "identified_confounders": identified_confounders,
        "identified_mediators": identified_mediators,
        "screening_results": screening,
        "mediation_results": mediation_results,
    }

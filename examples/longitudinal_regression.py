# %% [markdown]
"""
Example: Longitudinal Linear Regression with Autoregressive AR(p) Correction
Dataset: Bike Sharing (UCI Machine Learning Repository ID=275)

Dataset Context & Theoretical Background:
    The Bike Sharing dataset (Fanaee-T & Gama, 2014) contains 17,379 hourly
    records of bicycle rental demand in the Capital Bikeshare system across
    Washington, D.C. during 2011-2012, matched with meteorological and seasonal
    environmental measurements.

    Longitudinal panel data structures are characterized by repeated observations
    over time nested within discrete cross-sectional units. In this dataset,
    each calendar day functions as a panel unit (Level 2), with 24 hourly time
    points nested within each day (Level 1). Consecutive hourly observations
    within a day exhibit strong temporal inertia and autocorrelation: an
    unexpected spike in bike usage at hour t persistently echoes into hour t+1.

    Standard permutation tests assume that residuals are exchangeable under the
    null hypothesis. When residuals follow an autoregressive AR(p) process:

        eps_{i,t} = phi_1 * eps_{i,t-1} + ... + phi_p * eps_{i,t-p} + u_{i,t}

    permuting residuals across time shuffles apart the temporal lag structure,
    destroying positive covariance and yielding a permuted reference distribution
    with spuriously small dispersion. This produces severe anticonservatism and
    inflates false-positive rates (Type I error).

Methodological Rationale for Score-Based AR(p) Correction:
    1. AR Estimation with Panel Decontamination:
       Within-panel Frisch-Waugh-Lovell demeaning strips unobserved daily random
       intercepts before estimating pooled autoregressive parameters phi,
       preventing cluster random-intercept variance from contaminating AR lag
       coefficients. Nickell (1981) small-T bias adjustments are applied.
    2. Precision Matrix Factoring:
       The panel AR(p) process induces a block-diagonal covariance matrix
       Omega = diag(Omega_1, ..., Omega_N). Its exact inverse Omega^{-1} is
       constructed via stationary Yule-Walker equations.
    3. Generalized Score Projection:
       The score vector U_j = X_j^T Omega^{-1} (Y - X_{-j} hat{gamma}) and its
       Fisher information metric fold the temporal autocorrelation structure
       directly into the score test statistic, ensuring that permutation inference
       remains asymptotically exact and robust to serial correlation.

Features Selected for Demonstration:
    - temp: Normalized ambient temperature (deg C). Exhibits true within-day
      exogenous variation (cooler mornings, warmer afternoons) driving bike
      demand; remains statistically significant across all AR orders.
    - weathersit: Categorical weather severity scale (1: Clear, 2: Mist/Cloudy,
      3: Light Rain/Snow, 4: Heavy Rain). Appears highly significant in naive
      OLS/permutation, but collapses to non-significance under AR correction
      because weather persistence mirrors the temporal momentum of the day
      (a classic autocorrelation artifact).
    - workingday: Binary workday indicator (1: Workday, 0: Weekend/Holiday).
      Constant within each day (zero within-panel variation); serves as a null
      baseline that remains invariant across AR orders.

Demonstrates:
    - panel_id= and time_id= longitudinal panel specification
    - Sequential AR order progression: Unadjusted -> AR(1) -> AR(2) -> AR(3)
    - Diagnostic serial correlation validation: Durbin-Watson and Ljung-Box tests
    - print_comparison_table() multi-order comparative summary
    - Execution and protocol artifacts inspection via print_protocol_usage_table

References:
    - Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble
      detectors and background knowledge. Progress in Artificial Intelligence,
      2(2-3), 113-127.
    - Nickell, S. (1981). Biases in dynamic models with fixed effects.
      Econometrica, 49(6), 1417-1426.
    - Durbin, J., & Watson, G. S. (1950). Testing for serial correlation in
      least squares regression. Biometrika, 37(3/4), 409-428.
    - Ljung, G. M., & Box, G. E. (1978). On a measure of lack of fit in time
      series models. Biometrika, 65(2), 297-303.
"""

# %%
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    print_comparison_table,
    print_dataset_info_table,
    print_diagnostics_table,
    print_protocol_usage_table,
    print_results_table,
    randomization_test_regression,
)

# %%
# ============================================================================
# Load data
# ============================================================================

ds = fetch_ucirepo(id=275)
X_all = ds.data.features.copy()
y_all = ds.data.targets.copy()

# ============================================================================
# Construct panel structure: days as panels, hours as time
# ============================================================================

# Use dteday (date string) as panel identifier, hr as time index.
dates = X_all["dteday"].values
hours = X_all["hr"].values

# Keep only complete days (24 observations each) for balanced panels.
day_counts = pd.Series(dates).groupby(dates).transform("count")
complete_mask = day_counts == 24
X_complete = X_all.loc[complete_mask].copy()
y_complete = y_all.loc[complete_mask].copy()
dates_complete = X_complete["dteday"].values
hours_complete = X_complete["hr"].values

n_complete_days = len(np.unique(dates_complete))

# Subsample to 100 days for tractable computation.
unique_days = np.unique(dates_complete)
rng = np.random.default_rng(42)
sampled_days = rng.choice(unique_days, size=min(100, len(unique_days)), replace=False)
sampled_days.sort()
sample_mask = np.isin(dates_complete, sampled_days)

X_sample = X_complete.loc[sample_mask].reset_index(drop=True)
y_sample = y_complete.loc[sample_mask].reset_index(drop=True)

# Build integer panel and time identifiers.
day_labels = X_sample["dteday"].values
_, panel_id = np.unique(day_labels, return_inverse=True)
time_id = X_sample["hr"].values

# Select features that illustrate different AR correction behaviors.
# - temp: genuine within-day variation → survives AR correction
# - weathersit: slowly varying within day → autocorrelation artifact
# - workingday: constant within day → null baseline (never significant)
feature_cols = ["workingday", "weathersit", "temp"]
X = X_sample[feature_cols].copy()
y = y_sample[["cnt"]].copy()

n_obs = len(y)
n_panels = len(np.unique(panel_id))
n_features = len(feature_cols)

print_dataset_info_table(
    name="Bike Sharing (hourly, 100-day sample)",
    X=X,
    y=y,
    target_description="hourly bike rental count",
    extra_stats={
        "Panels (days)": str(n_panels),
        "Time points (hours)": "24",
    },
)

# %%
# ============================================================================
# Score test WITHOUT AR correction
# ============================================================================

results_no_ar = randomization_test_regression(
    X,
    y,
    method="score",
    family="linear",
    panel_id=panel_id,
    time_id=time_id,
    n_randomizations=2_000,
    random_state=42,
)
print_results_table(
    results_no_ar,
    title="Longitudinal Linear Regression — Score Test (No AR)",
)
print_diagnostics_table(results_no_ar)

# %%
# ============================================================================
# Score test WITH AR(1) correction
# ============================================================================

results_ar1 = randomization_test_regression(
    X,
    y,
    method="score",
    family="linear",
    panel_id=panel_id,
    time_id=time_id,
    ar_order=1,
    n_randomizations=2_000,
    random_state=42,
)
print_results_table(
    results_ar1,
    title="Longitudinal Linear Regression — Score Test (AR(1))",
)
print_diagnostics_table(results_ar1)

# %%
# ============================================================================
# Score test WITH AR(2) correction
# ============================================================================

results_ar2 = randomization_test_regression(
    X,
    y,
    method="score",
    family="linear",
    panel_id=panel_id,
    time_id=time_id,
    ar_order=2,
    n_randomizations=2_000,
    random_state=42,
)
print_results_table(
    results_ar2,
    title="Longitudinal Linear Regression — Score Test (AR(2))",
)
print_diagnostics_table(results_ar2)

# %%
# ============================================================================
# Score test WITH AR(3) correction
# ============================================================================

results_ar3 = randomization_test_regression(
    X,
    y,
    method="score",
    family="linear",
    panel_id=panel_id,
    time_id=time_id,
    ar_order=3,
    n_randomizations=2_000,
    random_state=42,
)
print_results_table(
    results_ar3,
    title="Longitudinal Linear Regression — Score Test (AR(3))",
)
print_diagnostics_table(results_ar3)

# %%
# ============================================================================
# P-value comparison across AR orders
# ============================================================================

print_comparison_table(
    [
        ("No AR", results_no_ar),
        ("AR(1)", results_ar1),
        ("AR(2)", results_ar2),
        ("AR(3)", results_ar3),
    ],
    title="P-Value Comparison: No AR vs. AR(3)",
)

# %%
# ============================================================================
# Execution & Protocol Artifacts
# ============================================================================
# Inspect the internal execution context from the completed test result:
# backend acceleration, batch convergence, fit metrics, and protocol properties.

print_protocol_usage_table(results_ar3)

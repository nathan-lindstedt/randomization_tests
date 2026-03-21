"""
Test Case 10: Longitudinal Linear Regression with AR Correction
Bike Sharing dataset (UCI ML Repository ID=275)

Demonstrates:
- ``ar_order=1`` and ``ar_order=2`` — score-based AR(p) correction
- ``panel_id=`` / ``time_id=`` — longitudinal data specification
- ``method="score"`` — required for AR correction
- Before/after Durbin–Watson and Ljung–Box diagnostics
- Comparison of results without AR, AR(1), and AR(2) correction
- How autocorrelation inflates significance for some features but not
  others, depending on a feature's within-day variation structure

Dataset
-------
17,379 hourly records of bike-sharing rental counts in Washington, D.C.
over 2011–2012, paired with weather and seasonal information.  The
natural panel structure treats each **calendar day** as a panel (unit)
and each **hour** as a within-panel time point:

    Level 2: Days (up to 731 unique dates)
    Level 1: Hourly observations within each day (24 per complete day)

Hourly bike demand within a day exhibits strong temporal autocorrelation
— consecutive hours have correlated residuals that violate the
independence assumption of standard permutation tests.  The ``ar_order``
correction estimates pooled within-day AR coefficients and folds the
working-correlation precision matrix into the score projection, yielding
properly calibrated inference.

Feature selection rationale
---------------------------
Three features are chosen to illustrate different behaviors under AR
correction:

- **temp** (normalized temperature): Has genuine within-day causal
  variation — cool mornings, warm afternoons — that drives hourly
  bike demand independently of temporal momentum.  Remains
  significant under AR(1) and AR(2) correction.

- **weathersit** (weather situation: 1–4 scale): Appears significant
  in the baseline model but collapses under AR correction.  Weather
  conditions are slowly varying within a day and correlated with the
  autocorrelated demand pattern.  Once temporal momentum is removed,
  the apparent association disappears — a classic autocorrelation
  artifact.

- **workingday** (binary: workday vs. weekend/holiday): Constant
  within each day (zero within-panel variation), so all variation
  is between-panel.  With only 100 days, there is not enough
  between-panel contrast for significance.  A null baseline that
  is never significant regardless of AR correction.

Reference
---------
Fanaee-T, H. & Gama, J. (2014). Event labeling combining ensemble
detectors and background knowledge. *Progress in Artificial
Intelligence*, 2(2–3), 113–127.
"""

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    print_ar_comparison_table,
    print_dataset_info_table,
    print_diagnostics_table,
    print_results_table,
    randomization_test_regression,
)

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
print(f"Complete days: {n_complete_days}")

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
    n_observations=n_obs,
    n_features=n_features,
    feature_names=feature_cols,
    target_name="cnt",
    target_description="hourly bike rental count",
    y_range=(float(y.values.min()), float(y.values.max())),
    y_mean=float(y.values.mean()),
    y_var=float(y.values.var()),
    extra_stats={
        "Panels (days)": str(n_panels),
        "Time points (hours)": "24",
    },
)

# ============================================================================
# Score test WITHOUT AR correction
# ============================================================================

print("\n" + "=" * 72)
print("Score test WITHOUT AR correction")
print("=" * 72)

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
    title="Score Permutation Test — No AR Correction",
)

# ============================================================================
# Score test WITH AR(1) correction
# ============================================================================

print("\n" + "=" * 72)
print("Score test WITH AR(1) correction")
print("=" * 72)

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
    title="Score Permutation Test — AR(1) Correction",
)

# ============================================================================
# Score test WITH AR(2) correction
# ============================================================================

print("\n" + "=" * 72)
print("Score test WITH AR(2) correction")
print("=" * 72)

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
    title="Score Permutation Test — AR(2) Correction",
)

# ============================================================================
# Extended diagnostics — each model
# ============================================================================

print_diagnostics_table(
    results_no_ar,
    title="Extended Diagnostics — No AR Correction",
)

print_diagnostics_table(
    results_ar1,
    title="Extended Diagnostics — AR(1) Corrected",
)

print_diagnostics_table(
    results_ar2,
    title="Extended Diagnostics — AR(2) Corrected",
)

# ============================================================================
# P-value comparison across AR orders
# ============================================================================

print_ar_comparison_table(
    [
        ("No AR", results_no_ar),
        ("AR(1)", results_ar1),
        ("AR(2)", results_ar2),
    ],
    title="P-Value Comparison: No AR vs. AR(1) vs. AR(2)",
)

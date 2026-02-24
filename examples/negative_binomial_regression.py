"""
Test Case 4: Negative Binomial Regression (Overdispersed Count Outcome)
Bike Sharing dataset (UCI ML Repository ID=275)

Demonstrates:
- ``family="negative_binomial"`` — explicit family selection
- ``calibrate()`` nuisance-parameter estimation (dispersion α)
- All five permutation methods routed through ``NegativeBinomialFamily``
- Direct ``ModelFamily`` protocol usage including ``calibrate``
- Massive overdispersion: marginal Var/Mean ≈ 174

The target variable *cnt* is the hourly count of total rental bikes.
The count distribution is heavily overdispersed — a Poisson model would
grossly understate standard errors.  NB2 (Var = μ + α·μ²) is the
natural choice.
"""

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    NegativeBinomialFamily,
    identify_confounders,
    permutation_test_regression,
    print_confounder_table,
    print_diagnostics_table,
    print_joint_results_table,
    print_results_table,
)

# ============================================================================
# Load data
# ============================================================================

bike_sharing = fetch_ucirepo(id=275)
X_full = bike_sharing.data.features.copy()
y_full = bike_sharing.data.targets

# Subsample to 300 rows (permutation tests re-fit B × p GLMs, so
# keeping n moderate avoids excessive runtime for an example script).
rng = np.random.default_rng(42)
idx = rng.choice(len(X_full), size=300, replace=False)
X_sub = X_full.iloc[idx].reset_index(drop=True)
y_sub = y_full.iloc[idx].reset_index(drop=True)

# Select predictors: normalised temperature, humidity, wind speed,
# working-day indicator, and weather situation (ordinal 1–4).
X = pd.DataFrame(
    {
        "temp": X_sub["temp"].astype(float),
        "hum": X_sub["hum"].astype(float),
        "windspeed": X_sub["windspeed"].astype(float),
        "workingday": X_sub["workingday"].astype(float),
        "weathersit": X_sub["weathersit"].astype(float),
    }
)
y = y_sub.copy()
feature_names = X.columns.tolist()

y_arr = np.ravel(y).astype(float)
print(f"Dataset:           {bike_sharing.metadata.name}")
print(f"Sample size:       {len(y_arr)}  (subsampled from {len(y_full)})")
print(f"Predictors:        {feature_names}")
print(f"Mean(y):           {y_arr.mean():.2f}")
print(f"Var(y):            {y_arr.var():.2f}")
print(f"Var/Mean ratio:    {y_arr.var() / y_arr.mean():.2f}  (>>1 → overdispersed)")
print()

# ============================================================================
# ter Braak (1992) — family="negative_binomial" (explicit)
# ============================================================================

results_ter_braak = permutation_test_regression(
    X, y, method="ter_braak", family="negative_binomial"
)
assert results_ter_braak.family.name == "negative_binomial"
print_results_table(
    results_ter_braak,
    title="ter Braak (1992) Permutation Test (family='negative_binomial')",
)
print_diagnostics_table(
    results_ter_braak,
    title="ter Braak (1992) Diagnostics (family='negative_binomial')",
)

# ============================================================================
# Confounder identification
# ============================================================================
# Identify which features are confounders for each predictor
# (correlated with both the predictor of interest and the outcome).

all_confounder_results = {}
for predictor in X.columns:
    all_confounder_results[predictor] = identify_confounders(X, y, predictor=predictor)

print_confounder_table(
    all_confounder_results,
    title="Confounder Identification for All Predictors (Negative Binomial)",
)

predictors_with_confounders = {
    pred: res["identified_confounders"]
    for pred, res in all_confounder_results.items()
    if res["identified_confounders"]
}

# ============================================================================
# Kennedy (1995) individual — family="negative_binomial"
# ============================================================================

if predictors_with_confounders:
    example_predictor = list(predictors_with_confounders.keys())[0]
    example_confounders = predictors_with_confounders[example_predictor]

    results_kennedy = permutation_test_regression(
        X,
        y,
        method="kennedy",
        confounders=example_confounders,
        family="negative_binomial",
    )
    print_results_table(
        results_kennedy,
        title=(
            f"Kennedy (1995) for '{example_predictor}' "
            f"(controlling for {', '.join(example_confounders)}) "
            f"[family='negative_binomial']"
        ),
    )
    print_diagnostics_table(
        results_kennedy,
        title="Kennedy (1995) Individual Diagnostics (family='negative_binomial')",
    )

# ============================================================================
# Kennedy (1995) joint — family="negative_binomial"
# ============================================================================

if predictors_with_confounders:
    results_kennedy_joint = permutation_test_regression(
        X,
        y,
        method="kennedy_joint",
        confounders=example_confounders,
        family="negative_binomial",
    )
    print_joint_results_table(
        results_kennedy_joint,
        title=(
            f"Kennedy (1995) Joint "
            f"(controlling for {', '.join(example_confounders)}) "
            f"[family='negative_binomial']"
        ),
    )

# ============================================================================
# Freedman–Lane (1983) individual — family="negative_binomial"
# ============================================================================

if predictors_with_confounders:
    results_fl = permutation_test_regression(
        X,
        y,
        method="freedman_lane",
        confounders=example_confounders,
        family="negative_binomial",
    )
    print_results_table(
        results_fl,
        title=(
            f"Freedman–Lane (1983) Individual "
            f"(controlling for {', '.join(example_confounders)}) "
            f"[family='negative_binomial']"
        ),
    )
    print_diagnostics_table(
        results_fl,
        title="Freedman–Lane (1983) Individual Diagnostics (family='negative_binomial')",
    )

# ============================================================================
# Freedman–Lane (1983) joint — family="negative_binomial"
# ============================================================================

if predictors_with_confounders:
    results_fl_joint = permutation_test_regression(
        X,
        y,
        method="freedman_lane_joint",
        confounders=example_confounders,
        family="negative_binomial",
    )
    print_joint_results_table(
        results_fl_joint,
        title=(
            f"Freedman–Lane (1983) Joint "
            f"(controlling for {', '.join(example_confounders)}) "
            f"[family='negative_binomial']"
        ),
    )

# ============================================================================
# Direct NegativeBinomialFamily protocol usage
# ============================================================================

n = len(y_arr)
p = X.shape[1]
family = NegativeBinomialFamily()

print(f"\n{'=' * 60}")
print("Direct NegativeBinomialFamily protocol usage")
print(f"{'=' * 60}")
print(f"  name:               {family.name}")
print(f"  residual_type:      {family.residual_type}")
print(f"  direct_permutation: {family.direct_permutation}")
print(f"  metric_label:       {family.metric_label}")

# validate_y
family.validate_y(y_arr)
print("  validate_y:         passed")

# calibrate — estimate α from the observed data
print(f"  alpha (before):     {family.alpha}")
calibrated = family.calibrate(X, y_arr, fit_intercept=True)
assert isinstance(calibrated, NegativeBinomialFamily)
print(f"  alpha (after):      {calibrated.alpha:.4f}")

# Idempotency check
recalibrated = calibrated.calibrate(X, y_arr, fit_intercept=True)
assert recalibrated is calibrated
print("  idempotent:         True (recalibrate returns self)")

# fit / predict / coefs / residuals (using calibrated instance)
model = calibrated.fit(X, y_arr, fit_intercept=True)
preds = calibrated.predict(model, X)
coefs = calibrated.coefs(model)
resids = calibrated.residuals(model, X, y_arr)
print(f"  coefs:              {np.round(coefs, 4)}")
print(f"  mean |residual|:    {np.mean(np.abs(resids)):.4f}")

# fit_metric (NB deviance)
deviance = calibrated.fit_metric(y_arr, preds)
print(f"  NB deviance:        {deviance:.2f}")

# reconstruct_y — NB-sampled reconstruction
rng2 = np.random.default_rng(42)
perm_resids = rng2.permutation(resids)
y_star = calibrated.reconstruct_y(preds, perm_resids, rng2)
print(f"  reconstruct_y:      shape={y_star.shape}, mean={np.mean(y_star):.4f}")

# batch_fit
n_batch = 50
perm_indices = np.array([rng2.permutation(n) for _ in range(n_batch)])
Y_matrix = y_arr[perm_indices]
batch_coefs = calibrated.batch_fit(X, Y_matrix, fit_intercept=True)
print(f"  batch_fit:          shape={batch_coefs.shape} (B={n_batch}, p={p})")

# diagnostics
diag = calibrated.diagnostics(X, y_arr, fit_intercept=True)
print(f"  diagnostics:        deviance={diag['deviance']}, α={diag['alpha']}")

# classical_p_values
p_classical = calibrated.classical_p_values(X, y_arr, fit_intercept=True)
print(f"  classical_p_values: {np.round(p_classical, 6)}")

# exchangeability_cells (v0.4.0 stub)
cells = calibrated.exchangeability_cells(X, y_arr)
assert cells is None
print("  exchangeability:    None (global)")

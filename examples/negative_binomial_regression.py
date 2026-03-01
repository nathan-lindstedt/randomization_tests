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

import warnings

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    NegativeBinomialFamily,
    identify_confounders,
    permutation_test_regression,
    print_confounder_table,
    print_dataset_info_table,
    print_diagnostics_table,
    print_family_info_table,
    print_joint_results_table,
    print_protocol_usage_table,
    print_results_table,
    resolve_family,
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
var_mean_ratio = y_arr.var() / y_arr.mean()

print_dataset_info_table(
    name=bike_sharing.metadata.name,
    n_observations=len(X),
    n_features=X.shape[1],
    feature_names=feature_names,
    target_name=y.columns[0],
    target_description="hourly rental bike count",
    y_range=(int(y_arr.min()), int(y_arr.max())),
    y_mean=float(y_arr.mean()),
    y_var=float(y_arr.var()),
    extra_stats={"Var/Mean": f"{var_mean_ratio:.2f}  (>>1 → overdispersed)"},
)

# ============================================================================
# Family resolution
# ============================================================================

nb_family = resolve_family("negative_binomial")
assert nb_family.name == "negative_binomial"

print_family_info_table(
    explicit_family=nb_family,
)

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
# Kennedy (1995) individual — family="negative_binomial"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_kennedy = permutation_test_regression(
        X, y, method="kennedy", confounders=[], family="negative_binomial"
    )
print_results_table(
    results_kennedy,
    title="Kennedy (1995) Individual Permutation Test (family='negative_binomial')",
)
print_diagnostics_table(
    results_kennedy,
    title="Kennedy (1995) Individual Diagnostics (family='negative_binomial')",
)

# ============================================================================
# Kennedy (1995) joint — family="negative_binomial"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_kennedy_joint = permutation_test_regression(
        X, y, method="kennedy_joint", confounders=[], family="negative_binomial"
    )
print_joint_results_table(
    results_kennedy_joint,
    title="Kennedy (1995) Joint Permutation Test (family='negative_binomial')",
)

# ============================================================================
# Freedman–Lane (1983) individual — family="negative_binomial"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_fl = permutation_test_regression(
        X, y, method="freedman_lane", confounders=[], family="negative_binomial"
    )
print_results_table(
    results_fl,
    title="Freedman–Lane (1983) Individual Permutation Test (family='negative_binomial')",
)
print_diagnostics_table(
    results_fl,
    title="Freedman–Lane (1983) Individual Diagnostics (family='negative_binomial')",
)

# ============================================================================
# Freedman–Lane (1983) joint — family="negative_binomial"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_fl_joint = permutation_test_regression(
        X, y, method="freedman_lane_joint", confounders=[], family="negative_binomial"
    )
print_joint_results_table(
    results_fl_joint,
    title="Freedman–Lane (1983) Joint Permutation Test (family='negative_binomial')",
)

# ============================================================================
# Confounder identification
# ============================================================================

all_confounder_results = {}
for predictor in X.columns:
    all_confounder_results[predictor] = identify_confounders(
        X, y, predictor=predictor, family="negative_binomial"
    )

print_confounder_table(
    all_confounder_results,
    title="Confounder Identification for All Predictors (Negative Binomial)",
)

predictors_with_confounders = {
    pred: res.identified_confounders
    for pred, res in all_confounder_results.items()
    if res.identified_confounders
}

# ============================================================================
# Kennedy with identified confounders — family="negative_binomial"
# ============================================================================

if predictors_with_confounders:
    example_predictor = list(predictors_with_confounders.keys())[0]
    example_confounders = predictors_with_confounders[example_predictor]

    results_kc = permutation_test_regression(
        X,
        y,
        method="kennedy",
        confounders=example_confounders,
        family="negative_binomial",
    )
    print_results_table(
        results_kc,
        title=(
            f"Kennedy (1995) for '{example_predictor}' "
            f"(controlling for {', '.join(example_confounders)}) "
            f"(family='negative_binomial')"
        ),
    )
    print_diagnostics_table(
        results_kc,
        title=(
            f"Kennedy (1995) Diagnostics for '{example_predictor}' "
            f"(family='negative_binomial')"
        ),
    )

# ============================================================================
# Direct NegativeBinomialFamily protocol usage
# ============================================================================
# The ModelFamily protocol encapsulates every model-specific operation —
# fitting, prediction, residual extraction, Y-reconstruction, batch
# fitting, diagnostics, and classical p-values.  Below we exercise
# each method directly for NegativeBinomialFamily.

X_np = X.values.astype(float)
n = len(y_arr)
p = X_np.shape[1]
family = NegativeBinomialFamily()

# validate_y
family.validate_y(y_arr)

# calibrate — estimate α from the observed data
calibrated = family.calibrate(X_np, y_arr, fit_intercept=True)
assert isinstance(calibrated, NegativeBinomialFamily)

# Idempotency check
recalibrated = calibrated.calibrate(X_np, y_arr, fit_intercept=True)
assert recalibrated is calibrated

# fit / predict / coefs / residuals (using calibrated instance)
model = calibrated.fit(X_np, y_arr, fit_intercept=True)
preds = calibrated.predict(model, X_np)
coefs = calibrated.coefs(model)
resids = calibrated.residuals(model, X_np, y_arr)

# fit_metric (NB deviance)
deviance = calibrated.fit_metric(y_arr, preds)

# reconstruct_y — NB-sampled reconstruction
rng2 = np.random.default_rng(42)
perm_resids = rng2.permutation(resids)
y_star = calibrated.reconstruct_y(
    preds[np.newaxis, :], perm_resids[np.newaxis, :], rng2
)

# batch_fit — fit NB GLM on B permuted Y vectors at once
n_batch = 50
perm_indices = np.array([rng2.permutation(n) for _ in range(n_batch)])
Y_matrix = y_arr[perm_indices]
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    batch_coefs = calibrated.batch_fit(X_np, Y_matrix, fit_intercept=True)
n_nan = int(np.sum(np.any(np.isnan(batch_coefs), axis=1)))

# diagnostics
diag = calibrated.diagnostics(X_np, y_arr, fit_intercept=True)
if diag["dispersion"] > 1.5:
    dispersion_status = "⚠ OVERDISPERSION DETECTED — NB2 IS APPROPRIATE"
else:
    dispersion_status = "✓ NO OVERDISPERSION (α handles it)"

# classical_p_values
p_classical = calibrated.classical_p_values(X_np, y_arr, fit_intercept=True)

# exchangeability_cells (v0.4.0 stub)
cells = calibrated.exchangeability_cells(X_np, y_arr)
assert cells is None

print_protocol_usage_table(
    results_ter_braak,
    title="Direct NegativeBinomialFamily Protocol Usage",
)

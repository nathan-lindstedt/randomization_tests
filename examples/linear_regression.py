"""
Test Case 1: Linear Regression (Continuous Outcome)
Real Estate Valuation dataset (UCI ML Repository ID=477)

Demonstrates:
- ``family="linear"`` — explicit family selection
- All five permutation methods routed through ``LinearFamily``
- Direct ``ModelFamily`` protocol usage (fit / predict / residuals /
  reconstruct_y / fit_metric / diagnostics / classical_p_values /
  batch_fit)
"""

import warnings

import numpy as np
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    LinearFamily,
    identify_confounders,
    permutation_test_regression,
    print_confounder_table,
    print_diagnostics_table,
    print_joint_results_table,
    print_results_table,
    resolve_family,
)

# ============================================================================
# Load data
# ============================================================================

real_estate_valuation = fetch_ucirepo(id=477)
X = real_estate_valuation.data.features
y = real_estate_valuation.data.targets

# ============================================================================
# Verify resolve_family auto-detects "linear" for continuous Y
# ============================================================================

auto_family = resolve_family("auto", np.ravel(y))
assert auto_family.name == "linear", f"Expected 'linear', got {auto_family.name!r}"
print(f"resolve_family('auto', y) → {auto_family.name!r}")

# ============================================================================# ter Braak (1992) \u2014 family="auto" (auto-detection)
# ============================================================================

results_ter_braak_auto = permutation_test_regression(
    X, y, method="ter_braak", family="auto"
)
assert results_ter_braak_auto.family.name == "linear"
print_results_table(
    results_ter_braak_auto,
    title="ter Braak (1992) Permutation Test (family='auto' \u2192 linear)",
)

# ============================================================================# ter Braak (1992) — family="linear" (explicit)
# ============================================================================

results_ter_braak = permutation_test_regression(
    X, y, method="ter_braak", family="linear"
)
print_results_table(
    results_ter_braak,
    title="ter Braak (1992) Permutation Test (family='linear')",
)
print_diagnostics_table(
    results_ter_braak,
    title="ter Braak (1992) Extended Diagnostics (family='linear')",
)
assert results_ter_braak.family.name == "linear"

# ============================================================================
# Kennedy (1995) individual — family="linear"
# ============================================================================

results_kennedy = permutation_test_regression(
    X, y, method="kennedy", confounders=[], family="linear"
)
print_results_table(
    results_kennedy,
    title="Kennedy (1995) Individual Permutation Test (family='linear')",
)
print_diagnostics_table(
    results_kennedy,
    title="Kennedy (1995) Individual Diagnostics (family='linear')",
)

# ============================================================================
# Kennedy (1995) joint — family="linear"
# ============================================================================

results_kennedy_joint = permutation_test_regression(
    X, y, method="kennedy_joint", confounders=[], family="linear"
)
print_joint_results_table(
    results_kennedy_joint,
    title="Kennedy (1995) Joint Permutation Test (family='linear')",
)

# ============================================================================
# Freedman–Lane (1983) individual — family="linear"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_fl = permutation_test_regression(
        X, y, method="freedman_lane", confounders=[], family="linear"
    )
print_results_table(
    results_fl,
    title="Freedman–Lane (1983) Individual Permutation Test (family='linear')",
)
print_diagnostics_table(
    results_fl,
    title="Freedman–Lane (1983) Individual Diagnostics (family='linear')",
)

# ============================================================================
# Freedman–Lane (1983) joint — family="linear"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_fl_joint = permutation_test_regression(
        X, y, method="freedman_lane_joint", confounders=[], family="linear"
    )
print_joint_results_table(
    results_fl_joint,
    title="Freedman–Lane (1983) Joint Permutation Test (family='linear')",
)

# ============================================================================
# Confounder identification
# ============================================================================

all_confounder_results = {}
for predictor in X.columns:
    all_confounder_results[predictor] = identify_confounders(X, y, predictor=predictor)

print_confounder_table(
    all_confounder_results,
    title="Confounder Identification for All Predictors (Linear)",
)

predictors_with_confounders = {
    pred: res["identified_confounders"]
    for pred, res in all_confounder_results.items()
    if res["identified_confounders"]
}

# ============================================================================
# Kennedy with identified confounders — family="linear"
# ============================================================================

if predictors_with_confounders:
    example_predictor = list(predictors_with_confounders.keys())[0]
    example_confounders = predictors_with_confounders[example_predictor]

    results_kc = permutation_test_regression(
        X,
        y,
        method="kennedy",
        confounders=example_confounders,
        family="linear",
    )
    print_results_table(
        results_kc,
        title=(
            f"Kennedy (1995) for '{example_predictor}' "
            f"(controlling for {', '.join(example_confounders)}) "
            f"(family='linear')"
        ),
    )
    print_diagnostics_table(
        results_kc,
        title=f"Kennedy (1995) Diagnostics for '{example_predictor}' (family='linear')",
    )

# ============================================================================
# Direct ModelFamily protocol usage
# ============================================================================
# The ModelFamily protocol encapsulates every model-specific operation —
# fitting, prediction, residual extraction, Y-reconstruction, batch
# fitting, diagnostics, and classical p-values.  Below we exercise
# each method directly.

family = LinearFamily()
X_np = X.values.astype(float)
y_np = np.ravel(y).astype(float)

print(f"\n{'=' * 60}")
print("Direct LinearFamily protocol usage")
print(f"{'=' * 60}")
print(f"  name:               {family.name}")
print(f"  residual_type:      {family.residual_type}")
print(f"  direct_permutation: {family.direct_permutation}")

# validate_y — should pass without error for continuous Y
family.validate_y(y_np)
print("  validate_y:         passed")

# fit / predict / coefs / residuals
model = family.fit(X_np, y_np, fit_intercept=True)
preds = family.predict(model, X_np)
coefs = family.coefs(model)
resids = family.residuals(model, X_np, y_np)
print(f"  coefs:              {coefs}")
print(f"  mean |residual|:    {np.mean(np.abs(resids)):.4f}")

# fit_metric (RSS)
rss = family.fit_metric(y_np, preds)
print(f"  RSS:                {rss:.2f}")

# reconstruct_y — additive: ŷ + π(e)
rng = np.random.default_rng(42)
perm_resids = rng.permutation(resids)
y_star = family.reconstruct_y(preds, perm_resids, rng)
print(f"  reconstruct_y:      shape={y_star.shape}, mean={np.mean(y_star):.4f}")

# batch_fit — fit OLS on B permuted Y vectors at once
n_batch = 100
perm_indices = np.array([rng.permutation(len(y_np)) for _ in range(n_batch)])
Y_matrix = y_np[perm_indices]  # shape (B, n)
batch_coefs = family.batch_fit(X_np, Y_matrix, fit_intercept=True)
print(
    f"  batch_fit:          shape={batch_coefs.shape} (B={n_batch}, p={X_np.shape[1]})"
)

# diagnostics — OLS summary via statsmodels
diag = family.diagnostics(X_np, y_np, fit_intercept=True)
print(f"  diagnostics:        R²={diag['r_squared']:.4f}, F={diag['f_statistic']:.2f}")

# classical_p_values — Wald t-test via statsmodels
p_classical = family.classical_p_values(X_np, y_np, fit_intercept=True)
print(f"  classical_p_values: {np.round(p_classical, 6)}")

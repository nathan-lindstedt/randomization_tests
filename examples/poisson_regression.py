"""
Test Case 3: Poisson Regression (Count Outcome)
Abalone dataset (UCI ML Repository ID=1)

Demonstrates:
- ``family="poisson"`` — explicit family selection
- All five permutation methods routed through ``PoissonFamily``
- Direct ``ModelFamily`` protocol usage (fit / predict / residuals /
  reconstruct_y / fit_metric / diagnostics / classical_p_values /
  batch_fit)
- Poisson-specific diagnostics (deviance, Pearson χ², dispersion)

The target variable *Rings* is a natural count (number of growth rings
visible in a cross-section of the shell).  Adding 1.5 gives the age
in years.  Equi-dispersion is excellent: the marginal variance/mean
ratio is ≈ 1.05, and the model-conditional dispersion is ≈ 0.60,
making this a textbook Poisson outcome.
"""

import warnings

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    PoissonFamily,
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

abalone = fetch_ucirepo(id=1)
X_full = abalone.data.features.copy()
y_full = abalone.data.targets

# Subsample to 300 rows (permutation tests re-fit B × p GLMs, so
# keeping n moderate avoids excessive runtime for an example script).
rng = np.random.default_rng(42)
idx = rng.choice(len(X_full), size=300, replace=False)
X_sub = X_full.iloc[idx].reset_index(drop=True)
y_sub = y_full.iloc[idx].reset_index(drop=True)

# Feature engineering: dummy-code Sex (M/F/I) into two indicators.
# Male is the reference category.
X = pd.DataFrame(
    {
        "Shell_weight": X_sub["Shell_weight"].astype(float),
        "Shucked_weight": X_sub["Shucked_weight"].astype(float),
        "Height": X_sub["Height"].astype(float),
        "is_female": (X_sub["Sex"] == "F").astype(float),
        "is_infant": (X_sub["Sex"] == "I").astype(float),
    }
)
y = y_sub.copy()

print_dataset_info_table(
    name=abalone.metadata.name,
    n_observations=len(X),
    n_features=X.shape[1],
    feature_names=list(X.columns),
    target_name=y.columns[0],
    target_description="growth-ring count",
    y_range=(int(y.values.min()), int(y.values.max())),
    y_mean=float(y.values.mean()),
    y_var=float(y.values.var()),
)

# ============================================================================
# Family resolution
# ============================================================================

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    auto_family = resolve_family("auto", np.ravel(y))
assert auto_family.name == "linear", (
    f"Expected 'linear' from auto-detection (count auto-detect not yet "
    f"implemented), got {auto_family.name!r}"
)

# Explicit selection required for Poisson.
poisson_family = resolve_family("poisson", np.ravel(y))
assert poisson_family.name == "poisson"

print_family_info_table(
    auto_family=auto_family,
    explicit_family=poisson_family,
    advisory=[str(w.message) for w in caught],
)

# ============================================================================
# ter Braak (1992) — family="poisson" (explicit)
# ============================================================================

results_ter_braak = permutation_test_regression(
    X, y, method="ter_braak", family="poisson"
)
print_results_table(
    results_ter_braak,
    title="ter Braak (1992) Permutation Test (family='poisson')",
)
print_diagnostics_table(
    results_ter_braak,
    title="ter Braak (1992) Extended Diagnostics (family='poisson')",
)
assert results_ter_braak.family.name == "poisson"
assert results_ter_braak.family.name == "poisson"

# ============================================================================
# Kennedy (1995) individual — family="poisson"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_kennedy = permutation_test_regression(
        X, y, method="kennedy", confounders=[], family="poisson"
    )
print_results_table(
    results_kennedy,
    title="Kennedy (1995) Individual Permutation Test (family='poisson')",
)
print_diagnostics_table(
    results_kennedy,
    title="Kennedy (1995) Individual Diagnostics (family='poisson')",
)

# ============================================================================
# Kennedy (1995) joint — family="poisson"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_kennedy_joint = permutation_test_regression(
        X, y, method="kennedy_joint", confounders=[], family="poisson"
    )
print_joint_results_table(
    results_kennedy_joint,
    title="Kennedy (1995) Joint Permutation Test (family='poisson')",
)

# ============================================================================
# Freedman–Lane (1983) individual — family="poisson"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_fl = permutation_test_regression(
        X, y, method="freedman_lane", confounders=[], family="poisson"
    )
print_results_table(
    results_fl,
    title="Freedman–Lane (1983) Individual Permutation Test (family='poisson')",
)
print_diagnostics_table(
    results_fl,
    title="Freedman–Lane (1983) Individual Diagnostics (family='poisson')",
)

# ============================================================================
# Freedman–Lane (1983) joint — family="poisson"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_fl_joint = permutation_test_regression(
        X, y, method="freedman_lane_joint", confounders=[], family="poisson"
    )
print_joint_results_table(
    results_fl_joint,
    title="Freedman–Lane (1983) Joint Permutation Test (family='poisson')",
)

# ============================================================================
# Confounder identification
# ============================================================================

all_confounder_results = {}
for predictor in X.columns:
    all_confounder_results[predictor] = identify_confounders(
        X, y, predictor=predictor, family="poisson"
    )

print_confounder_table(
    all_confounder_results,
    title="Confounder Identification for All Predictors (Poisson)",
)

predictors_with_confounders = {
    pred: res.identified_confounders
    for pred, res in all_confounder_results.items()
    if res.identified_confounders
}

# ============================================================================
# Kennedy with identified confounders — family="poisson"
# ============================================================================

if predictors_with_confounders:
    example_predictor = list(predictors_with_confounders.keys())[0]
    example_confounders = predictors_with_confounders[example_predictor]

    results_kc = permutation_test_regression(
        X,
        y,
        method="kennedy",
        confounders=example_confounders,
        family="poisson",
    )
    print_results_table(
        results_kc,
        title=(
            f"Kennedy (1995) for '{example_predictor}' "
            f"(controlling for {', '.join(example_confounders)}) "
            f"(family='poisson')"
        ),
    )
    print_diagnostics_table(
        results_kc,
        title=f"Kennedy (1995) Diagnostics for '{example_predictor}' (family='poisson')",
    )

# ============================================================================
# Direct ModelFamily protocol usage
# ============================================================================
# The ModelFamily protocol encapsulates every model-specific operation —
# fitting, prediction, residual extraction, Y-reconstruction, batch
# fitting, diagnostics, and classical p-values.  Below we exercise
# each method directly for PoissonFamily.

family = PoissonFamily()
X_np = X.values.astype(float)
y_np = np.ravel(y).astype(float)

# validate_y — should pass for non-negative integer counts
family.validate_y(y_np)

# fit / predict / coefs / residuals
model = family.fit(X_np, y_np, fit_intercept=True)
preds = family.predict(model, X_np)
coefs = family.coefs(model)
resids = family.residuals(model, X_np, y_np)

# fit_metric (deviance)
deviance = family.fit_metric(y_np, preds)

# reconstruct_y — Poisson sampling (stochastic!)
rng = np.random.default_rng(42)
perm_resids = rng.permutation(resids)
y_star = family.reconstruct_y(preds[np.newaxis, :], perm_resids[np.newaxis, :], rng)

# batch_fit — Poisson GLM on B permuted Y vectors via joblib
n_batch = 50
perm_indices = np.array([rng.permutation(len(y_np)) for _ in range(n_batch)])
Y_matrix = y_np[perm_indices]  # shape (B, n)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    batch_coefs = family.batch_fit(X_np, Y_matrix, fit_intercept=True)
n_nan = int(np.sum(np.any(np.isnan(batch_coefs), axis=1)))

# diagnostics — Poisson GLM summary via statsmodels
diag = family.diagnostics(X_np, y_np, fit_intercept=True)
if diag["dispersion"] > 1.5:
    dispersion_status = (
        "⚠ OVERDISPERSION DETECTED — CONSIDER family='negative_binomial'"
    )
else:
    dispersion_status = "✓ NO OVERDISPERSION (GOOD POISSON FIT)"

# classical_p_values — Wald z-test via statsmodels
p_classical = family.classical_p_values(X_np, y_np, fit_intercept=True)

# exchangeability_cells — stub (returns None for global exchangeability)
cells = family.exchangeability_cells(X_np, y_np)

print_protocol_usage_table(
    results_ter_braak,
    title="Direct PoissonFamily Protocol Usage",
)

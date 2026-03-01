"""
Test Case 5: Ordinal Regression (Ordered Categorical Outcome)
Wine Quality dataset (UCI ML Repository ID=186)

Demonstrates:
- ``family="ordinal"`` — proportional-odds logistic regression via
  ``OrdinalFamily``
- Three supported permutation methods: ``ter_braak``, ``kennedy``,
  ``kennedy_joint``
- Freedman-Lane rejection with informative error message
- Direct ``ModelFamily`` protocol usage (fit / predict / coefs /
  diagnostics / classical_p_values / batch_fit)
- ``score`` / ``null_score`` for joint-test deviance

**Why only three methods?**

Ordinal residuals are not well-defined because the proportional-odds
model produces K-class probability vectors rather than scalar
residuals.  The Freedman-Lane method requires residuals for the
partial regression approach (residuals → permute → reconstruct Y*),
so it is incompatible with ordinal outcomes.

The ter Braak path uses direct Y permutation (Manly 1997), which is
valid under H₀ without residuals.  The Kennedy methods permute
exposure-model residuals, which are always from a linear OLS model
regardless of the outcome family.
"""

import warnings

import numpy as np
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    OrdinalFamily,
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
# Wine Quality dataset (combined red + white, 6497 observations).
# Y = quality score (integer 3–9), treated as ordinal.
# We subsample and select features for a manageable demo runtime.

wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets

# Select a representative feature subset (5 features)
selected_features = [
    "alcohol",
    "volatile_acidity",
    "sulphates",
    "citric_acid",
    "residual_sugar",
]
X = X[selected_features]

# Subsample for demo speed (ordinal BFGS is expensive per permutation)
rng = np.random.default_rng(42)
n_sub = 500
idx = rng.choice(X.shape[0], size=n_sub, replace=False)
X = X.iloc[idx].reset_index(drop=True)
y = y.iloc[idx].reset_index(drop=True)

# Convert quality to 0-indexed ordinal
y_min = int(y.values.min())
y = y - y_min

y_vals = np.ravel(y.values).astype(int)
ordinal_levels = sorted(np.unique(y_vals).tolist())

print_dataset_info_table(
    name=wine_quality.metadata.name,
    n_observations=len(X),
    n_features=X.shape[1],
    feature_names=list(X.columns),
    target_name="quality",
    target_description="wine quality score (ordinal)",
    extra_stats={
        "Outcome Levels": str(ordinal_levels),
        "Unique Categories": str(len(ordinal_levels)),
    },
)

# ============================================================================
# Family resolution
# ============================================================================

ordinal_family = resolve_family("ordinal", np.ravel(y))
assert ordinal_family.name == "ordinal"
assert isinstance(ordinal_family, OrdinalFamily)

print_family_info_table(
    explicit_family=ordinal_family,
)

# ============================================================================
# ter Braak (1992) — direct Y permutation (Manly 1997)
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Inverting hessian")
    results_ter_braak = permutation_test_regression(
        X, y, method="ter_braak", family="ordinal", n_permutations=999
    )
print_results_table(
    results_ter_braak,
    title="ter Braak (1992) Permutation Test (family='ordinal')",
)
print_diagnostics_table(
    results_ter_braak,
    title="ter Braak (1992) Extended Diagnostics (family='ordinal')",
)

# ============================================================================
# Kennedy (1995) individual — family="ordinal"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    warnings.filterwarnings("ignore", message="Inverting hessian")
    results_kennedy = permutation_test_regression(
        X,
        y,
        method="kennedy",
        confounders=[],
        family="ordinal",
        n_permutations=999,
    )
print_results_table(
    results_kennedy,
    title="Kennedy (1995) Individual Permutation Test (family='ordinal')",
)
print_diagnostics_table(
    results_kennedy,
    title="Kennedy (1995) Individual Diagnostics (family='ordinal')",
)

# ============================================================================
# Kennedy (1995) joint — family="ordinal"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    warnings.filterwarnings("ignore", message="Inverting hessian")
    results_kennedy_joint = permutation_test_regression(
        X,
        y,
        method="kennedy_joint",
        confounders=[],
        family="ordinal",
        n_permutations=999,
    )
print_joint_results_table(
    results_kennedy_joint,
    title="Kennedy (1995) Joint Permutation Test (family='ordinal')",
)

# ============================================================================
# Freedman-Lane rejection (expected ValueError)
# ============================================================================

print("\n" + "=" * 80)
print("Freedman-Lane rejection (expected)")
print("=" * 80)

for fl_method in ("freedman_lane", "freedman_lane_joint"):
    try:
        permutation_test_regression(
            X,
            y,
            method=fl_method,
            family="ordinal",
            confounders=[],
            n_permutations=999,
        )
        print(f"ERROR: {fl_method} should have raised ValueError!")
    except ValueError as e:
        print(f"✓ {fl_method} correctly rejected: {str(e)[:80]}...")

# ============================================================================
# Confounder identification
# ============================================================================

all_confounder_results = {}
for predictor in X.columns:
    all_confounder_results[predictor] = identify_confounders(
        X, y, predictor=predictor, family="ordinal"
    )

print_confounder_table(
    all_confounder_results,
    title="Confounder Identification for All Predictors (Ordinal)",
)

predictors_with_confounders = {
    pred: res.identified_confounders
    for pred, res in all_confounder_results.items()
    if res.identified_confounders
}

# ============================================================================
# Kennedy with identified confounders — family="ordinal"
# ============================================================================

if predictors_with_confounders:
    example_predictor = list(predictors_with_confounders.keys())[0]
    example_confounders = predictors_with_confounders[example_predictor]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Inverting hessian")
        results_kc = permutation_test_regression(
            X,
            y,
            method="kennedy",
            family="ordinal",
            confounders=example_confounders,
            n_permutations=999,
        )
    print_results_table(
        results_kc,
        title=(
            f"Kennedy (1995) for '{example_predictor}' "
            f"(controlling for {', '.join(example_confounders)}) "
            f"(family='ordinal')"
        ),
    )
    print_diagnostics_table(
        results_kc,
        title=(
            f"Kennedy (1995) Diagnostics for '{example_predictor}' (family='ordinal')"
        ),
    )

# ============================================================================
# Direct ModelFamily protocol usage
# ============================================================================
# The ModelFamily protocol encapsulates every model-specific operation —
# fitting, prediction, residual extraction, Y-reconstruction, batch
# fitting, diagnostics, and classical p-values.  Below we exercise
# each method directly for OrdinalFamily.

family = OrdinalFamily()
X_np = X.values.astype(float)
y_np = np.ravel(y.values).astype(float)

# validate_y
family.validate_y(y_np)

# fit / predict / coefs
model = family.fit(X_np, y_np, fit_intercept=True)
preds = family.predict(model, X_np)
coefs = family.coefs(model)

# score / null_score — deviance
deviance = family.score(model, X_np, y_np)
null_deviance = family.null_score(y_np)

# NotImplementedError checks — ordinal does not support residuals,
# reconstruct_y, or fit_metric (the engine uses direct Y permutation).
for method_name in ("residuals", "reconstruct_y", "fit_metric"):
    try:
        if method_name == "residuals":
            family.residuals(model, X_np, y_np)
        elif method_name == "reconstruct_y":
            rng = np.random.default_rng(0)
            family.reconstruct_y(np.zeros((1, 5)), np.zeros((1, 5)), rng)
        elif method_name == "fit_metric":
            family.fit_metric(y_np, preds)
    except NotImplementedError:
        pass  # Expected: ordinal does not support these methods

# batch_fit — fit ordinal on B permuted Y vectors at once
rng = np.random.default_rng(42)
n_batch = 50
perm_indices = np.array([rng.permutation(len(y_np)) for _ in range(n_batch)])
Y_matrix = y_np[perm_indices]  # shape (B, n)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="Inverting hessian")
    batch_coefs = family.batch_fit(X_np, Y_matrix, fit_intercept=True)
n_nan = int(np.sum(np.any(np.isnan(batch_coefs), axis=1)))

# diagnostics — proportional-odds summary
diag = family.diagnostics(X_np, y_np, fit_intercept=True)

# classical_p_values — Wald z-test via statsmodels
p_classical = family.classical_p_values(X_np, y_np, fit_intercept=True)

# exchangeability_cells — stub (returns None for global exchangeability)
cells = family.exchangeability_cells(X_np, y_np)

print_protocol_usage_table(
    results_ter_braak,
    title="Direct OrdinalFamily Protocol Usage",
)

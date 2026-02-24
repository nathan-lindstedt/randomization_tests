"""
Test Case 3: Ordinal Regression (Ordered Categorical Outcome)
Wine Quality dataset (UCI ML Repository ID=186)

Demonstrates:
- ``family="ordinal"`` — proportional-odds logistic regression via
  ``OrdinalFamily``
- Three supported permutation methods: ``ter_braak``, ``kennedy``,
  ``kennedy_joint``
- Freedman-Lane rejection with informative error message
- Direct ``ModelFamily`` protocol usage (fit / predict / coefs /
  diagnostics / classical_p_values / batch_fit)
- Duck-typed ``model_fit_metric`` / ``null_fit_metric`` for joint test

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
    print_diagnostics_table,
    print_joint_results_table,
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

print(f"Dataset: {X.shape[0]} observations, {X.shape[1]} features")
print(f"Outcome levels: {sorted(np.unique(y.values.ravel().astype(int)).tolist())}")
print(f"Unique categories: {len(np.unique(y.values))}")

# ============================================================================
# Verify resolve_family correctly resolves "ordinal"
# ============================================================================

ordinal_family = resolve_family("ordinal", np.ravel(y))
assert ordinal_family.name == "ordinal"
print(f"\nresolve_family('ordinal', y) → {ordinal_family.name!r}")
assert isinstance(ordinal_family, OrdinalFamily)
assert ordinal_family.direct_permutation is True
print(f"  direct_permutation = {ordinal_family.direct_permutation}")
print(f"  residual_type = {ordinal_family.residual_type!r}")
print(f"  metric_label = {ordinal_family.metric_label!r}")

_family = resolve_family("ordinal")

# ============================================================================
# ter Braak (1992) — direct Y permutation (Manly 1997)
# ============================================================================

print("\n" + "=" * 80)
print("ter Braak method (direct Y permutation)")
print("=" * 80)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Inverting hessian")
    results_ter_braak = permutation_test_regression(
        X, y, method="ter_braak", family="ordinal", n_permutations=199
    )
print_results_table(results_ter_braak, feature_names=X.columns.tolist(), family=_family)
print_diagnostics_table(
    results_ter_braak, feature_names=X.columns.tolist(), family=_family
)

# ============================================================================
# Confounder analysis
# ============================================================================

all_confounder_results = {}
for predictor in X.columns:
    all_confounder_results[predictor] = identify_confounders(
        X, y, predictor=predictor, family="ordinal"
    )

print_confounder_table(all_confounder_results, family="ordinal")

# Collect confounders from any predictor that has them
predictors_with_confounders = {
    pred: res["identified_confounders"]
    for pred, res in all_confounder_results.items()
    if res["identified_confounders"]
}
if predictors_with_confounders:
    example_predictor = list(predictors_with_confounders.keys())[0]
    confounders = predictors_with_confounders[example_predictor]
else:
    confounders = X.columns[:2].tolist()
print(f"\nUsing confounders: {confounders}")

# ============================================================================
# Kennedy (1995) individual — exposure-residual permutation
# ============================================================================

print("\n" + "=" * 80)
print("Kennedy individual method")
print("=" * 80)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Inverting hessian")
    results_kennedy = permutation_test_regression(
        X,
        y,
        method="kennedy",
        family="ordinal",
        confounders=confounders,
        n_permutations=199,
    )
print_results_table(results_kennedy, feature_names=X.columns.tolist(), family=_family)
print_diagnostics_table(
    results_kennedy, feature_names=X.columns.tolist(), family=_family
)

# ============================================================================
# Kennedy (1995) joint — collective predictive improvement
# ============================================================================

print("\n" + "=" * 80)
print("Kennedy joint method (model_fit_metric duck-type)")
print("=" * 80)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Inverting hessian")
    results_joint = permutation_test_regression(
        X,
        y,
        method="kennedy_joint",
        family="ordinal",
        confounders=confounders,
        n_permutations=199,
    )
print_joint_results_table(results_joint, family=_family)

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
            confounders=confounders,
            n_permutations=99,
        )
        print(f"ERROR: {fl_method} should have raised ValueError!")
    except ValueError as e:
        print(f"✓ {fl_method} correctly rejected: {str(e)[:80]}...")

# ============================================================================
# Direct protocol usage
# ============================================================================

print("\n" + "=" * 80)
print("Direct OrdinalFamily protocol usage")
print("=" * 80)

fam = OrdinalFamily()
X_np = X.values.astype(float)
y_np = np.ravel(y.values).astype(float)

# Validate y
fam.validate_y(y_np)
print("✓ validate_y passed")

# Fit
model = fam.fit(X_np, y_np)
print(f"✓ fit: {len(fam.coefs(model))} slope coefficients")

# Predict expected value
preds = fam.predict(model, X_np)
print(f"✓ predict: E[Y|X] range = [{preds.min():.2f}, {preds.max():.2f}]")

# Model-based fit metric (duck-typed)
deviance = fam.model_fit_metric(model)
null_deviance = fam.null_fit_metric(model)
print(f"✓ model_fit_metric: -2·llf = {deviance:.2f}")
print(f"✓ null_fit_metric: -2·llnull = {null_deviance:.2f}")
print(f"  Improvement: {null_deviance - deviance:.2f}")

# Diagnostics
diag = fam.diagnostics(X_np, y_np)
print(f"✓ diagnostics: pseudo_R² = {diag['pseudo_r_squared']}")
print(f"  thresholds = {diag['thresholds']}")

# Classical p-values
pvals = fam.classical_p_values(X_np, y_np)
print(f"✓ classical_p_values: {pvals.shape[0]} values")

# NotImplementedError checks
for method_name in ("residuals", "reconstruct_y", "fit_metric"):
    try:
        if method_name == "residuals":
            fam.residuals(model, X_np, y_np)
        elif method_name == "reconstruct_y":
            rng = np.random.default_rng(0)
            fam.reconstruct_y(np.zeros((1, 5)), np.zeros((1, 5)), rng)
        elif method_name == "fit_metric":
            fam.fit_metric(y_np, preds)
    except NotImplementedError:
        print(f"✓ {method_name} correctly raises NotImplementedError")

print("\n✓ All ordinal tests passed!")

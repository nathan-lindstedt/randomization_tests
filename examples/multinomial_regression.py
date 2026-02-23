"""
Test Case 6: Multinomial Logistic Regression (Unordered Categorical Outcome)
Wine dataset (UCI ML Repository ID=109)

Demonstrates:
- ``family="multinomial"`` — explicit family selection (no auto-detection)
- ``ter_braak``, ``kennedy``, and ``kennedy_joint`` permutation methods
  (Freedman-Lane methods are not supported for multinomial because
  residuals are not well-defined)
- Wald χ² test statistics as the per-predictor scalar summary
- Direct ``ModelFamily`` protocol usage (fit / predict / coefs /
  category_coefs / batch_fit / diagnostics / classical_p_values)
"""

import warnings

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    MultinomialFamily,
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

wine = fetch_ucirepo(id=109)
X_wine = wine.data.features
y_wine = wine.data.targets

# Wine classes are 1, 2, 3 — recode to 0, 1, 2 for MultinomialFamily.
y_wine = pd.DataFrame(
    y_wine.iloc[:, 0].values - 1,
    columns=["class"],
)

# Select 5 features with low multicollinearity (all VIF ≤ 1.5).
selected_features = ["Alcohol", "Malicacid", "Ash", "Magnesium", "Hue"]
X_wine = X_wine[selected_features]

y_values = np.ravel(y_wine)
class_names = ["class_0", "class_1", "class_2"]
print(f"Classes: {dict(enumerate(class_names))}")
print(f"Class counts: {dict(zip(class_names, np.bincount(y_values), strict=True))}")

# ============================================================================
# ter Braak (1992) — family="multinomial"
# ============================================================================

results_ter_braak = permutation_test_regression(
    X_wine, y_wine, method="ter_braak", family="multinomial"
)
print_results_table(
    results_ter_braak,
    feature_names=X_wine.columns.tolist(),
    target_name="class",
    title="ter Braak (1992) Permutation Test (family='multinomial')",
)
print_diagnostics_table(
    results_ter_braak,
    feature_names=X_wine.columns.tolist(),
    title="ter Braak (1992) Diagnostics (family='multinomial')",
)
assert results_ter_braak["model_type"] == "multinomial"

# ============================================================================
# Kennedy (1995) individual — family="multinomial"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_kennedy = permutation_test_regression(
        X_wine, y_wine, method="kennedy", confounders=[], family="multinomial"
    )
print_results_table(
    results_kennedy,
    feature_names=X_wine.columns.tolist(),
    target_name="class",
    title="Kennedy (1995) Individual Permutation Test (family='multinomial')",
)
print_diagnostics_table(
    results_kennedy,
    feature_names=X_wine.columns.tolist(),
    title="Kennedy (1995) Individual Diagnostics (family='multinomial')",
)

# ============================================================================
# Kennedy (1995) joint — family="multinomial"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_kennedy_joint = permutation_test_regression(
        X_wine,
        y_wine,
        method="kennedy_joint",
        confounders=[],
        family="multinomial",
    )
print_joint_results_table(
    results_kennedy_joint,
    target_name="class",
    title="Kennedy (1995) Joint Permutation Test (family='multinomial')",
)

# ============================================================================
# Confounder identification — multinomial
# ============================================================================

all_confounder_results = {}
for predictor in X_wine.columns:
    all_confounder_results[predictor] = identify_confounders(
        X_wine, y_wine, predictor=predictor
    )

print_confounder_table(
    all_confounder_results,
    title="Confounder Identification for All Predictors (Multinomial)",
)

predictors_with_confounders = {
    pred: res["identified_confounders"]
    for pred, res in all_confounder_results.items()
    if res["identified_confounders"]
}

# ============================================================================
# Kennedy with identified confounders — family="multinomial"
# ============================================================================

if predictors_with_confounders:
    example_predictor = list(predictors_with_confounders.keys())[0]
    example_confounders = predictors_with_confounders[example_predictor]

    results_kc = permutation_test_regression(
        X_wine,
        y_wine,
        method="kennedy",
        confounders=example_confounders,
        family="multinomial",
    )
    print_results_table(
        results_kc,
        feature_names=X_wine.columns.tolist(),
        target_name="class",
        title=(
            f"Kennedy (1995) for '{example_predictor}' "
            f"(controlling for {', '.join(example_confounders)}) "
            f"(family='multinomial')"
        ),
    )
    print_diagnostics_table(
        results_kc,
        feature_names=X_wine.columns.tolist(),
        title=(
            f"Kennedy (1995) Diagnostics for '{example_predictor}' "
            f"(family='multinomial')"
        ),
    )

# ============================================================================
# Direct ModelFamily protocol usage
# ============================================================================
# MultinomialFamily implements the ModelFamily protocol with key
# differences from scalar-response families:
#
# - coefs() returns per-predictor Wald χ² statistics (scalar per
#   predictor), since each predictor has K−1 coefficients.
# - category_coefs() (duck-typed, not on the protocol) returns the
#   full (p, K−1) coefficient matrix.
# - residuals(), reconstruct_y(), and fit_metric() raise
#   NotImplementedError — the engine uses direct Y permutation.

family = MultinomialFamily()
X_np = X_wine.values.astype(float)
y_np = y_values.astype(float)

print(f"\n{'=' * 60}")
print("Direct MultinomialFamily protocol usage")
print(f"{'=' * 60}")
print(f"  name:               {family.name}")
print(f"  residual_type:      {family.residual_type}")
print(f"  direct_permutation: {family.direct_permutation}")

# validate_y — should pass for {0, 1, 2}
family.validate_y(y_np)
print("  validate_y:         passed")

# fit / predict / coefs
model = family.fit(X_np, y_np, fit_intercept=True)
preds = family.predict(model, X_np)
wald_chi2 = family.coefs(model)
print(f"  Wald χ²:            {np.round(wald_chi2, 4)}")
print(f"  E[Y|X] range:      [{preds.min():.4f}, {preds.max():.4f}]")

# category_coefs — (p, K-1) coefficient matrix
cat_coefs = family.category_coefs(model)
print(f"  category_coefs:     shape={cat_coefs.shape}")
for j in range(cat_coefs.shape[0]):
    print(f"    {selected_features[j]}: {np.round(cat_coefs[j], 4)}")

# model_fit_metric / null_fit_metric — deviance (duck-typed)
dev = family.model_fit_metric(model)
dev_null = family.null_fit_metric(model)
print(f"  deviance (full):    {dev:.2f}")
print(f"  deviance (null):    {dev_null:.2f}")

# batch_fit — fit multinomial on B permuted Y vectors at once
rng = np.random.default_rng(42)
n_batch = 50
perm_indices = np.array([rng.permutation(len(y_np)) for _ in range(n_batch)])
Y_matrix = y_np[perm_indices]  # shape (B, n)
batch_stats = family.batch_fit(X_np, Y_matrix, fit_intercept=True)
print(
    f"  batch_fit:          shape={batch_stats.shape} (B={n_batch}, p={X_np.shape[1]})"
)

# diagnostics — pseudo-R², LL, AIC, BIC, category counts
diag = family.diagnostics(X_np, y_np, fit_intercept=True)
print(
    f"  diagnostics:        pseudo_R²={diag['pseudo_r_squared']:.4f}, "
    f"AIC={diag['aic']:.2f}"
)
print(f"  category counts:    {diag['category_counts']}")

# classical_p_values — Wald χ²(K-1) p-values
p_classical = family.classical_p_values(X_np, y_np, fit_intercept=True)
print(f"  classical_p_values: {np.round(p_classical, 6)}")

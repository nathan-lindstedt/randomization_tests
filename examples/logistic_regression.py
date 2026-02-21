"""
Test Case 2: Logistic Regression (Binary Outcome)
Breast Cancer Wisconsin (Diagnostic) dataset (UCI ML Repository ID=17)
"""

from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    identify_confounders,
    permutation_test_regression,
    print_joint_results_table,
    print_results_table,
)

# ============================================================================
# Load data
# ============================================================================

breast_cancer = fetch_ucirepo(id=17)
X_bc = breast_cancer.data.features
y_bc = breast_cancer.data.targets

# Convert target to binary: malignant (M) -> 1, benign (B) -> 0
y_bc = (y_bc == "M").astype(int)

selected_features = ["radius1", "texture1", "perimeter1", "smoothness1", "compactness1"]
X_bc = X_bc[selected_features]

# ============================================================================
# ter Braak (1992) — logistic
# ============================================================================

results_ter_braak_bc = permutation_test_regression(X_bc, y_bc, method="ter_braak")
print_results_table(
    results_ter_braak_bc,
    feature_names=X_bc.columns.tolist(),
    target_name=y_bc.columns[0],
    title="ter Braak (1992) Permutation Test (Logistic)",
)

# ============================================================================
# Kennedy (1995) individual — logistic
# ============================================================================

results_kennedy_bc = permutation_test_regression(X_bc, y_bc, method="kennedy", confounders=[])
print_results_table(
    results_kennedy_bc,
    feature_names=X_bc.columns.tolist(),
    target_name=y_bc.columns[0],
    title="Kennedy (1995) Individual Coefficient Permutation Test (Logistic)",
)

# ============================================================================
# Kennedy (1995) joint — logistic
# ============================================================================

results_kennedy_joint_bc = permutation_test_regression(X_bc, y_bc, method="kennedy_joint", confounders=[])
print_joint_results_table(
    results_kennedy_joint_bc,
    target_name=y_bc.columns[0],
    title="Kennedy (1995) Joint Permutation Test (Logistic)",
)

# ============================================================================
# Confounder identification — logistic
# ============================================================================

print("Confounder Identification for All Predictors (Logistic)\n")

all_confounder_results_bc = {}
for predictor in X_bc.columns:
    all_confounder_results_bc[predictor] = identify_confounders(X_bc, y_bc, predictor=predictor)

for predictor, results in all_confounder_results_bc.items():
    print(f"Predictor: '{predictor}'")
    print(f"  Identified Confounders: {results['identified_confounders']}")
    print(f"  Identified Mediators: {results['identified_mediators']}")
    if results["identified_confounders"] or results["identified_mediators"]:
        print(f"  Recommendation: {results['recommendation']}")
    print()

predictors_with_confounders_bc = {
    pred: res["identified_confounders"]
    for pred, res in all_confounder_results_bc.items()
    if res["identified_confounders"]
}

print("Summary: Predictors with Identified Confounders (Logistic)\n")
if predictors_with_confounders_bc:
    for pred, confounders in predictors_with_confounders_bc.items():
        print(f"  {pred}: control for {confounders}")
else:
    print("  No confounders identified for any predictor.")
    print("  This suggests the predictors are relatively independent.")
print()

# ============================================================================
# Kennedy with identified confounders — logistic
# ============================================================================

if predictors_with_confounders_bc:
    example_predictor_bc = list(predictors_with_confounders_bc.keys())[0]
    example_confounders_bc = predictors_with_confounders_bc[example_predictor_bc]

    results_kc_bc = permutation_test_regression(
        X_bc, y_bc, method="kennedy", confounders=example_confounders_bc,
    )
    print_results_table(
        results_kc_bc,
        feature_names=X_bc.columns.tolist(),
        target_name=y_bc.columns[0],
        title=f"Kennedy (1995) Method for '{example_predictor_bc}' (controlling for {example_confounders_bc}) (Logistic)",
    )

"""
Test Case 2: Logistic Regression (Binary Outcome)
Breast Cancer Wisconsin (Diagnostic) dataset (UCI ML Repository ID=17)
"""

import warnings

from ucimlrepo import fetch_ucirepo

from randomization_tests import (
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
print_diagnostics_table(
    results_ter_braak_bc,
    feature_names=X_bc.columns.tolist(),
    title="ter Braak (1992) Extended Diagnostics (Logistic)",
)

# ============================================================================
# Kennedy (1995) individual — logistic
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_kennedy_bc = permutation_test_regression(
        X_bc, y_bc, method="kennedy", confounders=[]
    )
print_results_table(
    results_kennedy_bc,
    feature_names=X_bc.columns.tolist(),
    target_name=y_bc.columns[0],
    title="Kennedy (1995) Individual Coefficient Permutation Test (Logistic)",
)
print_diagnostics_table(
    results_kennedy_bc,
    feature_names=X_bc.columns.tolist(),
    title="Kennedy (1995) Individual Extended Diagnostics (Logistic)",
)

# ============================================================================
# Kennedy (1995) joint — logistic
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_kennedy_joint_bc = permutation_test_regression(
        X_bc, y_bc, method="kennedy_joint", confounders=[]
    )
print_joint_results_table(
    results_kennedy_joint_bc,
    target_name=y_bc.columns[0],
    title="Kennedy (1995) Joint Permutation Test (Logistic)",
)

# ============================================================================
# Confounder identification — logistic
# ============================================================================

all_confounder_results_bc = {}
for predictor in X_bc.columns:
    all_confounder_results_bc[predictor] = identify_confounders(
        X_bc, y_bc, predictor=predictor
    )

print_confounder_table(
    all_confounder_results_bc,
    title="Confounder Identification for All Predictors (Logistic)",
)

predictors_with_confounders_bc = {
    pred: res["identified_confounders"]
    for pred, res in all_confounder_results_bc.items()
    if res["identified_confounders"]
}

# ============================================================================
# Kennedy with identified confounders — logistic
# ============================================================================

if predictors_with_confounders_bc:
    example_predictor_bc = list(predictors_with_confounders_bc.keys())[0]
    example_confounders_bc = predictors_with_confounders_bc[example_predictor_bc]

    results_kc_bc = permutation_test_regression(
        X_bc,
        y_bc,
        method="kennedy",
        confounders=example_confounders_bc,
    )
    print_results_table(
        results_kc_bc,
        feature_names=X_bc.columns.tolist(),
        target_name=y_bc.columns[0],
        title=f"Kennedy (1995) Method for '{example_predictor_bc}' (controlling for {', '.join(example_confounders_bc)}) (Logistic)",
    )
    print_diagnostics_table(
        results_kc_bc,
        feature_names=X_bc.columns.tolist(),
        title=f"Kennedy (1995) Extended Diagnostics for '{example_predictor_bc}' (Logistic)",
    )

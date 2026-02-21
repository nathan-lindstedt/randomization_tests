"""
Test Case 1: Linear Regression (Continuous Outcome)
Real Estate Valuation dataset (UCI ML Repository ID=477)
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

real_estate_valuation = fetch_ucirepo(id=477)
X = real_estate_valuation.data.features
y = real_estate_valuation.data.targets

# ============================================================================
# ter Braak (1992)
# ============================================================================

results_ter_braak = permutation_test_regression(X, y, method="ter_braak")
print_results_table(
    results_ter_braak,
    feature_names=X.columns.tolist(),
    target_name=y.columns[0],
    title="ter Braak (1992) Permutation Test (Linear)",
)

# ============================================================================
# Kennedy (1995) individual (no confounders)
# ============================================================================

results_kennedy = permutation_test_regression(X, y, method="kennedy", confounders=[])
print_results_table(
    results_kennedy,
    feature_names=X.columns.tolist(),
    target_name=y.columns[0],
    title="Kennedy (1995) Individual Coefficient Permutation Test (Linear)",
)

# ============================================================================
# Kennedy (1995) joint
# ============================================================================

results_kennedy_joint = permutation_test_regression(X, y, method="kennedy_joint", confounders=[])
print_joint_results_table(
    results_kennedy_joint,
    target_name=y.columns[0],
    title="Kennedy (1995) Joint Permutation Test (Linear)",
)

# ============================================================================
# Confounder identification
# ============================================================================

print("Confounder Identification for All Predictors\n")

all_confounder_results = {}
for predictor in X.columns:
    all_confounder_results[predictor] = identify_confounders(X, y, predictor=predictor)

for predictor, results in all_confounder_results.items():
    print(f"Predictor: '{predictor}'")
    print(f"  Identified Confounders: {results['identified_confounders']}")
    print(f"  Identified Mediators: {results['identified_mediators']}")
    if results["identified_confounders"] or results["identified_mediators"]:
        print(f"  Recommendation: {results['recommendation']}")
    print()

predictors_with_confounders = {
    pred: res["identified_confounders"]
    for pred, res in all_confounder_results.items()
    if res["identified_confounders"]
}

print("Summary: Predictors with Identified Confounders\n")
if predictors_with_confounders:
    for pred, confounders in predictors_with_confounders.items():
        print(f"  {pred}: control for {confounders}")
else:
    print("  No confounders identified for any predictor.")
    print("  This suggests the predictors are relatively independent.")
print()

# ============================================================================
# Kennedy with identified confounders
# ============================================================================

if predictors_with_confounders:
    example_predictor = list(predictors_with_confounders.keys())[0]
    example_confounders = predictors_with_confounders[example_predictor]

    results_kc = permutation_test_regression(
        X, y, method="kennedy", confounders=example_confounders,
    )
    print_results_table(
        results_kc,
        feature_names=X.columns.tolist(),
        target_name=y.columns[0],
        title=f"Kennedy (1995) Method for '{example_predictor}' (controlling for {example_confounders}) (Linear)",
    )

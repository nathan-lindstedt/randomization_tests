"""
Test Case 2: Logistic Regression (Binary Outcome)
Breast Cancer Wisconsin (Diagnostic) dataset (UCI ML Repository ID=17)

Demonstrates:
- ``family="logistic"`` — explicit family selection
- All five permutation methods routed through ``LogisticFamily``
- Direct ``ModelFamily`` protocol usage (fit / predict / residuals /
  reconstruct_y / fit_metric / diagnostics / classical_p_values /
  batch_fit)
"""

import warnings

import numpy as np
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    LogisticFamily,
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

breast_cancer = fetch_ucirepo(id=17)
X_bc = breast_cancer.data.features
y_bc = breast_cancer.data.targets

# Convert target to binary: malignant (M) -> 1, benign (B) -> 0
y_bc = (y_bc == "M").astype(int)

selected_features = ["radius1", "texture1", "perimeter1", "smoothness1", "compactness1"]
X_bc = X_bc[selected_features]

print_dataset_info_table(
    name=breast_cancer.metadata.name,
    n_observations=len(X_bc),
    n_features=X_bc.shape[1],
    feature_names=list(X_bc.columns),
    target_name="Diagnosis",
    target_description="malignant (1) vs benign (0)",
    y_range=(int(y_bc.values.min()), int(y_bc.values.max())),
    y_mean=float(y_bc.values.mean()),
    extra_stats={"Prevalence": f"{float(y_bc.values.mean()):.2%}"},
)

# ============================================================================
# Verify resolve_family auto-detects "logistic" for binary Y
# ============================================================================

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    auto_family = resolve_family("auto", np.ravel(y_bc))
assert auto_family.name == "logistic", f"Expected 'logistic', got {auto_family.name!r}"

logistic_family = resolve_family("logistic", np.ravel(y_bc))
assert logistic_family.name == "logistic"

print_family_info_table(
    auto_family=auto_family,
    explicit_family=logistic_family,
    advisory=[str(w.message) for w in caught],
)

# ============================================================================
# ter Braak (1992) — family="auto" (auto-detection)
# ============================================================================

results_ter_braak_auto_bc = permutation_test_regression(
    X_bc, y_bc, method="ter_braak", family="auto"
)
assert results_ter_braak_auto_bc.family.name == "logistic"
print_results_table(
    results_ter_braak_auto_bc,
    title="ter Braak (1992) Permutation Test (family='auto' \u2192 logistic)",
)

# ============================================================================
# ter Braak (1992) — family="logistic" (explicit)
# ============================================================================

results_ter_braak_bc = permutation_test_regression(
    X_bc, y_bc, method="ter_braak", family="logistic"
)
print_results_table(
    results_ter_braak_bc,
    title="ter Braak (1992) Permutation Test (family='logistic')",
)
print_diagnostics_table(
    results_ter_braak_bc,
    title="ter Braak (1992) Diagnostics (family='logistic')",
)
assert results_ter_braak_bc.family.name == "logistic"

# ============================================================================
# Kennedy (1995) individual — family="logistic"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_kennedy_bc = permutation_test_regression(
        X_bc, y_bc, method="kennedy", confounders=[], family="logistic"
    )
print_results_table(
    results_kennedy_bc,
    title="Kennedy (1995) Individual Permutation Test (family='logistic')",
)
print_diagnostics_table(
    results_kennedy_bc,
    title="Kennedy (1995) Individual Diagnostics (family='logistic')",
)

# ============================================================================
# Kennedy (1995) joint — family="logistic"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_kennedy_joint_bc = permutation_test_regression(
        X_bc, y_bc, method="kennedy_joint", confounders=[], family="logistic"
    )
print_joint_results_table(
    results_kennedy_joint_bc,
    title="Kennedy (1995) Joint Permutation Test (family='logistic')",
)

# ============================================================================
# Freedman–Lane (1983) individual — family="logistic"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_fl_bc = permutation_test_regression(
        X_bc, y_bc, method="freedman_lane", confounders=[], family="logistic"
    )
print_results_table(
    results_fl_bc,
    title="Freedman–Lane (1983) Individual Permutation Test (family='logistic')",
)
print_diagnostics_table(
    results_fl_bc,
    title="Freedman–Lane (1983) Individual Diagnostics (family='logistic')",
)

# ============================================================================
# Freedman–Lane (1983) joint — family="logistic"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_fl_joint_bc = permutation_test_regression(
        X_bc, y_bc, method="freedman_lane_joint", confounders=[], family="logistic"
    )
print_joint_results_table(
    results_fl_joint_bc,
    title="Freedman–Lane (1983) Joint Permutation Test (family='logistic')",
)

# ============================================================================
# Confounder identification — logistic
# ============================================================================

all_confounder_results_bc = {}
for predictor in X_bc.columns:
    all_confounder_results_bc[predictor] = identify_confounders(
        X_bc, y_bc, predictor=predictor, family="logistic"
    )

print_confounder_table(
    all_confounder_results_bc,
    title="Confounder Identification for All Predictors (Logistic)",
)

predictors_with_confounders_bc = {
    pred: res.identified_confounders
    for pred, res in all_confounder_results_bc.items()
    if res.identified_confounders
}

# ============================================================================
# Kennedy with identified confounders — family="logistic"
# ============================================================================

if predictors_with_confounders_bc:
    example_predictor_bc = list(predictors_with_confounders_bc.keys())[0]
    example_confounders_bc = predictors_with_confounders_bc[example_predictor_bc]

    results_kc_bc = permutation_test_regression(
        X_bc,
        y_bc,
        method="kennedy",
        confounders=example_confounders_bc,
        family="logistic",
    )
    print_results_table(
        results_kc_bc,
        title=(
            f"Kennedy (1995) for '{example_predictor_bc}' "
            f"(controlling for {', '.join(example_confounders_bc)}) "
            f"(family='logistic')"
        ),
    )
    print_diagnostics_table(
        results_kc_bc,
        title=(
            f"Kennedy (1995) Diagnostics for '{example_predictor_bc}' "
            f"(family='logistic')"
        ),
    )

# ============================================================================
# Direct ModelFamily protocol usage
# ============================================================================
# The ModelFamily protocol encapsulates every model-specific operation —
# fitting, prediction, residual extraction, Y-reconstruction, batch
# fitting, diagnostics, and classical p-values.  Below we exercise
# each method directly for LogisticFamily.

family = LogisticFamily()
X_np = X_bc.values.astype(float)
y_np = np.ravel(y_bc).astype(float)

# validate_y — should pass for binary {0, 1}
family.validate_y(y_np)

# fit / predict / coefs / residuals
model = family.fit(X_np, y_np, fit_intercept=True)
preds = family.predict(model, X_np)
coefs = family.coefs(model)
resids = family.residuals(model, X_np, y_np)

# fit_metric (deviance)
deviance = family.fit_metric(y_np, preds)

# reconstruct_y — clip + Bernoulli sampling (stochastic!)
rng = np.random.default_rng(42)
perm_resids = rng.permutation(resids)
y_star = family.reconstruct_y(preds[np.newaxis, :], perm_resids[np.newaxis, :], rng)

# batch_fit — fit logistic on B permuted Y vectors at once
n_batch = 50
perm_indices = np.array([rng.permutation(len(y_np)) for _ in range(n_batch)])
Y_matrix = y_np[perm_indices]  # shape (B, n)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    batch_coefs = family.batch_fit(X_np, Y_matrix, fit_intercept=True)
n_nan = int(np.sum(np.any(np.isnan(batch_coefs), axis=1)))

# diagnostics — pseudo-R², LLR, AIC, BIC via statsmodels
diag = family.diagnostics(X_np, y_np, fit_intercept=True)

# classical_p_values — Wald z-test via statsmodels
p_classical = family.classical_p_values(X_np, y_np, fit_intercept=True)

# exchangeability_cells — stub (returns None for global exchangeability)
cells = family.exchangeability_cells(X_np, y_np)

print_protocol_usage_table(
    results_ter_braak_bc,
    title="Direct LogisticFamily Protocol Usage",
)

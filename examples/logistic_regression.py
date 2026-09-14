# %% [markdown]
r"""
Example: Logistic Regression (Binary Outcome)
Breast Cancer Wisconsin (Diagnostic) dataset (UCI ML Repository ID=17)

Demonstrates:
- ``family="logistic"`` — explicit and auto-detected (``family="auto"``) binary logistic models
- ter Braak (1992) permutation test with deviance residuals and recentred null
- Freedman–Lane (1983) individual and joint permutation tests
- Kennedy (1995) individual and joint permutation tests
- Stochastic reconstruction in binary models: :math:`Y^* \sim \mathrm{Bernoulli}(\mathrm{clip}(\hat{\mu} + \pi(e)))`
- Four-stage confounder sieve (screen → collider → mediator → moderator) for binary classification
- Confounder-controlled permutation testing
- Execution and protocol artifacts inspection via ``print_protocol_usage_table``

Dataset
-------
569 digitized cell nuclear measurements from fine needle aspirates (FNA) of
breast masses, collected at the University of Wisconsin Hospitals.  The target
variable is ``Diagnosis``, recoded as binary: malignant (1, n = 212, 37.3% prevalence)
versus benign (0, n = 357, 62.7%).

Feature selection rationale
---------------------------
Five nuclear morphology features describe cell geometry and texture:

- **radius1**: Mean of distances from the center to points on the perimeter.
  Core indicator of nuclear enlargement in malignant cells.
- **texture1**: Standard deviation of gray-scale pixel values. Quantifies chromatin
  granularity and nuclear heterogeneity.
- **perimeter1**: Nuclear perimeter length. Strong geometric correlate of tumor mass.
- **smoothness1**: Local variation in radius lengths, reflecting irregularity in the
  nuclear envelope.
- **compactness1**: Computed as :math:`\mathrm{perimeter}^2 / \mathrm{area} - 1.0`.
  Measures nuclear contour complexity.

Methodological rationale
-------------------------
In logistic regression, classical inference relies on asymptotic Wald z-tests derived
from the inverted Fisher information matrix. In moderate samples or when covariates
exhibit near-separation, Wald tests suffer from the Hauck–Donner effect (where standard
errors are wildly inflated, falsely dampening significance). Permutation tests bypass
Fisher asymptotics entirely by evaluating exact likelihood-ratio deviance reductions
under the null.

Reference
---------
Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature
extraction for breast tumor diagnosis. *IS&T/SPIE 1993 International Symposium
on Electronic Imaging: Science and Technology*, 1905, 861–870.
"""

# %%
import numpy as np
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    identify_confounders,
    print_confounder_table,
    print_dataset_info_table,
    print_diagnostics_table,
    print_family_info_table,
    print_joint_results_table,
    print_protocol_usage_table,
    print_results_table,
    randomization_test_regression,
    resolve_family,
)

# %%
# ============================================================================
# Load data
# ============================================================================

breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features
y = breast_cancer.data.targets

# Convert target to binary: malignant (M) -> 1, benign (B) -> 0
y = (y == "M").astype(int)

selected_features = ["radius1", "texture1", "perimeter1", "smoothness1", "compactness1"]
X = X[selected_features]

print_dataset_info_table(
    name=breast_cancer.metadata.name,
    X=X,
    y=y,
    target_description="malignant (1) vs benign (0)",
    extra_stats={"Prevalence": f"{float(y.values.mean()):.2%}"},
)

# %%
# ============================================================================
# Verify resolve_family auto-detects "logistic" for binary Y
# ============================================================================

auto_family = resolve_family("auto", np.ravel(y))
logistic_family = resolve_family("logistic", np.ravel(y))

print_family_info_table(
    auto_family=auto_family,
    explicit_family=logistic_family,
)

# %%
# ============================================================================
# ter Braak (1992) — family="auto" (auto-detection)
# ============================================================================

results_ter_braak_auto = randomization_test_regression(
    X, y, method="ter_braak", family="auto"
)
print_results_table(results_ter_braak_auto)

# %%
# ============================================================================
# ter Braak (1992) — family="logistic" (explicit)
# ============================================================================

results_ter_braak = randomization_test_regression(
    X, y, method="ter_braak", family="logistic"
)
print_results_table(results_ter_braak)
print_diagnostics_table(results_ter_braak)

# %%
# ============================================================================
# Kennedy (1995) individual — family="logistic"
# ============================================================================

results_kennedy = randomization_test_regression(
    X, y, method="kennedy", confounders=[], family="logistic"
)
print_results_table(results_kennedy)
print_diagnostics_table(results_kennedy)

# %%
# ============================================================================
# Kennedy (1995) joint — family="logistic"
# ============================================================================

results_kennedy_joint = randomization_test_regression(
    X, y, method="kennedy_joint", confounders=[], family="logistic"
)
print_joint_results_table(results_kennedy_joint)

# %%
# ============================================================================
# Freedman–Lane (1983) individual — family="logistic"
# ============================================================================

results_fl = randomization_test_regression(
    X, y, method="freedman_lane", confounders=[], family="logistic"
)
print_results_table(results_fl)
print_diagnostics_table(results_fl)

# %%
# ============================================================================
# Freedman–Lane (1983) joint — family="logistic"
# ============================================================================

results_fl_joint = randomization_test_regression(
    X, y, method="freedman_lane_joint", confounders=[], family="logistic"
)
print_joint_results_table(results_fl_joint)

# %%
# ============================================================================
# Confounder identification — logistic
# ============================================================================

all_confounder_results = identify_confounders(X, y, family="logistic")
print_confounder_table(all_confounder_results)

# %%
# ============================================================================
# Kennedy with identified covariate — family="logistic"
# ============================================================================
# While the four-stage sieve revealed extensive mediation and collider structures
# among nuclear geometry metrics, epidemiological analysis often controls for
# texture variation when evaluating tumor diameter. We test 'radius1' controlling
# for 'texture1' via Kennedy exposure residualization.

results_kc = randomization_test_regression(
    X,
    y,
    method="kennedy",
    confounders=["texture1"],
    family="logistic",
)
print_results_table(results_kc)
print_diagnostics_table(results_kc)

# %%
# ============================================================================
# Execution & Protocol Artifacts
# ============================================================================
# Inspect the internal execution context from the completed test result:
# backend acceleration, batch convergence, fit metrics, and protocol properties.

print_protocol_usage_table(results_kc)

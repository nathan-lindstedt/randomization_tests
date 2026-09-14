# %% [markdown]
"""
Example: Linear Regression (Continuous Outcome)
Real Estate Valuation dataset (UCI ML Repository ID=477)

Demonstrates:
- ``family="linear"`` — explicit and auto-detected (``family="auto"``) continuous models
- ter Braak (1992) permutation test (full-model residual permutation with recentred null)
- Freedman–Lane (1983) individual and joint permutation tests (reduced-model residual permutation)
- Kennedy (1995) individual and joint permutation tests (exposure residualization)
- Four-stage confounder sieve (screen → collider → mediator → moderator)
- Confounder-controlled permutation testing
- Execution and protocol artifacts inspection via ``print_protocol_usage_table``

Dataset
-------
414 real estate transactions collected from Sindian District, New Taipei City,
Taiwan (2012–2013).  The target variable is ``Y house price of unit area``
(measured in 10,000 New Taiwan Dollar / Ping, where 1 Ping = 3.3 square meters).
The market valuation distribution is continuous and unimodal (mean ≈ 38.0,
variance ≈ 185.0), providing an ideal testbed for ordinary least squares and
nonparametric residual permutation.

Feature selection rationale
---------------------------
Six physical and geographical predictors capture primary determinants of property
value:

- **X1 transaction date**: Continuous transaction timing (e.g. 2013.250). Captures
  temporal price momentum and seasonal fluctuations.
- **X2 house age**: Structural age of the building in years (range 0–43.8). Models
  physical depreciation over time.
- **X3 distance to the nearest MRT station**: Proximity in meters to the nearest
  Mass Rapid Transit station. Critical public transportation accessibility metric.
- **X4 number of convenience stores**: Count of major convenience stores (7-Eleven,
  FamilyMart, etc.) accessible on foot within the living circle.
- **X5 latitude** & **X6 longitude**: Geographic coordinate coordinates in degrees.
  Capture spatial clustering, neighborhood desirability, and submarket premiums.

Methodological rationale
-------------------------
Standard OLS inference assumes homoscedastic, Gaussian disturbances. When property
residuals exhibit skewness, kurtosis, or spatial heteroscedasticity, classical
t-statistics can be miscalibrated. Freedman–Lane and ter Braak permutation tests
provide asymptotically exact Type I error control without parametric error assumptions,
while Kennedy's exposure residualization preserves predictor correlations.

Reference
---------
Yeh, I. C., & Hsu, T. K. (2018). Building real estate valuation models with
comparative approach through case-based reasoning. *Applied Soft Computing*,
65, 260–271.
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

real_estate_valuation = fetch_ucirepo(id=477)
X = real_estate_valuation.data.features
y = real_estate_valuation.data.targets

print_dataset_info_table(
    name=real_estate_valuation.metadata.name,
    X=X,
    y=y,
)

# %%
# ============================================================================
# Verify resolve_family auto-detects "linear" for continuous Y
# ============================================================================

auto_family = resolve_family("auto", np.ravel(y))
linear_family = resolve_family("linear", np.ravel(y))

print_family_info_table(
    auto_family=auto_family,
    explicit_family=linear_family,
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
# ter Braak (1992) — family="linear" (explicit)
# ============================================================================

results_ter_braak = randomization_test_regression(
    X, y, method="ter_braak", family="linear"
)
print_results_table(results_ter_braak)
print_diagnostics_table(results_ter_braak)

# %%
# ============================================================================
# Kennedy (1995) individual — family="linear"
# ============================================================================

results_kennedy = randomization_test_regression(
    X, y, method="kennedy", confounders=[], family="linear"
)
print_results_table(results_kennedy)
print_diagnostics_table(results_kennedy)

# %%
# ============================================================================
# Kennedy (1995) joint — family="linear"
# ============================================================================

results_kennedy_joint = randomization_test_regression(
    X, y, method="kennedy_joint", confounders=[], family="linear"
)
print_joint_results_table(results_kennedy_joint)

# %%
# ============================================================================
# Freedman–Lane (1983) individual — family="linear"
# ============================================================================

results_fl = randomization_test_regression(
    X, y, method="freedman_lane", confounders=[], family="linear"
)
print_results_table(results_fl)
print_diagnostics_table(results_fl)

# %%
# ============================================================================
# Freedman–Lane (1983) joint — family="linear"
# ============================================================================

results_fl_joint = randomization_test_regression(
    X, y, method="freedman_lane_joint", confounders=[], family="linear"
)
print_joint_results_table(results_fl_joint)

# %%
# ============================================================================
# Confounder identification
# ============================================================================

all_confounder_results = identify_confounders(X, y, family="linear")
print_confounder_table(all_confounder_results)

# %%
# ============================================================================
# Kennedy with identified confounders — family="linear"
# ============================================================================
# The confounder sieve identified that 'X1 transaction date' is confounded
# by 'X6 longitude'. We now execute a Kennedy permutation test controlling
# for 'X6 longitude' to evaluate the partial effect.

target_predictor = "X1 transaction date"
confounders = all_confounder_results[target_predictor].identified_confounders

results_kc = randomization_test_regression(
    X,
    y,
    method="kennedy",
    confounders=confounders,
    family="linear",
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

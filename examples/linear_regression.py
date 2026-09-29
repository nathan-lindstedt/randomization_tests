# %% [markdown]
"""
Example: Linear Regression (Continuous Outcome)
Dataset: Real Estate Valuation (UCI Machine Learning Repository ID=477)

Dataset Context & Theoretical Background:
    The Real Estate Valuation dataset (Yeh & Hsu, 2018) comprises 414 residential
    property transactions from Sindian District, New Taipei City, Taiwan (2012–2013).
    The response variable is `Y house price of unit area` (measured in 10,000 New
    Taiwan Dollar per Ping, where 1 Ping ≈ 3.3 square meters).

    The market price distribution is continuous, unimodal, and approximately
    symmetric (mean ≈ 37.98, variance ≈ 184.69), providing an ideal empirical
    testbed for Ordinary Least Squares (OLS) regression and non-parametric
    resampling methods. Standard OLS inference relies on the Gauss–Markov theorem
    and normal-theory assumptions (homoscedastic, normally distributed disturbances).
    In applied real estate economics, residential sale residuals often exhibit
    moderate skewness, kurtosis, spatial heteroscedasticity, and localized market
    clustering. Under such departures, asymptotic t-statistics and Wald confidence
    intervals can be miscalibrated.

    Permutation testing replaces normality assumptions with exact, finite-sample
    or asymptotically exact exchangeability properties, deriving reference null
    distributions directly from data permutations without parametric distribution
    commitments.

Features Selected for Modeling:
    - X1 transaction date: Continuous transaction timing (2012.667 to 2013.583).
      Captures macroeconomic price appreciation and seasonal housing cycles.
    - X2 house age: Age of the residential structure in years (0 to 43.8 years).
      Models physical structural depreciation over time.
    - X3 distance to the nearest MRT station: Metric distance to the nearest Mass
      Rapid Transit station. Primary spatial accessibility and transportation metric.
    - X4 number of convenience stores: Count of major convenience stores (7-Eleven,
      FamilyMart, etc.) accessible on foot within the living radius. Measures local
      commercial density and pedestrian amenity access.
    - X5 latitude & X6 longitude: Geographic coordinates in decimal degrees.
      Capture neighborhood desirability, elevation premiums, and spatial location.

Methodological Rationale for Resampling Tests:
    1. Full-Model vs. Reduced-Model Residual Permutation:
       - ter Braak (1992): Computes residuals from the single full OLS model,
         shuffles them about the full fit, and centers the permutation distribution
         at beta* - beta_hat. Highly efficient single-fit procedure.
       - Freedman & Lane (1983): Fits a separate reduced model omitting predictor j,
         shuffles the reduced residuals, and refits the full design matrix. Canonical
         gold standard for conditioning on nuisance covariates.
    2. Exposure Residualization (Kennedy 1995):
       Partials out confounders Z from the exposure X_j to produce clean exposure
       residuals, and permutes the exposure rather than the outcome. Preserves
       exact design correlations and avoids outcome reconstruction entirely.
    3. Four-Stage Confounder Sieve:
       Before hypothesis testing, the automated causal sieve partitions candidate
       covariates into true confounders (to control), colliders (do not control),
       mediators (transmit causal effect; do not control), and moderators (informational).

Demonstrates:
    - family="linear" -- explicit and auto-detected (family="auto") continuous models
    - ter Braak (1992) permutation test with recentered null distribution
    - Freedman–Lane (1983) individual and joint permutation tests
    - Kennedy (1995) individual and joint permutation tests
    - Four-stage confounder sieve (screen -> collider -> mediator -> moderator)
    - Confounder-controlled permutation testing with side-by-side tables
    - Execution and protocol artifacts inspection via print_protocol_usage_table

References:
    - Yeh, I. C., & Hsu, T. K. (2018). Building real estate valuation models with
      comparative approach through case-based reasoning. Applied Soft Computing,
      65, 260-271.
    - Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported
      significance levels. Journal of Business & Economic Statistics, 1(4), 292-298.
    - ter Braak, C. J. F. (1992). Permutation versus bootstrap significance tests
      in multiple regression and ANOVA. Handbook of Statistics, 9, 79-104.
    - Kennedy, P. E. (1995). Randomization tests in econometrics. Journal of
      Business & Economic Statistics, 13(1), 85-94.
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

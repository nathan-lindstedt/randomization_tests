# %% [markdown]
"""
Example: Linear Multilevel Regression (Continuous Outcome with Hierarchical Clustering)
Dataset: Parkinsons Telemonitoring (UCI Machine Learning Repository ID=189)

Dataset Context & Theoretical Background:
    The Parkinson's Telemonitoring dataset (Tsanas et al., 2010; Little et al.,
    2007) comprises 5,875 biomedical voice acoustic recordings from 42 patients
    with early-stage Parkinson's disease, tracked longitudinally over a 6-month
    period. The primary clinical outcome is `motor_UPDRS` (Unified Parkinson's
    Disease Rating Scale, motor impairment subscore), assessed by licensed
    medical clinicians.

    Because each patient contributes approximately 140 repeated voice recordings
    across time, the dataset embodies a classic two-level hierarchical panel
    design:
        Level 2: Patients (N = 42 clusters)
        Level 1: Longitudinal voice recordings nested within patients (n_j approx 140)

    The empirical Intraclass Correlation (ICC approx 0.925) indicates that over
    90% of total variance in motor scores is between-patient rather than
    within-patient. Standard Ordinary Least Squares (OLS) regression operates
    under the assumption of independent and identically distributed errors;
    applying OLS to this clustered design severely understates standard errors,
    producing spuriously tight confidence intervals and elevated Type I error.
    The Linear Mixed-Effects Model (LMM; Laird & Ware, 1982) models this
    hierarchical structure directly via random patient intercepts:

        Y_{ij} = alpha + X_{ij} * beta + u_j + eps_{ij},   u_j ~ N(0, tau^2),  eps_{ij} ~ N(0, sigma^2)

    Variance components (tau^2, sigma^2) are estimated via Restricted Maximum
    Likelihood (REML; Harville, 1977), which accounts for degrees of freedom
    lost to fixed effects.

Features Selected for Modeling:
    - test_time: Time elapsed since patient clinical trial recruitment (days).
      Captures the natural longitudinal progression of motor symptoms.
    - HNR: Harmonics-to-noise ratio in decibels (dB). Measures vocal fold vibration
      periodicity vs. turbulent glottal noise; lower HNR signifies vocal dysphonia.
    - RPDE: Recurrence period density entropy. A non-linear dynamical complexity
      measure quantifying the regularity and predictability of vocal tract oscillations.
    - DFA: Detrended fluctuation analysis. Quantifies the fractal self-similarity
      and scaling exponent of turbulent vocal tremor.
    - PPE: Pitch period entropy. A non-linear measure of fundamental frequency
      perturbations and impaired vocal stability.

Methodological Rationale for Resampling Tests:
    1. Restricted Block Exchangeability:
       Observations within the same patient cannot be freely shuffled across
       different patients without destroying the massive between-cluster variance
       structure. Valid permutation requires either:
       (a) Permuting whitened GLS residuals across observations (whitened LMM),
       (b) Restricted within-cluster permutation preserving individual subject blocks.
    2. Freedman-Lane and ter Braak in Multilevel Models:
       Both methods project data onto the null space of confounding covariates.
       Under block exchangeability, permuted test statistics accurately mimic the
       finite-sample null distribution while rigorously controlling for between-patient
       heterogeneity and intraclass correlation.

Demonstrates:
    - family="linear_mixed" -- linear mixed-effects model via LinearMixedFamily
    - REML variance component estimation (tau^2, sigma^2, ICC)
    - ter Braak (1992) permutation testing under cluster exchangeability
    - Kennedy (1995) and Freedman-Lane (1983) individual and joint permutation tests
    - Cluster-aware confounder sieve with cluster bootstrap
    - Random slopes for time-varying longitudinal predictors
    - Execution and protocol artifacts inspection via print_protocol_usage_table

References:
    - Tsanas, A., Little, M. A., McSharry, P. E., & Ramig, L. O. (2010). Accurate
      telemonitoring of Parkinson's disease progression by non-invasive speech tests.
      IEEE Transactions on Biomedical Engineering, 57(4), 884-893.
    - Little, M. A., McSharry, P. E., Roberts, S. J., et al. (2007). Exploiting
      nonlinear recurrence and fractal scaling properties for voice disorder detection.
      BioMedical Engineering OnLine, 6(1), 23.
    - Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal data.
      Biometrics, 38(4), 963-974.
    - Harville, D. A. (1977). Maximum likelihood approaches to variance component
      estimation and to related problems. Journal of the American Statistical
      Association, 72(358), 320-338.
    - Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported
      significance levels. Journal of Business & Economic Statistics, 1(4), 292-298.
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

ds = fetch_ucirepo(id=189)
X_all = ds.data.features
y_df = ds.data.targets[["motor_UPDRS"]]
subjects = ds.data.ids["subject#"].values

# Select key voice biomarkers
cols = ["test_time", "HNR", "RPDE", "DFA", "PPE"]
X = X_all[cols]
y = np.ravel(y_df).astype(float)

print_dataset_info_table(
    name="Parkinsons Telemonitoring",
    X=X,
    y=y,
    target_name="motor_UPDRS",
    target_description="motor subscore",
    extra_stats={"Subjects": str(len(np.unique(subjects)))},
)

# %%
# ============================================================================
# Verify resolve_family detects "linear_mixed" for continuous Y + groups
# ============================================================================

auto_family = resolve_family("linear_mixed", y)

print_family_info_table(
    explicit_family=auto_family,
)

# %%
# ============================================================================
# ter Braak (1992) — family="linear_mixed"
# ============================================================================

results_ter_braak = randomization_test_regression(
    X,
    y_df,
    method="ter_braak",
    family="linear_mixed",
    groups=subjects,
    n_randomizations=999,
    random_state=42,
)
print_results_table(results_ter_braak)
print_diagnostics_table(results_ter_braak)

# %%
# ============================================================================
# Kennedy (1995) individual — family="linear_mixed"
# ============================================================================

results_kennedy = randomization_test_regression(
    X,
    y_df,
    method="kennedy",
    family="linear_mixed",
    groups=subjects,
    confounders=[],
    n_randomizations=999,
    random_state=42,
)
print_results_table(results_kennedy)
print_diagnostics_table(results_kennedy)

# %%
# ============================================================================
# Kennedy (1995) joint — family="linear_mixed"
# ============================================================================

results_kennedy_joint = randomization_test_regression(
    X,
    y_df,
    method="kennedy_joint",
    family="linear_mixed",
    groups=subjects,
    confounders=[],
    n_randomizations=999,
    random_state=42,
)
print_joint_results_table(results_kennedy_joint)

# %%
# ============================================================================
# Freedman–Lane (1983) individual — family="linear_mixed"
# ============================================================================

results_fl = randomization_test_regression(
    X,
    y_df,
    method="freedman_lane",
    family="linear_mixed",
    groups=subjects,
    confounders=[],
    n_randomizations=999,
    random_state=42,
)
print_results_table(results_fl)
print_diagnostics_table(results_fl)

# %%
# ============================================================================
# Freedman–Lane (1983) joint — family="linear_mixed"
# ============================================================================

results_fl_joint = randomization_test_regression(
    X,
    y_df,
    method="freedman_lane_joint",
    family="linear_mixed",
    groups=subjects,
    confounders=[],
    n_randomizations=999,
    random_state=42,
)
print_joint_results_table(results_fl_joint)

# %%
# ============================================================================
# Confounder identification with cluster bootstrap
# ============================================================================

all_confounder_results = identify_confounders(
    X,
    y_df,
    family="linear",
    groups=subjects,
    random_state=42,
)
print_confounder_table(all_confounder_results)

# %%
# ============================================================================
# Kennedy with covariate control — family="linear_mixed"
# ============================================================================
# The sieve revealed that RPDE and PPE act as mutual mediators of vocal
# dysphonia. In clinical acoustic research, test_time (disease progression)
# is often evaluated while controlling for baseline vocal periodicity (HNR).
# We run a Kennedy test for 'test_time' controlling for 'HNR'.

results_kc = randomization_test_regression(
    X,
    y_df,
    method="kennedy",
    confounders=["HNR"],
    family="linear_mixed",
    groups=subjects,
    n_randomizations=999,
    random_state=42,
)
print_results_table(results_kc)
print_diagnostics_table(results_kc)

# %%
# ============================================================================
# Random slopes: test_time as random slope (disease progression over time)
# ============================================================================

results_slopes = randomization_test_regression(
    X,
    y_df,
    method="ter_braak",
    family="linear_mixed",
    groups=subjects,
    random_slopes=[0],  # test_time is column index 0
    n_randomizations=999,
    random_state=42,
)
print_results_table(
    results_slopes,
    title="Linear Mixed Model — ter Braak (1992) Test (Random Slopes)",
)

# Compare random-intercept vs random-slopes diagnostics
print_diagnostics_table(
    results_ter_braak,
    title="Permutation Diagnostics — Random Intercept",
)
print_diagnostics_table(
    results_slopes,
    title="Permutation Diagnostics — Random Slopes",
)

# %%
# ============================================================================
# Execution & Protocol Artifacts
# ============================================================================
# Inspect the internal execution context from the completed test result:
# backend acceleration, batch convergence, fit metrics, and protocol properties.

print_protocol_usage_table(results_slopes)

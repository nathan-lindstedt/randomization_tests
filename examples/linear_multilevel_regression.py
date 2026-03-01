"""
Test Case 7: Linear Multilevel Regression (Continuous Outcome, Clustered Data)
Parkinsons Telemonitoring dataset (UCI ML Repository ID=189)

Demonstrates:
- ``family="linear_mixed"`` — linear mixed-effects model
- ter Braak (1992) permutation test with within-cluster exchangeability
- Kennedy (1995) individual and joint tests
- Freedman–Lane (1983) individual and joint tests
- External validation against statsmodels MixedLM (β̂, σ², τ², ICC)
- Direct ``LinearMixedFamily`` protocol usage (calibrate / fit / predict /
  residuals / diagnostics / classical_p_values / batch_fit)
- Random slopes for time-varying effects

Dataset
-------
5,875 voice recordings from 42 patients with early-stage Parkinson's
disease.  Each patient has ~140 recordings over ~6 months.  The outcome
is ``motor_UPDRS`` (Unified Parkinson's Disease Rating Scale, motor
subscore).  The natural grouping by ``subject#`` creates a classic
two-level hierarchy:

    Level 2: Patients (n = 42)
    Level 1: Repeated voice recordings within patients (~140 each)

With ICC ≈ 0.925, most variance is between patients rather than within.
This makes the dataset ideal for demonstrating mixed-effects models:
standard (flat) regression would underestimate standard errors by
ignoring the within-patient correlation.
"""

import warnings

import numpy as np
import statsmodels.api as sm
import statsmodels.regression.mixed_linear_model as mlm
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    LinearMixedFamily,
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
    n_observations=len(y),
    n_features=X.shape[1],
    feature_names=cols,
    target_name="motor_UPDRS",
    target_description="motor subscore",
    y_range=(float(y.min()), float(y.max())),
    y_mean=float(y.mean()),
    y_var=float(y.var()),
    extra_stats={"Subjects": str(len(np.unique(subjects)))},
)

# ============================================================================
# External validation: statsmodels MixedLM
# ============================================================================

print("=" * 80)
print("External validation: statsmodels MixedLM (random intercept)")
print("=" * 80)

X_np = X.values.astype(float)
X_sm = sm.add_constant(X_np)

sm_model = mlm.MixedLM(y, X_sm, groups=subjects).fit(reml=True, disp=0)

tau2_sm = float(np.asarray(sm_model.cov_re).flat[0])
icc_sm = tau2_sm / (tau2_sm + sm_model.scale)

# ============================================================================
# Our calibration: verify β̂ and variance components match
# ============================================================================

family = LinearMixedFamily()
family_cal = family.calibrate(X_np, y, fit_intercept=True, groups=subjects)

model = family_cal.fit(X_np, y, fit_intercept=True)
beta = model.beta
tau2 = float(family_cal.re_covariances[0][0, 0])
icc_ours = tau2 / (tau2 + family_cal.sigma2)

# Validate agreement
beta_match = np.allclose(beta, sm_model.fe_params, atol=1e-3)
sigma2_match = abs(family_cal.sigma2 - sm_model.scale) < 0.01
tau2_match = abs(tau2 - tau2_sm) < 0.1
assert beta_match, f"β̂ mismatch: {beta} vs {sm_model.fe_params}"
assert sigma2_match, f"σ² mismatch: {family_cal.sigma2} vs {sm_model.scale}"
assert tau2_match, f"τ² mismatch: {tau2} vs {tau2_sm}"

# ============================================================================
# Verify resolve_family detects "linear_mixed" for continuous Y + groups
# ============================================================================

auto_family = resolve_family("linear_mixed", y)
assert auto_family.name == "linear_mixed"

print_family_info_table(
    explicit_family=auto_family,
)

# ============================================================================
# ter Braak (1992) — family="linear_mixed"
# ============================================================================

results_ter_braak = permutation_test_regression(
    X,
    y_df,
    method="ter_braak",
    family="linear_mixed",
    groups=subjects,
    n_permutations=999,
    random_state=42,
)
print_results_table(
    results_ter_braak,
    title="ter Braak (1992) Permutation Test (family='linear_mixed')",
)
print_diagnostics_table(
    results_ter_braak,
    title="ter Braak (1992) Extended Diagnostics (family='linear_mixed')",
)

# ============================================================================
# Kennedy (1995) individual — family="linear_mixed"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_kennedy = permutation_test_regression(
        X,
        y_df,
        method="kennedy",
        family="linear_mixed",
        groups=subjects,
        confounders=[],
        n_permutations=999,
        random_state=42,
    )
print_results_table(
    results_kennedy,
    title="Kennedy (1995) Individual Permutation Test (family='linear_mixed')",
)
print_diagnostics_table(
    results_kennedy,
    title="Kennedy (1995) Individual Diagnostics (family='linear_mixed')",
)

# ============================================================================
# Kennedy (1995) joint — family="linear_mixed"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_kennedy_joint = permutation_test_regression(
        X,
        y_df,
        method="kennedy_joint",
        family="linear_mixed",
        groups=subjects,
        confounders=[],
        n_permutations=999,
        random_state=42,
    )
print_joint_results_table(
    results_kennedy_joint,
    title="Kennedy (1995) Joint Permutation Test (family='linear_mixed')",
)

# ============================================================================
# Freedman–Lane (1983) individual — family="linear_mixed"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_fl = permutation_test_regression(
        X,
        y_df,
        method="freedman_lane",
        family="linear_mixed",
        groups=subjects,
        confounders=[],
        n_permutations=999,
        random_state=42,
    )
print_results_table(
    results_fl,
    title="Freedman–Lane (1983) Individual Permutation Test (family='linear_mixed')",
)
print_diagnostics_table(
    results_fl,
    title="Freedman–Lane (1983) Individual Diagnostics (family='linear_mixed')",
)

# ============================================================================
# Freedman–Lane (1983) joint — family="linear_mixed"
# ============================================================================

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_fl_joint = permutation_test_regression(
        X,
        y_df,
        method="freedman_lane_joint",
        family="linear_mixed",
        groups=subjects,
        confounders=[],
        n_permutations=999,
        random_state=42,
    )
print_joint_results_table(
    results_fl_joint,
    title="Freedman–Lane (1983) Joint Permutation Test (family='linear_mixed')",
)

# ============================================================================
# Confounder identification with cluster bootstrap
# ============================================================================

all_confounder_results = {}
for predictor in X.columns:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        all_confounder_results[predictor] = identify_confounders(
            X,
            y_df,
            predictor=predictor,
            family="linear",
            groups=subjects,
            random_state=42,
        )

print_confounder_table(
    all_confounder_results,
    title="Confounder Identification for All Predictors (Linear Mixed)",
)

# Extract confounders using ConfounderAnalysisResult field access
predictors_with_confounders = {
    pred: res.identified_confounders
    for pred, res in all_confounder_results.items()
    if res.identified_confounders
}

# ============================================================================
# Kennedy with identified confounders — family="linear_mixed"
# ============================================================================

if predictors_with_confounders:
    example_predictor = list(predictors_with_confounders.keys())[0]
    example_confounders = predictors_with_confounders[example_predictor]

    results_kc = permutation_test_regression(
        X,
        y_df,
        method="kennedy",
        confounders=example_confounders,
        family="linear_mixed",
        groups=subjects,
        n_permutations=999,
        random_state=42,
    )
    print_results_table(
        results_kc,
        title=(
            f"Kennedy (1995) for '{example_predictor}' "
            f"(controlling for {', '.join(example_confounders)}) "
            f"(family='linear_mixed')"
        ),
    )
    print_diagnostics_table(
        results_kc,
        title=(
            f"Kennedy (1995) Diagnostics for '{example_predictor}' "
            f"(family='linear_mixed')"
        ),
    )

# ============================================================================
# Direct LinearMixedFamily protocol usage
# ============================================================================
# The ModelFamily protocol encapsulates every model-specific operation —
# fitting, prediction, residual extraction, Y-reconstruction, batch
# fitting, diagnostics, and classical p-values.  Below we exercise
# each method directly for LinearMixedFamily.

# validate_y — should pass without error for continuous Y
family_cal.validate_y(y)

# fit / predict / coefs / residuals
preds = family_cal.predict(model, X_np)
coefs = family_cal.coefs(model)
resids = family_cal.residuals(model, X_np, y)

# fit_metric (RSS)
rss = family_cal.fit_metric(y, preds)

# reconstruct_y — additive: ŷ + π(e)
rng = np.random.default_rng(42)
perm_resids = rng.permutation(resids)
y_star = family_cal.reconstruct_y(preds[np.newaxis, :], perm_resids[np.newaxis, :], rng)

# batch_fit — fit LMM on B permuted Y vectors at once
n_batch = 50
perm_indices = np.array([rng.permutation(len(y)) for _ in range(n_batch)])
Y_matrix = y[perm_indices]  # (B, n)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    batch_coefs = family_cal.batch_fit(X_np, Y_matrix, fit_intercept=True)
n_nan = int(np.sum(np.any(np.isnan(batch_coefs), axis=1)))

# diagnostics — marginal/conditional R², ICC, variance components
diag = family_cal.diagnostics(X_np, y, fit_intercept=True)

# classical_p_values — Wald t-test from LMM fixed effects
p_classical = family_cal.classical_p_values(X_np, y, fit_intercept=True)

# exchangeability_cells — within-cluster exchangeability
cells = family_cal.exchangeability_cells(X_np, y)

print_protocol_usage_table(
    results_ter_braak,
    title="Direct LinearMixedFamily Protocol Usage",
)

# ============================================================================
# Random slopes: test_time as random slope (disease progression over time)
# ============================================================================

results_slopes = permutation_test_regression(
    X,
    y_df,
    method="ter_braak",
    family="linear_mixed",
    groups=subjects,
    random_slopes=[0],  # test_time is column index 0
    n_permutations=999,
    random_state=42,
)
print_results_table(
    results_slopes,
    title="ter Braak (1992) with Random Slopes (family='linear_mixed')",
)

# Compare random-intercept vs random-slopes diagnostics
print_diagnostics_table(
    results_ter_braak,
    title="Random-Intercept Diagnostics (family='linear_mixed')",
)
print_diagnostics_table(
    results_slopes,
    title="Random-Slopes Diagnostics (family='linear_mixed')",
)

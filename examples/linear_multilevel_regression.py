"""
Test Case 7: Linear Multilevel Regression (Continuous Outcome, Clustered Data)
Parkinsons Telemonitoring dataset (UCI ML Repository ID=189)

Demonstrates:
- ``family="linear_mixed"`` — linear mixed-effects model
- ter Braak (1992) permutation test with within-cluster exchangeability
- Freedman–Lane (1983) permutation test with within-cluster exchangeability
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

import textwrap
import warnings

import numpy as np
import statsmodels.api as sm
import statsmodels.regression.mixed_linear_model as mlm
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    LinearMixedFamily,
    permutation_test_regression,
    print_diagnostics_table,
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

print("Dataset: Parkinsons Telemonitoring (UCI ID=189)")
print(f"  Observations:  {len(y)}")
print(f"  Subjects:      {len(np.unique(subjects))}")
print(f"  Features:      {X.shape[1]} ({', '.join(cols)})")
print(f"  Outcome:       motor_UPDRS (range {y.min():.1f}–{y.max():.1f})")
print()

# ============================================================================
# External validation: statsmodels MixedLM
# ============================================================================

print("=" * 80)
print("External validation: statsmodels MixedLM (random intercept)")
print("=" * 80)

X_np = X.values.astype(float)
X_sm = sm.add_constant(X_np)

sm_model = mlm.MixedLM(y, X_sm, groups=subjects).fit(reml=True, disp=0)

print(f"\nstatsmodels β̂ (intercept + {len(cols)} slopes):")
for i, name in enumerate(["intercept"] + cols):
    print(f"  {name:>12s}: {sm_model.fe_params[i]:>10.4f}")
print(f"  σ² (residual): {sm_model.scale:.4f}")
tau2_sm = float(np.asarray(sm_model.cov_re).flat[0])
print(f"  τ² (intercept): {tau2_sm:.4f}")
icc_sm = tau2_sm / (tau2_sm + sm_model.scale)
print(f"  ICC:            {icc_sm:.4f}")
print(f"  AIC:            {sm_model.aic:.4f}")
print(f"  BIC:            {sm_model.bic:.4f}")

# ============================================================================
# Our calibration: verify β̂ and variance components match
# ============================================================================

print(f"\n{'=' * 80}")
print("Our REML calibration (random intercept)")
print("=" * 80)

family = LinearMixedFamily()
family_cal = family.calibrate(X_np, y, fit_intercept=True, groups=subjects)

model = family_cal.fit(X_np, y, fit_intercept=True)
beta = model.beta
print(f"\nOur β̂ (intercept + {len(cols)} slopes):")
for i, name in enumerate(["intercept"] + cols):
    print(f"  {name:>12s}: {beta[i]:>10.4f}")
print(f"  σ² (residual): {family_cal.sigma2:.4f}")
tau2 = float(family_cal.re_covariances[0][0, 0])
print(f"  τ² (intercept): {tau2:.4f}")
icc_ours = tau2 / (tau2 + family_cal.sigma2)
print(f"  ICC:            {icc_ours:.4f}")
print(f"  REML converged: {family_cal.converged}")

# Validate agreement
print("\nValidation (our vs statsmodels):")
beta_match = np.allclose(beta, sm_model.fe_params, atol=1e-3)
sigma2_match = abs(family_cal.sigma2 - sm_model.scale) < 0.01
tau2_match = abs(tau2 - tau2_sm) < 0.1
print(f"  β̂ agree (atol=1e-3):  {beta_match}")
print(f"  σ² agree (atol=0.01): {sigma2_match}")
print(f"  τ² agree (atol=0.1):  {tau2_match}")
assert beta_match, f"β̂ mismatch: {beta} vs {sm_model.fe_params}"
assert sigma2_match, f"σ² mismatch: {family_cal.sigma2} vs {sm_model.scale}"
assert tau2_match, f"τ² mismatch: {tau2} vs {tau2_sm}"

# ============================================================================
# Verify resolve_family detects "linear_mixed" for continuous Y + groups
# ============================================================================

auto_family = resolve_family("linear_mixed", y)
assert auto_family.name == "linear_mixed"
print(f"\nresolve_family('linear_mixed', y) → {auto_family.name!r}")

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

# ============================================================================
# Direct LinearMixedFamily protocol usage
# ============================================================================

W = 80

# ── Compute all protocol outputs ─────────────────────────────────────────── #

family_cal.validate_y(y)
preds = family_cal.predict(model, X_np)
coefs = family_cal.coefs(model)
resids = family_cal.residuals(model, X_np, y)
rss = family_cal.fit_metric(y, preds)

rng = np.random.default_rng(42)
perm_resids = rng.permutation(resids)
y_star = family_cal.reconstruct_y(preds, perm_resids, rng)

n_batch = 100
perm_indices = np.array([rng.permutation(len(y)) for _ in range(n_batch)])
Y_matrix = y[perm_indices]  # (B, n)
batch_coefs = family_cal.batch_fit(X_np, Y_matrix, fit_intercept=True)

diag = family_cal.diagnostics(X_np, y, fit_intercept=True)
p_classical = family_cal.classical_p_values(X_np, y, fit_intercept=True)
cells = family_cal.exchangeability_cells(X_np, y)

# ── Render table ──────────────────────────────────────────────────────────── #

lw = 38  # label column width
vw = W - lw - 4  # value column (minus indent + gap)

direct_perm = family_cal.direct_permutation
direct_label = "Yes" if direct_perm else "No"

print()
print("=" * W)
title = "LinearMixedFamily: ModelFamily Protocol Usage"
print(f"{title:^{W}}")
print("=" * W)

# ── Model identity ────────────────────────────────────────────────────────── #
print(f"  {'Model Family:':<{lw}}{family_cal.name:>{vw}}")
print(f"  {'Residual Type:':<{lw}}{family_cal.residual_type:>{vw}}")
print(f"  {'Direct Permutation:':<{lw}}{direct_label:>{vw}}")
print(f"  {'Outcome Validation:':<{lw}}{'Passed':>{vw}}")

# ── Coefficient estimates ─────────────────────────────────────────────────── #
print("-" * W)
print(f"  {'Coefficient Estimates (Fixed Effects)':^{W - 4}}")
print("-" * W)

# Column layout: 2-indent + feature(20) + coef(9) + p-value(47) = 78+2=80
fc = 20  # feature column
cw = 9  # coefficient column
pw = W - 4 - fc - cw  # p-value column
print(f"  {'Feature':<{fc}}{'Coef':>{cw}}{'P>|t| (Asy)':>{pw}}")
print("  " + "-" * (W - 4))
for i, name in enumerate(cols):
    c_str = f"{coefs[i]:>{cw}.4f}"
    p_str = f"{p_classical[i]:>{pw}.6f}" if i < len(p_classical) else ""
    print(f"  {name:<{fc}}{c_str}{p_str}")

# ── Residual diagnostics ──────────────────────────────────────────────────── #
print("-" * W)
print(f"  {'Residual & Fit Diagnostics':^{W - 4}}")
print("-" * W)
print(f"  {'Mean |Residual|:':<{lw}}{np.mean(np.abs(resids)):>{vw}.4f}")
print(f"  {'Residual Sum of Squares:':<{lw}}{rss:>{vw}.2f}")
r2m_label = "Marginal R\u00b2:"
r2c_label = "Conditional R\u00b2:"
print(f"  {r2m_label:<{lw}}{diag['r_squared_marginal']:>{vw}}")
print(f"  {r2c_label:<{lw}}{diag['r_squared_conditional']:>{vw}}")
print(f"  {'ICC:':<{lw}}{diag['icc']:>{vw}}")

# ── Permutation infrastructure ────────────────────────────────────────────── #
print("-" * W)
print(f"  {'Permutation Infrastructure':^{W - 4}}")
print("-" * W)
print(
    f"  {'Reconstructed Y:':<{lw}}"
    f"{'n=' + str(y_star.shape[0]) + ', mean=' + f'{np.mean(y_star):.4f}':>{vw}}"
)
print(
    f"  {'Batch Refit (GLS):':<{lw}}"
    f"{'B=' + str(n_batch) + ', p=' + str(X_np.shape[1]):>{vw}}"
)
n_cells = len(np.unique(cells))
print(
    f"  {'Exchangeability Cells:':<{lw}}"
    f"{str(n_cells) + ' groups, ' + str(len(y)) + ' obs':>{vw}}"
)

# ── Explanatory note on direct permutation ────────────────────────────────── #
print("-" * W)
print("Notes")
print("-" * W)
if not direct_perm:
    note = (
        "No Direct Permutation: Observations are not exchangeable "
        "under the null because the mixed-effects error structure "
        "induces within-cluster correlation. Permutation tests must "
        "therefore operate on residuals (e.g., ter Braak 1992 or "
        "Freedman\u2013Lane 1983) rather than permuting raw Y values. "
        "This preserves the covariance structure while still "
        "generating a valid null distribution for each fixed-effect "
        "coefficient."
    )
else:
    note = (
        "Direct Permutation = Yes: Observations are exchangeable "
        "under the null hypothesis, so the permutation test shuffles "
        "raw outcome values (or class labels) across observations. "
        "This is appropriate when the model assumes independent, "
        "identically distributed errors and no hierarchical grouping "
        "structure needs to be respected."
    )
for line in textwrap.wrap(f"  {note}", width=W - 2, subsequent_indent="  "):
    print(line)
print("=" * W)
print()

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
diag_slopes = results_slopes.family.diagnostics(X_np, y, fit_intercept=True)
print("\nRandom-intercept model:")
print(f"  Marginal R²:     {diag['r_squared_marginal']}")
print(f"  Conditional R²:  {diag['r_squared_conditional']}")
print(f"  ICC:             {diag['icc']}")
print(f"  σ²:              {diag['sigma2']}")

print("\nRandom-slopes model (test_time):")
print(f"  Marginal R²:     {diag_slopes['r_squared_marginal']}")
print(f"  Conditional R²:  {diag_slopes['r_squared_conditional']}")
print(f"  ICC:             {diag_slopes['icc']}")
print(f"  σ²:              {diag_slopes['sigma2']}")
vc = diag_slopes["variance_components"]
for factor in vc.get("factors", []):
    k = factor["index"]
    d_k = factor["d_k"]
    print(f"  τ² (intercept):  {factor['intercept_var']:.4f}")
    if d_k > 1:
        for s_idx, sv in enumerate(factor.get("slope_vars", [])):
            print(f"  τ² (slope {s_idx}):    {sv:.4f}")
        for corr in factor.get("correlations", []):
            print(f"  ρ ({corr['label']}):  {corr['value']:.4f}")

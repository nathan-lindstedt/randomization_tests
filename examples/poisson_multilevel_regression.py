"""
Test Case 9: Poisson Multilevel Regression (Count Outcome, Clustered Data)
SUPPORT2 dataset (UCI ML Repository ID=880)

Demonstrates:
- ``family="poisson_mixed"`` — Poisson mixed-effects model
- Score projection permutation test (individual)
- Four-stage confounder sieve with cluster bootstrap (``groups=``)
- Score-with-confounders permutation test
- Direct ``PoissonMixedFamily`` protocol usage (calibrate / fit /
  predict / residuals / diagnostics / classical_p_values /
  score_project)

**Why ``method='score'``?**

GLMM families (logistic_mixed, poisson_mixed) do not support
``batch_fit()`` — each permutation would require iterative PQL/REML,
which is prohibitively expensive.  The score projection strategy
computes permuted test statistics via a single matrix-vector product,
making it orders of magnitude faster while remaining asymptotically
equivalent.

Dataset
-------
9,105 seriously-ill hospitalised patients from five U.S. medical
centres (Study to Understand Prognoses and Preferences for Outcomes
and Risks of Treatments, Phase 2).  The outcome is ``num.co``
(number of comorbidities, 0–9), which exhibits near-perfect Poisson
equi-dispersion (variance / mean ≈ 0.97).

The natural grouping by ``dzgroup`` (disease group) creates a
two-level hierarchy with 8 clusters:

    Level 2: Disease groups (n = 8)
        ARF/MOSF w/Sepsis, CHF, COPD, Lung Cancer,
        MOSF w/Malig, Coma, Colon Cancer, Cirrhosis
    Level 1: Patients within disease groups

Different disease groups have systematically different comorbidity
burdens (e.g. MOSF w/Malig vs. Coma), making the random intercept
clinically meaningful.
"""

import warnings

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    PoissonMixedFamily,
    identify_confounders,
    permutation_test_regression,
    print_confounder_table,
    print_dataset_info_table,
    print_diagnostics_table,
    print_family_info_table,
    print_protocol_usage_table,
    print_results_table,
    resolve_family,
)

# ============================================================================
# Load data
# ============================================================================

support2 = fetch_ucirepo(id=880)
X_all = support2.data.features.copy()

# Target: comorbidity count (stored in features, not targets)
y_series = X_all["num.co"].astype(float).astype(int)

# Grouping variable: disease group
dz_groups = X_all["dzgroup"].copy()
group_labels, group_uniques = pd.factorize(dz_groups)

# Select continuous clinical predictors with <1% missing.
# age       — patient age (years)
# meanbp    — mean arterial blood pressure (mmHg)
# hrt       — heart rate (bpm)
# resp      — respiratory rate (breaths/min)
# temp      — body temperature (°C)
feature_cols = ["age", "meanbp", "hrt", "resp", "temp"]
X_raw = X_all[feature_cols].copy()

# Drop any rows with NaN in features or target (< 0.01% for these cols)
valid = X_raw.notna().all(axis=1) & y_series.notna()
X_raw = X_raw[valid].reset_index(drop=True)
y_series = y_series[valid].reset_index(drop=True)
group_labels = group_labels[valid.values]

# Subsample to 500 rows for example runtime.
rng = np.random.default_rng(42)
sel = rng.choice(len(X_raw), size=500, replace=False)
X = X_raw.iloc[sel].reset_index(drop=True).astype(float)
y = pd.DataFrame(y_series.iloc[sel].reset_index(drop=True), columns=["num.co"])
groups = group_labels[sel]

y_np = np.ravel(y).astype(float)

print_dataset_info_table(
    name="SUPPORT2 (Comorbidities)",
    n_observations=len(y),
    n_features=X.shape[1],
    feature_names=list(X.columns),
    target_name="num.co",
    target_description="number of comorbidities (count)",
    y_range=(int(y.values.min()), int(y.values.max())),
    y_mean=float(y.values.mean()),
    y_var=float(y.values.var()),
    extra_stats={
        "Disease groups": str(len(np.unique(groups))),
        "Var / Mean": f"{float(y.values.var()) / float(y.values.mean()):.3f}",
    },
)

# ============================================================================
# Verify resolve_family detects "poisson_mixed"
# ============================================================================

auto_family = resolve_family("poisson_mixed", y_np)
assert auto_family.name == "poisson_mixed"

print_family_info_table(
    explicit_family=auto_family,
)

# ============================================================================
# Score individual — family="poisson_mixed"
# ============================================================================
# Score projection computes permuted test statistics via a single
# matrix-vector product — orders of magnitude faster than full IRLS
# refitting for each permutation.

results_score = permutation_test_regression(
    X,
    y,
    method="score",
    family="poisson_mixed",
    groups=groups,
    n_permutations=999,
    random_state=42,
)
print_results_table(
    results_score,
    title="Score Individual Permutation Test (family='poisson_mixed')",
)
print_diagnostics_table(
    results_score,
    title="Score Individual Diagnostics (family='poisson_mixed')",
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
            y,
            predictor=predictor,
            family="poisson",
            groups=groups,
            random_state=42,
        )

print_confounder_table(
    all_confounder_results,
    title="Confounder Identification for All Predictors (Poisson Mixed)",
)

# Extract confounders using ConfounderAnalysisResult field access
predictors_with_confounders = {
    pred: res.identified_confounders
    for pred, res in all_confounder_results.items()
    if res.identified_confounders
}

# ============================================================================
# Score with identified confounders — family="poisson_mixed"
# ============================================================================

if predictors_with_confounders:
    example_predictor = list(predictors_with_confounders.keys())[0]
    example_confounders = predictors_with_confounders[example_predictor]

    results_sc = permutation_test_regression(
        X,
        y,
        method="score",
        confounders=example_confounders,
        family="poisson_mixed",
        groups=groups,
        n_permutations=999,
        random_state=42,
    )
    print_results_table(
        results_sc,
        title=(
            f"Score for '{example_predictor}' "
            f"(controlling for {', '.join(example_confounders)}) "
            f"(family='poisson_mixed')"
        ),
    )
    print_diagnostics_table(
        results_sc,
        title=(f"Score Diagnostics for '{example_predictor}' (family='poisson_mixed')"),
    )

# ============================================================================
# Direct PoissonMixedFamily protocol usage
# ============================================================================
# The ModelFamily protocol encapsulates every model-specific operation —
# fitting, prediction, residual extraction, Y-reconstruction,
# diagnostics, and classical p-values.  Below we exercise each method
# directly for PoissonMixedFamily.
#
# Note: batch_fit() raises NotImplementedError for GLMM families —
# the score projection strategy is used instead for permutation tests.

family = PoissonMixedFamily()
X_np = X.values.astype(float)

# validate_y — should pass for non-negative integer counts
family.validate_y(y_np)

# calibrate — estimate variance components via PQL/REML
family_cal = family.calibrate(X_np, y_np, fit_intercept=True, groups=groups)

# fit / predict / coefs / residuals
model = family_cal.fit(X_np, y_np, fit_intercept=True)
preds = family_cal.predict(model, X_np)
coefs = family_cal.coefs(model)
resids = family_cal.residuals(model, X_np, y_np)

# fit_metric (deviance)
deviance = family_cal.fit_metric(y_np, preds)

# reconstruct_y — Poisson sampling (stochastic!)
rng = np.random.default_rng(42)
perm_resids = rng.permutation(resids)
y_star = family_cal.reconstruct_y(preds[np.newaxis, :], perm_resids[np.newaxis, :], rng)

# batch_fit — not supported for GLMM families (use score projection)
try:
    n_batch = 50
    perm_indices = np.array([rng.permutation(len(y_np)) for _ in range(n_batch)])
    Y_matrix = y_np[perm_indices]
    family_cal.batch_fit(X_np, Y_matrix, fit_intercept=True)
except NotImplementedError:
    pass  # Expected: GLMM requires method='score'

# diagnostics — Poisson GLMM: deviance, dispersion, ICC, variance components
diag = family_cal.diagnostics(X_np, y_np, fit_intercept=True)
if diag["dispersion"] > 1.5:
    dispersion_status = (
        "⚠ OVERDISPERSION DETECTED — CONSIDER family='negative_binomial'"
    )
else:
    dispersion_status = "✓ NO OVERDISPERSION (GOOD POISSON FIT)"

# classical_p_values — Wald z-test from GLMM fixed effects
p_classical = family_cal.classical_p_values(X_np, y_np, fit_intercept=True)

# exchangeability_cells — within-cluster exchangeability
cells = family_cal.exchangeability_cells(X_np, y_np)

print_protocol_usage_table(
    results_score,
    title="Direct PoissonMixedFamily Protocol Usage",
)

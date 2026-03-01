"""
Test Case 8: Logistic Multilevel Regression (Binary Outcome, Clustered Data)
Adult Census Income dataset (UCI ML Repository ID=2)

Demonstrates:
- ``family="logistic_mixed"`` — logistic mixed-effects model
- Score projection permutation test (individual)
- Four-stage confounder sieve with cluster bootstrap (``groups=``)
- Score-with-confounders permutation test
- Direct ``LogisticMixedFamily`` protocol usage (calibrate / fit /
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
48,842 records from the 1994 U.S. Census Bureau database.  The outcome
is binary: income > $50 K vs. ≤ $50 K.  The natural grouping by
``occupation`` (14 occupational categories after removing unknowns)
creates a two-level hierarchy:

    Level 2: Occupations (n = 14)
    Level 1: Individuals within occupations (~350 each after subsampling)

The occupation-level intercept variance captures between-occupation
differences in base income probability (e.g., "Exec-managerial" vs.
"Handlers-cleaners"), yielding a meaningful logistic random intercept.
"""

import warnings

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    LogisticMixedFamily,
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

adult = fetch_ucirepo(id=2)
X_all = adult.data.features.copy()
y_raw = adult.data.targets.iloc[:, 0].copy()

# Clean target: strip whitespace and trailing period
y_raw = y_raw.str.strip().str.rstrip(".")
y_binary = (y_raw == ">50K").astype(int)

# Extract occupation as grouping variable
occupations = X_all["occupation"].copy()

# Drop rows with unknown occupation ("?") or Armed-Forces (n=15)
keep_mask = ~occupations.isin(["?", "Armed-Forces"])
X_all = X_all[keep_mask].reset_index(drop=True)
y_binary = y_binary[keep_mask].reset_index(drop=True)
occupations = occupations[keep_mask].reset_index(drop=True)

# Select continuous features
X = pd.DataFrame(
    {
        "age": X_all["age"].astype(float),
        "education_num": X_all["education-num"].astype(float),
        "hours_per_week": X_all["hours-per-week"].astype(float),
        "capital_gain": X_all["capital-gain"].astype(float),
        "capital_loss": X_all["capital-loss"].astype(float),
    }
)
y = y_binary.copy()

# Encode groups as integer labels
group_labels, group_uniques = pd.factorize(occupations)
groups = group_labels

# Subsample to 500 rows (stratified by outcome) for example runtime.
rng = np.random.default_rng(42)
pos_idx = np.where(y.values == 1)[0]
neg_idx = np.where(y.values == 0)[0]
n_pos = min(120, len(pos_idx))  # ~24% prevalence preserved
n_neg = 500 - n_pos
sel = np.concatenate(
    [
        rng.choice(pos_idx, size=n_pos, replace=False),
        rng.choice(neg_idx, size=n_neg, replace=False),
    ]
)
rng.shuffle(sel)

X = X.iloc[sel].reset_index(drop=True)
y = pd.DataFrame(y.iloc[sel].reset_index(drop=True), columns=["income"])
groups = groups[sel]

print_dataset_info_table(
    name="Adult Census Income",
    n_observations=len(y),
    n_features=X.shape[1],
    feature_names=list(X.columns),
    target_name="income",
    target_description=">50K (1) vs <=50K (0)",
    y_range=(int(y.values.min()), int(y.values.max())),
    y_mean=float(y.values.mean()),
    extra_stats={
        "Occupations": str(len(np.unique(groups))),
        "Prevalence": f"{float(y.values.mean()):.2%}",
    },
)

# ============================================================================
# Verify resolve_family detects "logistic_mixed"
# ============================================================================

auto_family = resolve_family("logistic_mixed", np.ravel(y))
assert auto_family.name == "logistic_mixed"

print_family_info_table(
    explicit_family=auto_family,
)

# ============================================================================
# Score individual — family="logistic_mixed"
# ============================================================================
# Score projection computes permuted test statistics via a single
# matrix-vector product — orders of magnitude faster than full IRLS
# refitting for each permutation.

results_score = permutation_test_regression(
    X,
    y,
    method="score",
    family="logistic_mixed",
    groups=groups,
    n_permutations=999,
    random_state=42,
)
print_results_table(
    results_score,
    title="Score Individual Permutation Test (family='logistic_mixed')",
)
print_diagnostics_table(
    results_score,
    title="Score Individual Diagnostics (family='logistic_mixed')",
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
            family="logistic",
            groups=groups,
            random_state=42,
        )

print_confounder_table(
    all_confounder_results,
    title="Confounder Identification for All Predictors (Logistic Mixed)",
)

# Extract confounders using ConfounderAnalysisResult field access
predictors_with_confounders = {
    pred: res.identified_confounders
    for pred, res in all_confounder_results.items()
    if res.identified_confounders
}

# ============================================================================
# Score with identified confounders — family="logistic_mixed"
# ============================================================================

if predictors_with_confounders:
    example_predictor = list(predictors_with_confounders.keys())[0]
    example_confounders = predictors_with_confounders[example_predictor]

    results_sc = permutation_test_regression(
        X,
        y,
        method="score",
        confounders=example_confounders,
        family="logistic_mixed",
        groups=groups,
        n_permutations=999,
        random_state=42,
    )
    print_results_table(
        results_sc,
        title=(
            f"Score for '{example_predictor}' "
            f"(controlling for {', '.join(example_confounders)}) "
            f"(family='logistic_mixed')"
        ),
    )
    print_diagnostics_table(
        results_sc,
        title=(
            f"Score Diagnostics for '{example_predictor}' (family='logistic_mixed')"
        ),
    )

# ============================================================================
# Direct LogisticMixedFamily protocol usage
# ============================================================================
# The ModelFamily protocol encapsulates every model-specific operation —
# fitting, prediction, residual extraction, Y-reconstruction,
# diagnostics, and classical p-values.  Below we exercise each method
# directly for LogisticMixedFamily.
#
# Note: batch_fit() raises NotImplementedError for GLMM families —
# the score projection strategy is used instead for permutation tests.

family = LogisticMixedFamily()
X_np = X.values.astype(float)
y_np = np.ravel(y).astype(float)

# validate_y — should pass for binary {0, 1}
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

# reconstruct_y — clip + Bernoulli sampling (stochastic!)
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

# diagnostics — deviance, ICC (latent scale), variance components
diag = family_cal.diagnostics(X_np, y_np, fit_intercept=True)

# classical_p_values — Wald z-test from GLMM fixed effects
p_classical = family_cal.classical_p_values(X_np, y_np, fit_intercept=True)

# exchangeability_cells — within-cluster exchangeability
cells = family_cal.exchangeability_cells(X_np, y_np)

print_protocol_usage_table(
    results_score,
    title="Direct LogisticMixedFamily Protocol Usage",
)

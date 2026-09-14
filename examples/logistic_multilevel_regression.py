# %% [markdown]
"""
Example: Logistic Multilevel Regression (Binary Outcome with Hierarchical Clustering)
Dataset: Adult Census Income (UCI Machine Learning Repository ID=2)

Dataset Context & Theoretical Background:
    The Adult Census Income dataset (Kohavi, 1996), extracted from the 1994
    Current Population Survey (CPS) by the U.S. Census Bureau, is an established
    benchmark for binary classification and socio-economic econometric analysis.
    The prediction task models whether an individual's annual income exceeds
    $50,000 (encoded as binary Y in {0, 1}).

    Individual workers operate within distinct occupational structures (e.g.,
    Executive/Managerial, Specialized Professional, Administrative Support,
    Craft/Repair, Handlers/Cleaners, Sales). Occupational sectors establish
    institutional pay grades, union wage floors, and baseline productivity
    differentials that induce significant between-occupation variance.

    Fitting a standard single-level logistic regression assumes all workers are
    independent, ignoring the shared occupational random effect. This assumption
    artificially deflates standard errors for socio-demographic covariates and
    distorts statistical inference. The Logistic Generalized Linear Mixed Model
    (Logistic GLMM; Breslow & Clayton, 1993) parameterizes occupational clustering
    via a latent random intercept:

        logit(P(Y_{ij} = 1 | X_{ij}, u_j)) = alpha + X_{ij} * beta + u_j,  u_j ~ N(0, sigma_u^2)

    On the latent logistic scale, the residual error variance is fixed at
    pi^2 / 3 approx 3.29, yielding a well-defined Intraclass Correlation (ICC):

        ICC = sigma_u^2 / (sigma_u^2 + pi^2 / 3)

Features Selected for Modeling:
    - age: Worker age in years. Captures career lifecycle progression, experience
      accumulation, and seniority premiums.
    - education_num: Continuous educational attainment index (ranging from 1 for
      early primary school to 16 for Doctorate). Strongest structural determinant
      of human capital.
    - hours_per_week: Reported usual working hours per week. Measures labor
      supply and overtime intensity.
    - capital_gain: Annual recorded capital gains from asset and equity sales ($),
      reflecting non-labor asset wealth.
    - capital_loss: Annual recorded capital losses from investment assets ($).

Methodological Rationale for Score Projection Permutation:
    1. Score Projection for Binary GLMMs:
       Refitting non-linear logistic GLMMs via Penalized Quasi-Likelihood (PQL)
       or Laplace approximation across B = 999 permutation iterations is
       prohibitively slow. The score test strategy (Rao, 1948; Commenges, 2003)
       evaluates the gradient of the log-likelihood (the score vector) under the
       null model. Invariance under cluster-preserving permutations is assessed
       in closed form via matrix-vector projections.
    2. Cluster-Aware Confounder Control:
       Covariate selection via `identify_confounders()` accounts for grouping
       using a cluster bootstrap, ensuring that candidate confounder selection
       respects the multi-level dependency structure.

Demonstrates:
    - family="logistic_mixed" -- binary logistic GLMM via LogisticMixedFamily
    - Efficient score projection permutation testing for GLMMs
    - Automated cluster-aware confounder sieve with cluster bootstrap (groups=)
    - Confounder-adjusted score projection testing
    - Execution and protocol artifacts inspection via print_protocol_usage_table

References:
    - Kohavi, R. (1996). Scaling up the accuracy of naive-bayes classifiers:
      A decision-tree hybrid. In Proceedings of the Second International
      Conference on Knowledge Discovery and Data Mining (KDD-96), 202-207.
    - Breslow, N. E., & Clayton, D. G. (1993). Approximate inference in
      generalized linear mixed models. Journal of the American Statistical
      Association, 88(421), 9-25.
    - Rao, C. R. (1948). Large sample tests of statistical hypotheses concerning
      several parameters with applications to problems of estimation. Mathematical
      Proceedings of the Cambridge Philosophical Society, 44(1), 50-57.
    - Commenges, D. (2003). Transformations which preserve exchangeability and
      randomization tests. Statistics & Probability Letters, 63(3), 277-285.
"""

# %%
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    identify_confounders,
    print_confounder_table,
    print_dataset_info_table,
    print_diagnostics_table,
    print_family_info_table,
    print_protocol_usage_table,
    print_results_table,
    randomization_test_regression,
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
    X=X,
    y=y,
    target_description=">50K (1) vs <=50K (0)",
    extra_stats={
        "Occupations": str(len(np.unique(groups))),
        "Prevalence": f"{float(y.values.mean()):.2%}",
    },
)

# %%
# ============================================================================
# Verify resolve_family detects "logistic_mixed"
# ============================================================================

auto_family = resolve_family("logistic_mixed", np.ravel(y))

print_family_info_table(
    explicit_family=auto_family,
)

# %%
# ============================================================================
# Score individual — family="logistic_mixed"
# ============================================================================
# Score projection computes permuted test statistics via a single
# matrix-vector product — orders of magnitude faster than full IRLS
# refitting for each permutation.

results_score = randomization_test_regression(
    X,
    y,
    method="score",
    family="logistic_mixed",
    groups=groups,
    n_randomizations=999,
    random_state=42,
)
print_results_table(results_score)
print_diagnostics_table(results_score)

# %%
# ============================================================================
# Confounder identification with cluster bootstrap
# ============================================================================

all_confounder_results = identify_confounders(
    X,
    y,
    family="logistic",
    groups=groups,
    random_state=42,
)
print_confounder_table(all_confounder_results)

# %%
# ============================================================================
# Score with covariate control — family="logistic_mixed"
# ============================================================================
# In labor economics analysis, evaluating the wage premium of education
# (education_num) often adjusts for labor supply intensity (hours_per_week).
# We run a cluster-adjusted score test controlling for 'hours_per_week'.

results_sc = randomization_test_regression(
    X,
    y,
    method="score",
    confounders=["hours_per_week"],
    family="logistic_mixed",
    groups=groups,
    n_randomizations=999,
    random_state=42,
)
print_results_table(results_sc)
print_diagnostics_table(results_sc)

# %%
# ============================================================================
# Execution & Protocol Artifacts
# ============================================================================
# Inspect the internal execution context from the completed test result:
# backend acceleration, batch convergence, fit metrics, and protocol properties.

print_protocol_usage_table(results_sc)

# %% [markdown]
"""
Example: Poisson Multilevel Regression (Count Outcome with Hierarchical Clustering)
Dataset: SUPPORT2 (UCI Machine Learning Repository ID=880)

Dataset Context & Theoretical Background:
    The SUPPORT2 dataset (Study to Understand Prognoses and Preferences for
    Outcomes and Risks of Treatments; Knaus et al., 1995) comprises clinical,
    physiological, and diagnostic records from 9,105 critically ill hospitalized
    adults across five major United States medical academic centers.

    The response variable is `num.co` (number of diagnosed chronic comorbidities,
    ranging from 0 to 9), exhibiting near-perfect equi-dispersion (sample
    variance / sample mean = 0.97). The observations are clustered hierarchically
    across distinct disease categories (`dzgroup`), such as Acute Respiratory
    Failure (ARF/MOSF w/ Sepsis), Congestive Heart Failure (CHF), Chronic
    Obstructive Pulmonary Disease (COPD), Cirrhosis, Coma, and Malignancies.
    Patients treated under the same disease diagnosis share unobserved baseline
    frailty and institutional management protocols, inducing positive intraclass
    correlation (ICC > 0).

    Ignoring this hierarchical clustering by fitting a naive single-level Poisson
    GLM violates observation independence, underestimating standard errors and
    artificially inflating false positive rates. The Poisson Generalized Linear
    Mixed Model (Poisson GLMM; Breslow & Clayton, 1993) addresses this via a
    random cluster intercept:

        log(lambda_{ij}) = alpha + X_{ij} * beta + u_j,    u_j ~ N(0, sigma_u^2)
        Y_{ij} ~ Poisson(lambda_{ij})

Features Selected for Modeling:
    - age: Patient chronological age in years. Comorbidity accumulation is
      naturally progressive with biological aging.
    - meanbp: Mean arterial blood pressure (mmHg). Reflects baseline vascular tone
      and hemodynamic stability.
    - hrt: Heart rate in beats per minute (bpm). Key vital sign indicative of
      physiologic stress, tachycardia, or autonomic response.
    - resp: Respiratory rate in breaths per minute. Sensitive marker of respiratory
      distress, metabolic acid-base compensation, or sepsis.
    - temp: Core body temperature in degrees Celsius (deg C). Hypothermia or
      hyperthermia reflects systemic inflammatory response.

Methodological Rationale for Score Projection Permutation:
    1. The Computational Bottleneck in GLMM Permutation:
       In linear mixed models (LMMs), batch OLS/GLS can be vectorized over B
       permutations. In GLMMs, however, each fit requires non-linear Penalized
       Quasi-Likelihood (PQL) or Adaptive Gauss-Hermite Quadrature with iterative
       numerical optimization. Refitting full GLMMs across B = 999 or 9,999
       permutations is computationally prohibitive.
    2. Score Projection Strategy (Rao 1948; Commenges 2003):
       Under the null hypothesis H_0: beta_j = 0, the variance components and
       reduced-model fixed effects are estimated once on the original sample.
       The efficient score vector U = X^T W (y - mu) is projected onto the null
       subspace. Permuted score test statistics S*(b) are then evaluated via
       vectorized matrix-vector multiplication without iterative refitting.
    3. Block-Permutation Invariance:
       To respect the exchangeability structure of clustered data, observations
       are randomized according to the hierarchical tree: either permuting entire
       cluster blocks or permuting within homogeneous strata, preserving the
       intraclass correlation structure under the null hypothesis.

Demonstrates:
    - family="poisson_mixed" -- Poisson generalized linear mixed model via
      PoissonMixedFamily
    - Efficient score projection permutation testing for GLMMs
    - Automated cluster-aware confounder sieve with cluster bootstrap (groups=)
    - Confounder-adjusted score projection testing
    - Execution and protocol artifacts inspection via print_protocol_usage_table

References:
    - Knaus, W. A., Harrell, F. E., Lynn, J., et al. (1995). The SUPPORT
      prognostic model: Objective estimates of survival for seriously ill
      hospitalized adults. Annals of Internal Medicine, 122(3), 191-203.
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
    X=X,
    y=y,
    target_name="num.co",
    target_description="number of comorbidities (count)",
    extra_stats={
        "Disease groups": str(len(np.unique(groups))),
        "Var / Mean": f"{float(y.values.var()) / float(y.values.mean()):.3f}",
    },
)

# %%
# ============================================================================
# Verify resolve_family detects "poisson_mixed"
# ============================================================================

auto_family = resolve_family("poisson_mixed", y_np)

print_family_info_table(
    explicit_family=auto_family,
)

# %%
# ============================================================================
# Score individual — family="poisson_mixed"
# ============================================================================
# Score projection computes permuted test statistics via a single
# matrix-vector product — orders of magnitude faster than full IRLS
# refitting for each permutation.

results_score = randomization_test_regression(
    X,
    y,
    method="score",
    family="poisson_mixed",
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
    family="poisson",
    groups=groups,
    random_state=42,
)
print_confounder_table(all_confounder_results)

# %%
# ============================================================================
# Score with identified confounders — family="poisson_mixed"
# ============================================================================
# The confounder sieve identified that 'hrt' (heart rate) is confounded by
# 'temp' (body temperature). We execute a cluster-adjusted score test for
# 'hrt' controlling for 'temp'.

target_predictor = "hrt"
confounders = all_confounder_results[target_predictor].identified_confounders

results_sc = randomization_test_regression(
    X,
    y,
    method="score",
    confounders=confounders,
    family="poisson_mixed",
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

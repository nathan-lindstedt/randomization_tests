# %% [markdown]
"""
Example: Ordinal Logistic Regression (Proportional Odds Model)
Dataset: Wine Quality (UCI Machine Learning Repository ID=186)

Dataset Context & Theoretical Background:
    The Wine Quality dataset (Cortez et al., 2009) comprises physicochemical
    laboratory measurements and sensory quality evaluations of red and white
    variants of Portuguese "Vinho Verde". Quality was graded on an ordered
    integer sensory scale from 0 (very bad) to 10 (very excellent) by certified
    wine assessors following double-blind sensory protocols.

    In statistical modeling, an ordered categorical response violates the
    assumptions of both metric linear regression (which assumes equal spacing
    between adjacent scale increments) and multinomial logistic regression
    (which ignores ordinal ordering and unnecessarily inflates parameter
    dimensionality). The proportional odds cumulative logit model (McCullagh,
    1980) addresses this by parameterizing cumulative category probabilities:

        logit(P(Y <= k | X)) = alpha_k - X * beta,   for k = 1, ..., K - 1

    where alpha_1 <= alpha_2 <= ... <= alpha_{K-1} are strictly monotonic
    threshold intercepts (cutpoints) partitioning the latent continuous quality
    continuum, and beta is an invariant vector of regression slopes shared
    across all ordinal transitions (the proportional odds assumption).

Features Selected for Modeling:
    - alcohol: Alcohol percent volume (% vol). Higher alcohol levels are
      strongly associated with perceived body, warmth, and flavor extraction,
      consistently receiving higher preference ratings from expert tasters.
    - volatile_acidity: Acetic acid content (g/dm^3). Elevated volatile
      acidity imparts an unpleasant vinegar aroma and sensory sharpness,
      constituting a major sensory defect.
    - sulphates: Potassium sulphate additive (g/dm^3). Contributes to free
      and bound sulfur dioxide (SO2) equilibrium, serving as an essential
      antioxidant and antimicrobial preservative that protects fruit freshness.
    - citric_acid: Citric acid concentration (g/dm^3). Adds tartness, sensory
      freshness, and structural crispness to the palate profile.
    - residual_sugar: Remaining fermentable hexose sugars after fermentation
      ceases (g/dm^3), balancing perceived acidity and dry mouthfeel.

Methodological Rationale for Resampling Tests:
    1. Direct Y Permutation vs. Residual Permutation:
       Unlike continuous regression models, ordinal logistic regression yields
       discrete cumulative class probabilities rather than continuous additive
       errors. Well-defined continuous residuals do not exist. Consequently,
       residual-based resampling methods such as Freedman-Lane (1983) and
       ter Braak (1992) are theoretically and mechanically invalid for ordinal
       outcomes, and are explicitly rejected with informative error messages.
    2. Manly (1997) Direct Permutation:
       The canonical exact permutation test for ordinal regression permutes the
       discrete response vector Y directly across observational units (Manly,
       1997). Under the global null hypothesis H_0: beta = 0, the joint
       distribution of Y is invariant under the symmetric group S_n.
    3. Kennedy (1995) Exposure-Residual Permutation:
       When partial regression effects must be tested while controlling for
       confounding covariates Z, Kennedy's (1995) method permutes the residuals
       e_X = (I - H_Z) X obtained from regressing exposure X on Z. Because e_X
       is continuous and derived from an ordinary least squares projection, it
       remains fully defined regardless of whether the outcome Y is metric,
       binary, count, or ordinal.

Demonstrates:
    - family="ordinal" -- proportional-odds cumulative logit regression
      via OrdinalFamily
    - Manly (1997) direct permutation testing (individual and joint deviance)
    - Kennedy (1995) exposure-residual permutation testing
    - Expected rejection of Freedman-Lane and ter Braak on ordinal models
    - Automated confounder identification and partial permutation testing
    - Execution and protocol artifacts inspection via print_protocol_usage_table

References:
    - Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
      Modeling wine preferences by data mining from physicochemical properties.
      Decision Support Systems, 47(4), 547-553.
    - McCullagh, P. (1980). Regression models for ordinal data. Journal of
      the Royal Statistical Society: Series B (Methodological), 42(2), 109-127.
    - Manly, B. F. J. (1997). Randomization, Bootstrap and Monte Carlo Methods
      in Biology (2nd ed.). Chapman & Hall/CRC.
    - Kennedy, P. E. (1995). Randomization tests in econometrics. Journal of
      Business & Economic Statistics, 13(1), 85-94.
"""

# %%
import numpy as np
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    identify_confounders,
    print_compatibility_table,
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
# Wine Quality dataset (combined red + white, 6497 observations).
# Y = quality score (integer 3–9), treated as ordinal.
# We subsample and select features for a manageable demo runtime.

wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets

# Select a representative feature subset (5 features)
selected_features = [
    "alcohol",
    "volatile_acidity",
    "sulphates",
    "citric_acid",
    "residual_sugar",
]
X = X[selected_features]

# Subsample for demo speed (ordinal BFGS is expensive per permutation)
rng = np.random.default_rng(42)
n_sub = 500
idx = rng.choice(X.shape[0], size=n_sub, replace=False)
X = X.iloc[idx].reset_index(drop=True)
y = y.iloc[idx].reset_index(drop=True)

# Convert quality to 0-indexed ordinal
y_min = int(y.values.min())
y = y - y_min

y_vals = np.ravel(y.values).astype(int)
ordinal_levels = sorted(np.unique(y_vals).tolist())

print_dataset_info_table(
    name=wine_quality.metadata.name,
    X=X,
    y=y,
    target_description="wine quality score (ordinal)",
    extra_stats={
        "Outcome Levels": str(ordinal_levels),
        "Unique Categories": str(len(ordinal_levels)),
    },
)

# %%
# ============================================================================
# Family resolution
# ============================================================================

ordinal_family = resolve_family("ordinal", np.ravel(y))

print_family_info_table(
    explicit_family=ordinal_family,
)

# %%
# ============================================================================
# Manly (1997) individual — direct Y permutation (family="ordinal")
# ============================================================================

results_manly = randomization_test_regression(
    X, y, method="manly", family="ordinal", n_randomizations=999
)
print_results_table(results_manly)
print_diagnostics_table(results_manly)

# %%
# ============================================================================
# Manly (1997) joint — direct Y permutation (family="ordinal")
# ============================================================================

results_manly_joint = randomization_test_regression(
    X, y, method="manly_joint", family="ordinal", n_randomizations=999
)
print_joint_results_table(results_manly_joint)

# %%
# ============================================================================
# Kennedy (1995) individual — family="ordinal"
# ============================================================================

results_kennedy = randomization_test_regression(
    X,
    y,
    method="kennedy",
    confounders=[],
    family="ordinal",
    n_randomizations=999,
)
print_results_table(results_kennedy)
print_diagnostics_table(results_kennedy)

# %%
# ============================================================================
# Kennedy (1995) joint — family="ordinal"
# ============================================================================

results_kennedy_joint = randomization_test_regression(
    X,
    y,
    method="kennedy_joint",
    confounders=[],
    family="ordinal",
    n_randomizations=999,
)
print_joint_results_table(results_kennedy_joint)

# %%
# ============================================================================
# Method compatibility
# ============================================================================
# Ordinal models have discrete cumulative probability distributions rather than
# scalar continuous errors. Residual-based methods like Freedman-Lane and
# ter Braak are incompatible and rejected with ValueError. We inspect the
# structured compatibility matrix for the ordinal family.

print_compatibility_table("ordinal")

# %%
# ============================================================================
# Confounder identification
# ============================================================================

all_confounder_results = identify_confounders(X, y, family="ordinal")
print_confounder_table(all_confounder_results)

# %%
# ============================================================================
# Kennedy with covariate control — family="ordinal"
# ============================================================================
# Enological research often evaluates the sensory penalty of volatile acidity
# while adjusting for alcohol content (which masks sensory acidity). We execute
# a Kennedy permutation test for 'volatile_acidity' controlling for 'alcohol'.

results_kc = randomization_test_regression(
    X,
    y,
    method="kennedy",
    confounders=["alcohol"],
    family="ordinal",
    n_randomizations=999,
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

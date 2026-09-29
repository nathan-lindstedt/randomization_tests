# %% [markdown]
"""
Example: Poisson Regression (Count Outcome)
Dataset: Abalone (UCI Machine Learning Repository ID=1)

Dataset Context & Theoretical Background:
    The Abalone dataset (Nash et al., 1994) comprises 4,177 biological and physical
    measurements of Tasmanian blacklip abalone (*Haliotis rubra*) harvested from
    the Bass Strait, subsampled here to 300 observations for rapid interactive
    execution. The response variable is `Rings`, an integer count representing the
    number of concentric growth rings counted under microscopic examination after
    histological sectioning and staining. Chronological age in years corresponds
    approximately to Rings + 1.5.

    Count responses are naturally discrete, non-negative, and heteroscedastic,
    violating standard Gaussian linear assumptions. The Poisson Generalized
    Linear Model (GLM; Nelder & Wedderburn, 1972) models the expected count via
    the canonical log link:

        log(E[Y | X]) = log(mu) = alpha + X * beta  ==>  mu = exp(alpha + X * beta)

    The standard Poisson likelihood imposes the equi-dispersion property:

        Var(Y | X) = E[Y | X] = mu

    In this subsample, the outcome exhibits mild underdispersion (sample variance /
    sample mean ≈ 1.05; conditional Pearson dispersion ratio ≈ 0.60), satisfying
    the equi-dispersion requirement without overdispersion.

Features Selected for Modeling:
    - Shell_weight: Dry shell weight in grams. Because shell accretion continues
      unabated throughout the lifespan, shell mass represents the most reliable
      continuous physical marker of biological age.
    - Shucked_weight: Weight of abalone meat in grams. Soft tissue mass reflects
      nutritional state but exhibits diminishing returns as metabolic senescence occurs.
    - Height: Shell vertical thickness in millimeters (mm). Measures vertical
      conical shell elevation and growth volume.
    - is_female: Binary indicator for female specimens (male is the reference class).
    - is_infant: Binary indicator for sexually immature juveniles lacking
      differentiated gonad development.

Methodological Rationale for Resampling Tests:
    1. Exact Deviance Testing vs. Asymptotic Wald Inference:
       Classical Poisson GLM inference relies on asymptotic Wald z-statistics derived
       from the inverted Fisher information matrix. In finite samples or under
       moderate skewness, Wald tests can be inaccurate. Permutation tests evaluate
       exact likelihood-ratio deviance reductions directly from the data.
    2. Response-Scale Stochastic Poisson Reconstruction:
       Because count models require non-negative integer outcomes, residual
       reconstruction must preserve the count data-generating process. The engine
       permutes response-scale residuals e = y - mu_hat, reconstructs rate parameters
       mu* = clip(mu_hat + pi(e), 1e-10, 1e8), and draws permuted responses
       stochastically via Y* ~ Poisson(mu*).
    3. Freedman–Lane and Kennedy Conditioning:
       Freedman–Lane (1983) and Kennedy (1995) condition on allometric confounders
       (such as total shell weight), isolating the partial contributions of soft
       tissue mass and sexual dimorphism independently of overall body scale.

Demonstrates:
    - family="poisson" -- explicit family selection for count data
    - Conservative auto-detection behavior (family="auto" selects "linear" for continuous compatibility)
    - ter Braak (1992) permutation test with Poisson deviance residuals
    - Freedman–Lane (1983) individual and joint permutation tests
    - Kennedy (1995) individual and joint permutation tests
    - Poisson-specific goodness-of-fit diagnostics (deviance, Pearson χ², dispersion ratio)
    - Stochastic reconstruction for count models: Y* ~ Poisson(clip(mu_hat + pi(e)))
    - Four-stage confounder sieve with Poisson regression
    - Execution and protocol artifacts inspection via print_protocol_usage_table

References:
    - Nash, W. J., Sellers, T. L., Talbot, S. R., Cawthorn, A. J., & Wesney, B. (1994).
      The Population Biology of Abalone (Haliotis species) in Tasmania. I. Blacklip
      Abalone (H. rubra) from the North Coast and Islands of Bass Strait. Sea Fisheries
      Division, Technical Report No. 48, ISSN 1034-3288.
    - Nelder, J. A., & Wedderburn, R. W. (1972). Generalized linear models. Journal of
      the Royal Statistical Society: Series A (General), 135(3), 370-384.
    - Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported
      significance levels. Journal of Business & Economic Statistics, 1(4), 292-298.
    - ter Braak, C. J. F. (1992). Permutation versus bootstrap significance tests in
      multiple regression and ANOVA. Handbook of Statistics, 9, 79-104.
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

abalone = fetch_ucirepo(id=1)
X_full = abalone.data.features.copy()
y_full = abalone.data.targets

# Subsample to 300 rows (permutation tests re-fit B × p GLMs, so
# keeping n moderate avoids excessive runtime for an example script).
rng = np.random.default_rng(42)
idx = rng.choice(len(X_full), size=300, replace=False)
X_sub = X_full.iloc[idx].reset_index(drop=True)
y_sub = y_full.iloc[idx].reset_index(drop=True)

# Feature engineering: dummy-code Sex (M/F/I) into two indicators.
# Male is the reference category.
X = pd.DataFrame(
    {
        "Shell_weight": X_sub["Shell_weight"].astype(float),
        "Shucked_weight": X_sub["Shucked_weight"].astype(float),
        "Height": X_sub["Height"].astype(float),
        "is_female": (X_sub["Sex"] == "F").astype(float),
        "is_infant": (X_sub["Sex"] == "I").astype(float),
    }
)
y = y_sub.copy()

print_dataset_info_table(
    name=abalone.metadata.name,
    X=X,
    y=y,
    target_description="growth-ring count",
)

# %%
# ============================================================================
# Family resolution
# ============================================================================

poisson_family = resolve_family("poisson", np.ravel(y))

print_family_info_table(
    y=y,
    explicit_family=poisson_family,
)

# %%
# ============================================================================
# ter Braak (1992) — family="poisson" (explicit)
# ============================================================================

results_ter_braak = randomization_test_regression(
    X, y, method="ter_braak", family="poisson"
)
print_results_table(results_ter_braak)
print_diagnostics_table(results_ter_braak)

# %%
# ============================================================================
# Kennedy (1995) individual — family="poisson"
# ============================================================================

results_kennedy = randomization_test_regression(
    X, y, method="kennedy", confounders=[], family="poisson"
)
print_results_table(results_kennedy)
print_diagnostics_table(results_kennedy)

# %%
# ============================================================================
# Kennedy (1995) joint — family="poisson"
# ============================================================================

results_kennedy_joint = randomization_test_regression(
    X, y, method="kennedy_joint", confounders=[], family="poisson"
)
print_joint_results_table(results_kennedy_joint)

# %%
# ============================================================================
# Freedman–Lane (1983) individual — family="poisson"
# ============================================================================

results_fl = randomization_test_regression(
    X, y, method="freedman_lane", confounders=[], family="poisson"
)
print_results_table(results_fl)
print_diagnostics_table(results_fl)

# %%
# ============================================================================
# Freedman–Lane (1983) joint — family="poisson"
# ============================================================================

results_fl_joint = randomization_test_regression(
    X, y, method="freedman_lane_joint", confounders=[], family="poisson"
)
print_joint_results_table(results_fl_joint)

# %%
# ============================================================================
# Confounder identification
# ============================================================================

all_confounder_results = identify_confounders(X, y, family="poisson")
print_confounder_table(all_confounder_results)

# %%
# ============================================================================
# Kennedy with identified confounders — family="poisson"
# ============================================================================
# The confounder sieve identified that 'Shell_weight' is confounded by the
# sex indicators ('is_female', 'is_infant'). We now execute Kennedy's
# permutation test controlling for these confounders to isolate the partial
# allometric accretion effect.

target_predictor = "Shell_weight"
confounders = all_confounder_results[target_predictor].identified_confounders

results_kc = randomization_test_regression(
    X,
    y,
    method="kennedy",
    confounders=confounders,
    family="poisson",
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

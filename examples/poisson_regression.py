# %% [markdown]
"""
Example: Poisson Regression (Count Outcome)
Abalone dataset (UCI ML Repository ID=1)

Demonstrates:
- ``family="poisson"`` — explicit family selection for count data
- Conservative auto-detection behavior (``family="auto"`` selects ``"linear"`` for continuous compatibility)
- ter Braak (1992) permutation test with Poisson deviance residuals
- Freedman–Lane (1983) individual and joint permutation tests
- Kennedy (1995) individual and joint permutation tests
- Poisson-specific goodness-of-fit diagnostics (deviance, Pearson χ², dispersion ratio)
- Stochastic reconstruction for count models: :math:`Y^* \\sim \\mathrm{Poisson}(\\mathrm{clip}(\\hat{\\mu} + \\pi(e)))`
- Four-stage confounder sieve with Poisson regression
- Execution and protocol artifacts inspection via ``print_protocol_usage_table``

Dataset
-------
4,177 physical measurements of Tasmanian blacklip abalone (*Haliotis rubra*),
subsampled to 300 observations for demo runtime. The target variable is ``Rings``
(an integer count representing concentric growth rings in the shell cone, where
age in years is approximately :math:`\\mathrm{Rings} + 1.5`).

With marginal variance-to-mean ratio ≈ 1.05 and conditional dispersion ratio ≈ 0.60,
the outcome satisfies Poisson equi-dispersion assumptions without overdispersion.

Feature selection rationale
---------------------------
Five physical predictors capture developmental allometry:

- **Shell_weight**: Dry shell weight in grams. Shell accretion continues throughout
  life, making shell mass the single most reliable physical marker of age.
- **Shucked_weight**: Weight of abalone meat in grams. Soft tissue mass exhibits
  diminishing returns as metabolic senescence sets in.
- **Height**: Shell vertical thickness in mm.
- **is_female** & **is_infant**: Indicator variables for sex categories (male is the
  reference category). Infantile abalones lack differentiated gonad development.

Methodological rationale
-------------------------
Poisson regression models expected counts as :math:`\\mathbb{E}[Y \\mid X] = \\exp(X\beta)`.
Classical inference relies on Wald z-statistics that assume asymptotic normality of the
maximum likelihood estimator. In finite samples, permutation testing provides exact
likelihood-ratio deviance reduction tests under the null hypothesis of no association.

Reference
---------
Nash, W. J., Sellers, T. L., Talbot, S. R., Cawthorn, A. J., & Wesney, B. (1994).
The Population Biology of Abalone (*Haliotis* species) in Tasmania. I. Blacklip
Abalone (*H. rubra*) from the North Coast and Islands of Bass Strait. *Sea Fisheries
Division, Technical Report No. 48*, ISSN 1034-3288.
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

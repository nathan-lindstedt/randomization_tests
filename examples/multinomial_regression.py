# %% [markdown]
"""
Example: Multinomial Logistic Regression (Unordered Categorical Outcome)
Dataset: Wine (UCI Machine Learning Repository ID=109)

Dataset Context & Theoretical Background:
    The Wine dataset (Forina et al., 1991; Aeberhard et al., 1992) reports the
    chemical analysis of wines grown in the Piedmont region of Italy, derived
    from three distinct cultivars: Barolo (Class 1), Grignolino (Class 2), and
    Barbera (Class 3). The objective is to identify which chemical attributes
    statistically distinguish the three wine cultivars.

    When the response variable Y is nominal (unordered categorical with K >= 3
    classes), neither standard metric regression nor ordinal regression is
    appropriate. Multinomial logistic regression models the log-odds of
    belonging to class k relative to a chosen baseline class K (the reference):

        log( P(Y = k | X) / P(Y = K | X) ) = alpha_k + X * beta_k,  k = 1, ..., K - 1

    Because each predictor has K - 1 independent slope coefficients across the
    contrast equations, testing whether a predictor significantly affects class
    membership requires a joint multidimensional test. Under the null
    hypothesis H_0: beta_{j,1} = beta_{j,2} = ... = beta_{j,K-1} = 0, the
    scalar test statistic is the Wald chi-square statistic:

        W_j = hat{beta}_j^T * [Cov(hat{beta}_j)]^{-1} * hat{beta}_j ~ chi^2(K - 1)

Features Selected for Modeling:
    - Alcohol: Alcohol percent by volume (% vol). Reflects grape maturity, sugar
      accumulation, and yeast fermentation characteristics across cultivars.
    - Malicacid: Malic acid concentration (g/L). A key dicarboxylic organic acid
      strongly dependent on vine microclimate, altitude, and harvest timing.
    - Ash: Total inorganic mineral residue remaining after sample incineration (g/L),
      reflecting soil uptake and vineyard geology.
    - Magnesium: Magnesium mineral content (mg/L), acting as an enzyme cofactor in
      fermentation and vine nutrition.
    - Hue: The color ratio of absorbance at 420 nm to 520 nm, characterizing
      anthocyanin coloration and oxidation state.

Methodological Rationale for Resampling Tests:
    1. Direct Y Permutation vs. Residual Permutation:
       Multinomial models produce K-dimensional probability simplex predictions
       rather than scalar continuous residuals. Scalar error reconstruction
       is undefined. Therefore, residual-based resampling algorithms such as
       Freedman-Lane (1983) and ter Braak (1992) cannot be applied, and are
       safeguarded against by raising informative ValueErrors.
    2. Manly (1997) Direct Permutation:
       The exact finite-sample test for multinomial regression permutes the
       discrete class labels Y directly across observation units (Manly, 1997).
       Under the global null hypothesis that cultivar identity is independent
       of chemical composition, the exchangeability of Y holds unconditionally
       under the symmetric group S_n.
    3. Kennedy (1995) Exposure-Residual Permutation:
       When assessing the partial significance of a chemical attribute while
       adjusting for confounding covariates Z, Kennedy's (1995) method permutes
       the exposure residuals e_X = (I - H_Z) X. Because e_X is continuous and
       derived from an OLS projection of X onto Z, it remains valid regardless
       of the outcome's discrete multinomial nature.

Demonstrates:
    - family="multinomial" -- unordered multinomial logit via MultinomialFamily
    - Manly (1997) direct permutation testing (individual Wald chi^2 and joint deviance)
    - Kennedy (1995) exposure-residual permutation testing
    - Method compatibility matrix via print_compatibility_table
    - Automated confounder identification and partial permutation testing
    - Execution and protocol artifacts inspection via print_protocol_usage_table

References:
    - Forina, M., Armanino, C., Castino, M., & Ubigli, M. (1991). Multivariate
      data analysis as a discriminating tool of the origin of wines. Vitis,
      25, 189-201.
    - Aeberhard, S., Coomans, D., & de Vel, O. (1992). Comparative analysis of
      statistical pattern recognition methods in high dimensional settings.
      Pattern Recognition, 27(8), 1065-1077.
    - Manly, B. F. J. (1997). Randomization, Bootstrap and Monte Carlo Methods
      in Biology (2nd ed.). Chapman & Hall/CRC.
    - Kennedy, P. E. (1995). Randomization tests in econometrics. Journal of
      Business & Economic Statistics, 13(1), 85-94.
"""

# %%
import numpy as np
import pandas as pd
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

wine = fetch_ucirepo(id=109)
X = wine.data.features
y = wine.data.targets

# Wine classes are 1, 2, 3 — recode to 0, 1, 2 for MultinomialFamily.
y = pd.DataFrame(
    y.iloc[:, 0].values - 1,
    columns=["class"],
)

# Select 5 features with low multicollinearity (all VIF ≤ 1.5).
selected_features = ["Alcohol", "Malicacid", "Ash", "Magnesium", "Hue"]
X = X[selected_features]

y_values = np.ravel(y)
class_names = ["class_0", "class_1", "class_2"]
counts = np.bincount(y_values)

print_dataset_info_table(
    name=wine.metadata.name,
    X=X,
    y=y,
    target_description="wine cultivar (3 classes)",
    extra_stats={
        "Classes": ", ".join(class_names),
        "Class Counts": ", ".join(
            f"{c}: {n}" for c, n in zip(class_names, counts, strict=True)
        ),
    },
)

_family = resolve_family("multinomial")

print_family_info_table(
    explicit_family=_family,
)

# %%
# ============================================================================
# Manly (1997) individual — direct Y permutation (family="multinomial")
# ============================================================================

results_manly = randomization_test_regression(
    X, y, method="manly", family="multinomial", n_randomizations=999
)
print_results_table(results_manly)
print_diagnostics_table(results_manly)

# %%
# ============================================================================
# Manly (1997) joint — direct Y permutation (family="multinomial")
# ============================================================================

results_manly_joint = randomization_test_regression(
    X, y, method="manly_joint", family="multinomial", n_randomizations=999
)
print_joint_results_table(results_manly_joint)

# %%
# ============================================================================
# Kennedy (1995) individual — family="multinomial"
# ============================================================================

results_kennedy = randomization_test_regression(
    X, y, method="kennedy", confounders=[], family="multinomial", n_randomizations=999
)
print_results_table(results_kennedy)
print_diagnostics_table(results_kennedy)

# %%
# ============================================================================
# Kennedy (1995) joint — family="multinomial"
# ============================================================================

results_kennedy_joint = randomization_test_regression(
    X,
    y,
    method="kennedy_joint",
    confounders=[],
    family="multinomial",
    n_randomizations=999,
)
print_joint_results_table(results_kennedy_joint)

# %%
# ============================================================================
# Method compatibility
# ============================================================================
# Multinomial models predict K-class probabilities rather than continuous
# scalar errors. Residual-based methods like Freedman-Lane and ter Braak are
# incompatible and rejected with ValueError. We inspect the compatibility matrix.

print_compatibility_table("multinomial")

# %%
# ============================================================================
# Confounder identification — multinomial
# ============================================================================

all_confounder_results = identify_confounders(X, y, family="multinomial")
print_confounder_table(all_confounder_results, family=_family)

# %%
# ============================================================================
# Kennedy with identified confounders — family="multinomial"
# ============================================================================
# The confounder sieve identified that 'Alcohol' is confounded by 'Magnesium'.
# We execute a Kennedy permutation test for 'Alcohol' controlling for 'Magnesium'.

target_predictor = "Alcohol"
confounders = all_confounder_results[target_predictor].identified_confounders

results_kc = randomization_test_regression(
    X,
    y,
    method="kennedy",
    confounders=confounders,
    family="multinomial",
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

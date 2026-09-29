# %% [markdown]
r"""
Example: Logistic Regression (Binary Outcome)
Dataset: Breast Cancer Wisconsin Diagnostic (UCI Machine Learning Repository ID=17)

Dataset Context & Theoretical Background:
    The Breast Cancer Wisconsin Diagnostic dataset (Street, Wolberg, & Mangasarian,
    1993) comprises 569 digitized cell nuclear measurements derived from fine needle
    aspirates (FNA) of human breast masses collected at the University of Wisconsin
    Hospitals. The diagnostic response is `Diagnosis`, encoded as binary: malignant
    (1, n = 212, 37.26% prevalence) versus benign (0, n = 357, 62.74%).

    Binary regression models the conditional probability of malignancy via the
    canonical logit link:

        logit(P(Y = 1 | X)) = log(P / (1 - P)) = alpha + X * beta

    Classical parametric inference relies on maximum likelihood estimation (MLE)
    and the inverted Fisher information matrix:

        Var(beta_hat) = (X^T W X)^{-1},    W = diag(mu_hat * (1 - mu_hat))

    In finite samples or when covariates strongly predict separation between benign
    and malignant cell clusters, Wald z-tests suffer from the Hauck–Donner effect
    (Hauck & Donner, 1977): as the true parameter magnitude increases, the estimated
    standard error grows faster than the coefficient itself, causing Wald test
    statistics to paradoxically plunge toward zero and falsely failing to reject
    the null.

    Permutation testing bypasses Fisher information asymptotics entirely by
    evaluating empirical likelihood-ratio deviance reductions directly under the
    exact permutation null distribution.

Features Selected for Modeling:
    - radius1: Mean distance from the nuclear center to perimeter points. Primary
      marker of nuclear enlargement in neoplastic cytology.
    - texture1: Standard deviation of gray-scale pixel values. Quantifies chromatin
      clumping, texture granularity, and nuclear heterogeneity.
    - perimeter1: Mean nuclear perimeter length. Core geometric proxy for total tumor
      nuclear cross-sectional dimension.
    - smoothness1: Local variation in radius lengths, quantifying structural
      irregularity and indentations in the nuclear envelope.
    - compactness1: Dimensionless shape metric computed as perimeter^2 / area - 1.0.
      Measures contour complexity and pleomorphic irregularity.

Methodological Rationale for Resampling Tests:
    1. Immunity to the Hauck–Donner Effect:
       Permutation test statistics evaluate deviance reductions or score projections
       rather than inverted Hessian standard errors, preserving power even when
       morphological features near-perfectly separate malignant from benign biopsies.
    2. Stochastic Bernoulli Reconstruction:
       Because binary residuals e_i = y_i - p_hat_i lie on the probability scale
       (-1, 1), adding permuted residuals p* = clip(p_hat + pi(e), 0.001, 0.999)
       followed by Bernoulli sampling Y* ~ Bernoulli(p*) converts continuous permuted
       probabilities into valid binary outcomes, preserving the binomial likelihood.
    3. Freedman–Lane and Kennedy Conditioning:
       Freedman–Lane (1983) and Kennedy (1995) condition on cytological confounders,
       distinguishing direct nuclear shape predictors from indirect correlates
       transmitted through overall nuclear perimeter or volume.

Demonstrates:
    - family="logistic" -- explicit and auto-detected (family="auto") binary logistic models
    - ter Braak (1992) permutation test with deviance residuals and recentred null
    - Freedman–Lane (1983) individual and joint permutation tests
    - Kennedy (1995) individual and joint permutation tests
    - Stochastic reconstruction in binary models: Y* ~ Bernoulli(clip(mu_hat + pi(e)))
    - Four-stage confounder sieve (screen -> collider -> mediator -> moderator)
    - Confounder-controlled permutation testing
    - Execution and protocol artifacts inspection via print_protocol_usage_table

References:
    - Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature
      extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium
      on Electronic Imaging: Science and Technology, 1905, 861-870.
    - Hauck, W. W., & Donner, A. (1977). Wald's test as applied to hypotheses in
      logit analysis. Journal of the American Statistical Association, 72(360), 851-853.
    - Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported
      significance levels. Journal of Business & Economic Statistics, 1(4), 292-298.
    - ter Braak, C. J. F. (1992). Permutation versus bootstrap significance tests
      in multiple regression and ANOVA. Handbook of Statistics, 9, 79-104.
"""

# %%
import numpy as np
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

breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features
y = breast_cancer.data.targets

# Convert target to binary: malignant (M) -> 1, benign (B) -> 0
y = (y == "M").astype(int)

selected_features = ["radius1", "texture1", "perimeter1", "smoothness1", "compactness1"]
X = X[selected_features]

print_dataset_info_table(
    name=breast_cancer.metadata.name,
    X=X,
    y=y,
    target_description="malignant (1) vs benign (0)",
    extra_stats={"Prevalence": f"{float(y.values.mean()):.2%}"},
)

# %%
# ============================================================================
# Verify resolve_family auto-detects "logistic" for binary Y
# ============================================================================

auto_family = resolve_family("auto", np.ravel(y))
logistic_family = resolve_family("logistic", np.ravel(y))

print_family_info_table(
    auto_family=auto_family,
    explicit_family=logistic_family,
)

# %%
# ============================================================================
# ter Braak (1992) — family="auto" (auto-detection)
# ============================================================================

results_ter_braak_auto = randomization_test_regression(
    X, y, method="ter_braak", family="auto"
)
print_results_table(results_ter_braak_auto)

# %%
# ============================================================================
# ter Braak (1992) — family="logistic" (explicit)
# ============================================================================

results_ter_braak = randomization_test_regression(
    X, y, method="ter_braak", family="logistic"
)
print_results_table(results_ter_braak)
print_diagnostics_table(results_ter_braak)

# %%
# ============================================================================
# Kennedy (1995) individual — family="logistic"
# ============================================================================

results_kennedy = randomization_test_regression(
    X, y, method="kennedy", confounders=[], family="logistic"
)
print_results_table(results_kennedy)
print_diagnostics_table(results_kennedy)

# %%
# ============================================================================
# Kennedy (1995) joint — family="logistic"
# ============================================================================

results_kennedy_joint = randomization_test_regression(
    X, y, method="kennedy_joint", confounders=[], family="logistic"
)
print_joint_results_table(results_kennedy_joint)

# %%
# ============================================================================
# Freedman–Lane (1983) individual — family="logistic"
# ============================================================================

results_fl = randomization_test_regression(
    X, y, method="freedman_lane", confounders=[], family="logistic"
)
print_results_table(results_fl)
print_diagnostics_table(results_fl)

# %%
# ============================================================================
# Freedman–Lane (1983) joint — family="logistic"
# ============================================================================

results_fl_joint = randomization_test_regression(
    X, y, method="freedman_lane_joint", confounders=[], family="logistic"
)
print_joint_results_table(results_fl_joint)

# %%
# ============================================================================
# Confounder identification — logistic
# ============================================================================

all_confounder_results = identify_confounders(X, y, family="logistic")
print_confounder_table(all_confounder_results)

# %%
# ============================================================================
# Kennedy with identified covariate — family="logistic"
# ============================================================================
# While the four-stage sieve revealed extensive mediation and collider structures
# among nuclear geometry metrics, epidemiological analysis often controls for
# texture variation when evaluating tumor diameter. We test 'radius1' controlling
# for 'texture1' via Kennedy exposure residualization.

results_kc = randomization_test_regression(
    X,
    y,
    method="kennedy",
    confounders=["texture1"],
    family="logistic",
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

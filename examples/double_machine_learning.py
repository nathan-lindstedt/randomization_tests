# %% [markdown]
"""
Example: Double/Debiased Machine Learning (DML) Regression (Continuous Outcome)
Dataset: Real Estate Valuation (UCI Machine Learning Repository ID=477)

Dataset Context & Theoretical Background:
    The Real Estate Valuation dataset (Yeh & Hsu, 2018) comprises 414 residential
    property transactions from Sindian District, New Taipei City, Taiwan (2012–2013).
    The dependent variable is `Y house price of unit area` (measured in 10,000 New
    Taiwan Dollar per Ping, where 1 Ping ≈ 3.3 square meters).

    Property valuation is governed by complex spatial sorting: physical land
    attributes, municipal transit corridors, topographical barriers, and zoning
    regulations induce inherently non-linear relationships across space. While
    standard multiple linear regression assumes additive linear confounding:

        Y = X * beta + Z * gamma + eps

    in observational spatial econometrics, the true conditional expectations
    g(Z) = E[Y | Z] and m(Z) = E[X | Z] are non-linear. When confounding is
    non-linear, classical linear residualization (OLS projection) fails to absorb
    the non-linear confounder signal, leaving omitted variable bias in the
    residuals. This leaks spurious correlation into the inference stage, inflating
    Type I error rates up to 100% or destroying statistical power.

    Double/Debiased Machine Learning (DML; Robinson 1988; Chernozhukov et al. 2018)
    resolves this through two fundamental innovations:
    1. Neyman-Orthogonal Double Residualization:
       Flexible machine learning models estimate both the outcome nuisance
       function g(Z) = E[Y | Z] and the treatment/predictor nuisance functions
       m(Z) = E[X | Z]. Purging non-linear confounding from BOTH Y and X ensures
       the score equation is first-order insensitive (orthogonal) to small errors
       in nuisance function estimation.
    2. Out-of-Fold Cross-Fitting:
       Nuisance models are trained strictly on auxiliary folds and evaluated on
       held-out folds. This breaks own-observation sample dependence and prevents
       overfitting bias from distorting unit exchangeability.

Features Selected for Modeling:
    - X1 transaction date: Continuous transaction timing (2012.667 to 2013.583).
      Captures macroeconomic price appreciation and seasonal housing cycles.
    - X2 house age: Age of the residential structure in years (0 to 43.8 years).
      Reflects physical depreciation and structural wear.
    - X4 number of convenience stores: Count of major convenience stores (7-Eleven,
      FamilyMart, Hi-Life, OK Mart) within walking radius. A key proxy for local
      commercial activity and neighborhood amenities.
    - Confounders (Z):
      * X3 distance to the nearest MRT station: Metric distance to the nearest
        Mass Rapid Transit station. Accessibility to rapid transit is a primary
        spatial determinant of land valuation in Taipei metropolitan area.
      * X5 latitude & X6 longitude: Geographic coordinates in decimal degrees.
        Topographical elevation, school district boundaries, and municipal
        planning boundaries create non-linear spatial price surfaces.

Methodological Rationale for Double/Debiased Resampling:
    1. Why Linear Confounder Adjustment Fails in Spatial Data:
       Because distance to transit and geographic coordinates interact non-linearly,
       linear regression cannot absorb the full spatial surface. Consequently,
       commercial amenities such as convenience stores (X4) appear strongly
       statistically significant (p = 0.002) in naive OLS, when in reality stores
       are endogenously sited by corporate algorithms in areas of high foot traffic
       and high land values.
    2. Why Gradient Boosted Decision Trees (GBDT) for Nuisance Estimation:
       Chernozhukov et al. (2018, Theorem 3.1) establishes that DML estimators
       achieve root-n consistency and asymptotic normality provided nuisance
       estimators achieve an L2 error rate of o(n^{-1/4}). Gradient boosting
       naturally captures non-linear geographic curves, sharp boundary thresholds,
       and multi-way spatial interactions without requiring ad-hoc polynomial
       expansions or splines.
    3. Permutation Null Distribution on Cross-Fitted Residuals:
       Under the null hypothesis H_0: beta_j = 0, the cross-fitted residuals
       Y_tilde = Y - g_hat(Z) and X_tilde = X - m_hat(Z) are conditionally
       independent. Permuting Y_tilde* = pi(Y_tilde) evaluates the Neyman-orthogonal
       estimator across B permutations in a single vectorized matrix multiplication,
       yielding zero-centered, asymptotically exact permutation inference:
           Guarantee: Asymptotically Exact (DML)

Demonstrates:
    - Double/Debiased Machine Learning (DML) via reduced_model= parameter
    - Continuous regression DML with GradientBoostingRegressor
    - Demonstration of linear confounder leakage in standard Freedman-Lane (1983)
    - Elimination of spurious significance via Double-Residualized Freedman-Lane
    - Side-by-side comparison table contrasting Standard FL vs. DML (DIVERGENT verdict)
    - Omnibus joint hypothesis testing under DML (method="freedman_lane_joint")
    - Full protocol and execution artifacts inspection via print_protocol_usage_table

References:
    - Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
      Newey, W., & Robins, J. (2018). Double/debiased machine learning for
      treatment and structural parameters. The Econometrics Journal, 21(1), C1-C68.
    - Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
      Econometrica, 56(4), 931-954.
    - Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported
      significance levels. Journal of Business & Economic Statistics, 1(4), 292-298.
    - Yeh, I. C., & Hsu, T. K. (2018). Building real estate valuation models with
      comparative approach through case-based reasoning. Applied Soft Computing,
      65, 260-271.
"""

# %%
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    identify_confounders,
    print_comparison_table,
    print_confounder_table,
    print_dataset_info_table,
    print_diagnostics_table,
    print_joint_results_table,
    print_protocol_usage_table,
    print_results_table,
    randomization_test_regression,
)

# %%
# ============================================================================
# Continuous Regression DML — Real Estate Valuation
# ============================================================================
# Load data — Real Estate Valuation
# ============================================================================

real_estate_valuation = fetch_ucirepo(id=477)
X = real_estate_valuation.data.features
y = real_estate_valuation.data.targets

print_dataset_info_table(
    name=real_estate_valuation.metadata.name,
    X=X,
    y=y,
)

# %%
# ============================================================================
# Confounder screening
# ============================================================================

confounder_results = identify_confounders(X, y)
print_confounder_table(confounder_results)

# Geographic coordinates and transit accessibility form the continuous
# confounder block
confounders = [
    "X3 distance to the nearest MRT station",
    "X5 latitude",
    "X6 longitude",
]

# %%
# ============================================================================
# Freedman–Lane (1983) individual — baseline (linear confounding)
# ============================================================================

res_standard_fl = randomization_test_regression(
    X,
    y,
    confounders=confounders,
    method="freedman_lane",
    n_randomizations=500,
    random_state=42,
)
print_results_table(res_standard_fl)

# %%
# ============================================================================
# Double/Debiased Machine Learning (DML) via GradientBoostingRegressor
# ============================================================================
# In standard linear FL, convenience store density (X4) appears significant
# (p = 0.002) due to spatial sorting. Cross-fitting both Y and X on spatial
# confounders (Z) via GBDT purges non-linear geographic bias, revealing that
# store density has no independent causal effect on house prices.

reduced_model_gbdt = GradientBoostingRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
)

res_dml = randomization_test_regression(
    X,
    y,
    confounders=confounders,
    method="freedman_lane",
    reduced_model=reduced_model_gbdt,
    n_randomizations=500,
    random_state=42,
)
print_results_table(res_dml)
print_diagnostics_table(res_dml)

# %%
# ============================================================================
# Side-by-side comparison: Standard FL vs. DML
# ============================================================================

print_comparison_table([("Standard FL", res_standard_fl), ("DML (GBDT)", res_dml)])

# %%
# ============================================================================
# Freedman–Lane (1983) joint — Double/Debiased Machine Learning (DML)
# ============================================================================

res_dml_joint = randomization_test_regression(
    X,
    y,
    confounders=confounders,
    method="freedman_lane_joint",
    reduced_model=reduced_model_gbdt,
    n_randomizations=500,
    random_state=42,
)
print_joint_results_table(res_dml_joint)

# %%
# ============================================================================
# Protocol and execution artifacts
# ============================================================================

print_protocol_usage_table(res_dml)

# %% [markdown]
"""
Example: Logistic Double/Debiased Machine Learning (DML) Regression (Binary Outcome)
Dataset: Breast Cancer Wisconsin Diagnostic (UCI Machine Learning Repository ID=17)

Dataset Context & Theoretical Background:
    The Breast Cancer Wisconsin Diagnostic dataset (Street, Wolberg, & Mangasarian,
    1993) records 569 digitized cell nuclear measurements derived from fine needle
    aspirates (FNA) of breast masses. The clinical response is `Diagnosis`, encoded
    as binary: malignant (1, n = 212, 37.26%) versus benign (0, n = 357, 62.74%).

    Nuclear shape analysis in oncology presents complex non-linear confounding.
    Overall nuclear dimensions (perimeter, cross-sectional area, volume) scale
    allometrically and non-linearly with cell mass. Subtle diagnostic markers of
    malignancy, such as nuclear contour irregularity (compactness = perimeter^2 / area - 1.0),
    are frequently collinear with and non-linearly confounded by primary size metrics.

    In standard linear logistic regression, additive linear conditioning on nuclear
    size fails to isolate the non-linear morphological scaling relationship. This
    attenuates the estimated effect of contour irregularity, causing classical tests
    and standard Freedman–Lane to report compactness as non-significant (p = 0.615).

    Double/Debiased Machine Learning (DML) with paired non-linear learners:
        reduced_model = (RandomForestClassifier(), RandomForestRegressor())
    models the probability of malignancy P(Y=1 | Z) via random forest classification
    and the non-linear scaling of nuclear features E[X | Z] via random forest regression.
    Purging non-linear size confounding out-of-fold uncovers previously suppressed
    diagnostic signal, demonstrating how DML prevents false-negative conclusions.

Features Selected for Modeling:
    - radius1: Mean distance from the nuclear center to perimeter points. Primary
      marker of overall nuclear enlargement in neoplastic cytology.
    - texture1: Standard deviation of gray-scale pixel intensity values. Measures
      chromatin heterogeneity and textural roughness.
    - smoothness1: Local variation in radius lengths, quantifying microscopic
      indentations and folds in the nuclear envelope.
    - compactness1: Dimensionless shape metric computed as perimeter^2 / area - 1.0.
      Measures nuclear pleomorphism and contour complexity.
    - Confounders (Z):
      * perimeter1: Overall nuclear perimeter length in micrometers. Primary
        geometric size confounder governing cellular allometry.

Methodological Rationale for Double/Debiased Resampling:
    1. Paired Classification-Regression Nuisance Architecture:
       Binary outcomes Y in {0, 1} require probability estimation P(Y=1 | Z) via
       classifiers implementing predict_proba(). Continuous predictors X in R
       require conditional expectation estimation E[X | Z] via regressors.
       Supplying an explicit tuple (model_y, model_x) provides estimator-agnostic
       nuisance modeling tailored to the mathematical structure of each target.
    2. Overcoming Confounder Masking in Oncology:
       When morphological features are non-linearly confounded by cell scale,
       linear residualization fails to separate intrinsic contour roughness from
       overall size. Non-parametric random forest cross-fitting purges non-linear
       size dependence out-of-fold, restoring statistical power to detect true
       diagnostic biomarkers.
    3. Permutation Score Test for Binary Outcomes:
       Under H_0: beta_j = 0, probability-scale residuals y - p_hat(Z) are
       evaluated against the Fisher-weighted score projection across B permutations,
       providing asymptotically exact inference without parametric distributional
       assumptions:
           Guarantee: Asymptotically Exact (DML)

Demonstrates:
    - Logistic Double/Debiased Machine Learning (DML) via reduced_model= parameter
    - Paired estimator tuple: (RandomForestClassifier, RandomForestRegressor)
    - Demonstration of confounder masking in standard logistic Freedman-Lane
    - Recovery of masked diagnostic significance via Double-Residualized DML
    - Side-by-side comparison table contrasting Standard FL vs. DML (DIVERGENT verdict)
    - Omnibus joint hypothesis testing under logistic DML (method="freedman_lane_joint")
    - Full protocol and execution artifacts inspection via print_protocol_usage_table

References:
    - Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature
      extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium
      on Electronic Imaging: Science and Technology, 1905, 861-870.
    - Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
      Newey, W., & Robins, J. (2018). Double/debiased machine learning for
      treatment and structural parameters. The Econometrics Journal, 21(1), C1-C68.
    - Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
    - Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported
      significance levels. Journal of Business & Economic Statistics, 1(4), 292-298.
"""

# %%
# ============================================================================
# Load data — Breast Cancer Wisconsin Diagnostic
# ============================================================================

breast_cancer = fetch_ucirepo(id=17)
features_bc = ["radius1", "texture1", "smoothness1", "compactness1", "perimeter1"]
X_bc = breast_cancer.data.features[features_bc]
y_bc = pd.DataFrame(
    {"Diagnosis": (breast_cancer.data.targets["Diagnosis"] == "M").astype(int)}
)

print_dataset_info_table(
    name=breast_cancer.metadata.name,
    X=X_bc,
    y=y_bc,
    target_description="malignant (1) vs. benign (0)",
)

# %%
# ============================================================================
# Confounder screening
# ============================================================================

confounder_results_bc = identify_confounders(X_bc, y_bc, family="logistic")
print_confounder_table(confounder_results_bc)

# Nuclear perimeter represents the dominant morphological size confounder
confounders_bc = ["perimeter1"]

# %%
# ============================================================================
# Logistic Freedman–Lane (1983) individual — baseline (linear confounding)
# ============================================================================

res_standard_fl_bc = randomization_test_regression(
    X_bc,
    y_bc,
    confounders=confounders_bc,
    family="logistic",
    method="freedman_lane",
    n_randomizations=500,
    random_state=42,
)
print_results_table(res_standard_fl_bc)

# %%
# ============================================================================
# Logistic Double/Debiased Machine Learning (DML) via Random Forest
# ============================================================================
# In standard linear logistic FL, nuclear compactness (compactness1) is masked
# by collinearity with overall perimeter (p = 0.615). Cross-fitting both the binary
# outcome and continuous predictors via paired Random Forest learners debiases
# non-linear cell morphology, revealing compactness as a significant marker (p = 0.037).

reduced_model_rf = (
    RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42),
    RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42),
)

res_dml_bc = randomization_test_regression(
    X_bc,
    y_bc,
    confounders=confounders_bc,
    family="logistic",
    method="freedman_lane",
    reduced_model=reduced_model_rf,
    n_randomizations=500,
    random_state=42,
)
print_results_table(res_dml_bc)
print_diagnostics_table(res_dml_bc)

# %%
# ============================================================================
# Side-by-side comparison: Standard FL vs. DML
# ============================================================================

print_comparison_table([("Standard FL", res_standard_fl_bc), ("DML (RF)", res_dml_bc)])

# %%
# ============================================================================
# Logistic Freedman–Lane (1983) joint — Double/Debiased Machine Learning (DML)
# ============================================================================

res_dml_joint_bc = randomization_test_regression(
    X_bc,
    y_bc,
    confounders=confounders_bc,
    method="freedman_lane_joint",
    reduced_model=reduced_model_rf,
    n_randomizations=500,
    random_state=42,
)
print_joint_results_table(res_dml_joint_bc)

# %%
# ============================================================================
# Protocol and execution artifacts
# ============================================================================

print_protocol_usage_table(res_dml_bc)

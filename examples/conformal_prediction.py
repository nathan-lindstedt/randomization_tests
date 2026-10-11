# %% [markdown]
"""
Example: Conformal Prediction (Distribution-Free Predictive Uncertainty in Continuous Regression)
Dataset: Real Estate Valuation (UCI Machine Learning Repository ID=477)

Dataset Context & Theoretical Background:
    The Real Estate Valuation dataset (Yeh & Hsu, 2018) comprises 414 residential
    property transactions from Sindian District, New Taipei City, Taiwan (2012–2013).
    The dependent variable is `Y house price of unit area` (measured in 10,000 New
    Taiwan Dollar per Ping, where 1 Ping ≈ 3.3 square meters).

    While hypothesis testing via permutation regression evaluates whether spatial
    covariates (such as proximity to transit or structural building age) have
    statistically significant effects on property valuation, real-world decision-makers
    require out-of-sample prediction intervals: "Given an unseen property, in what
    range will its market transaction price fall with 95% confidence?"

    Classical parametric prediction intervals rely on rigid Gaussian and homoscedasticity
    assumptions:
        y_hat_new ± t_{n-p, 1-alpha/2} * s * sqrt(1 + x_new^T (X^T X)^{-1} x_new)
    In observational real estate markets, these assumptions fail catastrophically:
    1. Spatial Heteroscedasticity: Properties situated in dense transit corridors exhibit
       tight price clustering, whereas properties in suburban or mountain outskirts display
       immense price variance.
    2. Non-Gaussian Disturbances: Speculative luxury purchases induce heavy right-tailed
       residuals that violate Gaussian error models, causing classical intervals to under-cover.

Methodological Rationale for Resampling Tests:
    1. Bridging Permutation Exchangeability to Predictive Uncertainty:
       Just as permutation tests derive valid p-values exclusively from exchangeability
       rather than parametric distribution laws, conformal prediction (Vovk et al. 2005)
       inverts nonconformity order statistics to guarantee valid out-of-sample coverage:
           P(Y_new in C(X_new)) >= 1 - alpha
       The calibration constant rank ceil((n_cal + 1)(1 - alpha)) shares the exact
       mathematical +1 foundation with Phipson & Smyth (2010) permutation p-values:
       both stem from the uniform ordering distribution of n+1 exchangeable variables.
    2. Locally-Weighted Kernel Adaptation:
       Standard split conformal intervals have fixed width across all test points.
       By integrating the Kernel protocol (GaussianKernel with median heuristic;
       Papadopoulos et al. 2008), squared calibration residuals are smoothed to
       estimate local variance sigma_hat(X). Nonconformity scores are studentized by
       local scale, producing narrower intervals for urban properties and wider
       intervals for high-uncertainty remote homes.
    3. Sample-Efficient Leave-One-Out Uncertainty via Jackknife+:
       Jackknife+ (Barber, Candès, Ramdas & Tibshirani 2021) evaluates leave-one-out
       residuals without splitting the dataset, utilizing all training points for
       both model fitting and uncertainty calibration under a distribution-free
       1 - 2*alpha theoretical guarantee.
    4. Hierarchical Clustered Conformal Prediction:
       When observations share spatial correlation within geographic zones, observation
       exchangeability is violated. Clustered conformal prediction (Dunn, Wasserman &
       Ramdas 2022) calibrates cluster-maximum nonconformity scores, guaranteeing
       valid simultaneous coverage across clusters.

Features Selected for Modeling:
    - X1 transaction date: Macroeconomic appreciation and transaction timing.
    - X2 house age: Structural wear and building depreciation (0 to 43.8 years).
    - X3 distance to the nearest MRT station: Metric walking distance to rapid transit hubs.
    - X4 number of convenience stores: Local commercial density and foot traffic.
    - X5 latitude & X6 longitude: Geographic coordinates establishing the spatial surface.

Demonstrates:
    - Finite-sample exact split conformal prediction (method="split")
    - Locally-weighted adaptive conformal prediction with GaussianKernel
    - Jackknife+ leave-one-out conformal prediction (method="jackknife_plus")
    - Clustered conformal prediction respecting spatial neighborhood exchangeability
    - Out-of-sample empirical test coverage evaluation (result.evaluate / evaluate_conformal)
    - Feature-stratified adaptivity analysis across transit accessibility corridors
    - Binary classification prediction sets with multi-class ambiguous biopsy detection
    - Formatted ASCII display tables via print_conformal_table and print_conformal_evaluation_table

References:
    - Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
    - Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive inference
      with the jackknife+. Annals of Statistics, 49(1), 486–507.
    - Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2008). Inductive confidence
      machines for regression. ECML 2002, LNAI 2430, 345–356.
    - Dunn, R., Wasserman, L., & Ramdas, A. (2022). Distribution-free prediction sets for two-layer
      hierarchical models. Journal of the American Statistical Association, 1–12.
    - Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and
      distribution-free uncertainty quantification. arXiv:2107.07511.
    - Yeh, I. C., & Hsu, T. K. (2018). Building real estate valuation models with comparative
      approach through case-based reasoning. Applied Soft Computing, 65, 260-271.
"""

# %%
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    conformal_prediction,
    print_conformal_evaluation_table,
    print_conformal_table,
    print_dataset_info_table,
)
from randomization_tests._kernels import GaussianKernel

# %%
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
    target_description="transaction price per unit area (10,000 NTD/Ping)",
)

# %%
# ============================================================================
# Out-of-Sample Train/Test Partition (80% Train-Calibration / 20% Test)
# ============================================================================
# In rigorous statistical practice, conformal prediction must be validated on
# an independent held-out test cohort to verify that empirical coverage
# satisfies P(Y_test in C(X_test)) >= 1 - alpha across unseen data points.

rng = np.random.default_rng(42)
n_total = len(X)
n_test = int(np.round(0.20 * n_total))
perm_indices = rng.permutation(n_total)
train_idx, test_idx = perm_indices[:-n_test], perm_indices[-n_test:]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

print(
    f"Data Partition: {len(X_train)} training/calibration observations, {len(X_test)} held-out test observations."
)

# %%
# ============================================================================
# Population Test Evaluation — Split Conformal vs. Locally-Weighted Kernel
# vs. Jackknife+
# ============================================================================
# We evaluate out-of-sample prediction intervals across all 83 held-out test
# properties.

# 1. Standard Split Conformal (Homoscedastic baseline)
res_split_test = conformal_prediction(
    X_train,
    y_train,
    X_test,
    confidence_level=0.95,
    method="split",
    random_state=42,
)
eval_split = res_split_test.evaluate(y_test)
print_conformal_evaluation_table(
    eval_split,
    title="Split Conformal — Out-of-Sample Test Set Evaluation (N=83)",
)

# 2. Locally-Weighted Adaptive Conformal (Gaussian Kernel)
kernel_adaptive = GaussianKernel(sigma="median")
res_kernel_test = conformal_prediction(
    X_train,
    y_train,
    X_test,
    kernel=kernel_adaptive,
    confidence_level=0.95,
    method="split",
    random_state=42,
)
eval_kernel = res_kernel_test.evaluate(y_test)
print_conformal_evaluation_table(
    eval_kernel,
    title="Adaptive Kernel Conformal — Out-of-Sample Test Set Evaluation (N=83)",
)

# 3. Jackknife+ Predictive Inference (Leave-One-Out)
res_jk_test = conformal_prediction(
    X_train,
    y_train,
    X_test,
    method="jackknife_plus",
    confidence_level=0.95,
)
eval_jk = res_jk_test.evaluate(y_test)
print_conformal_evaluation_table(
    eval_jk,
    title="Jackknife+ Conformal — Out-of-Sample Test Set Evaluation (N=83)",
)

# %%
# ============================================================================
# Feature-Stratified Coverage (FSC) Across Transit Distance Corridors
# ============================================================================
# Standard split conformal guarantees marginal coverage averaged over the
# population,but can exhibit conditional under-coverage across difficult
# feature sub-populations.Using the library's built-in Feature-Stratified
# Coverage (FSC) evaluation (Angelopoulos & Bates 2021), we partition
# properties into 3 spatial corridors:
# - Near Station (< 500m)
# - Intermediate (500 - 1,500m)
# - Remote Outskirts (> 1,500m)

dist_mrt = X_test["X3 distance to the nearest MRT station"].values
transit_corridors = np.where(
    dist_mrt < 500.0,
    "1. Near Station (< 500m)",
    np.where(dist_mrt <= 1500.0, "2. Mid Distance (500-1500m)", "3. Remote (> 1500m)"),
)

eval_kernel_stratified = res_kernel_test.evaluate(y_test, groups=transit_corridors)
print_conformal_evaluation_table(
    eval_kernel_stratified,
    title="Locally-Weighted Conformal — Feature-Stratified Coverage (Transit)",
)

# %%
# ============================================================================
# Exemplar Case Studies — Urban Transit-Adjacent vs. Remote Outskirts
# ============================================================================
# To demonstrate what a decision-maker receives when querying an individual property,
# we inspect two real exemplar properties from the held-out test cohort:
# Property A: A modern apartment located immediately adjacent to an MRT station (23.4m).
# Property B: A remote home located deep in the rural mountain outskirts (6,396.3m).
# Using the index= parameter, print_conformal_table renders detailed single-case predictions
# directly from the batch-calibrated models without re-running calibration.

test_idx_near = int(np.argmin(X_test["X3 distance to the nearest MRT station"].values))
test_idx_far = int(np.argmax(X_test["X3 distance to the nearest MRT station"].values))

dist_near = X_test.iloc[test_idx_near]["X3 distance to the nearest MRT station"]
dist_far = X_test.iloc[test_idx_far]["X3 distance to the nearest MRT station"]

# Split Conformal (Fixed Width) on Near Property
print_conformal_table(
    res_split_test,
    index=test_idx_near,
    title=f"Split Conformal — Transit-Adjacent Property ({dist_near:.1f}m to MRT)",
)

# Locally-Weighted Adaptive Conformal (Adapts width to local covariate density)
print_conformal_table(
    res_kernel_test,
    index=test_idx_near,
    title=f"Locally-Weighted Conformal — Transit-Adjacent Property ({dist_near:.1f}m to MRT)",
)

print_conformal_table(
    res_kernel_test,
    index=test_idx_far,
    title=f"Locally-Weighted Conformal — Remote Outskirts ({dist_far:.1f}m to MRT)",
)

# Jackknife+ Predictive Inference (Leave-One-Out)
print_conformal_table(
    res_jk_test,
    index=test_idx_near,
    title="Jackknife+ Conformal — Full Sample Leave-One-Out (Worst-Case 1-2α)",
)

# Clustered Conformal Prediction (Spatial Transit Zones)
# For clustered conformal, we calibrate with groups and inspect the near property
transit_zones_train = (
    X_train["X3 distance to the nearest MRT station"].values // 500
).astype(int)

res_clustered_near = conformal_prediction(
    X_train,
    y_train,
    X_test.iloc[test_idx_near : test_idx_near + 1],
    groups=transit_zones_train,
    confidence_level=0.95,
    random_state=42,
)
print_conformal_table(
    res_clustered_near,
    title="Clustered Conformal — Spatial Transit Neighborhoods",
)


# %% [markdown]
"""
Example: Conformal Prediction Sets (Distribution-Free Classification Sets)
Dataset: Breast Cancer Wisconsin Diagnostic (UCI Machine Learning Repository ID=17)

Dataset Context & Theoretical Background:
    The Breast Cancer Wisconsin Diagnostic dataset (Street, Wolberg & Mangasarian,
    1993) contains 569 fine needle aspiration (FNA) biopsies of breast masses.
    The clinical outcome is histological malignancy diagnosis (`M` = Malignant,
    coded as 1; `B` = Benign, coded as 0).

    In clinical oncology, forced binary classifications based on arbitrary probability
    thresholds (e.g. y_hat = 1 iff P(Malignant | X) >= 0.50) present profound ethical
    and clinical hazards. When a biopsy exhibits ambiguous nuclear contour morphology,
    a forced decision suppresses diagnostic uncertainty: a false negative delays
    critical oncological therapy, while a false positive triggers invasive surgery.

    Conformal prediction sets (Vovk et al. 2005; Sadinle, Lei & Wasserman 2019)
    resolve this by producing set-valued predictions C(X_new) that guarantee
    marginal coverage at 1 - alpha:
        P(Y_new in C(X_new)) >= 1 - alpha
    - Unambiguous biopsies yield singleton prediction sets: {0} for benign or {1} for malignant.
    - Borderline, morphologically atypical biopsies yield multi-class prediction sets:
      {0, 1}, explicitly signaling to clinical pathologists that the case is diagnostically
      ambiguous and warrants immediate secondary biopsy review.

Features Selected for Modeling:
    - radius1: Mean of distances from center to perimeter points of cell nuclei.
    - texture1: Standard deviation of gray-scale values in the nuclear image.
    - smoothness1: Local variation in radius lengths (surface contour irregularity).
    - compactness1: Perimeter^2 / area - 1.0 (nuclear boundary concavity and packing).
    - perimeter1: Nuclear size perimeter measurement.

Methodological Rationale for Resampling Tests:
    1. Calibrating on Nonconformity Scores:
       The nonconformity score for candidate class y_i is S_i = 1 - P_hat(Y = y_i | X_i).
       Calibration determines the exact quantile threshold q_hat at rank
       ceil((n_cal + 1)(1 - alpha)). Any class whose predicted probability satisfies
       P_hat(Y = c | X_new) >= 1 - q_hat is included in the conformal prediction set.
    2. Finite-Sample Clinical Guarantee:
       Under exchangeability of patient biopsies, the true histological outcome is
       guaranteed to be contained in the conformal prediction set at least 95% of the time.

Demonstrates:
    - Split conformal prediction sets for binary classification (family="logistic")
    - Nonconformity evaluation directly on class probability margins
    - Singleton vs. ambiguous prediction set identification ({0}, {1}, vs {0, 1})
    - Formatted ASCII display tables via print_conformal_table with guarantee tier banners

References:
    - Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature
      extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium
      on Electronic Imaging: Science and Technology, 1905, 861-870.
    - Sadinle, M., Lei, J., & Wasserman, L. (2019). Least ambiguous prediction sets
      for classification with an application to medical diagnostics. Journal of the
      American Statistical Association, 114(526), 723–734.
    - Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
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
# Out-of-Sample Cohort Partition (80% Train-Calibration / 20% Test Cohort)
# ============================================================================
# To test whether the 95% marginal coverage guarantee holds on unseen biopsies and
# quantify the proportion of decisive vs. ambiguous prediction sets, we partition
# the 569 biopsies into 455 training/calibration cases and 114 held-out test patients.

perm_bc = rng.permutation(len(X_bc))
n_test_bc = int(np.round(0.20 * len(X_bc)))
train_bc_idx, test_bc_idx = perm_bc[:-n_test_bc], perm_bc[-n_test_bc:]

X_bc_train, y_bc_train = X_bc.iloc[train_bc_idx], y_bc.iloc[train_bc_idx]
X_bc_test, y_bc_test = X_bc.iloc[test_bc_idx], y_bc.iloc[test_bc_idx]

print(
    f"Biopsy Cohort Partition: {len(X_bc_train)} training/calibration biopsies, {len(X_bc_test)} held-out test biopsies."
)

# %%
# ============================================================================
# Population Cohort Evaluation — Conformal Prediction Sets (N=114)
# ============================================================================
# We evaluate conformal prediction sets across all 114 test biopsies at 95% confidence.

res_bc_cohort = conformal_prediction(
    X_bc_train,
    y_bc_train,
    X_bc_test,
    family="logistic",
    confidence_level=0.95,
    random_state=42,
)
eval_bc = res_bc_cohort.evaluate(y_bc_test)
print_conformal_evaluation_table(
    eval_bc,
    title="Conformal Prediction Sets — Test Biopsy Cohort Evaluation (N=114)",
)

# %%
# ============================================================================
# Clinical Case Studies — Malignant Singleton vs. Benign Singleton vs. Ambiguous Set
# ============================================================================
# To illustrate individual clinical decision support, we inspect three actual patient
# biopsies directly from the test cohort result res_bc_cohort using index=:
# Patient A (test row 0): Confirmed malignancy -> Decisive Singleton {1}.
# Patient B (test row 6): Confirmed benign lesion -> Decisive Singleton {0}.
# Patient C (test row 4): Borderline nuclear morphology -> Ambiguous {0, 1}.

# Patient A: Confirmed Malignant Case (Singleton {1})
print_conformal_table(
    res_bc_cohort,
    index=0,
    title="Clinical Case 1: Decisive Malignant Biopsy (Singleton Set {1})",
)

# Patient B: Confirmed Benign Case (Singleton {0})
print_conformal_table(
    res_bc_cohort,
    index=6,
    title="Clinical Case 2: Decisive Benign Biopsy (Singleton Set {0})",
)

# Patient C: Borderline Ambiguous Case (Multi-class {0, 1} Trigger for Secondary Review)
print_conformal_table(
    res_bc_cohort,
    index=4,
    title="Clinical Case 3: Borderline Biopsy (Ambiguous {0, 1} Set — Secondary Review Trigger)",
)

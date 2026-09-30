# %% [markdown]
"""
Example: Knockoff Filters for False Discovery Rate (FDR) Controlled Feature Selection
Techniques: Barber & Candès (2015) Fixed-X Exact FDR & Candès et al. (2018) Model-X Approximate FDR

Theoretical Context & Problem Formulation:
    In modern high-dimensional inference, researchers routinely test tens, hundreds, or
    thousands of candidate explanatory variables (genomic markers, spectroscopic features,
    or macroeconomic indicators) against an outcome of interest. Testing marginal
    associations via individual hypothesis tests introduces severe multiple testing inflation:
    at a nominal significance level alpha = 0.05, testing 100 null features yields ~5 false
    positives in expectation.

    While classical corrections like Benjamini-Hochberg (1995) control the False Discovery Rate
    (FDR) under independent or PRDS test statistics, they rely on valid marginal p-values from
    univariate regressions, which suffer from omitted variable bias in collinear designs.

    The Knockoff framework solves this by constructing synthetic "knockoff" features X_tilde
    that mimic the exact correlation structure of the original design X, but are conditionally
    independent of the response Y given X:
        H_0: beta_j = 0  ==>  Swapping X_j with X_tilde_j preserves the joint distribution.

Theoretical Guarantee Tiers:
    1. Fixed-X Knockoffs (Barber & Candès, 2015, Ann. Statist.):
       - Guarantee: Finite-sample exact FDR control: E[FDP] <= q.
       - Assumptions: Linear Gaussian model Y | X ~ N(X beta, sigma^2 I). NO distributional
         assumptions on X (design X is treated as fixed and deterministic).
       - Geometric Construction: Requires n >= 2p. Uses the orthonormal null-space complement
         C in R^{n x p} from the QR factorization of X (C^T X = 0, C^T C = I).
       - Invariant: Exactly preserves the Gram matrix to machine precision:
             X_tilde^T X_tilde == X^T X
             X^T X_tilde == X^T X - diag(s)
    2. Model-X Knockoffs (Candès, Fan, Janson & Lv, 2018, JRSS-B):
       - Guarantee: Model-X approximate FDR control under estimated Sigma: E[FDP] <= q + TV error.
       - Assumptions: X ~ N(0, Sigma). Makes NO assumptions on the response distribution Y | X
         (supports non-linear, logistic, Poisson, or deep neural networks).
       - Geometric Construction: Generates conditional Gaussian knockoffs:
             X_tilde | X ~ N(X(I - Sigma^{-1} diag(s)), 2 diag(s) - diag(s) Sigma^{-1} diag(s)).
       - Shrinkage: Uses Ledoit-Wolf shrinkage to ensure lambda_min(Sigma_hat) > 0 and prevent
         eigenvalue collapse under collinear designs.
       - Swap Diagnostic: Uses the package's own MMD two-sample test (Gretton et al., 2012)
         to empirically audit the swap-exchangeability assumption.

Knockoff+ Threshold:
    Feature importance statistics W_j = |beta_j| - |beta_tilde_j| satisfy the coin-flip
    property: under the null H_0 (beta_j = 0), W_j is symmetric around zero.
    The Barber & Candès (2015) knockoff+ threshold is:
        tau_+ = min { t > 0 : (1 + #{j: W_j <= -t}) / (#{j: W_j >= t} v 1) <= q }
    Selecting features with W_j >= tau_+ provably guarantees E[FDP] <= q.
"""

# %%
import warnings

import numpy as np
import pandas as pd

from randomization_tests import knockoff_test, print_knockoff_table

# %% [markdown]
# ### 1. Synthetic Biomarker Experiment with Planted Signals
# We generate an observational study with $n = 200$ subjects and $p = 25$ candidate biomarkers.
# Covariates exhibit realistic autoregressive AR(1) spatial/temporal correlation ($\rho = 0.4$).
# Exactly 8 of the 25 biomarkers have true non-zero regulatory effects on the response.

# %%
rng = np.random.default_rng(2026)
n, p = 200, 25
rho = 0.4

# AR(1) feature covariance
idx = np.arange(p)
Sigma_true = rho ** np.abs(np.subtract.outer(idx, idx))

# Generate features
X_raw = rng.multivariate_normal(np.zeros(p), Sigma_true, size=n)
feature_names = [f"Biomarker_{i + 1:02d}" for i in range(p)]
df_X = pd.DataFrame(X_raw, columns=feature_names)

# Plant 8 true positive signals (Biomarkers 1 to 8)
beta_true = np.zeros(p)
beta_true[:8] = [3.5, -3.2, 3.8, 3.0, -3.4, 3.1, 3.6, -3.3]

# Generate Gaussian response
y = X_raw @ beta_true + rng.normal(0.0, 1.0, size=n)

print(f"Dataset generated: {n} observations, {p} candidate biomarkers.")
print("Planted causal signals: Biomarker_01 through Biomarker_08 (8 true positives).")
print("Null biomarkers: Biomarker_09 through Biomarker_25 (17 null features).")

# %% [markdown]
# ### 2. Model-X Knockoffs with Ledoit-Wolf Covariance Shrinkage
# We execute Model-X knockoffs at a target False Discovery Rate $q = 0.20$.
# Ledoit-Wolf shrinkage stabilizes the estimated covariance matrix, and MMD tests
# audit the pairwise swap-exchangeability of candidate features.

# %%
with warnings.catch_warnings():
    # Suppress the advisory notice suggesting Fixed-X for linear models with n >= 2p
    warnings.simplefilter("ignore", UserWarning)
    res_model_x = knockoff_test(
        df_X,
        y,
        method="equicorrelated",
        fdr_level=0.20,
        shrinkage=True,
        random_state=42,
    )

print_knockoff_table(res_model_x, title="Model-X Knockoff Filter (Biomarker Selection)")

# %%
# Verify empirical FDR and power
selected_mx = res_model_x.selected_features
true_positives_mx = [f for f in selected_mx if int(f.split("_")[1]) <= 8]
false_positives_mx = [f for f in selected_mx if int(f.split("_")[1]) > 8]

fdp_mx = len(false_positives_mx) / max(1, len(selected_mx))
power_mx = len(true_positives_mx) / 8.0

print(
    f"Model-X Selections: {len(selected_mx)} features ({len(true_positives_mx)} TP, {len(false_positives_mx)} FP)"
)
print(f"Empirical FDP: {fdp_mx:.2%} (Target FDR bound: 20.0%)")
print(f"Empirical Power: {power_mx:.2%}")

# %% [markdown]
# ### 3. Fixed-X Knockoffs (Barber & Candès 2015 Finite-Sample Exact FDR)
# In the $n \ge 2p$ linear regression regime ($200 \ge 50$), Fixed-X knockoffs provide
# a strictly stronger guarantee: finite-sample exact FDR control treating $X$ as fixed,
# without requiring Gaussianity of the feature matrix.

# %%
res_fixed_x = knockoff_test(
    df_X,
    y,
    method="fixed_x",
    fdr_level=0.20,
    family="linear",
    random_state=42,
)

print_knockoff_table(
    res_fixed_x, title="Fixed-X Knockoff Filter (Finite-Sample Exact FDR)"
)

# %%
selected_fx = res_fixed_x.selected_features
true_positives_fx = [f for f in selected_fx if int(f.split("_")[1]) <= 8]
false_positives_fx = [f for f in selected_fx if int(f.split("_")[1]) > 8]

fdp_fx = len(false_positives_fx) / max(1, len(selected_fx))
power_fx = len(true_positives_fx) / 8.0

print(
    f"Fixed-X Selections: {len(selected_fx)} features ({len(true_positives_fx)} TP, {len(false_positives_fx)} FP)"
)
print(f"Empirical FDP: {fdp_fx:.2%} (Target FDR bound: 20.0%)")
print(f"Empirical Power: {power_fx:.2%}")

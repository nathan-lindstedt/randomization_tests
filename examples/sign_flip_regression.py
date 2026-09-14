# %% [markdown]
"""
Example: Sign-Flip Randomization Test for Linear Regression
Dataset: Energy Efficiency (UCI Machine Learning Repository ID=242)

Dataset Context & Theoretical Background:
    The Energy Efficiency dataset (Tsanas & Xifara, 2012) investigates the
    thermal performance of residential architectural building forms simulated
    using Ecotect software across 12 distinct building configurations. The primary
    outcome is `Y1` (Heating Load, measured in kilowatt-hours per square meter,
    kWh/m^2), representing the thermal energy required to maintain conditioned
    interior comfort.

    Classical permutation tests rely on the assumption of exchangeability: under
    the null hypothesis H_0: beta_j = 0, the joint distribution of the error
    vector eps = (eps_1, ..., eps_n)^T is invariant under all n! permutations pi in S_n.
    When observations exhibit conditional heteroscedasticity (e.g. Var(eps_i | X_i)
    varies across design points), standard permutation tests can suffer inflated
    false positive rates because swapping errors across points with different
    variances violates exchangeability.

    The Sign-Flip Randomization Test (Fisher, 1935; Pitman, 1937; DiCiccio & Efron,
    1992) replaces exchangeability with the weaker assumption of **conditional
    symmetry**: under the null hypothesis, the distribution of each error is
    symmetric about zero:

        P(eps_i <= -u | X_i) = P(eps_i >= u | X_i),   forall u > 0,  i = 1, ..., n

    Instead of permuting the residual vector across observation indices, each
    residual e_i is multiplied independently by a Rademacher random variable
    s_i in {-1, +1} where P(s_i = +1) = P(s_i = -1) = 1/2. Because sign-flipping
    operates point-by-point, it remains exact even when errors are heteroscedastic,
    provided each conditional error distribution remains symmetric.

Features Selected for Modeling:
    - compactness (X1): Relative compactness ratio of building shape. Lower
      compactness implies greater exterior surface area relative to volume,
      increasing conductive thermal heat loss.
    - height (X5): Overall building height (meters). Directly dictates interior
      stack effect and buoyancy-driven thermal stratification.
    - orientation (X6): Cardinal building orientation (2: North, 3: East,
      4: South, 5: West), governing solar radiation capture.
    - glazing_area (X7): Glazing area percentage of exterior facade (0%, 10%, 25%, 40%).
      Primary vector of window conductive and radiant thermal transfer.
    - glazing_dist (X8): Spatial distribution of fenestration (uniform, north,
      south, east, west).

    Note: Features X2 (Surface Area), X3 (Wall Area), and X4 (Roof Area) are
    exact deterministic linear combinations of X1 and X5; including them causes
    singular design matrices and infinite VIFs, and they are appropriately omitted.

Methodological Highlights:
    1. Pre-test Symmetry Verification:
       Before executing a sign-flip test, the conditional symmetry assumption must
       be empirically evaluated. The function `validate_symmetry()` performs a
       Wilcoxon signed-rank test and evaluates sample skewness on OLS residuals.
    2. Freedman-Lane Sign-Flip Framework:
       Sign-flipping is embedded into the Freedman-Lane partial regression
       framework: for each predictor j, Y is regressed onto X_{-j} to obtain
       reduced-model residuals e_{-j}. Rademacher signs are flipped: e* = s * e_{-j},
       reconstructing Y* = X_{-j} hat{gamma} + e* to re-estimate beta_j.
    3. Head-to-Head Comparison:
       Side-by-side comparison with the standard Freedman-Lane permutation test
       isolates the effect of the randomization mechanism (permutation vs.
       Rademacher sign-flip) holding the partial regression architecture constant.

Demonstrates:
    - randomization="sign_flip" -- Rademacher sign-flip test via Freedman-Lane
    - validate_symmetry() -- Wilcoxon signed-rank diagnostic table
    - print_symmetry_table() -- structured presentation of symmetry diagnostics
    - Direct head-to-head comparison table between permutation and sign-flip
    - Execution and protocol artifacts inspection via print_protocol_usage_table

References:
    - Tsanas, A., & Xifara, A. (2012). Accurate quantitative estimation of energy
      performance of residential buildings using statistical machine learning tools.
      Energy and Buildings, 49, 560-567.
    - Fisher, R. A. (1935). The Design of Experiments. Oliver & Boyd.
    - Pitman, E. J. G. (1937). Significance tests which may be applied to samples
      from any populations. Journal of the Royal Statistical Society, 4(1), 119-130.
    - DiCiccio, T. J., & Efron, B. (1992). More accurate confidence intervals in
      exponential families. Biometrika, 79(2), 231-245.
    - Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported
      significance levels. Journal of Business & Economic Statistics, 1(4), 292-298.
"""

# %%
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    print_comparison_table,
    print_dataset_info_table,
    print_diagnostics_table,
    print_protocol_usage_table,
    print_results_table,
    print_symmetry_table,
    randomization_test_regression,
)

# %%
# ============================================================================
# Load data
# ============================================================================

ds = fetch_ucirepo(id=242)

# Drop X2, X3, X4 (surface/wall/roof area) — linearly derived from
# X1 (relative compactness), causing degenerate VIF values.
feature_cols = ["X1", "X5", "X6", "X7", "X8"]
X = ds.data.features[feature_cols].copy()
X.columns = ["compactness", "height", "orientation", "glazing_area", "glazing_dist"]
y = ds.data.targets[["Y1"]].copy()  # Heating load

print_dataset_info_table(
    name="Energy Efficiency",
    X=X,
    y=y,
    target_name="Y1",
    target_description="heating load (kWh/m²)",
)

# %%
# ============================================================================
# Sign-flip test
# ============================================================================

results_sf = randomization_test_regression(
    X,
    y,
    family="linear",
    n_randomizations=2_000,
    random_state=42,
    randomization="sign_flip",
)

print_results_table(results_sf)
print_diagnostics_table(results_sf)

# %%
# ============================================================================
# Symmetry diagnostic — validate the sign-flip assumption
# ============================================================================
# The Wilcoxon signed-rank test evaluates whether residuals are symmetric about
# zero, validating the conditional symmetry assumption required by sign-flipping.
# The diagnostic is extracted directly from the completed test result.

print_symmetry_table(results_sf)

# %%
# ============================================================================
# Freedman–Lane permutation test for comparison
# ============================================================================

# The sign-flip test uses the Freedman–Lane framework internally:
# for each feature j, it fits Y ~ X_{-j}, extracts reduced-model
# residuals, then flips their signs.  Comparing against
# method="freedman_lane" isolates the randomization mechanism
# (permutation vs Rademacher sign-flip) while holding the framework
# constant.  This is an apples-to-apples comparison.

results_perm = randomization_test_regression(
    X,
    y,
    method="freedman_lane",
    confounders=[],
    family="linear",
    n_randomizations=2_000,
    random_state=42,
)

print_results_table(results_perm)
print_diagnostics_table(results_perm)

# %%
# ============================================================================
# Side-by-side comparison
# ============================================================================

print_comparison_table(
    [
        ("Sign-Flip", results_sf),
        ("Permutation", results_perm),
    ],
    title="Sign-Flip vs. Freedman\u2013Lane P-Value Comparison",
)

# %%
# ============================================================================
# Execution & Protocol Artifacts
# ============================================================================
# Inspect the internal execution context from the completed test result:
# backend acceleration, batch convergence, fit metrics, and protocol properties.

print_protocol_usage_table(results_sf)

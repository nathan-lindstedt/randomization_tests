"""
Test Case 11: Sign-Flip Test for Linear Regression
Energy Efficiency dataset (UCI ML Repository ID=242)

Demonstrates:
- ``randomization_test_regression(randomization="sign_flip")`` — the
  sign-flip analogue of permutation testing for symmetric residual
  distributions
- ``validate_symmetry()`` — Wilcoxon signed-rank diagnostic
- Side-by-side comparison with Freedman–Lane permutation test
- Why Freedman–Lane is the natural comparator (both use reduced-
  model residuals; the only difference is permutation vs sign-flip)

Background
----------
The sign-flip test replaces the exchangeability assumption of
standard permutation tests with a weaker **symmetry** assumption:
under the null, residuals are symmetric about zero.  Instead of
permuting the residual vector, each residual is independently
multiplied by a Rademacher variable (+1 or -1).

This is particularly useful when:
- Residuals are symmetric but not identically distributed
  (heteroscedastic but symmetric errors)
- The analyst wants an alternative randomization scheme to
  cross-validate permutation test findings

The procedure follows the Freedman-Lane framework, replacing
permutation with sign-flipping in the residual randomization step.

Dataset
-------
768 samples of simulated building shapes.  Eight structural
features describe each building; we select five with low
collinearity (VIF < 5):

- **compactness** (X1 — relative compactness)
- **height** (X5 — overall height)
- **orientation** (X6 — orientation: 2/3/4/5)
- **glazing_area** (X7 — glazing area)
- **glazing_dist** (X8 — glazing area distribution)

Three geometry features (X2 surface area, X3 wall area,
X4 roof area) are dropped because they are linearly derived
from X1, producing degenerate VIF values.

Reference
---------
Tsanas, A. & Xifara, A. (2012). Accurate quantitative estimation
of energy performance of residential buildings using statistical
machine learning tools. *Energy and Buildings*, 49, 560-567.
"""

import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    print_comparison_table,
    print_dataset_info_table,
    print_diagnostics_table,
    print_results_table,
    print_symmetry_table,
    randomization_test_regression,
    validate_symmetry,
)

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
    n_observations=len(X),
    n_features=X.shape[1],
    feature_names=list(X.columns),
    target_name="Y1",
    target_description="heating load (kWh/m²)",
    y_range=(float(y.values.min()), float(y.values.max())),
    y_mean=float(y.values.mean()),
    y_var=float(y.values.var()),
)

# ============================================================================
# Symmetry diagnostic — validate the sign-flip assumption
# ============================================================================

# Fit a linear model to obtain residuals for the symmetry check.
model = LinearRegression().fit(X.values, np.ravel(y))
residuals = np.ravel(y) - model.predict(X.values)

sym = validate_symmetry(residuals)
print_symmetry_table(sym)

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

print_results_table(
    results_sf,
    title="Sign-Flip Test — Linear Regression",
)

print_diagnostics_table(
    results_sf,
    title="Sign-Flip Test — Extended Diagnostics",
)

# ============================================================================
# Freedman–Lane permutation test for comparison
# ============================================================================

# The sign-flip test uses the Freedman–Lane framework internally:
# for each feature j, it fits Y ~ X_{-j}, extracts reduced-model
# residuals, then flips their signs.  Comparing against
# method="freedman_lane" isolates the randomization mechanism
# (permutation vs Rademacher sign-flip) while holding the framework
# constant.  This is an apples-to-apples comparison.

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*without confounders.*")
    results_perm = randomization_test_regression(
        X,
        y,
        method="freedman_lane",
        confounders=[],
        family="linear",
        n_randomizations=2_000,
        random_state=42,
    )

print_results_table(
    results_perm,
    title="Freedman\u2013Lane (1983) Permutation Test \u2014 Linear Regression",
)

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

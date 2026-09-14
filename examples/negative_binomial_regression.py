# %% [markdown]
"""
Example: Negative Binomial Regression (Overdispersed Count Outcome)
Bike Sharing dataset (UCI ML Repository ID=275)

Demonstrates:
- ``family="negative_binomial"`` — explicit family selection for overdispersed counts
- ``calibrate()`` nuisance-parameter estimation: estimation of dispersion parameter :math:`\alpha`
- ter Braak (1992) permutation test with negative binomial deviance residuals
- Freedman–Lane (1983) individual and joint permutation tests
- Kennedy (1995) individual and joint permutation tests
- Negative-binomial-specific diagnostics (deviance, Pearson χ², dispersion ratio, alpha estimate)
- Stochastic reconstruction for overdispersed counts: :math:`Y^* \\sim \\mathrm{NegBin}(\\mu^*, \alpha)`
- Four-stage confounder sieve with negative binomial regression
- Execution and protocol artifacts inspection via ``print_protocol_usage_table``

Dataset
-------
17,379 hourly records of bike rental counts from the Capital Bikeshare system in
Washington, D.C. (2011–2012), subsampled to 300 observations for demo runtime.
The target variable is ``cnt`` (total count of rental bikes per hour).

The count distribution is severely overdispersed, with a marginal variance-to-mean
ratio of ≈ 174.4 (:math:`\\mathrm{Var} \\gg \\mathbb{E}`). Fitting a standard Poisson model
would severely understate standard errors and generate anticonservative inference.
The NB2 parameterisation (:math:`\\mathrm{Var}(Y \\mid X) = \\mu + \alpha\\mu^2`) incorporates
gamma-distributed unobserved heterogeneity to restore proper likelihood calibration.

Feature selection rationale
---------------------------
Five environmental and calendar predictors capture hourly commuter demand:

- **temp**: Normalized temperature in Celsius (divided by 41 max). Warmer weather
  increases cycling propensity.
- **hum**: Normalized relative humidity (divided by 100). High humidity dampens outdoor activity.
- **windspeed**: Normalized wind speed (divided by 67). Strong headwinds impede cycling.
- **workingday**: Indicator (1 = workday, 0 = weekend or holiday). Distinguishes commuter
  peaks from leisure cycling.
- **weathersit**: Categorical weather severity index (1: Clear, 2: Mist/Cloudy,
  3: Light Rain/Snow, 4: Heavy Precipitation). Adverse weather suppresses demand.

Methodological rationale
-------------------------
Negative binomial regression estimates the dispersion parameter :math:`\alpha` once on the
observed data during calibration and holds it fixed across the permutation loop. This
ensures the null hypothesis conditions on the calibrated nuisance variance structure,
avoiding unstable numerical optimization during permutation refitting.

Reference
---------
Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors
and background knowledge. *Progress in Artificial Intelligence*, 2(2–3), 113–127.
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

bike_sharing = fetch_ucirepo(id=275)
X_full = bike_sharing.data.features.copy()
y_full = bike_sharing.data.targets

# Subsample to 300 rows (permutation tests re-fit B × p GLMs, so
# keeping n moderate avoids excessive runtime for an example script).
rng = np.random.default_rng(42)
idx = rng.choice(len(X_full), size=300, replace=False)
X_sub = X_full.iloc[idx].reset_index(drop=True)
y_sub = y_full.iloc[idx].reset_index(drop=True)

# Select predictors: normalised temperature, humidity, wind speed,
# working-day indicator, and weather situation (ordinal 1–4).
X = pd.DataFrame(
    {
        "temp": X_sub["temp"].astype(float),
        "hum": X_sub["hum"].astype(float),
        "windspeed": X_sub["windspeed"].astype(float),
        "workingday": X_sub["workingday"].astype(float),
        "weathersit": X_sub["weathersit"].astype(float),
    }
)
y = y_sub.copy()

y_arr = np.ravel(y).astype(float)
var_mean_ratio = y_arr.var() / y_arr.mean()

print_dataset_info_table(
    name=bike_sharing.metadata.name,
    X=X,
    y=y,
    target_description="hourly rental bike count",
    extra_stats={"Var/Mean": f"{var_mean_ratio:.2f}  (>>1 → overdispersed)"},
)

# %%
# ============================================================================
# Family resolution
# ============================================================================

nb_family = resolve_family("negative_binomial")

print_family_info_table(
    y=y,
    explicit_family=nb_family,
)

# %%
# ============================================================================
# ter Braak (1992) — family="negative_binomial" (explicit)
# ============================================================================

results_ter_braak = randomization_test_regression(
    X, y, method="ter_braak", family="negative_binomial"
)
print_results_table(results_ter_braak)
print_diagnostics_table(results_ter_braak)

# %%
# ============================================================================
# Kennedy (1995) individual — family="negative_binomial"
# ============================================================================

results_kennedy = randomization_test_regression(
    X, y, method="kennedy", confounders=[], family="negative_binomial"
)
print_results_table(results_kennedy)
print_diagnostics_table(results_kennedy)

# %%
# ============================================================================
# Kennedy (1995) joint — family="negative_binomial"
# ============================================================================

results_kennedy_joint = randomization_test_regression(
    X, y, method="kennedy_joint", confounders=[], family="negative_binomial"
)
print_joint_results_table(results_kennedy_joint)

# %%
# ============================================================================
# Freedman–Lane (1983) individual — family="negative_binomial"
# ============================================================================

results_fl = randomization_test_regression(
    X, y, method="freedman_lane", confounders=[], family="negative_binomial"
)
print_results_table(results_fl)
print_diagnostics_table(results_fl)

# %%
# ============================================================================
# Freedman–Lane (1983) joint — family="negative_binomial"
# ============================================================================

results_fl_joint = randomization_test_regression(
    X, y, method="freedman_lane_joint", confounders=[], family="negative_binomial"
)
print_joint_results_table(results_fl_joint)

# %%
# ============================================================================
# Confounder identification
# ============================================================================

all_confounder_results = identify_confounders(X, y, family="negative_binomial")
print_confounder_table(all_confounder_results)

# %%
# ============================================================================
# Kennedy with identified confounders — family="negative_binomial"
# ============================================================================
# The confounder sieve identified that 'hum' (relative humidity) is confounded
# by 'weathersit' (weather severity). We execute Kennedy's permutation test
# controlling for 'weathersit' to estimate the partial effect of humidity.

target_predictor = "hum"
confounders = all_confounder_results[target_predictor].identified_confounders

results_kc = randomization_test_regression(
    X,
    y,
    method="kennedy",
    confounders=confounders,
    family="negative_binomial",
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

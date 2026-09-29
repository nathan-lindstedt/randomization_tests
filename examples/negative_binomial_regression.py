# %% [markdown]
"""
Example: Negative Binomial Regression (Overdispersed Count Outcome)
Dataset: Bike Sharing (UCI Machine Learning Repository ID=275)

Dataset Context & Theoretical Background:
    The Bike Sharing dataset (Fanaee-T & Gama, 2014) comprises 17,379 hourly
    records of rental bike counts from the Capital Bikeshare system in Washington,
    D.C. (2011–2012), subsampled here to 300 observations for rapid interactive
    execution. The response variable is `cnt`, the total number of bicycles
    rented per hour by registered and casual commuters.

    While count outcomes are discrete and non-negative, empirical social and
    behavioral processes frequently violate the equi-dispersion property of the
    Poisson distribution (where Var(Y) = E[Y]). In this hourly transit dataset,
    the marginal variance-to-mean ratio exceeds 170 (sample variance ≈ 32,800 vs.
    sample mean ≈ 189), representing extreme overdispersion driven by peak-hour
    clustering, unobserved commuter preferences, and sudden weather fluctuations.

    Fitting a standard Poisson GLM to severely overdispersed data severely
    understates standard errors, producing spuriously narrow confidence intervals
    and catastrophic false discovery rates. The Negative Binomial type-2 model
    (NB2; Cameron & Trivedi, 2013) resolves this by compounding the Poisson rate
    parameter with a gamma-distributed latent heterogeneity shock, yielding a
    quadratic variance function:

        Var(Y | X) = mu + alpha * mu^2,    mu = exp(alpha_0 + X * beta)

    where alpha > 0 represents the unobserved heterogeneity dispersion parameter.

Features Selected for Modeling:
    - temp: Normalized ambient temperature in degrees Celsius (scaled by 41 deg C max).
      Warmer temperatures strongly increase commuter willingness to cycle.
    - hum: Normalized relative humidity (scaled by 100 max). Oppressive humidity
      dampens outdoor exertion and suppresses cycling demand.
    - windspeed: Normalized wind speed (scaled by 67 max). Strong aerodynamic drag
      and head-winds impede cycling adoption.
    - workingday: Binary calendar indicator (1 = workday, 0 = weekend or holiday).
      Distinguishes structured bi-modal commuter peaks from leisure weekend trips.
    - weathersit: Categorical weather severity scale (1: Clear/Few clouds, 2: Mist/Cloudy,
      3: Light Rain/Snow, 4: Heavy Rain/Thunderstorm). Severe precipitation directly
      suppresses open-air transit demand.

Methodological Rationale for Resampling Tests:
    1. Fixed Dispersion Calibration:
       The dispersion parameter alpha is estimated once on the unpermuted observed
       data during the calibration phase (calibrate()) via maximum likelihood.
       Holding alpha fixed across all B permutations prevents numerical convergence
       failures on extreme permuted realizations, while correctly conditioning
       the null hypothesis on the observed overdispersion structure.
    2. Overdispersed Stochastic Negative Binomial Reconstruction:
       Permuted response-scale residuals e = y - mu_hat are reconstructed on the rate
       scale mu* = clip(mu_hat + pi(e), 1e-10, 1e8) and transformed into NB2
       parameters: size n = 1 / alpha, probability p = 1 / (1 + alpha * mu*).
       Drawing permuted outcomes stochastically via Y* ~ NegBin(n, p) preserves
       both the discrete count support and the overdispersed variance structure.
    3. Freedman–Lane and Kennedy Conditioning:
       Freedman–Lane (1983) and Kennedy (1995) condition on primary meteorological
       confounders (such as ambient temperature), isolating the partial contributions
       of humidity, wind, and calendar classification.

Demonstrates:
    - family="negative_binomial" -- explicit family selection for overdispersed counts
    - calibrate() nuisance-parameter estimation: estimation of dispersion parameter alpha
    - ter Braak (1992) permutation test with negative binomial deviance residuals
    - Freedman–Lane (1983) individual and joint permutation tests
    - Kennedy (1995) individual and joint permutation tests
    - Negative-binomial-specific diagnostics (deviance, Pearson χ², dispersion ratio, alpha estimate)
    - Stochastic reconstruction for overdispersed counts: Y* ~ NegBin(mu*, alpha)
    - Four-stage confounder sieve with negative binomial regression
    - Execution and protocol artifacts inspection via print_protocol_usage_table

References:
    - Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors
      and background knowledge. Progress in Artificial Intelligence, 2(2-3), 113-127.
    - Cameron, A. C., & Trivedi, P. K. (2013). Regression Analysis of Count Data
      (2nd ed.). Cambridge University Press.
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

# Quick Start

## Installation

### From source (editable)

```bash
git clone https://github.com/nathanlindstedt/randomization_tests.git
cd randomization_tests
pip install -e .
```

### With optional JAX backend

```bash
pip install -e ".[jax]"
```

## Minimal example

```python
import pandas as pd
from randomization_tests import (
    randomization_test_regression,
    print_diagnostics_table,
    print_results_table,
)

# Prepare data as DataFrames (pandas or Polars)
X = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [5, 4, 3, 2, 1]})
y = pd.DataFrame({"y": [2.1, 4.0, 5.8, 8.2, 9.9]})

# Run a ter Braak (1992) permutation test
results = randomization_test_regression(
    X, y,
    n_randomizations=1_000,
    method="ter_braak",
    random_state=42,
)

# Display a statsmodels-style results table
print_results_table(results)

# Display extended diagnostics (VIF, Cook's D, etc.)
print_diagnostics_table(results)
```

## Available methods

| Method | `method=` | Description |
|---|---|---|
| ter Braak (1992) | `"ter_braak"` | Permute residuals under the reduced model. Default. |
| Kennedy (1995) individual | `"kennedy"` | Partial out confounders, permute exposure residuals. |
| Kennedy (1995) joint | `"kennedy_joint"` | Test whether predictors collectively improve fit beyond confounders. |
| Freedman–Lane (1983) individual | `"freedman_lane"` | Permute full-model residuals, reconstruct from reduced-model fitted values. Better power than Kennedy when predictors are correlated. |
| Freedman–Lane (1983) joint | `"freedman_lane_joint"` | Joint version of Freedman–Lane. |
| Manly (1997) individual | `"manly"` | Permute Y directly; valid for ordinal/multinomial families. |
| Manly (1997) joint | `"manly_joint"` | Joint version of Manly. |
| Score individual | `"score"` | Score-test projection; supports all families including mixed-effects. |
| Score joint | `"score_joint"` | Joint version of score test. |
| Score exact | `"score_exact"` | Exact score test via PQL-fixed vmap IRLS; GLMM families only. |

Kennedy and Freedman–Lane methods require the `confounders` parameter
(a list of column names).

> **Note:** Ordinal and multinomial families do not support Freedman–Lane
> methods (residuals are ill-defined for these model types).

## Sign-flip tests

Sign-flip tests replace permutation with Rademacher (±1) multipliers
applied to residuals.  They are valid when the error distribution is
symmetric about zero — a weaker assumption than exchangeability.

```python
from randomization_tests import randomization_test_regression, validate_symmetry

# Check symmetry assumption before running
sym = validate_symmetry(y.values.ravel() - y.values.mean())
print(f"Symmetric: {sym['is_symmetric']}  (p={sym['p_value']:.3f})")

# Run a sign-flip test (Freedman–Lane with ±1 multipliers)
results = randomization_test_regression(
    X, y,
    n_randomizations=5_000,
    random_state=42,
    randomization="sign_flip",
)

print_results_table(results)
```

Sign-flip tests support the same families as permutation tests, except
ordinal and multinomial (which raise `ValueError`).  Confounders are
supported via the `confounders` parameter, identical to Freedman–Lane.

## Confounder identification

```python
from randomization_tests import identify_confounders, print_confounder_table, resolve_family

result = identify_confounders(X, y, predictor="x1", random_state=42)
print_confounder_table(result)
```

For non-linear families, pass `family=` so that the b-path and
total-effect regressions use the appropriate GLM:

```python
result = identify_confounders(
    X, y, predictor="x1", family="poisson", random_state=42,
)
print_confounder_table(result, family=resolve_family("poisson"))
```

For all predictors at once:

```python
all_results = {}
for predictor in X.columns:
    all_results[predictor] = identify_confounders(
        X, y, predictor=predictor, random_state=42,
    )
print_confounder_table(all_results)
```

## Input formats

All public functions accept both **pandas** and **Polars** DataFrames:

```python
import polars as pl

X_pl = pl.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [5, 4, 3, 2, 1]})
y_pl = pl.DataFrame({"y": [2.1, 4.0, 5.8, 8.2, 9.9]})

results = randomization_test_regression(X_pl, y_pl, random_state=42)
```

## Intercept control

By default an intercept is included.  For through-origin regression:

```python
results = randomization_test_regression(
    X, y, fit_intercept=False, random_state=42,
)
```

## JAX backend

The optional JAX backend accelerates logistic regression permutation
tests via `jax.vmap` over a custom Newton–Raphson solver.

### Tested versions

`jax>=0.4.20` through current (0.5.x).  Older 0.4.x releases may work
but are not tested in CI.

### Installation

```bash
pip install randomization-tests[jax]
```

Or from a local clone:

```bash
pip install -e ".[jax]"
```

### Known limitations

- **No Windows GPU support.** JAX does not ship Windows GPU wheels.
  CPU-only works on all platforms.
- **Memory.** JAX pre-allocates 75 % of GPU memory by default.  Set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` to disable this.
- **First-call latency.** JIT compilation adds a one-time overhead on
  the first call per session.

### Verify installation

```python
from randomization_tests import get_backend
print(get_backend())  # "jax" if detected, "numpy" otherwise
```

## Further reading

- [API Reference](API.md)
- [Background & motivation](../README.md)

## Model families

By default (`family="auto"`), binary targets trigger logistic regression
and all other targets use linear regression.  Pass an explicit `family=`
string for count, ordinal, or multinomial outcomes.

### Poisson (count data)

```python
results = randomization_test_regression(
    X, y, family="poisson", n_randomizations=1_000, random_state=42,
)
```

### Negative binomial (overdispersed counts)

```python
results = randomization_test_regression(
    X, y, family="negative_binomial", n_randomizations=1_000, random_state=42,
)
```

### Ordinal (ordered categories)

```python
# y must be integer-coded with ≥ 3 levels (0, 1, 2, ...)
results = randomization_test_regression(
    X, y, family="ordinal", n_randomizations=1_000, random_state=42,
)
```

### Multinomial (unordered categories)

```python
# y must be integer-coded with ≥ 3 classes (0, 1, 2, ...)
results = randomization_test_regression(
    X, y, family="multinomial", n_randomizations=1_000, random_state=42,
)
```

See `examples/` for complete worked examples of each family.

## Longitudinal data with AR errors

For panel / longitudinal data with serially correlated errors, use
`ar_order=` to apply an FGLS autoregressive correction that
Cholesky-whitens the design matrix and residuals before score projection.
This requires `panel_id=` (unit identifier),
`time_id=` (temporal index), and a score-based method.

```python
import numpy as np
import pandas as pd
from randomization_tests import randomization_test_regression

# Simulated panel: 20 subjects, 10 time points each
rng = np.random.default_rng(42)
n_panels, n_times = 20, 10
n = n_panels * n_times
panel_id = np.repeat(np.arange(n_panels), n_times)
time_id = np.tile(np.arange(n_times), n_panels)
x = rng.standard_normal(n)

# AR(1) errors with rho = 0.7
eps = np.zeros(n)
for p in range(n_panels):
    s = p * n_times
    eps[s] = rng.standard_normal()
    for t in range(1, n_times):
        eps[s + t] = 0.7 * eps[s + t - 1] + rng.standard_normal()

y = pd.DataFrame({"y": 1.5 * x + eps})
X = pd.DataFrame({"x": x})

results = randomization_test_regression(
    X, y,
    method="score",
    panel_id=panel_id,
    time_id=time_id,
    ar_order=1,
    n_randomizations=1_000,
    random_state=42,
)
```

The `ar_order=1` correction estimates a pooled within-panel AR(1)
coefficient and Cholesky-whitens X and residuals before projection,
ensuring permuted residuals are approximately exchangeable.  Diagnostics include before/after Durbin–Watson
and Ljung–Box statistics in
`results.extended_diagnostics["panel_diagnostics"]`.

## Backend configuration

By default the package auto-detects JAX and uses it when available.
To override:

```python
import randomization_tests

# Force numpy/sklearn (no JAX)
randomization_tests.set_backend("numpy")

# Restore auto-detection
randomization_tests.set_backend("auto")

# Check current backend
print(randomization_tests.get_backend())  # "numpy" or "jax"
```

Or set the `RANDOMIZATION_TESTS_BACKEND` environment variable:

```bash
export RANDOMIZATION_TESTS_BACKEND=numpy
```

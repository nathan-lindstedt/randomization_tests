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
    permutation_test_regression,
    print_diagnostics_table,
    print_results_table,
)

# Prepare data as DataFrames (pandas or Polars)
X = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [5, 4, 3, 2, 1]})
y = pd.DataFrame({"y": [2.1, 4.0, 5.8, 8.2, 9.9]})

# Run a ter Braak (1992) permutation test
results = permutation_test_regression(
    X, y,
    n_permutations=1_000,
    method="ter_braak",
    random_state=42,
)

# Display a statsmodels-style results table
print_results_table(results, feature_names=list(X.columns), target_name="y")

# Display extended diagnostics (VIF, Cook's D, etc.)
print_diagnostics_table(results, feature_names=list(X.columns))
```

## Available methods

| Method | `method=` | Description |
|---|---|---|
| ter Braak (1992) | `"ter_braak"` | Permute residuals under the reduced model. Default. |
| Kennedy (1995) individual | `"kennedy"` | Partial out confounders, permute exposure residuals. |
| Kennedy (1995) joint | `"kennedy_joint"` | Test whether predictors collectively improve fit beyond confounders. |
| Freedman–Lane (1983) individual | `"freedman_lane"` | Permute full-model residuals, reconstruct from reduced-model fitted values. Better power than Kennedy when predictors are correlated. |
| Freedman–Lane (1983) joint | `"freedman_lane_joint"` | Joint version of Freedman–Lane. |

Kennedy and Freedman–Lane methods require the `confounders` parameter
(a list of column names).

> **Note:** Ordinal and multinomial families do not support Freedman–Lane
> methods (residuals are ill-defined for these model types).

## Confounder identification

```python
from randomization_tests import identify_confounders, print_confounder_table

result = identify_confounders(X, y, predictor="x1", random_state=42)
print_confounder_table(result)
```

For non-linear families, pass `family=` so that the b-path and
total-effect regressions use the appropriate GLM:

```python
result = identify_confounders(
    X, y, predictor="x1", family="poisson", random_state=42,
)
print_confounder_table(result, family="poisson")
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

results = permutation_test_regression(X_pl, y_pl, random_state=42)
```

## Intercept control

By default an intercept is included.  For through-origin regression:

```python
results = permutation_test_regression(
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
results = permutation_test_regression(
    X, y, family="poisson", n_permutations=1_000, random_state=42,
)
```

### Negative binomial (overdispersed counts)

```python
results = permutation_test_regression(
    X, y, family="negative_binomial", n_permutations=1_000, random_state=42,
)
```

### Ordinal (ordered categories)

```python
# y must be integer-coded with ≥ 3 levels (0, 1, 2, ...)
results = permutation_test_regression(
    X, y, family="ordinal", n_permutations=1_000, random_state=42,
)
```

### Multinomial (unordered categories)

```python
# y must be integer-coded with ≥ 3 classes (0, 1, 2, ...)
results = permutation_test_regression(
    X, y, family="multinomial", n_permutations=1_000, random_state=42,
)
```

See `examples/` for complete worked examples of each family.

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

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

Kennedy methods require the `confounders` parameter (a list of column
names).

## Confounder identification

```python
from randomization_tests import identify_confounders, print_confounder_table

result = identify_confounders(X, y, predictor="x1", random_state=42)
print_confounder_table(result)
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

## Further reading

- [API Reference](API.md)
- [Background & motivation](../README.md)

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

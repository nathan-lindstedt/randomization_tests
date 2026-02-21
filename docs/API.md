# API Reference

> **Package:** `randomization_tests`
>
> Permutation tests for regression models using ter Braak (1992) and
> Kennedy (1995) methods with vectorised OLS, optional JAX autodiff for
> logistic regression, and pre‑generated unique permutations.
>
> All functions accepting data inputs support both **pandas** and
> **Polars** DataFrames (coerced internally via `_ensure_pandas_df`).

---

## Core

### `permutation_test_regression`

```python
permutation_test_regression(
    X: DataFrameLike,
    y: DataFrameLike,
    n_permutations: int = 5_000,
    precision: int = 3,
    p_value_threshold_one: float = 0.05,
    p_value_threshold_two: float = 0.01,
    method: str = "ter_braak",
    confounders: list[str] | None = None,
    random_state: int | None = None,
    fit_intercept: bool = True,
) -> dict
```

Run a permutation test for regression coefficients.

Automatically detects binary vs. continuous outcomes and uses logistic or
linear regression accordingly.

| Parameter | Description |
|---|---|
| `X` | Feature matrix, shape `(n_samples, n_features)`. Accepts pandas or Polars DataFrames. |
| `y` | Target values, shape `(n_samples,)`. Binary targets (`{0, 1}`) trigger logistic regression; otherwise linear regression is used. |
| `n_permutations` | Number of unique permutations to generate. |
| `precision` | Decimal places for reported p‑values. |
| `p_value_threshold_one` | First significance level (marked `*`). |
| `p_value_threshold_two` | Second significance level (marked `**`). |
| `method` | `"ter_braak"`, `"kennedy"`, or `"kennedy_joint"`. |
| `confounders` | Column names of confounders (required for Kennedy methods). |
| `random_state` | Seed for reproducibility. |
| `fit_intercept` | Whether to include an intercept in the regression model. Set to `False` for through‑origin regression. |

**Returns:** A dictionary containing model coefficients, empirical
(permutation) and classical (asymptotic) p‑values, extended diagnostics,
and method metadata. When `method="kennedy_joint"`, the dictionary
instead contains the observed improvement statistic and a single joint
p‑value.

**Raises:** `ValueError` if *method* is not one of the recognised
options.

**References:**

- ter Braak, C. J. F. (1992). Permutation versus bootstrap significance
  tests in multiple regression and ANOVA. *Handbook of Statistics*,
  Vol. 9.
- Kennedy, P. E. (1995). Randomization tests in econometrics.
  *J. Business & Economic Statistics*, 13(1), 85–94.
- Phipson, B. & Smyth, G. K. (2010). Permutation p‑values should never
  be zero. *Stat. Appl. Genet. Mol. Biol.*, 9(1), Article 39.

---

## Permutations

### `generate_unique_permutations`

```python
generate_unique_permutations(
    n_samples: int,
    n_permutations: int,
    random_state: int | None = None,
    exclude_identity: bool = True,
    max_exhaustive: int = 10,
) -> np.ndarray
```

Pre‑generate a matrix of unique permutation index arrays.

For small `n_samples` (≤ `max_exhaustive`), Lehmer‑code sampling draws
permutations by their lexicographic rank — O(B·n) time and memory,
independent of *n*!.  For larger inputs a vectorised batch strategy
generates all B permutations in one NumPy call, with optional post‑hoc
deduplication when the birthday‑paradox collision bound warrants it.

| Parameter | Description |
|---|---|
| `n_samples` | Length of the array to permute. |
| `n_permutations` | Number of unique permutations requested. |
| `random_state` | Seed for reproducibility. |
| `exclude_identity` | Exclude `[0, 1, …, n−1]` so the observed data is never counted as a null sample. |
| `max_exhaustive` | Threshold below which exhaustive enumeration is used. |

**Returns:** `np.ndarray` of shape `(n_permutations, n_samples)`.

**Raises:** `ValueError` if `n_permutations` exceeds the number of
available unique permutations.

---

## P‑Values

### `calculate_p_values`

```python
calculate_p_values(
    X: DataFrameLike,
    y: DataFrameLike,
    permuted_coefs: np.ndarray,
    model_coefs: np.ndarray,
    precision: int = 3,
    p_value_threshold_one: float = 0.05,
    p_value_threshold_two: float = 0.01,
    fit_intercept: bool = True,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]
```

Calculate empirical (permutation) and classical (asymptotic) p‑values.

Uses the Phipson & Smyth (2010) correction `p = (b + 1) / (B + 1)` to
ensure p‑values are never exactly zero.

| Parameter | Description |
|---|---|
| `X` | Feature matrix, shape `(n_samples, n_features)`. Accepts pandas or Polars DataFrames. |
| `y` | Target values. |
| `permuted_coefs` | Coefficients from each permutation, shape `(n_permutations, n_features)`. |
| `model_coefs` | Observed (unpermuted) coefficients, shape `(n_features,)`. |
| `precision` | Decimal places for rounding. |
| `p_value_threshold_one` | First significance threshold. |
| `p_value_threshold_two` | Second significance threshold. |
| `fit_intercept` | Whether to include an intercept in the asymptotic model. |

**Returns:** A 4‑tuple
`(permuted_p_values, classic_p_values, raw_permuted_p, raw_classic_p)`
where the first two are lists of formatted p‑value strings with
significance markers (`*`, `**`, or `ns`) and the last two are raw
`np.ndarray` values.

---

## Confounders

### `identify_confounders`

```python
identify_confounders(
    X: DataFrameLike,
    y: DataFrameLike,
    predictor: str,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> dict
```

Two‑step confounder identification.

1. Screen for variables correlated with both predictor and outcome.
2. Use mediation analysis to filter out mediators.

Variables that pass screening but are **not** identified as mediators are
classified as likely confounders.

| Parameter | Description |
|---|---|
| `X` | Feature matrix. Accepts pandas or Polars DataFrames. |
| `y` | Target variable. |
| `predictor` | Predictor of interest. |
| `correlation_threshold` | Minimum absolute Pearson *r*. |
| `p_value_threshold` | Significance cutoff. |
| `n_bootstrap` | Bootstrap iterations for mediation analysis. |
| `confidence_level` | Confidence‑interval level. |
| `random_state` | Seed for reproducibility. |

**Returns:** Dictionary with keys `identified_confounders`,
`identified_mediators`, `screening_results`, and `mediation_results`.

---

### `screen_potential_confounders`

```python
screen_potential_confounders(
    X: DataFrameLike,
    y: DataFrameLike,
    predictor: str,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
) -> dict
```

Screen for variables correlated with both predictor and outcome.

A potential confounder *Z* satisfies `|r(Z, X)| >= threshold` **and**
`|r(Z, Y)| >= threshold`, both with `p < p_value_threshold`.

| Parameter | Description |
|---|---|
| `X` | Feature matrix. Accepts pandas or Polars DataFrames. |
| `y` | Target variable. |
| `predictor` | Name of the predictor of interest. |
| `correlation_threshold` | Minimum absolute Pearson *r* to flag a variable. |
| `p_value_threshold` | Maximum p‑value for significance. |

**Returns:** Dictionary with keys `predictor`, `potential_confounders`,
`correlations_with_predictor`, `correlations_with_outcome`, and
`excluded_variables`.

---

### `mediation_analysis`

```python
mediation_analysis(
    X: DataFrameLike,
    y: DataFrameLike,
    predictor: str,
    mediator: str,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    precision: int = 4,
    random_state: int | None = None,
) -> dict
```

Preacher & Hayes (2004, 2008) bootstrap test of the indirect effect.

Decomposes the total effect of *predictor* on *y* into a direct effect
(c′) and an indirect effect through *mediator* (a × b). The indirect
effect is tested via bias‑corrected and accelerated (BCa) bootstrap
confidence intervals (Efron, 1987). Unlike the Baron & Kenny (1986)
causal‑steps approach, this method does **not** require the total effect
to be significant as a prerequisite — the bootstrap CI of the indirect
effect is the sole criterion for mediation.

| Parameter | Description |
|---|---|
| `X` | Feature matrix. Accepts pandas or Polars DataFrames. |
| `y` | Target variable. |
| `predictor` | Predictor (X in X → M → Y). |
| `mediator` | Potential mediator (M). |
| `n_bootstrap` | Number of bootstrap samples. Preacher & Hayes recommend ≥ 5 000 for BCa intervals. |
| `confidence_level` | Confidence‑interval level. |
| `precision` | Decimal places for rounding. |
| `random_state` | Seed for reproducibility. |

**Returns:** Dictionary containing the mediation decomposition
(`total_effect`, `direct_effect`, `indirect_effect`, `a_path`, `b_path`),
BCa bootstrap CI (`indirect_effect_ci`), `ci_method` (`"BCa"`),
`proportion_mediated`, `is_mediator` flag, and a textual
`interpretation`.

---

## Display

### `print_results_table`

```python
print_results_table(
    results: dict,
    feature_names: list[str],
    target_name: str | None = None,
    title: str = "Permutation Test Results",
) -> None
```

Print regression results in a formatted ASCII table similar to
statsmodels output.  Shows model summary statistics (top panel) and
per‑feature coefficients with empirical (permutation) and classical
(asymptotic) p‑values side by side (bottom panel).

| Parameter | Description |
|---|---|
| `results` | Results dictionary from `permutation_test_regression`. |
| `feature_names` | Names of the features/predictors. |
| `target_name` | Name of the target variable. |
| `title` | Title for the output table. |

---

### `print_joint_results_table`

```python
print_joint_results_table(
    results: dict,
    target_name: str | None = None,
    title: str = "Joint Permutation Test Results",
) -> None
```

Print joint test results (Kennedy joint method) in a formatted ASCII
table.

| Parameter | Description |
|---|---|
| `results` | Results dictionary from `permutation_test_regression` with `method="kennedy_joint"`. |
| `target_name` | Name of the target variable. |
| `title` | Title for the output table. |

---

### `print_diagnostics_table`

```python
print_diagnostics_table(
    results: dict,
    feature_names: list[str],
    title: str = "Extended Diagnostics",
) -> None
```

Print extended model diagnostics in a formatted ASCII table.

Complements `print_results_table` with additional per‑predictor and
model‑level diagnostics.  The table has four sections:

1. **Per‑predictor Diagnostics** — standardised coefficients, VIF,
   Monte Carlo SE, optional Exposure R² (Kennedy method), and
   empirical‑vs‑asymptotic divergence flags.
2. **Legend** — compact key explaining each column.
3. **Model‑level Diagnostics** — Breusch‑Pagan (linear) or deviance
   residuals (logistic), Cook's distance counts, permutation coverage.
4. **Notes** (conditional) — plain‑language warnings for flagged
   diagnostics.

| Parameter | Description |
|---|---|
| `results` | Results dictionary from `permutation_test_regression`. Must contain `extended_diagnostics`. |
| `feature_names` | Names of the features/predictors. |
| `title` | Title for the output table. |

---

### `print_confounder_table`

```python
print_confounder_table(
    confounder_results: dict,
    title: str = "Confounder Identification Results",
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> None
```

Print confounder identification results in a formatted ASCII table.

Accepts either a single `identify_confounders` result dict or a
`dict[str, dict]` mapping predictor names to their individual result
dicts.

| Parameter | Description |
|---|---|
| `confounder_results` | Single result dict or `{predictor: result_dict}` mapping. |
| `title` | Title for the output table. |
| `correlation_threshold` | The `\|r\|` threshold used during screening (shown in header). |
| `p_value_threshold` | Screening p‑value cutoff (shown in header). |
| `n_bootstrap` | Bootstrap iterations for mediation (shown in header). |
| `confidence_level` | CI level for mediation (shown in header). |

---

## Backend Configuration

The package auto‑detects JAX at import time.  You can override the
backend via an environment variable or at runtime.

### `get_backend`

```python
get_backend() -> str
```

Return the active backend name (`"jax"` or `"numpy"`).

Resolution order (first match wins):

1. Programmatic override set by `set_backend()`.
2. The `RANDOMIZATION_TESTS_BACKEND` environment variable.
3. Auto‑detection: `"jax"` if importable, else `"numpy"`.

### `set_backend`

```python
set_backend(name: str) -> None
```

Override the backend selection.

| Parameter | Description |
|---|---|
| `name` | `"jax"`, `"numpy"`, or `"auto"` (case‑insensitive). `"auto"` restores the default resolution order. |

**Raises:** `ValueError` if *name* is not recognised.

**Examples:**

```bash
# Disable JAX globally (shell)
export RANDOMIZATION_TESTS_BACKEND=numpy
```

```python
# Disable JAX programmatically
import randomization_tests
randomization_tests.set_backend("numpy")
```

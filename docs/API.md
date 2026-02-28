# API Reference

> **Package:** `randomization_tests`
>
> Permutation tests for regression models using ter Braak (1992),
> Kennedy (1995), and Freedman–Lane (1983) methods with vectorised
> batch fitting, optional JAX autodiff for all GLM families, and
> pre‑generated unique permutations.
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
    family: str = "auto",
    n_jobs: int = 1,
) -> IndividualTestResult | JointTestResult
```

Run a permutation test for regression coefficients.

By default (`family="auto"`), detects binary vs. continuous outcomes and
uses logistic or linear regression accordingly.  Pass an explicit family
string (`"linear"`, `"logistic"`, `"poisson"`, `"negative_binomial"`,
`"ordinal"`, `"multinomial"`) to override auto‑detection.

| Parameter | Description |
|---|---|
| `X` | Feature matrix, shape `(n_samples, n_features)`. Accepts pandas or Polars DataFrames. |
| `y` | Target values, shape `(n_samples,)`. When `family="auto"`, binary targets (`{0, 1}`) trigger logistic regression; otherwise linear regression is used. |
| `n_permutations` | Number of unique permutations to generate. |
| `precision` | Decimal places for reported p‑values. |
| `p_value_threshold_one` | First significance level (marked `*`). |
| `p_value_threshold_two` | Second significance level (marked `**`). |
| `method` | `"ter_braak"`, `"kennedy"`, `"kennedy_joint"`, `"freedman_lane"`, or `"freedman_lane_joint"`. |
| `confounders` | Column names of confounders (required for Kennedy and Freedman–Lane methods). |
| `random_state` | Seed for reproducibility. |
| `fit_intercept` | Whether to include an intercept in the regression model. Set to `False` for through‑origin regression. |
| `family` | Model family string. `"auto"` (default) detects binary `{0, 1}` targets → logistic, otherwise linear. Explicit values: `"linear"`, `"logistic"`, `"poisson"`, `"negative_binomial"`, `"ordinal"`, `"multinomial"`. |
| `n_jobs` | Number of parallel jobs for the permutation batch‑fit loop. `1` (default) is sequential; `-1` uses all CPU cores. Ignored when the JAX backend is active. |

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

## Families

Model‑family abstraction layer.  Each family encapsulates all
model‑specific operations (fitting, prediction, residuals,
reconstruction, diagnostics) behind a common protocol so that the
permutation engine is family‑agnostic.

### `ModelFamily` (Protocol)

Runtime‑checkable protocol that every family class must satisfy.

| Property / Method | Description |
|---|---|
| `name: str` | Short label, e.g. `"linear"`, `"logistic"`, `"poisson"`, `"negative_binomial"`, `"ordinal"`, `"multinomial"`. |
| `residual_type: str` | Residual flavour (`"raw"`, `"probability"`, `"response"`). |
| `direct_permutation: bool` | `True` when the family supports direct Y‑permutation (ter Braak shortcut). |
| `metric_label: str` | Human‑readable label for the joint‑test metric (e.g. `"RSS Reduction"`). |
| `validate_y(y)` | Raise `ValueError` if *y* is incompatible with the family. |
| `fit(X, y, fit_intercept)` | Fit the model; returns a fitted‑model object. |
| `predict(model, X)` | Predicted values on the response scale. |
| `coefs(model)` | Slope coefficients (intercept excluded). |
| `residuals(model, X, y)` | Residuals appropriate for the family. |
| `reconstruct_y(predictions, permuted_residuals, rng)` | Build permuted response vectors. |
| `fit_metric(y_true, y_pred)` | Scalar goodness‑of‑fit metric (e.g. RSS, deviance). |
| `diagnostics(X, y, fit_intercept)` | Dict of model‑level diagnostics. |
| `classical_p_values(X, y, fit_intercept)` | Asymptotic p‑values (t‑ or z‑test). |
| `exchangeability_cells(X, y)` | Exchangeability cell labels (or `None` for global). |
| `batch_fit(X, Y_matrix, fit_intercept, **kw)` | Fit B models with varying Y; shape `(B, p)`. |
| `batch_fit_varying_X(X_batch, y, fit_intercept, **kw)` | Fit B models with varying X; shape `(B, p)`. |

### `LinearFamily`

OLS regression for continuous outcomes.

- `residual_type = "raw"` — raw residuals `y − ŷ`.
- `direct_permutation = True` — ter Braak shortcut via pseudoinverse.
- `metric_label = "RSS Reduction"`.
- `batch_fit` uses NumPy pseudoinverse (single matrix multiply).

### `LogisticFamily`

Logistic regression for binary `{0, 1}` outcomes.

- `residual_type = "probability"` — probability‑scale residuals `y − P̂(Y=1)`.
- `direct_permutation = False`.
- `metric_label = "Deviance Reduction"`.
- `reconstruct_y` clips then draws Bernoulli.
- `batch_fit` uses JAX vmap or sklearn fallback.

### `PoissonFamily`

Poisson GLM for non‑negative integer count outcomes.

- `residual_type = "response"` — response‑scale residuals `y − μ̂`.
- `direct_permutation = False`.
- `metric_label = "Deviance Reduction"`.
- `reconstruct_y` adds permuted residuals on the response scale, then
  draws `Y* ~ Poisson(μ*)`.
- `batch_fit` uses a joblib‑parallelised statsmodels loop.

### `NegativeBinomialFamily`

```python
NegativeBinomialFamily(alpha: float | None = None)
```

NB2 GLM for overdispersed count outcomes (`Var(Y) = μ + α·μ²`).

- `residual_type = "response"` — response‑scale residuals `y − μ̂`.
- `direct_permutation = False`.
- `metric_label = "Deviance Reduction"`.
- `alpha` — the NB dispersion parameter.  `None` means uncalibrated
  (must call `calibrate()` before fitting).
- `reconstruct_y` clips fitted values, then draws
  `Y* ~ NB(n=1/α, p=1/(1+α·μ*))`.
- `batch_fit` uses a joblib‑parallelised statsmodels loop with fixed α.

#### `calibrate`

```python
NegativeBinomialFamily.calibrate(
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = True,
) -> NegativeBinomialFamily
```

Estimate the NB dispersion parameter α from the data via MLE
(`statsmodels.discrete.NegativeBinomial`).  Returns a **new frozen
instance** with `alpha` set.  Idempotent: calling on an
already‑calibrated instance returns `self`.

This method is **not** part of the `ModelFamily` protocol — it is
duck‑typed and called via `hasattr` guard in the permutation engine.
Only families with nuisance parameters need implement it.

| Parameter | Description |
|---|---|
| `X` | Feature matrix, shape `(n, p)`. |
| `y` | Response vector, shape `(n,)`. |
| `fit_intercept` | Whether to include an intercept column. |

**Returns:** A new `NegativeBinomialFamily` instance with `alpha` set.

**Raises:** `RuntimeError` if MLE estimation fails to converge.

### `OrdinalFamily`

Proportional‑odds logistic regression for ordered categorical outcomes
with ≥ 3 levels (integer‑coded 0, 1, …, K−1).

- `residual_type = "none"` — ordinal residuals are ill‑defined.
- `direct_permutation = True` — first family to use direct Y permutation.
- `metric_label = "Deviance Reduction"`.
- `fit` uses `statsmodels.miscmodels.ordinal_model.OrderedModel` with
  `distr="logit"` and `method="bfgs"` for reliable convergence.
- `fit_intercept` is accepted but ignored — thresholds serve as
  category‑specific intercepts.
- `residuals`, `reconstruct_y`, `fit_metric` raise `NotImplementedError`.
- `batch_fit` uses a joblib‑parallelised OrderedModel loop.

**Supported methods:** `ter_braak`, `kennedy`, `kennedy_joint`.

**Rejected methods:** `freedman_lane`, `freedman_lane_joint` raise
`ValueError` because ordinal residuals are not well‑defined.

#### `score`

```python
OrdinalFamily.score(model: Any, X: np.ndarray, y: np.ndarray) -> float
```

Returns `−2 × log‑likelihood` from the fitted ordinal model.
Protocol method used by all joint‑test strategies.

#### `null_score`

```python
OrdinalFamily.null_score(y: np.ndarray, fit_intercept: bool = True) -> float
```

Returns the thresholds‑only (no predictors) deviance, computed
analytically from empirical category proportions:
`−2 ∑ n_k log(n_k / n)`.

### `MultinomialFamily`

Multinomial logistic regression (softmax link) for unordered categorical
outcomes with ≥ 3 classes (integer‑coded 0, 1, …, K−1).

- `residual_type = "none"` — multinomial residuals are ill‑defined.
- `direct_permutation = True` — permutes class labels directly.
- `metric_label = "χ²"`.
- `fit` uses `statsmodels.discrete.discrete_model.MNLogit`.
- `coefs` returns a scalar LRT chi‑squared per predictor (shape `(p,)`)
  — the standard reduction for testing whether a predictor affects a
  multi‑class outcome.
- `residuals`, `reconstruct_y`, `fit_metric` raise `NotImplementedError`.
- `batch_fit` uses a joblib‑parallelised MNLogit loop.

**Supported methods:** `ter_braak`, `kennedy`, `kennedy_joint`.

**Rejected methods:** `freedman_lane`, `freedman_lane_joint` raise
`ValueError` because multinomial residuals are not well‑defined.

#### `category_coefs` (convenience)

```python
MultinomialFamily.category_coefs(model: Any) -> np.ndarray
```

Returns the full `(p, K−1)` slope matrix from a fitted MNLogit model
for detailed post‑hoc inspection of per‑category contrasts.

#### `score`

```python
MultinomialFamily.score(model: Any, X: np.ndarray, y: np.ndarray) -> float
```

Returns `−2 × log‑likelihood` from the fitted multinomial model.
Protocol method used by all joint‑test strategies.

#### `null_score`

```python
MultinomialFamily.null_score(y: np.ndarray, fit_intercept: bool = True) -> float
```

Returns the intercept‑only deviance from a fitted `MNLogit` null model.

### `LinearMixedFamily`

Linear mixed‑effects model (LMM) with Henderson REML estimation.

- `residual_type = "raw"` — conditional (BLUP) residuals.
- `direct_permutation = False`.
- `metric_label = "RSS Reduction"`.
- Calibrated via `calibrate(X, y, fit_intercept, groups=...)`.
  Builds random‑effects design Z from group labels, estimates
  variance components via LM-Nielsen Newton on REML NLL.
- `score_project()` uses cached GLS projection A row for O(n·B)
  batch permutation.
- `batch_fit` uses A @ Y — single matmul for all B permutations.
- `exchangeability_cells` returns first‑factor group labels.
- Diagnostics: marginal/conditional R², variance components, ICC.

**Supported methods:** `ter_braak`, `freedman_lane`,
`freedman_lane_joint`, `score`, `score_joint`.

Requires `groups=` keyword (via `FitContext` or direct `calibrate()`
call).

### `LogisticMixedFamily`

Logistic GLMM for clustered binary `{0, 1}` outcomes via Laplace
approximation.

- `residual_type = "deviance"` — deviance residuals conditional on û.
- `direct_permutation = False`.
- `stat_label = "z"`.
- Calibrated via `calibrate(X, y, fit_intercept, groups=...)`.
  Inner IRLS pre‑scales by √W to reuse Henderson algebra; outer
  θ optimisation via `_reml_newton_solve()` on Laplace NLL.
- `score_project()` uses one‑step Le Cam corrector — O(n·B) matmul,
  no IRLS in the permutation loop.
- `reconstruct_y` draws `Y* ~ Bern(clip(ŷ + π(e)))`.
- `exchangeability_cells` returns first‑factor group labels.
- Diagnostics: AUC, deviance, variance components, ICC,
  random‑effect covariance recovery.

**Supported methods:** `score`, `score_exact`.

**Rejected methods:** `ter_braak`, `freedman_lane`, `kennedy` raise
`ValueError` — GLMM families require score‑based permutation.

Requires `groups=` keyword.

### `PoissonMixedFamily`

Poisson GLMM for clustered non‑negative integer count outcomes via
Laplace approximation.

- `residual_type = "deviance"` — deviance residuals conditional on û.
- `direct_permutation = False`.
- `stat_label = "z"`.
- Same architecture as `LogisticMixedFamily` — only the conditional
  NLL and IRLS working response/weight functions differ.
- `score_project()` uses one‑step Le Cam corrector.
- `reconstruct_y` draws `Y* ~ Poisson(max(ŷ + π(e), ε))`.
- `exchangeability_cells` returns first‑factor group labels.
- Diagnostics: pseudo‑R², deviance, Pearson overdispersion (χ²/df),
  variance components, ICC.

**Supported methods:** `score`, `score_exact`.

**Rejected methods:** `ter_braak`, `freedman_lane`, `kennedy` raise
`ValueError`.

Requires `groups=` keyword.

### `resolve_family`

```python
resolve_family(family: str, y: np.ndarray) -> ModelFamily
```

Resolve a family string to a `ModelFamily` instance.

| Value | Behaviour |
|---|---|
| `"auto"` | Binary `{0, 1}` → `LogisticFamily`; otherwise `LinearFamily`. |
| `"linear"` | `LinearFamily()`. |
| `"logistic"` | `LogisticFamily()`. |
| `"poisson"` | `PoissonFamily()`. |
| `"negative_binomial"` | `NegativeBinomialFamily()` (uncalibrated; α estimated during calibration). |
| `"ordinal"` | `OrdinalFamily()`. Requires ≥ 3 categories; supports ter Braak, Kennedy only. |
| `"multinomial"` | `MultinomialFamily()`. Requires ≥ 3 unordered categories; supports ter Braak, Kennedy only. |
| `"linear_mixed"` | `LinearMixedFamily()`. Requires `groups=`. |
| `"logistic_mixed"` | `LogisticMixedFamily()`. Requires `groups=`; supports score methods only. |
| `"poisson_mixed"` | `PoissonMixedFamily()`. Requires `groups=`; supports score methods only. |

**Raises:** `ValueError` for unrecognised family strings.

### `register_family`

```python
register_family(name: str, cls: type) -> None
```

Register a custom `ModelFamily` implementation under a string key.
The class must satisfy the `ModelFamily` protocol.

**Raises:** `TypeError` if *cls* does not implement the protocol.

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
    family: str = "auto",
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
| `family` | Model family string (`"auto"`, `"linear"`, `"logistic"`, `"poisson"`, `"negative_binomial"`, `"ordinal"`, `"multinomial"`). When non‑linear, the b‑path and total‑effect regressions use the family‑appropriate GLM instead of OLS. |

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
    family: str = "auto",
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
| `family` | Model family string. When non‑linear, the b‑path (Y ~ X + M) and total‑effect (Y ~ X) regressions use the family‑appropriate GLM. |

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
    results: IndividualTestResult,
    *,
    title: str = "Permutation Test Results",
) -> None
```

Print regression results in a formatted ASCII table similar to
statsmodels output.  Shows model summary statistics (top panel) and
per‑feature coefficients with empirical (permutation) and classical
(asymptotic) p‑values side by side (bottom panel).

Feature names, target name, and model family are read directly from
the result object.

| Parameter | Description |
|---|---|
| `results` | `IndividualTestResult` from `permutation_test_regression`. |
| `title` | Title for the output table. |

---

### `print_joint_results_table`

```python
print_joint_results_table(
    results: JointTestResult,
    *,
    title: str = "Joint Permutation Test Results",
) -> None
```

Print joint test results (Kennedy/Freedman–Lane joint methods) in a
formatted ASCII table.

Target name and model family are read directly from the result object.

| Parameter | Description |
|---|---|
| `results` | `JointTestResult` from `permutation_test_regression` (joint methods). |
| `title` | Title for the output table. |

---

### `print_diagnostics_table`

```python
print_diagnostics_table(
    results: IndividualTestResult,
    *,
    title: str = "Extended Diagnostics",
) -> None
```

Print extended model diagnostics in a formatted ASCII table.

Complements `print_results_table` with additional per‑predictor and
model‑level diagnostics.  Feature names and model family are read
directly from the result object.  The table has four sections:

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
| `results` | `IndividualTestResult` from `permutation_test_regression`. Must contain `extended_diagnostics`. |
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
    family: ModelFamily | None = None,
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
| `family` | Optional `ModelFamily` instance (e.g. `PoissonFamily()`).  When supplied, the family name is shown in the table header. |

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

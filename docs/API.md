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
    p_value_threshold_three: float = 0.001,
    method: str = "ter_braak",
    confounders: list[str] | None = None,
    random_state: int | None = None,
    fit_intercept: bool = True,
    family: str | ModelFamily = "auto",
    n_jobs: int = 1,
    backend: str | None = None,
    groups: np.ndarray | list[int] | pd.Series | pd.DataFrame | None = None,
    permutation_strategy: str | None = None,
    permutation_constraints: Callable[[np.ndarray], np.ndarray] | None = None,
    random_slopes: list[int] | dict[str, list[int]] | None = None,
    confidence_level: float = 0.95,
    panel_id: np.ndarray | list[int] | pd.Series | str | None = None,
    time_id: np.ndarray | list[int] | pd.Series | str | None = None,
) -> IndividualTestResult | JointTestResult
```

Run a permutation test for regression coefficients.

By default (`family="auto"`), detects binary vs. continuous outcomes and
uses logistic or linear regression accordingly.  Pass an explicit family
string (`"linear"`, `"logistic"`, `"poisson"`, `"negative_binomial"`,
`"ordinal"`, `"multinomial"`) or a pre‑configured `ModelFamily` instance
to override auto‑detection.

| Parameter | Description |
|---|---|
| `X` | Feature matrix, shape `(n_samples, n_features)`. Accepts pandas or Polars DataFrames. |
| `y` | Target values, shape `(n_samples,)`. When `family="auto"`, binary targets (`{0, 1}`) trigger logistic regression; otherwise linear regression is used. |
| `n_permutations` | Number of unique permutations to generate. |
| `precision` | Decimal places for reported p‑values. |
| `p_value_threshold_one` | First significance level (marked `*`). |
| `p_value_threshold_two` | Second significance level (marked `**`). |
| `p_value_threshold_three` | Third significance level (marked `***`). |
| `method` | `"ter_braak"`, `"kennedy"`, `"kennedy_joint"`, `"freedman_lane"`, `"freedman_lane_joint"`, `"score"`, `"score_joint"`, or `"score_exact"`. |
| `confounders` | Column names of confounders (required for Kennedy and Freedman–Lane methods). |
| `random_state` | Seed for reproducibility. |
| `fit_intercept` | Whether to include an intercept in the regression model. Set to `False` for through‑origin regression. |
| `family` | Model family string **or** a `ModelFamily` instance. `"auto"` (default) detects binary `{0, 1}` targets → logistic, otherwise linear. Explicit values: `"linear"`, `"logistic"`, `"poisson"`, `"negative_binomial"`, `"ordinal"`, `"multinomial"`, `"linear_mixed"`, `"logistic_mixed"`, `"poisson_mixed"`. Pre‑configured instances (e.g. `NegativeBinomialFamily(alpha=2.0)`) are accepted directly. |
| `n_jobs` | Number of parallel jobs for the permutation batch‑fit loop. `1` (default) is sequential; `-1` uses all CPU cores. Ignored when the JAX backend is active. |
| `backend` | `"numpy"`, `"jax"`, or `None` (default). When `None`, the global policy from `set_backend()` is used. An explicit value overrides the global setting for this call only. |
| `groups` | Exchangeability group labels. When provided, permutations are constrained to respect group structure rather than shuffling globally. Accepts a 1‑D array‑like of integer labels `(n_samples,)` or a `DataFrame` with one or more blocking columns (multi‑column DataFrames are cross‑classified into integer cell labels). |
| `permutation_strategy` | Cell‑level permutation strategy: `"within"` shuffles only within cells, `"between"` permutes entire cells as units, `"two-stage"` composes both. Defaults to `"within"` when `groups` is provided. Cannot be set without `groups`. |
| `permutation_constraints` | Optional post‑filter callback. Receives a `(B, n)` permutation array and must return a `(B', n)` array with `B' ≤ B` rows. Used for domain‑specific constraints that cannot be expressed via cell structure alone. |
| `random_slopes` | Random‑slope column indices or `{factor: [indices]}` mapping for mixed‑effects families. Passed to `calibrate()` to build the random‑effects design matrix Z. |
| `confidence_level` | Confidence level for all CI types (permutation, Wald, Clopper–Pearson, standardised). Defaults to `0.95`. |
| `panel_id` | Panel (subject/unit) identifier for longitudinal data. When provided, automatically sets `groups=panel_id` and `permutation_strategy="within"`. Accepts a 1‑D array‑like of labels or a column name (string) in `X`. Cannot be used together with an explicit `groups=` argument. |
| `time_id` | Time‑period identifier for longitudinal data. Only meaningful when `panel_id` is also provided. Used for validation checks: warns if data is not sorted by `(panel_id, time_id)` or if panels are unbalanced. |

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
| `classical_p_values(X, y, fit_intercept, *, robust_se)` | Asymptotic p‑values (t‑ or z‑test). `robust_se=True` uses sandwich SEs. |
| `score_project(X, y, feature_idx, confounders, *, y)` | Score‑test projection row for `method="score"`. |
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
resolve_family(
    family: str | ModelFamily,
    y: np.ndarray | None = None,
) -> ModelFamily
```

Resolve a family string or instance to a concrete `ModelFamily`.

When *family* is already a `ModelFamily` instance, it is returned
as‑is (pass‑through).  This enables callers to pass pre‑configured
instances (e.g. `NegativeBinomialFamily(alpha=2.0)`) without
triggering fresh resolution or construction.

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
    n_bootstrap_mediation: int = 1000,
    n_bootstrap_moderation: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
    family: str | ModelFamily = "auto",
    correlation_method: str = "pearson",
    correction_method: str | None = None,
    groups: np.ndarray | None = None,
) -> ConfounderAnalysisResult
```

Four‑stage confounder sieve.

Classifies candidate variables as colliders, mediators, moderators,
or true confounders via sequential testing:

1. **Screen** — dual‑correlation with both predictor and outcome.
2. **Collider test** — removes colliders (X → Z ← Y).
3. **Mediator test** — removes mediators (X → M → Y).
4. **Moderator test** — labels moderators (informational; stays in
   confounder pool).

The sieve is an **exploratory** tool for data‑driven confounder
selection.  For guaranteed Type I error control, specify
`confounders=` based on domain knowledge.

| Parameter | Description |
|---|---|
| `X` | Feature matrix. Accepts pandas or Polars DataFrames. |
| `y` | Target variable. |
| `predictor` | Predictor of interest. |
| `correlation_threshold` | Minimum absolute correlation to flag. |
| `p_value_threshold` | Significance cutoff for screening. |
| `n_bootstrap_mediation` | Bootstrap iterations for mediation tests. |
| `n_bootstrap_moderation` | Bootstrap iterations for moderation tests. |
| `confidence_level` | Confidence‑interval level. |
| `random_state` | Seed for reproducibility. |
| `family` | Model family string or object. Mixed families resolved to base. |
| `correlation_method` | `"pearson"` (default), `"partial"`, or `"distance"`. |
| `correction_method` | `None`, `"holm"`, or `"fdr_bh"` for multiple‑testing correction. |
| `groups` | Optional group labels for cluster bootstrap (passed to mediation/moderation). |

**Returns:** `ConfounderAnalysisResult` dataclass with fields
`predictor`, `identified_confounders`, `identified_mediators`,
`identified_moderators`, `identified_colliders`, `screening_results`,
`mediation_results`, `moderation_results`, and `collider_results`.
Supports `to_dict()` and dict‑style `[]` access for backward
compatibility.

---

### `screen_potential_confounders`

```python
screen_potential_confounders(
    X: DataFrameLike,
    y: DataFrameLike,
    predictor: str,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
    correlation_method: str = "pearson",
    correction_method: str | None = None,
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
| `correlation_threshold` | Minimum absolute Pearson *r* (or distance correlation) to flag. |
| `p_value_threshold` | Maximum p‑value for significance. |
| `correlation_method` | `"pearson"` (default), `"partial"`, or `"distance"`. When `"partial"`, the Z‑Y leg computes `partial_r(Z, Y | X)`. When `"distance"`, Székely & Rizzo's bias‑corrected distance correlation is used for both legs. |
| `correction_method` | `None`, `"holm"`, or `"fdr_bh"`. Multiple‑testing correction applied per‑leg via `statsmodels.stats.multitest.multipletests`. |

**Returns:** Dictionary with keys `predictor`, `potential_confounders`,
`correlations_with_predictor`, `correlations_with_outcome`,
`excluded_variables`, `correlation_method`, `correction_method`, and
`adjusted_p_values`.

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
    family: str | ModelFamily = "auto",
    groups: np.ndarray | None = None,
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
| `family` | Model family string or object. Mixed families are automatically resolved to their base family. |
| `groups` | Optional 1‑D array of group labels for cluster bootstrap. When provided, uses cluster bootstrap (resample whole groups) and cluster jackknife for BCa acceleration. |

**Returns:** Dictionary containing the mediation decomposition
(`total_effect`, `direct_effect`, `indirect_effect`, `a_path`, `b_path`),
BCa bootstrap CI (`indirect_effect_ci`), `ci_method` (`"BCa"`),
`proportion_mediated`, `is_mediator` flag, and a textual
`interpretation`.

---

### `moderation_analysis`

```python
moderation_analysis(
    X: DataFrameLike,
    y: DataFrameLike,
    predictor: str,
    moderator: str,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    precision: int = 4,
    random_state: int | None = None,
    family: str | ModelFamily = "auto",
    groups: np.ndarray | None = None,
) -> dict
```

Bootstrap test for moderation (interaction effect).

Mean‑centers *predictor* and *moderator* **per resample** (no
information leakage), constructs the interaction term X_c × Z_c,
and fits Y ~ X_c + Z_c + X_c × Z_c.  The interaction coefficient
is tested via BCa bootstrap CIs.

| Parameter | Description |
|---|---|
| `X` | Feature matrix. Accepts pandas or Polars DataFrames. |
| `y` | Target variable. |
| `predictor` | Predictor (X). |
| `moderator` | Candidate moderator (Z). |
| `n_bootstrap` | Number of bootstrap samples. |
| `confidence_level` | Confidence‑interval level. |
| `precision` | Decimal places for rounding. |
| `random_state` | Seed for reproducibility. |
| `family` | Outcome family. Mixed families are automatically resolved to their base family. |
| `groups` | Optional 1‑D array of group labels for cluster bootstrap. |

**Returns:** Dictionary with keys `predictor`, `moderator`,
`x_coef`, `z_coef`, `interaction_coef`, `interaction_ci`,
`ci_method`, `is_moderator`, and `interpretation`.

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
    confounder_results: dict | ConfounderAnalysisResult,
    title: str = "Confounder Identification Results",
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    family: ModelFamily | None = None,
    correlation_method: str = "pearson",
    correction_method: str | None = None,
) -> None
```

Print confounder identification results in a formatted ASCII table.

Accepts either a single `identify_confounders` result dict,
a `ConfounderAnalysisResult` dataclass, or a `dict[str, dict]`
mapping predictor names to their individual result dicts.

| Parameter | Description |
|---|---|
| `confounder_results` | Single result dict, `ConfounderAnalysisResult`, or `{predictor: result_dict}` mapping. |
| `title` | Title for the output table. |
| `correlation_threshold` | The `\|r\|` threshold used during screening (shown in header). |
| `p_value_threshold` | Screening p‑value cutoff (shown in header). |
| `n_bootstrap` | Bootstrap iterations for mediation (shown in header). |
| `confidence_level` | CI level for mediation (shown in header). |
| `family` | Optional `ModelFamily` instance. When supplied, the family name is shown in the header. |
| `correlation_method` | `"pearson"`, `"partial"`, or `"distance"` (shown in header). |
| `correction_method` | `None`, `"holm"`, or `"fdr_bh"` (shown in header). |

---

## Sensitivity Analysis

### `compute_e_value`

```python
compute_e_value(
    coefficient: float,
    family: str,
    ci_bound: float | None = None,
    sd_x: float | None = None,
    sd_y: float | None = None,
    baseline_prevalence: float | None = None,
) -> dict
```

Compute the E‑value for unmeasured confounding sensitivity.

The E‑value is the minimum strength of association (on the RR scale)
that an unmeasured confounder would need to have with both the treatment
and the outcome to fully explain away the observed association.

Family‑dispatched conversion:

* **linear** — Standardize β → Cohen's d, then RR = exp(0.91 × d).
  Requires `sd_x` and `sd_y`.
* **logistic** / **ordinal** — OR = exp(β), then Cornfield inequality.
  When `baseline_prevalence` is provided, computes exact RR.
* **poisson** / **negative_binomial** — RR = exp(β) directly.
* **multinomial** — Returns NaN with UserWarning.

| Parameter | Description |
|---|---|
| `coefficient` | Estimated coefficient (log‑OR for logistic/ordinal, log‑RR for Poisson/NegBin, raw β for linear). |
| `family` | Family name string. Mixed names (e.g. `"linear_mixed"`) are stripped to base. |
| `ci_bound` | Optional CI bound for E‑value of CI. |
| `sd_x` | Standard deviation of predictor (required for linear). |
| `sd_y` | Standard deviation of outcome (required for linear). |
| `baseline_prevalence` | Baseline outcome probability for exact RR conversion (logistic/ordinal). |

**Returns:** Dictionary with keys `e_value`, `e_value_ci`, `rr`,
`family`, and `interpretation`.

**References:** VanderWeele, T. J. & Ding, P. (2017). Sensitivity
analysis in observational research: introducing the E‑value. *Annals
of Internal Medicine*, 167(4), 268–274.

---

### `rosenbaum_bounds`

```python
rosenbaum_bounds(
    result: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    gammas: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    predictor_index: int | None = None,
    alpha: float = 0.05,
) -> dict
```

Rosenbaum sensitivity bounds for unmeasured confounding.

Computes worst‑case p‑values under varying degrees of hidden bias Γ.
**Restricted to LinearFamily with binary predictors.**

| Parameter | Description |
|---|---|
| `result` | Permutation test result dict (must contain `"p_values"` and `"family"`). |
| `X` | Design matrix used in the permutation test. |
| `y` | Outcome vector. |
| `gammas` | Sequence of Γ values to evaluate. |
| `predictor_index` | Column index of the predictor (defaults to 0). |
| `alpha` | Significance level for critical Γ. |

**Returns:** Dictionary with `predictor`, `observed_p`, `gamma_values`,
`worst_case_p`, `critical_gamma`, and `interpretation`.

**Raises:** `NotImplementedError` for non‑linear or mixed families.
`ValueError` for continuous predictors.

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

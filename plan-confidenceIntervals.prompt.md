## Plan: Confidence Intervals & Standardisation Fixes

This plan adds four types of confidence intervals to `IndividualTestResult` and fixes two correctness issues found during research. The four CI types are: (1) permutation-based CIs for regression coefficients, with BCa adjustment via a generic helper factored from the existing mediation code; (2) Clopper-Pearson exact binomial CIs for p-values, aligned with the Phipson & Smyth estimator; (3) classical Wald CIs from statsmodels, covering all seven families including per-category CIs for multinomial; and (4) CIs for standardised (beta-weight) coefficients, computed by applying the family-appropriate scaling factor to the permutation CI endpoints. Two prerequisite correctness fixes are folded in: the standardisation formula in `compute_standardized_coefs()` currently misclassifies four of six families (Poisson, NB, ordinal, multinomial all fall into the wrong branch), and the permuted-coefficient distributions from ter Braak / Freedman–Lane are centred on zero rather than on the observed coefficient, making naïve percentile CIs invalid without a shift. Both are corrected as part of the CI pipeline.

---

### Step 1 — Fix `compute_standardized_coefs()` in `diagnostics.py` (L119–155)

Expand the two-branch `if family.name == "logistic"` / `else` to cover all families with the correct link-function logic:

- `"linear"`, `"linear_mixed"` → `β · SD(X) / SD(Y)` — identity link, no change from current behaviour.
- `"logistic"`, `"ordinal"` → `β · SD(X)` — log-odds link families. Ordinal coefficients at `families.py` L3704 are log-odds slopes (same scale as logistic), so dividing by `SD(Y)` of an integer-coded ordinal outcome is conceptually wrong.
- `"poisson"`, `"negative_binomial"` → `β · SD(X)` — log-link families. Coefficients live on the log-rate scale; there is no natural `SD(Y)` denominator on the link scale, identical reasoning to logistic.
- `"multinomial"` → return `np.full_like(model_coefs, np.nan)`. The `coefs()` method at `families.py` L4429 returns Wald χ² statistics (not regression coefficients), so standardisation is meaningless.

Update the docstring to document all six family branches. Existing tests for linear and logistic standardisation must remain green.

---

### Step 2 — Factor out generic `_bca_percentile()` from `confounders.py` (L463–596)

Create a statistic-agnostic BCa percentile function that encapsulates the three generic pieces of the current `_bca_ci()`:

```python
_bca_percentile(
    boot_dist: np.ndarray,       # (B,) replicates of the statistic
    observed_stat: float,        # point estimate
    jackknife_stats: np.ndarray, # (n,) leave-one-out estimates
    alpha: float,                # 1 - confidence_level
) -> tuple[float, float]        # (ci_lower, ci_upper)
```

The three reusable blocks, currently at:

- **Bias correction z₀** — `confounders.py` L496–L507: `z0 = Φ⁻¹(clip(mean(boot_dist < observed_stat)))`.
- **Acceleration â** — `confounders.py` L539–L549: `â = Σ(θ̄ − θ̂₍₋ᵢ₎)³ / [6·(Σ(θ̄ − θ̂₍₋ᵢ₎)²)^{3/2} + ε]`.
- **Adjusted percentiles** — `confounders.py` L555–L582: BCa formula with clipping, then `np.percentile`.

Place the new function in `diagnostics.py` (the natural home for CI computation; avoids a backwards import from confounders into diagnostics).

Refactor the existing `_bca_ci()` in `confounders.py` to import and delegate to `_bca_percentile()`, passing its mediation-specific `jackknife_indirect = a_jack_slopes * b_jack_slopes` array as the `jackknife_stats` argument. The mediation-specific jackknife computation (a-path OLS at L524–L529, b-path family fit at L531–L541, product at L543) stays in `_bca_ci()`.

---

### Step 3 — Add `compute_jackknife_coefs()` to `diagnostics.py`

New function that produces leave-one-out coefficient estimates for the BCa acceleration constant:

```python
compute_jackknife_coefs(
    family: ModelFamily,
    X: np.ndarray,        # (n, p)
    y_values: np.ndarray, # (n,)
    fit_intercept: bool,
) -> np.ndarray | None    # (n, p) or None if n > 500
```

Implementation:

- Build the `(n, n−1)` leave-one-out index array (reuse the pattern from `confounders.py` L519–L522).
- For each observation `i`, construct `X_loo = X[loo_idx[i]]`, `y_loo = y[loo_idx[i]]`.
- Fit via `family.fit(X_loo, y_loo, fit_intercept)` and extract `family.coefs(model)`.
- If `family` has a `batch_fit_paired()` method (present on linear, logistic, Poisson, NB), use it for vectorised leave-one-out fits: `family.batch_fit_paired(X_loo_3d, Y_loo_2d, fit_intercept)` → `(n, p)`. Fall back to a sequential loop otherwise (ordinal, multinomial, LMM).
- **Guard**: if `n > 500`, return `None` instead of fitting. The caller falls back to simple shifted-percentile CIs. BCa's coverage advantage over percentile CIs diminishes at large n, and `n` refits is too expensive for interactive use at that scale.
- For multinomial, `family.coefs()` returns Wald χ² per predictor — the jackknife operates on those same χ² values (BCa is agnostic to the statistic's interpretation).

---

### Step 4 — Add CI computation functions to `diagnostics.py`

Four new module-level functions:

#### 4a. `compute_permutation_ci()`

```python
compute_permutation_ci(
    permuted_coefs: np.ndarray,  # (B, p)
    model_coefs: np.ndarray,     # (p,)
    method: str,                 # strategy name
    alpha: float,                # 1 - confidence_level
    jackknife_coefs: np.ndarray | None,  # (n, p) or None
    confounders: list[str],
    feature_names: list[str],
) -> np.ndarray                  # (p, 2)
```

- **Strategy-aware centering**: if `method` in `{"ter_braak", "freedman_lane"}`, add `model_coefs[j]` to `permuted_coefs[:, j]` before computing percentiles, because those strategies produce null distributions centred on zero (ter Braak at `ter_braak.py` L147, Freedman–Lane at `freedman_lane.py` L131). If `method` in `{"score", "kennedy"}`, no shift — score strategy explicitly re-centres on β̂ at `score.py` L164; Kennedy individual also produces coefficients centred on the observed estimate.
- **BCa path**: if `jackknife_coefs is not None`, call `_bca_percentile(shifted[:, j], model_coefs[j], jackknife_coefs[:, j], alpha)` for each feature `j`.
- **Percentile fallback**: if `jackknife_coefs is None`, compute `np.percentile(shifted[:, j], [100*alpha/2, 100*(1-alpha/2)])`.
- **Confounder masking**: features named in `confounders` → `[NaN, NaN]`.

#### 4b. `compute_pvalue_ci()`

```python
compute_pvalue_ci(
    counts: np.ndarray,    # (p,) — number of permuted |β*| ≥ observed |β|
    n_permutations: int,   # B
    alpha: float,          # 1 - confidence_level
) -> np.ndarray            # (p, 2)
```

- Clopper-Pearson exact binomial CI via `scipy.stats.beta.ppf`:
  - `successes = counts + 1` (aligned with the `+1` numerator in the Phipson & Smyth estimator at `pvalues.py` L139).
  - `trials = n_permutations + 1`.
  - `lower = beta.ppf(alpha/2, successes, trials - successes + 1)`.
  - `upper = beta.ppf(1 - alpha/2, successes + 1, trials - successes)`.
- Edge case: when `successes == 0`, lower is 0; when `successes == trials`, upper is 1.

#### 4c. `compute_wald_ci()`

```python
compute_wald_ci(
    observed_model: Any,
    family: ModelFamily,
    n_features: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray | None]
    # returns (wald_ci, category_wald_ci)
    # wald_ci shape: (p, 2) for per-predictor CIs
    # category_wald_ci shape: (p, K-1, 2) for multinomial, else None
```

Family dispatch:

- **Linear / Logistic / Poisson / Negative Binomial**: `observed_model.conf_int(alpha)` returns `(p_aug, 2)` (intercept + slopes). Strip the intercept row: `ci_matrix[1:n_features+1]`. Return `(wald_ci, None)`.
- **Ordinal**: `observed_model.conf_int(alpha)` returns `(p + K−1, 2)` (slopes first, then thresholds). Slice `[:n_features]` for slope-only CIs. Return `(wald_ci, None)`.
- **Multinomial**: `observed_model.conf_int(alpha)` returns `((K−1)·p_aug, 2)`. Reshape to `(K−1, p_aug, 2)`, strip intercept column → `(K−1, p, 2)`, transpose to `(p, K−1, 2)` to match `category_coefs()` convention. Per-predictor Wald CI is `[NaN, NaN]` (χ² statistic has no direct Wald CI on the coefficient scale). Return `(nan_array, category_wald_ci)`.
- **Linear Mixed**: call `observed_model.conf_int(alpha)` if available from the statsmodels `MixedLMResults` stored in `FitContext.sm_mixed_model`; if unavailable, return `(nan_array, None)`.
- Wrap in `try/except` — degenerate fits (separation, singular Hessian) may produce `inf` or raise; return `NaN`-filled arrays on failure.

#### 4d. `compute_standardized_ci()`

```python
compute_standardized_ci(
    permutation_ci: np.ndarray,  # (p, 2)
    model_coefs: np.ndarray,     # (p,)
    X: pd.DataFrame,
    y_values: np.ndarray,
    family: ModelFamily,
) -> np.ndarray                  # (p, 2)
```

- Compute the same per-feature scaling factor used in the corrected `compute_standardized_coefs()`:
  - `"linear"`, `"linear_mixed"` → `scale[j] = SD(X_j) / SD(Y)`.
  - `"logistic"`, `"ordinal"`, `"poisson"`, `"negative_binomial"` → `scale[j] = SD(X_j)`.
  - `"multinomial"` → return `np.full((p, 2), np.nan)`.
- Apply: `standardized_ci[:, 0] = permutation_ci[:, 0] * scale`, `standardized_ci[:, 1] = permutation_ci[:, 1] * scale`. This works because standardisation is a linear transformation of the coefficient CIs.

---

### Step 5 — Expose `counts` from `pvalues.py`

Modify `calculate_p_values()` at `pvalues.py` L60 to return `counts` (the raw `np.sum(np.abs(permuted_coefs) >= np.abs(model_coefs), axis=0)` array, currently computed at L130–L132) as a fifth element in its return tuple.

Update the unpacking in `core.py` L700–L710:

```python
permuted_p_values, classic_p_values, raw_empirical_p, raw_classic_p, counts = (
    calculate_p_values(...)
)
```

---

### Step 6 — Add `confidence_intervals` field to `_results.py`

On `IndividualTestResult` (frozen dataclass at L132), add a new optional field after `extended_diagnostics`:

```python
confidence_intervals: dict[str, Any] = field(default_factory=dict)
```

The dict structure:

| Key | Type | Description |
|---|---|---|
| `"permutation_ci"` | `list[list[float]]` (p×2) | BCa or percentile CIs for regression coefficients |
| `"pvalue_ci"` | `list[list[float]]` (p×2) | Clopper-Pearson CIs for empirical p-values |
| `"wald_ci"` | `list[list[float]]` (p×2) | Classical Wald CIs (NaN for multinomial per-predictor) |
| `"standardized_ci"` | `list[list[float]]` (p×2) | CIs for standardised coefficients |
| `"category_wald_ci"` | `list[list[list[float]]]` (p×(K-1)×2) | Per-category Wald CIs (multinomial only; absent otherwise) |
| `"confidence_level"` | `float` | e.g. 0.95 |
| `"ci_method"` | `str` | `"bca"` or `"percentile"` |

On `JointTestResult` (at L248), add `extended_diagnostics: dict[str, Any] = field(default_factory=dict)` to bring it to parity with `IndividualTestResult`.

Update `_SERIALIZERS` if needed to handle the new field in `to_dict()`.

---

### Step 7 — Add `confidence_level` parameter to the public API

Add `confidence_level: float = 0.95` to:

- `permutation_test_regression()` signature at `core.py` L46.
- `PermutationEngine.__init__()` signature at `engine.py` L59.
- Store as `self.ctx.confidence_level` on `FitContext`.

Add `confidence_level: float | None = None` field to `FitContext` at `_context.py`.

Thread the parameter through the internal call chain so it reaches `_package_individual_result()`.

---

### Step 8 — Wire CI computation into `_package_individual_result()` at `core.py` (L681–766)

After the existing `compute_all_diagnostics()` call (≈L731), add the CI computation pipeline:

1. **Alpha**: `alpha = 1 - confidence_level`
2. **Jackknife**: `jackknife_coefs = compute_jackknife_coefs(engine.family, X.values.astype(float), y_values, fit_intercept)` — returns `(n, p)` or `None` if n > 500.
3. **Permutation CI**: `perm_ci = compute_permutation_ci(permuted_coefs, engine.model_coefs, method, alpha, jackknife_coefs, confounders, list(X.columns))`
4. **P-value CI**: `pval_ci = compute_pvalue_ci(counts, n_permutations, alpha)`
5. **Wald CI**: `wald_ci, cat_ci = compute_wald_ci(engine.ctx.observed_model, engine.family, len(engine.model_coefs), alpha)`
6. **Standardised CI**: `std_ci = compute_standardized_ci(perm_ci, engine.model_coefs, X, y_values, engine.family)`
7. **Bundle**:

   ```python
   ci_dict = {
       "permutation_ci": perm_ci.tolist(),
       "pvalue_ci": pval_ci.tolist(),
       "wald_ci": wald_ci.tolist(),
       "standardized_ci": std_ci.tolist(),
       "confidence_level": confidence_level,
       "ci_method": "bca" if jackknife_coefs is not None else "percentile",
   }
   if cat_ci is not None:
       ci_dict["category_wald_ci"] = cat_ci.tolist()
   ```

8. Pass `confidence_intervals=ci_dict` to the `IndividualTestResult` constructor.

Mask confounder columns across all CI arrays with `[NaN, NaN]`, matching the existing confounder p-value masking pattern at `core.py` L717–L721.

---

### Step 9 — Tests

#### 9a. Standardisation fix

- Verify `compute_standardized_coefs()` returns `β · SD(X)` (no SD(Y) denominator) for Poisson, NB, and ordinal families on a synthetic dataset.
- Verify multinomial returns an array of NaN values.
- Verify linear and logistic remain unchanged (regression guard).

#### 9b. BCa helper

- Unit test `_bca_percentile()` with a known normal distribution (z₀ ≈ 0, â ≈ 0 → should recover standard percentile CI).
- Test with a skewed distribution where z₀ ≠ 0 and verify the CI is asymmetric.
- Test edge case: all jackknife stats identical → â = 0, no division-by-zero.

#### 9c. Jackknife

- Verify `compute_jackknife_coefs()` shape is `(n, p)` on a tiny linear example (n=20, p=2).
- Verify leave-one-out correctness: manually fit with observation 0 removed, compare to row 0.
- Verify n > 500 guard returns `None`.

#### 9d. Permutation CI

- Verify strategy-aware centering: run ter Braak and score on the same data with forced permuted_coefs; confirm ter Braak CIs are shifted by β̂ and score CIs are unshifted.
- Verify BCa vs percentile fallback: with jackknife=None, function returns simple percentile CI.
- Verify confounder columns return `[NaN, NaN]`.

#### 9e. Clopper-Pearson

- Hand-compute `scipy.stats.beta.ppf` for counts=3, B=99 and verify the function matches.
- Verify edge cases: counts=0, counts=B.

#### 9f. Wald CI

- Linear: compare function output to `model.conf_int()` directly.
- Logistic: same comparison.
- Ordinal: verify slicing excludes threshold parameters (output shape is `(p, 2)`, not `(p + K−1, 2)`).
- Multinomial: verify per-predictor returns NaN; verify `category_wald_ci` shape is `(p, K−1, 2)`.
- Degenerate case: singular design matrix → returns NaN-filled array, no crash.

#### 9g. Standardised CI

- Verify scaling matches `compute_standardized_coefs()` ratio for linear (SD(X)/SD(Y)) and logistic (SD(X)).
- Verify multinomial returns NaN.

#### 9h. Integration

- Call `permutation_test_regression()` with `confidence_level=0.95` for each of: linear, logistic, Poisson, NB, ordinal, multinomial, linear_mixed.
- Verify `result.confidence_intervals` dict has all expected keys.
- Verify shapes: `permutation_ci` is `(p, 2)`, `pvalue_ci` is `(p, 2)`, etc.
- Verify `confidence_level` is `0.95` and `ci_method` is `"bca"` or `"percentile"`.
- For multinomial, verify `category_wald_ci` key is present with shape `(p, K−1, 2)`.

#### 9i. Regression guard

- Full test suite (851 existing tests) must pass unchanged.

---

### Step 10 — Housekeeping

- Update `CHANGELOG.md` with a new section documenting the four CI types, the standardisation fix, and the BCa helper factoring.
- Update `docs/ROADMAP.md` to check off CI-related items.
- Update `plan-unifiedMixedModel.prompt.md` with a new "Plan D: Confidence Intervals" section and strike through completed steps.
- Update example scripts in `examples/` to demonstrate accessing `result.confidence_intervals`.
- Run `ruff check`, `mypy`, and full `pytest` suite.

---

### Verification

- `conda activate randomization-tests && python -m pytest tests/ -x -q` — all tests pass (851 existing + ~30 new CI tests)
- `conda activate randomization-tests && python -m ruff check src/ tests/` — clean
- `conda activate randomization-tests && python -m mypy src/randomization_tests/` — clean
- Manual: run `examples/linear_regression.py`, inspect `result.confidence_intervals` for correct structure and plausible values

---

### Decisions

- **Strategy-aware centering**: ter Braak and Freedman–Lane null distributions shifted by `+model_coefs[j]` before CI computation; score and Kennedy left unshifted (already centred on β̂_obs by construction — score at `score.py` L164, Kennedy via full-model refit)
- **BCa vs percentile threshold**: jackknife computed only when n ≤ 500; above that, simple shifted-percentile CIs. BCa's coverage advantage is principally for small n, which is this project's focus
- **BCa helper placement**: `_bca_percentile()` in `diagnostics.py`; existing `_bca_ci()` in `confounders.py` refactored to delegate
- **Multinomial CIs**: per-category Wald CIs via `MNLogitResults.conf_int()` reshaped to `(p, K−1, 2)` matching `category_coefs()` convention. Per-predictor Wald CI, permutation CI standardised CI all NaN for multinomial (Wald χ² statistics are not on a coefficient scale). Permutation CIs are computed on the χ² values as-is (BCa is agnostic to the statistic's interpretation)
- **Ordinal Wald CIs**: slice `[:n_features]` from `conf_int()` output, excluding threshold parameters
- **Clopper-Pearson parameterisation**: `successes = counts + 1`, `trials = B + 1`, exactly matching the Phipson & Smyth `(b+1)/(B+1)` estimator
- **Standardisation fix**: ordinal reclassified as log-odds link (same as logistic); Poisson and NB reclassified as log-link (no SD(Y) denominator); multinomial returns NaN
- **CI storage**: single `confidence_intervals: dict` field on `IndividualTestResult`; `JointTestResult` gets `extended_diagnostics` for parity but no CIs (joint tests produce a single scalar p-value, not per-predictor coefficients)
- **`counts` exposure**: `calculate_p_values()` return type extended to 5-tuple to avoid recomputing the comparison

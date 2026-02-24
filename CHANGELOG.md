# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-23

### Added

- **`permute_hook()` extension point (Step 8):**
  `PermutationEngine.permute_indices()` converted from `@staticmethod`
  to instance method.  New `_permute_hook()` method serves as the
  extension point for v0.4.1 exchangeability-constrained permutations.
  Subclasses override `_permute_hook()` to restrict permutations
  within exchangeability cells.  Two new tests added:
  `test_permute_indices_is_instance_method` and
  `test_permute_hook_override`.
- **`_classical_p_values_fallback` removal + family warning hygiene (Step 7 + 7a):**
  Deleted the dead `_classical_p_values_fallback()` function from
  `pvalues.py`.  `family` parameter on `calculate_p_values()` changed
  from `ModelFamily | None = None` to required `ModelFamily`.
  Removed `statsmodels` imports (`sm`, `SmConvergenceWarning`,
  `PerfectSeparationWarning`) that were only used by the fallback.
  Updated module and function docstrings to remove "backward-compatible
  call" / "falls back" language.
- **Family warning hygiene (7a rider):**  Normalised
  `warnings.catch_warnings()` suppression across all 6 families'
  `diagnostics()`, `classical_p_values()`, and `fit()` methods.
  `LinearFamily.diagnostics()` now suppresses `RuntimeWarning`;
  `LogisticFamily.diagnostics()` now suppresses
  `SmConvergenceWarning`, `PerfectSeparationWarning`, `RuntimeWarning`;
  `PoissonFamily` and `NegativeBinomialFamily` methods now suppress
  `PerfectSeparationWarning` in addition to existing warnings;
  `OrdinalFamily.fit()` and `MultinomialFamily.fit()` now suppress
  `PerfectSeparationWarning` in addition to existing warnings.
  Removed the belt-and-suspenders `warnings.catch_warnings()` block
  from `engine.py` (wrapping `self.family.diagnostics()`), along with
  the `SmConvergenceWarning`/`PerfectSeparationWarning` imports.
  Every family is now self-contained for warning suppression.
- **`model_type` removal + self-contained display (Step 6):**
  `model_type` field deleted from both `IndividualTestResult` and
  `JointTestResult`.  `family` field changed from `str` to
  `ModelFamily` instance.  Zero `model_type` references remain
  anywhere in the codebase.
- **`_SERIALIZERS` registry:** `_DictAccessMixin` gains a
  `_SERIALIZERS: ClassVar[dict[str, Any]]` class variable.
  `to_dict()` composes serializers (e.g. `family → family.name`)
  with the existing `_numpy_to_python()` conversion.
- **Self-contained display:** `print_results_table(results, *, title=...)`,
  `print_joint_results_table(results, *, title=...)`, and
  `print_diagnostics_table(results, *, title=...)` now extract
  `family`, `feature_names`, and `target_name` directly from the
  result object — no parameter passing required.
- **New result fields:** `feature_names`, `target_name`,
  `n_permutations`, `groups`, and `permutation_strategy` added to
  both `IndividualTestResult` and `JointTestResult`.

### Changed

- `compute_standardized_coefs` and `compute_cooks_distance` now take
  `family: ModelFamily` instead of `model_type: str`.
- `print_confounder_table` `family` parameter changed from
  `str | None` to `ModelFamily | None`.
- All 6 example scripts updated to use self-contained display calls.
- `API.md` and `QUICKSTART.md` updated for new display signatures.

- **`compute_extended_diagnostics()` protocol method:** each
  `ModelFamily` now owns its family-specific model-level diagnostic
  computation (`breusch_pagan`, `deviance_residuals`, `poisson_gof`,
  `nb_gof`, `ordinal_gof`, `multinomial_gof`).  The 170-line
  `model_type` branch block in `diagnostics.py` is deleted; replaced
  by a single `result.update(family.compute_extended_diagnostics(...))`
  dispatch.

### Changed

- `compute_all_diagnostics()` parameter `model_type: str` replaced
  by `family: ModelFamily`.  `core.py` call site updated from
  `model_type=engine.family.name` to `family=engine.family`.
- Family-specific diagnostic logic migrated from `diagnostics.py`
  into `LinearFamily`, `LogisticFamily`, `PoissonFamily`,
  `NegativeBinomialFamily`, `OrdinalFamily`, and `MultinomialFamily`.
  Each implementation includes try/except with NaN-filled sentinel
  dicts for graceful degradation on degenerate data.
- Tests updated: `test_diagnostics.py` passes `family=` instances
  instead of `model_type=` strings; `test_families.py` gains 6 new
  `compute_extended_diagnostics` assertions.

- **`ModelFamily` protocol:** strategy pattern decoupling model fitting
  from the permutation engine.  Each family implements `validate_y`,
  `fit`, `predict`, `coefs`, `residuals`, `reconstruct_y`, `fit_metric`,
  `diagnostics`, `classical_p_values`, `exchangeability_cells`,
  `batch_fit`, and `batch_fit_varying_X`.
- **`_backends/` package:** `BackendProtocol` with `NumpyBackend` and
  `JaxBackend` implementations.  `resolve_backend()` auto-detects JAX.
- **Four new GLM families:** Poisson, negative binomial (NB2),
  ordinal (proportional-odds logistic), and multinomial (softmax)
  regression — all with JAX Newton–Raphson batch solvers and
  NumPy/statsmodels fallback.
- **Negative binomial `calibrate()` pattern:** dispersion α estimated
  once on observed data via MLE, held fixed for all permutation refits.
  Duck-typed, idempotent, frozen-dataclass design.
- **Multinomial LRT test statistic:** per-predictor likelihood-ratio
  chi-squared preserves the `(p,)` protocol contract.
  `category_coefs(model)` convenience method for `(p, K−1)` inspection.
- **Ordinal proportional-odds test:** Brant-like χ² test comparing
  pooled slopes to category-specific binary logit slopes, reported in
  diagnostics.
- **Freedman–Lane (1983) permutation method:** `method="freedman_lane"`
  (individual) and `method="freedman_lane_joint"` — full-model residual
  permutation with reduced-model fitted values.  Better power than
  Kennedy when predictors are correlated.
- **`PermutationEngine` class:** resolves family, backend, calibration,
  and permutation indices at construction time.
  `permutation_test_regression()` is now a thin wrapper.
- **`family=` parameter** on `permutation_test_regression()`,
  `identify_confounders()`, `mediation_analysis()`, and
  `print_confounder_table()`.
- **`n_jobs=` parameter** for joblib parallelisation of batch-fit loops
  (NumPy backend).  Ignored when JAX is active.
- **Typed result objects:** `IndividualTestResult` and `JointTestResult`
  frozen dataclasses with `_DictAccessMixin` for backward-compatible
  dict-like access and `.to_dict()` JSON serialisation.
- **Count auto-detection warning:** `family="auto"` warns when Y looks
  like count data (non-negative integers, >2 unique values).
- **JAX improvements:** float64 precision, triple convergence criteria,
  `while_loop` dynamic early exit (25× speedup), damped Hessian
  regularisation, aggregated convergence warnings.
- **Integration tests:** 50 new tests covering all families × methods,
  Freedman–Lane rejection for ordinal/multinomial, cross-family schema
  consistency, confounder module with each family.
- **Example scripts:** `poisson_regression.py`,
  `negative_binomial_regression.py`, `ordinal_regression.py`,
  `multinomial_regression.py`.

### Changed

- `core.py` refactored from `is_binary` branching to family-dispatched
  method calls via `_strategies/` package.
- `compute_all_diagnostics` accepts `model_type: str` instead of
  `is_binary: bool`.
- `calculate_p_values` delegates to `family.classical_p_values()`.
- Display tables support all six families with family-specific headers,
  diagnostics panels, and stat labels.
- Test suite expanded from 163 to ~580 tests.

### Removed

- `_batch_ols_coefs()` from `core.py` (superseded by
  `NumpyBackend.batch_ols()`).
- `_compute_diagnostics()` from `core.py` (superseded by
  `family.diagnostics()`).
- Top-level `_jax.py` shim (superseded by `_backends/_jax.py`).

## [0.2.0] - 2026-02-22

### Added

- **Permutation deduplication profiling:** benchmarked the three-tier
  strategy (Lehmer, dedup, vectorised) across a (n, B) grid from n=8
  to n=10,000.  Published analysis at
  `docs/permutation-dedup-performance.md` with methodology, heatmaps,
  and regime analysis.  Current thresholds (`max_exhaustive=10`,
  collision bound < 1e-9) validated — no changes needed.
- **Edge-case tests** (`test_edge_cases.py`): 30 tests covering empty
  DataFrames, single-feature models, constant columns, perfect
  separation, and permutation-count boundaries.
- **JAX convergence tests** (`test_jax_convergence.py`): 9 tests for
  ill-conditioned Hessians, rank-deficient designs, and perfect
  separation under the custom Newton-Raphson solver.
- **Large-n smoke tests** (`test_smoke.py`): 5 tests verifying memory
  and runtime at n=10,000 for all methods and model types.
- **Pre-commit configuration** (`.pre-commit-config.yaml`): ruff lint +
  format, mypy with strict settings, trailing-whitespace and
  end-of-file-fixer hooks.
- **Lower-bound CI job** (`test-deps-lower`): validates against pinned
  older dependency versions (numpy 1.24, pandas 2.0, etc.) on
  Python 3.10.
- **JAX CI job** (`test-jax`): dedicated workflow job for JAX backend
  tests including slow convergence tests.
- **Slow-test CI job** (`test-slow`): runs large-n smoke tests
  separately from the fast test matrix.
- JAX backend documentation in `docs/QUICKSTART.md` (tested versions,
  install, limitations, verification).
- Branch strategy and dependency compatibility docs in
  `CONTRIBUTING.md`.

### Changed

- **JAX code extracted into `_jax.py`:** `_logistic_nll`,
  `_logistic_grad`, `_logistic_hessian_diag`, `fit_logistic_batch_jax`,
  and `fit_logistic_varying_X_jax` moved from `core.py` to a dedicated
  `_jax.py` module.  `core.py` imports from `_jax` — no protocol or
  backend abstraction yet (deferred to v0.3.0).
- CI triggers retargeted from `main` to `experimental` branch.
- mypy strict mode: `disallow_untyped_defs = true`, zero `type: ignore`
  suppressions via `TYPE_CHECKING` pattern for optional dependencies.
- `pytest` configured with `filterwarnings = ["error"]` and `slow`
  marker; `addopts` excludes slow tests by default.
- Test suite expanded from 125 to 163 tests (+ 6 slow/skipped).

## [0.1.5] - 2026-02-21

### Added

- Extended diagnostics module (`diagnostics.py`): standardised coefficients,
  VIF, Monte Carlo SE, divergence flags, Breusch-Pagan (linear),
  deviance residuals (logistic), Cook's distance, permutation coverage,
  and exposure R² (Kennedy method with confounders).
- `print_diagnostics_table` for formatted ASCII display of extended
  diagnostics with context-aware Notes section.
- Omnibus test footer clarification in `print_joint_results_table`.
- `fit_intercept` parameter threaded through all methods, diagnostics,
  and p-value calculations for consistent model specification.
- New tests for diagnostics, display, and core intercept handling
  (125 total, up from 100).

### Fixed

- **Intercept mismatch in permutation refits (all methods).** The old
  code fitted permutation models without an intercept while the observed
  model used `fit_intercept=True`. This caused permuted slope coefficients
  to absorb the response mean, producing spurious p-values (e.g. p = 1.0
  for X6 longitude in the linear example, p = 0.0 for strong predictors).
  All permutation refits now include an intercept column matching the
  observed model specification.
- P-value formatting: `_fmt()` now uses fixed-width `f"{val:.{precision}f}"`
  instead of `f"{rounded}"`, preventing misleading `0.0` / `1.0` display.
- `calculate_p_values` now returns raw numeric arrays alongside formatted
  strings, eliminating redundant statsmodels refits in diagnostics.
- Exposure R² column suppressed when no confounders are specified
  (was displaying a wall of `0.0000` values).

### Changed

- `calculate_p_values` return type expanded from 2-tuple to 4-tuple
  `(permuted_str, classic_str, raw_empirical, raw_classic)`.
- Examples refactored to use the new `print_diagnostics_table` API.
- Docs (API.md, QUICKSTART.md, ROADMAP.md) updated for v0.1.5.

## [0.1.1] - 2026-02-21

### Added

- Optional Polars input support: all public API functions (`permutation_test_regression`,
  `screen_potential_confounders`, `mediation_analysis`, `identify_confounders`,
  `calculate_p_values`) now accept `polars.DataFrame` and `polars.LazyFrame`
  in addition to `pandas.DataFrame`. Polars inputs are converted to pandas at
  the API boundary; internal computation remains NumPy-based.
- New `_compat.py` module with `_ensure_pandas_df` converter and runtime Polars
  detection (no hard dependency).
- `polars` optional dependency group (`pip install randomization-tests[polars]`).
- Comprehensive test suite for Polars input compatibility, including a parity
  test verifying identical results from Polars and pandas inputs.

## [0.1.0] - 2026-02-20

### Added

- `permutation_test_regression` supporting ter Braak (1992), Kennedy (1995)
  individual, and Kennedy (1995) joint methods.
- Vectorised OLS via batch pseudoinverse multiplication.
- Optional JAX backend (`jax.vmap` + `jax.grad` Newton-Raphson) for batched
  logistic regression with transparent numpy/sklearn fallback.
- Pre-generated unique permutation indices with hash-based deduplication and
  exhaustive enumeration for small *n*.
- Phipson & Smyth (2010) corrected p-values that are never exactly zero.
- Confounder identification pipeline: correlation screening + Preacher & Hayes
  (2004, 2008) mediation analysis with BCa bootstrap CIs.
- Formatted ASCII table output (`print_results_table`,
  `print_joint_results_table`).
- PEP 561 `py.typed` marker for downstream type-checking support.
- GitHub Actions CI (lint, type check, test matrix across Python 3.10–3.13).
- `get_backend()` / `set_backend()` runtime API and
  `RANDOMIZATION_TESTS_BACKEND` environment variable for selecting the
  compute backend (`"jax"`, `"numpy"`, or `"auto"`).

### Changed

- Replaced Baron & Kenny (1986) causal-steps mediation with Preacher & Hayes
  (2004, 2008) bootstrap test using bias-corrected and accelerated (BCa)
  confidence intervals (Efron, 1987). Default bootstrap samples increased
  from 1 000 to 5 000. Return dict now includes `ci_method` key.

### Fixed

- JAX logistic solvers omitting the intercept column, which caused
  the Newton-Raphson solver to fit a different model specification than
  sklearn's `LogisticRegression(fit_intercept=True)` default. ter Braak
  and Kennedy empirical p-values now match across backends.

### Removed

- Unused `_typing.py` module and dead `_fit_logistic_jax` single-fit
  function (superseded by the batched `_fit_logistic_batch_jax`).

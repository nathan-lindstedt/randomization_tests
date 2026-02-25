# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - Unreleased

### Added

- **`groups=` parameter** on `permutation_test_regression()`:
  accepts array-like integer labels, `pd.Series`, or `pd.DataFrame`
  (single-column extracted as 1-D; multi-column cross-classified into
  integer cell labels).  When provided, permutations respect group
  structure instead of shuffling globally.
- **`permutation_strategy=` parameter**: `"within"` (default when
  `groups` is provided), `"between"`, or `"two-stage"`.  Controls
  how permutations interact with exchangeability cells.
- **`permutation_constraints=` callback parameter**: optional
  post-filter that receives `(B, n)` and returns `(B', n)`.  Engine
  back-fills gaps by generating more permutations and re-filtering.
  Validated on construction via shape probe.
- **Three new permutation generators** in `permutations.py`:
  `generate_within_cell_permutations()`,
  `generate_between_cell_permutations()`,
  `generate_two_stage_permutations()`.  All support hash-based dedup,
  identity exclusion, budget capping with warnings, and
  reproducibility via `random_state`.
- **`_validate_groups()` and `_validate_constraints()`** in `core.py`:
  input validation for group labels, strategy strings, callback
  shape probing, singleton cell warnings, two-stage imbalance
  warnings, and minimum-group-count enforcement (G ≥ 5 for
  `"between"`).
- **Engine dispatch** in `engine.py`: `_permute_hook()` now routes
  to cell-constrained generators based on `permutation_strategy`,
  falls back to `family.exchangeability_cells()` when no explicit
  `groups` are provided, and applies callback post-filters.
- **Result wiring**: `IndividualTestResult.groups`,
  `.permutation_strategy` and `JointTestResult.groups`,
  `.permutation_strategy` are now populated from the engine instead
  of hard-coded to `None`.
- **48 new tests** across `test_permutations.py` (23),
  `test_core.py` (21), and `test_engine.py` (4) covering generators,
  validation, dispatch, budget warnings, singleton warnings,
  imbalance warnings, callback validation, and end-to-end integration.
- **Mixed-radix Lehmer-code sampling** for within-cell and two-stage
  (balanced) generators: when the composite reference set
  ∏ n_c! ≤ 50,000, permutation ranks are sampled without replacement
  via the factorial number system — zero collisions, exact enumeration
  when B ≥ available.  New `_unrank_within_cell()` helper decomposes
  composite ranks into per-cell sub-permutations.
- **Vectorised batch generation** for within-cell permutations:
  large-cell regimes (∏ n_c! > threshold) now generate all B
  candidates in one `np.tile` + per-cell `rng.permuted(axis=1)` call
  with post-hoc hash dedup, replacing one-at-a-time Python loop.
- **Safety cap on global generator gap-fill**: the previously
  unbounded `while count < n_permutations` gap-fill loop in
  `generate_unique_permutations` now has a `max_attempts` bound,
  preventing hangs in degenerate edge cases.
- **19 new tests** for the new code paths: `TestUnrankWithinCell` (4),
  `TestWithinCellLehmerPath` (4), `TestWithinCellVectorisedPath` (4),
  `TestTwoStageLehmerPath` (5), `TestGlobalGapFillCap` (2).
- **Cell generator benchmark** (`benchmarks/profile_cell_generators.py`):
  26-scenario timing harness for all four generators with `--tag`
  support for before/after comparison.
- **Between-cell infeasibility validation** in `core.py`:
  `permutation_strategy='between'` now raises a `ValueError` with
  actionable guidance when all cells have unique sizes (no valid
  permutations exist), and emits a `UserWarning` when fewer than 100
  between-cell permutations are available, directing users to
  `'within'` or `'two-stage'` as alternatives.
- **End-to-end engine benchmark** (`benchmarks/profile_endtoend.py`):
  measures full `permutation_test_regression()` pipeline wall time
  across 14 scenarios varying predictor count (p=1–20), sample size
  (n=100–1000), model family (linear, logistic, Poisson), and
  permutation strategy (global, within, two-stage).  Generates 3
  charts: time vs. p, strategy overhead ratio, family comparison.
- **2 new tests** in `test_core.py`:
  `test_between_infeasible_all_unique_sizes_raises` and
  `test_between_low_budget_warns` covering the new validation logic.

### Changed

- `PermutationEngine.__init__()` accepts three new keyword arguments
  (`groups`, `permutation_strategy`, `permutation_constraints`) and
  stores them as instance attributes before permutation generation.
- `_permute_hook()` rewritten from simple `generate_unique_permutations`
  passthrough to full strategy dispatcher with family cell fallback.
- `permutations.py` module docstring expanded to document all 6
  generation strategies (3 global + 3 cell-constrained).
- **Two-stage within-cell shuffle vectorised**: the B×G Python loop
  that applied per-row `rng.permutation()` within each cell is
  replaced by a G-iteration loop using `rng.permuted(block, axis=1)`
  — all B rows shuffled at C-level in a single call per cell.
  Speedups: `paired_design_30` 8.2× (1.84s → 0.23s),
  `triplet_design_20` 6.1× (1.19s → 0.20s), large-n designs
  1.1–1.5× at B=9,999.
- **`@final` decorators on all 6 family classes** and
  `_InterceptOnlyOLS`: marks them as non-subclassable, enabling
  mypy to resolve `Self` as the exact concrete type (eliminates the
  `type: ignore[return-value]` on `NegativeBinomialFamily.calibrate()`).
- **`@dataclass(frozen=True)` added to `MultinomialFamily`**:
  aligns with the other 5 families that already had it, providing
  consistent `__eq__`, `__hash__`, and immutability semantics.
- **`NegativeBinomialFamily` method ordering normalised**: Internal
  helpers and Calibration sections moved to match all other families'
  canonical section order (Display → Validation → Single-model →
  Permutation → Scoring → Diagnostics → Exchangeability → Calibration
  → Batch fitting).
- **Comprehensive structural-subtyping comments** added to
  `ModelFamily` protocol methods (`exchangeability_cells()`,
  `calibrate()`) and all 5 no-op `calibrate()` implementations,
  explaining why explicit overrides are required under duck-typed
  Protocol semantics.
- **`_TrackingFamily` in `test_engine.py`** rewritten from
  `LinearFamily` subclass to delegation wrapper, compatible with
  `@final` sealing.
- **37 unnecessary `# noqa: ARG002` annotations removed** from
  `families.py`: the `ARG` rule set is not enabled in the project's
  ruff configuration, so these annotations were inert.

## [0.4.0] - Unreleased

### Added

- **`score()` / `null_score()` protocol methods:**
  `score(model, X, y) -> float` returns a deviance-like scalar for the
  fitted model (prediction-based families use `fit_metric`, model-object
  families use `−2 × llf`).  `null_score(y, fit_intercept) -> float`
  returns the null-model baseline.  Eliminates all `hasattr` /
  `model_fit_metric` / `null_fit_metric` duck-typing on `OrdinalFamily`
  and `MultinomialFamily`.  Implemented on all 6 families.
- **`fit_reduced()` module-level function** (in `families.py`):
  centralises the "fit confounders-only or fall back to intercept-only
  predictions" logic previously duplicated in four strategy files.
- **`batch_fit_and_score()` / `batch_fit_and_score_varying_X()`:**
  new protocol methods combining fitting and scoring in a single
  vectorised call.  Used by Kennedy joint
  (`batch_fit_and_score_varying_X`) and Freedman-Lane joint
  (`batch_fit_and_score`) strategies.  JAX backend uses `vmap`; NumPy
  backend uses sequential loop with `n_jobs` support.
- **`batch_fit_paired()` — confounder bootstrap/jackknife vectorisation:**
  new `batch_fit_paired(X_batch, Y_batch, fit_intercept)` method where
  both X and Y vary per replicate (bootstrap resamples or jackknife
  leave-one-out).  JAX backend uses `vmap(_solve_one, in_axes=(0, 0))`.
  `confounders.py` bootstrap loop (1,000 iterations) and jackknife
  loop (*n* iterations) refactored from sequential `fit()` calls to
  single `batch_fit_paired()` calls.  6 new JAX methods, 6 NumPy
  fallbacks, protocol + 6 family implementations.
- **`backend=` parameter** on `permutation_test_regression()` /
  `PermutationEngine`: `"numpy"`, `"jax"`, or `None` (auto-resolve).
  Enables test injection and per-call backend selection.
- **`family: str | ModelFamily`** parameter widening: users can pass
  pre-configured family instances (e.g.
  `NegativeBinomialFamily(alpha=2.0)`) directly.  `resolve_family()`
  returns instances as-is (pass-through).
- **6 new `TestBatchFitPaired` tests** verifying shape, finiteness,
  and correctness for all families (linear, logistic, Poisson, negative
  binomial, ordinal, multinomial).

### Changed

- All four strategy files (`ter_braak.py`, `kennedy.py`,
  `freedman_lane.py`) refactored to use `fit_reduced()` and
  `family.score()` / `family.null_score()`.
- Kennedy joint strategy uses `batch_fit_and_score_varying_X()`.
- Freedman-Lane joint strategy uses `batch_fit_and_score()`.
- `confounders.py` bootstrap and jackknife loops vectorised via
  `batch_fit_paired()` (was sequential `fit()` per replicate).

### Fixed

- **JAX solver return-type annotations:** 15 annotations in `_jax.py`
  corrected from 2-tuple to 3-tuple `(beta, nll, converged)` — 5
  solver function signatures + 10 `_solve_one` inner functions.

### Removed

- Duck-typed `model_fit_metric()` and `null_fit_metric()` from
  `OrdinalFamily` and `MultinomialFamily` (subsumed by `score()` and
  `null_score()`).

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

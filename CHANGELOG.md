# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.2] - Unreleased

### Fixed

- **Logistic null-score prediction** (`LogisticFamily.null_score`):
  intercept-free null now predicts σ(0)=0.5 instead of 0, matching
  the logistic link at η=0.
- **Poisson/NB null-score prediction** (`PoissonFamily.null_score`,
  `NegativeBinomialFamily.null_score`): intercept-free null now
  predicts exp(0)=1 instead of 0, matching the log-link at η=0.
- **Logistic clipping harmonisation**: `LogisticFamily.fit_metric`
  clipping changed from `[0.001, 0.999]` to `[1e-15, 1−1e-15]`,
  consistent with the numerically stable softplus NLL path.
- **Engine error message**: removed hardcoded family list from
  score-project error message; now dynamically names the family
  that lacks `score_project()`.
- **Clopper-Pearson dead branch**: removed unreachable
  `np.maximum(successes, 1)` guard and `np.where(successes == 0)`
  branch in `_clopper_pearson_ci`, since `successes` is always ≥ 1.
- **Confounder p-value label**: changed `"N/A (confounder)"` to
  `"(confounder)"` — the `"N/A"` prefix was redundant clutter.
- **Confounder `pval_ci` masking**: Clopper-Pearson CIs are now
  set to NaN for confounder features at the data layer (`core.py`),
  preventing misleading CI display in both results and diagnostics
  tables.
- **Confounder `± margin` sub-row**: confounders now render a blank
  sub-row instead of a `± margin` line, preventing spurious `[!]`
  markers and `n_permutations` recommendations.
- **Confounder P-Val CI in diagnostics**: NaN CIs now render as em
  dash `—` in the diagnostics table P-Val CI column.

### Changed

- **Fisher SE singularity guard** (`_fisher_information_se` in
  `_jax.py`): replaced bare `jnp.linalg.inv` with
  `jnp.linalg.solve` + finiteness check → `jnp.linalg.pinv`
  fallback.  JAX does not raise on singular matrices, so the guard
  checks for NaN/Inf explicitly instead of using try/except.
- **Score projection singularity guard**
  (`_glm_score_projection_row` in `_jax.py`): replaced
  `np.linalg.inv(fisher)` with `np.linalg.solve(fisher, e_j)` +
  `try/except LinAlgError` → `np.linalg.pinv` fallback.
- **Poisson eta overflow guard** (`_poisson_nll`, `_poisson_grad`,
  `_poisson_hessian` in `_jax.py`): clipped η to `[-20, 20]`
  before `jnp.exp(η)` to prevent overflow.
- **Score strategy regularisation** (`score.py`): added
  `Σ_k += 1e-10·I` before `np.linalg.solve(Σ_k, …)` to stabilise
  near-singular cluster scatter matrices.
- **GLMM Fisher upgrade** (`LogisticMixedFamily.score_project`,
  `PoissonMixedFamily.score_project`): upgraded from diagonal
  `U_j / I_{jj}` to full inverse `[I⁻¹]_{jj}` with `try/except
  LinAlgError` → `pinv` fallback.
- **Distance correlation denominator** (`confounders.py`): changed
  `dvar_x * dvar_y - dcov² + 1e-300` to
  `max(dvar_x * dvar_y - dcov², 1e-300)` to prevent negative
  argument to `np.sqrt`.

### Added

- **GLMM deviance note** (`LogisticMixedFamily.diagnostics`,
  `PoissonMixedFamily.diagnostics`): diagnostics dict now includes
  `"deviance_note": "marginal (fixed-effects only)"` to clarify
  that deviance excludes BLUPs Zb̂.
- **Rosenbaum homoscedasticity documentation**: added `Notes`
  section to `rosenbaum_bounds` docstring and inline comment
  documenting the equal-variance assumption and its implications
  for heteroscedastic data.
- **BCa multinomial limitation comment**: added inline comment in
  `_jackknife_coefs` documenting that BCa bootstrap assumes an
  approximate pivot, which the multinomial χ² statistic does not
  satisfy exactly.
- **13 new tests**: `TestNoInterceptNullScore` (4),
  `TestSingularityGuards` (1), `TestSingularHessianSE` (1),
  `TestPoissonEtaOverflow` (1),
  `test_small_n_near_constant_returns_finite` (1),
  `TestConfounderDisplay` (5).

### Changed (Model Diagnostics Polish)

- **3-column model-level diagnostics grid**: restructured the
  Model-level Diagnostics section of `print_diagnostics_table` into
  a consistent Label (28ch) | Stat (14ch) | Detail layout, matching
  the per-feature diagnostics alignment.
- **`display_diagnostics()` protocol**: return type changed from
  `list[tuple[str, str]]` to `list[tuple[str, str, str]]` across
  all 10 family implementations (7 in `families.py`, 3 in
  `families_mixed.py`) and the `ModelFamily` protocol signature.
- **Cook's D rendering**: now shows `"3 obs."` in the stat column
  and `"threshold = 0.0400"` in the detail column.
- **Coverage label**: shortened from `"Permutation coverage:"` to
  `"Coverage:"`.
- **Coverage sufficiency verdict**: coverage line now ends with
  `"— sufficient"` or `"— borderline"`, computed inline from
  Clopper-Pearson CIs (whether any non-confounder CI straddles a
  significance threshold).
- **Coverage `n!` notation**: factorial overflow denominator now
  renders as `414!` instead of `> 10^414`.
- **`compute_permutation_coverage`**: returns two additional keys
  `coverage_pct` and `n_factorial_str` for display decomposition.
- **7 new tests**: `TestModelLevelDiagnosticsRendering` (7) covering
  3-column alignment, Cook's D, coverage sufficient/borderline,
  line width ≤ 80, B/denominator, and factorial overflow notation.

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
- **Residual-based permutation for `LinearMixedFamily`**: the
  mixed-effects family supports the full residual-based permutation
  pipeline (ter Braak 1992, Freedman–Lane 1983).  `fit()` rebuilds
  the GLS projection on the fly from stored Henderson components
  (`C₂₂`, `Z`) when called with a reduced design matrix, keeping
  REML variance components fixed — no per-permutation re-solve.
- **Manly (1997) fallback warning**: families with
  `direct_permutation = True` (ordinal, multinomial) now emit a
  `UserWarning` when the ter Braak strategy falls back to direct Y
  permutation, explaining the power implications (marginal vs partial
  association) and suggesting the Kennedy method as an alternative.
- **`examples/linear_multilevel_regression.py`**: new example script
  using the Parkinsons Telemonitoring dataset (UCI ID=189) —
  demonstrates `family="linear_mixed"` with ter Braak and
  Freedman–Lane, external validation against statsmodels MixedLM
  (β̂, σ², τ², ICC agree), direct `LinearMixedFamily` protocol
  usage, and random slopes for time-varying disease progression.
- **Score projection strategy** (`method="score"`,
  `method="score_joint"`, `method="score_exact"`): batch permutation
  via a single matmul per feature instead of refitting B times.
  `score_project()` added to `ModelFamily` protocol (default raises
  `NotImplementedError`), implemented on `LinearFamily` (OLS
  pseudoinverse row) and `LinearMixedFamily` (GLS projection A row).
  All 5 GLM families (`LogisticFamily`, `PoissonFamily`,
  `NegativeBinomialFamily`, `OrdinalFamily`, `MultinomialFamily`)
  implement the protocol method with `NotImplementedError` to satisfy
  structural subtyping.  Three strategy classes in
  `_strategies/score.py`: `ScoreIndividualStrategy` (per-feature
  coefficient-scale results with constant offset),
  `ScoreJointStrategy` (RSS reduction via `batch_fit_and_score`),
  and `ScoreExactStrategy` (Plan C placeholder).  Engine guard
  rejects score methods for families without `score_project()`.
  Confounder masking extended to `method="score"`.  n_jobs warning
  extended to cover score methods on linear family.
  `LinearMixedFamily.batch_fit_and_score` now falls back to OLS
  when X dimensions don't match the calibrated projection (enables
  joint tests with reduced confounder-only model).
- **37 new tests** in `tests/test_score_strategy.py`: protocol
  conformance, registry wiring, linear individual (result type,
  p-value range, power detection, null non-rejection), LMM individual,
  Score ≡ ter Braak equivalence (linear), Score ≡ Freedman–Lane
  equivalence (LMM, exact p-value match with and without confounders),
  linear joint, LMM joint, unsupported family rejection (logistic,
  Poisson), `score_exact` placeholder, confounder masking, n_jobs
  warning, determinism.
- **Stabilization Phase 5: Autodiff-powered capabilities** — Five
  new capabilities built on JAX autodiff infrastructure:
  - **Fisher information SEs** (`_fisher_information_se`): Wald p-values
    from observed Fisher information (H⁻¹ diagonal) for all 5 GLM
    families.  JAX-first, statsmodels fallback.
  - **Sandwich (robust) SEs** (`_sandwich_se`): Eicker–Huber–White
    heteroscedasticity-robust SEs via per-observation score outer
    product.  Available as `robust_se=True` on `classical_p_values()`.
  - **GLM score projection** (`score_project()`): real implementations
    for `LogisticFamily` (W = μ(1−μ)), `PoissonFamily` (W = μ),
    `NegativeBinomialFamily` (W = μ/(1+αμ)).  Unlocks
    `method="score"` and `method="score_joint"` for all GLM families.
  - **Autodiff Cook's D** (`_autodiff_cooks_distance`): influence-
    function Cook's D via D_i = (1/p) sᵢ' H⁻¹ sᵢ.  Extends Cook's
    D to ordinal and multinomial families where statsmodels lacks
    support.  JAX-first, statsmodels fallback for linear/logistic.
  - **Profile likelihood CIs** (`compute_profile_ci`): bisection on
    the profile deviance 2[NLL_profile − NLL_full] = χ²₁(α) with
    inner Newton loop for constrained optimisation.  Surfaces as
    `"profile_ci"` in `result.confidence_intervals`.  Supported for
    logistic, Poisson, NB, and ordinal families.
- **Stabilization Phase 6: P-value range surfacing** — Four display
  enhancements that surface Clopper-Pearson CI precision directly in
  the results tables:
  - **P-value CIs in results table** (`print_results_table`): each
    empirical p-value now has a sub-row showing `± margin` from the
    Clopper-Pearson CI, decimal-aligned beneath the p-value.
  - **Borderline significance marker** (`_significance_marker`):
    when the CI straddles any significance threshold, a `[!]` marker
    is appended to the `± margin` display, flagging that the
    significance conclusion is unstable.
  - **P-value CIs in diagnostics table** (`print_diagnostics_table`):
    new `P-Val CI` column shows `[lo, hi]` alongside the MC SE
    column (non-Exp-R² layouts).
  - **`n_permutations` recommendation**: when borderline p-values
    are detected, a Notes line recommends minimum B to resolve the
    ambiguity, computed by inverting the Clopper-Pearson width formula.
  - **`_recommend_n_permutations` helper**: computes minimum B using
    normal approximation to Clopper-Pearson half-width, clamped to
    `[100, 10_000_000]`.
  - 18 new tests covering all Phase 6 display features.
- **Stabilization Phase 7: Inaccuracies & staleness fixes** —
  Three corrections to bring documentation and imports up to date:
  - **`__init__.py` docstring rewrite**: module docstring now reflects
    full scope (all GLM families, mixed-effects, Freedman–Lane, score
    strategies, JAX across all families).  Six missing symbols added
    to the autosummary block: `moderation_analysis`, `compute_e_value`,
    `rosenbaum_bounds`, `ConfounderAnalysisResult`,
    `IndividualTestResult`, `JointTestResult`.
  - **API.md signature update**: `permutation_test_regression` now
    documents all 22 parameters including `p_value_threshold_three`,
    `backend`, `groups`, `permutation_strategy`,
    `permutation_constraints`, `random_slopes`, `confidence_level`,
    `panel_id`, `time_id`.  `family` widened to `str | ModelFamily`.
    `method` values updated to include `score`, `score_joint`,
    `score_exact`.  `resolve_family` signature updated to
    `(family: str | ModelFamily, y: np.ndarray | None = None)`.
  - **Relative import fix** (`display.py`): changed
    `from randomization_tests.families import ModelFamily` to
    `from .families import ModelFamily`.
  - **`core.py` docstring update**: `method` parameter now lists all
    8 strategies including score variants.  Added missing
    `random_slopes` parameter documentation.
- **Stabilization Phase 8: Comment quality lift** —
  Codebase-wide narration pass bringing ~15 under-documented
  locations up to the tutorial-quality standard set by `_jax.py`,
  `diagnostics.py`, and `confounders.py`.  Highlights:
  - **`engine.py`**: score probe duck-typing rationale, Anderson &
    Robinson (2001) citation in `_generate_for_strategy()` docstring,
    `deficit * 2` overgeneration explanation in `_apply_constraints()`.
  - **`families.py`**: `NegativeBinomialFamily.coefs()` docstring
    expanded with intercept-stripping explanation.
  - **`families_mixed.py`**: all 4 ICC formulas now carry full
    citations — Snijders & Bosker (2012) §17.2 for logistic (π²/3
    latent-scale variance), Goldstein, Browne & Rasbash (2002) for
    Poisson (log-normal approximation, level-1 variance = 1).
    Woodbury identity derivation added to `_calibrate_statsmodels()`
    with per-variable dimensional annotations.
  - **`_jax.py`**: McCullagh & Nelder (1989, §2.5) IRLS citation
    added to `_fit_glm_irls()`.  Deviance inline comment at first
    `2.0 * NLL` return.
  - **`_strategies/score.py`**: 17-line narration block in Cholesky
    reconstruction loop explaining log-Cholesky parameterisation
    (θ → L_k → Σ_k = L_k L_k'), Kronecker block-diagonal structure
    (I_G ⊗ Σ_k⁻¹), and assembly into Γ⁻¹.
  - **`display.py`**: table geometry comment (W=80, fc=22, sub-column
    widths summing to 80).
  - **`_results.py`**: `np.bool_` edge-case comment explaining NumPy
    ≥ 2.0 decoupling from `builtins.bool`.
  - **`_config.py`**: thread-safety note on `_backend_override`.
  - **`diagnostics.py`**: Clopper-Pearson (1934) citation added;
    Chinn (2000) citation for the 0.91 probit/logistic bridge
    constant; Rosenbaum (2002, §4.3) derivation for shifted null
    mean and variance; orphaned aggregate-helper comment repaired.
- **`LogisticMixedFamily`** (`family="logistic_mixed"`): GLMM for
  clustered binary data via Laplace approximation.  Inner IRLS
  pre-scales by √W to reuse unweighted Henderson algebra from LMM.
  Outer θ optimisation plugs a Laplace NLL into the existing
  `_reml_newton_solve()` — no new solver code.  Supports random
  intercepts/slopes, crossed designs, and arbitrary nesting.
  Diagnostics: AUC, deviance, variance components, ICC, random-effect
  covariance recovery.  Permutation via `method="score"` (one-step
  corrector) or `method="score_exact"` (PQL-fixed vmap).
- **`PoissonMixedFamily`** (`family="poisson_mixed"`): GLMM for
  clustered count data via Laplace approximation.  Same architecture
  as `LogisticMixedFamily` — only the conditional NLL and working
  response/weight functions differ.  Diagnostics: pseudo-R², deviance,
  overdispersion (Pearson dispersion), variance components, ICC.
- **Laplace solver** (`_build_laplace_nll` + `_laplace_solve` in
  `_backends/_jax.py`): generic two-loop Laplace engine.  Outer
  Newton via `_reml_newton_solve()`, inner IRLS unrolled for
  `jax.grad`/`jax.hessian` differentiability.  Returns `LaplaceResult`
  dataclass with β̂, û, W, μ̂, V⁻¹ diagonal, Fisher information.
- **`_fit_glm_irls()`** (`_backends/_jax.py`): robust pure-NumPy GLM
  solver with η-clipping, weight capping, Hessian modification, and
  step-halving line search.  Used as reliable initialisation for
  Laplace and as a standalone solver for GLMM diagnostics.
- **`_pql_fixed_irls_vmap()`** (`_backends/_jax.py`): pure JAX
  function running full IRLS at fixed Γ⁻¹ via `jax.vmap`.  Takes
  `(B, n)` permuted y, returns `(B, p)` β̂.  20 unrolled iterations,
  same Henderson algebra/η-clipping/weight-capping as
  `_build_laplace_nll`.
- **`ScoreExactStrategy`** (`method="score_exact"`) in
  `_strategies/score.py`: PQL-fixed exact permutation for GLMM
  families.  Validates family via `log_chol` attribute, rebuilds
  Γ⁻¹ from stored log-Cholesky params, single vmap call on full
  model, extracts slope coefficients with confounder masking.
  Rejects non-GLMM families with `ValueError`.
- **164 new GLMM tests** in `tests/test_families_mixed.py`: protocol
  conformance, Laplace accuracy vs R `glmer()` references, IRLS
  pre-scaling verification, calibration round-trip, residual
  properties, diagnostics completeness, display formatting,
  exchangeability cells, crossed designs, one-step corrector validity
  (KS uniformity), end-to-end integration with
  `permutation_test_regression()`, PQL-fixed smoke tests.
- **Four-stage confounder sieve** in `confounders.py`:
  `identify_confounders()` now runs screen → collider → mediator →
  moderator stages.  Returns `ConfounderAnalysisResult` frozen
  dataclass (backward-compatible via `to_dict()` and `[]` access).
  New parameters: `correlation_method`, `correction_method`, `groups`,
  `n_bootstrap_mediation`, `n_bootstrap_moderation`.
- **Collider detection** (`_collider_test()`): linear OLS t-test +
  Pearson r amplification comparison.  GLM families use
  `fam.fit()`/`fam.coefs()` with permutation-calibrated
  non-collapsibility guard (200 permutations, 95th percentile null).
  Multinomial returns `(False, NaN, NaN)`.
- **Moderation analysis** (`moderation_analysis()`): BCa bootstrap
  test for interaction X_c × Z_c.  Per-resample mean-centering,
  collinearity guard (rank < 3 → skip with warning),
  quasi-separation guard for GLM families, cluster bootstrap support.
- **Screening upgrades** in `screen_potential_confounders()`:
  - `correlation_method="partial"`: asymmetric partial correlation
    (Z-Y partials out X; Z-X keeps marginal Pearson r).
  - `correlation_method="distance"`: Székely & Rizzo bias-corrected
    distance correlation with asymptotic t-test.  O(n²) complexity
    warning when n > 10,000.
  - `correction_method="holm"` / `"fdr_bh"`: per-leg multiple-testing
    correction via `statsmodels.stats.multitest.multipletests`.
- **Cluster bootstrap** for mediation and moderation:
  `_cluster_bootstrap_indices()` resamples whole groups;
  `_cluster_jackknife_indices()` for BCa acceleration.  Activated
  via `groups=` parameter on `mediation_analysis()`,
  `moderation_analysis()`, and `identify_confounders()`.
- **Mixed-family fallback** (`_resolve_base_family()`): automatically
  strips `_mixed` suffix for mediation/moderation/collider analysis,
  reverting to the base family fit/coefs methods.
- **E-value sensitivity analysis** (`compute_e_value()` in
  `diagnostics.py`): family-dispatched conversion (linear → Cohen's d
  → RR, logistic/ordinal → OR via Cornfield inequality,
  poisson/negbin → direct RR, multinomial → NaN).  Optional
  `baseline_prevalence` for exact RR.  Mixed-family names stripped.
- **Rosenbaum bounds** (`rosenbaum_bounds()` in `diagnostics.py`):
  worst-case p-values under hidden bias Γ.  Linear-only, binary
  predictors only.  Rejects non-linear (NotImplementedError),
  linear_mixed (exact name check), continuous predictors (ValueError).
- **`ConfounderAnalysisResult`** frozen dataclass in `_results.py`:
  9 fields with `_DictAccessMixin` for backward compatibility.
- **Updated `print_confounder_table()`**: accepts
  `ConfounderAnalysisResult`, new `correlation_method` and
  `correction_method` parameters, shows colliders and moderators.
- **60 confounder tests** covering partial/distance correlation,
  multiple-testing correction, collider detection (linear + logistic
  + multinomial + permutation guard), moderation, mixed-family
  fallback, cluster bootstrap, collinearity guard, multinomial
  exclusion, full sieve orchestrator, E-value (10 tests), and
  Rosenbaum bounds (5 tests).
- **`panel_id=` and `time_id=` convenience parameters** on
  `permutation_test_regression()`: syntactic sugar for
  `groups=panel_id, permutation_strategy="within"`.  When
  `panel_id` is provided, permutations are automatically
  constrained to within-panel shuffling.  `time_id` enables
  panel-balance and sort-order validation warnings.  Conflicts
  with explicit `groups=` or `permutation_strategy=` raise
  `ValueError`.  Panel-level diagnostics (number of panels,
  observations per panel min/max/mean, balance flag) are added
  to `extended_diagnostics["panel_diagnostics"]`.
- **10 new tests** in `TestPanelData` covering balanced panels,
  diagnostics, equivalence to explicit groups, conflict detection,
  sort-order warnings, column-name resolution, and unbalanced
  panel warnings.

### Changed

- **Stabilization Phase 1: Cross-cutting helpers** — Extracted 9
  shared helpers replacing ~2,400 lines of near-identical boilerplate:
  `_suppress_sm_warnings` context manager (43 sites),
  `_augment_intercept` helper (51+ sites), `_extract_variance_components`
  (6 loops), `_format_variance_components` (3 loops),
  `_require_calibrated` (3 copies), `_validate_count_y` /
  `_validate_categorical_y` (4 bodies), VIF via matrix inverse
  (diagnostics.py), vectorized Clopper-Pearson CIs, vectorized BCa CIs.
- **Stabilization Phase 2: Batch dispatch dedup** — Added module-level
  `_dispatch_batch()` helper and `_backend_slug: ClassVar[str]` to all
  6 family classes.  Collapsed 30 batch methods (each 15–40 lines of
  repeated backend-resolution boilerplate) to 1–5-line wrappers.
  Net: ~490 lines removed from `families.py`.
- **NB2 gradient/Hessian → autodiff** — Replaced 60 lines of
  hand-coded NB2 gradient and Hessian in `_jax.py` with
  `jit(grad(_make_negbin_nll(alpha)))` and
  `jit(hessian(_make_negbin_nll(alpha)))`.  Added 2 cross-validation
  tests verifying autodiff matches finite-difference approximations.
- **Stabilization Phase 3: `_numpy.py` batch dispatch dedup** — Added
  two module-level generic helpers (`_batch_coefs` and
  `_batch_coefs_and_scores`) that factor out the sequential-vs-parallel
  loop/vstack scaffolding shared by all 29 non-vectorized batch methods
  in `NumpyBackend`.  Each method is now a thin wrapper: preamble →
  `_fit_one` closure → single `return _batch_coefs(...)` or
  `return _batch_coefs_and_scores(...)` call.  3 pure-BLAS vectorized
  methods (`batch_ols`, `batch_ols_fit_and_score`, `batch_mixed_lm`)
  remain untouched.  Net: ~228 lines removed from `_numpy.py`
  (1,839 → 1,611).
- **Stabilization Phase 4: GLMM stub & calibrate dedup** — Added
  `_GLMMBatchStubMixin` (5 `batch_*` `NotImplementedError` stubs) and
  `_calibrate_glmm()` shared Laplace-approximation calibration helper
  in `families_mixed.py`.  Both `LogisticMixedFamily` and
  `PoissonMixedFamily` now inherit the mixin and delegate `calibrate()`
  to the shared helper (parameterised by family name, working-response
  function, and conditional NLL).  Net: ~65 lines removed
  (2,630 → 2,565).
- **Confounder sieve batch optimisation (29× speedup)**:
  `_collider_test()`, `mediation_analysis()`, `_bca_ci()`, and
  `moderation_analysis()` in `confounders.py` now use
  `batch_fit_paired()` and `batch_fit_varying_X()` (JAX vmap) for
  bootstrap, jackknife, and permutation loops instead of sequential
  `fam.fit()` calls.  Ordinal confounder identification on the Wine
  Quality dataset (n=500, 5 predictors) drops from ~109s to ~3.8s.
  Falls back to the sequential loop when the family lacks batch
  methods or when cluster-bootstrap produces ragged index arrays.
- **Ordinal example `n_permutations` increased**: all methods in
  `examples/ordinal_regression.py` now use `n_permutations=999`
  (previously 199), providing tighter p-value resolution with
  negligible additional runtime via JAX vmap.
- **`_reml_newton_solve` LM-Nielsen damping cap**: `lambda_max`
  changed from `0.5 * spectral_norm` to `1e16 * spectral_norm`.
  The previous cap permanently trapped the solver when the Laplace
  NLL was highly non-quadratic (λ at cap, ρ < 0 every iteration,
  zero accepted steps).  Nielsen (1999) places no upper bound on λ;
  `1e16 ×` is a pure overflow guard that never constrains in
  practice.  No regression on REML (792 non-GLMM tests pass).
- **`np.where` eager-evaluation fix** in GLMM `fit_metric` methods:
  both `LogisticMixedFamily.fit_metric` and
  `PoissonMixedFamily.fit_metric` wrap `np.log` calls in
  `np.errstate(divide="ignore", invalid="ignore")` to suppress
  `log(0)` warnings from NumPy's eager evaluation of both branches
  in `np.where`.
- **`ScoreExactStrategy`** upgraded from Plan C placeholder
  (`NotImplementedError`) to full PQL-fixed implementation.
  Non-GLMM families now raise `ValueError` (not `NotImplementedError`).

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
- **`JaxBackend` batch dispatch refactor**: 30 GLM `batch_*` methods
  (logistic, Poisson, negative binomial, ordinal, multinomial, OLS ×
  5 patterns each) consolidated via 3 generic private helpers
  (`_batch_shared_X`, `_batch_varying_X`, `_batch_paired`) and
  3 static module-level helpers (`_augment_intercept_2d`,
  `_augment_intercept_3d`, `_strip_intercept`).  Each public method
  is now a thin one-liner delegate.  ~900 lines of duplicated
  boilerplate eliminated.  No public API change; compiled XLA
  identical.  Mixed-LM methods excluded (different solver architecture).
- **GLMM test fixture scoping**: 6 module-level fixtures in
  `test_families_mixed.py` re-scoped from `scope="function"` to
  `scope="module"`, avoiding redundant Laplace REML calibration
  per test (~10× speedup: 21 min → <2 min).

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

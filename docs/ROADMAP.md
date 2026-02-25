# Roadmap

This document outlines the planned development trajectory for
**randomization-tests**, from its current alpha state through a stable
1.0 release.  The overarching architectural goal is a **hypergraph
model specification layer** that unifies single-equation regression,
multi-equation path models, and group-level hypothesis testing under a
single typed graph structure, with exchangeability-aware permutations
dispatched automatically.  Each milestone builds toward this goal: the
`ModelFamily` protocol (v0.3.0) provides node-level equation solvers,
exchangeability cells (v0.4.0) formalise permutation constraints, the
graph specification layer (v0.5.0) ties them together, and the
structured result interface (v0.6.0) exposes the graph structure to
users.

Version numbers are indicative — scopes may shift as the project
evolves — but the ordering reflects deliberate dependency reasoning:
statistical machinery is finalised before the result interface is
locked down, so that each abstraction only needs to be designed once.

Versions v0.1.0 through v0.3.0 are complete. Their detailed
checklists have been retired from this document. Key design
decisions that carry forward:

### Design note — `calibrate()` pattern

**v0.4.0 implications:**

- `calibrate()` becomes the natural hook for
  `PermutationEngine.__init__`: the engine calls it once during
  construction, yielding a fully-resolved family for the loop.
- `calibrate()` is orthogonal to `exchangeability_cells()` — they
  solve different problems (nuisance estimation vs. permutation
  constraints) and compose independently.
- Future families with nuisance parameters (e.g. zero-inflated
  models with π, Gamma with shape k) adopt the same pattern
  without protocol changes.

---

## v0.4.0 — Exchangeability & Multilevel Frameworks

Formalises the permutation structure: which observations are
exchangeable with which, under what constraints.  These
exchangeability cells are the permutation-side complement to the
model-side `ModelFamily` protocol — together they define a complete
permutation test specification for any single-equation model.

**Forward-compatibility note:** the `exchangeability_cells()` method
was added to the `ModelFamily` protocol in v0.3.0 Step 6b
(stabilisation) as a no-op stub returning `None` (global
exchangeability) on all six families (linear, logistic, Poisson,
negative binomial, ordinal, multinomial).  This release fills it in
with real cell structures.  `calibrate()` is now a formal
`ModelFamily` protocol method (Step 14a) called unconditionally by
`PermutationEngine.__init__`; the former `hasattr` guard has been
removed.  All six concrete families are sealed with `@final`.
The duck-typed `model_fit_metric()` and `null_fit_metric()`
on ordinal and multinomial families are replaced by new `score()`
and `null_score()` protocol methods, eliminating the `hasattr`
branching in Kennedy joint.

### `fit_reduced()` in `families.py` + `score()` / `null_score()` on protocol

Deferred from v0.3.0 audit.  Three related problems resolved together:

1. **Reduced-model fitting duplication** — four strategies repeat the
   same “fit Z or fall back to intercept-only predictions” block.
2. **Duck-typed metric branching** — Kennedy joint uses
   `hasattr(family, "model_fit_metric")` to branch between
   prediction-based families and model-object families (ordinal,
   multinomial).  Fragile and forces every joint strategy to know
   about both API surfaces.
3. **Deferred null-metric assignment** — when Kennedy joint has no
   confounders and the family is ordinal/multinomial, it defers
   `base_metric` until after the full model is fitted, then pulls
   `model.llnull`.  Hard to follow.

**Architecture decisions:**

- **`fit_reduced()` is a module-level function in `families.py`**
  alongside `resolve_family()`.  It operates entirely on the
  `ModelFamily` interface (`family.fit()` + `family.predict()`)
  with a zero-column fallback — it is not family-specific, so
  it cannot go on the protocol.  Placing it next to `fit()` and
  `resolve_family()` co-locates all family-interface utilities
  in one module, maintains a consistent pattern (module-level
  functions that delegate to the protocol), and keeps the
  dependency direction correct (strategies import from families,
  never the reverse).  The v0.5.0 graph compiler and any new
  callers can import it from `families` without reaching into
  internal packages.
- **`score(model, X, y) -> float`** is a new `ModelFamily` protocol
  method.  Prediction-based families delegate to
  `self.fit_metric(y, self.predict(model, X))`.  Model-object
  families (ordinal, multinomial) return `−2 × model.llf`.  This
  eliminates all `hasattr` / `_uses_model_metric` branching.
- **`null_score(y, fit_intercept) -> float`** is a new `ModelFamily`
  protocol method (Option 1 — safe and explicit).  Prediction-based
  families compute `fit_metric(y, mean(y))` or
  `fit_metric(y, zeros)`.  Model-object families fit a
  thresholds/intercept-only model and return `−2 × llf`.  This
  replaces the `null_fit_metric()` duck-typed method.

- [X] Add `fit_reduced()` module-level function in `families.py`
  (alongside `resolve_family()`) returning
  `(model | None, predictions)`.
- [X] Add `score(model, X, y)` to the `ModelFamily` protocol and
  implement on all 6 families.
- [X] Add `null_score(y, fit_intercept)` to the `ModelFamily` protocol
  and implement on all 6 families.
- [X] Refactor all four strategy files to use `fit_reduced()` and
  `family.score()` / `family.null_score()`.
- [X] Remove duck-typed `model_fit_metric()` and `null_fit_metric()`
  from `OrdinalFamily` and `MultinomialFamily` (subsumed by
  `score()` and `null_score()`).
- [ ] Performance: the reduced model is fitted once and cached,
  avoiding redundant refits when the same reduced model applies to
  multiple exposure variables.

### `batch_fit_paired` — confounder bootstrap/jackknife vectorisation

The `mediation_analysis()` bootstrap loop (1,000 iterations) and the
`_bca_ci()` jackknife loop (*n* iterations) call `family.fit()`
sequentially for every replicate.  For non-linear families (ordinal,
multinomial, negative binomial) with large *n*, this is the dominant
bottleneck — e.g. ordinal with *n*=500 and 5 candidate confounders
requires ~7,500 individual BFGS fits.

A new `batch_fit_paired(X_batch, Y_batch, fit_intercept, **kwargs)`
method where **both** X and Y vary per replicate (unlike
`batch_fit` which holds Y fixed) provides the exactly the semantics
needed for bootstrap (resampled rows affect both X and Y) and
jackknife (leave-one-out affects both).

- [X] Add 6 `batch_*_paired` methods to `_backends/_jax.py` using
  `vmap(_solve_one, in_axes=(0, 0))` over both X and Y.
- [X] Add 6 `batch_*_paired` fallback methods to `_backends/_numpy.py`
  with sequential loop and `n_jobs` support.
- [X] Add `batch_fit_paired()` to the `ModelFamily` protocol and
  implement on all 6 families.
- [X] Refactor `confounders.py` bootstrap loop: pre-build
  `XM_boot = xm_full[boot_idx]` (B, n, 2) and
  `Y_boot = y_values[boot_idx]` (B, n), single call to
  `fam.batch_fit_paired()`.
- [X] Refactor `confounders.py` jackknife loop: build leave-one-out
  index array (n, n−1), same pattern.
- [X] Fix 15 solver return-type annotations in `_jax.py` (5 solver
  functions + 10 `_solve_one` inner functions) from 2-tuple to
  3-tuple `(beta, nll, converged)`.
- [X] 6 new tests in `TestBatchFitPaired` verifying shape,
  finiteness, and correctness for all families.

### `batch_fit_and_score` — Kennedy joint vectorisation

The Kennedy joint strategy needs to fit a full model per permutation
and compute a deviance score.  `batch_fit_and_score()` and
`batch_fit_and_score_varying_X()` combine fitting and scoring in a
single vectorised call, avoiding the overhead of separate
`fit()` + `score()` per permutation.

- [X] Add `batch_fit_and_score()` to backends, protocol, and all 6
  families (fixed X, permuted Y).
- [X] Add `batch_fit_and_score_varying_X()` to backends, protocol,
  and all 6 families (both X and Y vary per permutation).
- [X] Refactor Kennedy joint strategy to use
  `batch_fit_and_score_varying_X()`.
- [X] Refactor Freedman-Lane joint strategy to use
  `batch_fit_and_score()`.

### `backend=` parameter on `PermutationEngine`

- [X] Add `backend=` parameter to `PermutationEngine` constructor
  and `permutation_test_regression()`.  When supplied, skip
  auto-resolution via `resolve_backend()`.

### Display refactor — family-driven formatting

The current `display.py` uses `if/elif` chains on `model_type`
strings to format headers and diagnostics tables.  This scales
poorly — each new family adds branches in three locations
(`print_results_table`, `print_joint_results_table`, diagnostics
rendering).  v0.4.0 moves formatting responsibility into the
`ModelFamily` protocol via a clean break — no fallback paths,
no deprecation warnings, all `model_type` branching deleted:

- [X] `ModelFamily.display_header(diagnostics: dict) -> list[tuple[str, str, str, str]]` — returns structured row
  descriptors for the results table header.  Each 4-tuple is
  `(left_label, left_value, right_label, right_value)`.  Families
  own content and value formatting; `display.py` owns all layout.
- [X] `ModelFamily.display_diagnostics(diagnostics: dict) -> tuple[list[tuple[str, str]], list[str]]` — returns `(lines, notes)`:
  label/value pairs for model-level diagnostics and warning strings
  for the Notes section.  Bundling notes with lines avoids residual
  `model_type` branching in `display.py`.
- [X] `ModelFamily.stat_label: str` property — `"t"`, `"z"`, or
  `"χ²"` per family.  Used in result table column headers.
- [X] `display.py` becomes family-agnostic: `family: ModelFamily` is
  a **required** parameter on `print_results_table()`,
  `print_joint_results_table()`, and `print_diagnostics_table()`.
  All `model_type` branch blocks are deleted.
- [X] Unblocks user-defined families — a custom `ModelFamily` can
  control its own display output without modifying `display.py`.

### Diagnostics refactor — family-driven computation

The same `model_type` branching pattern in `diagnostics.py`
(`compute_all_diagnostics()` L950–1117) is eliminated by moving
family-specific diagnostic computation into the protocol:

- [X] `ModelFamily.compute_extended_diagnostics(X, y, fit_intercept) -> dict[str, Any]` — returns family-specific diagnostic key–value
  pairs.  The dict keys must match what `display_diagnostics()` reads
  (contract locked in by the display refactor).
- [X] `compute_all_diagnostics()`: `family: ModelFamily` replaces
  `model_type: str`.  The entire `model_type` branch block is
  deleted and replaced by
  `result.update(family.compute_extended_diagnostics(...))`.
- [X] Generic diagnostics (VIF, standardized coefs, Monte Carlo SE,
  divergence flags, Cook's distance, permutation coverage) remain in
  `diagnostics.py` — they are family-agnostic.

### `model_type` removal from result objects + self-contained display

- [X] `model_type` field deleted from `IndividualTestResult` and
  `JointTestResult` dataclass definitions.  No property alias, no
  `to_dict()` key — zero `model_type` anywhere in the codebase.
- [X] `family: str` field changed to `family: ModelFamily` instance
  on both result classes.  All consumers use the instance directly
  (`results.family.stat_label`, `results.family.display_header()`,
  etc.).
- [X] `_SERIALIZERS: ClassVar[dict[str, Any]]` registry on
  `_DictAccessMixin`.  `to_dict()` consults the registry to convert
  non-primitive fields (e.g. `family` → `family.name`), then always
  runs `_numpy_to_python()` on the result (composing, not exclusive).
  Prepares infrastructure for v0.5.0 `GraphTestResult`.
- [X] Self-contained display functions: `print_results_table(results, *, title=...)`, `print_joint_results_table(results, *, title=...)`,
  `print_diagnostics_table(results, *, title=...)`.  All metadata
  (`family`, `feature_names`, `target_name`) extracted from the
  result object internally — no parameter passing.
- [X] `compute_standardized_coefs` and `compute_cooks_distance` take
  `family: ModelFamily` (not `model_type: str`).
- [X] `print_confounder_table`: `family: ModelFamily | None = None`
  (drop `str` from union type).
- [X] New fields added to result objects: `feature_names: list[str]`,
  `target_name: str`, `n_permutations: int`,
  `groups: np.ndarray | None` (populated in v0.4.1),
  `permutation_strategy: str | None` (populated in v0.4.1).

### Sign-flip test

Deferred from v0.3.0 (Step 8).  Depends on Step 1
(`family.residuals()`, `family.reconstruct_y()`).

Sign-flipping rests on the assumption of symmetric error
distributions, which is strictly weaker than exchangeability but
applies only to paired or within-subject designs where the
difference scores are symmetric about zero under the null.  Because
the assumption domain differs from that of permutation tests —
symmetry governs within-unit comparisons, exchangeability governs
between-unit comparisons — sign-flipping is exposed as a **separate
public entry point** rather than as a `method` on
`permutation_test_regression()`.

- [ ] `sign_flip_test_regression()`: separate public function with
  input validation requiring paired structure (two-column response or
  pre-computed difference vector).
- [ ] Resampling module `sign_flips.py` (parallel to `permutations.py`)
  generating the 2ⁿ reference distribution of sign-flip assignments.
- [ ] Integrates with `ModelFamily`: sign-flip the residuals returned
  by `family.residuals()`, reconstruct via `family.reconstruct_y()`.
  No family-specific code needed.
- [ ] Documentation clearly distinguishes the symmetry assumption from
  the exchangeability assumption, including when each is appropriate
  and when each is violated.
- [ ] Standard in neuroimaging (FSL PALM supports both permutation
  and sign-flip).

### Exchangeability cells

- [X] New `groups=` parameter accepting a column name or array that
  identifies clusters (schools, firms, regions, panel units, etc.).
- [X] `permutation_strategy=` argument controlling the exchangeability
  structure:
  - `"within"` — permute observations within each cluster, holding
    cluster membership fixed.  Appropriate when the research question
    concerns within-cluster variation and cluster-level confounding is
    controlled by design.
  - `"between"` — permute entire clusters as units.  Appropriate when
    the treatment is assigned at the cluster level.
  - `"two-stage"` — permute clusters first, then permute observations
    within clusters.  Suitable for crossed or partially nested designs.
- [X] Generalisation to arbitrary blocking factors: permutations occur
  only within strata defined by one or more categorical variables.
- [X] User-supplied permutation constraints via a callback for
  non-standard exchangeability structures.
- [X] Validation logic to ensure the requested strategy is compatible
  with the data (e.g., `"between"` requires sufficient clusters for
  meaningful permutation; raises `ValueError` when all cells have
  unique sizes with guidance to use `"within"` or `"two-stage"`;
  warns when fewer than 100 between-cell permutations are available).

### Mixed-effects models

Depends on exchangeability cells (permutation must respect the
nesting structure that the random effects encode).

- [ ] Integration with statsmodels `MixedLM` (linear) and a suitable
  backend for generalised linear mixed models (GLMM) as new
  `ModelFamily` implementations.
- [ ] The permutation target becomes the conditional residual (BLUP
  residual) under the reduced model, with random-effect structure held
  fixed across permutations.
- [ ] Diagnostics: marginal and conditional R², random-effect variance
  components, ICC.

### Longitudinal / panel data

Depends on exchangeability cells (within-panel permutation is a
special case of within-group exchangeability).

- [ ] Within-panel permutation as the default strategy when a time or
  wave variable is provided, preserving the temporal dependency
  structure within each unit.
- [ ] Support for both balanced and unbalanced panels.
- [ ] Optional autoregressive residual structure for the reduced model.

### Inferential improvements

- [ ] Exact binomial or Clopper-Pearson confidence intervals on
  permutation p-values reflecting Monte Carlo uncertainty when only a
  finite number of permutations are drawn.
- [ ] Adaptive stopping: optionally halt the permutation loop early
  once the CI for the p-value is narrow enough to determine
  significance with a specified confidence.
- [ ] Conditional Monte Carlo: permute within the sufficient-statistic
  strata of a nuisance parameter for exact conditional tests.

### Compatibility validation module

Consolidates the distributed method-incompatibility checks scattered
across `core.py` and `engine.py` into a single `_validation.py`
module with a structured compatibility matrix.  This is the last
v0.4.x item — it lands after mixed-effects (new constraint
dimensions), sign-flip (new family/method incompatibilities), and
inferential improvements have expanded the constraint surface to the
point where centralisation pays for itself.  The v0.5.0 graph
compiler consumes this module directly for per-equation validation.

- [ ] New `_validation.py` module containing a compatibility matrix
  that maps `(family, method, strategy, data_shape)` tuples to
  outcomes: proceed, warn with guidance, or raise with guidance.
- [ ] Structured guidance messages with a consistent format across all
  incompatibilities: what was requested, why it is incompatible, and
  which alternatives are valid.  Messages reference the correct
  `permutation_strategy` or `method` parameter names so users can
  act on them directly.
- [ ] Refactor existing distributed checks to delegate to the
  compatibility matrix:
  - Between-cell infeasibility (`ValueError` / `UserWarning` in
    `core.py` `_resolve_groups_and_strategy()`)
  - Freedman–Lane rejection for ordinal/multinomial (`ValueError`
    in `engine.py`)
  - Kennedy without confounders (`UserWarning` in `engine.py` +
    Notes section in display)
  - Freedman–Lane without confounders (`UserWarning` — reduces to
    ter Braak)
  - `n_jobs != 1` with JAX backend (`UserWarning` in `core.py`)
  - `n_jobs != 1` with vectorised linear OLS (`UserWarning` in
    `core.py`)
  - Sign-flip rejection for direct-permutation families
    (`ValueError`)
  - Mixed-effects family + incompatible permutation strategy
- [ ] `validate_compatibility(family, method, strategy, groups, data_shape) -> list[ValidationIssue]` public function returning a
  list of typed issue objects (`ValidationIssue(level, code, message, suggestion)`).  `level` is `"error"` or `"warning"`.
  The engine calls this once during construction; the graph compiler
  calls it per equation during compilation.
- [ ] Forward-compatibility hook: the `ValidationIssue.code` field
  (e.g. `"BETWEEN_INFEASIBLE"`, `"FREEDMAN_LANE_NO_RESIDUALS"`,
  `"SIGN_FLIP_DIRECT_PERMUTATION"`) enables the v0.5.0 graph
  compiler to programmatically inspect and handle issues rather
  than catching exceptions.
- [ ] Tests: one test per existing check verifying it still fires
  through the new centralised path, plus tests for the new
  mixed-effects and sign-flip constraint combinations.

---

## v0.5.0 — Graph Specification & Multi-Equation Orchestration

The architectural centrepiece: a typed hypergraph data structure that
lets users declare multi-equation models and have the package
automatically derive which equations to fit, which families to use,
which permutation strategies to apply, and which null hypotheses to
test.  A standard regression is a single-layer graph with all
predictors pointing at one outcome; mediation, path models, and
multi-equation systems are deeper graphs composed of the same
node-level equation solvers built in v0.3.0, constrained by the
exchangeability cells built in v0.4.0.

### Specification data structure

- [ ] A `CausalGraph` class supporting:
  - **Nodes** — observed variables, each optionally annotated with a
    model family and role (exposure, outcome, mediator, confounder).
  - **Directed edges** — pairwise causal claims (X → Y), each
    testable individually via per-coefficient permutation tests.
  - **Hyperedges** — group-level causal claims ({X₁, X₂} → Y as an
    irreducible unit), testable via generalised joint tests.
  - **Exchangeability cells** — optional per-node or global
    permutation constraints inherited from the v0.4.0 cell system.
- [ ] Validation: acyclicity check, connected-component analysis,
  family compatibility per equation, identification of
  under-determined nodes.

### Graph compiler

- [ ] Topological sort of the DAG to determine equation fitting order.
- [ ] For each outcome node, derive the structural equation:
  `(outcome, predictors, family, permutation_strategy, null_type)`.
- [ ] Resolve families automatically (from outcome type) or from
  per-node annotation.
- [ ] Map hyperedges to Kennedy joint tests; map simple edges to
  ter Braak or Kennedy individual tests based on the presence of
  declared confounders.

### Multi-equation orchestrator

- [ ] Execute permutation tests for each structural equation in
  topological order, dispatching to the `ModelFamily` protocol.
- [ ] Independent permutation per equation by default: each equation
  is tested in isolation, conditioning on observed values of upstream
  nodes.
- [ ] Optional propagated mode: permute upstream, refit downstream,
  enabling permutation-based indirect-effect testing through
  multi-step paths.
- [ ] Collect per-equation results into a unified graph-level result
  object.

### Hyperedge testing

- [ ] Generalise the Kennedy joint test to arbitrary hyperedges
  declared in the specification.  A hyperedge {X₁, X₂, X₃} → Y
  triggers joint row-wise permutation of exposure residuals for
  X₁, X₂, X₃ simultaneously, with the test statistic being the
  improvement in the family's fit metric.
- [ ] Support hyperedges targeting different outcome nodes within the
  same graph.
- [ ] Support mixed-family hyperedges (e.g., Poisson outcome with
  linear exposure models).

### Indirect effect extraction

- [ ] For a declared path X → M → Y, compute the product of per-edge
  coefficients (a × b) and test via permutation.
- [ ] Extend the existing BCa bootstrap framework to support
  permutation-based indirect effect p-values for paths declared in
  the graph specification.
- [ ] Support arbitrary-length causal chains (X → M₁ → M₂ → Y) with
  product-of-coefficients test statistics.

### Model specification syntax

- [ ] Primary API: Python method calls —
  `g.add_node("Y", family="linear")`,
  `g.add_edge("X1", "Y")`,
  `g.add_hyperedge(["X1", "X2"], "Y")`.
- [ ] Convenience: formula-style string parser —
  `"Y ~ X1 + X2; M ~ X1; Y ~ M"` with equations parsed and DAG
  inferred.
- [ ] Convenience: dictionary specification for programmatic and
  configuration-file workflows.

### Interoperability

- [ ] The `CausalGraph` internal representation should use a directed
  incidence matrix as its canonical form (sparse, with +1/−1 entries
  for head/tail of each hyperedge), enabling natural conversion to
  and from external hypergraph libraries.
- [ ] Node attribute and hyperedge attribute dictionaries for metadata
  (family, role, exchangeability cell, test results) that survive
  round-trip conversion.
- [ ] Conversion utilities in a `compat` module:
  - `to_hypernetx()` / `from_hypernetx()` — HyperNetX `Hypergraph`
    objects.
  - `to_toponetx()` / `from_toponetx()` — TopoNetX
    `CombinatorialComplex` objects.
  - `to_networkx()` / `from_networkx()` — NetworkX `DiGraph` objects
    (hyperedges expanded to bipartite auxiliary nodes).
- [ ] This ensures the package can participate in broader hypergraph
  analysis pipelines as a permutation-testing module.

---

## v0.6.0 — Structured Results & Dual API

With the graph specification and all statistical machinery finalised,
this release extends the result interface for graph-structured models
and adds academic output formats.

**Note:** The core result dataclasses (`IndividualTestResult`,
`JointTestResult`) and their dict-like access layer
(`_DictAccessMixin`, `.to_dict()`) were pulled forward into v0.3.0
Step 6b (stabilisation).  v0.4.0 Step 6 enriched these with
`family: ModelFamily` instances, a `_SERIALIZERS` registry,
self-contained display functions, and new metadata fields.
This release builds on that foundation.

### Extended result types

- [X] `IndividualTestResult` — per-coefficient tests (pulled forward
  to v0.3.0 Step 6b).
- [X] `JointTestResult` — group-level improvement tests (pulled
  forward to v0.3.0 Step 6b).
- [X] `.to_dict()` with full JSON serialisability (pulled forward).
- [ ] `GraphTestResult` — multi-equation results from the graph
  specification layer (v0.5.0), containing per-equation results,
  per-edge p-values, per-hyperedge p-values, and per-path indirect
  effects.
- [ ] Direct attribute access for graph results: `.equations`,
  `.edge_p_values`, `.hyperedge_p_values`, `.indirect_effects`.

### Academic output formats

- [ ] `.summary()` prints the ASCII table (current behaviour,
  already functional via display functions).
- [ ] `.to_latex()` produces a publication-ready LaTeX table.
- [ ] `.to_html()` renders in Jupyter notebooks.
- [ ] `.to_markdown()` generates a Markdown table for inclusion in
  reports or documentation.

### Display decoupling

**Pulled forward to v0.4.0** ("`model_type` removal from result
objects + self-contained display" above).  Display functions now
accept typed result objects with `family: ModelFamily` and extract
all metadata internally.  No remaining work here.

- [X] `print_results_table()` and `print_joint_results_table()` accept
  `IndividualTestResult` / `JointTestResult` natively with
  keyword-only `title` parameter.  All context (`family`,
  `feature_names`, `target_name`) extracted from the result object.
- [X] Type annotations on all display function signatures updated to
  reflect the typed inputs.
- [X] Internal string-key assumptions eliminated — display functions
  use typed attribute access exclusively.

### Programmatic access

- [ ] `.to_dataframe()` returns a tidy pandas DataFrame of
  coefficients, standard errors, and p-values.
- [ ] Graph-level summary: `.graph_summary()` showing per-edge and
  per-path results in a single view.

### Scikit-learn estimator interface

- [ ] `PermutationTestRegressor` and `PermutationTestClassifier`
  wrappers conforming to the scikit-learn estimator contract: `fit`,
  `predict`, `get_params`, `set_params`, `score`.
- [ ] Compatible with `Pipeline`, `GridSearchCV`, `cross_val_score`,
  and other scikit-learn meta-estimators.
- [ ] For graph specifications, a `GraphPermutationTest` estimator
  that accepts a `CausalGraph` and exposes results via `.results_`.
- [ ] The `fit` method runs the permutation test; results are
  accessible via the `.results_` attribute.  `predict` delegates to
  the underlying regression model.

### Flexible input handling

- [ ] Feature names inferred from DataFrame columns or supplied
  explicitly.
- [ ] numpy array, pandas, and Polars inputs accepted uniformly
  across all API surfaces (single-equation and graph).

---

## v0.7.0 — Causal Discovery & Visualisation

Adds methods for data-driven discovery of graph structure, feeding
directly into the graph specification layer, plus visualisation
utilities for permutation distributions and graph topology.

### Causal screening

- [ ] Supplement the existing Preacher & Hayes mediation analysis with
  more robust approaches for distinguishing confounders from mediators:
  - **LiNGAM** (Shimizu et al., 2006) — exploits non-Gaussianity to
    identify causal direction in linear models.
  - **Additive noise models (ANM)** — nonparametric causal direction
    test based on independence of residuals.
  - **PC algorithm** — constraint-based causal discovery from
    conditional independence tests.
- [ ] Output a `CausalGraph` specification directly from discovery,
  enabling a **discover → specify → test** pipeline.
- [ ] Document the assumptions and limitations of each approach clearly
  in the API reference.

### Visualisation utilities

- [ ] Permutation distribution histograms with observed test statistic
  annotated.
- [ ] Coefficient forest plots comparing observed vs. permutation null
  distributions across predictors.
- [ ] Graph topology visualisation: render the `CausalGraph` with
  edges coloured and weighted by permutation p-values.
- [ ] Option to return matplotlib Figure/Axes objects for further
  customisation, or render inline in Jupyter notebooks.

---

## v1.0.0 — Stable Release & Ecosystem

Marks the first stable public API with semantic versioning guarantees.
The graph specification layer, `ModelFamily` protocol, exchangeability
cell system, and structured result objects are all frozen.

### API freeze

- [ ] All public function signatures, graph specification methods,
  result object attributes, and parameter names are frozen.  Breaking
  changes after this point require a major version bump.
- [ ] Comprehensive deprecation policy for any future interface changes.

### PyPI publication

- [ ] Publish to PyPI so the package is installable via
  `pip install randomization-tests`.
- [ ] Automated release workflow in GitHub Actions: tag a version, build
  the sdist/wheel, upload to PyPI.

### Documentation site

- [ ] Sphinx or MkDocs documentation hosted on ReadTheDocs or GitHub
  Pages.
- [ ] Auto-generated API reference from docstrings.
- [ ] Tutorials: single-equation quickstart, multi-equation path model,
  exchangeability cells, causal discovery → testing pipeline.
- [ ] Gallery of worked examples.

### Benchmarks

- [ ] Runtime benchmarks across *n*, *n_permutations*, *n_features*,
  and *n_equations* for each model family.
- [ ] Comparison against naive (non-vectorised) implementations to
  quantify the performance gains from batch algebra and JAX.
- [ ] Published benchmark results in the documentation.

### GPU acceleration

- [ ] Evaluate CuPy as an additional backend by implementing
  `_backends/_cupy.py` with `BackendProtocol` — all model families
  gain GPU support automatically through the existing backend
  dispatch system established in v0.3.0.
- [ ] Document hardware requirements and expected speedups.
- [ ] Published GPU-vs-CPU benchmark results in the documentation.

### Community

- [ ] Issue and pull request templates.
- [ ] GitHub Discussions board for questions and feature requests.
- [ ] Citation file (`CITATION.cff`) for academic use.

---

## Cross-cutting concerns *(ongoing)*

These items are not tied to any single release and are maintained
continuously.

- [ ] **CI matrix:** keep the test matrix current as new Python versions
  are released (3.14+, etc.).
- [ ] **Dependency floors:** periodically review and update minimum
  dependency versions.
- [ ] **Security:** monitor dependencies for vulnerabilities via
  Dependabot or similar tooling.
- [ ] **Documentation:** keep the API reference, quickstart guide, and
  changelog in sync with every release.
- [ ] **Property-based tests:** add Hypothesis-based property tests for
  core invariants (p-values in [0, 1], permuted arrays are true
  permutations, result schema completeness, determinism under fixed
  seed, commutativity of confounder ordering).  Not tied to a
  specific release — expand incrementally as new families and methods
  are added.

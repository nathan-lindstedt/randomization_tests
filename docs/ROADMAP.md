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

---

## v0.1.0 — Foundation

The initial packaged release.  Establishes the src-layout structure,
core algorithms, and development infrastructure.

- [X] **Permutation methods:** ter Braak (1992) residual permutation,
  Kennedy (1995) individual exposure-residual permutation, and Kennedy
  (1995) joint F-style test.
- [X] **Vectorised OLS** via batch pseudoinverse multiplication — the
  design matrix is inverted once and applied to all permuted response
  vectors simultaneously.
- [X] **Optional JAX backend** for logistic regression: `jax.vmap` over a
  custom Newton-Raphson solver with `jax.grad`-computed gradients.
  Transparent fallback to scikit-learn when JAX is not installed.
- [X] **Unique permutation pre-generation** with hash-based deduplication
  for large *n* and exhaustive enumeration for small *n* (≤ 12).
- [X] **Phipson & Smyth (2010) corrected p-values** that are guaranteed
  never to be exactly zero.
- [X] **Confounder identification pipeline:** Pearson correlation screening
  followed by Preacher & Hayes (2004, 2008) mediation analysis with
  bias-corrected and accelerated (BCa) bootstrap confidence intervals
  (Efron, 1987).  The indirect effect (a × b) is tested directly via
  5 000 bootstrap resamples; the jackknife-based acceleration
  correction accounts for skewness in the sampling distribution.
- [X] **Formatted ASCII table output** modeled after statsmodels, with
  separate layouts for individual-coefficient and joint tests.
- [X] **Development infrastructure:** Google-style docstrings, PEP 561
  `py.typed` marker, modern Python 3.10+ type annotations, GitHub
  Actions CI (ruff, mypy, pytest across Python 3.10–3.13), and a test
  suite of 40 tests.

---

## v0.1.1 — Polish

- [X] **Polars input support:** all public functions accept both pandas
  and Polars DataFrames via an internal `_ensure_pandas_df` adapter in
  `_compat.py`.  A `DataFrameLike` type alias documents the supported
  input types.

---

## v0.1.5 — Diagnostics & intercept fix

Adds extended diagnostics, fixes a correctness bug in all permutation
refit paths, and expands the test suite.

- [X] **Intercept mismatch fix (all methods).** Permutation refits in
  `_batch_ols_coefs`, `_ter_braak_linear`, `_ter_braak_logistic`,
  `_kennedy_individual_linear`, `_kennedy_individual_logistic`, and
  `_kennedy_joint` were fitting without an intercept while the observed
  model used `fit_intercept=True`.  This caused permuted slope
  coefficients to absorb the response mean, producing spurious p-values
  (e.g. p = 1.0 for X6 longitude in the linear example, p = 0.0 for
  strong predictors).  All permutation refits now include an intercept
  column matching the observed model specification.
- [X] **Extended diagnostics module** (`diagnostics.py`): standardised
  coefficients, variance inflation factors (VIF), Monte Carlo standard
  error, empirical-vs-asymptotic divergence flags, Breusch-Pagan
  heteroscedasticity test (linear), deviance residuals and runs test
  (logistic), Cook's distance influence counts, and permutation
  coverage reporting.
- [X] **Exposure R² column** in the diagnostics table for the Kennedy
  individual method, quantifying how much of each predictor's variance
  is explained by the confounders.  Suppressed when no confounders are
  specified (was displaying a wall of `0.0000` values).
- [X] **`fit_intercept` parameter** added to all code paths (10
  functions in `core.py` and `pvalues.py`), enabling intercept-free
  models for through-origin regression.
- [X] **Confounder display table** (`print_confounder_table`): formatted
  80-character ASCII table for confounder identification results,
  replacing raw `print()` output.  Supports single- and multi-predictor
  inputs, parameter header, clean-predictor summary, and conditional
  mediator warning notes.
- [X] **Display improvements:** 80-character line-width constraint via
  `_wrap()` helper, `textwrap.wrap`-based title centering, consistent
  4-decimal p-value formatting (`"0.000"` / `"1.000"`), redesigned
  four-section diagnostics table layout, omnibus test footer in joint
  results table.
- [X] **P-value formatting fix:** `_fmt()` now uses fixed-width
  `f"{val:.{precision}f}"` instead of `f"{rounded}"`, preventing
  misleading `0.0` / `1.0` display.
- [X] **`calculate_p_values` return type** expanded from 2-tuple to
  4-tuple `(permuted_str, classic_str, raw_empirical, raw_classic)`,
  eliminating redundant statsmodels refits in diagnostics.
- [X] **Cook's distance fix:** logistic Cook's D now delegates to
  `sm.GLM(family=Binomial()).get_influence().cooks_distance` instead of
  using `sm.Logit.fittedvalues` (which returns log-odds, not
  probabilities).
- [X] **Test suite expanded** from 40 to 125 tests covering diagnostics,
  fit_intercept, exposure R², Polars compatibility, and display
  formatting.

---

## v0.2.0 — Hardening

Focuses on reliability, code quality enforcement, and performance of
the existing feature set before any new statistical capabilities are
added.

### CI & code quality

- [X] Achieve a fully clean ruff and mypy pass with zero warnings or
  suppressed ignores.
- [X] Add a pre-commit configuration (ruff lint + format, mypy,
  trailing-whitespace, end-of-file-fixer) so contributors catch issues
  before pushing.

### Test coverage

- [X] Expand the test suite with edge-case coverage: empty DataFrames,
  single-feature models, constant columns, perfect separation in
  logistic regression, and permutation requests that approach or exceed
  the available unique permutation count.
- [X] Add convergence-failure tests for the JAX Newton-Raphson solver
  (ill-conditioned Hessians, rank-deficient designs).
- [X] Add large-*n* smoke tests (e.g., *n* = 10,000, modest
  *n_permutations*) to verify memory footprint and runtime stay within
  reasonable bounds.

### CI & branching

- [X] Retarget all CI triggers from `main` to `experimental`.  The `main`
  branch is the stable instructional default; `experimental` is the
  active development branch.  Version branches (e.g. `v0.3.0`) branch
  off `experimental` and merge back into it when complete.
- [X] Document the branch strategy in `CONTRIBUTING.md`.

### Dependency management

- [X] Establish a tested compatibility matrix of lower- and upper-bound
  dependency versions (numpy, pandas, scipy, statsmodels, scikit-learn)
  and verify in CI.
- [X] Pin or document JAX version compatibility for the optional backend.

### Performance

- [X] Profile the hash-based permutation deduplication to identify any
  bottlenecks when *n_permutations* is large relative to *n*!.
  Published analysis: `docs/permutation-dedup-performance.md`.
- [X] Extract JAX-specific code from `core.py` into a standalone
  `_jax.py` module (pure separation of concerns — no protocol or
  backend abstraction yet).  This keeps `core.py` cleaner and
  provides a clean extraction point for v0.3.0 when `_jax.py` becomes
  `_backends/_jax.py`.

---

## v0.3.0 — GLM Family Extensions

Introduces a `ModelFamily` strategy pattern that decouples model
fitting from the permutation engine, then implements new permutation
methods and GLM families on top of it.  The protocol defines how each
family fits a model, extracts residuals, reconstructs permuted
outcomes, and computes diagnostics — making the core engine
family-agnostic and extensible to future model types including
mixed-effects and multi-equation specifications.

The ordering within this milestone is deliberate: the abstraction
layer (`ModelFamily`, `_backends/`) must exist before any code that
depends on it.  The core refactor converts existing linear and
logistic paths to the new protocol.  New permutation methods and GLM
families are then built on the stabilised abstractions, inheriting
all existing methods from the start.

**Progress:** Steps 1–5 complete (abstraction layer, core refactor,
JAX improvements, parallelisation).  278 tests passing.  Remaining:
confounder module update, Freedman–Lane, new GLM families (Poisson,
negative binomial, ordinal, multinomial), and sign-flip test.

### Step 1 — `ModelFamily` protocol

- [X] A `typing.Protocol` class defining the interface every family
  must implement: `fit`, `predict`, `coefs`, `residuals`,
  `reconstruct_y`, `fit_metric`, `diagnostics`, `classical_p_values`,
  and a `batch_fit` method that delegates to the active backend.
- [X] `LinearFamily` and `LogisticFamily` implementations refactored
  from existing `core.py` logic.
- [X] `resolve_family()` dispatch: `"auto"` resolves to `"linear"` or
  `"logistic"` via current binary detection; explicit strings map
  directly.

### Step 2 — `_backends/` package

- [X] Promote v0.2.0's `_jax.py` module into a `_backends/` package
  with a `BackendProtocol` defining `batch_ols`, `batch_logistic`,
  `batch_poisson`, `batch_negbin`, and `batch_ordinal` methods.
- [X] `_backends/_numpy.py`: NumPy/sklearn fallback (always available).
- [X] `_backends/_jax.py`: JAX accelerated path (optional dependency).
- [X] `resolve_backend()` reads `_config.get_backend()` and returns the
  appropriate backend object.  Future accelerators (CuPy, etc.) slot
  in as additional modules implementing `BackendProtocol` with no
  changes to `families.py` or `core.py`.

### Step 3 — Core refactor

Depends on Steps 1–2.  Converts the existing engine from hard-coded
`is_binary` branching to family-dispatched method calls.

- [X] Replace `is_binary` branching in `core.py` with family method
  calls: generic `_ter_braak_generic`, `_kennedy_individual_generic`,
  and `_kennedy_joint` functions dispatch to the active family.
- [X] `family=` parameter on `permutation_test_regression()`, with
  `"auto"` as the default.
- [X] Resolved family name included in the result dict.
- [X] `compute_all_diagnostics` accepts `model_type: str` instead of
  `is_binary: bool`.
- [X] `calculate_p_values` accepts a `ModelFamily` instance and
  delegates classical p-value computation to
  `family.classical_p_values()`, removing the duplicate statsmodels
  refit that was previously hard-coded in `pvalues.py`.
- [ ] Confounder module updated: `family` parameter on
  `identify_confounders` and `mediation_analysis`, using the
  family-appropriate model for the b-path and total-effect equations.

### Step 4 — JAX improvements

Depends on Step 2 (`_backends/_jax.py`).

- [X] JAX convergence control: `max_iter`, `tol`, convergence warnings
  for all Newton–Raphson solvers in `_backends/_jax.py`.
- [X] **Float64 precision:** `jax.config.update("jax_enable_x64", True)`
  at import, all dtypes changed from `float32` to `float64`.
  `_DEFAULT_TOL` set to `1e-8` (matching statsmodels' IRLS default).
  Float64 noise floor is `κ(H) × ε_f64 ≈ 1e-12`, well below
  tolerance — eliminates spurious convergence failures on
  ill-conditioned Hessians (`κ ≈ 10,000`) that plagued float32
  (noise floor `≈ 6e-4`, above `tol = 1e-4`).
- [X] **Triple convergence criteria (OR):** (1) gradient `|g|_∞ < tol`,
  (2) parameter-change `|Δβ|_∞ < tol`, (3) relative NLL change
  `|Δf|/max(|f|, 1) < tol`.  All gated on `jnp.isfinite(beta_new)`.
  OR is safe because logistic regression is strictly convex.
- [X] **`while_loop` dynamic early exit:** replaced `fori_loop` + `cond`
  skip pattern (which ran all 100 iterations at ≈10s for B=5000) with
  `jax.lax.while_loop` that exits at iteration 3–4 (≈0.4s), a 25×
  speedup.  Float64 makes convergence checking reliable, so dynamic
  exit is the correct choice.
- [X] **Damped Hessian regularisation:** `_MIN_DAMPING = 1e-8` added to
  Hessian diagonal for near-singular cases.
- [X] **Aggregated convergence warnings:** single summary warning with
  percentage and actionable guidance (VIF, quasi-complete separation)
  instead of per-feature warnings.
- [X] **JAX-accelerated Kennedy individual linear path:**
  `jax.vmap` over `jnp.linalg.lstsq` to eliminate the
  per-permutation Python loop, matching the existing JAX logistic
  architecture.  Implemented as `LinearFamily.batch_fit` dispatching
  through the JAX backend.
  The ter Braak and joint linear paths already use a single NumPy
  pseudoinverse multiply and gain little from JAX, but all families
  should provide a JAX path for consistency and to prefer autodiff
  over manual gradient implementations wherever possible.

### Step 4 Quickfix — Display & result-dict improvements

- [X] **Kennedy no-confounders Notes section:** `print_results_table`
  and `print_joint_results_table` now render a styled Notes block
  when the Kennedy method is called without confounders, matching the
  existing Notes style in `print_confounder_table` and
  `print_diagnostics_table`.  The programmatic `UserWarning` is
  retained for non-display consumers; the example script suppresses
  it via `warnings.catch_warnings()` so the table note is the only
  user-facing message.
- [X] **`confounders` key in non-joint result dict:** the return dict
  from `permutation_test_regression` for `method="kennedy"` now
  includes a `"confounders"` key (already present for
  `"kennedy_joint"`), enabling display functions to detect the
  no-confounders condition without inspecting the method string alone.

### Step 5 — Parallelisation

Depends on Step 1 (`ModelFamily.batch_fit()`).

- [X] Parallelise the scikit-learn fallback loops via
  `joblib.Parallel(prefer="threads")` inside `NumpyBackend`:
  `batch_logistic()`, `batch_logistic_varying_X()`, and
  `batch_ols_varying_X()`.  Threads are effective because
  BLAS/LAPACK routines and sklearn's L-BFGS solver release the GIL.
- [X] `_kennedy_joint` per-permutation refit loop parallelised
  directly via joblib when `n_jobs != 1`.
- [X] `batch_ols()` is already fully vectorised (single pseudoinverse
  multiply), so `n_jobs` has no effect there.
- [X] Add `n_jobs: int = 1` parameter to
  `permutation_test_regression()`.  `n_jobs=-1` uses all cores.
- [X] Family layer strips `n_jobs` from kwargs and forwards it only
  to `NumpyBackend` methods.  JAX backend methods never see
  `n_jobs` — their signatures remain clean.
- [X] When the JAX backend is active and `n_jobs != 1`, a
  `UserWarning` is emitted and `n_jobs` resets to 1 (JAX uses
  `vmap` vectorisation, so joblib parallelism is redundant).
- [X] Null distributions exposed in result dicts: `"permuted_coefs"`
  `(B, p)` array for ter Braak / Kennedy individual;
  `"permuted_improvements"` length-B array for Kennedy joint.
  Enables user-side density plots and custom p-value calculations.
- [X] Example scripts rewritten to pass `family=` explicitly and
  exercise the full `ModelFamily` protocol: `resolve_family`,
  `validate_y`, `fit`, `predict`, `coefs`, `residuals`,
  `reconstruct_y`, `fit_metric`, `batch_fit`, `diagnostics`, and
  `classical_p_values`.

### Step 6 — Freedman–Lane permutation method

Depends on Step 3 (family-dispatched core).  New permutation method
for all existing families.

- [X] `method="freedman_lane"` (individual) and
  `method="freedman_lane_joint"` — permutes residuals from the
  **full** model and adds them to fitted values from the **reduced**
  model.  Better power than Kennedy when predictors are correlated
  (Anderson & Legendre 1999; Winkler et al. 2014).  The default
  permutation method in FSL PALM, AFNI, and FreeSurfer.
- [X] Shares >90% of the Kennedy codepath — only the residual source
  and fitted-value base differ.  Reuses `family.residuals()`,
  `family.reconstruct_y()`, and `family.batch_fit()` pipeline.
- [X] When `confounders=[]`, Freedman–Lane reduces to ter Braak;
  a `UserWarning` guides users to the simpler method.
- [X] All five methods (ter Braak, Kennedy individual/joint,
  Freedman–Lane individual/joint) available for every family.
- [X] Result dicts carry `"family"` and `"backend"` provenance keys
  alongside the existing `"model_type"` key (Step 17).

### Step 7 — New GLM families

Depends on Steps 1–3 and 6.  Each new family inherits all five
permutation methods and both backends from the start.

#### Poisson regression

- [ ] Permutation tests for count outcomes with an exponential mean
  function.  The ter Braak residual-permutation approach generalises
  naturally: fit the reduced Poisson model, extract deviance residuals,
  permute, and refit the full model.
- [ ] Diagnostics: deviance, Pearson chi-squared, AIC/BIC,
  overdispersion test.

#### Negative binomial regression

- [ ] Handles overdispersed count data where the Poisson assumption
  fails.
- [ ] Requires estimation of the dispersion parameter once on the
  observed data, held fixed throughout the permutation loop.
- [ ] Diagnostics: deviance, AIC/BIC, alpha (dispersion) estimate.

#### Ordinal logistic regression

- [ ] Proportional-odds model for ordered categorical outcomes.
- [ ] Direct permutation of the ordinal response (residuals are not
  well-defined for ordinal); threshold parameters re-estimated on each
  permutation.

#### Multinomial logistic regression

- [ ] Extends binary logistic to unordered categorical outcomes with
  *K* classes, producing *K* − 1 coefficient vectors.
- [ ] The test statistic becomes a vector (one per contrast) or can be
  reduced to a scalar via the log-likelihood ratio or deviance.
- [ ] Table output should support both a stacked all-contrasts view and
  individual per-contrast tables.

### Step 8 — Sign-flip test

Depends on Step 1 (`family.residuals()`, `family.reconstruct_y()`).

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

---

## v0.4.0 — Exchangeability & Multilevel Frameworks

Formalises the permutation structure: which observations are
exchangeable with which, under what constraints.  These
exchangeability cells are the permutation-side complement to the
model-side `ModelFamily` protocol — together they define a complete
permutation test specification for any single-equation model.

### Exchangeability cells

- [ ] New `groups=` parameter accepting a column name or array that
  identifies clusters (schools, firms, regions, panel units, etc.).
- [ ] `permutation_strategy=` argument controlling the exchangeability
  structure:
  - `"within"` — permute observations within each cluster, holding
    cluster membership fixed.  Appropriate when the research question
    concerns within-cluster variation and cluster-level confounding is
    controlled by design.
  - `"between"` — permute entire clusters as units.  Appropriate when
    the treatment is assigned at the cluster level.
  - `"two-stage"` — permute clusters first, then permute observations
    within clusters.  Suitable for crossed or partially nested designs.
- [ ] Generalisation to arbitrary blocking factors: permutations occur
  only within strata defined by one or more categorical variables.
- [ ] User-supplied permutation constraints via a callback for
  non-standard exchangeability structures.
- [ ] Validation logic to ensure the requested strategy is compatible
  with the data (e.g., `"between"` requires sufficient clusters for
  meaningful permutation).

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
this release redesigns the output interface to expose graph-structured
results for both academic reporting and programmatic pipeline usage.

### `PermutationTestResult` dataclass

- [ ] Replaces plain-dictionary return values with typed dataclasses:
  - `PermutationTestResult` — single-equation individual-coefficient
    tests (backward-compatible with current usage).
  - `JointPermutationTestResult` — single-equation joint tests.
  - `GraphTestResult` — multi-equation results from the graph
    specification layer, containing per-equation results, per-edge
    p-values, per-hyperedge p-values, and per-path indirect effects.
- [ ] Direct attribute access: `.coefs`, `.p_values`,
  `.permuted_p_values`, `.classic_p_values`, `.diagnostics`,
  `.method`, `.family`.
- [ ] Academic output: `.summary()` prints the ASCII table (current
  behaviour).  `.to_latex()` produces a publication-ready LaTeX
  table.  `.to_html()` renders in Jupyter notebooks.
  `.to_markdown()` generates a Markdown table for inclusion in
  reports or documentation.
- [ ] Programmatic access: `.to_dataframe()` returns a tidy pandas
  DataFrame of coefficients, standard errors, and p-values.
  `.to_dict()` preserves backward compatibility with the current
  dictionary interface.
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

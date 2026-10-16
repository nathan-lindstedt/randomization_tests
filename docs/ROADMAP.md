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

Versions v0.1.0 through v0.4.0 are complete. Their detailed
checklists have been retired from this document. Key design
decisions that carry forward:

## Deferred from v0.4.0 to v0.5.0

### Inferential improvements

- [ ] Adaptive stopping: optionally halt the permutation loop early
  once significance or non-significance is established.  To avoid
  optional-stopping bias (super-uniformity of $(b+1)/(B+1)$ holds only
  for fixed $B$), sequential evaluation adopts the Besag & Clifford (1991)
  stopping boundary: halt when the exceedance count reaches a fixed target
  $h$ (e.g. $h=10$) with total randomizations $L$, computing the exact
  sequential p-value $h/L$; or continue to the maximum $B$ cap.
  Implementation operates via chunked batches — the engine pre-generates
  randomization indices and evaluates them in fixed-size blocks with
  vectorised batch fitting.  (Deferred to v0.5.2).
- [ ] Conditional Monte Carlo: permute within the sufficient-statistic
  strata of a nuisance parameter for exact conditional tests.
  `ScoreExactStrategy` currently implements full PQL-fixed IRLS on GLMM
  working responses via `_pql_fixed_irls_vmap`; a general exact-enumeration
  and network-algorithm mode for non-mixed GLMs (conditioning on sufficient
  statistics $\sum X_j y$) is needed first for sufficient-statistic
  conditioning to have a meaningful integration point across general
  families.  (Deferred to v0.5.x).

### Compatibility validation module

Consolidates the distributed method-incompatibility checks scattered
across `core.py` and `engine.py` into a single `_validation.py`
module with a structured compatibility matrix.  Audit of the codebase
(142 validation checks total) found that existing checks are already
organized into natural groupings: method/family guards in
`engine.py __init__()`, sign-flip/AR/confounder guards in
`core.py _validate_and_prepare_inputs()`, and group/cell guards in
`core.py _validate_groups()`.  Messages are already consistent (what
was requested, why incompatible, what to use instead).  The v0.5.0
graph compiler — which requires programmatic `validate_compatibility()`
access for per-equation validation — serves as the natural trigger
for centralisation without adding redundant runtime indirection.

- [X] `_validation.py` module with compatibility matrix and
  `validate_compatibility()` public function.
- [X] `ValidationIssue(level, code, message, suggestion)` typed
  objects for programmatic handling by the graph compiler.
- [ ] Refactor 19 compatibility checks from `core.py` and `engine.py`
  to delegate to the compatibility matrix (deferred to graph compiler
  integration).

---

## v0.5.0 — Graph Specification & Inference Abstraction Tower

The architectural centrepiece: a typed **Directed Acyclic Hypergraph (DAHG)**
data structure based on Dong et al. (2026) that lets users declare
multi-equation structural causal models with higher-order parental
interactions and have the package automatically derive which equations to
fit, which families to use, which permutation strategies to apply, and which
null hypotheses to test.  A standard regression is a single-layer DAHG with
parents pointing at one outcome; mediation, path models, and multi-equation
systems are deeper DAHGs composed of the same node-level equation solvers
built in v0.3.0, constrained by the exchangeability cells and trees built in
v0.4.0.  This milestone also builds out the inference abstraction tower,
incorporating nonparametric kernel tests, text representations, debiased
machine learning, conformal prediction, and invariance testing.

### Kernel protocol & nonparametric tests

- [X] `Kernel` protocol and `KernelEval` unified representation
  supporting full Gram and low-rank Nyström factorisations with
  leverage score sampling (`_kernels.py`).
- [X] Concrete kernel implementations: `GaussianKernel` (with median
  heuristic), `CosineKernel`, `LinearKernel`, `LaplacianKernel`,
  and `PrecomputedKernel`.
- [X] Factored Gram-matrix operations (`gram_trace_product`,
  `gram_centering`, `gram_permute`, `gram_row_sums`).
- [X] Nonparametric two-sample testing via Maximum Mean Discrepancy
  (`mmd_test`).
- [X] Nonparametric independence testing via Hilbert-Schmidt Independence
  Criterion (`hsic_test`).
- [X] Nonparametric regression testing in the RKHS (`kernel_regression_test`).

### Text processing pipeline

- [X] Text vectorisation via TF-IDF (`docs_to_tfidf`) and topic
  proportions via KL-NMF (`docs_to_topics`).
- [X] Topic model diagnostic tools: `coherence_score` (NPMI),
  `exclusivity_score`, and joint model selection `select_n_components`.
- [X] End-to-end `text_mmd_test` with automatic pooled-vocabulary
  handling and custom embedding support.

### Mixed-model inference-space refactor

- [X] Whitened tangent-space linear model for GLMM score projection,
  eliminating score projection offset to machine precision.
- [X] Longitudinal AR estimation decontaminated from cluster random
  effects via within-panel Frisch–Waugh–Lovell demeaning and Nickell
  bias correction (`estimate_panel_ar_coefficients`).
- [X] Composite cluster covariance whitening ($V_g = \Omega_g + Z_g \Gamma Z_g^T$)
  via block Cholesky for longitudinal AR mixed models.
- [X] Unified varying-X batch solver delegating to vectorised whitened
  OLS across LMM and GLMM, unblocking Kennedy and Kennedy joint for GLMM families.
- [X] Decoupled model-structure grouping from permutation strategy via
  `permutation_strategy="unrestricted"`.
- [X] Exact Woodbury GLS projection in `LinearMixedFamily.batch_fit_and_score()`
  for reduced designs, eliminating scale mismatches across batch fits.
- [X] Poisson/Binomial Rule of Three borderline $B^*$ recommendation in
  display tables.

### Double/Debiased Machine Learning (DML)

- [X] Cross-fitting implementation (`_cross_fit_residualize`) with
  $K$-fold and `GroupKFold` support, estimator cloning, and deterministic
  seed propagation.
- [X] Polymorphic `_DMLReducedModel` container satisfying `predict()` and
  `predict_proba()` across all model families.
- [X] Freedman–Lane integration with `reduced_model=` parameter on
  `randomization_test_regression` and `kernel_regression_test`.

### Conformal prediction

- [X] Distribution-free split conformal prediction intervals and sets for
  regression and classification (`conformal_prediction`).
- [X] Jackknife+ conformal prediction with leave-one-out cross-validation.
- [X] Out-of-sample empirical coverage and Feature-Stratified Coverage (FSC)
  evaluation (`evaluate_conformal`, `ConformalEvaluationResult`).

### Invariance testing

- [X] Testing invariance of conditional distributions ($Y \perp E \mid X$)
  across environments (`invariance_test`, `InvarianceResult`).
- [X] Multi-tier test dispatch: exact discrete stratification (`"stratified"`),
  residualized joint regression testing (`"freedman_lane"`, `"kennedy"`, `"score"`),
  and conditional permutation tests (`"conditional"` via CPT MCMC).

### Knockoff filters

- [X] False Discovery Rate (FDR) controlled variable selection via
  knockoff filters (`knockoff_test`, `KnockoffResult`).
- [X] Fixed-X knockoffs for linear regression ($n \ge 2p$, Barber & Candès 2015, finite-sample exact FDR)
  and Model-X knockoffs for general designs (Candès et al. 2018, Ledoit-Wolf shrinkage).
- [X] MMD swap exchangeability diagnostic to verify knockoff construction
  quality (`mmd_test` on feature-knockoff pairs).

### Specification data structure

- [ ] A `CausalGraph` class implementing a **Directed Acyclic Hypergraph (DAHG)**
  per Dong et al. (2026), supporting:
  - **Nodes** — observed variables, each optionally annotated with a
    model family, causal role (exposure, outcome, mediator, confounder), and
    equation-specific exchangeability tree.
  - **Directed edges** — pairwise causal claims ($|T(e)| = 1$, $X \to Y$), each
    testable individually via per-coefficient permutation tests.
  - **Directed hyperedges** — higher-order parental mechanisms ($|T(e)| \ge 2$, $\{X_1, X_2\} \to Y$),
    representing multilinear tail-tuple interaction terms $\prod_{u \in T(e)} X_u$ in the SEM
    design matrix (Dong et al. 2026).
  - **Exchangeability trees** — optional per-node or global
    permutation constraints inherited from the `ExchangeabilityTree` system,
    enabling different structural equations to carry different exchangeability
    hierarchies (e.g. between-cluster for exposures, within-cluster for outcomes).
- [ ] Validation: DAHG acyclicity verification via topological sorting on the
  projected shadow dependency graph $\mathcal{E}_{\text{shadow}} = \bigcup_e \{(u, h(e)) : u \in T(e)\}$,
  connected-component analysis, family compatibility per equation, and identification
  of under-determined nodes.

### Graph compiler

- [ ] DAHG topological sort via Kahn's algorithm on the shadow dependency graph
  (formally grounded in the Fourier-slice nilpotency theorem of Dong et al. 2026).
- [ ] For each outcome (head) node, derive exactly **one structural equation**:
  `(outcome, predictors, confounders, family, method, exchangeability_tree)`.
- [ ] Resolve families automatically (from outcome type) or from
  per-node annotation.
- [ ] Role-based confounder separation: incoming edges/hyperedges with `role="confounder"`
  (or right of `|` in formulas) populate `confounders=`; non-confounder sources populate
  `predictors=`.
- [ ] Map pairwise edges to individual permutation tests (Freedman–Lane or Kennedy);
  map interaction hyperedges to derived product columns (`X1:X2`).

### Multi-equation orchestrator

- [ ] Execute permutation tests for each structural equation in
  topological order, dispatching to the `ModelFamily` protocol.
- [ ] Dynamic interaction materialization: dynamically compute interaction columns
  (`"A:B" = A * B`) in working copies during execution, preserving input DataFrame immutability.
- [ ] Independent permutation per equation by default: each equation
  is tested in isolation, conditioning on observed values of upstream
  nodes.
- [ ] Optional path-residual mode: conduct path-specific edge permutations
  (e.g., residual permutation of mediator $M$ conditional on $X$ to isolate $M \to Y$),
  enabling exact testing across multi-step paths without composite-null inflation.
- [ ] Collect per-equation results into a unified graph-level result
  object (`GraphTestResult`).

### Indirect effect extraction

- [ ] For a declared path X → M → Y, compute the product of per-edge coefficients ($a \times b$)
  and test $H_0: ab = 0$ via the maximum $p$-value rule ($p_{ab} = \max(p_a, p_b)$ across edge permutations)
  or path-residual permutation nulls.
- [ ] Path confidence intervals evaluated via case-resampling bootstrap
  percentiles (resampling rows/clusters and refitting all equations),
  strictly distinguishing hypothesis testing (zero-centred permutation null)
  from parameter estimation.
- [ ] Support arbitrary-length causal chains ($X \to M_1 \to M_2 \to Y$)
  with product-of-coefficients test statistics.
- [ ] Automated detection and reporting of inconsistent mediation (suppression),
  flagging cases where direct ($c'$) and indirect ($a \times b$) paths have
  statistically significant opposite signs.

### Hyperedges vs. Omnibus Joint Testing

- [ ] Strict category separation between structural mechanisms and inferential tests:
  - **DAHG Hyperedge Mechanisms** (Dong et al. 2026): A structural non-linear synergy
    $\{X_1, X_2\} \to Y$ contributing $\beta_{12} (X_1 \cdot X_2)$ to the SEM, tested
    as a 1-df hypothesis ($H_0: \beta_{12} = 0$) via interaction column permutation.
  - **Omnibus Joint Hypotheses** (Anderson & Robinson 2001): An inferential hypothesis
    grouping over multiple linear predictors (e.g. categorical factor levels, `{X1, X2}`),
    tested as a $k$-df subspace hypothesis ($H_0: \beta_1 = \beta_2 = 0$) via
    `method="kennedy_joint"` or `method="freedman_lane_joint"` without multiplying variables.
- [ ] Support mixed-family equations within the same DAHG (e.g., Poisson outcome with
  logistic mediator).

### Model specification syntax

- [ ] Primary API: Python method calls —
  `g.add_node("Y", family="linear")`,
  `g.add_edge("X1", "Y")`,
  `g.add_hyperedge(["X1", "X2"], "Y")`.
- [ ] Convenience: arrow-style string parser —
  `"Y <- X1 + X2 [linear]; M <- X1 [linear]; Y <- M"` with equations
  parsed and DAHG inferred.
  - Supports Wilkinson–Rogers interaction syntax: `X1 * X2` (expands to main effects
    and interaction hyperedge `X1:X2`).
  - Supports inferential joint grouping braces: `{X1, X2}` (triggers omnibus joint test).
  - Supports confounder partition: `Y <- Tested | Controlled :: Grouping`.
- [ ] Convenience: dictionary specification for programmatic and
  configuration-file workflows.

### Markov compatibility testing

- [ ] Test the Local Markov Condition for declared DAHG structures
  (`markov_compatibility_test`) via permutation conditional independence
  tests across parents and non-descendants.
- [ ] Three-tier rigor ladder: Tier 1 (exact discrete stratification), Tier 2 (CPT resampling),
  Tier 3 (residualized permutation fallback).
- [ ] Westfall–Young stepdown resampling for family-wise error rate
  (FWER) control across graph constraints.
- [ ] Return structured `MarkovCompatibilityResult` with per-node
  diagnostics and constraint summaries.

### Model guidance & diagnostic engine

- [ ] Automated pre-test and post-test diagnostic evaluation to assess
  model assumptions (linearity, dispersion, zero-inflation, clustering,
  temporal correlation).
- [ ] Context-aware recommendation engine guiding users to appropriate
  families, strategies, and permutation configurations.

### Interoperability (deferred to v0.5.1)

- [ ] The `CausalGraph` internal representation should use a directed
  incidence matrix as its canonical form (sparse, with $+1$ for head node,
  $-1$ for tail nodes of each hyperedge), enabling natural conversion to
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
- [ ] Synergies with causal discovery libraries (`causal-learn`):
  permute-based validation of discovered DAG/DAHG skeletons.

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
- [X] `KernelTestResult`, `ConformalResult`, `InvarianceResult`,
  `KnockoffResult` — extended result dataclasses (pulled forward to v0.5.0).
- [ ] `GraphTestResult` — multi-equation results from the graph
  specification layer (v0.5.0), containing per-equation results,
  per-edge p-values, per-hyperedge p-values, and per-path indirect
  effects.
- [ ] `MarkovCompatibilityResult` — results from Markov compatibility
  testing (v0.5.0).
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

## v0.7.0 — Causal Discovery & Visualization

Adds methods for data-driven discovery of graph structure, feeding
directly into the graph specification layer, plus visualization
utilities for permutation distributions and graph topology.

### Causal screening

- [X] **v0.4.1** — Four-stage confounder sieve (screen → collider →
  mediator → moderator), partial and distance correlation screening,
  multiple-testing correction (Holm/FDR-BH), cluster bootstrap for
  mediation/moderation, E-value and Rosenbaum bounds sensitivity
  analysis, `ConfounderAnalysisResult` dataclass.
- [ ] Supplement the sieve with causal direction testing for
  distinguishing confounders from mediators:
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

### Visualization utilities

- [ ] Permutation distribution histograms with observed test statistic
  annotated.
- [ ] Coefficient forest plots comparing observed vs. permutation null
  distributions across predictors.
- [ ] Graph topology visualization: render the `CausalGraph` with
  edges colored and weighted by permutation p-values.
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

- [X] Runtime benchmarks across *n*, *n_randomizations*, *n_features*,
  and model families (`profile_endtoend.py`, `profile_solvers.py`, `profile_subsystems.py`).
- [X] Comparison against naive (non-vectorised) implementations to
  quantify the performance gains from batch algebra, Woodbury whitening, and tangent-space score tests.
- [ ] Published benchmark results and architecture reports in `benchmarks/summaries/`.

### GPU acceleration

- [ ] Implement a PyTorch backend (`_backends/_torch.py`) conforming
  to `BackendProtocol`.  PyTorch's `torch.func` module provides
  near-1:1 parity with the JAX functional APIs used in the existing
  JAX backend — `torch.func.vmap`, `torch.func.grad`, and
  `torch.func.hessian` map directly to their JAX equivalents, and
  `torch.compile` replaces `jax.jit`.  The two `jax.lax.while_loop`
  call sites translate to standard Python loops.  This gives all
  model families GPU support automatically through the existing
  backend dispatch system while providing native Windows CUDA
  support (the primary motivation — JAX offers only experimental
  CPU-only wheels on Windows with no GPU path).
- [ ] Document hardware requirements (CUDA toolkit, supported GPU
  architectures) and expected speedups.
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
- [X] **Documentation:** keep the API reference, quickstart guide, and
  changelog in sync with every release.  (Phase 7: `__init__.py`
  docstring rewritten, API.md signature updated with all 22
  parameters, `resolve_family` signature corrected, relative import
  fix applied.  Phase 8: codebase-wide comment quality lift — ICC
  citations, Woodbury derivation, IRLS citation, Cholesky
  narration, Clopper-Pearson citation, Rosenbaum derivation, table
  geometry, and `np.bool_`/thread-safety notes added.)
- [ ] **Property-based tests:** add Hypothesis-based property tests for
  core invariants (p-values in [0, 1], permuted arrays are true
  permutations, result schema completeness, determinism under fixed
  seed, commutativity of confounder ordering).  Not tied to a
  specific release — expand incrementally as new families and methods
  are added.

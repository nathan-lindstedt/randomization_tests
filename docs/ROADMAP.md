# Roadmap

This document outlines the planned development trajectory for
**randomization-tests**, from its current alpha state through a stable
1.0 release.  Version numbers are indicative — scopes may shift as the
project evolves — but the ordering of milestones reflects deliberate
dependency reasoning: all statistical machinery is finalized before the
public-facing result interface is locked down, so that abstraction only
needs to be designed once.

---

## v0.1.0 — Foundation

The initial packaged release.  Establishes the src-layout structure,
core algorithms, and development infrastructure.

- [x] **Permutation methods:** ter Braak (1992) residual permutation,
  Kennedy (1995) individual exposure-residual permutation, and Kennedy
  (1995) joint F-style test.
- [x] **Vectorised OLS** via batch pseudoinverse multiplication — the
  design matrix is inverted once and applied to all permuted response
  vectors simultaneously.
- [x] **Optional JAX backend** for logistic regression: `jax.vmap` over a
  custom Newton-Raphson solver with `jax.grad`-computed gradients.
  Transparent fallback to scikit-learn when JAX is not installed.
- [x] **Unique permutation pre-generation** with hash-based deduplication
  for large *n* and exhaustive enumeration for small *n* (≤ 12).
- [x] **Phipson & Smyth (2010) corrected p-values** that are guaranteed
  never to be exactly zero.
- [x] **Confounder identification pipeline:** Pearson correlation screening
  followed by Preacher & Hayes (2004, 2008) mediation analysis with
  bias-corrected and accelerated (BCa) bootstrap confidence intervals
  (Efron, 1987).  The indirect effect (a × b) is tested directly via
  5 000 bootstrap resamples; the jackknife-based acceleration
  correction accounts for skewness in the sampling distribution.
- [x] **Formatted ASCII table output** modeled after statsmodels, with
  separate layouts for individual-coefficient and joint tests.
- [x] **Development infrastructure:** Google-style docstrings, PEP 561
  `py.typed` marker, modern Python 3.10+ type annotations, GitHub
  Actions CI (ruff, mypy, pytest across Python 3.10–3.13), and a test
  suite of 40 tests.

---

## v0.1.1 — Polish & diagnostics

Adds extended diagnostics, input flexibility, display improvements, and
expands the test suite.

- [x] **Polars input support:** all public functions accept both pandas
  and Polars DataFrames via an internal `_ensure_pandas_df` adapter in
  `_compat.py`.  A `DataFrameLike` type alias documents the supported
  input types.
- [x] **Extended diagnostics module** (`diagnostics.py`): standardised
  coefficients, variance inflation factors (VIF), Monte Carlo standard
  error, empirical-vs-asymptotic divergence flags, Breusch-Pagan
  heteroscedasticity test (linear), deviance residuals and runs test
  (logistic), Cook's distance influence counts, and permutation
  coverage reporting.
- [x] **Exposure R² column** in the diagnostics table for the Kennedy
  individual method, quantifying how much of each predictor's variance
  is explained by the confounders.  A near-1.0 value explains why the
  permuted p-value is inflated.
- [x] **`fit_intercept` parameter** added to all code paths (10
  functions in `core.py` and `pvalues.py`), enabling intercept-free
  models for through-origin regression.
- [x] **Confounder display table** (`print_confounder_table`): formatted
  80-character ASCII table for confounder identification results,
  replacing raw `print()` output.  Supports single- and multi-predictor
  inputs, parameter header, clean-predictor summary, and conditional
  mediator warning notes.
- [x] **Display improvements:** 80-character line-width constraint via
  `_wrap()` helper, `textwrap.wrap`-based title centering, consistent
  4-decimal p-value formatting (`"0.000"` / `"1.000"`), redesigned
  four-section diagnostics table layout.
- [x] **Cook's distance fix:** logistic Cook's D now delegates to
  `sm.GLM(family=Binomial()).get_influence().cooks_distance` instead of
  using `sm.Logit.fittedvalues` (which returns log-odds, not
  probabilities).
- [x] **Test suite expanded** from 40 to 124 tests covering diagnostics,
  fit_intercept, exposure R², Polars compatibility, and display
  formatting.

---

## v0.2.0 — Hardening

Focuses on reliability, code quality enforcement, and performance of
the existing feature set before any new statistical capabilities are
added.

### CI & code quality
- [ ] Achieve a fully clean ruff and mypy pass with zero warnings or
  suppressed ignores.
- [ ] Add a pre-commit configuration (ruff lint + format, mypy,
  trailing-whitespace, end-of-file-fixer) so contributors catch issues
  before pushing.

### Test coverage
- [ ] Expand the test suite with edge-case coverage: empty DataFrames,
  single-feature models, constant columns, perfect separation in
  logistic regression, and permutation requests that approach or exceed
  the available unique permutation count.
- [ ] Add convergence-failure tests for the JAX Newton-Raphson solver
  (ill-conditioned Hessians, rank-deficient designs).
- [ ] Add large-*n* smoke tests (e.g., *n* = 10,000, modest
  *n_permutations*) to verify memory footprint and runtime stay within
  reasonable bounds.

### Dependency management
- [ ] Establish a tested compatibility matrix of lower- and upper-bound
  dependency versions (numpy, pandas, scipy, statsmodels, scikit-learn)
  and verify in CI.
- [ ] Pin or document JAX version compatibility for the optional backend.

### Performance
- [ ] Parallelise the scikit-learn logistic regression fallback loop via
  joblib or multiprocessing, since each permutation fit is independent.
- [ ] Profile the hash-based permutation deduplication to identify any
  bottlenecks when *n_permutations* is large relative to *n*!.

---

## v0.3.0 — GLM family extensions

Extends the package beyond OLS and binary logistic regression to cover
the most commonly encountered generalised linear model families in
applied research.

### Poisson regression
- [ ] Permutation tests for count outcomes with an exponential mean
  function.  The ter Braak residual-permutation approach generalises
  naturally: fit the reduced Poisson model, extract deviance or Pearson
  residuals, permute, and refit the full model.
- [ ] Diagnostics: deviance, Pearson chi-squared, AIC/BIC,
  overdispersion test.

### Negative binomial regression
- [ ] Handles overdispersed count data where the Poisson assumption fails.
- [ ] Requires estimation of the dispersion parameter in addition to the
  linear predictor — the permutation loop must account for this extra
  nuisance parameter.
- [ ] Diagnostics: deviance, AIC/BIC, alpha (dispersion) estimate.

### Multinomial logistic regression
- [ ] Extends binary logistic to unordered categorical outcomes with *K*
  classes, producing *K* − 1 coefficient vectors.
- [ ] The test statistic becomes a vector (one per contrast) or can be
  reduced to a scalar via the log-likelihood ratio or deviance.
- [ ] Table output should support both a stacked all-contrasts view and
  individual per-contrast tables.

### Ordinal logistic regression
- [ ] Proportional-odds model for ordered categorical outcomes.
- [ ] Permutation of the ordinal response preserves the category labels;
  the threshold parameters are re-estimated on each permutation.

### Shared infrastructure
- [ ] A `family=` or `model_type=` parameter on the public API, with
  automatic detection retained as the default for binary and continuous
  outcomes.
- [ ] Per-family diagnostic dictionaries and table formatting.

---

## v0.4.0 — Multilevel & longitudinal frameworks

Adds support for clustered, hierarchical, and panel data structures
where observations are not exchangeable across the full sample.

### Cluster-aware permutations
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
- [ ] Validation logic to ensure the requested strategy is compatible with
  the data (e.g., `"between"` requires sufficient clusters for
  meaningful permutation).

### Mixed-effects models
- [ ] Integration with statsmodels `MixedLM` (linear) and a suitable
  backend for generalised linear mixed models (GLMM).
- [ ] The permutation target becomes the conditional residual (BLUP
  residual) under the reduced model, with random-effect structure held
  fixed across permutations.
- [ ] Diagnostics: marginal and conditional R², random-effect variance
  components, ICC.

### Longitudinal / panel data
- [ ] Within-panel permutation as the default strategy when a time or
  wave variable is provided, preserving the temporal dependency
  structure within each unit.
- [ ] Support for both balanced and unbalanced panels.
- [ ] Optional autoregressive residual structure for the reduced model.

### Block / stratified permutations
- [ ] Generalisation of the `groups=` concept to arbitrary blocking
  factors: permutations occur only within strata defined by one or more
  categorical variables.
- [ ] Useful for randomised block designs, matched-pairs analyses, and
  stratified observational studies.

---

## v0.5.0 — Methodological improvements

Revisits and strengthens the statistical methods available in the
package, incorporating more robust alternatives and additional
inferential tools.

### Causal screening
- [ ] Supplement the existing Preacher & Hayes mediation analysis with more
  robust approaches for distinguishing confounders from mediators:
  - **LiNGAM** (Shimizu et al., 2006) — exploits non-Gaussianity to
    identify causal direction in linear models.
  - **Additive noise models (ANM)** — nonparametric causal direction
    test based on independence of residuals.
  - **PC algorithm** — constraint-based causal discovery from
    conditional independence tests.
- [ ] Document the assumptions and limitations of each approach clearly in
  the API reference.

### Confidence intervals on permutation p-values
- [ ] Exact binomial or Clopper-Pearson confidence intervals reflecting
  Monte Carlo uncertainty when only a finite number of permutations are
  drawn.
- [ ] Adaptive stopping: optionally halt the permutation loop early once
  the CI for the p-value is narrow enough to determine significance
  with a specified confidence.

### Stratified / constrained permutations
- [ ] Beyond block permutations (v0.4.0), support user-supplied
  permutation constraints via a callback or constraint specification
  language, for non-standard exchangeability structures.
- [ ] Conditional Monte Carlo: permute within the sufficient-statistic
  strata of a nuisance parameter for exact conditional tests.

### Visualisation utilities
- [ ] Permutation distribution histograms with observed test statistic
  annotated.
- [ ] Coefficient forest plots comparing observed vs. permutation null
  distributions across predictors.
- [ ] Option to return matplotlib Figure/Axes objects for further
  customisation, or render inline in Jupyter notebooks.

---

## v0.6.0 — Structured results & dual API

With all statistical machinery finalised, this release redesigns the
output interface to serve both academic reporting and programmatic
pipeline usage from a single set of result objects.

### `PermutationTestResult` dataclass
- [ ] Replaces the current plain-dictionary return values with a typed
  dataclass that supports:
  - **Direct attribute access:** `.coefs`, `.p_values`,
    `.permuted_p_values`, `.classic_p_values`, `.diagnostics`,
    `.method`, `.model_type`.
  - **Academic output:** `.summary()` prints the ASCII table (current
    behaviour).  `.to_latex()` produces a publication-ready LaTeX
    table.  `.to_html()` renders in Jupyter notebooks.
    `.to_markdown()` generates a Markdown table for inclusion in
    reports or documentation.
  - **Programmatic access:** `.to_dataframe()` returns a tidy pandas
    DataFrame of coefficients, standard errors, and p-values.
    `.to_dict()` preserves backward compatibility with the current
    dictionary interface.
- [ ] Separate result classes for individual-coefficient tests
  (`PermutationTestResult`) and joint tests
  (`JointPermutationTestResult`) with appropriate attributes for each.

### Scikit-learn estimator interface
- [ ] `PermutationTestRegressor` and `PermutationTestClassifier` wrappers
  conforming to the scikit-learn estimator contract: `fit`, `predict`,
  `get_params`, `set_params`, `score`.
- [ ] Compatible with `Pipeline`, `GridSearchCV`, `cross_val_score`, and
  other scikit-learn meta-estimators.
- [ ] The `fit` method runs the permutation test; results are accessible
  via the `.results_` attribute.  `predict` delegates to the underlying
  regression model.

### Flexible input handling
- [x] ~~Accept `numpy` arrays, `pandas` DataFrames/Series, and (optionally)
  `polars` DataFrames as inputs, coercing internally as needed.~~
  *(Moved to v0.1.1 — pandas and Polars DataFrames accepted via
  `_ensure_pandas_df`.)*
- [ ] Feature names inferred from DataFrame columns or supplied
  explicitly.

---

## v1.0.0 — Stable release & ecosystem

Marks the first stable public API with semantic versioning guarantees.

### API freeze
- [ ] All public function signatures, result object attributes, and
  parameter names are frozen.  Breaking changes after this point
  require a major version bump.
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
- [ ] Tutorials, worked examples, and a gallery of use cases.
- [ ] Cross-referenced with the academic background in the README.

### Benchmarks
- [ ] Comprehensive runtime benchmarks across *n*, *n_permutations*, and
  *n_features* for each model family.
- [ ] Comparison against naive (non-vectorised) implementations to
  quantify the performance gains from batch algebra and JAX.
- [ ] Published benchmark results in the documentation.

### GPU acceleration
- [ ] Extend the JAX backend to cover all model families (currently
  logistic only).
- [ ] Evaluate CuPy as an alternative GPU path for numpy-based batch
  operations.
- [ ] Document hardware requirements and expected speedups.

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

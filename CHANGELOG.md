# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Alpha] - Main Branch

- Replaced Baron & Kenny (1986) causal-steps mediation with Preacher & Hayes
  (2004, 2008) bootstrap test using bias-corrected and accelerated (BCa)
  confidence intervals (Efron, 1987). Default bootstrap samples increased
  from 1 000 to 5 000. Return dict now includes `ci_method` key.
- Added `get_backend()` / `set_backend()` runtime API and
  `RANDOMIZATION_TESTS_BACKEND` environment variable for selecting the
  compute backend (`"jax"`, `"numpy"`, or `"auto"`).
- Fixed JAX logistic solvers omitting the intercept column, which caused
  the Newton-Raphson solver to fit a different model specification than
  sklearn's `LogisticRegression(fit_intercept=True)` default. ter Braak
  and Kennedy empirical p-values now match across backends.
- Removed unused `_typing.py` module and dead `_fit_logistic_jax` single-fit
  function (superseded by the batched `_fit_logistic_batch_jax`).

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

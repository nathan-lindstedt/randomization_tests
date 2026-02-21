# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Alpha] - Main Branch

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
- Confounder identification pipeline: correlation screening + Baron & Kenny
  (1986) mediation analysis with bootstrap CIs.
- Formatted ASCII table output (`print_results_table`,
  `print_joint_results_table`).
- PEP 561 `py.typed` marker for downstream type-checking support.
- GitHub Actions CI (lint, type check, test matrix across Python 3.10–3.13).

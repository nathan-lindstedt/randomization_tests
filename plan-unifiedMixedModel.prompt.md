## Revised Plan: Unified Mixed-Model Architecture (LMM + GLMM)

**Summary:** Three plans executed in dependency order. Plan A is done. Plan B adds score projection — a batched permutation strategy that produces identical p-values to Freedman–Lane for LMM and serves as the required permutation mechanism for GLMM. Plan C adds GLMM families by plugging a Laplace NLL into the existing Newton solver, with a one-step corrector upgrade for second-order accuracy while remaining fully batched.

---

## Plan A: Henderson LMM (DONE)

**Status:** Complete. `_reml_newton_solve()` in `_backends/_jax.py` is the default solver. `LinearMixedFamily` in `families_mixed.py` implements full protocol with projection A. `FitContext` caches groups, Z, variance components. Exchangeability cells return first-factor group labels.

No further work needed.

---

## Plan B: Score Projection Strategy (DONE)

### Goal

Add `method="score"` and `method="score_joint"` that compute permutation test statistics via a single matmul per feature — no iterative solver in the permutation loop. For LMM this produces bit-for-bit identical p-values to Freedman–Lane. For GLMM (Plan C) this is the only supported permutation mechanism.

### Dependencies

- Plan A (done): `LinearMixedFamily` with calibrated `projection_A`.
- Existing infrastructure: `PermutationStrategy` protocol in `_strategies/__init__.py`, `_STRATEGY_REGISTRY`, `resolve_strategy()`, `fit_reduced()` in `families.py`.

### Mathematical Foundation

Score test statistic for feature j:

$$U_j = X_j^\top \hat{V}_0^{-1}(y - \hat{\mu}_0)$$

This is linear in y. For B permuted responses:

$$U_{j,\pi} = X_j^\top \hat{V}_0^{-1} y_\pi - \underbrace{X_j^\top \hat{V}_0^{-1} \hat{\mu}_0}_{\text{constant}}$$

Batch computation: `E_pi @ projection_row` — shape `(B, n) @ (n,) → (B,)`.

For LMM, score projection is mathematically identical to Freedman–Lane refit because the j-th row of projection A is exactly $X_j^\top \hat{V}_0^{-1}$ (up to normalization). The rank ordering across permutations is identical, so p-values match exactly.

### ~~Step B.1: `score_project()` protocol method~~

~~**File:** `families.py`~~

~~Add `score_project()` to `ModelFamily` protocol with a default body that raises `NotImplementedError`.~~

~~Implementation on `LinearFamily`: pseudoinverse row j, then `E_pi @ pinv_row` → `(B,)`.~~

~~Implementation on `LinearMixedFamily`: cached `projection_A` row j, then `E_pi @ projection_row` → `(B,)`.~~

~~All 5 GLM families implement with `NotImplementedError` to satisfy structural subtyping.~~

### ~~Step B.2: Score strategies~~

~~**New file:** `src/randomization_tests/_strategies/score.py`~~

~~`ScoreIndividualStrategy` (`is_joint = False`): per-feature score projection with constant offset.~~

~~`ScoreJointStrategy` (`is_joint = True`): RSS reduction via `batch_fit_and_score`.~~

~~`ScoreExactStrategy` (`is_joint = False`): Plan C placeholder raising `NotImplementedError`.~~

### ~~Step B.3: Registration and guard~~

~~`_strategies/__init__.py`: added `score`, `score_joint`, `score_exact` to `_ensure_registry()`.~~

~~`engine.py`: probe-based `try/NotImplementedError` guard for unsupported families.~~

~~Confounder masking in `core.py` extended to include `method="score"`.~~

~~n_jobs warning extended to cover score/score_joint on linear family.~~

### ~~Step B.4: Tests~~

~~**New file:** `tests/test_score_strategy.py` — 37 tests covering:~~
~~Score ≡ Freedman–Lane equivalence (LMM), Score ≡ ter Braak equivalence (linear),~~
~~protocol conformance, registry wiring, confounder masking, unsupported family rejection,~~
~~score_exact placeholder, n_jobs warning, determinism, p-value validity, LMM joint.~~

### Verification gate (all passed)

1. ~~Score p-values = Freedman–Lane p-values for LMM (exact equality)~~ ✅
2. ~~Score p-values = ter Braak p-values for linear family (exact equality)~~ ✅
3. ~~Score rejects families that don't implement `score_project()`~~ ✅
4. ~~`resolve_strategy("score")` and `resolve_strategy("score_joint")` work~~ ✅
5. ~~`mypy src/` clean, `ruff check src/ tests/` clean, full test suite green~~ ✅ (851 passed, 2 pre-existing)

---

## Plan C: GLMM via Laplace + One-Step Corrector

### Goal

Ship `LogisticMixedFamily` and `PoissonMixedFamily` using Laplace approximation for estimation and batched one-step corrector for permutation. The outer θ optimization plugs a Laplace NLL into the **existing `_reml_newton_solve()`** — no new solver code. The inner IRLS loop pre-scales inputs by √W to reuse **unweighted Henderson algebra from Plan A** — no new linear algebra. Batch permutation uses Plan B's `score_project()` with an optional one-step Newton correction for second-order accuracy — fully batched, O(n·B), no IRLS in the permutation loop.

### Dependencies

- Plan A (done): `_reml_newton_solve()`, `_build_reml_nll()`, Henderson algebra, `LinearMixedFamily` patterns, `FitContext` caching.
- Plan B: `score_project()` protocol method, `ScoreIndividualStrategy` / `ScoreJointStrategy`.

### Mathematical Foundation

#### GLMM formulation

$$g(\mu) = X\beta + Zu, \quad u \sim N(0, \Gamma(\theta)), \quad y|u \sim \text{ExponentialFamily}$$

#### Two-loop Laplace estimation

**Outer loop** — Newton on θ via `_reml_newton_solve(laplace_nll, total_chol_params, max_iter, tol)`:

$$\ell_L(\theta) = \ell(y|\hat\beta, \hat u; \theta) - \tfrac{1}{2}\hat u^\top \Gamma^{-1}\hat u - \tfrac{1}{2}\log|H_u|$$

where $(\hat\beta, \hat u)$ are the joint mode from the inner loop and $H_u$ is the Hessian of the penalized log-likelihood w.r.t. u at the mode.

**Inner loop** — IRLS on (β, u) for fixed θ:

Penalized log-likelihood:

$$\ell_P(\beta, u) = \sum_i \ell_i(y_i|\eta_i) - \tfrac{1}{2}u^\top\Gamma^{-1}u, \quad \eta = X\beta + Zu$$

IRLS working response and weights:

$$\tilde y_i = \eta_i + \frac{y_i - \mu_i}{g'(\mu_i)\cdot\text{Var}(y_i|u)}, \quad w_i = \frac{[g'(\mu_i)]^2}{\text{Var}(y_i|u)}$$

Pre-scaling: $\tilde X = \sqrt W \odot X$, $\tilde Z = \sqrt W \odot Z$, $\tilde y^* = \sqrt W \odot \tilde y$. Then the IRLS update is the **unweighted Henderson system** applied to $(\tilde X, \tilde Z, \tilde y^*)$ — identical to Plan A:

$$\begin{bmatrix}\tilde X^\top\tilde X & \tilde X^\top\tilde Z \\ \tilde Z^\top\tilde X & \tilde Z^\top\tilde Z + \Gamma^{-1}\end{bmatrix}\begin{bmatrix}\beta \\ u\end{bmatrix} = \begin{bmatrix}\tilde X^\top\tilde y^* \\ \tilde Z^\top\tilde y^*\end{bmatrix}$$

#### Batch permutation: one-step corrector

Under H₀ (feature j absent), fit the null model once → get $\hat\beta_0$, $\hat u_0$, $W_0$, $\mathcal I_0$.

**Score projection** (first-order):

$$U_{j,\pi} = (X_j \odot V_0^{-1\text{diag}})^\top(y_\pi - \hat\mu_0)$$

**One-step corrector** (second-order, Le Cam estimator):

$$\hat\beta_{j,\pi}^{(1)} = \hat\beta_{0,j} + [\mathcal I_0]_{jj}^{-1} \cdot U_{j,\pi}$$

Both are matmuls — `(B, n) @ (n,) → (B,)` for the score, then a scalar multiply for the correction. Fully batched. O(n·B) total. No IRLS in the permutation loop.

The Fisher information $\mathcal I_0 = X^\top W_0 X + X^\top W_0 Z\, C_{22}^{-1}\, Z^\top W_0 X$ (the Schur complement of the Henderson system under H₀) is computed once during calibration and cached on the family instance.

### ~~Step C.1: Per-family conditional NLL + IRLS functions~~

~~**File:** `_backends/_jax.py`~~

~~Four pure JAX functions (~40 lines total):~~

~~**Logistic:**~~

~~```python~~
~~def _logistic_conditional_nll(y, eta):~~
~~    return -jnp.sum(y * eta - jnp.logaddexp(0.0, eta))~~

~~def _logistic_working_response_and_weights(y, eta):~~
~~    mu = jax.nn.sigmoid(eta)~~
~~    w = jnp.clip(mu * (1.0 - mu), 1e-10)~~
~~    z_tilde = eta + (y - mu) / w~~
~~    return z_tilde, w~~
~~```~~

~~**Poisson:**~~

~~```python~~
~~def _poisson_conditional_nll(y, eta):~~
~~    return -jnp.sum(y * eta - jnp.exp(eta) - jax.scipy.special.gammaln(y + 1))~~

~~def _poisson_working_response_and_weights(y, eta):~~
~~    mu = jnp.exp(eta)~~
~~    w = jnp.clip(mu, 1e-10)~~
~~    z_tilde = eta + (y - mu) / w~~
~~    return z_tilde, w~~
~~```~~

~~These are the **only** lines in the entire GLMM stack that differ between families.~~

### ~~Step C.2: Laplace solver~~

~~**File:** `_backends/_jax.py`~~

~~**New dataclass:**~~

```python
@dataclass(frozen=True)
class LaplaceResult:
    beta: np.ndarray            # (p,) fixed effects
    u: np.ndarray               # (q,) BLUPs
    re_covariances: tuple[np.ndarray, ...]  # σ̂²·Σ_k per factor
    log_chol: np.ndarray        # optimized θ
    W: np.ndarray               # (n,) IRLS weights at convergence
    mu: np.ndarray              # (n,) fitted values at convergence
    V_inv_diag: np.ndarray      # (n,) for score projection
    fisher_info: np.ndarray     # (p, p) for one-step corrector
    converged: bool
    n_iter_outer: int
    n_iter_inner_total: int
    nll: float
```

~~**`_build_laplace_nll(X, Z, y, re_struct, working_fn, cond_nll_fn)`** (~80 lines):~~

~~Returns pure function `laplace_nll(log_chol_params) → scalar`. For given θ:~~

1. Build Γ⁻¹ from log-Cholesky params (same as `_build_reml_nll`).
2. **Inner IRLS (unrolled, fixed number of iterations for JAX tracing):**
   - Compute η = Xβ + Zu.
   - Call `working_fn(y, η)` → z̃, W.
   - Pre-scale: X̃ = √W ⊙ X, Z̃ = √W ⊙ Z, ỹ* = √W ⊙ z̃.
   - Solve unweighted Henderson on (X̃, Z̃, ỹ*) with Γ⁻¹ → (β_new, u_new).
   - Update β, u.
3. Compute Laplace NLL: $-\ell(y|\hat\beta,\hat u) + \tfrac{1}{2}\hat u^\top\Gamma^{-1}\hat u + \tfrac{1}{2}\log|H_u|$.

The inner loop is unrolled (not `lax.while_loop`) so that `jax.grad`/`jax.hessian` can differentiate through it. Fixed at `max_inner` iterations (default 10). This is standard practice for Laplace approximation — the inner loop converges in 3–8 iterations for well-conditioned problems, and extra iterations at convergence are identity operations.

~~**`_laplace_solve(X_raw, Z, y, re_struct, *, working_fn, cond_nll_fn, ...)`** (~120 lines):~~

~~1. Build `laplace_nll` via `_build_laplace_nll(...)`.~~
~~2. Call `_reml_newton_solve(laplace_nll, total_chol_params, max_iter, tol)` — **the existing Newton solver, unchanged**.~~
~~3. Post-convergence: recover β̂, û, W, μ̂, V⁻¹_diag, Fisher information.~~
~~4. Return `LaplaceResult`.~~

### ~~Step C.3: GLMM family classes~~

~~**File:** `families_mixed.py`~~

~~**`LogisticMixedFamily`** and **`PoissonMixedFamily`** — frozen dataclasses following `LinearMixedFamily` pattern (~350 lines each):~~

**Fields** (calibration state):

```python
re_struct, W, mu, V_inv_diag, fisher_info, beta, u,
re_covariances, log_chol, Z, C22, converged, n_iter,
_groups_arr, _raw_groups
```

**Protocol constants:**

| | `LogisticMixedFamily` | `PoissonMixedFamily` |
|---|---|---|
| `name` | `"logistic_mixed"` | `"poisson_mixed"` |
| `residual_type` | `"deviance"` | `"deviance"` |
| `direct_permutation` | `False` | `False` |
| `stat_label` | `"z"` | `"z"` |

**Key methods:**

- **`calibrate(X, y, fit_intercept, **kwargs)`**: Reads `groups=` from kwargs (via `FitContext` caching). Builds Z via `_build_random_effects_design()`. Calls `_laplace_solve(... working_fn=_logistic_working_response_and_weights, cond_nll_fn=_logistic_conditional_nll)`. Stores W, μ, V⁻¹_diag, Fisher info on the returned frozen instance.

- **`score_project(X, feature_idx, residuals, perm_indices, *, fit_intercept)`**: One-step corrector:

  ```python
  j = feature_idx + 1 if fit_intercept else feature_idx
  score_weights = X_full[:, j] * self.V_inv_diag      # (n,)
  E_pi = residuals[perm_indices]                        # (B, n)
  U_j = E_pi @ score_weights                            # (B,) — score
  correction = U_j / self.fisher_info[j, j]             # (B,) — one-step
  return correction
  ```

  Fully batched. O(n·B). No IRLS.

- **`validate_y(y)`**: Logistic: asserts binary y ∈ {0, 1}. Poisson: asserts non-negative integer.

- **`fit(X, y, fit_intercept)`**: Returns stored calibrated β, u.

- **`predict(model, X)`**: Logistic: `expit(X @ β̂)`. Poisson: `exp(X @ β̂)`. Marginal predictions (integrating out u).

- **`residuals(model, X, y)`**: Deviance residuals conditional on û.

- **`batch_fit()` / `batch_fit_and_score()`**: Raise `NotImplementedError` with message: "GLMM families require method='score'. Use permutation_test_regression(..., method='score')."

- **`exchangeability_cells(X, y)`**: Returns `_groups_arr` — same as `LinearMixedFamily`.

- **`diagnostics(X, y, fit_intercept)`**: Logistic: AUC, deviance, variance components, ICC. Poisson: Pseudo-R², deviance, overdispersion, variance components, ICC.

- **`display_header(diagnostics)` / `display_diagnostics(diagnostics)`**: Family-appropriate formatting.

### ~~Step C.4: Registry, exports, FitContext~~

~~**`families.py`:** Register `"logistic_mixed"` and `"poisson_mixed"` in `_FAMILIES`.~~

~~**`__init__.py`:** Export `LogisticMixedFamily`, `PoissonMixedFamily`.~~

~~**`_context.py`:** Add optional fields: `irls_weights: np.ndarray | None`, `fitted_values: np.ndarray | None`, `v_inv_diag: np.ndarray | None`, `random_effect_blups: np.ndarray | None`, `fisher_info: np.ndarray | None`.~~

~~**`engine.py`:** After calibration, populate GLMM-specific `FitContext` fields via `hasattr` checks (same pattern as existing LMM fields).~~

### ~~Step C.5: Tests~~

**Extend `tests/test_families_mixed.py`** (~350 lines):

1. **Protocol conformance** for both GLMM families.

2. **Laplace accuracy (logistic):** Simulate clustered binary data (30 groups × 20 obs). Compare β̂, τ̂² against pre-computed R `glmer()` reference values. Tolerance: β̂ within 0.05, τ̂² within 0.1.

3. **Laplace accuracy (Poisson):** Same approach with clustered count data.

4. **IRLS pre-scaling = weighted Henderson:** For fixed θ, verify one IRLS iteration with pre-scaling produces identical (β, u) to solving the weighted Henderson system directly. Machine epsilon tolerance.

5. **W = I reduces to LMM:** Gaussian data through Laplace path (with identity link, W = I) produces same β̂ as Henderson REML.

6. **One-step corrector validity:** Score projection and one-step corrector produce p-values ∈ [0, 1]. Under H₀, approximately uniform (KS test p > 0.01).

7. **One-step vs serial refit:** For small B (50 permutations), compare one-step corrector p-values with serial Laplace refit. Rank correlation ρ > 0.95.

8. **End-to-end integration:**
   - `permutation_test_regression(X, y, family="logistic_mixed", groups=g, method="score")` completes.
   - `permutation_test_regression(X, y, family="poisson_mixed", groups=g, method="score")` completes.

9. **GLMM rejects non-score strategies:** `method="ter_braak"` with `family="logistic_mixed"` → `ValueError`.

10. **Crossed design (logistic):** Subject × item binary data with `groups={"subject": ..., "item": ...}`. `method="score"` produces valid p-values.

### ~~Step C.6: PQL-Fixed Exact Permutation (`method="score_exact"`)~~

~~**Why:** The one-step corrector is second-order accurate and sufficient for most use cases, but for small n, rare events, or high ICC, it may be insufficient. PQL-fixed provides exact MLE estimates at fixed θ̂ per permutation.~~

~~**Algorithm:** Hold θ̂ from null model fixed. For each permutation, run full IRLS inner loop to convergence on y_π, vectorized across B permutations via `jax.vmap`. Each lane converges in 3–8 iterations; cost ~500× one-step but fully vectorized.~~

~~**Implementation:**~~
~~- `_pql_fixed_irls_vmap()` in `_backends/_jax.py` (~50 lines) — reuses IRLS inner loop from `_build_laplace_nll`~~
~~- `ScoreExactStrategy` in `_strategies/score.py` (~30 lines) — dispatches to PQL-fixed vmap~~
~~- Registry: `"score_exact": ScoreExactStrategy`~~
~~- Both `LogisticMixedFamily` and `PoissonMixedFamily` support `method="score_exact"`~~

~~**Tests:**~~
~~- PQL-fixed β̂ matches serial Laplace refit to 1e-4~~
~~- PQL-fixed vs one-step rank correlation ρ > 0.95~~
~~- PQL-fixed end-to-end for logistic_mixed and poisson_mixed~~
~~- PQL-fixed rejects non-GLMM families~~

### Estimated scope

| Location | Change | Lines |
|---|---|---|
| `_backends/_jax.py` | 4 NLL/IRLS functions + `_build_laplace_nll` + `_laplace_solve` + `LaplaceResult` | +280 |
| `families_mixed.py` | Two GLMM family classes | +700 |
| `families.py` | Registry entries | +5 |
| `__init__.py` | Exports | +3 |
| `_context.py` | GLMM fields | +10 |
| `engine.py` | GLMM context population | +15 |
| `tests/test_families_mixed.py` | GLMM tests | +350 |
| `_backends/_jax.py` (C.6) | PQL-fixed vmap | +50 |
| `_strategies/score.py` (C.6) | ScoreExactStrategy | +30 |
| `tests/test_families_mixed.py` (C.6) | PQL-fixed tests | +80 |
| **Total** | | **~1,520** |

### Verification gate

1. Laplace β̂ within 0.05 of `glmer()` reference for logistic and Poisson
2. IRLS pre-scaling = weighted Henderson to machine epsilon
3. W = I reduces to LMM (shared code path validation)
4. One-step corrector p-values valid and ~uniform under H₀
5. One-step rank-correlated (ρ > 0.95) with serial refit
6. End-to-end `permutation_test_regression(family="logistic_mixed", method="score")` on JAX
7. GLMM families reject non-score strategies with clear messages
8. Crossed designs produce valid exchangeability cells and permutation results
9. PQL-fixed (method="score_exact") β̂ matches serial refit to 1e-4
10. PQL-fixed end-to-end for logistic_mixed and poisson_mixed
11. `mypy src/` clean, `ruff check src/ tests/` clean, full test suite green

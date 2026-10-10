"""Empirical evaluation of GLM solver damping and permutation batch size flexibility.

Tests:
  1. Permutation count flexibility: B in [100, 500, 1000, 5000, 10000, 25000]
     - Compile time vs execution time
     - Memory scaling and whether chunking / fixed-block execution is beneficial
  2. Damping behavior:
     - Current static damping (1e-8 * I)
     - Scale-invariant relative spectral damping (delta * max(diag(H)) * I)
     - Branchless adaptive Levenberg-Marquardt (gain ratio rho via jnp.where)
  3. Stress testing under:
     - Standard well-conditioned logistic regression
     - Arbitrarily scaled features (X * 1e3, X * 1e-3)
     - Ill-conditioned collinear features (kappa(X) = 1e5)
     - Sparse / separated binary outcome draws

Usage:
    conda activate randomization-tests && python research/test_solver_damping_and_batch_flexibility.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, hessian, jit, vmap

# Float64 precision
jax.config.update("jax_enable_x64", True)

# ======================================================================
# 1. Define Candidate Solvers
# ======================================================================


def _logistic_nll(beta: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = X @ beta
    return jnp.sum(jnp.logaddexp(0.0, -logits * (2.0 * y - 1.0)))


_logistic_grad = jit(grad(_logistic_nll))
_logistic_hessian = jit(hessian(_logistic_nll))


# Candidate A: Current static damping (1e-8 * I)
def solve_static_damping(
    X: jnp.ndarray,
    y: jnp.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    min_damping: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n_params = X.shape[1]
    damping_matrix = min_damping * jnp.eye(n_params, dtype=jnp.float64)
    beta_init = jnp.zeros(n_params, dtype=jnp.float64)

    init_state = (
        jnp.array(0),
        beta_init,
        jnp.array(jnp.inf, dtype=jnp.float64),
        jnp.array(False),
    )

    def cond(state):
        i, _, _, converged = state
        return (i < max_iter) & (~converged)

    def body(state):
        i, beta, nll_prev, converged_prev = state
        g = _logistic_grad(beta, X, y)
        H = _logistic_hessian(beta, X, y)
        step = jnp.linalg.solve(H + damping_matrix, g)
        beta_step = beta - step
        beta_new = jnp.where(converged_prev, beta, beta_step)
        nll_step = _logistic_nll(beta_new, X, y)
        nll_new = jnp.where(converged_prev, nll_prev, nll_step)

        grad_small = jnp.max(jnp.abs(g)) < tol
        step_small = jnp.max(jnp.abs(step)) < tol
        nll_rel = jnp.abs(nll_new - nll_prev) / jnp.maximum(
            jnp.maximum(jnp.abs(nll_prev), jnp.abs(nll_new)), 1.0
        )
        func_small = nll_rel < tol
        converged_now = (grad_small | step_small | func_small) & jnp.all(
            jnp.isfinite(beta_new)
        )
        converged = converged_prev | converged_now
        return (i + 1, beta_new, nll_new, converged)

    n_iters, beta_final, nll_final, converged = jax.lax.while_loop(
        cond, body, init_state
    )
    return beta_final, nll_final, converged, n_iters


# Candidate B: Scale-Invariant Relative Spectral Damping
def solve_relative_spectral_damping(
    X: jnp.ndarray,
    y: jnp.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    delta: float = 1.5e-8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n_params = X.shape[1]
    beta_init = jnp.zeros(n_params, dtype=jnp.float64)

    init_state = (
        jnp.array(0),
        beta_init,
        jnp.array(jnp.inf, dtype=jnp.float64),
        jnp.array(False),
    )

    def cond(state):
        i, _, _, converged = state
        return (i < max_iter) & (~converged)

    def body(state):
        i, beta, nll_prev, converged_prev = state
        g = _logistic_grad(beta, X, y)
        H = _logistic_hessian(beta, X, y)
        # Scale-invariant diagonal floor based on local Hessian spectrum
        scale = jnp.maximum(jnp.max(jnp.diag(H)), 1e-12)
        damping_matrix = (delta * scale) * jnp.eye(n_params, dtype=jnp.float64)

        step = jnp.linalg.solve(H + damping_matrix, g)
        beta_step = beta - step
        beta_new = jnp.where(converged_prev, beta, beta_step)
        nll_step = _logistic_nll(beta_new, X, y)
        nll_new = jnp.where(converged_prev, nll_prev, nll_step)

        grad_small = jnp.max(jnp.abs(g)) < tol
        step_small = jnp.max(jnp.abs(step)) < tol
        nll_rel = jnp.abs(nll_new - nll_prev) / jnp.maximum(
            jnp.maximum(jnp.abs(nll_prev), jnp.abs(nll_new)), 1.0
        )
        func_small = nll_rel < tol
        converged_now = (grad_small | step_small | func_small) & jnp.all(
            jnp.isfinite(beta_new)
        )
        converged = converged_prev | converged_now
        return (i + 1, beta_new, nll_new, converged)

    n_iters, beta_final, nll_final, converged = jax.lax.while_loop(
        cond, body, init_state
    )
    return beta_final, nll_final, converged, n_iters


# Candidate C: Branchless Vectorized Levenberg-Marquardt (LM)
def solve_adaptive_lm(
    X: jnp.ndarray,
    y: jnp.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n_params = X.shape[1]
    beta_init = jnp.zeros(n_params, dtype=jnp.float64)

    # Initial lambda: 1e-3 * scale
    H0 = _logistic_hessian(beta_init, X, y)
    lam0 = 1e-3 * jnp.maximum(jnp.max(jnp.diag(H0)), 1.0)

    # State: (iteration, beta, nll, lam, converged)
    nll0 = _logistic_nll(beta_init, X, y)
    init_state = (
        jnp.array(0),
        beta_init,
        nll0,
        lam0,
        jnp.array(False),
    )

    def cond(state):
        i, _, _, _, converged = state
        return (i < max_iter) & (~converged)

    def body(state):
        i, beta, nll_prev, lam, converged_prev = state
        g = _logistic_grad(beta, X, y)
        H = _logistic_hessian(beta, X, y)

        scale = jnp.maximum(jnp.max(jnp.diag(H)), 1e-12)
        lam_eff = jnp.maximum(lam, 1e-12 * scale)
        H_damped = H + lam_eff * jnp.eye(n_params, dtype=jnp.float64)

        step = jnp.linalg.solve(H_damped, g)
        beta_trial = beta - step
        nll_trial = _logistic_nll(beta_trial, X, y)

        pred_reduction = 0.5 * jnp.dot(step, lam_eff * step + g)
        actual_reduction = nll_prev - nll_trial
        rho = actual_reduction / jnp.maximum(pred_reduction, 1e-16)

        step_accepted = (actual_reduction > 0.0) & jnp.isfinite(nll_trial)

        beta_next = jnp.where(step_accepted & ~converged_prev, beta_trial, beta)
        nll_next = jnp.where(step_accepted & ~converged_prev, nll_trial, nll_prev)

        # Tune lambda
        lam_down = lam * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
        lam_up = lam * 2.0
        lam_next = jnp.where(step_accepted, lam_down, lam_up)

        grad_small = jnp.max(jnp.abs(g)) < tol
        converged_now = grad_small & jnp.all(jnp.isfinite(beta_next))
        converged = converged_prev | converged_now

        return (i + 1, beta_next, nll_next, lam_next, converged)

    n_iters, beta_final, nll_final, _, converged = jax.lax.while_loop(
        cond, body, init_state
    )
    return beta_final, nll_final, converged, n_iters


# ======================================================================
# Run Benchmark Tests
# ======================================================================


def run_tests():
    print("=" * 70)
    print("EMPIRICAL TEST: SOLVER DAMPING & PERMUTATION BATCH SIZE FLEXIBILITY")
    print("=" * 70)

    np.random.seed(42)
    n = 200
    p = 5
    X_raw = np.random.randn(n, p)
    # Add intercept
    X = np.column_stack([np.ones(n), X_raw])
    true_beta = np.array([0.5, -1.0, 0.8, 0.0, 0.0, 1.2])
    logits = X @ true_beta
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (np.random.rand(n) < probs).astype(float)

    X_j = jnp.array(X, dtype=jnp.float64)
    y_j = jnp.array(y, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Test 1: Baseline Convergence on Well-Conditioned Data
    # ------------------------------------------------------------------
    print("\n--- Test 1: Single-fit baseline on well-conditioned data ---")
    b_stat, nll_stat, conv_stat, it_stat = solve_static_damping(X_j, y_j)
    b_rel, nll_rel, conv_rel, it_rel = solve_relative_spectral_damping(X_j, y_j)
    b_lm, nll_lm, conv_lm, it_lm = solve_adaptive_lm(X_j, y_j)

    print(
        f"Static damping:     iters={int(it_stat)}, NLL={float(nll_stat):.6f}, converged={bool(conv_stat)}"
    )
    print(
        f"Relative spectral:  iters={int(it_rel)}, NLL={float(nll_rel):.6f}, converged={bool(conv_rel)}"
    )
    print(
        f"Adaptive LM:        iters={int(it_lm)}, NLL={float(nll_lm):.6f}, converged={bool(conv_lm)}"
    )
    print(f"Max abs diff (Rel vs Stat): {float(jnp.max(jnp.abs(b_rel - b_stat))):.2e}")
    print(f"Max abs diff (LM vs Stat):  {float(jnp.max(jnp.abs(b_lm - b_stat))):.2e}")

    # ------------------------------------------------------------------
    # Test 2: Scale Invariance (X * 10^3)
    # ------------------------------------------------------------------
    print(
        "\n--- Test 2: Scale Invariance Stress Test (Features multiplied by 1,000) ---"
    )
    X_scaled = X.copy()
    X_scaled[:, 1:] *= 1000.0  # scale slope columns
    X_scaled_j = jnp.array(X_scaled, dtype=jnp.float64)

    b_stat_sc, _, conv_stat_sc, it_stat_sc = solve_static_damping(X_scaled_j, y_j)
    b_rel_sc, _, conv_rel_sc, it_rel_sc = solve_relative_spectral_damping(
        X_scaled_j, y_j
    )
    b_lm_sc, _, conv_lm_sc, it_lm_sc = solve_adaptive_lm(X_scaled_j, y_j)

    # Scaled parameters should be exactly true_beta[1:] / 1000
    expected_slopes = b_stat[1:] / 1000.0
    err_stat = float(jnp.max(jnp.abs(b_stat_sc[1:] - expected_slopes)))
    err_rel = float(jnp.max(jnp.abs(b_rel_sc[1:] - expected_slopes)))
    err_lm = float(jnp.max(jnp.abs(b_lm_sc[1:] - expected_slopes)))

    print(
        f"Static damping:     iters={int(it_stat_sc)}, max slope error={err_stat:.2e}, conv={bool(conv_stat_sc)}"
    )
    print(
        f"Relative spectral:  iters={int(it_rel_sc)}, max slope error={err_rel:.2e}, conv={bool(conv_rel_sc)}"
    )
    print(
        f"Adaptive LM:        iters={int(it_lm_sc)}, max slope error={err_lm:.2e}, conv={bool(conv_lm_sc)}"
    )

    # ------------------------------------------------------------------
    # Test 3: Collinear / Ill-Conditioned Design (kappa = 10^6)
    # ------------------------------------------------------------------
    print("\n--- Test 3: Severe Multicollinearity Stress Test (kappa(X) = 10^6) ---")
    X_col = X.copy()
    X_col[:, 2] = X_col[:, 1] + 1e-6 * np.random.randn(n)
    X_col_j = jnp.array(X_col, dtype=jnp.float64)

    b_stat_col, _, conv_stat_col, it_stat_col = solve_static_damping(X_col_j, y_j)
    b_rel_col, _, conv_rel_col, it_rel_col = solve_relative_spectral_damping(
        X_col_j, y_j
    )
    b_lm_col, _, conv_lm_col, it_lm_col = solve_adaptive_lm(X_col_j, y_j)

    print(
        f"Static damping:     iters={int(it_stat_col)}, conv={bool(conv_stat_col)}, beta finite={bool(jnp.all(jnp.isfinite(b_stat_col)))}"
    )
    print(
        f"Relative spectral:  iters={int(it_rel_col)}, conv={bool(conv_rel_col)}, beta finite={bool(jnp.all(jnp.isfinite(b_rel_col)))}"
    )
    print(
        f"Adaptive LM:        iters={int(it_lm_col)}, conv={bool(conv_lm_col)}, beta finite={bool(jnp.all(jnp.isfinite(b_lm_col)))}"
    )

    # ------------------------------------------------------------------
    # Test 4: Batch Permutation Scaling across B in [100, 500, 1000, 5000, 10000, 20000]
    # ------------------------------------------------------------------
    print("\n--- Test 4: Batch Permutation Scaling Across Variable B ---")

    # Generate permuted Y matrices
    B_values = [100, 500, 1000, 5000, 10000, 20000]

    # Create batch vmap solvers
    @jit
    def batch_static(Y_mat):
        def _one(y_vec):
            b, nll, c, _ = solve_static_damping(X_j, y_vec)
            return b, nll, c

        return vmap(_one)(Y_mat)

    @jit
    def batch_rel(Y_mat):
        def _one(y_vec):
            b, nll, c, _ = solve_relative_spectral_damping(X_j, y_vec)
            return b, nll, c

        return vmap(_one)(Y_mat)

    for B in B_values:
        Y_perm = np.array(
            [np.random.permutation(y) for _ in range(B)], dtype=np.float64
        )
        Y_perm_j = jnp.array(Y_perm)

        # Warmup / compile time measurement
        t0 = time.perf_counter()
        res_stat = batch_static(Y_perm_j)
        # Block until ready
        res_stat[0].block_until_ready()
        t1 = time.perf_counter()
        time_first_call = t1 - t0

        # Second call (pure execution, compiled)
        t2 = time.perf_counter()
        res_stat = batch_static(Y_perm_j)
        res_stat[0].block_until_ready()
        t3 = time.perf_counter()
        time_second_call = t3 - t2

        # Relative solver
        t4 = time.perf_counter()
        res_rel = batch_rel(Y_perm_j)
        res_rel[0].block_until_ready()
        t5 = time.perf_counter()

        n_conv = int(np.sum(np.asarray(res_stat[2])))
        max_diff = float(
            np.max(np.abs(np.asarray(res_stat[0]) - np.asarray(res_rel[0])))
        )

        print(
            f"B = {B:5d} | Compile+Run: {time_first_call:.3f}s | "
            f"Run only: {time_second_call:.3f}s | "
            f"Rel solver run: {(t5 - t4):.3f}s | "
            f"Converged: {n_conv}/{B} | Max |diff|: {max_diff:.2e}"
        )

    # ------------------------------------------------------------------
    # Test 5: Chunked / Blocked Permutation Batching
    # ------------------------------------------------------------------
    print("\n--- Test 5: Fixed-Block Chunking vs. Monolithic vmap for Large B ---")
    B_total = 25000
    block_size = 5000
    Y_perm_big = np.array(
        [np.random.permutation(y) for _ in range(B_total)], dtype=np.float64
    )

    # Monolithic solve
    t0 = time.perf_counter()
    Y_big_j = jnp.array(Y_perm_big)
    res_mono = batch_static(Y_big_j)
    res_mono[0].block_until_ready()
    t_mono = time.perf_counter() - t0

    # Chunked solve in blocks of 5,000 using the SAME compiled kernel
    # Reuse batch_static on blocks of size 5,000!
    t0 = time.perf_counter()
    chunk_results = []
    for start in range(0, B_total, block_size):
        end = min(start + block_size, B_total)
        block = jnp.array(Y_perm_big[start:end])
        # Note: if the last block is smaller than block_size, we can pad or call directly
        out = batch_static(block)
        out[0].block_until_ready()
        chunk_results.append(np.asarray(out[0]))
    t_chunk = time.perf_counter() - t0
    stacked_chunk = np.vstack(chunk_results)

    max_chunk_diff = float(np.max(np.abs(np.asarray(res_mono[0]) - stacked_chunk)))
    print(f"Total B = {B_total}:")
    print(f"  Monolithic vmap time:    {t_mono:.3f}s")
    print(f"  Chunked (blocks of {block_size}) time: {t_chunk:.3f}s")
    print(f"  Max difference between chunked and monolithic: {max_chunk_diff:.2e}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

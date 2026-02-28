"""Head-to-head comparison of REML solvers for random slopes.

Benchmarks five solver approaches against statsmodels reference:

  A. Modified Newton–Raphson (JAX autodiff, eigenvalue-clamped Hessian)
  B. EM algorithm (trace-corrected REML-EM, pure NumPy)
  C. Hybrid EM → Newton (EM for 20 iters, then switch to Newton)
  D. AI-REML (Average Information, analytical score + AI matrix)
  E. L-BFGS-B (scipy.optimize with JAX autodiff gradient)

Across five scenarios of increasing difficulty:

  1. Intercept-only (d=1) — baseline sanity check
  2. Intercept + 1 slope, moderate correlation (d=2, ρ≈0.3)
  3. Intercept + 1 slope, high correlation (d=2, ρ≈0.9)
  4. Intercept + 2 slopes (d=3)
  5. Near-zero slope variance (boundary case)
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg
import statsmodels.api as sm
import statsmodels.regression.mixed_linear_model as mlm

from randomization_tests.families_mixed import _build_random_effects_design

# ──────────────────────────────────────────────────────────────────────
# Option A: Modified Newton–Raphson via JAX autodiff
# ──────────────────────────────────────────────────────────────────────


def solve_modified_newton(
    X_raw: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    re_struct: list[tuple[int, int]],
    fit_intercept: bool = True,
    max_iter: int = 500,
    tol: float = 1e-8,
    min_damping: float = 1e-6,
) -> dict:
    """Modified Newton REML solver (Option A)."""
    import jax
    import jax.numpy as jnp
    from jax import grad, jit

    from randomization_tests._backends._jax import (
        _build_reml_nll,
        _fill_lower_triangular_np,
    )

    if fit_intercept:
        X = np.column_stack([np.ones(X_raw.shape[0]), X_raw])
    else:
        X = np.asarray(X_raw)

    n, p = X.shape
    q = Z.shape[1]

    total_chol_params = sum(d * (d + 1) // 2 for _, d in re_struct)

    X_j = jnp.array(X, dtype=jnp.float64)
    Z_j = jnp.array(Z, dtype=jnp.float64)
    y_j = jnp.array(y, dtype=jnp.float64)

    nll_fn = _build_reml_nll(X_j, Z_j, y_j, re_struct)
    grad_fn = jit(grad(nll_fn))
    hess_fn = jit(jax.hessian(nll_fn))
    nll_jit = jit(nll_fn)

    params = jnp.zeros(total_chol_params, dtype=jnp.float64)

    t0 = time.perf_counter()
    converged = False
    nll_prev = float("inf")

    for _i in range(max_iter):
        nll_val = float(nll_jit(params))
        if not np.isfinite(nll_val):
            break

        g = grad_fn(params)
        H = hess_fn(params)

        # Eigenvalue-modified Newton (Gill-Murray)
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eigvals_clamped = jnp.maximum(eigvals, min_damping)
        H_pd = eigvecs @ jnp.diag(eigvals_clamped) @ eigvecs.T
        step = jnp.linalg.solve(H_pd, g)

        # Trust-region step clamping
        max_step_norm = 5.0
        step_norm = float(jnp.linalg.norm(step))
        scale = min(1.0, max_step_norm / max(step_norm, 1e-20))
        params_new = params - scale * step

        nll_new = float(nll_jit(params_new))

        # Convergence checks
        grad_max = float(jnp.max(jnp.abs(g)))
        step_max = float(jnp.max(jnp.abs(params_new - params)))
        nll_rel = abs(nll_new - nll_prev) / max(abs(nll_prev), abs(nll_new), 1.0)

        if (grad_max < tol or step_max < tol or nll_rel < tol) and np.isfinite(nll_new):
            params = params_new
            converged = True
            break

        params = params_new
        nll_prev = nll_new

    elapsed = time.perf_counter() - t0
    n_iter = _i + 1

    # Post-convergence recovery
    params_np = np.asarray(params)
    Gamma_inv = np.zeros((q, q))
    L_factors = []
    factor_offset = 0
    theta_offset = 0
    for _k, (G_k, d_k) in enumerate(re_struct):
        n_chol_k = d_k * (d_k + 1) // 2
        theta_k = params_np[theta_offset : theta_offset + n_chol_k]
        L_k = _fill_lower_triangular_np(theta_k, d_k)
        L_factors.append(L_k)
        Sigma_k_inv = np.linalg.solve(L_k @ L_k.T, np.eye(d_k))
        block_k = np.kron(np.eye(G_k), Sigma_k_inv)
        size_k = G_k * d_k
        Gamma_inv[
            factor_offset : factor_offset + size_k,
            factor_offset : factor_offset + size_k,
        ] = block_k
        factor_offset += size_k
        theta_offset += n_chol_k

    XtX = X.T @ X
    XtZ = X.T @ Z
    Xty = X.T @ y
    Zty = Z.T @ y
    C22 = Z.T @ Z + Gamma_inv
    C22_inv_ZtX = np.linalg.solve(C22, XtZ.T)
    S = XtX - XtZ @ C22_inv_ZtX
    C22_inv_Zty = np.linalg.solve(C22, Zty)
    rhs = Xty - XtZ @ C22_inv_Zty
    beta = np.linalg.solve(S, rhs)

    w = Zty - XtZ.T @ beta
    r2 = float(y @ y) - 2 * Xty @ beta + beta @ XtX @ beta
    C22_inv_w = np.linalg.solve(C22, w)
    Q = float(r2 - w @ C22_inv_w)
    sigma2 = Q / (n - p)

    re_covs = [sigma2 * (L @ L.T) for L in L_factors]
    nll_final = float(nll_jit(params))

    return {
        "method": "Modified Newton",
        "converged": converged,
        "n_iter": n_iter,
        "nll": nll_final,
        "sigma2": sigma2,
        "re_covariances": re_covs,
        "beta": beta,
        "elapsed": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────
# Option B: EM algorithm for REML
# ──────────────────────────────────────────────────────────────────────


def solve_em(
    X_raw: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    re_struct: list[tuple[int, int]],
    fit_intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> dict:
    """EM algorithm for REML estimation (Option B).

    Uses the Henderson mixed-model equations at each iteration.
    Guaranteed monotone decrease of REML deviance.

    E-step: Solve Henderson for β̂, û given current Γ.
    M-step: Update Σ_k from û'û / G_k + trace correction.
    Profiled σ² from residual quadratic form.
    """
    if fit_intercept:
        X = np.column_stack([np.ones(X_raw.shape[0]), X_raw])
    else:
        X = np.asarray(X_raw)

    n, p = X.shape
    q = Z.shape[1]

    # Sufficient statistics
    XtX = X.T @ X
    XtZ = X.T @ Z
    ZtZ = Z.T @ Z
    Xty = X.T @ y
    Zty = Z.T @ y
    yty = float(y @ y)

    # Initialise: Σ_k = I for each factor, σ² = var(y)
    Sigma_list = [np.eye(d_k) for _, d_k in re_struct]
    sigma2 = float(np.var(y))

    t0 = time.perf_counter()
    converged = False
    nll_prev = float("inf")

    for iteration in range(max_iter):
        # Build Γ⁻¹ from current Σ_k estimates
        Gamma_inv = np.zeros((q, q))
        log_det_Gamma = 0.0
        factor_offset = 0
        for k, (G_k, d_k) in enumerate(re_struct):
            Sigma_k = Sigma_list[k]
            try:
                L_k = np.linalg.cholesky(Sigma_k)
                Sigma_k_inv = scipy.linalg.cho_solve((L_k, True), np.eye(d_k))
                log_det_Sigma_k = 2.0 * np.sum(np.log(np.diag(L_k)))
            except np.linalg.LinAlgError:
                Sigma_k_reg = Sigma_k + 1e-8 * np.eye(d_k)
                L_k = np.linalg.cholesky(Sigma_k_reg)
                Sigma_k_inv = scipy.linalg.cho_solve((L_k, True), np.eye(d_k))
                log_det_Sigma_k = 2.0 * np.sum(np.log(np.diag(L_k)))

            block_k = np.kron(np.eye(G_k), Sigma_k_inv)
            size_k = G_k * d_k
            Gamma_inv[
                factor_offset : factor_offset + size_k,
                factor_offset : factor_offset + size_k,
            ] = block_k
            log_det_Gamma += G_k * log_det_Sigma_k
            factor_offset += size_k

        # Henderson system: C₂₂ = Z'Z + Γ⁻¹
        C22 = ZtZ + Gamma_inv
        try:
            C22_chol = np.linalg.cholesky(C22)
        except np.linalg.LinAlgError:
            C22 += 1e-8 * np.eye(q)
            C22_chol = np.linalg.cholesky(C22)

        # β̂ = S⁻¹(X'y - X'Z C₂₂⁻¹ Z'y)
        C22_inv_ZtX = scipy.linalg.cho_solve((C22_chol, True), XtZ.T)
        S = XtX - XtZ @ C22_inv_ZtX
        C22_inv_Zty = scipy.linalg.cho_solve((C22_chol, True), Zty)
        rhs = Xty - XtZ @ C22_inv_Zty
        beta = np.linalg.solve(S, rhs)

        # û = Γ Z'Ṽ⁻¹(y - Xβ̂) = C₂₂⁻¹(Z'y - Z'Xβ̂) via Henderson
        w = Zty - XtZ.T @ beta
        u_hat = scipy.linalg.cho_solve((C22_chol, True), w)

        # M-step: update Σ_k and σ²
        # C₂₂⁻¹ is needed for trace corrections in both updates
        C22_inv = scipy.linalg.cho_solve((C22_chol, True), np.eye(q))

        # σ² REML-EM update with trace correction:
        #   σ̂² = [‖y - Xβ̂ - Zû‖² + σ²_old · tr(C₂₂⁻¹ Z'Z)] / (n - p)
        # The trace term accounts for uncertainty in û (BLUPs absorb
        # residual variance).
        r = y - X @ beta - Z @ u_hat
        Q = float(r @ r)
        trace_correction = sigma2 * np.trace(C22_inv @ ZtZ)
        sigma2_new = (Q + trace_correction) / (n - p)

        # Σ_k update:
        #   Σ̂_k = (1/G_k) Σⱼ [ûⱼ ûⱼ' + σ² (C₂₂⁻¹)_jj]
        # where ûⱼ is the d_k-dimensional RE vector for group j.
        factor_offset = 0
        for k, (G_k, d_k) in enumerate(re_struct):
            size_k = G_k * d_k
            u_k = u_hat[factor_offset : factor_offset + size_k].reshape(G_k, d_k)
            C22_inv_block = C22_inv[
                factor_offset : factor_offset + size_k,
                factor_offset : factor_offset + size_k,
            ]

            Sigma_new = np.zeros((d_k, d_k))
            for j in range(G_k):
                u_j = u_k[j]  # (d_k,)
                # C₂₂⁻¹ block for group j
                c22_jj = C22_inv_block[j * d_k : (j + 1) * d_k, j * d_k : (j + 1) * d_k]
                Sigma_new += np.outer(u_j, u_j) + sigma2_new * c22_jj
            Sigma_new /= G_k

            # Ensure positive definite
            eigv, eigvec = np.linalg.eigh(Sigma_new)
            eigv = np.maximum(eigv, 1e-10)
            Sigma_list[k] = eigvec @ np.diag(eigv) @ eigvec.T

            factor_offset += size_k

        sigma2 = sigma2_new

        # Compute REML NLL for convergence check
        r2 = yty - 2 * Xty @ beta + beta @ XtX @ beta
        C22_inv_w2 = scipy.linalg.cho_solve((C22_chol, True), w)
        Q_nll = float(r2 - w @ C22_inv_w2)
        log_det_C22 = 2.0 * np.sum(np.log(np.diag(C22_chol)))
        _, log_det_S = np.linalg.slogdet(S)
        nll = 0.5 * (
            (n - p) * np.log(max(Q_nll, 1e-20))
            + log_det_C22
            + log_det_Gamma
            + log_det_S
        )

        nll_rel = abs(nll - nll_prev) / max(abs(nll_prev), abs(nll), 1.0)
        if nll_rel < tol and iteration > 0:
            converged = True
            break
        nll_prev = nll

    elapsed = time.perf_counter() - t0
    re_covs = [sigma2 * Sigma_list[k] for k, _ in enumerate(re_struct)]

    return {
        "method": "EM",
        "converged": converged,
        "n_iter": iteration + 1,
        "nll": nll,
        "sigma2": sigma2,
        "re_covariances": re_covs,
        "beta": beta,
        "elapsed": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────
# Option C: Hybrid EM → Newton (EM warm-start, Newton finish)
# ──────────────────────────────────────────────────────────────────────


def solve_hybrid(
    X_raw: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    re_struct: list[tuple[int, int]],
    fit_intercept: bool = True,
    em_iters: int = 30,
    newton_max_iter: int = 200,
    tol: float = 1e-8,
    min_damping: float = 1e-6,
) -> dict:
    """Hybrid EM → Newton REML solver (Option C).

    Runs EM for em_iters to reach the basin of convergence,
    extracts the log-Cholesky parameters from the EM Σ_k estimates,
    then switches to Newton–Raphson with JAX autodiff for quadratic
    convergence to the exact optimum.
    """
    import jax
    import jax.numpy as jnp
    from jax import grad, jit

    from randomization_tests._backends._jax import (
        _build_reml_nll,
        _fill_lower_triangular_np,
    )

    if fit_intercept:
        X = np.column_stack([np.ones(X_raw.shape[0]), X_raw])
    else:
        X = np.asarray(X_raw)

    n, p = X.shape
    q = Z.shape[1]

    sum(d * (d + 1) // 2 for _, d in re_struct)

    # ── Phase 1: EM warm-start ──
    # Sufficient statistics
    XtX = X.T @ X
    XtZ = X.T @ Z
    ZtZ = Z.T @ Z
    Xty = X.T @ y
    Zty = Z.T @ y

    Sigma_list = [np.eye(d_k) for _, d_k in re_struct]
    sigma2 = float(np.var(y))

    t0 = time.perf_counter()

    for _ in range(em_iters):
        Gamma_inv = np.zeros((q, q))
        factor_offset = 0
        for k, (G_k, d_k) in enumerate(re_struct):
            Sigma_k = Sigma_list[k]
            try:
                L_k = np.linalg.cholesky(Sigma_k)
                Sigma_k_inv = scipy.linalg.cho_solve((L_k, True), np.eye(d_k))
            except np.linalg.LinAlgError:
                Sigma_k_reg = Sigma_k + 1e-8 * np.eye(d_k)
                L_k = np.linalg.cholesky(Sigma_k_reg)
                Sigma_k_inv = scipy.linalg.cho_solve((L_k, True), np.eye(d_k))
            block_k = np.kron(np.eye(G_k), Sigma_k_inv)
            size_k = G_k * d_k
            Gamma_inv[
                factor_offset : factor_offset + size_k,
                factor_offset : factor_offset + size_k,
            ] = block_k
            factor_offset += size_k

        C22 = ZtZ + Gamma_inv
        try:
            C22_chol = np.linalg.cholesky(C22)
        except np.linalg.LinAlgError:
            C22 += 1e-8 * np.eye(q)
            C22_chol = np.linalg.cholesky(C22)

        C22_inv_ZtX = scipy.linalg.cho_solve((C22_chol, True), XtZ.T)
        S = XtX - XtZ @ C22_inv_ZtX
        C22_inv_Zty = scipy.linalg.cho_solve((C22_chol, True), Zty)
        rhs = Xty - XtZ @ C22_inv_Zty
        beta = np.linalg.solve(S, rhs)

        w = Zty - XtZ.T @ beta
        u_hat = scipy.linalg.cho_solve((C22_chol, True), w)

        C22_inv = scipy.linalg.cho_solve((C22_chol, True), np.eye(q))
        r = y - X @ beta - Z @ u_hat
        Q_em = float(r @ r)
        trace_corr = sigma2 * np.trace(C22_inv @ ZtZ)
        sigma2 = (Q_em + trace_corr) / (n - p)

        factor_offset = 0
        for k, (G_k, d_k) in enumerate(re_struct):
            size_k = G_k * d_k
            u_k = u_hat[factor_offset : factor_offset + size_k].reshape(G_k, d_k)
            C22_inv_block = C22_inv[
                factor_offset : factor_offset + size_k,
                factor_offset : factor_offset + size_k,
            ]
            Sigma_new = np.zeros((d_k, d_k))
            for j in range(G_k):
                u_j = u_k[j]
                c22_jj = C22_inv_block[j * d_k : (j + 1) * d_k, j * d_k : (j + 1) * d_k]
                Sigma_new += np.outer(u_j, u_j) + sigma2 * c22_jj
            Sigma_new /= G_k
            eigv, eigvec = np.linalg.eigh(Sigma_new)
            eigv = np.maximum(eigv, 1e-10)
            Sigma_list[k] = eigvec @ np.diag(eigv) @ eigvec.T
            factor_offset += size_k

    # ── Extract log-Cholesky init from EM Σ_k ──
    init_params = []
    for k, (_G_k, d_k) in enumerate(re_struct):
        Sigma_k = Sigma_list[k]
        ratio_k = Sigma_k / max(sigma2, 1e-20)
        try:
            L_k = np.linalg.cholesky(ratio_k)
        except np.linalg.LinAlgError:
            L_k = np.eye(d_k)
        for i in range(d_k):
            for j in range(i):
                init_params.append(L_k[i, j])
            init_params.append(np.log(max(L_k[i, i], 1e-20)))
    init_params = np.array(init_params)

    # ── Phase 2: Newton with JAX autodiff ──
    X_j = jnp.array(X, dtype=jnp.float64)
    Z_j = jnp.array(Z, dtype=jnp.float64)
    y_j = jnp.array(y, dtype=jnp.float64)

    nll_fn = _build_reml_nll(X_j, Z_j, y_j, re_struct)
    grad_fn = jit(grad(nll_fn))
    hess_fn = jit(jax.hessian(nll_fn))
    nll_jit = jit(nll_fn)

    params = jnp.array(init_params, dtype=jnp.float64)
    converged = False
    nll_prev = float(nll_jit(params))
    total_iter = em_iters

    for i in range(newton_max_iter):
        g = grad_fn(params)
        H = hess_fn(params)

        # Eigenvalue clamping (should be PSD near optimum after EM)
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eigvals_clamped = jnp.maximum(eigvals, min_damping)
        H_pd = eigvecs @ jnp.diag(eigvals_clamped) @ eigvecs.T
        step = jnp.linalg.solve(H_pd, g)

        # Trust-region
        max_step_norm = 5.0
        step_norm = float(jnp.linalg.norm(step))
        scale = min(1.0, max_step_norm / max(step_norm, 1e-20))
        params_new = params - scale * step
        nll_new = float(nll_jit(params_new))

        grad_max = float(jnp.max(jnp.abs(g)))
        step_max = float(jnp.max(jnp.abs(params_new - params)))
        nll_rel = abs(nll_new - nll_prev) / max(abs(nll_prev), abs(nll_new), 1.0)

        if (grad_max < tol or step_max < tol or nll_rel < tol) and np.isfinite(nll_new):
            params = params_new
            converged = True
            total_iter += i + 1
            break

        params = params_new
        nll_prev = nll_new
    else:
        total_iter += newton_max_iter

    elapsed = time.perf_counter() - t0

    # Post-convergence recovery
    params_np = np.asarray(params)
    Gamma_inv = np.zeros((q, q))
    L_factors = []
    factor_offset = 0
    theta_offset = 0
    for _k, (G_k, d_k) in enumerate(re_struct):
        n_chol_k = d_k * (d_k + 1) // 2
        theta_k = params_np[theta_offset : theta_offset + n_chol_k]
        L_k = _fill_lower_triangular_np(theta_k, d_k)
        L_factors.append(L_k)
        Sigma_k_inv = np.linalg.solve(L_k @ L_k.T, np.eye(d_k))
        block_k = np.kron(np.eye(G_k), Sigma_k_inv)
        size_k = G_k * d_k
        Gamma_inv[
            factor_offset : factor_offset + size_k,
            factor_offset : factor_offset + size_k,
        ] = block_k
        factor_offset += size_k
        theta_offset += n_chol_k

    C22 = ZtZ + Gamma_inv
    C22_inv_ZtX = np.linalg.solve(C22, XtZ.T)
    S = XtX - XtZ @ C22_inv_ZtX
    C22_inv_Zty = np.linalg.solve(C22, Zty)
    rhs_final = Xty - XtZ @ C22_inv_Zty
    beta_final = np.linalg.solve(S, rhs_final)

    w_final = Zty - XtZ.T @ beta_final
    r2 = float(y @ y) - 2 * Xty @ beta_final + beta_final @ XtX @ beta_final
    C22_inv_w = np.linalg.solve(C22, w_final)
    Q_final = float(r2 - w_final @ C22_inv_w)
    sigma2_final = Q_final / (n - p)

    re_covs = [sigma2_final * (L @ L.T) for L in L_factors]
    nll_final = float(nll_jit(params))

    return {
        "method": "Hybrid EM→Newton",
        "converged": converged,
        "n_iter": total_iter,
        "nll": nll_final,
        "sigma2": sigma2_final,
        "re_covariances": re_covs,
        "beta": beta_final,
        "elapsed": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────
# Option D: AI-REML (Average Information, analytical)
# ──────────────────────────────────────────────────────────────────────


def _build_gamma_inv_and_derivs(
    re_struct: list[tuple[int, int]],
    Sigma_list: list[np.ndarray],
    q: int,
) -> tuple[np.ndarray, float, list[np.ndarray]]:
    """Build Γ⁻¹, log|Γ|, and ∂Γ/∂θ_m for each free parameter.

    Each Σ_k is parameterised directly (not via Cholesky). The free
    parameters are the lower-triangular elements of each Σ_k, with
    diagonal elements on a log scale for positivity.

    Returns:
        Gamma_inv: (q, q) block-diagonal inverse.
        log_det_Gamma: scalar log|Γ|.
        dGamma_list: List of (q, q) matrices ∂Γ/∂Σ_k[i,j] for each
            unique lower-triangular element.
    """
    Gamma_inv = np.zeros((q, q))
    log_det_Gamma = 0.0
    dGamma_list: list[np.ndarray] = []

    factor_offset = 0
    for k, (G_k, d_k) in enumerate(re_struct):
        Sigma_k = Sigma_list[k]
        try:
            L_k = np.linalg.cholesky(Sigma_k)
            Sigma_k_inv = scipy.linalg.cho_solve((L_k, True), np.eye(d_k))
            log_det_Sk = 2.0 * np.sum(np.log(np.diag(L_k)))
        except np.linalg.LinAlgError:
            Sigma_k_reg = Sigma_k + 1e-8 * np.eye(d_k)
            L_k = np.linalg.cholesky(Sigma_k_reg)
            Sigma_k_inv = scipy.linalg.cho_solve((L_k, True), np.eye(d_k))
            log_det_Sk = 2.0 * np.sum(np.log(np.diag(L_k)))

        block_k = np.kron(np.eye(G_k), Sigma_k_inv)
        size_k = G_k * d_k
        Gamma_inv[
            factor_offset : factor_offset + size_k,
            factor_offset : factor_offset + size_k,
        ] = block_k
        log_det_Gamma += G_k * log_det_Sk

        # ∂Γ/∂Σ_k[i,j] — each unique lower-tri element of Σ_k
        for i in range(d_k):
            for j in range(i + 1):
                E_ij = np.zeros((d_k, d_k))
                if i == j:
                    E_ij[i, i] = 1.0
                else:
                    E_ij[i, j] = 1.0
                    E_ij[j, i] = 1.0
                dGamma_m = np.zeros((q, q))
                dGamma_block = np.kron(np.eye(G_k), E_ij)
                dGamma_m[
                    factor_offset : factor_offset + size_k,
                    factor_offset : factor_offset + size_k,
                ] = dGamma_block
                dGamma_list.append(dGamma_m)

        factor_offset += size_k

    return Gamma_inv, log_det_Gamma, dGamma_list


def solve_ai_reml(
    X_raw: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    re_struct: list[tuple[int, int]],
    fit_intercept: bool = True,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict:
    """AI-REML solver with analytical score and Average Information (Option D).

    The Average Information matrix AI[m,n] = ½ (Py)' (∂V/∂θ_m) P (∂V/∂θ_n) (Py)
    is always PSD by construction. The score vector is computed analytically
    from Henderson quantities. No autodiff required.

    Parameterisation: direct elements of Σ_k (lower-triangular, with σ²
    profiled out).  After convergence, converts to log-Cholesky for the
    returned result.
    """
    if fit_intercept:
        X = np.column_stack([np.ones(X_raw.shape[0]), X_raw])
    else:
        X = np.asarray(X_raw)

    n, p = X.shape
    q = Z.shape[1]

    XtX = X.T @ X
    XtZ = X.T @ Z
    ZtZ = Z.T @ Z
    Xty = X.T @ y
    Zty = Z.T @ y
    float(y @ y)

    # Number of free parameters (lower-tri of each Σ_k)
    n_params = sum(d_k * (d_k + 1) // 2 for _, d_k in re_struct)

    # Init Σ_k = I
    Sigma_list = [np.eye(d_k) for _, d_k in re_struct]

    t0 = time.perf_counter()
    converged = False

    for iteration in range(max_iter):
        Gamma_inv, log_det_Gamma, dGamma_list = _build_gamma_inv_and_derivs(
            re_struct, Sigma_list, q
        )

        C22 = ZtZ + Gamma_inv
        try:
            C22_chol = np.linalg.cholesky(C22)
        except np.linalg.LinAlgError:
            C22 += 1e-8 * np.eye(q)
            C22_chol = np.linalg.cholesky(C22)

        C22_inv_ZtX = scipy.linalg.cho_solve((C22_chol, True), XtZ.T)
        S = XtX - XtZ @ C22_inv_ZtX
        C22_inv_Zty = scipy.linalg.cho_solve((C22_chol, True), Zty)
        rhs = Xty - XtZ @ C22_inv_Zty
        beta = np.linalg.solve(S, rhs)

        # P = Ṽ⁻¹ - Ṽ⁻¹X S⁻¹ X'Ṽ⁻¹  (not built explicitly)
        # Py = Ṽ⁻¹(y - Xβ̂) via Woodbury
        r = y - X @ beta
        w = Z.T @ r  # Z'r
        C22_inv_w = scipy.linalg.cho_solve((C22_chol, True), w)
        Py = r - Z @ C22_inv_w  # Ṽ⁻¹r = r - Z C₂₂⁻¹ Z'r

        # Profiled σ² = y'Py / (n-p)
        sigma2 = float(y @ Py) / (n - p)

        # Score vector: s_m = -½ tr(PV_m) + ½ (Py)'V_m(Py) / σ²
        # where V_m = Z (∂Γ/∂θ_m) Z'  (the m-th derivative of Ṽ)
        # tr(PV_m) = tr(P Z dΓ_m Z') = tr(Z'P Z · dΓ_m)
        # But P involves Ṽ⁻¹ which is implicit. Use:
        #   tr(PV_m) = tr(dΓ_m) - tr(C₂₂⁻¹ dΓ_m Γ⁻¹ ... )
        # Simpler: compute ZtPZ = Z'Ṽ⁻¹Z - Z'Ṽ⁻¹X S⁻¹ X'Ṽ⁻¹Z
        C22_inv = scipy.linalg.cho_solve((C22_chol, True), np.eye(q))
        scipy.linalg.cho_solve((C22_chol, True), Z.T)

        # Z'PZ = Z'Ṽ⁻¹Z - Z'Ṽ⁻¹X S⁻¹ X'Ṽ⁻¹Z
        # Z'Ṽ⁻¹ = Z' - Z'Z C₂₂⁻¹ Z'  ... this is getting complex.
        # Simpler approach: ZtPy = Z'Py, and compute score/AI from Py directly.

        ZtPy = Z.T @ Py  # (q,)

        score = np.zeros(n_params)
        for m, dGamma_m in enumerate(dGamma_list):
            # ∂V/∂θ_m = Z dΓ_m Z', so (Py)' V_m (Py) = (Z'Py)' dΓ_m (Z'Py)
            quad = float(ZtPy @ dGamma_m @ ZtPy)

            # tr(P V_m) = tr((Ṽ⁻¹ - Ṽ⁻¹X S⁻¹ X'Ṽ⁻¹) Z dΓ_m Z')
            # = tr(dΓ_m Z'PZ)
            # Z'PZ = Z'Ṽ⁻¹Z - (Z'Ṽ⁻¹X) S⁻¹ (X'Ṽ⁻¹Z)
            # Z'Ṽ⁻¹Z = ZtZ - ZtZ C₂₂⁻¹ ZtZ  ... use Woodbury
            # Actually: Z'Ṽ⁻¹ = C₂₂⁻¹ Z' Gamma_inv  ... no.
            # Let's use: Ṽ⁻¹ = I - Z C₂₂⁻¹ Z' (Woodbury)
            # Z'Ṽ⁻¹Z = ZtZ - ZtZ C₂₂⁻¹ ZtZ
            ZtVinvZ = ZtZ - ZtZ @ C22_inv @ ZtZ
            ZtVinvX = XtZ.T - ZtZ @ C22_inv @ XtZ.T
            S_inv = np.linalg.solve(S, np.eye(p))
            ZtPZ = ZtVinvZ - ZtVinvX @ S_inv @ ZtVinvX.T

            trace_PVm = np.trace(dGamma_m @ ZtPZ)
            score[m] = -0.5 * trace_PVm + 0.5 * quad / max(sigma2, 1e-20)

        # AI matrix: AI[m,n] = ½ (Py)' V_m P V_n (Py) / σ²
        # ≈ ½ (Z'Py)' dΓ_m (Z'PZ) dΓ_n (Z'Py) / σ²
        # But the standard AI approximation simplifies to:
        # AI[m,n] = ½ (Py)' (∂V/∂θ_m) P (∂V/∂θ_n) (Py)
        # = ½ (Z'Py)' dΓ_m ZtPZ dΓ_n (Z'Py)
        AI = np.zeros((n_params, n_params))
        # Precompute dΓ_m @ ZtPy and dΓ_m @ ZtPZ
        dG_ZtPy = [dG @ ZtPy for dG in dGamma_list]
        dG_ZtPZ = [dG @ ZtPZ for dG in dGamma_list]
        for m in range(n_params):
            for nn in range(m, n_params):
                # AI[m,n] = ½ (ZtPy)' dΓ_m ZtPZ dΓ_n (ZtPy)
                val = 0.5 * float(dG_ZtPy[m] @ dG_ZtPZ[nn] @ ZtPy)
                AI[m, nn] = val
                AI[nn, m] = val

        # Ensure PSD
        eigvals, eigvecs = np.linalg.eigh(AI)
        eigvals = np.maximum(eigvals, 1e-8)
        AI_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Newton step: Δθ = AI⁻¹ score
        try:
            delta = np.linalg.solve(AI_pd, score)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(AI_pd, score, rcond=None)[0]

        # Clamp step
        max_step = 2.0
        step_norm = np.linalg.norm(delta)
        if step_norm > max_step:
            delta *= max_step / step_norm

        # Update Σ_k elements
        param_offset = 0
        for _k_idx, (_G_k, d_k) in enumerate(re_struct):
            Sigma_new = Sigma_list[_k_idx].copy()
            for i in range(d_k):
                for j in range(i + 1):
                    Sigma_new[i, j] += delta[param_offset]
                    if i != j:
                        Sigma_new[j, i] += delta[param_offset]
                    param_offset += 1
            # Ensure PSD
            eigv, eigvec = np.linalg.eigh(Sigma_new)
            eigv = np.maximum(eigv, 1e-10)
            Sigma_list[_k_idx] = eigvec @ np.diag(eigv) @ eigvec.T

        # Convergence
        if np.max(np.abs(score)) < tol and iteration > 0:
            converged = True
            break

    elapsed = time.perf_counter() - t0

    # Convert to log-Cholesky for output compatibility
    re_covs = [sigma2 * Sigma_list[k] for k, _ in enumerate(re_struct)]

    # Build Γ⁻¹ and recover projection
    Gamma_inv_final = np.zeros((q, q))
    factor_offset = 0
    for k, (G_k, d_k) in enumerate(re_struct):
        Sigma_k_inv = np.linalg.solve(Sigma_list[k], np.eye(d_k))
        block_k = np.kron(np.eye(G_k), Sigma_k_inv)
        size_k = G_k * d_k
        Gamma_inv_final[
            factor_offset : factor_offset + size_k,
            factor_offset : factor_offset + size_k,
        ] = block_k
        factor_offset += size_k

    C22_final = ZtZ + Gamma_inv_final
    C22_inv_ZtX_f = np.linalg.solve(C22_final, XtZ.T)
    S_f = XtX - XtZ @ C22_inv_ZtX_f
    C22_inv_Zty_f = np.linalg.solve(C22_final, Zty)
    rhs_f = Xty - XtZ @ C22_inv_Zty_f
    beta_final = np.linalg.solve(S_f, rhs_f)

    w_f = Zty - XtZ.T @ beta_final
    r2_f = float(y @ y) - 2 * Xty @ beta_final + beta_final @ XtX @ beta_final
    C22_inv_w_f = np.linalg.solve(C22_final, w_f)
    Q_f = float(r2_f - w_f @ C22_inv_w_f)
    sigma2_final = Q_f / (n - p)

    re_covs = [sigma2_final * Sigma_list[k] for k, _ in enumerate(re_struct)]

    # REML NLL
    log_det_C22_f = float(np.linalg.slogdet(C22_final)[1])
    _, log_det_S_f = np.linalg.slogdet(S_f)
    _, log_det_Gamma_f = 0.0, 0.0
    factor_offset = 0
    log_det_Gamma_f = 0.0
    for _k, (G_k, _d_k) in enumerate(re_struct):
        _, ld = np.linalg.slogdet(Sigma_list[_k])
        log_det_Gamma_f += G_k * ld
    nll_final = 0.5 * (
        (n - p) * np.log(max(Q_f, 1e-20))
        + log_det_C22_f
        + log_det_Gamma_f
        + log_det_S_f
    )

    return {
        "method": "AI-REML",
        "converged": converged,
        "n_iter": iteration + 1,
        "nll": nll_final,
        "sigma2": sigma2_final,
        "re_covariances": re_covs,
        "beta": beta_final,
        "elapsed": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────
# Option E: L-BFGS-B (scipy.optimize + JAX autodiff gradient)
# ──────────────────────────────────────────────────────────────────────


def solve_lbfgsb(
    X_raw: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    re_struct: list[tuple[int, int]],
    fit_intercept: bool = True,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> dict:
    """L-BFGS-B solver with JAX autodiff gradient (Option E).

    Uses the same _build_reml_nll + jax.grad as the Newton solver,
    but delegates optimisation to scipy's L-BFGS-B which maintains
    its own PSD quasi-Newton Hessian approximation internally.

    No Hessian computation, no eigenvalue clamping. Just exact
    gradient + scipy quasi-Newton. Should be robust and precise.
    """
    import jax.numpy as jnp
    from jax import grad, jit
    from scipy.optimize import minimize

    from randomization_tests._backends._jax import (
        _build_reml_nll,
        _fill_lower_triangular_np,
    )

    if fit_intercept:
        X = np.column_stack([np.ones(X_raw.shape[0]), X_raw])
    else:
        X = np.asarray(X_raw)

    n, p = X.shape
    q = Z.shape[1]

    total_chol_params = sum(d * (d + 1) // 2 for _, d in re_struct)

    X_j = jnp.array(X, dtype=jnp.float64)
    Z_j = jnp.array(Z, dtype=jnp.float64)
    y_j = jnp.array(y, dtype=jnp.float64)

    nll_fn = _build_reml_nll(X_j, Z_j, y_j, re_struct)
    nll_jit = jit(nll_fn)
    grad_jit = jit(grad(nll_fn))

    n_feval = [0]  # mutable counter

    def objective(params_np):
        n_feval[0] += 1
        p_jax = jnp.array(params_np, dtype=jnp.float64)
        val = float(nll_jit(p_jax))
        g = np.asarray(grad_jit(p_jax), dtype=np.float64)
        return val, g

    x0 = np.zeros(total_chol_params)

    t0 = time.perf_counter()
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )
    elapsed = time.perf_counter() - t0

    params_np = result.x
    converged = result.success

    # Post-convergence recovery
    Gamma_inv = np.zeros((q, q))
    L_factors = []
    factor_offset = 0
    theta_offset = 0
    for _k, (G_k, d_k) in enumerate(re_struct):
        n_chol_k = d_k * (d_k + 1) // 2
        theta_k = params_np[theta_offset : theta_offset + n_chol_k]
        L_k = _fill_lower_triangular_np(theta_k, d_k)
        L_factors.append(L_k)
        Sigma_k_inv = np.linalg.solve(L_k @ L_k.T, np.eye(d_k))
        block_k = np.kron(np.eye(G_k), Sigma_k_inv)
        size_k = G_k * d_k
        Gamma_inv[
            factor_offset : factor_offset + size_k,
            factor_offset : factor_offset + size_k,
        ] = block_k
        factor_offset += size_k
        theta_offset += n_chol_k

    XtX = X.T @ X
    XtZ = X.T @ Z
    Xty = X.T @ y
    Zty = Z.T @ y
    C22 = Z.T @ Z + Gamma_inv
    C22_inv_ZtX = np.linalg.solve(C22, XtZ.T)
    S = XtX - XtZ @ C22_inv_ZtX
    C22_inv_Zty = np.linalg.solve(C22, Zty)
    rhs = Xty - XtZ @ C22_inv_Zty
    beta = np.linalg.solve(S, rhs)

    w = Zty - XtZ.T @ beta
    r2 = float(y @ y) - 2 * Xty @ beta + beta @ XtX @ beta
    C22_inv_w = np.linalg.solve(C22, w)
    Q = float(r2 - w @ C22_inv_w)
    sigma2 = Q / (n - p)

    re_covs = [sigma2 * (L @ L.T) for L in L_factors]
    nll_final = float(nll_jit(jnp.array(params_np, dtype=jnp.float64)))

    return {
        "method": "L-BFGS-B",
        "converged": converged,
        "n_iter": result.nit,
        "nll": nll_final,
        "sigma2": sigma2,
        "re_covariances": re_covs,
        "beta": beta,
        "elapsed": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────
# Reference: statsmodels MixedLM
# ──────────────────────────────────────────────────────────────────────


def solve_statsmodels(
    X_raw: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    re_struct: list[tuple[int, int]],
    fit_intercept: bool = True,
    random_slopes: list[int] | None = None,
) -> dict:
    """statsmodels reference solution."""
    import warnings

    X_sm = sm.add_constant(X_raw) if fit_intercept else np.asarray(X_raw)

    exog_re_kw = {}
    if random_slopes:
        exog_re_cols = [np.ones(len(y))]
        for idx in random_slopes:
            exog_re_cols.append(X_raw[:, idx])
        exog_re_kw["exog_re"] = np.column_stack(exog_re_cols)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = mlm.MixedLM(y, X_sm, groups=groups, **exog_re_kw).fit(reml=True, disp=0)
    elapsed = time.perf_counter() - t0

    beta = np.asarray(model.fe_params)
    sigma2 = float(model.scale)
    cov_re = np.atleast_2d(np.asarray(model.cov_re))

    return {
        "method": "statsmodels",
        "converged": model.converged,
        "n_iter": 0,
        "nll": float(-model.llf),
        "sigma2": sigma2,
        "re_covariances": [cov_re],
        "beta": beta,
        "elapsed": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────
# Test scenarios
# ──────────────────────────────────────────────────────────────────────


def make_scenario(
    name: str,
    n_groups: int,
    n_per: int,
    p: int,
    beta_true: np.ndarray,
    Sigma_true: np.ndarray,
    sigma_e: float,
    random_slopes: list[int] | None,
    seed: int = 42,
) -> dict:
    """Generate a scenario dataset."""
    rng = np.random.default_rng(seed)
    n = n_groups * n_per
    groups = np.repeat(np.arange(n_groups), n_per)
    X = rng.standard_normal((n, p))

    d = Sigma_true.shape[0]
    L_true = np.linalg.cholesky(Sigma_true)
    u = rng.standard_normal((n_groups, d)) @ L_true.T

    y = X @ beta_true + u[groups, 0]  # random intercept
    for s_idx, col_idx in enumerate(random_slopes or []):
        y += u[groups, 1 + s_idx] * X[:, col_idx]
    y += rng.normal(0, sigma_e, n)

    Z, re_struct = _build_random_effects_design(
        groups, X=X, random_slopes=random_slopes
    )

    return {
        "name": name,
        "X": X,
        "y": y,
        "Z": Z,
        "groups": groups,
        "re_struct": re_struct,
        "random_slopes": random_slopes,
        "Sigma_true": Sigma_true,
        "beta_true": beta_true,
        "sigma_e_true": sigma_e,
    }


def print_result(result: dict, ref: dict | None = None) -> None:
    """Pretty-print a solver result."""
    print(
        f"  {result['method']:20s}: "
        f"conv={result['converged']!s:5s}  "
        f"iter={result['n_iter']:4d}  "
        f"NLL={result['nll']:12.4f}  "
        f"σ²={result['sigma2']:.4f}  "
        f"time={result['elapsed']:.3f}s"
    )
    for k, cov in enumerate(result["re_covariances"]):
        d = cov.shape[0]
        if d == 1:
            print(f"    Σ̂_{k}: τ²={cov[0, 0]:.4f}")
        else:
            print(f"    Σ̂_{k}:")
            for row in cov:
                print(f"      [{', '.join(f'{v:8.4f}' for v in row)}]")
    print(f"    β̂ = [{', '.join(f'{v:.4f}' for v in result['beta'])}]")

    if ref is not None:
        sigma2_err = abs(result["sigma2"] - ref["sigma2"]) / max(
            abs(ref["sigma2"]), 1e-10
        )
        beta_err = np.linalg.norm(result["beta"] - ref["beta"]) / max(
            np.linalg.norm(ref["beta"]), 1e-10
        )
        cov_errs = []
        for c1, c2 in zip(
            result["re_covariances"], ref["re_covariances"], strict=False
        ):
            cov_errs.append(np.linalg.norm(c1 - c2) / max(np.linalg.norm(c2), 1e-10))
        print(
            f"    vs ref: σ²_err={sigma2_err:.2e}  β_err={beta_err:.2e}  "
            f"Σ_err={[f'{e:.2e}' for e in cov_errs]}"
        )


def main() -> None:
    scenarios = [
        make_scenario(
            name="1. Intercept-only (d=1)",
            n_groups=10,
            n_per=20,
            p=2,
            beta_true=np.array([2.0, -1.0]),
            Sigma_true=np.array([[4.0]]),
            sigma_e=1.0,
            random_slopes=None,
        ),
        make_scenario(
            name="2. Slope, moderate ρ (d=2)",
            n_groups=10,
            n_per=20,
            p=2,
            beta_true=np.array([2.0, -1.0]),
            Sigma_true=np.array([[4.0, 0.5], [0.5, 0.8]]),
            sigma_e=1.0,
            random_slopes=[0],
        ),
        make_scenario(
            name="3. Slope, high ρ (d=2)",
            n_groups=10,
            n_per=20,
            p=2,
            beta_true=np.array([2.0, -1.0]),
            Sigma_true=np.array([[4.0, 1.7], [1.7, 0.8]]),
            sigma_e=1.0,
            random_slopes=[0],
        ),
        make_scenario(
            name="4. Two slopes (d=3)",
            n_groups=15,
            n_per=20,
            p=3,
            beta_true=np.array([1.5, -0.5, 0.8]),
            Sigma_true=np.array(
                [
                    [2.0, 0.3, 0.1],
                    [0.3, 0.5, 0.05],
                    [0.1, 0.05, 0.3],
                ]
            ),
            sigma_e=0.8,
            random_slopes=[0, 1],
        ),
        make_scenario(
            name="5. Near-zero slope var (boundary)",
            n_groups=10,
            n_per=25,
            p=2,
            beta_true=np.array([2.0, -1.0]),
            Sigma_true=np.array([[4.0, 0.01], [0.01, 0.001]]),
            sigma_e=1.0,
            random_slopes=[0],
        ),
    ]

    for sc in scenarios:
        print(f"\n{'=' * 70}")
        print(f"Scenario: {sc['name']}")
        print(
            f"  n={len(sc['y'])}, p={sc['X'].shape[1]}, "
            f"re_struct={sc['re_struct']}, "
            f"Σ_true={sc['Sigma_true'].tolist()}"
        )
        print(f"{'=' * 70}")

        ref = solve_statsmodels(
            sc["X"],
            sc["Z"],
            sc["y"],
            sc["groups"],
            sc["re_struct"],
            random_slopes=sc["random_slopes"],
        )
        print_result(ref)

        result_a = solve_modified_newton(sc["X"], sc["Z"], sc["y"], sc["re_struct"])
        print_result(result_a, ref)

        result_b = solve_em(sc["X"], sc["Z"], sc["y"], sc["re_struct"])
        print_result(result_b, ref)

        result_c = solve_hybrid(sc["X"], sc["Z"], sc["y"], sc["re_struct"])
        print_result(result_c, ref)

        result_d = solve_ai_reml(sc["X"], sc["Z"], sc["y"], sc["re_struct"])
        print_result(result_d, ref)

        result_e = solve_lbfgsb(sc["X"], sc["Z"], sc["y"], sc["re_struct"])
        print_result(result_e, ref)


if __name__ == "__main__":
    main()

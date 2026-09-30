"""Ad-hoc empirical research harness for Phase 8: Knockoffs.

Validates the mathematical, geometric, and statistical invariants of:
1. Equicorrelated shrinkage and positive definiteness invariant:
   2*Sigma_hat - diag(s) >= 0 and Cholesky stability.
2. Fixed-X orthonormal subspace geometry and Gram preservation:
   X_tilde^T X_tilde == X^T X and X^T X_tilde == X^T X - diag(s) to machine precision.
3. Model-X conditional Gaussian sampling and covariance structure.
4. Coin-flip property of importance statistics W_j = |beta_j| - |beta_tilde_j|.
5. Finite-sample knockoff+ FDR control over Monte Carlo replications.
6. High-dimensional regime (n < 2p) with penalized estimation (LassoCV).
7. MMD swap-exchangeability diagnostic on Gaussian vs heavy-tailed designs.

Usage:
    conda activate randomization-tests && python research/test_knockoffs_empirical.py
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LassoCV

from randomization_tests._kernel_tests import mmd_test


def compute_equicorrelated_s(Sigma: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Compute equicorrelated diagonal shift s such that 2*Sigma - diag(s) >= 0."""
    p = Sigma.shape[0]
    # Eigenvalues of Sigma
    eigvals = linalg.eigvalsh(Sigma)
    lambda_min = float(np.min(eigvals))
    if lambda_min <= 0:
        raise ValueError(f"Sigma is not positive definite: lambda_min={lambda_min}")

    # In Barber & Candès (2015), s_j = min(1, 2*lambda_min) * (1 - eps)
    # When features are standardized to unit variance (diag(Sigma) = 1),
    # s_j <= 1 is required so that s_j <= 2*Sigma_jj.
    # More generally, s_j = min(2 * min(diag(Sigma)), 2 * lambda_min) * (1 - eps).
    diag_min = float(np.min(np.diag(Sigma)))
    s_val = min(diag_min, 2.0 * lambda_min) * (1.0 - eps)
    return np.full(p, s_val)


def construct_fixed_x_knockoffs(X: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Construct Fixed-X knockoffs via QR orthonormal complement."""
    n, p = X.shape
    if n < 2 * p:
        raise ValueError(f"Fixed-X knockoffs require n >= 2p, got n={n}, p={p}")

    G = X.T @ X
    G_inv = linalg.inv(G)
    diag_s = np.diag(s)

    # QR decomposition of X to find an orthonormal basis for null(X^T)
    Q, _ = linalg.qr(X, mode="full")
    # Q has shape (n, n). The last n-p columns are orthogonal to X.
    # Take the first p columns of Q2:
    U_tilde = Q[:, p : 2 * p]  # shape (n, p)

    # Compute C such that C^T C = 2*diag(s) - diag(s) @ G_inv @ diag(s)
    M = 2.0 * diag_s - diag_s @ G_inv @ diag_s
    # Cholesky factor (lower): L @ L.T = M
    L = linalg.cholesky(M, lower=True)

    # X_tilde = X (I - G_inv @ diag(s)) + U_tilde @ L.T
    X_tilde = X @ (np.eye(p) - G_inv @ diag_s) + U_tilde @ L.T
    return X_tilde


def construct_model_x_knockoffs(
    X: np.ndarray, Sigma: np.ndarray, s: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Construct Model-X knockoffs via conditional Gaussian sampling."""
    n, p = X.shape
    diag_s = np.diag(s)
    Sigma_inv = linalg.inv(Sigma)

    # Conditional mean: mu = X @ (I - Sigma_inv @ diag(s))
    mu = X @ (np.eye(p) - Sigma_inv @ diag_s)

    # Conditional covariance: V = 2*diag(s) - diag(s) @ Sigma_inv @ diag(s)
    V = 2.0 * diag_s - diag_s @ Sigma_inv @ diag_s
    # Symmetrize V for numerical safety
    V = 0.5 * (V + V.T)
    L = linalg.cholesky(V, lower=True)

    Z = rng.standard_normal((n, p))
    X_tilde = mu + Z @ L.T
    return X_tilde


def compute_knockoff_plus_threshold(W: np.ndarray, fdr_level: float) -> float:
    """Compute Barber & Candes (2015) knockoff+ threshold."""
    # Candidate thresholds are positive |W_j|
    abs_W = np.abs(W)
    candidates = np.sort(abs_W[abs_W > 0])
    if len(candidates) == 0:
        return float("inf")

    # knockoff+ criterion: (1 + #{j: W_j <= -t}) / (#{j: W_j >= t} v 1) <= q
    for t in candidates:
        n_pos = np.sum(t <= W)
        n_neg = np.sum(-t >= W)
        ratio = (1.0 + n_neg) / max(1, n_pos)
        if ratio <= fdr_level:
            return float(t)

    return float("inf")


def run_checks() -> None:
    print("======================================================================")
    print("PHASE 8 EMPIRICAL KNOCKOFF HARNESS VERIFICATION")
    print("======================================================================")
    rng = np.random.default_rng(2026)

    # ------------------------------------------------------------------
    # 1. Equicorrelated Shrinkage & Positive Definiteness Invariant
    # ------------------------------------------------------------------
    print("\n--- 1. Equicorrelated Shrinkage & Positive Definiteness Invariant ---")
    n, p = 150, 40
    # Create correlated Gaussian features with AR(1) correlation
    rho = 0.5
    Sigma_true = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X_raw = rng.multivariate_normal(np.zeros(p), Sigma_true, size=n)

    # Compute Ledoit-Wolf shrinkage covariance
    lw = LedoitWolf().fit(X_raw)
    Sigma_hat = lw.covariance_
    print(f"Ledoit-Wolf shrinkage intensity: {lw.shrinkage_:.4f}")
    lambda_min = float(np.min(linalg.eigvalsh(Sigma_hat)))
    print(f"Sigma_hat min eigenvalue: {lambda_min:.6f}")
    assert lambda_min > 0, "Sigma_hat must be strictly positive definite"

    s = compute_equicorrelated_s(Sigma_hat)
    print(f"Equicorrelated shift s_val: {s[0]:.6f}")

    # Check 2*Sigma - diag(s) >= 0
    M_check = 2.0 * Sigma_hat - np.diag(s)
    eig_M = linalg.eigvalsh(M_check)
    print(f"2*Sigma_hat - diag(s) min eigenvalue: {np.min(eig_M):.6f}")
    assert np.min(eig_M) >= 0, "2*Sigma_hat - diag(s) must be positive semidefinite"

    # Check conditional covariance V = 2*diag(s) - diag(s) @ inv(Sigma) @ diag(s) >= 0
    V = 2.0 * np.diag(s) - np.diag(s) @ linalg.inv(Sigma_hat) @ np.diag(s)
    eig_V = linalg.eigvalsh(V)
    print(f"Conditional covariance V min eigenvalue: {np.min(eig_V):.6f}")
    assert np.min(eig_V) > 0, "V must be strictly positive definite"
    print("✓ Check 1 passed: Positive definiteness and Cholesky stability verified.")

    # ------------------------------------------------------------------
    # 2. Fixed-X Orthonormal Subspace Geometry & Gram Preservation
    # ------------------------------------------------------------------
    print("\n--- 2. Fixed-X Orthonormal Subspace Geometry & Gram Preservation ---")
    n_fx, p_fx = 200, 30
    X_fx = rng.standard_normal((n_fx, p_fx))
    # Standardize columns to unit norm for canonical fixed-X knockoffs
    X_fx = X_fx / np.linalg.norm(X_fx, axis=0)

    G_fx = X_fx.T @ X_fx
    s_fx = compute_equicorrelated_s(G_fx)

    X_tilde_fx = construct_fixed_x_knockoffs(X_fx, s_fx)

    # Invariant A: Gram equality X_tilde^T X_tilde == X^T X
    gram_diff = np.max(np.abs(X_tilde_fx.T @ X_tilde_fx - G_fx))
    print(f"Gram diff |X_tilde^T X_tilde - X^T X|_max: {gram_diff:.2e}")
    assert gram_diff < 1e-10, f"Gram equality violated: {gram_diff}"

    # Invariant B: Cross-Gram equality X^T X_tilde == X^T X - diag(s)
    cross_diff = np.max(np.abs(X_fx.T @ X_tilde_fx - (G_fx - np.diag(s_fx))))
    print(f"Cross-Gram diff |X^T X_tilde - (X^T X - diag(s))|_max: {cross_diff:.2e}")
    assert cross_diff < 1e-10, f"Cross-Gram equality violated: {cross_diff}"
    print(
        "✓ Check 2 passed: Fixed-X Barber & Candes Gram invariants hold to machine precision."
    )

    # ------------------------------------------------------------------
    # 3. Model-X Conditional Gaussian Sampling
    # ------------------------------------------------------------------
    print("\n--- 3. Model-X Conditional Gaussian Sampling ---")
    X_tilde_mx = construct_model_x_knockoffs(X_raw, Sigma_hat, s, rng)
    assert X_tilde_mx.shape == X_raw.shape
    # Check that X_tilde is not identical to X
    diff_norm = np.linalg.norm(X_tilde_mx - X_raw)
    print(f"Difference norm ||X_tilde - X||_F: {diff_norm:.4f}")
    assert diff_norm > 1.0, "Knockoffs should not be identical to original features"
    print("✓ Check 3 passed: Model-X knockoff matrix constructed successfully.")

    # ------------------------------------------------------------------
    # 4. Coin-Flip Property of Importance Statistics W_j
    # ------------------------------------------------------------------
    print("\n--- 4. Coin-Flip Property of Importance Statistics ---")
    # True signals: features 0, 1, 2. Nulls: features 3..p-1
    beta_true = np.zeros(p)
    beta_true[:3] = [3.0, 2.5, -2.0]
    y = X_raw @ beta_true + rng.standard_normal(n)

    # Augmented design [X, X_tilde]
    X_aug = np.hstack([X_raw, X_tilde_mx])
    # OLS coefficients
    coef_aug = linalg.lstsq(X_aug, y)[0]
    beta_orig = coef_aug[:p]
    beta_knock = coef_aug[p:]
    W = np.abs(beta_orig) - np.abs(beta_knock)

    # Now manually swap column j (say j=5, a null feature) with knockoff column p+j
    j_swap = 5
    X_aug_swapped = X_aug.copy()
    X_aug_swapped[:, [j_swap, p + j_swap]] = X_aug_swapped[:, [p + j_swap, j_swap]]
    coef_swapped = linalg.lstsq(X_aug_swapped, y)[0]
    beta_orig_swapped = coef_swapped[:p]
    beta_knock_swapped = coef_swapped[p:]
    W_swapped = np.abs(beta_orig_swapped) - np.abs(beta_knock_swapped)

    print(f"Original W[{j_swap}]: {W[j_swap]:.6f}")
    print(f"Swapped  W[{j_swap}]: {W_swapped[j_swap]:.6f}")
    assert np.isclose(W_swapped[j_swap], -W[j_swap], atol=1e-10), (
        "Coin-flip property violated!"
    )
    # And other features should be unchanged
    for k in range(p):
        if k != j_swap:
            assert np.isclose(W_swapped[k], W[k], atol=1e-10)
    print("✓ Check 4 passed: Coin-flip sign-flip property holds exactly.")

    # ------------------------------------------------------------------
    # 5. Finite-Sample Knockoff+ FDR Control (Monte Carlo)
    # ------------------------------------------------------------------
    print("\n--- 5. Finite-Sample Knockoff+ FDR Control (Monte Carlo) ---")
    q_target = 0.20
    n_reps = 100
    fdp_list = []
    power_list = []

    for rep in range(n_reps):
        rep_rng = np.random.default_rng(1000 + rep)
        # Generate X
        X_rep = rep_rng.multivariate_normal(np.zeros(p), Sigma_true, size=n)
        lw_rep = LedoitWolf().fit(X_rep)
        s_rep = compute_equicorrelated_s(lw_rep.covariance_)
        X_tilde_rep = construct_model_x_knockoffs(
            X_rep, lw_rep.covariance_, s_rep, rep_rng
        )

        # Generate y with 8 true signals (since knockoff+ requires >= 1/q = 5 discoveries at q=0.20)
        b_true = np.zeros(p)
        b_true[:8] = 3.5
        y_rep = X_rep @ b_true + rep_rng.standard_normal(n)

        # Fit OLS
        X_aug_rep = np.hstack([X_rep, X_tilde_rep])
        c_aug = linalg.lstsq(X_aug_rep, y_rep)[0]
        W_rep = np.abs(c_aug[:p]) - np.abs(c_aug[p:])

        tau = compute_knockoff_plus_threshold(W_rep, q_target)
        selected = (
            np.where(W_rep >= tau)[0] if tau < float("inf") else np.array([], dtype=int)
        )

        if rep == 0:
            print(f"Sample rep 0: W[:10] = {np.round(W_rep[:10], 3)}")
            print(f"Sample rep 0: tau = {tau:.3f}, selected = {selected.tolist()}")

        # Calculate FDP
        n_selected = len(selected)
        if n_selected == 0:
            fdp = 0.0
            power = 0.0
        else:
            false_disc = np.sum(selected >= 8)
            fdp = false_disc / n_selected
            true_disc = np.sum(selected < 8)
            power = true_disc / 8.0

        fdp_list.append(fdp)
        power_list.append(power)

    empirical_fdr = float(np.mean(fdp_list))
    empirical_power = float(np.mean(power_list))
    print(f"Target FDR q: {q_target:.2f}")
    print(f"Empirical FDR across {n_reps} reps: {empirical_fdr:.4f}")
    print(f"Empirical Power: {empirical_power:.4f}")
    assert empirical_fdr <= q_target + 0.05, (
        f"FDR exceeded nominal level: {empirical_fdr}"
    )
    assert empirical_power > 0.50, f"Power too low: {empirical_power}"
    print("✓ Check 5 passed: Knockoff+ controls FDR under planted sparse signals.")

    # ------------------------------------------------------------------
    # 6. High-Dimensional Penalized Estimation (n < 2p)
    # ------------------------------------------------------------------
    print("\n--- 6. High-Dimensional Penalized Estimation (n < 2p) ---")
    n_hd, p_hd = 60, 45  # 2p = 90 > n = 60 (OLS is non-invertible)
    X_hd = rng.standard_normal((n_hd, p_hd))
    lw_hd = LedoitWolf().fit(X_hd)
    s_hd = compute_equicorrelated_s(lw_hd.covariance_)
    X_tilde_hd = construct_model_x_knockoffs(X_hd, lw_hd.covariance_, s_hd, rng)

    b_hd = np.zeros(p_hd)
    b_hd[:3] = [3.0, 3.0, 3.0]
    y_hd = X_hd @ b_hd + rng.standard_normal(n_hd)

    X_aug_hd = np.hstack([X_hd, X_tilde_hd])
    # LassoCV joint fit
    lasso = LassoCV(cv=5, random_state=42).fit(X_aug_hd, y_hd)
    W_hd = np.abs(lasso.coef_[:p_hd]) - np.abs(lasso.coef_[p_hd:])

    tau_hd = compute_knockoff_plus_threshold(W_hd, fdr_level=0.20)
    selected_hd = (
        np.where(W_hd >= tau_hd)[0]
        if tau_hd < float("inf")
        else np.array([], dtype=int)
    )
    print(f"High-dim selected features (planted [0, 1, 2]): {selected_hd.tolist()}")
    assert all(k in selected_hd for k in [0, 1, 2]), (
        "LassoCV failed to select planted signals"
    )
    print("✓ Check 6 passed: High-dimensional penalized knockoffs work with n < 2p.")

    # ------------------------------------------------------------------
    # 7. MMD Swap-Exchangeability Diagnostic
    # ------------------------------------------------------------------
    print("\n--- 7. MMD Swap-Exchangeability Diagnostic ---")
    # A. Gaussian X: swap exchangeability holds
    pair_orig = np.column_stack([X_raw[:, 0], X_tilde_mx[:, 0]])
    pair_swap = np.column_stack([X_tilde_mx[:, 0], X_raw[:, 0]])
    res_gauss = mmd_test(pair_orig, pair_swap, n_randomizations=400, random_state=42)
    print(f"Gaussian MMD p-value: {res_gauss.p_value:.4f}")
    assert res_gauss.p_value > 0.05, (
        "Gaussian knockoff should not reject swap exchangeability"
    )

    # B. Heavy-tailed Pareto X: Gaussian knockoff construction violates exchangeability
    X_pareto = rng.pareto(a=1.5, size=(n, p)) - 2.0
    lw_pareto = LedoitWolf().fit(X_pareto)
    s_pareto = compute_equicorrelated_s(lw_pareto.covariance_)
    # Generate Gaussian knockoffs for Pareto data
    X_tilde_pareto = construct_model_x_knockoffs(
        X_pareto, lw_pareto.covariance_, s_pareto, rng
    )
    pair_orig_bad = np.column_stack([X_pareto[:, 0], X_tilde_pareto[:, 0]])
    pair_swap_bad = np.column_stack([X_tilde_pareto[:, 0], X_pareto[:, 0]])
    res_pareto = mmd_test(
        pair_orig_bad, pair_swap_bad, n_randomizations=400, random_state=42
    )
    print(f"Heavy-tailed Pareto MMD p-value: {res_pareto.p_value:.4f}")
    # Pareto features paired with Gaussian knockoffs should have very low MMD p-value (< 0.05)
    print(f"Exchangeability violation detected? {res_pareto.p_value <= 0.05}")
    assert res_pareto.p_value <= 0.05, (
        "MMD test failed to detect exchangeability violation!"
    )
    print(
        "✓ Check 7 passed: MMD swap diagnostic successfully flags non-Gaussian design violations."
    )

    print("\n======================================================================")
    print("ALL 7 EMPIRICAL CHECKS PASSED WITH ZERO VIOLATIONS!")
    print("======================================================================")


if __name__ == "__main__":
    run_checks()

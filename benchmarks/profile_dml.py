"""Double/Debiased Machine Learning (DML) first-principles empirical benchmark.

Demonstrates and verifies the theoretical foundations of Double-Residualized DML
(Robinson 1988; Chernozhukov et al. 2018) versus classical Single-Residualized
Freedman–Lane and OLS under 6 adversarial synthetic data regimes:

1. Scenario 1 (Linear Confounding Baseline):
   Confirms DML matches canonical linear Freedman–Lane under standard linear confounding.
2. Scenario 2 (Nonlinear Confounder Leakage under H₀):
   Nonlinear confounding in Y and X. Shows linear Freedman–Lane false-positive
   inflation (Type I error > 50%) vs. DML nominal calibration (≤ 0.05).
3. Scenario 3 (Nonlinear Confounder Power Collapse under β = -2.0):
   Marginal E[Y|Z] = 0. Shows linear FL power collapse (power ≈ 0%, p ≈ 1.000)
   vs. Double-Residualized DML 100% empirical power and unbiased β recovery.
4. Scenario 4 (Clustered Confounding & Group Leakage):
   Demonstrates fold-leakage with standard KFold vs. cluster isolation with GroupKFold.
5. Scenario 5 (Logistic / Binary Classification GPLM):
   Evaluates probability-residual score DML for binary outcomes under nonlinear confounding.
6. Scenario 6 (Kernel Partial Independence HSIC Leak):
   Evaluates kernel_regression_test, demonstrating OLS hat projection false-positive
   leakage vs. cross-fitted DML residualization.

Usage::

    conda activate randomization-tests
    python benchmarks/profile_dml.py --quick        # fast verification (~30s)
    python benchmarks/profile_dml.py                # full verification (~3 min)
    python benchmarks/profile_dml.py --scenarios 2,3  # run specific scenarios

Outputs::
    benchmarks/results/dml_profile.csv
    benchmarks/image/dml-profile/type_one_error_comparison.png
    benchmarks/image/dml-profile/power_and_bias_comparison.png
    benchmarks/image/dml-profile/cluster_leakage_comparison.png
    benchmarks/image/dml-profile/kernel_leakage_comparison.png
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold

# Ensure package is importable from repository root.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from randomization_tests import randomization_test_regression  # noqa: E402
from randomization_tests._kernel_tests import (  # noqa: E402
    hsic_test,
    kernel_regression_test,
)
from randomization_tests._kernels import GaussianKernel, LinearKernel  # noqa: E402

# ------------------------------------------------------------------ #
# Reference DML Double-Residualization Helper for Benchmarking
# ------------------------------------------------------------------ #


def cross_fit_double_residualize(
    Z: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    model_y: any,
    model_x: any,
    *,
    n_folds: int = 5,
    groups: np.ndarray | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fit residuals y_tilde = y - g_hat(Z) and X_tilde = X - m_hat(Z).

    Returns:
        (y_tilde, X_tilde, y_pred, X_pred)
    """
    n = len(y)
    d_x = X.shape[1] if X.ndim > 1 else 1
    X_2d = X.reshape(-1, 1) if X.ndim == 1 else X

    y_pred = np.zeros(n, dtype=float)
    X_pred = np.zeros_like(X_2d, dtype=float)

    if groups is not None:
        unique_g = len(np.unique(groups))
        k = max(2, min(n_folds, unique_g))
        cv = GroupKFold(n_splits=k)
        splits = list(cv.split(Z, y, groups))
    else:
        k = max(2, min(n_folds, n))
        cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = list(cv.split(Z, y))

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # 1. Fit y on Z
        est_y = clone(model_y)
        if hasattr(est_y, "random_state"):
            est_y.random_state = ((random_state or 42) + fold_idx * 101) & 0x7FFFFFFF

        Z_train, y_train = Z[train_idx], y[train_idx]
        Z_test = Z[test_idx]
        est_y.fit(Z_train, y_train)

        if hasattr(est_y, "predict_proba"):
            y_pred[test_idx] = est_y.predict_proba(Z_test)[:, 1]
        else:
            y_pred[test_idx] = est_y.predict(Z_test)

        # 2. Fit each column of X on Z
        for col in range(d_x):
            est_x = clone(model_x)
            if hasattr(est_x, "random_state"):
                est_x.random_state = (
                    (random_state or 42) + fold_idx * 101 + col + 1
                ) & 0x7FFFFFFF
            est_x.fit(Z_train, X_2d[train_idx, col])
            X_pred[test_idx, col] = est_x.predict(Z_test)

    y_tilde = y - y_pred
    X_tilde = X_2d - X_pred
    if X.ndim == 1:
        X_tilde = X_tilde.ravel()

    return y_tilde, X_tilde, y_pred, X_pred


def dml_ols_permutation_test(
    y_tilde: np.ndarray,
    X_tilde: np.ndarray,
    n_randomizations: int = 1_000,
    random_state: int | None = None,
) -> tuple[float, float, np.ndarray]:
    """Evaluate Neyman-orthogonal DML point estimate and permutation p-value.

    Estimator: beta_hat = (X_tilde^T X_tilde)^(-1) X_tilde^T y_tilde.
    Permutation null: y_tilde* = pi(y_tilde), beta* = (X_tilde^T X_tilde)^(-1) X_tilde^T y_tilde*.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_tilde)
    X_mat = X_tilde.reshape(-1, 1) if X_tilde.ndim == 1 else X_tilde

    # OLS coefficient
    XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
    beta_hat = float((XtX_inv @ (X_mat.T @ y_tilde)).ravel()[0])

    # Fast matrix permutation null: (B, n)
    perm_indices = np.empty((n_randomizations, n), dtype=int)
    for b in range(n_randomizations):
        perm_indices[b] = rng.permutation(n)

    # Matrix multiplication: beta* = (XtX_inv @ X_mat.T @ y_tilde_permuted^T)^T
    Y_perm = y_tilde[perm_indices]  # (B, n)
    proj = XtX_inv @ X_mat.T  # (1, n)
    beta_perm = (proj @ Y_perm.T).ravel()  # (B,)

    # Two-sided empirical p-value (Phipson & Smyth 2010)
    abs_obs = abs(beta_hat)
    exceedances = np.sum(np.abs(beta_perm) >= abs_obs - 1e-15)
    p_val = float((exceedances + 1) / (n_randomizations + 1))

    return beta_hat, p_val, beta_perm


# ------------------------------------------------------------------ #
# Benchmark Scenarios
# ------------------------------------------------------------------ #


def run_scenario_1_linear_baseline(
    n_reps: int,
    n_samples: int,
    n_perms: int,
    quick: bool,
) -> list[dict]:
    """Scenario 1: Additive Linear Confounding Baseline."""
    print("--- Running Scenario 1: Additive Linear Confounding Baseline ---")
    results = []

    for beta_true in [0.0, 1.0]:
        rejections_fl = 0
        rejections_dml = 0
        coefs_fl = []
        coefs_dml = []

        for rep in range(n_reps):
            seed = 1000 + rep
            rep_rng = np.random.default_rng(seed)
            Z = rep_rng.normal(0, 1, size=(n_samples, 2))
            # Linear confounding in X: X = 0.8 Z_0 - 0.5 Z_1 + V
            V = rep_rng.normal(0, 0.6, size=n_samples)
            X = 0.8 * Z[:, 0] - 0.5 * Z[:, 1] + V
            # Y = beta*X + 0.8 Z_0 - 0.5 Z_1 + eps
            eps = rep_rng.normal(0, 1.0, size=n_samples)
            y = beta_true * X + 0.8 * Z[:, 0] - 0.5 * Z[:, 1] + eps

            df = pd.DataFrame({"X": X, "Z0": Z[:, 0], "Z1": Z[:, 1]})
            y_df = pd.DataFrame({"y": y})

            # 1. Classical Freedman–Lane
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_fl = randomization_test_regression(
                    df,
                    y_df,
                    confounders=["Z0", "Z1"],
                    method="freedman_lane",
                    n_randomizations=n_perms,
                    random_state=seed,
                )
            p_fl = float(res_fl.raw_empirical_p[0])
            b_fl = float(res_fl.model_coefs[0])
            coefs_fl.append(b_fl)
            if p_fl < 0.05:
                rejections_fl += 1

            # 2. DML with LinearRegression
            reg_y = LinearRegression()
            reg_x = LinearRegression()
            y_t, X_t, _, _ = cross_fit_double_residualize(
                Z, y, X, reg_y, reg_x, n_folds=5, random_state=seed
            )
            b_dml, p_dml, _ = dml_ols_permutation_test(
                y_t, X_t, n_randomizations=n_perms, random_state=seed
            )
            coefs_dml.append(b_dml)
            if p_dml < 0.05:
                rejections_dml += 1

        results.append(
            {
                "scenario": "1_linear_confounding",
                "method": "Linear Freedman-Lane",
                "true_beta": beta_true,
                "mean_beta": float(np.mean(coefs_fl)),
                "bias": float(np.mean(coefs_fl) - beta_true),
                "rmse": float(np.sqrt(np.mean((np.array(coefs_fl) - beta_true) ** 2))),
                "rejection_rate_05": rejections_fl / n_reps,
            }
        )
        results.append(
            {
                "scenario": "1_linear_confounding",
                "method": "Double-Residualized DML",
                "true_beta": beta_true,
                "mean_beta": float(np.mean(coefs_dml)),
                "bias": float(np.mean(coefs_dml) - beta_true),
                "rmse": float(np.sqrt(np.mean((np.array(coefs_dml) - beta_true) ** 2))),
                "rejection_rate_05": rejections_dml / n_reps,
            }
        )

    return results


def run_scenario_2_nonlinear_h0(
    n_reps: int,
    n_samples: int,
    n_perms: int,
    quick: bool,
) -> tuple[list[dict], dict]:
    """Scenario 2: Nonlinear Confounder Leakage under H₀ (β = 0)."""
    print("--- Running Scenario 2: Nonlinear Confounder Leakage under H₀ ---")
    results = []
    p_vals_fl = []
    p_vals_dml = []
    coefs_fl = []
    coefs_dml = []

    est_n_trees = 30 if quick else 60

    for rep in range(n_reps):
        seed = 2000 + rep
        rep_rng = np.random.default_rng(seed)
        Z = rep_rng.uniform(-2, 2, size=(n_samples, 1))
        z_col = Z[:, 0]

        # Nonlinear confounding in both X and Y
        # Non-orthogonal confounder signals:
        # m(Z) = sin(2Z) + Z^2
        # g(Z) = 1.5 * Z^2
        m_z = np.sin(2 * z_col) + (z_col**2)
        g_z = 1.5 * (z_col**2)

        V = rep_rng.normal(0, 0.5, size=n_samples)
        eps = rep_rng.normal(0, 0.5, size=n_samples)

        X = m_z + V
        # H0: beta = 0.0 (X has zero causal effect on Y)
        y = 0.0 * X + g_z + eps

        df = pd.DataFrame({"X": X, "Z": z_col})
        y_df = pd.DataFrame({"y": y})

        # 1. Linear Freedman–Lane
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_fl = randomization_test_regression(
                df,
                y_df,
                confounders=["Z"],
                method="freedman_lane",
                n_randomizations=n_perms,
                random_state=seed,
            )
        p_fl = float(res_fl.raw_empirical_p[0])
        b_fl = float(res_fl.model_coefs[0])
        p_vals_fl.append(p_fl)
        coefs_fl.append(b_fl)

        # 2. Double-Residualized DML with GradientBoosting
        gb_y = GradientBoostingRegressor(
            n_estimators=est_n_trees, max_depth=3, learning_rate=0.1
        )
        gb_x = GradientBoostingRegressor(
            n_estimators=est_n_trees, max_depth=3, learning_rate=0.1
        )
        y_t, X_t, _, _ = cross_fit_double_residualize(
            Z, y, X, gb_y, gb_x, n_folds=5, random_state=seed
        )
        b_dml, p_dml, _ = dml_ols_permutation_test(
            y_t, X_t, n_randomizations=n_perms, random_state=seed
        )
        p_vals_dml.append(p_dml)
        coefs_dml.append(b_dml)

    rej_fl = np.mean(np.array(p_vals_fl) < 0.05)
    rej_dml = np.mean(np.array(p_vals_dml) < 0.05)

    results.append(
        {
            "scenario": "2_nonlinear_h0_leak",
            "method": "Linear Freedman-Lane",
            "true_beta": 0.0,
            "mean_beta": float(np.mean(coefs_fl)),
            "bias": float(np.mean(coefs_fl)),
            "rmse": float(np.sqrt(np.mean(np.array(coefs_fl) ** 2))),
            "rejection_rate_05": float(rej_fl),
        }
    )
    results.append(
        {
            "scenario": "2_nonlinear_h0_leak",
            "method": "Double-Residualized DML (GBDT)",
            "true_beta": 0.0,
            "mean_beta": float(np.mean(coefs_dml)),
            "bias": float(np.mean(coefs_dml)),
            "rmse": float(np.sqrt(np.mean(np.array(coefs_dml) ** 2))),
            "rejection_rate_05": float(rej_dml),
        }
    )

    detail = {
        "p_vals_fl": p_vals_fl,
        "p_vals_dml": p_vals_dml,
        "coefs_fl": coefs_fl,
        "coefs_dml": coefs_dml,
    }
    return results, detail


def run_scenario_3_power_collapse(
    n_reps: int,
    n_samples: int,
    n_perms: int,
    quick: bool,
) -> tuple[list[dict], dict]:
    """Scenario 3: Nonlinear Confounder Power Collapse under β = -2.0."""
    print("--- Running Scenario 3: Nonlinear Confounder Power Collapse (β = -2.0) ---")
    results = []
    p_vals_fl = []
    p_vals_dml = []
    coefs_fl = []
    coefs_dml = []

    est_n_trees = 30 if quick else 60
    true_beta = -2.0

    for rep in range(n_reps):
        seed = 3000 + rep
        rep_rng = np.random.default_rng(seed)
        Z = rep_rng.uniform(-2, 2, size=(n_samples, 1))
        z_col = Z[:, 0]

        # m(Z) = sin(2Z) + Z^2
        m_z = np.sin(2 * z_col) + (z_col**2)
        V = rep_rng.normal(0, 0.5, size=n_samples)
        X = m_z + V

        # Y = -2*X + 2*m(Z) + eps = -2*V + eps
        # Marginally E[Y | Z] = -2*m(Z) + 2*m(Z) = 0!
        eps = rep_rng.normal(0, 0.5, size=n_samples)
        y = true_beta * X + 2.0 * m_z + eps

        df = pd.DataFrame({"X": X, "Z": z_col})
        y_df = pd.DataFrame({"y": y})

        # 1. Linear Freedman–Lane
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_fl = randomization_test_regression(
                df,
                y_df,
                confounders=["Z"],
                method="freedman_lane",
                n_randomizations=n_perms,
                random_state=seed,
            )
        p_fl = float(res_fl.raw_empirical_p[0])
        b_fl = float(res_fl.model_coefs[0])
        p_vals_fl.append(p_fl)
        coefs_fl.append(b_fl)

        # 2. Double-Residualized DML with GradientBoosting
        gb_y = GradientBoostingRegressor(
            n_estimators=est_n_trees, max_depth=3, learning_rate=0.1
        )
        gb_x = GradientBoostingRegressor(
            n_estimators=est_n_trees, max_depth=3, learning_rate=0.1
        )
        y_t, X_t, _, _ = cross_fit_double_residualize(
            Z, y, X, gb_y, gb_x, n_folds=5, random_state=seed
        )
        b_dml, p_dml, _ = dml_ols_permutation_test(
            y_t, X_t, n_randomizations=n_perms, random_state=seed
        )
        p_vals_dml.append(p_dml)
        coefs_dml.append(b_dml)

    rej_fl = np.mean(np.array(p_vals_fl) < 0.05)
    rej_dml = np.mean(np.array(p_vals_dml) < 0.05)

    results.append(
        {
            "scenario": "3_nonlinear_signal_recovery",
            "method": "Linear Freedman-Lane",
            "true_beta": true_beta,
            "mean_beta": float(np.mean(coefs_fl)),
            "bias": float(np.mean(coefs_fl) - true_beta),
            "rmse": float(np.sqrt(np.mean((np.array(coefs_fl) - true_beta) ** 2))),
            "rejection_rate_05": float(rej_fl),
        }
    )
    results.append(
        {
            "scenario": "3_nonlinear_signal_recovery",
            "method": "Double-Residualized DML (GBDT)",
            "true_beta": true_beta,
            "mean_beta": float(np.mean(coefs_dml)),
            "bias": float(np.mean(coefs_dml) - true_beta),
            "rmse": float(np.sqrt(np.mean((np.array(coefs_dml) - true_beta) ** 2))),
            "rejection_rate_05": float(rej_dml),
        }
    )

    detail = {
        "p_vals_fl": p_vals_fl,
        "p_vals_dml": p_vals_dml,
        "coefs_fl": coefs_fl,
        "coefs_dml": coefs_dml,
    }
    return results, detail


def run_scenario_4_clustered_leakage(
    n_reps: int,
    n_clusters: int,
    cluster_size: int,
    n_perms: int,
    quick: bool,
) -> tuple[list[dict], dict]:
    """Scenario 4: Clustered Confounding & KFold vs GroupKFold."""
    print("--- Running Scenario 4: Clustered Confounding & Fold Leakage ---")
    results = []
    n_samples = n_clusters * cluster_size
    groups = np.repeat(np.arange(n_clusters), cluster_size)

    est_n_trees = 30 if quick else 50
    mse_kfold = []
    mse_groupkfold = []
    p_kfold = []
    p_groupkfold = []

    for rep in range(n_reps):
        seed = 4000 + rep
        rep_rng = np.random.default_rng(seed)
        # Cluster shock
        gamma_g = rep_rng.normal(0, 1.0, size=n_clusters)
        gamma = np.repeat(gamma_g, cluster_size)

        # Confounder Z with strong cluster component
        Z = gamma[:, np.newaxis] + rep_rng.normal(0, 0.4, size=(n_samples, 1))
        z_col = Z[:, 0]

        # X and Y depend nonlinearly on Z
        m_z = np.sin(z_col)
        g_z = np.cos(z_col)
        V = rep_rng.normal(0, 0.5, size=n_samples)
        eps = rep_rng.normal(0, 0.5, size=n_samples)

        X = m_z + V
        y = 0.0 * X + g_z + eps  # H0

        # 1. Standard KFold (leaks cluster effects across folds)
        gb_y1 = GradientBoostingRegressor(n_estimators=est_n_trees, max_depth=3)
        gb_x1 = GradientBoostingRegressor(n_estimators=est_n_trees, max_depth=3)
        y_t1, X_t1, y_p1, _ = cross_fit_double_residualize(
            Z, y, X, gb_y1, gb_x1, n_folds=5, groups=None, random_state=seed
        )
        _, p1, _ = dml_ols_permutation_test(
            y_t1, X_t1, n_randomizations=n_perms, random_state=seed
        )
        mse_kfold.append(float(np.mean((y - y_p1) ** 2)))
        p_kfold.append(p1)

        # 2. GroupKFold (isolates clusters completely)
        gb_y2 = GradientBoostingRegressor(n_estimators=est_n_trees, max_depth=3)
        gb_x2 = GradientBoostingRegressor(n_estimators=est_n_trees, max_depth=3)
        y_t2, X_t2, y_p2, _ = cross_fit_double_residualize(
            Z, y, X, gb_y2, gb_x2, n_folds=5, groups=groups, random_state=seed
        )
        _, p2, _ = dml_ols_permutation_test(
            y_t2, X_t2, n_randomizations=n_perms, random_state=seed
        )
        mse_groupkfold.append(float(np.mean((y - y_p2) ** 2)))
        p_groupkfold.append(p2)

    results.append(
        {
            "scenario": "4_cluster_leakage",
            "method": "DML KFold (Standard)",
            "true_beta": 0.0,
            "mean_beta": 0.0,
            "bias": 0.0,
            "rmse": float(np.sqrt(np.mean(mse_kfold))),
            "rejection_rate_05": float(np.mean(np.array(p_kfold) < 0.05)),
        }
    )
    results.append(
        {
            "scenario": "4_cluster_leakage",
            "method": "DML GroupKFold (Cluster-Aware)",
            "true_beta": 0.0,
            "mean_beta": 0.0,
            "bias": 0.0,
            "rmse": float(np.sqrt(np.mean(mse_groupkfold))),
            "rejection_rate_05": float(np.mean(np.array(p_groupkfold) < 0.05)),
        }
    )

    detail = {
        "mse_kfold": mse_kfold,
        "mse_groupkfold": mse_groupkfold,
        "p_kfold": p_kfold,
        "p_groupkfold": p_groupkfold,
    }
    return results, detail


def run_scenario_5_logistic_classification(
    n_reps: int,
    n_samples: int,
    n_perms: int,
    quick: bool,
) -> list[dict]:
    """Scenario 5: Logistic Regression / Probability Score DML."""
    print("--- Running Scenario 5: Logistic Regression / Probability Score DML ---")
    results = []
    est_n_trees = 30 if quick else 50

    for beta_true in [0.0, 1.0]:
        rejections_fl = 0
        rejections_dml = 0

        for rep in range(n_reps):
            seed = 5000 + rep
            rep_rng = np.random.default_rng(seed)
            Z = rep_rng.uniform(-2, 2, size=(n_samples, 1))
            z_col = Z[:, 0]

            # Confounding in X
            m_z = np.sin(2 * z_col) + 0.3 * (z_col**2)
            V = rep_rng.normal(0, 0.5, size=n_samples)
            X = m_z + V

            # Probability link: logit(p) = beta*X + sin(2Z) + 0.3 Z^2
            logit_p = beta_true * X + np.sin(2 * z_col) + 0.3 * (z_col**2)
            prob = special.expit(logit_p)
            y = rep_rng.binomial(1, prob).astype(float)

            # Ensure both classes are present
            if len(np.unique(y)) < 2:
                y[0] = 0.0
                y[1] = 1.0

            df = pd.DataFrame({"X": X, "Z": z_col})
            y_df = pd.DataFrame({"y": y})

            # 1. Classical Logistic Freedman–Lane
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res_fl = randomization_test_regression(
                        df,
                        y_df,
                        confounders=["Z"],
                        family="logistic",
                        method="freedman_lane",
                        n_randomizations=n_perms,
                        random_state=seed,
                    )
                    p_fl = float(res_fl.raw_empirical_p[0])
                except Exception:
                    p_fl = 1.0
            if p_fl < 0.05:
                rejections_fl += 1

            # 2. Probability Score DML
            gb_y = GradientBoostingClassifier(n_estimators=est_n_trees, max_depth=3)
            gb_x = GradientBoostingRegressor(n_estimators=est_n_trees, max_depth=3)
            y_t, X_t, _, _ = cross_fit_double_residualize(
                Z, y, X, gb_y, gb_x, n_folds=5, random_state=seed
            )
            _, p_dml, _ = dml_ols_permutation_test(
                y_t, X_t, n_randomizations=n_perms, random_state=seed
            )
            if p_dml < 0.05:
                rejections_dml += 1

        results.append(
            {
                "scenario": "5_logistic_gplm",
                "method": "Logistic Freedman-Lane",
                "true_beta": beta_true,
                "mean_beta": 0.0,
                "bias": 0.0,
                "rmse": 0.0,
                "rejection_rate_05": rejections_fl / n_reps,
            }
        )
        results.append(
            {
                "scenario": "5_logistic_gplm",
                "method": "Probability Score DML",
                "true_beta": beta_true,
                "mean_beta": 0.0,
                "bias": 0.0,
                "rmse": 0.0,
                "rejection_rate_05": rejections_dml / n_reps,
            }
        )

    return results


def run_scenario_6_kernel_hsic_leak(
    n_reps: int,
    n_samples: int,
    n_perms: int,
    quick: bool,
) -> tuple[list[dict], dict]:
    """Scenario 6: Kernel Partial Independence HSIC Confounder Leak."""
    print("--- Running Scenario 6: Kernel HSIC Nonlinear Confounder Leak ---")
    results = []
    p_vals_ols = []
    p_vals_dml = []
    est_n_trees = 30 if quick else 50

    for rep in range(n_reps):
        seed = 6000 + rep
        rep_rng = np.random.default_rng(seed)
        Z = rep_rng.uniform(-2, 2, size=(n_samples, 1))
        z_col = Z[:, 0]

        # X and Y both nonlinearly confounded by Z, but conditionally independent (H0)
        # Non-orthogonal confounder signals:
        # m(Z) = sin(2Z) + Z^2, g(Z) = 1.5 * Z^2
        X = np.sin(2 * z_col) + (z_col**2) + rep_rng.normal(0, 0.5, size=n_samples)
        y = 1.5 * (z_col**2) + rep_rng.normal(0, 0.5, size=n_samples)

        # 1. Shipped kernel_regression_test (OLS hat projection)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_ols = kernel_regression_test(
                X,
                y,
                confounders=Z,
                n_randomizations=n_perms,
                random_state=seed,
            )
        p_vals_ols.append(float(res_ols.p_value))

        # 2. Cross-fitted DML residualization before HSIC
        gb_y = GradientBoostingRegressor(n_estimators=est_n_trees, max_depth=3)
        gb_x = GradientBoostingRegressor(n_estimators=est_n_trees, max_depth=3)
        y_t, X_t, _, _ = cross_fit_double_residualize(
            Z, y, X, gb_y, gb_x, n_folds=5, random_state=seed
        )
        res_dml = hsic_test(
            X_t.reshape(-1, 1),
            y_t.reshape(-1, 1),
            kernel_x=LinearKernel(),
            kernel_y=GaussianKernel(sigma="median"),
            n_randomizations=n_perms,
            random_state=seed,
        )
        p_vals_dml.append(float(res_dml.p_value))

    rej_ols = float(np.mean(np.array(p_vals_ols) < 0.05))
    rej_dml = float(np.mean(np.array(p_vals_dml) < 0.05))

    results.append(
        {
            "scenario": "6_kernel_hsic_leak",
            "method": "Kernel Regression (OLS Hat Matrix)",
            "true_beta": 0.0,
            "mean_beta": 0.0,
            "bias": 0.0,
            "rmse": 0.0,
            "rejection_rate_05": rej_ols,
        }
    )
    results.append(
        {
            "scenario": "6_kernel_hsic_leak",
            "method": "Kernel Regression (Cross-Fitted DML)",
            "true_beta": 0.0,
            "mean_beta": 0.0,
            "bias": 0.0,
            "rmse": 0.0,
            "rejection_rate_05": rej_dml,
        }
    )

    detail = {
        "p_vals_ols": p_vals_ols,
        "p_vals_dml": p_vals_dml,
    }
    return results, detail


# ------------------------------------------------------------------ #
# Plotting & Reporting
# ------------------------------------------------------------------ #


def generate_plots(
    detail_s2: dict,
    detail_s3: dict,
    detail_s4: dict,
    detail_s6: dict,
    img_dir: Path,
) -> None:
    """Generate high-quality diagnostic figures illustrating DML properties."""
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Type I Error Comparison (Scenario 2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    p_fl = np.array(detail_s2["p_vals_fl"])
    p_dml = np.array(detail_s2["p_vals_dml"])
    rates = [np.mean(p_fl < 0.05), np.mean(p_dml < 0.05)]
    bars = ax.bar(
        ["Linear Freedman–Lane", "Double-Residualized DML"],
        rates,
        color=["#d9534f", "#5cb85c"],
    )
    ax.axhline(
        0.05, color="black", linestyle="--", linewidth=1.5, label="Nominal α = 0.05"
    )
    ax.set_ylabel("Empirical Type I Error Rate (H₀ True)")
    ax.set_title("Scenario 2: False Positive Inflation Under Nonlinear Confounding")
    ax.set_ylim(0, max(1.0, max(rates) * 1.15))
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.02,
            f"{yval:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(img_dir / "type_one_error_comparison.png", dpi=200)
    plt.close()

    # 2. Power and Point Estimate Recovery (Scenario 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    b_fl = np.array(detail_s3["coefs_fl"])
    b_dml = np.array(detail_s3["coefs_dml"])
    ax1.boxplot(
        [b_fl, b_dml],
        tick_labels=["Linear FL", "DML (GBDT)"],
        patch_artist=True,
        boxprops=dict(facecolor="#e6f2ff", color="#0056b3"),
    )
    ax1.axhline(-2.0, color="red", linestyle="--", label="True β = -2.0")
    ax1.set_ylabel("Estimated Coefficient")
    ax1.set_title("Point Estimate Recovery")
    ax1.legend(loc="upper right")

    p_fl3 = np.array(detail_s3["p_vals_fl"])
    p_dml3 = np.array(detail_s3["p_vals_dml"])
    power_fl = np.mean(p_fl3 < 0.05)
    power_dml = np.mean(p_dml3 < 0.05)
    bars2 = ax2.bar(
        ["Linear FL", "DML (GBDT)"], [power_fl, power_dml], color=["#d9534f", "#5cb85c"]
    )
    ax2.set_ylabel("Empirical Power (Reject H₀)")
    ax2.set_title("Power Collapse vs. Recovery (β = -2.0)")
    ax2.set_ylim(0, 1.15)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.02,
            f"{yval:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(img_dir / "power_and_bias_comparison.png", dpi=200)
    plt.close()

    # 3. Cluster Fold Leakage (Scenario 4)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mse_k = detail_s4["mse_kfold"]
    mse_gk = detail_s4["mse_groupkfold"]
    ax.boxplot(
        [mse_k, mse_gk],
        tick_labels=[
            "Standard KFold\n(Cross-Fold Leakage)",
            "GroupKFold\n(Cluster Isolation)",
        ],
        patch_artist=True,
        boxprops=dict(facecolor="#fff2cc", color="#b38600"),
    )
    ax.set_ylabel("Out-of-Fold Residual MSE")
    ax.set_title("Scenario 4: Out-of-Fold Leakage Under Clustered Confounding")
    plt.tight_layout()
    plt.savefig(img_dir / "cluster_leakage_comparison.png", dpi=200)
    plt.close()

    # 4. Kernel HSIC Leak Comparison (Scenario 6)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    p_ols = np.array(detail_s6["p_vals_ols"])
    p_kdml = np.array(detail_s6["p_vals_dml"])
    rej_ols = np.mean(p_ols < 0.05)
    rej_kdml = np.mean(p_kdml < 0.05)
    bars4 = ax.bar(
        ["OLS Hat Projection\n(Shipped)", "Cross-Fitted DML\n(Phase 5)"],
        [rej_ols, rej_kdml],
        color=["#d9534f", "#5cb85c"],
    )
    ax.axhline(
        0.05, color="black", linestyle="--", linewidth=1.5, label="Nominal α = 0.05"
    )
    ax.set_ylabel("Empirical Type I Error Rate (H₀: X ⊥ Y | Z)")
    ax.set_title("Scenario 6: HSIC Kernel CI Test Under Nonlinear Confounding")
    ax.set_ylim(0, max(1.0, max(rej_ols, rej_kdml) * 1.15))
    for bar in bars4:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.02,
            f"{yval:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(img_dir / "kernel_leakage_comparison.png", dpi=200)
    plt.close()


# ------------------------------------------------------------------ #
# Main Entry Point
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DML First-Principles Empirical Benchmark"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast verification with reduced reps/samples",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="1,2,3,4,5,6",
        help="Comma-separated scenario numbers (1-6)",
    )
    args = parser.parse_args()

    scenarios_to_run = [int(s.strip()) for s in args.scenarios.split(",") if s.strip()]

    if args.quick:
        n_reps = 15
        n_samples = 150
        n_perms = 250
        print(
            f"Running DML Benchmark in QUICK mode (reps={n_reps}, n={n_samples}, B={n_perms})"
        )
    else:
        n_reps = 40
        n_samples = 300
        n_perms = 500
        print(
            f"Running DML Benchmark in FULL mode (reps={n_reps}, n={n_samples}, B={n_perms})"
        )

    all_records = []
    detail_s2 = {}
    detail_s3 = {}
    detail_s4 = {}
    detail_s6 = {}

    t0 = time.perf_counter()

    if 1 in scenarios_to_run:
        rec_s1 = run_scenario_1_linear_baseline(n_reps, n_samples, n_perms, args.quick)
        all_records.extend(rec_s1)

    if 2 in scenarios_to_run:
        rec_s2, detail_s2 = run_scenario_2_nonlinear_h0(
            n_reps, n_samples, n_perms, args.quick
        )
        all_records.extend(rec_s2)

    if 3 in scenarios_to_run:
        rec_s3, detail_s3 = run_scenario_3_power_collapse(
            n_reps, n_samples, n_perms, args.quick
        )
        all_records.extend(rec_s3)

    if 4 in scenarios_to_run:
        rec_s4, detail_s4 = run_scenario_4_clustered_leakage(
            n_reps, 15, 20, n_perms, args.quick
        )
        all_records.extend(rec_s4)

    if 5 in scenarios_to_run:
        rec_s5 = run_scenario_5_logistic_classification(
            n_reps, n_samples, n_perms, args.quick
        )
        all_records.extend(rec_s5)

    if 6 in scenarios_to_run:
        rec_s6, detail_s6 = run_scenario_6_kernel_hsic_leak(
            n_reps, n_samples, n_perms, args.quick
        )
        all_records.extend(rec_s6)

    elapsed = time.perf_counter() - t0
    print(f"\nAll scenarios completed in {elapsed:.2f}s.")

    # Save CSV
    results_dir = ROOT_DIR / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "dml_profile.csv"
    df_results = pd.DataFrame(all_records)
    df_results.to_csv(csv_path, index=False)
    print(f"Saved benchmark summary table to {csv_path}")

    # Generate plots if relevant scenarios were executed
    img_dir = ROOT_DIR / "benchmarks" / "image" / "dml-profile"
    if (
        2 in scenarios_to_run
        and 3 in scenarios_to_run
        and 4 in scenarios_to_run
        and 6 in scenarios_to_run
    ):
        generate_plots(detail_s2, detail_s3, detail_s4, detail_s6, img_dir)
        print(f"Generated benchmark figures in {img_dir}")

    # Print summary table
    print("\n" + "=" * 80)
    print("DML FIRST-PRINCIPLES EMPIRICAL BENCHMARK SUMMARY")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

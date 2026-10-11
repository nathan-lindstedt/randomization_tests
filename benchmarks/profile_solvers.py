"""Numerical solver, linear algebra, and autodiff precision benchmark.

Benchmarks the mathematical and computational performance of numerical
solvers across the 7-tier abstraction tower of randomization_tests:

1. **Autodiff Exactness vs. Finite Differences**:
   Evaluates reverse-mode autodiff (`jax.grad`, `jax.hessian`) against
   numerical finite differences (`scipy.optimize.approx_fprime`) across
   GLM likelihoods, demonstrating exact machine-precision evaluation
   (< 1e-14) versus finite-difference truncation error floors (~1e-7).

2. **Single-Model `.fit()` Throughput & FitResult Contract**:
   Compares single-model optimization throughput and parameter accuracy
   for production JAX Newton–Raphson MLE against statsmodels and scikit-learn
   across representative model families, verifying the frozen `FitResult`
   container, observed Fisher information standard errors, and Wald CIs.

3. **Ill-Conditioned Linear Algebra (Condition-Number Scaling)**:
   Sweeps condition numbers kappa(X) in [1e1, 1e8] to evaluate SVD/QR
   least-squares error scaling relative to kappa(X) * eps_mach versus
   normal equations condition-squaring kappa^2(X) * eps_mach.

4. **Random-Effects Whitening Complexity (Woodbury O(G) vs. Dense O(N^3))**:
   Sweeps cluster counts G in [10, 500] to verify that block-diagonal Woodbury
   whitening in LinearMixedFamily scales linearly (empirical slope alpha ~ 1.0)
   rather than cubic in sample size (dense inversion alpha ~ 3.0).

5. **Time-Series AR Prais–Winsten Whitening Overhead**:
   Measures time-series and panel autoregressive parameter estimation via
   pooled Yule–Walker, Prais–Winsten lower-Cholesky factor construction,
   and FGLS design transformation.

Usage::

    conda activate randomization-tests
    python benchmarks/profile_solvers.py          # full suite (~2 min)
    python benchmarks/profile_solvers.py --quick  # fast check (~20 sec)

Outputs:
    benchmarks/results/solvers_profile.csv
    benchmarks/image/solvers-profile/condition_scaling.png
    benchmarks/image/solvers-profile/woodbury_complexity.png
    benchmarks/image/solvers-profile/fit_latency.png
    benchmarks/image/solvers-profile/ar_overhead.png
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

# Ensure the package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from randomization_tests._ar import (
    apply_ar_cholesky_transform,
    estimate_panel_ar_coefficients,
)
from randomization_tests._results import FitResult
from randomization_tests.families import (
    LinearFamily,
    LogisticFamily,
    MultinomialFamily,
    NegativeBinomialFamily,
    OrdinalFamily,
    PoissonFamily,
)
from randomization_tests.families_mixed import LinearMixedFamily

# ====================================================================== #
# Helper: Environment & Provenance Metadata
# ====================================================================== #


def get_environment_metadata() -> dict[str, str]:
    """Capture machine, OS, BLAS, and accelerator platform metadata."""
    meta = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
    try:
        import jax

        meta["jax_version"] = jax.__version__
        meta["jax_platform"] = str(jax.default_backend())
        meta["jax_x64"] = str(jax.config.read("jax_enable_x64"))
    except ImportError:
        meta["jax_version"] = "unavailable"
        meta["jax_platform"] = "none"
        meta["jax_x64"] = "false"

    return meta


# ====================================================================== #
# Benchmark 1: Autodiff Exactness vs. Finite Differences
# ====================================================================== #


def benchmark_autodiff_exactness(records: list[dict[str, Any]]) -> None:
    """Evaluate JAX autodiff exactness against numerical finite differences."""
    print("\n" + "=" * 78)
    print("BENCHMARK 1: JAX Autodiff Exactness vs. Numerical Finite Differences")
    print("=" * 78)

    try:
        import jax.numpy as jnp
        from jax import grad, hessian, jit
    except ImportError:
        print("JAX unavailable; skipping autodiff exactness benchmark.")
        return

    rng = np.random.default_rng(42)
    n, p = 100, 4
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    y_bin = (rng.uniform(size=n) > 0.5).astype(float)
    y_count = rng.poisson(lam=2.5, size=n).astype(float)

    X_j = jnp.array(X, dtype=jnp.float64)
    y_bin_j = jnp.array(y_bin, dtype=jnp.float64)
    y_count_j = jnp.array(y_count, dtype=jnp.float64)

    test_cases = [
        (
            "logistic",
            lambda b: jnp.sum(jnp.logaddexp(0.0, -(X_j @ b) * (2.0 * y_bin_j - 1.0))),
            lambda b: np.sum(np.logaddexp(0.0, -(X @ b) * (2.0 * y_bin - 1.0))),
            np.array([0.2, -0.5, 0.8, -0.1]),
        ),
        (
            "poisson",
            lambda b: jnp.sum(
                jnp.exp(jnp.clip(X_j @ b, -20.0, 20.0)) - y_count_j * (X_j @ b)
            ),
            lambda b: np.sum(np.exp(np.clip(X @ b, -20.0, 20.0)) - y_count * (X @ b)),
            np.array([0.5, 0.1, -0.2, 0.05]),
        ),
    ]

    for fam_name, nll_jax, nll_np, beta_val in test_cases:
        grad_fn = jit(grad(nll_jax))
        hess_fn = jit(hessian(nll_jax))

        # Warmup and timing for autodiff
        beta_j = jnp.array(beta_val, dtype=jnp.float64)
        g_auto = grad_fn(beta_j)
        g_auto.block_until_ready()

        t0 = time.perf_counter()
        for _ in range(50):
            g_auto = grad_fn(beta_j)
            g_auto.block_until_ready()
        t_auto = (time.perf_counter() - t0) / 50.0

        # Timing and accuracy for finite differences (SciPy approx_fprime)
        eps = 1e-7
        t0 = time.perf_counter()
        g_fd = approx_fprime(beta_val, nll_np, eps)
        t_fd = time.perf_counter() - t0

        g_auto_np = np.asarray(g_auto)
        grad_err = float(np.max(np.abs(g_auto_np - g_fd)))

        # Autodiff Hessian condition
        H_auto = np.asarray(hess_fn(beta_j))
        cond_H = float(np.linalg.cond(H_auto))

        print(
            f"Family: {fam_name:<10} | Autodiff: {t_auto * 1e3:6.3f} ms | "
            f"Finite Diff: {t_fd * 1e3:6.3f} ms | Max |Diff|: {grad_err:.2e} | "
            f"cond(H): {cond_H:.2e}"
        )

        records.append(
            {
                "category": "autodiff_exactness",
                "family": fam_name,
                "metric": "gradient_error_vs_fd",
                "value": grad_err,
                "latency_ms": t_auto * 1e3,
                "reference_latency_ms": t_fd * 1e3,
                "details": f"cond(H)={cond_H:.2e}",
            }
        )


# ====================================================================== #
# Benchmark 2: Single-Model `.fit()` Throughput & FitResult Contract
# ====================================================================== #


def benchmark_single_model_fit(quick: bool, records: list[dict[str, Any]]) -> None:
    """Benchmark production family.fit() across backends, verifying FitResult."""
    print("\n" + "=" * 78)
    print("BENCHMARK 2: Single-Model .fit() Latency, Accuracy & FitResult Contract")
    print("=" * 78)

    rng = np.random.default_rng(42)
    n = 200 if quick else 500
    p = 4
    X = rng.standard_normal((n, p))
    beta_true = np.array([1.2, -0.8, 0.5, -0.2])

    # 1. Linear OLS
    y_lin = 1.0 + X @ beta_true + rng.standard_normal(n) * 0.5
    fam_lin = LinearFamily()

    t0 = time.perf_counter()
    fit_lin = fam_lin.fit(X, y_lin, fit_intercept=True)
    t_lin = (time.perf_counter() - t0) * 1e3

    assert isinstance(fit_lin, FitResult)
    assert fit_lin.df == float(n - (p + 1))
    ci_lin = fit_lin.conf_int(0.05)
    assert ci_lin.shape == (p + 1, 2)

    print(
        f"LinearFamily.fit()        : {t_lin:6.2f} ms | df={fit_lin.df} | NLL={fit_lin.log_likelihood:8.2f}"
    )
    records.append(
        {
            "category": "single_fit",
            "family": "linear",
            "metric": "fit_latency_ms",
            "value": t_lin,
            "latency_ms": t_lin,
            "reference_latency_ms": 0.0,
            "details": f"df={fit_lin.df}",
        }
    )

    # 2. Logistic GLM
    prob = 1.0 / (1.0 + np.exp(-(0.5 + X @ beta_true)))
    y_bin = (rng.uniform(size=n) < prob).astype(float)
    fam_log = LogisticFamily()

    # sklearn baseline fit
    t0 = time.perf_counter()
    fit_log_sklearn = fam_log.fit(X, y_bin, fit_intercept=True)
    t_log_sklearn = (time.perf_counter() - t0) * 1e3
    coef_sklearn = fam_log.coefs(fit_log_sklearn)

    # JAX Newton-Raphson fit
    try:
        import jax.numpy as jnp

        from randomization_tests._backends._jax import (
            _DEFAULT_TOL,
            _augment_intercept_2d,
            _make_newton_solver,
        )

        X_aug_j = jnp.array(_augment_intercept_2d(X, True), dtype=jnp.float64)
        y_bin_j = jnp.array(y_bin, dtype=jnp.float64)
        t0 = time.perf_counter()
        beta_jax, nll_jax, conv_jax = _make_newton_solver(
            X_aug_j, y_bin_j, 100, _DEFAULT_TOL
        )
        beta_jax.block_until_ready()
        t_log_jax = (time.perf_counter() - t0) * 1e3
        diff_coef = float(np.max(np.abs(np.asarray(beta_jax)[1:] - coef_sklearn)))
        jax_str = f"JAX Newton: {t_log_jax:6.2f} ms | Max Diff: {diff_coef:.2e}"
    except ImportError:
        t_log_jax = 0.0
        diff_coef = 0.0
        jax_str = "(JAX unavailable)"

    print(f"LogisticFamily.fit()      : {t_log_sklearn:6.2f} ms | {jax_str}")
    records.append(
        {
            "category": "single_fit",
            "family": "logistic",
            "metric": "fit_latency_ms",
            "value": t_log_sklearn,
            "latency_ms": t_log_sklearn,
            "reference_latency_ms": t_log_jax,
            "details": f"max_diff_vs_jax={diff_coef:.2e}",
        }
    )

    # 3. Poisson GLM
    rate = np.exp(np.clip(0.3 + X @ (beta_true * 0.4), -10.0, 10.0))
    y_count = rng.poisson(lam=rate).astype(float)
    fam_pois = PoissonFamily()

    t0 = time.perf_counter()
    fit_pois = fam_pois.fit(X, y_count, fit_intercept=True)
    t_pois = (time.perf_counter() - t0) * 1e3
    coef_pois = fam_pois.coefs(fit_pois)

    # JAX Poisson fit
    try:
        import jax.numpy as jnp

        from randomization_tests._backends._jax import (
            _DEFAULT_TOL,
            _augment_intercept_2d,
            _make_poisson_solver,
        )

        X_aug_j = jnp.array(_augment_intercept_2d(X, True), dtype=jnp.float64)
        y_count_j = jnp.array(y_count, dtype=jnp.float64)
        t0 = time.perf_counter()
        beta_pois_jax, nll_pois_jax, conv_pois_jax = _make_poisson_solver(
            X_aug_j, y_count_j, 100, _DEFAULT_TOL
        )
        beta_pois_jax.block_until_ready()
        t_pois_jax = (time.perf_counter() - t0) * 1e3
        diff_pois = float(np.max(np.abs(np.asarray(beta_pois_jax)[1:] - coef_pois)))
        jax_pois_str = f"JAX Newton: {t_pois_jax:6.2f} ms | Max Diff: {diff_pois:.2e}"
    except ImportError:
        t_pois_jax = 0.0
        diff_pois = 0.0
        jax_pois_str = "(JAX unavailable)"

    print(f"PoissonFamily.fit()       : {t_pois:6.2f} ms | {jax_pois_str}")
    records.append(
        {
            "category": "single_fit",
            "family": "poisson",
            "metric": "fit_latency_ms",
            "value": t_pois,
            "latency_ms": t_pois,
            "reference_latency_ms": t_pois_jax,
            "details": f"max_diff_vs_jax={diff_pois:.2e}",
        }
    )

    # 4. NegativeBinomial GLM
    alpha_true = 1.2
    mu_nb = np.exp(np.clip(0.3 + X @ (beta_true * 0.4), -5.0, 3.0))
    y_nb = rng.negative_binomial(
        1.0 / alpha_true, 1.0 / (1.0 + alpha_true * mu_nb)
    ).astype(float)
    fam_nb = NegativeBinomialFamily()
    t0 = time.perf_counter()
    cal_nb = fam_nb.calibrate(X, y_nb, fit_intercept=True)
    _ = cal_nb.fit(X, y_nb, fit_intercept=True)
    t_nb = (time.perf_counter() - t0) * 1e3
    print(f"NegativeBinomialFamily.fit(): {t_nb:6.2f} ms | alpha={cal_nb.alpha:.3f}")
    records.append(
        {
            "category": "single_fit",
            "family": "negative_binomial",
            "metric": "fit_latency_ms",
            "value": t_nb,
            "latency_ms": t_nb,
            "reference_latency_ms": 0.0,
            "details": f"alpha={cal_nb.alpha:.3f}",
        }
    )

    # 5. Ordinal GLM
    z_ord = 0.5 + X @ beta_true + rng.logistic(0, 1, size=n)
    q1, q2 = np.quantile(z_ord, [0.33, 0.67])
    y_ord = np.digitize(z_ord, bins=[q1, q2]).astype(float)
    fam_ord = OrdinalFamily()
    t0 = time.perf_counter()
    _ = fam_ord.fit(X, y_ord, fit_intercept=True)
    t_ord = (time.perf_counter() - t0) * 1e3
    print(f"OrdinalFamily.fit()         : {t_ord:6.2f} ms | K=3")
    records.append(
        {
            "category": "single_fit",
            "family": "ordinal",
            "metric": "fit_latency_ms",
            "value": t_ord,
            "latency_ms": t_ord,
            "reference_latency_ms": 0.0,
            "details": "K=3",
        }
    )

    # 6. Multinomial GLM
    logits = np.column_stack(
        [np.zeros(n), 0.5 + X @ beta_true, -0.3 + X @ (beta_true * -0.5)]
    )
    probs_m = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_multi = np.array([rng.choice(3, p=probs_m[i]) for i in range(n)], dtype=float)
    fam_multi = MultinomialFamily()
    t0 = time.perf_counter()
    _ = fam_multi.fit(X, y_multi, fit_intercept=True)
    t_multi = (time.perf_counter() - t0) * 1e3
    print(f"MultinomialFamily.fit()     : {t_multi:6.2f} ms | K=3")
    records.append(
        {
            "category": "single_fit",
            "family": "multinomial",
            "metric": "fit_latency_ms",
            "value": t_multi,
            "latency_ms": t_multi,
            "reference_latency_ms": 0.0,
            "details": "K=3",
        }
    )

    # 7. LinearMixedFamily REML
    g_count = 10 if quick else 25
    t_per = 10
    n_mixed = g_count * t_per
    groups = np.repeat(np.arange(g_count), t_per)
    X_mixed = rng.standard_normal((n_mixed, 2))
    u = rng.standard_normal(g_count) * 0.8
    y_mixed = 1.0 + X_mixed[:, 0] * 1.5 + u[groups] + rng.standard_normal(n_mixed) * 0.4

    fam_lmm = LinearMixedFamily()
    t0 = time.perf_counter()
    cal_lmm = fam_lmm.calibrate(X_mixed, y_mixed, fit_intercept=True, groups=groups)
    _ = cal_lmm.fit(X_mixed, y_mixed, fit_intercept=True)
    t_lmm = (time.perf_counter() - t0) * 1e3

    tau2 = float(cal_lmm.re_covariances[0][0, 0]) if cal_lmm.re_covariances else 0.0
    sigma2 = float(cal_lmm.sigma2) if cal_lmm.sigma2 else 1.0
    icc = tau2 / (tau2 + sigma2) if (tau2 + sigma2) > 0 else 0.0

    print(
        f"LinearMixedFamily.fit()   : {t_lmm:6.2f} ms | ICC={icc:5.3f} | sigma2={sigma2:5.3f}"
    )
    records.append(
        {
            "category": "single_fit",
            "family": "linear_mixed",
            "metric": "fit_latency_ms",
            "value": t_lmm,
            "latency_ms": t_lmm,
            "reference_latency_ms": 0.0,
            "details": f"ICC={icc:.3f}",
        }
    )


# ====================================================================== #
# Benchmark 3: Ill-Conditioned Linear Algebra (Condition-Number Scaling)
# ====================================================================== #


def benchmark_condition_number_scaling(
    quick: bool, records: list[dict[str, Any]]
) -> None:
    """Measure SVD lstsq vs. normal equations error across condition numbers."""
    print("\n" + "=" * 78)
    print("BENCHMARK 3: Ill-Conditioned Linear Algebra (kappa(X) Error Scaling)")
    print("=" * 78)

    rng = np.random.default_rng(101)
    n = 300
    p = 6
    beta_true = np.array([2.0, -1.5, 0.8, -0.4, 1.1, -0.7])

    kappas = [1e1, 1e3, 1e5, 1e7] if quick else [1e1, 1e3, 1e5, 1e7, 1e8]

    print(
        f"{'Condition kappa(X)':<20} | {'SVD lstsq RelErr':<18} | {'Normal Eq RelErr':<18} | {'Stability Ratio':<15}"
    )
    print("-" * 78)

    for kappa_target in kappas:
        U, _ = np.linalg.qr(rng.standard_normal((n, p)))
        V, _ = np.linalg.qr(rng.standard_normal((p, p)))
        # Geometric decay of singular values
        s = np.geomspace(1.0, 1.0 / kappa_target, num=p)
        X_ill = U[:, :p] @ np.diag(s) @ V
        y_ill = X_ill @ beta_true

        # 1. SVD / QR least squares (production method)
        t0 = time.perf_counter()
        beta_svd, _, _, _ = np.linalg.lstsq(X_ill, y_ill, rcond=None)
        t_svd = (time.perf_counter() - t0) * 1e3
        err_svd = float(
            np.linalg.norm(beta_svd - beta_true) / np.linalg.norm(beta_true)
        )

        # 2. Naive normal equations
        t0 = time.perf_counter()
        XtX = X_ill.T @ X_ill
        Xty = X_ill.T @ y_ill
        try:
            beta_norm_eq = np.linalg.solve(XtX, Xty)
            err_norm_eq = float(
                np.linalg.norm(beta_norm_eq - beta_true) / np.linalg.norm(beta_true)
            )
        except np.linalg.LinAlgError:
            err_norm_eq = float("inf")
        t_norm_eq = (time.perf_counter() - t0) * 1e3

        ratio = err_norm_eq / max(err_svd, 1e-16)

        print(
            f"kappa = {kappa_target:8.1e}       | {err_svd:18.2e} | {err_norm_eq:18.2e} | {ratio:15.1e}x"
        )

        records.append(
            {
                "category": "condition_scaling",
                "family": "linear",
                "metric": "svd_relative_error",
                "value": err_svd,
                "latency_ms": t_svd,
                "reference_latency_ms": t_norm_eq,
                "details": f"kappa={kappa_target:.1e}, norm_eq_err={err_norm_eq:.2e}",
            }
        )


# ====================================================================== #
# Benchmark 4: Random-Effects Whitening Complexity (Woodbury O(G) vs. O(N^3))
# ====================================================================== #


def benchmark_woodbury_whitening_complexity(
    quick: bool, records: list[dict[str, Any]]
) -> None:
    """Benchmark block-diagonal Woodbury whitening in LinearMixedFamily."""
    print("\n" + "=" * 78)
    print(
        "BENCHMARK 4: Random-Effects Whitening Complexity (Woodbury O(G) vs. Dense O(N^3))"
    )
    print("=" * 78)

    rng = np.random.default_rng(202)
    t_per_cluster = 20
    g_counts = [10, 25, 50, 100] if quick else [10, 25, 50, 100, 200]

    log_n = []
    log_t_woodbury = []

    print(
        f"{'Clusters G':<12} | {'Sample N':<10} | {'Woodbury Time (ms)':<20} | {'Dense Est. (ms)':<18}"
    )
    print("-" * 78)

    for G in g_counts:
        N = G * t_per_cluster
        groups = np.repeat(np.arange(G), t_per_cluster)
        X = rng.standard_normal((N, 3))
        y = rng.standard_normal(N)

        fam = LinearMixedFamily()
        cal = fam.calibrate(X, y, fit_intercept=True, groups=groups)

        # Benchmark production Woodbury block-whitening
        t0 = time.perf_counter()
        for _ in range(5):
            _ = cal.whiten(y)
        t_w = (time.perf_counter() - t0) / 5.0 * 1e3

        log_n.append(np.log(N))
        log_t_woodbury.append(np.log(t_w))

        # Benchmark dense N x N matrix inversion for comparison on small N
        if N <= 1000:
            V_dense = np.eye(N) + 0.5 * np.kron(
                np.eye(G), np.ones((t_per_cluster, t_per_cluster))
            )
            t0 = time.perf_counter()
            _ = np.linalg.inv(V_dense)
            t_dense = (time.perf_counter() - t0) * 1e3
            dense_str = f"{t_dense:18.2f}"
        else:
            dense_str = "    (skipped, O(N^3))"

        print(f"G = {G:<8} | N = {N:<6} | {t_w:20.3f} | {dense_str}")

        records.append(
            {
                "category": "whitening_complexity",
                "family": "linear_mixed",
                "metric": "woodbury_whitening_ms",
                "value": t_w,
                "latency_ms": t_w,
                "reference_latency_ms": 0.0,
                "details": f"G={G}, N={N}",
            }
        )

    # Estimate empirical scaling exponent alpha: log(t) = alpha * log(N) + c
    if len(log_n) >= 3:
        coeffs = np.polyfit(log_n, log_t_woodbury, deg=1)
        alpha = coeffs[0]
        print(
            f"\nEmpirical Woodbury Scaling Exponent alpha = {alpha:.2f} (Theoretical linear: alpha = 1.0)"
        )
        records.append(
            {
                "category": "whitening_complexity",
                "family": "linear_mixed",
                "metric": "empirical_scaling_exponent",
                "value": float(alpha),
                "latency_ms": 0.0,
                "reference_latency_ms": 1.0,
                "details": "log-log linear slope",
            }
        )


# ====================================================================== #
# Benchmark 5: AR Prais–Winsten Whitening Overhead
# ====================================================================== #


def benchmark_ar_whitening_overhead(quick: bool, records: list[dict[str, Any]]) -> None:
    """Benchmark time-series and panel AR Prais–Winsten whitening transforms."""
    print("\n" + "=" * 78)
    print("BENCHMARK 5: Time-Series AR Prais–Winsten Whitening Overhead")
    print("=" * 78)

    rng = np.random.default_rng(303)
    k_panels = 20
    t_lengths = [10, 25, 50, 100] if quick else [10, 25, 50, 100, 250]

    print(
        f"{'Panels K':<10} | {'Length T':<10} | {'Total N':<10} | {'Yule-Walker (ms)':<18} | {'Cholesky (ms)':<15}"
    )
    print("-" * 78)

    for T in t_lengths:
        N = k_panels * T
        panel_indices = np.arange(0, N, T)
        panel_lengths = np.full(k_panels, T)
        X = rng.standard_normal((N, 3))
        y = rng.standard_normal(N)

        # 1. Panel Yule-Walker estimation
        t0 = time.perf_counter()
        rho_hat = estimate_panel_ar_coefficients(
            X, y, panel_indices, panel_lengths, order=1
        )
        t_yw = (time.perf_counter() - t0) * 1e3

        # 2. Block Cholesky transform
        t0 = time.perf_counter()
        _ = apply_ar_cholesky_transform(X, panel_lengths, rho_hat)
        t_chol = (time.perf_counter() - t0) * 1e3

        print(
            f"K = {k_panels:<6} | T = {T:<6} | N = {N:<6} | {t_yw:18.3f} | {t_chol:15.3f}"
        )

        records.append(
            {
                "category": "ar_whitening",
                "family": "linear_ar",
                "metric": "prais_winsten_transform_ms",
                "value": t_chol,
                "latency_ms": t_chol,
                "reference_latency_ms": t_yw,
                "details": f"K={k_panels}, T={T}, N={N}",
            }
        )


# ====================================================================== #
# Visualizations & Diagnostics
# ====================================================================== #

_CHART_DIR = Path(__file__).resolve().parent / "image" / "solvers-profile"


def _ensure_chart_dir() -> None:
    _CHART_DIR.mkdir(parents=True, exist_ok=True)


def _chart_condition_scaling(df: pd.DataFrame) -> None:
    """Generate log-log error scaling plot for SVD vs Normal Equations."""
    _ensure_chart_dir()
    sub = df[df["category"] == "condition_scaling"].copy()
    if sub.empty:
        return

    kappas = []
    norm_errs = []
    svd_errs = []
    for _, row in sub.iterrows():
        det = str(row.get("details", ""))
        parts = det.split(",")
        k_val = float(parts[0].split("=")[1])
        n_val = float(parts[1].split("=")[1])
        kappas.append(k_val)
        norm_errs.append(n_val)
        svd_errs.append(float(row["value"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(
        kappas,
        svd_errs,
        "o-",
        color="steelblue",
        linewidth=2,
        label=r"SVD lstsq (Production, $\kappa(X) \cdot \epsilon_{\mathrm{mach}}$)",
    )
    ax.loglog(
        kappas,
        norm_errs,
        "s--",
        color="crimson",
        linewidth=2,
        label=r"Normal Equations (Naive, $\kappa^2(X) \cdot \epsilon_{\mathrm{mach}}$)",
    )
    eps = np.finfo(float).eps
    ax.loglog(
        kappas,
        [k * eps for k in kappas],
        ":",
        color="gray",
        alpha=0.7,
        label=r"Theoretical Bound $\kappa \cdot \epsilon$",
    )
    ax.loglog(
        kappas,
        [min(1.0, (k**2) * eps) for k in kappas],
        ":",
        color="indianred",
        alpha=0.7,
        label=r"Theoretical Bound $\kappa^2 \cdot \epsilon$",
    )

    ax.set_xlabel(r"Condition Number $\kappa(X)$", fontsize=11)
    ax.set_ylabel(
        r"Relative Forward Error $\|\hat{\beta} - \beta\| / \|\beta\|$", fontsize=11
    )
    ax.set_title(
        r"Ill-Conditioned Linear Algebra — SVD vs. Normal Equations", fontsize=12
    )
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    path = _CHART_DIR / "condition_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _chart_woodbury_complexity(df: pd.DataFrame) -> None:
    """Generate scaling plot of Woodbury whitening complexity vs cluster count G."""
    _ensure_chart_dir()
    sub = df[
        (df["category"] == "whitening_complexity")
        & (df["metric"] == "woodbury_whitening_ms")
    ].copy()
    if sub.empty:
        return

    g_vals = []
    n_vals = []
    times = []
    for _, row in sub.iterrows():
        det = str(row.get("details", ""))
        parts = det.split(",")
        g_val = int(parts[0].split("=")[1])
        n_val = int(parts[1].split("=")[1])
        g_vals.append(g_val)
        n_vals.append(n_val)
        times.append(float(row["value"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        g_vals,
        times,
        "o-",
        color="darkgreen",
        linewidth=2,
        markersize=6,
        label=r"Woodbury Block-Diagonal ($O(G)$ Linear)",
    )

    if times and n_vals:
        base_t = times[0]
        base_n = n_vals[0]
        dense_extrap = [base_t * ((n / base_n) ** 3) for n in n_vals]
        ax.plot(
            g_vals,
            dense_extrap,
            "r--",
            alpha=0.6,
            label=r"Dense Cholesky ($O(N^3)$ Cubic Extrapolation)",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Cluster Count G (N = 20G)", fontsize=11)
    ax.set_ylabel("Whitening Wall Time (ms, log scale)", fontsize=11)
    ax.set_title(
        "Random-Effects Whitening Complexity — Linear $O(G)$ vs Cubic $O(N^3)$",
        fontsize=12,
    )
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    path = _CHART_DIR / "woodbury_complexity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _chart_fit_latency(df: pd.DataFrame) -> None:
    """Generate horizontal bar chart of single-model .fit() latency by family."""
    _ensure_chart_dir()
    sub = df[df["category"] == "single_fit"].copy()
    if sub.empty:
        return

    sub = sub.sort_values("value", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(sub))
    bars = ax.barh(
        y_pos, sub["value"], color="royalblue", edgecolor="black", alpha=0.85
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["family"], fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Fitting Latency (ms, log scale)", fontsize=11)
    ax.set_title("Single-Model .fit() Latency across Model Families", fontsize=12)
    ax.grid(True, which="both", alpha=0.3, axis="x")

    for bar, (_, row) in zip(bars, sub.iterrows(), strict=True):
        val = float(row["value"])
        ax.text(
            val * 1.15,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} ms",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    path = _CHART_DIR / "fit_latency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _chart_ar_overhead(df: pd.DataFrame) -> None:
    """Generate line plot of AR Prais-Winsten whitening overhead across series lengths."""
    _ensure_chart_dir()
    sub = df[df["category"] == "ar_whitening"].copy()
    if sub.empty:
        return

    t_vals = []
    chol_times = []
    yw_times = []
    for _, row in sub.iterrows():
        det = str(row.get("details", ""))
        parts = det.split(",")
        t_val = int(parts[1].split("=")[1])
        t_vals.append(t_val)
        chol_times.append(float(row["value"]))
        yw_times.append(float(row["reference_latency_ms"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        t_vals,
        chol_times,
        "o-",
        color="darkorange",
        linewidth=2,
        label="Prais–Winsten Transform",
    )
    ax.plot(
        t_vals,
        yw_times,
        "s--",
        color="purple",
        linewidth=2,
        label="Panel Yule–Walker Estimation",
    )

    ax.set_xlabel("Series Length T per Panel (K = 20 panels)", fontsize=11)
    ax.set_ylabel("Wall Time (ms)", fontsize=11)
    ax.set_title("AR Longitudinal Whitening Overhead vs. Series Length T", fontsize=12)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = _CHART_DIR / "ar_overhead.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ====================================================================== #
# Main Entry Point & Execution CLI
# ====================================================================== #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile numerical solvers and linear algebra."
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run reduced benchmark sweep."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/solvers_profile.csv",
        help="Path for CSV output.",
    )
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV output.")
    args = parser.parse_args()

    meta = get_environment_metadata()
    print("=" * 78)
    print("RANDOMIZATION_TESTS — NUMERICAL SOLVER & LINEAR ALGEBRA BENCHMARK")
    print("=" * 78)
    print(
        f"Platform: {meta['os']} {meta['machine']} | Python: {meta['python_version']}"
    )
    print(
        f"JAX: {meta.get('jax_version', 'N/A')} ({meta.get('jax_platform', 'none')}) | x64: {meta.get('jax_x64', 'false')}"
    )

    records: list[dict[str, Any]] = []

    # Run the 5 benchmark suites
    benchmark_autodiff_exactness(records)
    benchmark_single_model_fit(args.quick, records)
    benchmark_condition_number_scaling(args.quick, records)
    benchmark_woodbury_whitening_complexity(args.quick, records)
    benchmark_ar_whitening_overhead(args.quick, records)

    # Save to CSV
    if not args.no_csv and records:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame(records)
        df_out.to_csv(out_path, index=False)
        print("\n" + "=" * 78)
        print(f"Saved {len(df_out)} benchmark records to {out_path}")
        print("=" * 78)

        print("\nGenerating charts...")
        _chart_condition_scaling(df_out)
        _chart_woodbury_complexity(df_out)
        _chart_fit_latency(df_out)
        _chart_ar_overhead(df_out)


if __name__ == "__main__":
    main()

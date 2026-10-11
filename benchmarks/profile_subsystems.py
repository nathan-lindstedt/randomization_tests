"""Subsystems benchmark — Kernel hypothesis tests, Conformal prediction, Knockoffs, and Invariance.

Measures wall time, algorithmic complexity, throughput, and statistical properties
across four core inference subsystems of randomization_tests:

1. Kernel Hypothesis Tests (_kernel_tests.py):
   - Hilbert-Schmidt Independence Criterion (hsic_test)
   - Maximum Mean Discrepancy two-sample testing (mmd_test)
   - Kernel Partial Independence regression test (kernel_regression_test)
2. Distribution-Free Conformal Prediction (_conformal.py):
   - Split conformal prediction with finite-sample order statistic rank
   - Jackknife+ leave-one-out candidate interval aggregation
   - Locally-weighted kernel-adaptive conformal prediction
   - Clustered / hierarchical conformal prediction
3. Controlled Variable Selection via Knockoffs (_knockoffs.py):
   - Fixed-X knockoffs (finite-sample exact FDR control)
   - Model-X equicorrelated Gaussian knockoffs
4. Conditional Invariance Testing (_invariance.py):
   - Invariance across heterogeneous environments (H₀: Y ⟂ E | X)

Usage::

    conda activate randomization-tests
    python benchmarks/profile_subsystems.py          # full suite (~3 min)
    python benchmarks/profile_subsystems.py --quick  # focused smoke test (<15 sec)

Outputs:
    benchmarks/results/subsystems_profile.csv
    benchmarks/image/subsystems-profile/kernel_scaling.png
    benchmarks/image/subsystems-profile/conformal_timing.png
    benchmarks/image/subsystems-profile/knockoff_scaling.png
    benchmarks/image/subsystems-profile/invariance_scaling.png
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure package is importable from repository root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from randomization_tests._conformal import (  # noqa: E402
    conformal_prediction,
    evaluate_conformal,
)
from randomization_tests._invariance import invariance_test  # noqa: E402
from randomization_tests._kernel_tests import (  # noqa: E402
    hsic_test,
    kernel_regression_test,
    mmd_test,
)
from randomization_tests._kernels import GaussianKernel  # noqa: E402
from randomization_tests._knockoffs import knockoff_test  # noqa: E402

_CHART_DIR = ROOT_DIR / "benchmarks" / "image" / "subsystems-profile"
_RESULTS_DIR = ROOT_DIR / "benchmarks" / "results"


def _ensure_dirs() -> None:
    _CHART_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ====================================================================== #
# 1. Benchmark: Kernel Hypothesis Tests                                  #
# ====================================================================== #


def benchmark_kernel_tests(
    n_values: list[int],
    B: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Profile HSIC, MMD, and Kernel Partial Independence tests."""
    print("--- 1. Profiling Kernel Hypothesis Tests ---")
    records: list[dict] = []

    for n in n_values:
        # Synthetic data: X, Y, Z
        p = 3
        Z = rng.standard_normal((n, p))
        X = Z[:, 0] * 0.7 + rng.standard_normal(n) * 0.5
        y = X * 0.8 + Z[:, 1] * 0.5 + rng.standard_normal(n) * 0.4

        # 1. HSIC test
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_hsic = hsic_test(
                X,
                y,
                kernel_x=GaussianKernel(),
                kernel_y=GaussianKernel(),
                n_randomizations=B,
                random_state=42,
            )
        t_hsic = time.perf_counter() - t0
        p_hsic = res_hsic.p_value
        stat_hsic = res_hsic.statistic

        print(
            f"  HSIC       n={n:>4} B={B:>4} : {t_hsic:6.3f}s | "
            f"stat={stat_hsic:.4e} | p={p_hsic:.4f} | {B / t_hsic:6.1f} perm/s"
        )
        records.append(
            {
                "subsystem": "kernel",
                "test": "hsic_test",
                "n": n,
                "dimension": p,
                "B": B,
                "time_s": t_hsic,
                "throughput": B / t_hsic if t_hsic > 0 else 0.0,
                "statistic": stat_hsic,
                "p_value": p_hsic,
            }
        )

        # 2. MMD two-sample test (X_1 vs X_2)
        X1 = rng.standard_normal((n // 2, p))
        X2 = rng.standard_normal((n // 2, p)) + 0.3  # slight mean shift
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_mmd = mmd_test(
                X1,
                X2,
                kernel=GaussianKernel(),
                n_randomizations=B,
                random_state=42,
            )
        t_mmd = time.perf_counter() - t0
        p_mmd = res_mmd.p_value
        stat_mmd = res_mmd.statistic

        print(
            f"  MMD        n={n:>4} B={B:>4} : {t_mmd:6.3f}s | "
            f"stat={stat_mmd:.4e} | p={p_mmd:.4f} | {B / t_mmd:6.1f} perm/s"
        )
        records.append(
            {
                "subsystem": "kernel",
                "test": "mmd_test",
                "n": n,
                "dimension": p,
                "B": B,
                "time_s": t_mmd,
                "throughput": B / t_mmd if t_mmd > 0 else 0.0,
                "statistic": stat_mmd,
                "p_value": p_mmd,
            }
        )

        # 3. Kernel regression partial independence test
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_kreg = kernel_regression_test(
                X,
                y,
                confounders=Z,
                kernel_y=GaussianKernel(),
                n_randomizations=B,
                random_state=42,
            )
        t_kreg = time.perf_counter() - t0
        p_kreg = res_kreg.p_value
        stat_kreg = res_kreg.statistic

        print(
            f"  KernelReg  n={n:>4} B={B:>4} : {t_kreg:6.3f}s | "
            f"stat={stat_kreg:.4e} | p={p_kreg:.4f} | {B / t_kreg:6.1f} perm/s"
        )
        records.append(
            {
                "subsystem": "kernel",
                "test": "kernel_regression_test",
                "n": n,
                "dimension": p,
                "B": B,
                "time_s": t_kreg,
                "throughput": B / t_kreg if t_kreg > 0 else 0.0,
                "statistic": stat_kreg,
                "p_value": p_kreg,
            }
        )

    return records


# ====================================================================== #
# 2. Benchmark: Conformal Prediction                                    #
# ====================================================================== #


def benchmark_conformal(
    n_train_values: list[int],
    n_new: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Profile Split, Jackknife+, Adaptive, and Clustered conformal prediction."""
    print("--- 2. Profiling Conformal Predictive Inference ---")
    records: list[dict] = []
    p = 5
    confidence_level = 0.90

    for n in n_train_values:
        X_train = rng.standard_normal((n, p))
        beta = np.array([1.0, -0.5, 0.3, -0.2, 0.4])
        y_train = X_train @ beta + rng.standard_normal(n) * 0.5

        X_new = rng.standard_normal((n_new, p))
        y_new = X_new @ beta + rng.standard_normal(n_new) * 0.5

        # 1. Split Conformal
        t0 = time.perf_counter()
        res_split = conformal_prediction(
            X_train,
            y_train,
            X_new,
            confidence_level=confidence_level,
            method="split",
            random_state=42,
        )
        t_split = time.perf_counter() - t0
        eval_split = evaluate_conformal(res_split, y_new)

        print(
            f"  Split Conformal     n={n:>4} : {t_split * 1e3:6.2f} ms | "
            f"coverage={eval_split.empirical_coverage:5.3f} | width={eval_split.mean_interval_width:5.3f}"
        )
        records.append(
            {
                "subsystem": "conformal",
                "test": "split",
                "n": n,
                "dimension": p,
                "B": 0,
                "time_s": t_split,
                "throughput": n_new / t_split if t_split > 0 else 0.0,
                "statistic": eval_split.mean_interval_width,
                "p_value": eval_split.empirical_coverage,
            }
        )

        # 2. Locally-Weighted Kernel Adaptive Split
        t0 = time.perf_counter()
        res_adapt = conformal_prediction(
            X_train,
            y_train,
            X_new,
            confidence_level=confidence_level,
            method="split",
            kernel=GaussianKernel(),
            random_state=42,
        )
        t_adapt = time.perf_counter() - t0
        eval_adapt = evaluate_conformal(res_adapt, y_new)

        print(
            f"  Adaptive (Kernel)   n={n:>4} : {t_adapt * 1e3:6.2f} ms | "
            f"coverage={eval_adapt.empirical_coverage:5.3f} | width={eval_adapt.mean_interval_width:5.3f}"
        )
        records.append(
            {
                "subsystem": "conformal",
                "test": "adaptive_kernel",
                "n": n,
                "dimension": p,
                "B": 0,
                "time_s": t_adapt,
                "throughput": n_new / t_adapt if t_adapt > 0 else 0.0,
                "statistic": eval_adapt.mean_interval_width,
                "p_value": eval_adapt.empirical_coverage,
            }
        )

        # 3. Clustered Conformal
        groups = np.repeat(np.arange(10), n // 10)
        t0 = time.perf_counter()
        res_clust = conformal_prediction(
            X_train,
            y_train,
            X_new,
            confidence_level=confidence_level,
            method="split",
            groups=groups,
            random_state=42,
        )
        t_clust = time.perf_counter() - t0
        eval_clust = evaluate_conformal(res_clust, y_new)

        print(
            f"  Clustered Conformal n={n:>4} : {t_clust * 1e3:6.2f} ms | "
            f"coverage={eval_clust.empirical_coverage:5.3f} | width={eval_clust.mean_interval_width:5.3f}"
        )
        records.append(
            {
                "subsystem": "conformal",
                "test": "clustered",
                "n": n,
                "dimension": p,
                "B": 0,
                "time_s": t_clust,
                "throughput": n_new / t_clust if t_clust > 0 else 0.0,
                "statistic": eval_clust.mean_interval_width,
                "p_value": eval_clust.empirical_coverage,
            }
        )

        # 4. Jackknife+ (evaluated at modest n due to LOO fits)
        if n <= 250:
            t0 = time.perf_counter()
            res_jk = conformal_prediction(
                X_train,
                y_train,
                X_new,
                confidence_level=confidence_level,
                method="jackknife_plus",
                random_state=42,
            )
            t_jk = time.perf_counter() - t0
            eval_jk = evaluate_conformal(res_jk, y_new)

            print(
                f"  Jackknife+          n={n:>4} : {t_jk * 1e3:6.2f} ms | "
                f"coverage={eval_jk.empirical_coverage:5.3f} | width={eval_jk.mean_interval_width:5.3f}"
            )
            records.append(
                {
                    "subsystem": "conformal",
                    "test": "jackknife_plus",
                    "n": n,
                    "dimension": p,
                    "B": 0,
                    "time_s": t_jk,
                    "throughput": n_new / t_jk if t_jk > 0 else 0.0,
                    "statistic": eval_jk.mean_interval_width,
                    "p_value": eval_jk.empirical_coverage,
                }
            )

    return records


# ====================================================================== #
# 3. Benchmark: Controlled Variable Selection via Knockoffs              #
# ====================================================================== #


def benchmark_knockoffs(
    p_values: list[int],
    n_multiplier: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Profile Fixed-X and Model-X equicorrelated knockoffs across feature dimensions."""
    print("--- 3. Profiling Knockoff Filter & Variable Selection ---")
    records: list[dict] = []

    for p in p_values:
        n = p * n_multiplier
        # Generate correlated features via autoregressive covariance
        rho = 0.5
        idx = np.arange(p)
        cov = rho ** np.abs(np.subtract.outer(idx, idx))
        L = np.linalg.cholesky(cov)
        X_raw = rng.standard_normal((n, p)) @ L.T
        col_names = [f"x{i}" for i in range(p)]
        X = pd.DataFrame(X_raw, columns=col_names)

        # True sparse signal: first 3 variables active
        beta = np.zeros(p)
        beta[: min(3, p)] = [2.0, -1.8, 1.5][: min(3, p)]
        y = X.values @ beta + rng.standard_normal(n) * 0.5

        # 1. Fixed-X knockoffs (Finite-sample exact FDR control)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_fixed = knockoff_test(
                X,
                y,
                method="fixed_x",
                fdr_level=0.10,
                random_state=42,
            )
        t_fixed = time.perf_counter() - t0
        k_fixed = len(res_fixed.selected_features)

        print(
            f"  Fixed-X Knockoffs  p={p:>3} n={n:>4} : {t_fixed * 1e3:6.2f} ms | "
            f"selected={k_fixed:>2}/{min(3, p)} | W_max={np.max(res_fixed.knockoff_statistics):6.3f}"
        )
        records.append(
            {
                "subsystem": "knockoff",
                "test": "fixed_x",
                "n": n,
                "dimension": p,
                "B": 0,
                "time_s": t_fixed,
                "throughput": p / t_fixed if t_fixed > 0 else 0.0,
                "statistic": float(k_fixed),
                "p_value": float(np.max(res_fixed.knockoff_statistics)),
            }
        )

        # 2. Model-X equicorrelated knockoffs
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_modelx = knockoff_test(
                X,
                y,
                method="equicorrelated",
                fdr_level=0.10,
                random_state=42,
            )
        t_modelx = time.perf_counter() - t0
        k_modelx = len(res_modelx.selected_features)

        print(
            f"  Model-X Knockoffs  p={p:>3} n={n:>4} : {t_modelx * 1e3:6.2f} ms | "
            f"selected={k_modelx:>2}/{min(3, p)} | W_max={np.max(res_modelx.knockoff_statistics):6.3f}"
        )
        records.append(
            {
                "subsystem": "knockoff",
                "test": "equicorrelated",
                "n": n,
                "dimension": p,
                "B": 0,
                "time_s": t_modelx,
                "throughput": p / t_modelx if t_modelx > 0 else 0.0,
                "statistic": float(k_modelx),
                "p_value": float(np.max(res_modelx.knockoff_statistics)),
            }
        )

    return records


# ====================================================================== #
# 4. Benchmark: Conditional Invariance Testing                           #
# ====================================================================== #


def benchmark_invariance(
    n_values: list[int],
    env_counts: list[int],
    B: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Profile conditional invariance testing across sample sizes and environment counts."""
    print("--- 4. Profiling Conditional Invariance Testing ---")
    records: list[dict] = []
    p = 4

    for n in n_values:
        for G in env_counts:
            # Generate invariant data across environments: Y = 1.2*X0 + 0.5*X1 + noise
            X_raw = rng.standard_normal((n, p))
            col_names = [f"x{i}" for i in range(p)]
            X = pd.DataFrame(X_raw, columns=col_names)
            y = X["x0"] * 1.2 + X["x1"] * 0.5 + rng.standard_normal(n) * 0.5

            # Balanced environment assignments
            envs = np.repeat(np.arange(G), n // G)
            if len(envs) < n:
                envs = np.concatenate([envs, np.full(n - len(envs), G - 1)])

            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_inv = invariance_test(
                    X,
                    y,
                    predictor="x0",
                    environments=envs,
                    method="freedman_lane",
                    n_randomizations=B,
                    random_state=42,
                )
            t_inv = time.perf_counter() - t0
            p_val = res_inv.p_value
            stat = res_inv.max_deviation

            print(
                f"  Invariance (FL)    n={n:>4} E={G:>2} B={B:>4} : {t_inv:6.3f}s | "
                f"max_dev={stat:6.3f} | p={p_val:5.3f} | {B / t_inv:6.1f} perm/s"
            )
            records.append(
                {
                    "subsystem": "invariance",
                    "test": f"env_{G}",
                    "n": n,
                    "dimension": G,
                    "B": B,
                    "time_s": t_inv,
                    "throughput": B / t_inv if t_inv > 0 else 0.0,
                    "statistic": stat,
                    "p_value": p_val,
                }
            )

    return records


# ====================================================================== #
#  Visualizations                                                        #
# ====================================================================== #


def _chart_kernel_scaling(df: pd.DataFrame) -> None:
    """Generate scaling chart for kernel hypothesis tests."""
    sub = df[df["subsystem"] == "kernel"].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for test in sub["test"].unique():
        ts = sub[sub["test"] == test].sort_values("n")
        ax.plot(ts["n"], ts["time_s"], "o-", label=test, linewidth=2, markersize=6)

    ax.set_xlabel("Sample Size (n)", fontsize=11)
    ax.set_ylabel("Wall Time (seconds)", fontsize=11)
    ax.set_title(
        "Kernel Hypothesis Tests — Computational Scaling vs Sample Size", fontsize=12
    )
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = _CHART_DIR / "kernel_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _chart_conformal_timing(df: pd.DataFrame) -> None:
    """Generate timing and coverage comparison for conformal prediction."""
    sub = df[df["subsystem"] == "conformal"].copy()
    if sub.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Latency by method (at largest n)
    max_n = sub["n"].max()
    sub_n = sub[sub["n"] == max_n].sort_values("time_s")

    ax1.bar(
        sub_n["test"],
        sub_n["time_s"] * 1e3,
        color="teal",
        edgecolor="black",
        alpha=0.85,
    )
    ax1.set_ylabel("Wall Time (ms)", fontsize=11)
    ax1.set_title(f"Conformal Calibration Latency (n={max_n})", fontsize=12)
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(True, alpha=0.3, axis="y")

    # Coverage vs Nominal
    ax2.bar(
        sub_n["test"],
        sub_n["p_value"],
        color="darkseagreen",
        edgecolor="black",
        alpha=0.85,
    )
    ax2.axhline(0.90, color="red", linestyle="--", label="Nominal 90% Coverage")
    ax2.set_ylabel("Empirical Out-of-Sample Coverage", fontsize=11)
    ax2.set_title(f"Empirical Coverage Rate (n={max_n})", fontsize=12)
    ax2.set_ylim([0.75, 1.0])
    ax2.tick_params(axis="x", rotation=25)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = _CHART_DIR / "conformal_timing.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _chart_knockoff_scaling(df: pd.DataFrame) -> None:
    """Generate scaling chart for knockoff generation."""
    sub = df[df["subsystem"] == "knockoff"].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for test in sub["test"].unique():
        ts = sub[sub["test"] == test].sort_values("dimension")
        ax.plot(
            ts["dimension"],
            ts["time_s"] * 1e3,
            "o-",
            label=f"{test} knockoffs",
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Feature Count (p)", fontsize=11)
    ax.set_ylabel("Wall Time (ms)", fontsize=11)
    ax.set_title(
        "Knockoff Filter Variable Selection — Scaling vs Dimension (p)", fontsize=12
    )
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = _CHART_DIR / "knockoff_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _chart_invariance_scaling(df: pd.DataFrame) -> None:
    """Generate scaling chart for conditional invariance testing."""
    sub = df[df["subsystem"] == "invariance"].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for env_name in sub["test"].unique():
        ts = sub[sub["test"] == env_name].sort_values("n")
        ax.plot(
            ts["n"],
            ts["time_s"],
            "o-",
            label=f"{env_name} environments",
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Sample Size (n)", fontsize=11)
    ax.set_ylabel("Wall Time (seconds)", fontsize=11)
    ax.set_title(
        "Conditional Invariance Test — Wall Time vs Sample Size (n)", fontsize=12
    )
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = _CHART_DIR / "invariance_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ====================================================================== #
#  Main Runner                                                           #
# ====================================================================== #


def main() -> None:
    parser = argparse.ArgumentParser(description="Subsystems benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run reduced grids for faster smoke-test execution",
    )
    args = parser.parse_args()

    _ensure_dirs()
    rng = np.random.default_rng(42)

    if args.quick:
        kernel_n = [100, 200]
        kernel_B = 100
        conformal_n = [100, 200]
        conformal_new = 50
        knockoff_p = [10, 25]
        invariance_n = [150, 300]
        invariance_envs = [2, 4]
        invariance_B = 100
    else:
        kernel_n = [100, 250, 500, 1000]
        kernel_B = 500
        conformal_n = [100, 250, 500, 1000]
        conformal_new = 200
        knockoff_p = [10, 25, 50, 100]
        invariance_n = [200, 500, 1000]
        invariance_envs = [2, 4, 8]
        invariance_B = 500

    print("=" * 80)
    print("RANDOMIZATION TESTS — SUBSYSTEMS BENCHMARK SUITE")
    print("=" * 80)
    print(f"Python:   {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Mode:     {'QUICK (smoke-test)' if args.quick else 'FULL (comprehensive)'}")
    print()

    all_records: list[dict] = []
    all_records.extend(benchmark_kernel_tests(kernel_n, kernel_B, rng))
    print()
    all_records.extend(benchmark_conformal(conformal_n, conformal_new, rng))
    print()
    all_records.extend(benchmark_knockoffs(knockoff_p, n_multiplier=5, rng=rng))
    print()
    all_records.extend(
        benchmark_invariance(invariance_n, invariance_envs, invariance_B, rng)
    )
    print()

    df = pd.DataFrame(all_records)
    csv_path = _RESULTS_DIR / "subsystems_profile.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} benchmark records to {csv_path}")

    print("\nGenerating charts...")
    _chart_kernel_scaling(df)
    _chart_conformal_timing(df)
    _chart_knockoff_scaling(df)
    _chart_invariance_scaling(df)
    print("\nAll subsystem benchmarks completed successfully.")


if __name__ == "__main__":
    main()

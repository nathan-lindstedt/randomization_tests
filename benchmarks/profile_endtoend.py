"""End-to-end engine benchmark — full permutation_test_regression() pipeline.

Measures wall time for the complete pipeline (permutation generation →
model fitting → scoring → p-value computation) across realistic
scenarios with varying predictor counts (p), observation counts (n),
group structures, families, and permutation strategies.

Key questions this benchmark answers
-------------------------------------
1. Does predictor count *p* dominate runtime over permutation strategy
   overhead?
2. At what n×p does permutation generation cost become negligible
   relative to model fitting?
3. Is within-cell / two-stage overhead observable when model fitting
   dominates?
4. How do GLM families compare at matched dimensions?

Usage::

    conda activate randomization-tests
    python benchmarks/profile_endtoend.py          # full suite (~15 min)
    python benchmarks/profile_endtoend.py --quick  # reduced (~3 min)

Outputs:
    benchmarks/results/endtoend_profile.csv
    benchmarks/image/endtoend-profile/time_by_p.png
    benchmarks/image/endtoend-profile/strategy_overhead.png
    benchmarks/image/endtoend-profile/family_comparison.png
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure the package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from randomization_tests import permutation_test_regression  # noqa: E402

# ====================================================================== #
#  Scenario definitions                                                   #
# ====================================================================== #
#
# Each scenario is a dict with:
#   name:     Human-readable label
#   n:        Number of observations
#   p:        Number of predictor columns (the treatment column is
#             always appended, so total columns = p + 1)
#   G:        Number of balanced groups (cell_size = n // G)
#   family:   Model family string
#   method:   Permutation method
#   strategies: List of permutation strategies to test

SCENARIOS: list[dict] = [
    # ---- Linear family: varying p at small n -------------------------
    {
        "name": "linear_n100_p5",
        "n": 100,
        "p": 5,
        "G": 5,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    {
        "name": "linear_n100_p10",
        "n": 100,
        "p": 10,
        "G": 5,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    {
        "name": "linear_n100_p20",
        "n": 100,
        "p": 20,
        "G": 5,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    {
        "name": "linear_n100_p30",
        "n": 100,
        "p": 30,
        "G": 5,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    # ---- Linear family: varying p at medium n ------------------------
    {
        "name": "linear_n500_p5",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    {
        "name": "linear_n500_p10",
        "n": 500,
        "p": 10,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    {
        "name": "linear_n500_p20",
        "n": 500,
        "p": 20,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    {
        "name": "linear_n500_p30",
        "n": 500,
        "p": 30,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    # ---- GLM families at medium n, p=10 ------------------------------
    {
        "name": "logistic_n500_p10",
        "n": 500,
        "p": 10,
        "G": 10,
        "family": "logistic",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    {
        "name": "poisson_n500_p10",
        "n": 500,
        "p": 10,
        "G": 10,
        "family": "poisson",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    # ---- Linear family: varying p at large n -------------------------
    {
        "name": "linear_n1000_p5",
        "n": 1000,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    {
        "name": "linear_n1000_p10",
        "n": 1000,
        "p": 10,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    {
        "name": "linear_n1000_p20",
        "n": 1000,
        "p": 20,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    {
        "name": "linear_n1000_p30",
        "n": 1000,
        "p": 30,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["global", "within", "two-stage"],
    },
    # ---- Joint tests (Kennedy joint) at medium n --------------------
    {
        "name": "joint_linear_n500_p10",
        "n": 500,
        "p": 10,
        "G": 10,
        "family": "linear",
        "method": "kennedy_joint",
        "strategies": ["global", "within", "two-stage"],
    },
    {
        "name": "joint_logistic_n500_p10",
        "n": 500,
        "p": 10,
        "G": 10,
        "family": "logistic",
        "method": "kennedy_joint",
        "strategies": ["global", "within", "two-stage"],
    },
]

# Quick-mode keeps a representative subset
QUICK_NAMES = {
    "linear_n100_p5",
    "linear_n100_p30",
    "linear_n500_p10",
    "linear_n500_p30",
    "logistic_n500_p10",
    "linear_n1000_p20",
    "joint_linear_n500_p10",
}

# ---- Shared constants ------------------------------------------------

B_VALUES = [999]
B_VALUES_FULL = [999]
RANDOM_STATE = 42
REPEATS = 3


# ====================================================================== #
#  Data generation                                                        #
# ====================================================================== #


def _generate_data(
    scenario: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Generate synthetic data for one scenario.

    Returns
    -------
    X_df : DataFrame of shape (n, p+1) with named columns
    y_df : DataFrame of shape (n, 1) with response values
    groups : (n,) integer array of balanced group labels
    confounders : list of confounder column names (first p columns)
    """
    n = scenario["n"]
    p = scenario["p"]
    G = scenario["G"]
    family = scenario["family"]

    # ---- Design matrix: p covariates + 1 treatment -------------------
    covariates = rng.standard_normal((n, p))
    treatment = rng.standard_normal((n, 1))
    X = np.hstack([covariates, treatment])

    col_names = [f"X{i}" for i in range(1, p + 1)] + ["treatment"]
    X_df = pd.DataFrame(X, columns=col_names)

    # ---- True coefficients (sparse) ----------------------------------
    beta = np.zeros(p + 1)
    beta[: min(3, p)] = [0.5, -0.3, 0.2][: min(3, p)]
    beta[-1] = 0.4  # treatment effect

    # ---- Linear predictor + noise ------------------------------------
    eta = X @ beta + rng.standard_normal(n) * 0.5

    # ---- Response (family-specific) ----------------------------------
    if family == "linear":
        y = eta
    elif family == "logistic":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob).astype(float)
    elif family == "poisson":
        eta_clamped = np.clip(eta, -5, 3)
        mu = np.exp(eta_clamped)
        y = rng.poisson(mu).astype(float)
    else:
        msg = f"Unknown family: {family}"
        raise ValueError(msg)

    y_df = pd.DataFrame({"y": y})

    # ---- Balanced group labels ---------------------------------------
    cell_size = n // G
    groups = np.repeat(np.arange(G), cell_size)
    if len(groups) < n:
        groups = np.concatenate([groups, np.full(n - len(groups), G - 1)])
    groups = groups.astype(int)

    # Confounders = all covariate columns (not treatment)
    confounders = [f"X{i}" for i in range(1, p + 1)] if p > 0 else []

    return X_df, y_df, groups, confounders


# ====================================================================== #
#  Single-run executor                                                    #
# ====================================================================== #


def _run_single(
    scenario: dict,
    strategy: str,
    B: int,
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    groups: np.ndarray,
    confounders: list[str],
) -> dict:
    """Run permutation_test_regression for one scenario/strategy/B combo.

    Returns a dict of timing + metadata for the results table.
    """
    method = scenario["method"]

    # Determine arguments based on strategy
    kwargs: dict = {
        "X": X_df,
        "y": y_df,
        "n_permutations": B,
        "method": method,
        "random_state": RANDOM_STATE,
        "family": scenario["family"],
    }

    # For kennedy_joint and freedman_lane, confounders are needed
    if method in ("kennedy", "kennedy_joint", "freedman_lane", "freedman_lane_joint"):
        kwargs["confounders"] = confounders if confounders else None

    if strategy == "global":
        # No groups, no strategy
        pass
    else:
        kwargs["groups"] = groups
        kwargs["permutation_strategy"] = strategy

    # Suppress warnings during benchmarking (convergence, n_jobs, etc.)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_test_regression(**kwargs)
    elapsed = time.perf_counter() - t0

    # Extract result type to verify it worked
    result_type = type(result).__name__

    return {
        "scenario": scenario["name"],
        "family": scenario["family"],
        "method": method,
        "strategy": strategy,
        "n": scenario["n"],
        "p": scenario["p"],
        "G": scenario["G"],
        "B": B,
        "time_s": elapsed,
        "result_type": result_type,
    }


# ====================================================================== #
#  Benchmark runner                                                       #
# ====================================================================== #


def run_benchmark(scenarios: list[dict], b_values: list[int]) -> pd.DataFrame:
    """Run all scenario × strategy × B combinations and return results."""
    rng = np.random.default_rng(RANDOM_STATE)

    # Count total runs for progress display
    total = sum(len(s["strategies"]) * len(b_values) * REPEATS for s in scenarios)

    print("=" * 72)
    print("END-TO-END ENGINE BENCHMARK")
    print("=" * 72)
    print(f"Python:   {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"B values: {b_values}")
    print(f"Repeats:  {REPEATS}")
    print(f"Scenarios: {len(scenarios)}")
    print()

    # Scenario overview
    print("Scenario overview:")
    print(
        f"  {'Name':<30} {'n':>5} {'p':>3} {'G':>3} {'Family':<10} "
        f"{'Method':<20} Strategies"
    )
    print(f"  {'-' * 100}")
    for s in scenarios:
        strats = ", ".join(s["strategies"])
        print(
            f"  {s['name']:<30} {s['n']:>5} {s['p']:>3} {s['G']:>3} "
            f"{s['family']:<10} {s['method']:<20} {strats}"
        )
    print()

    print("Running benchmarks...")
    print("-" * 72)

    rows: list[dict] = []
    run_idx = 0

    for s in scenarios:
        # Generate data once per scenario (same data for all strategies)
        X_df, y_df, groups, confounders = _generate_data(s, rng)

        for strategy in s["strategies"]:
            for B in b_values:
                times = []
                for _rep in range(REPEATS):
                    row = _run_single(s, strategy, B, X_df, y_df, groups, confounders)
                    times.append(row["time_s"])
                    run_idx += 1

                # Use median time across repeats
                median_time = float(np.median(times))
                row_final = {
                    "scenario": s["name"],
                    "family": s["family"],
                    "method": s["method"],
                    "strategy": strategy,
                    "n": s["n"],
                    "p": s["p"],
                    "G": s["G"],
                    "B": B,
                    "time_s": median_time,
                    "time_min": min(times),
                    "time_max": max(times),
                    "result_type": row["result_type"],
                }
                rows.append(row_final)

                status = "OK"
                print(
                    f"  [{run_idx:>4}/{total}] "
                    f"{s['name']:<30} {strategy:<12} "
                    f"n={s['n']:>5}  p={s['p']:>2}  G={s['G']:>2}  "
                    f"B={B:>5}  time={median_time:>8.4f}s  [{status}]"
                )

    print("-" * 72)
    print(f"Total scenario runs: {len(rows)}")
    print()

    return pd.DataFrame(rows)


# ====================================================================== #
#  Chart generation                                                       #
# ====================================================================== #

_CHART_DIR = Path(__file__).resolve().parent / "image" / "endtoend-profile"


def _ensure_chart_dir() -> None:
    _CHART_DIR.mkdir(parents=True, exist_ok=True)


def _chart_time_by_p(df: pd.DataFrame) -> None:
    """Wall time vs. predictor count (p) for linear family, by strategy.

    Shows how model-fitting cost scales with p, and whether strategy
    overhead is visible at each p.
    """
    _ensure_chart_dir()

    # Filter to linear family, largest B, individual tests only
    mask = (
        (df["family"] == "linear")
        & (df["B"] == df["B"].max())
        & (~df["method"].str.contains("joint"))
    )
    sub = df[mask].copy()

    if sub.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, n_val in zip(axes, sorted(sub["n"].unique()), strict=True):
        ns = sub[sub["n"] == n_val]
        for strategy in ["global", "within", "two-stage"]:
            ss = ns[ns["strategy"] == strategy].sort_values("p")
            if not ss.empty:
                ax.plot(ss["p"], ss["time_s"], "o-", label=strategy)

        ax.set_xlabel("Number of predictors (p)")
        ax.set_ylabel("Wall time (s)")
        ax.set_title(f"n = {n_val}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Wall Time vs. Predictor Count — Linear Family", fontsize=14)
    fig.tight_layout()
    path = _CHART_DIR / "time_by_p.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _chart_strategy_overhead(df: pd.DataFrame) -> None:
    """Strategy overhead ratio (within/global, two-stage/global).

    Shows at what n×p the permutation strategy overhead becomes
    negligible relative to model fitting.
    """
    _ensure_chart_dir()

    # Filter to linear family, largest B, individual tests
    mask = (
        (df["family"] == "linear")
        & (df["B"] == df["B"].max())
        & (~df["method"].str.contains("joint"))
    )
    sub = df[mask].copy()

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute overhead ratio vs. global for each scenario
    overhead_rows = []
    for _, row in sub.iterrows():
        if row["strategy"] == "global":
            continue
        global_row = sub[
            (sub["scenario"] == row["scenario"]) & (sub["strategy"] == "global")
        ]
        if not global_row.empty:
            ratio = row["time_s"] / global_row.iloc[0]["time_s"]
            overhead_rows.append(
                {
                    "scenario": row["scenario"],
                    "strategy": row["strategy"],
                    "n": row["n"],
                    "p": row["p"],
                    "n_x_p": row["n"] * row["p"],
                    "overhead": ratio,
                }
            )

    if not overhead_rows:
        plt.close(fig)
        return

    oh_df = pd.DataFrame(overhead_rows)

    for strategy in ["within", "two-stage"]:
        ss = oh_df[oh_df["strategy"] == strategy].sort_values("n_x_p")
        if not ss.empty:
            labels = [f"n{r['n']}p{r['p']}" for _, r in ss.iterrows()]
            ax.plot(range(len(ss)), ss["overhead"], "o-", label=strategy)
            ax.set_xticks(range(len(ss)))
            ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="global (1.0×)")
    ax.set_ylabel("Overhead ratio (strategy / global)")
    ax.set_xlabel("Scenario (sorted by n × p)")
    ax.set_title("Strategy Overhead vs. Global — Linear Family")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = _CHART_DIR / "strategy_overhead.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _chart_family_comparison(df: pd.DataFrame) -> None:
    """Compare families at matched dimensions (n=500, p=5).

    Shows runtime differences between linear, logistic, and Poisson
    families at the same data size, across strategies.
    """
    _ensure_chart_dir()

    # Filter to n=500, p=5, largest B, individual tests
    mask = (
        (df["n"] == 500)
        & (df["p"] == 5)
        & (df["B"] == df["B"].max())
        & (~df["method"].str.contains("joint"))
    )
    sub = df[mask].copy()

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    families = sub["family"].unique()
    strategies = ["global", "within", "two-stage"]
    x = np.arange(len(strategies))
    width = 0.8 / len(families)

    for i, fam in enumerate(sorted(families)):
        fs = sub[sub["family"] == fam]
        times = []
        for strat in strategies:
            ss = fs[fs["strategy"] == strat]
            times.append(ss["time_s"].values[0] if not ss.empty else 0)
        ax.bar(x + i * width, times, width, label=fam)

    ax.set_xticks(x + width * (len(families) - 1) / 2)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("Wall time (s)")
    ax.set_xlabel("Permutation strategy")
    ax.set_title("Family Comparison at n=500, p=5")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = _CHART_DIR / "family_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _print_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary table of results."""
    print("=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    # Group by scenario for cleaner output
    max_b = df["B"].max()
    sub = df[df["B"] == max_b].copy()

    print(
        f"  {'Scenario':<30} {'Strat':<12} {'Family':<10} "
        f"{'n':>5} {'p':>3} {'B':>5} {'Time':>8}  {'Type'}"
    )
    print(f"  {'-' * 95}")

    for _, row in sub.iterrows():
        print(
            f"  {row['scenario']:<30} {row['strategy']:<12} "
            f"{row['family']:<10} {row['n']:>5} {row['p']:>3} "
            f"{row['B']:>5} {row['time_s']:>8.4f}  {row['result_type']}"
        )

    print("=" * 72)


# ====================================================================== #
#  Main                                                                   #
# ====================================================================== #


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end engine benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced scenario set for faster iteration",
    )
    args = parser.parse_args()

    if args.quick:
        scenarios = [s for s in SCENARIOS if s["name"] in QUICK_NAMES]
        b_values = B_VALUES
    else:
        scenarios = SCENARIOS
        b_values = B_VALUES_FULL

    df = run_benchmark(scenarios, b_values)

    # ---- Save CSV ------------------------------------------------
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "endtoend_profile.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # ---- Generate charts -----------------------------------------
    print("Generating charts...")
    _chart_time_by_p(df)
    _chart_strategy_overhead(df)
    _chart_family_comparison(df)

    # ---- Summary table -------------------------------------------
    _print_summary(df)


if __name__ == "__main__":
    main()

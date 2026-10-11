"""End-to-end engine benchmark — full randomization_test_regression() pipeline.

Measures wall time, throughput (permutations per second), and algorithmic scaling
for the complete pipeline across all 7 model families (linear, logistic,
poisson, negative_binomial, ordinal, multinomial, linear_mixed), all statistical
methods (score, score_joint, freedman_lane, freedman_lane_joint, ter_braak,
kennedy, kennedy_joint, manly, manly_joint, sign_flip), and all canonical
permutation strategies (unrestricted, within, between, two-stage).

Key questions this benchmark answers
-------------------------------------
1. Method Shootout: Does the null tangent-space Score test achieve 10x-50x
   throughput speedup over full-refit Freedman-Lane and Manly methods?
2. Family Scaling: How do generalized linear, discrete overdispersed, ordinal,
   multinomial, and random-intercept mixed models compare at matched dimensions?
3. Predictor Scaling: How does wall time scale with predictor count (p) and sample
   size (n) under linear and GLM families?
4. Strategy Overhead: What is the empirical cost of within-cell, between-cell,
   and two-stage multi-level permutation vs flat unrestricted shuffling?
5. Randomization Mode: What is the performance characteristic of Rademacher
   sign-flipping vs full combinatorial permutation?

Usage::

    conda activate randomization-tests
    python benchmarks/profile_endtoend.py          # full suite (~10 min)
    python benchmarks/profile_endtoend.py --quick  # focused smoke test (<15 sec)

Outputs:
    benchmarks/results/endtoend_profile.csv
    benchmarks/image/endtoend-profile/method_comparison.png
    benchmarks/image/endtoend-profile/family_comparison.png
    benchmarks/image/endtoend-profile/strategy_overhead.png
    benchmarks/image/endtoend-profile/time_by_p.png
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

# Ensure the package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from randomization_tests import randomization_test_regression  # noqa: E402

# ====================================================================== #
#  Scenario definitions                                                   #
# ====================================================================== #

SCENARIOS: list[dict] = [
    # ------------------------------------------------------------------ #
    # 1. Method Shootout at matched dimensions (n=500, p=5, G=10)        #
    # ------------------------------------------------------------------ #
    {
        "name": "shootout_score",
        "category": "shootout",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "score",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "shootout_score_joint",
        "category": "shootout",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "score_joint",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "shootout_freedman_lane",
        "category": "shootout",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "shootout_freedman_lane_joint",
        "category": "shootout",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane_joint",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "shootout_ter_braak",
        "category": "shootout",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "ter_braak",
        "strategies": ["unrestricted"],
        "confounders": False,
    },
    {
        "name": "shootout_kennedy",
        "category": "shootout",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "kennedy",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "shootout_kennedy_joint",
        "category": "shootout",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "kennedy_joint",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "shootout_manly",
        "category": "shootout",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "manly",
        "strategies": ["unrestricted"],
        "confounders": False,
    },
    {
        "name": "shootout_manly_joint",
        "category": "shootout",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "manly_joint",
        "strategies": ["unrestricted"],
        "confounders": False,
    },
    {
        "name": "shootout_sign_flip",
        "category": "shootout",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "randomization": "sign_flip",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    # ------------------------------------------------------------------ #
    # 2. Family Comparison across all 7 families (n=500, p=5, G=10)      #
    # ------------------------------------------------------------------ #
    {
        "name": "family_linear",
        "category": "family_refit",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "family_logistic",
        "category": "family_refit",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "logistic",
        "method": "freedman_lane",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "family_poisson",
        "category": "family_refit",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "poisson",
        "method": "freedman_lane",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "family_negative_binomial",
        "category": "family_refit",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "negative_binomial",
        "method": "freedman_lane",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "family_ordinal",
        "category": "family_refit",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "ordinal",
        "method": "manly",
        "strategies": ["unrestricted"],
        "confounders": False,
    },
    {
        "name": "family_multinomial",
        "category": "family_refit",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "multinomial",
        "method": "manly",
        "strategies": ["unrestricted"],
        "confounders": False,
    },
    {
        "name": "family_linear_mixed",
        "category": "family_refit",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear_mixed",
        "method": "ter_braak",
        "strategies": ["within"],
        "confounders": False,
    },
    # Score tests across all families
    {
        "name": "score_linear",
        "category": "family_score",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "score",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "score_logistic",
        "category": "family_score",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "logistic",
        "method": "score",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "score_poisson",
        "category": "family_score",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "poisson",
        "method": "score",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "score_negative_binomial",
        "category": "family_score",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "negative_binomial",
        "method": "score",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "score_ordinal",
        "category": "family_score",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "ordinal",
        "method": "score",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "score_multinomial",
        "category": "family_score",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "multinomial",
        "method": "score",
        "strategies": ["unrestricted"],
        "confounders": True,
    },
    {
        "name": "score_linear_mixed",
        "category": "family_score",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear_mixed",
        "method": "score",
        "strategies": ["within"],
        "confounders": True,
    },
    # ------------------------------------------------------------------ #
    # 3. Strategy Overhead & Comparison (n=500, p=10, G=10)              #
    # ------------------------------------------------------------------ #
    {
        "name": "strategy_linear",
        "category": "strategy",
        "n": 500,
        "p": 10,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "between", "two-stage"],
        "confounders": True,
    },
    # ------------------------------------------------------------------ #
    # 4. Predictor Scaling: varying p across small/medium/large n        #
    # ------------------------------------------------------------------ #
    {
        "name": "linear_n100_p5",
        "category": "dimension",
        "n": 100,
        "p": 5,
        "G": 5,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "two-stage"],
        "confounders": True,
    },
    {
        "name": "linear_n100_p10",
        "category": "dimension",
        "n": 100,
        "p": 10,
        "G": 5,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "two-stage"],
        "confounders": True,
    },
    {
        "name": "linear_n100_p20",
        "category": "dimension",
        "n": 100,
        "p": 20,
        "G": 5,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "two-stage"],
        "confounders": True,
    },
    {
        "name": "linear_n100_p30",
        "category": "dimension",
        "n": 100,
        "p": 30,
        "G": 5,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "two-stage"],
        "confounders": True,
    },
    {
        "name": "linear_n500_p5",
        "category": "dimension",
        "n": 500,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "two-stage"],
        "confounders": True,
    },
    {
        "name": "linear_n500_p10",
        "category": "dimension",
        "n": 500,
        "p": 10,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "two-stage"],
        "confounders": True,
    },
    {
        "name": "linear_n500_p20",
        "category": "dimension",
        "n": 500,
        "p": 20,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "two-stage"],
        "confounders": True,
    },
    {
        "name": "linear_n500_p30",
        "category": "dimension",
        "n": 500,
        "p": 30,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "two-stage"],
        "confounders": True,
    },
    {
        "name": "linear_n1000_p5",
        "category": "dimension",
        "n": 1000,
        "p": 5,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "two-stage"],
        "confounders": True,
    },
    {
        "name": "linear_n1000_p10",
        "category": "dimension",
        "n": 1000,
        "p": 10,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "two-stage"],
        "confounders": True,
    },
    {
        "name": "linear_n1000_p20",
        "category": "dimension",
        "n": 1000,
        "p": 20,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "two-stage"],
        "confounders": True,
    },
    {
        "name": "linear_n1000_p30",
        "category": "dimension",
        "n": 1000,
        "p": 30,
        "G": 10,
        "family": "linear",
        "method": "freedman_lane",
        "strategies": ["unrestricted", "within", "two-stage"],
        "confounders": True,
    },
]

# Quick-mode keeps a representative subset covering all key domains
QUICK_NAMES = {
    "shootout_score",
    "shootout_freedman_lane",
    "shootout_manly",
    "shootout_sign_flip",
    "family_linear",
    "family_logistic",
    "family_poisson",
    "family_negative_binomial",
    "family_ordinal",
    "family_multinomial",
    "family_linear_mixed",
    "score_linear",
    "score_logistic",
    "linear_n100_p5",
    "linear_n500_p10",
    "strategy_linear",
}

# Shared constants
RANDOM_STATE = 42
B_VALUES_QUICK = [200]
B_VALUES_FULL = [999]
REPEATS_QUICK = 1
REPEATS_FULL = 3


# ====================================================================== #
#  Data generation                                                        #
# ====================================================================== #


def _generate_data(
    scenario: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Generate synthetic data tailored to each model family's assumptions.

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

    # Design matrix: p covariates + 1 treatment
    covariates = rng.standard_normal((n, p))
    treatment = rng.standard_normal((n, 1))
    X = np.hstack([covariates, treatment])

    col_names = [f"X{i}" for i in range(1, p + 1)] + ["treatment"]
    X_df = pd.DataFrame(X, columns=col_names)

    # True coefficients (sparse)
    beta = np.zeros(p + 1)
    beta[: min(3, p)] = [0.5, -0.3, 0.2][: min(3, p)]
    beta[-1] = 0.4  # treatment effect

    # Balanced group labels
    cell_size = n // G
    groups = np.repeat(np.arange(G), cell_size)
    if len(groups) < n:
        groups = np.concatenate([groups, np.full(n - len(groups), G - 1)])
    groups = groups.astype(int)

    # Linear predictor + noise
    eta = X @ beta + rng.standard_normal(n) * 0.4

    # Response generation
    if family == "linear":
        y = eta
    elif family == "logistic":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob).astype(float)
    elif family == "poisson":
        mu = np.exp(np.clip(eta, -5.0, 3.0))
        y = rng.poisson(mu).astype(float)
    elif family == "negative_binomial":
        alpha = 1.2
        mu = np.exp(np.clip(eta, -5.0, 3.0))
        n_param = 1.0 / alpha
        p_param = 1.0 / (1.0 + alpha * mu)
        y = rng.negative_binomial(n=n_param, p=p_param).astype(float)
    elif family == "ordinal":
        z = eta + rng.logistic(0, 1, size=n)
        q1, q2 = np.quantile(z, [0.33, 0.67])
        y = np.digitize(z, bins=[q1, q2]).astype(float)
    elif family == "multinomial":
        logits = np.column_stack(
            [
                np.zeros(n),
                eta,
                -0.5 * eta + rng.standard_normal(n) * 0.3,
            ]
        )
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        y = np.array([rng.choice(3, p=probs[i]) for i in range(n)], dtype=float)
    elif family == "linear_mixed":
        u = rng.standard_normal(G) * 0.6
        y = eta + u[groups]
    else:
        msg = f"Unknown family: {family}"
        raise ValueError(msg)

    y_df = pd.DataFrame({"y": y})
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
    """Run randomization_test_regression for one scenario/strategy/B combo."""
    method = scenario["method"]
    family = scenario["family"]
    randomization = scenario.get("randomization", "permute")
    use_confounders = scenario.get("confounders", True)

    kwargs: dict = {
        "X": X_df,
        "y": y_df,
        "n_randomizations": B,
        "method": method,
        "family": family,
        "random_state": RANDOM_STATE,
        "randomization": randomization,
    }

    if (
        use_confounders
        and confounders
        and method
        in (
            "kennedy",
            "kennedy_joint",
            "freedman_lane",
            "freedman_lane_joint",
            "score",
            "score_joint",
        )
    ):
        kwargs["confounders"] = confounders

    # Strategy / Groups wiring
    if family == "linear_mixed":
        kwargs["groups"] = groups
        if strategy != "unrestricted":
            kwargs["permutation_strategy"] = strategy
    elif strategy == "unrestricted":
        # Flat unconstrained permutation (groups omitted)
        pass
    else:
        kwargs["groups"] = groups
        kwargs["permutation_strategy"] = strategy

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = randomization_test_regression(**kwargs)
    elapsed = time.perf_counter() - t0

    result_type = type(result).__name__

    return {
        "scenario": scenario["name"],
        "category": scenario.get("category", "general"),
        "family": family,
        "method": method,
        "strategy": strategy,
        "randomization": randomization,
        "n": scenario["n"],
        "p": scenario["p"],
        "G": scenario["G"],
        "B": B,
        "time_s": elapsed,
        "perms_per_sec": B / elapsed if elapsed > 0 else 0.0,
        "result_type": result_type,
    }


# ====================================================================== #
#  Benchmark runner                                                       #
# ====================================================================== #


def run_benchmark(
    scenarios: list[dict],
    b_values: list[int],
    repeats: int = 1,
) -> pd.DataFrame:
    """Run all scenario x strategy x B combinations and return results."""
    rng = np.random.default_rng(RANDOM_STATE)
    total = sum(len(s["strategies"]) * len(b_values) * repeats for s in scenarios)

    print("=" * 80)
    print("END-TO-END ENGINE BENCHMARK")
    print("=" * 80)
    print(f"Python:    {sys.version.split()[0]}")
    print(f"Platform:  {platform.platform()}")
    print(f"B values:  {b_values}")
    print(f"Repeats:   {repeats}")
    print(f"Scenarios: {len(scenarios)}")
    print()

    print("Running benchmarks...")
    print("-" * 80)

    rows: list[dict] = []
    run_idx = 0

    for s in scenarios:
        X_df, y_df, groups, confounders = _generate_data(s, rng)

        for strategy in s["strategies"]:
            for B in b_values:
                times = []
                for _rep in range(repeats):
                    row = _run_single(s, strategy, B, X_df, y_df, groups, confounders)
                    times.append(row["time_s"])
                    run_idx += 1

                median_time = float(np.median(times))
                throughput = B / median_time if median_time > 0 else 0.0

                row_final = {
                    "scenario": s["name"],
                    "category": s.get("category", "general"),
                    "family": s["family"],
                    "method": s["method"],
                    "strategy": strategy,
                    "randomization": s.get("randomization", "permute"),
                    "n": s["n"],
                    "p": s["p"],
                    "G": s["G"],
                    "B": B,
                    "time_s": median_time,
                    "time_min": min(times),
                    "time_max": max(times),
                    "perms_per_sec": throughput,
                    "result_type": row["result_type"],
                }
                rows.append(row_final)

                print(
                    f"  [{run_idx:>4}/{total}] "
                    f"{s['name']:<28} {strategy:<12} "
                    f"n={s['n']:>5}  p={s['p']:>2}  B={B:>4}  "
                    f"time={median_time:>7.4f}s ({throughput:>7.1f} perm/s)"
                )

    print("-" * 80)
    print(f"Total scenario configurations completed: {len(rows)}")
    print()

    return pd.DataFrame(rows)


# ====================================================================== #
#  Chart generation                                                       #
# ====================================================================== #

_CHART_DIR = Path(__file__).resolve().parent / "image" / "endtoend-profile"


def _ensure_chart_dir() -> None:
    _CHART_DIR.mkdir(parents=True, exist_ok=True)


def _chart_method_comparison(df: pd.DataFrame) -> None:
    """Shootout chart comparing all methods at n=500, p=5."""
    _ensure_chart_dir()
    sub = df[df["category"] == "shootout"].copy()
    if sub.empty:
        return

    sub = sub.sort_values("time_s", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    y_pos = np.arange(len(sub))

    bars = ax.barh(
        y_pos, sub["time_s"], color="steelblue", edgecolor="black", alpha=0.85
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["scenario"])
    ax.set_xlabel("Wall time (seconds) — lower is faster")
    ax.set_title(
        "Method Shootout at Matched Dimensions (n=500, p=5, B=max)", fontsize=13
    )
    ax.grid(True, alpha=0.3, axis="x")

    for bar, (_, row) in zip(bars, sub.iterrows(), strict=True):
        width = bar.get_width()
        throughput = row["perms_per_sec"]
        ax.text(
            width + max(sub["time_s"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}s ({throughput:.0f} p/s)",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    path = _CHART_DIR / "method_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _chart_family_comparison(df: pd.DataFrame) -> None:
    """Compare families at matched dimensions across refit and score methods."""
    _ensure_chart_dir()
    sub = df[df["category"].isin(["family_refit", "family_score"])].copy()
    if sub.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    refit_sub = sub[sub["category"] == "family_refit"].sort_values("time_s")
    if not refit_sub.empty:
        ax1.barh(
            refit_sub["family"],
            refit_sub["time_s"],
            color="coral",
            edgecolor="black",
            alpha=0.85,
        )
        ax1.set_title("Refit / Exact Methods (FL, TB, Manly)", fontsize=12)
        ax1.set_xlabel("Wall time (s)")
        ax1.grid(True, alpha=0.3, axis="x")

    score_sub = sub[sub["category"] == "family_score"].sort_values("time_s")
    if not score_sub.empty:
        ax2.barh(
            score_sub["family"],
            score_sub["time_s"],
            color="mediumseagreen",
            edgecolor="black",
            alpha=0.85,
        )
        ax2.set_title("Score Test (Tangent Space Null Update)", fontsize=12)
        ax2.set_xlabel("Wall time (s)")
        ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Family Runtime Comparison at n=500, p=5", fontsize=14)
    fig.tight_layout()
    path = _CHART_DIR / "family_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _chart_strategy_overhead(df: pd.DataFrame) -> None:
    """Strategy overhead ratio relative to unrestricted permutation."""
    _ensure_chart_dir()
    sub = df[df["category"] == "strategy"].copy()
    if sub.empty:
        # Fall back to dimension scenarios with multiple strategies
        sub = df[(df["family"] == "linear") & (df["p"] == 10)].copy()

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    strats = sub["strategy"].unique()
    times = [sub[sub["strategy"] == st]["time_s"].mean() for st in strats]

    baseline = min(times) if times else 1.0
    ratios = [t / baseline for t in times]

    bars = ax.bar(strats, ratios, color="royalblue", edgecolor="black", alpha=0.85)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Baseline (1.0x)")
    ax.set_ylabel("Overhead Ratio (relative to fastest)")
    ax.set_title("Permutation Strategy Overhead (n=500, p=10, G=10)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    for bar, ratio in zip(bars, ratios, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{ratio:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    path = _CHART_DIR / "strategy_overhead.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _chart_time_by_p(df: pd.DataFrame) -> None:
    """Wall time vs predictor count (p) for linear family by strategy."""
    _ensure_chart_dir()
    sub = df[df["category"] == "dimension"].copy()
    if sub.empty:
        return

    n_values = sorted(sub["n"].unique())
    fig, axes = plt.subplots(
        1, len(n_values), figsize=(5 * len(n_values), 5), sharey=False
    )
    if len(n_values) == 1:
        axes = [axes]

    for ax, n_val in zip(axes, n_values, strict=True):
        ns = sub[sub["n"] == n_val]
        for strategy in sorted(ns["strategy"].unique()):
            ss = ns[ns["strategy"] == strategy].sort_values("p")
            if not ss.empty:
                ax.plot(ss["p"], ss["time_s"], "o-", label=strategy)

        ax.set_xlabel("Number of predictors (p)")
        ax.set_ylabel("Wall time (s)")
        ax.set_title(f"n = {n_val}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Wall Time vs. Predictor Count (p) — Linear Family", fontsize=14)
    fig.tight_layout()
    path = _CHART_DIR / "time_by_p.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _print_summary(df: pd.DataFrame) -> None:
    """Print formatted summary table."""
    print("=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(
        f"  {'Scenario':<28} {'Strat':<12} {'Family':<16} "
        f"{'n':>5} {'p':>3} {'B':>5} {'Time':>8} {'Throughput':>12}"
    )
    print(f"  {'-' * 88}")

    for _, row in df.iterrows():
        print(
            f"  {row['scenario']:<28} {row['strategy']:<12} "
            f"{row['family']:<16} {row['n']:>5} {row['p']:>3} "
            f"{row['B']:>5} {row['time_s']:>8.4f} {row['perms_per_sec']:>10.1f} p/s"
        )
    print("=" * 90)


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
        b_values = B_VALUES_QUICK
        repeats = REPEATS_QUICK
    else:
        scenarios = SCENARIOS
        b_values = B_VALUES_FULL
        repeats = REPEATS_FULL

    df = run_benchmark(scenarios, b_values, repeats=repeats)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "endtoend_profile.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    print("Generating charts...")
    _chart_method_comparison(df)
    _chart_family_comparison(df)
    _chart_strategy_overhead(df)
    _chart_time_by_p(df)

    _print_summary(df)


if __name__ == "__main__":
    main()

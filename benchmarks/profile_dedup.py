"""Profile permutation deduplication across (n, B) combinations.

Measures wall-clock time, peak memory, collision count, and dedup-needed
flag for ``generate_unique_permutations`` across a grid of sample sizes
and permutation counts.

Usage::

    conda activate randomization-tests
    python benchmarks/profile_dedup.py          # full grid (takes ~5 min)
    python benchmarks/profile_dedup.py --quick   # reduced grid for smoke test

Outputs:
    benchmarks/results/dedup_profile.csv
    docs/image/permutation-dedup-profile/heatmap.png
    docs/image/permutation-dedup-profile/time_vs_B.png
    docs/image/permutation-dedup-profile/collision_rate.png
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

# Ensure the package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from randomization_tests.permutations import generate_unique_permutations  # noqa: E402

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

N_VALUES_FULL = [8, 10, 12, 15, 20, 50, 100, 500, 1_000, 5_000, 10_000]
B_VALUES_FULL = [100, 1_000, 5_000, 10_000]

N_VALUES_QUICK = [8, 10, 12, 20, 100, 1_000]
B_VALUES_QUICK = [100, 1_000, 5_000]

REPEATS = 3
SEED_BASE = 42

RESULTS_DIR = Path(__file__).resolve().parent / "results"
IMAGE_DIR = (
    Path(__file__).resolve().parents[1] / "docs" / "image" / "permutation-dedup-profile"
)


# ------------------------------------------------------------------ #
# Benchmark helpers
# ------------------------------------------------------------------ #


def _collision_prob(n: int, B: int) -> float:
    """Birthday-paradox approximate collision probability.

    Computes in log space to avoid float underflow for large n.
    """
    log_numer = math.log(B) + math.log(max(B - 1, 1)) - math.log(2)
    log_denom = sum(math.log(i) for i in range(1, n + 1))
    log_prob = log_numer - log_denom
    if log_prob < -700:
        return 0.0
    return math.exp(log_prob)


def _is_feasible(n: int, B: int) -> bool:
    """Check whether B unique permutations can be drawn from n!"""
    n_fact = math.factorial(n)
    available = n_fact - 1  # exclude identity
    return available >= B


def _benchmark_one(n: int, B: int, seed: int) -> dict:
    """Run a single (n, B) benchmark and return metrics."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = generate_unique_permutations(
        n_samples=n,
        n_permutations=B,
        random_state=seed,
        exclude_identity=True,
    )
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Count collisions: for large n where dedup is skipped, collisions=0
    # For Lehmer-code path, collisions=0 by construction (without-replacement)
    # For dedup path: collisions = (initial batch size - unique kept before fill)
    # We can infer this from the result shape vs the batch size.
    # Since we can't directly observe the internal dedup, we proxy via
    # the birthday-paradox collision probability.
    coll_prob = _collision_prob(n, B)
    need_dedup = coll_prob >= 1e-9
    strategy = "lehmer" if n <= 10 else ("dedup" if need_dedup else "vectorised")

    return {
        "n": n,
        "B": B,
        "time_s": elapsed,
        "peak_memory_bytes": peak_bytes,
        "peak_memory_MB": peak_bytes / (1024 * 1024),
        "collision_prob": coll_prob,
        "need_dedup": need_dedup,
        "strategy": strategy,
        "result_shape": result.shape,
    }


def run_grid(
    n_values: list[int], b_values: list[int], repeats: int = REPEATS
) -> pd.DataFrame:
    """Run the full benchmark grid and return a DataFrame of results."""
    rows: list[dict] = []
    total = sum(1 for n in n_values for B in b_values if _is_feasible(n, B))
    done = 0

    for n in n_values:
        for B in b_values:
            if not _is_feasible(n, B):
                continue

            times: list[float] = []
            peak_mems: list[float] = []
            last_result: dict = {}

            for r in range(repeats):
                result = _benchmark_one(n, B, seed=SEED_BASE + r)
                times.append(result["time_s"])
                peak_mems.append(result["peak_memory_bytes"])
                last_result = result

            done += 1
            median_time = float(np.median(times))
            median_mem = float(np.median(peak_mems))

            # For large n, n! overflows string conversion limits.
            # Store log10(n!) instead; compute B/n! in log space.
            log10_nfact = sum(math.log10(i) for i in range(1, n + 1))
            log10_B_over_nfact = (
                math.log10(B) - log10_nfact if log10_nfact < 300 else float("-inf")
            )

            row = {
                "n": n,
                "B": B,
                "log10_n_factorial": log10_nfact,
                "log10_B_over_nfact": log10_B_over_nfact,
                "median_time_s": median_time,
                "median_peak_memory_bytes": median_mem,
                "median_peak_memory_MB": median_mem / (1024 * 1024),
                "collision_prob": last_result["collision_prob"],
                "need_dedup": last_result["need_dedup"],
                "strategy": last_result["strategy"],
            }
            rows.append(row)
            print(
                f"  [{done:3d}/{total}] n={n:4d}, B={B:6,d}, "
                f"strategy={row['strategy']:10s}, "
                f"time={median_time:.4f}s, "
                f"mem={row['median_peak_memory_MB']:.2f}MB"
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Chart generation
# ------------------------------------------------------------------ #


def _make_heatmap(df: pd.DataFrame, image_dir: Path) -> None:
    """Time heatmap as a function of (n, B) with log-scale colour."""
    pivot = df.pivot(index="n", columns="B", values="median_time_s")

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="YlOrRd",
        norm=LogNorm(
            vmin=max(pivot.values[pivot.values > 0].min(), 1e-6),
            vmax=pivot.values.max(),
        ),
        origin="lower",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{b:,}" for b in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("B (n_permutations)")
    ax.set_ylabel("n (n_samples)")
    ax.set_title("Permutation Generation Time (seconds, log scale)")

    # Annotate cells with time values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isnan(val):
                text = "—"
            elif val < 0.01:
                text = f"{val:.4f}"
            elif val < 1:
                text = f"{val:.3f}"
            else:
                text = f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(im, ax=ax, label="Time (s)")
    fig.tight_layout()
    fig.savefig(image_dir / "heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {image_dir / 'heatmap.png'}")


def _make_time_vs_B(df: pd.DataFrame, image_dir: Path) -> None:
    """Line plot: time vs. B for selected n values."""
    selected_n = [n for n in [8, 10, 12, 15, 20, 50, 100, 500] if n in df["n"].values]

    fig, ax = plt.subplots(figsize=(10, 6))
    for n in selected_n:
        subset = df[df["n"] == n].sort_values("B")
        marker = "o" if n <= 10 else ("s" if n <= 20 else "^")
        ax.plot(subset["B"], subset["median_time_s"], marker=marker, label=f"n={n}")

    ax.set_xlabel("B (n_permutations)")
    ax.set_ylabel("Median time (s)")
    ax.set_title("Time vs. B by Sample Size")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(title="n_samples")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(image_dir / "time_vs_B.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {image_dir / 'time_vs_B.png'}")


def _make_collision_rate(df: pd.DataFrame, image_dir: Path) -> None:
    """Collision probability vs. log10(B/n!) with dedup threshold."""
    # Only include rows where collision_prob > 0 and strategy != lehmer
    mask = (df["collision_prob"] > 0) & (df["strategy"] != "lehmer")
    subset = df[mask].copy()

    if subset.empty:
        print("  Skipping collision_rate plot — no dedup-eligible rows")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter actual collision probabilities coloured by n, with the
    # 1e-9 dedup threshold overlaid.
    for n in sorted(subset["n"].unique()):
        n_df = subset[subset["n"] == n]
        ax.scatter(
            n_df["log10_B_over_nfact"],
            n_df["collision_prob"],
            label=f"n={n}",
            s=60,
            zorder=5,
        )

    # Draw 1e-9 threshold line
    ax.axhline(
        y=1e-9, color="red", linestyle="--", alpha=0.7, label="Dedup threshold (1e-9)"
    )

    ax.set_xlabel("log₁₀(B / n!)")
    ax.set_ylabel("Collision probability")
    ax.set_yscale("log")
    ax.set_title("Birthday-Paradox Collision Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(image_dir / "collision_rate.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {image_dir / 'collision_rate.png'}")


def _make_strategy_breakdown(df: pd.DataFrame, image_dir: Path) -> None:
    """Stacked bar chart showing which strategy is used per (n, B)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    strategies = ["lehmer", "dedup", "vectorised"]
    colors = {"lehmer": "#4C72B0", "dedup": "#DD8452", "vectorised": "#55A868"}

    n_vals = sorted(df["n"].unique())
    x = np.arange(len(n_vals))
    width = 0.8 / len(df["B"].unique())

    for i, B in enumerate(sorted(df["B"].unique())):
        b_df = df[df["B"] == B]
        for n_idx, n in enumerate(n_vals):
            row = b_df[b_df["n"] == n]
            if row.empty:
                continue
            strategy = row.iloc[0]["strategy"]
            ax.bar(
                x[n_idx] + i * width,
                row.iloc[0]["median_time_s"],
                width,
                color=colors.get(strategy, "gray"),
                edgecolor="white",
                linewidth=0.5,
            )

    ax.set_xticks(x + width * len(df["B"].unique()) / 2)
    ax.set_xticklabels(n_vals)
    ax.set_xlabel("n (n_samples)")
    ax.set_ylabel("Time (s)")
    ax.set_yscale("log")
    ax.set_title("Strategy & Time by Sample Size")

    # Custom legend
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=colors[s], label=s) for s in strategies]
    ax.legend(handles=legend_elements, title="Strategy")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(image_dir / "strategy_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {image_dir / 'strategy_breakdown.png'}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile permutation deduplication")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced grid for quick smoke testing",
    )
    args = parser.parse_args()

    n_values = N_VALUES_QUICK if args.quick else N_VALUES_FULL
    b_values = B_VALUES_QUICK if args.quick else B_VALUES_FULL

    # Print environment info
    print("=" * 60)
    print("Permutation Deduplication Profile")
    print("=" * 60)
    print(f"  Platform:    {platform.platform()}")
    print(f"  Python:      {platform.python_version()}")
    print(f"  NumPy:       {np.__version__}")
    print(f"  CPU:         {platform.processor()}")
    print(f"  n values:    {n_values}")
    print(f"  B values:    {b_values}")
    print(f"  Repeats:     {REPEATS}")
    print()

    # Run benchmarks
    print("Running benchmarks...")
    df = run_grid(n_values, b_values, repeats=REPEATS)

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "dedup_profile.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Generate charts
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating charts...")
    _make_heatmap(df, IMAGE_DIR)
    _make_time_vs_B(df, IMAGE_DIR)
    _make_collision_rate(df, IMAGE_DIR)
    _make_strategy_breakdown(df, IMAGE_DIR)

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        df[
            [
                "n",
                "B",
                "strategy",
                "median_time_s",
                "median_peak_memory_MB",
                "collision_prob",
            ]
        ].to_string(index=False)
    )

    # Key findings
    print("\nKey findings:")
    lehmer = df[df["strategy"] == "lehmer"]
    dedup = df[df["strategy"] == "dedup"]
    vectorised = df[df["strategy"] == "vectorised"]

    if not lehmer.empty:
        print(
            f"  Lehmer:      {len(lehmer)} combos, "
            f"max time={lehmer['median_time_s'].max():.4f}s"
        )
    if not dedup.empty:
        print(
            f"  Dedup:       {len(dedup)} combos, "
            f"max time={dedup['median_time_s'].max():.4f}s"
        )
    if not vectorised.empty:
        print(
            f"  Vectorised:  {len(vectorised)} combos, "
            f"max time={vectorised['median_time_s'].max():.4f}s"
        )


if __name__ == "__main__":
    main()

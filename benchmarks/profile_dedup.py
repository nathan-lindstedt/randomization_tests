"""Profile permutation generation across all four generators.

Measures wall-clock time, peak memory, collision probability, and
strategy selection for the global, within-cell, between-cell, and
two-stage permutation generators across comprehensive grids.

Usage::

    conda activate randomization-tests
    python benchmarks/profile_dedup.py          # full grid (takes ~8 min)
    python benchmarks/profile_dedup.py --quick   # reduced grid for smoke test

Outputs:
    benchmarks/results/dedup_profile.csv
    benchmarks/image/permutation-dedup-profile/heatmap.png
    benchmarks/image/permutation-dedup-profile/time_vs_B.png
    benchmarks/image/permutation-dedup-profile/collision_rate.png
    benchmarks/image/permutation-dedup-profile/strategy_breakdown.png
    benchmarks/image/permutation-dedup-profile/within_cell_heatmap.png
    benchmarks/image/permutation-dedup-profile/cell_generator_comparison.png
    benchmarks/image/permutation-dedup-profile/lehmer_crossover.png
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
import tracemalloc
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

# Ensure the package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from randomization_tests.permutations import (  # noqa: E402
    generate_between_cell_permutations,
    generate_two_stage_permutations,
    generate_unique_permutations,
    generate_within_cell_permutations,
)

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

# Global generator grid
N_VALUES_FULL = [8, 10, 12, 15, 20, 50, 100, 500, 1_000, 5_000, 10_000]
B_VALUES_FULL = [100, 1_000, 5_000, 10_000]
N_VALUES_QUICK = [8, 10, 12, 20, 100, 1_000]
B_VALUES_QUICK = [100, 1_000, 5_000]

# Within-cell grid: (n_groups, cell_size)
WITHIN_GRID_FULL: list[tuple[int, int]] = [
    # Small cells — Lehmer (prod n_c! <= 50_000)
    (5, 3),  # prod = 6^5 = 7_776,       n = 15
    (3, 4),  # prod = 24^3 = 13_824,     n = 12
    (6, 3),  # prod = 6^6 = 46_656,      n = 18
    # Transition zone — just above Lehmer threshold
    (4, 4),  # prod = 24^4 = 331_776,    n = 16   > 50K → vectorised-batch
    (7, 3),  # prod = 6^7 = 279_936,     n = 21   > 50K → vectorised-batch
    # Medium cells — vectorised-batch + hash dedup
    (5, 5),  # n = 25
    (4, 8),  # n = 32
    (3, 10),  # n = 30
    (5, 10),  # n = 50
    # Large cells — collisions negligible
    (5, 20),  # n = 100
    (10, 10),  # n = 100
    (5, 100),  # n = 500
    (10, 50),  # n = 500
    (10, 100),  # n = 1_000
    (20, 50),  # n = 1_000
    # Very large — matching global generator scale
    (50, 100),  # n = 5_000
    (10, 500),  # n = 5_000
]
WITHIN_GRID_QUICK: list[tuple[int, int]] = [
    (5, 3),
    (3, 4),
    (5, 5),
    (3, 10),
    (5, 20),
    (10, 10),
    (10, 100),
    (50, 100),
]
WITHIN_B_FULL = [100, 1_000, 5_000, 10_000]
WITHIN_B_QUICK = [100, 1_000, 5_000]

# Between-cell grid: (G, cell_size)
BETWEEN_GRID_FULL: list[tuple[int, int]] = [
    # Lehmer (G ≤ 10)
    (8, 10),  # n = 80,    G! = 40_320
    (10, 10),  # n = 100,   G! = 3_628_800
    (8, 50),  # n = 400
    (10, 50),  # n = 500
    (10, 100),  # n = 1_000
    (10, 500),  # n = 5_000
    # Hash-dedup (G > 10)
    (15, 10),  # n = 150
    (20, 10),  # n = 200
    (20, 50),  # n = 1_000
    (50, 100),  # n = 5_000
]
BETWEEN_GRID_QUICK: list[tuple[int, int]] = [
    (8, 10),
    (10, 10),
    (10, 100),
    (15, 10),
    (50, 100),
]
BETWEEN_B_FULL = [100, 1_000, 5_000, 10_000]
BETWEEN_B_QUICK = [100, 1_000, 5_000]

# Two-stage grid: (G, cell_size) — balanced designs
TWO_STAGE_GRID_FULL: list[tuple[int, int]] = [
    # Lehmer candidates: G!*prod(n_c!) <= 50_000
    (3, 3),  # 6 * 6^3 = 1_296,       n = 9    → Lehmer
    (4, 3),  # 24 * 6^4 = 31_104,     n = 12   → Lehmer
    # Just above Lehmer threshold
    (3, 4),  # 6 * 24^3 = 82_944,     n = 12   > 50K → hash-dedup
    (5, 3),  # 120 * 6^5 = 933_120,   n = 15   > 50K → hash-dedup
    # Clearly hash-dedup — moderate
    (3, 5),  # n = 15
    (5, 5),  # n = 25
    (5, 10),  # n = 50
    (10, 10),  # n = 100
    # Hash-dedup — large, matching global scale
    (5, 100),  # n = 500
    (10, 50),  # n = 500
    (10, 100),  # n = 1_000
]
TWO_STAGE_GRID_QUICK: list[tuple[int, int]] = [
    (3, 3),
    (4, 3),
    (3, 4),
    (5, 5),
    (10, 10),
    (10, 100),
]
TWO_STAGE_B_FULL = [100, 1_000, 5_000, 10_000]
TWO_STAGE_B_QUICK = [100, 1_000, 5_000]

REPEATS = 3
SEED_BASE = 42

RESULTS_DIR = Path(__file__).resolve().parent / "results"
IMAGE_DIR = Path(__file__).resolve().parent / "image" / "permutation-dedup-profile"


# ------------------------------------------------------------------ #
# Collision probability helpers
# ------------------------------------------------------------------ #


def _collision_prob_global(n: int, B: int) -> float:
    """Birthday-paradox collision probability for the global generator.

    P(>=1 dup) ~ B(B-1) / (2 * n!).  Computed in log space.
    """
    log_numer = math.log(B) + math.log(max(B - 1, 1)) - math.log(2)
    log_denom = sum(math.log(i) for i in range(1, n + 1))
    log_prob = log_numer - log_denom
    return 0.0 if log_prob < -700 else math.exp(log_prob)


def _collision_prob_within(cell_sizes: list[int], B: int) -> float:
    """Birthday-paradox collision probability for within-cell generator.

    Reference set = prod(n_c!).
    """
    log_numer = math.log(B) + math.log(max(B - 1, 1)) - math.log(2)
    log_denom = sum(sum(math.log(j) for j in range(1, s + 1)) for s in cell_sizes)
    if log_denom == 0:
        return 1.0
    log_prob = log_numer - log_denom
    return 0.0 if log_prob < -700 else math.exp(log_prob)


def _collision_prob_between(G: int, B: int) -> float:
    """Birthday-paradox collision probability for between-cell generator.

    Reference set = G!.
    """
    log_numer = math.log(B) + math.log(max(B - 1, 1)) - math.log(2)
    log_denom = sum(math.log(i) for i in range(1, G + 1))
    log_prob = log_numer - log_denom
    return 0.0 if log_prob < -700 else math.exp(log_prob)


def _collision_prob_two_stage(G: int, cell_sizes: list[int], B: int) -> float:
    """Birthday-paradox collision probability for two-stage generator.

    Reference set = G! * prod(n_c!).
    """
    log_numer = math.log(B) + math.log(max(B - 1, 1)) - math.log(2)
    log_denom_g = sum(math.log(i) for i in range(1, G + 1))
    log_denom_w = sum(sum(math.log(j) for j in range(1, s + 1)) for s in cell_sizes)
    log_prob = log_numer - (log_denom_g + log_denom_w)
    return 0.0 if log_prob < -700 else math.exp(log_prob)


# ------------------------------------------------------------------ #
# Feasibility checks
# ------------------------------------------------------------------ #


def _is_feasible_global(n: int, B: int) -> bool:
    return math.factorial(n) - 1 >= B


def _is_feasible_within(cell_sizes: list[int], B: int) -> bool:
    _CAP = B + 2
    total = reduce(
        lambda a, b: min(a * b, _CAP),
        [math.factorial(s) for s in cell_sizes],
        1,
    )
    return total - 1 >= B


def _is_feasible_between(G: int, B: int) -> bool:
    return math.factorial(G) - 1 >= B


def _is_feasible_two_stage(G: int, cell_sizes: list[int], B: int) -> bool:
    _CAP = B + 2
    wp = reduce(
        lambda a, b: min(a * b, _CAP),
        [math.factorial(s) for s in cell_sizes],
        1,
    )
    return min(math.factorial(G) * wp, _CAP) - 1 >= B


# ------------------------------------------------------------------ #
# Strategy classification (mirrors permutations.py branching)
# ------------------------------------------------------------------ #


def _classify_global(n: int, B: int) -> str:
    if n <= 10:
        return "lehmer"
    return "dedup" if _collision_prob_global(n, B) >= 1e-9 else "vectorised"


def _classify_within(cell_sizes: list[int]) -> str:
    _LEHMER_THRESHOLD = 50_000
    total_exact = 1
    for s in cell_sizes:
        total_exact *= math.factorial(s)
        if total_exact > _LEHMER_THRESHOLD:
            return "vectorised-batch"
    if all(s <= 10 for s in cell_sizes):
        return "lehmer"
    return "vectorised-batch"


def _classify_between(G: int) -> str:
    return "lehmer" if G <= 10 else "hash-dedup"


def _classify_two_stage(G: int, cell_sizes: list[int]) -> str:
    _LEHMER_THRESHOLD = 50_000
    if len(set(cell_sizes)) != 1 or G > 10:
        return "hash-dedup"
    within_exact = 1
    for s in cell_sizes:
        within_exact *= math.factorial(s)
        if within_exact > _LEHMER_THRESHOLD:
            return "hash-dedup"
    if math.factorial(G) * within_exact <= _LEHMER_THRESHOLD:
        return "lehmer"
    return "hash-dedup"


# ------------------------------------------------------------------ #
# Log-space reference-set size
# ------------------------------------------------------------------ #


def _log10_ref_global(n: int) -> float:
    return sum(math.log10(i) for i in range(1, n + 1))


def _log10_ref_within(cell_sizes: list[int]) -> float:
    return sum(sum(math.log10(j) for j in range(1, s + 1)) for s in cell_sizes)


def _log10_ref_between(G: int) -> float:
    return sum(math.log10(i) for i in range(1, G + 1))


def _log10_ref_two_stage(G: int, cell_sizes: list[int]) -> float:
    return _log10_ref_between(G) + _log10_ref_within(cell_sizes)


# ------------------------------------------------------------------ #
# Individual benchmark runners
# ------------------------------------------------------------------ #


def _bench_global(n: int, B: int, seed: int) -> tuple[float, int]:
    tracemalloc.start()
    t0 = time.perf_counter()
    generate_unique_permutations(
        n_samples=n,
        n_permutations=B,
        random_state=seed,
        exclude_identity=True,
    )
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak


def _bench_within(
    n_groups: int,
    cell_size: int,
    B: int,
    seed: int,
) -> tuple[float, int]:
    cells = np.repeat(np.arange(n_groups), cell_size)
    tracemalloc.start()
    t0 = time.perf_counter()
    generate_within_cell_permutations(
        n_samples=len(cells),
        n_permutations=B,
        cells=cells,
        random_state=seed,
        exclude_identity=True,
    )
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak


def _bench_between(
    n_groups: int,
    cell_size: int,
    B: int,
    seed: int,
) -> tuple[float, int]:
    cells = np.repeat(np.arange(n_groups), cell_size)
    tracemalloc.start()
    t0 = time.perf_counter()
    generate_between_cell_permutations(
        n_samples=len(cells),
        n_permutations=B,
        cells=cells,
        random_state=seed,
        exclude_identity=True,
    )
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak


def _bench_two_stage(
    n_groups: int,
    cell_size: int,
    B: int,
    seed: int,
) -> tuple[float, int]:
    cells = np.repeat(np.arange(n_groups), cell_size)
    tracemalloc.start()
    t0 = time.perf_counter()
    generate_two_stage_permutations(
        n_samples=len(cells),
        n_permutations=B,
        cells=cells,
        random_state=seed,
        exclude_identity=True,
    )
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak


# ------------------------------------------------------------------ #
# Grid runners
# ------------------------------------------------------------------ #


def _run_repeats(
    bench_fn,
    repeats: int,
    **kwargs: int,
) -> tuple[float, float, float, float]:
    """Run *bench_fn* multiple times, return (median_t, min_t, max_t, median_mem)."""
    times: list[float] = []
    mems: list[float] = []
    for r in range(repeats):
        t, m = bench_fn(seed=SEED_BASE + r, **kwargs)
        times.append(t)
        mems.append(m)
    return float(np.median(times)), min(times), max(times), float(np.median(mems))


def run_global_grid(
    n_values: list[int],
    b_values: list[int],
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    combos = [(n, B) for n in n_values for B in b_values if _is_feasible_global(n, B)]
    for done, (n, B) in enumerate(combos, 1):
        med_t, min_t, max_t, med_m = _run_repeats(
            _bench_global,
            repeats,
            n=n,
            B=B,
        )
        strat = _classify_global(n, B)
        coll = _collision_prob_global(n, B)
        row = dict(
            generator="global",
            label=f"global_n{n}_B{B}",
            n=n,
            B=B,
            n_groups=0,
            cell_size=n,
            log10_ref_set=_log10_ref_global(n),
            median_time_s=med_t,
            min_time_s=min_t,
            max_time_s=max_t,
            median_peak_memory_MB=med_m / (1024 * 1024),
            collision_prob=coll,
            strategy=strat,
        )
        rows.append(row)
        print(
            f"  [{done:3d}/{len(combos)}] global  "
            f"n={n:5d}  B={B:5d}  "
            f"strategy={strat:16s}  time={med_t:.4f}s"
        )
    return pd.DataFrame(rows)


def run_within_grid(
    grid: list[tuple[int, int]],
    b_values: list[int],
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    combos = [
        (g, cs, B)
        for g, cs in grid
        for B in b_values
        if _is_feasible_within([cs] * g, B)
    ]
    for done, (ng, cs, B) in enumerate(combos, 1):
        n = ng * cs
        sizes = [cs] * ng
        med_t, min_t, max_t, med_m = _run_repeats(
            _bench_within,
            repeats,
            n_groups=ng,
            cell_size=cs,
            B=B,
        )
        strat = _classify_within(sizes)
        coll = _collision_prob_within(sizes, B)
        row = dict(
            generator="within",
            label=f"within_{ng}x{cs}_B{B}",
            n=n,
            B=B,
            n_groups=ng,
            cell_size=cs,
            log10_ref_set=_log10_ref_within(sizes),
            median_time_s=med_t,
            min_time_s=min_t,
            max_time_s=max_t,
            median_peak_memory_MB=med_m / (1024 * 1024),
            collision_prob=coll,
            strategy=strat,
        )
        rows.append(row)
        print(
            f"  [{done:3d}/{len(combos)}] within  "
            f"{ng:2d}x{cs:<3d}  n={n:5d}  B={B:5d}  "
            f"strategy={strat:16s}  time={med_t:.4f}s"
        )
    return pd.DataFrame(rows)


def run_between_grid(
    grid: list[tuple[int, int]],
    b_values: list[int],
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    combos = [
        (g, cs, B) for g, cs in grid for B in b_values if _is_feasible_between(g, B)
    ]
    for done, (ng, cs, B) in enumerate(combos, 1):
        n = ng * cs
        med_t, min_t, max_t, med_m = _run_repeats(
            _bench_between,
            repeats,
            n_groups=ng,
            cell_size=cs,
            B=B,
        )
        strat = _classify_between(ng)
        coll = _collision_prob_between(ng, B)
        row = dict(
            generator="between",
            label=f"between_{ng}x{cs}_B{B}",
            n=n,
            B=B,
            n_groups=ng,
            cell_size=cs,
            log10_ref_set=_log10_ref_between(ng),
            median_time_s=med_t,
            min_time_s=min_t,
            max_time_s=max_t,
            median_peak_memory_MB=med_m / (1024 * 1024),
            collision_prob=coll,
            strategy=strat,
        )
        rows.append(row)
        print(
            f"  [{done:3d}/{len(combos)}] between "
            f"G={ng:2d}  cs={cs:<3d}  n={n:5d}  B={B:5d}  "
            f"strategy={strat:16s}  time={med_t:.4f}s"
        )
    return pd.DataFrame(rows)


def run_two_stage_grid(
    grid: list[tuple[int, int]],
    b_values: list[int],
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    combos = [
        (g, cs, B)
        for g, cs in grid
        for B in b_values
        if _is_feasible_two_stage(g, [cs] * g, B)
    ]
    for done, (ng, cs, B) in enumerate(combos, 1):
        n = ng * cs
        sizes = [cs] * ng
        med_t, min_t, max_t, med_m = _run_repeats(
            _bench_two_stage,
            repeats,
            n_groups=ng,
            cell_size=cs,
            B=B,
        )
        strat = _classify_two_stage(ng, sizes)
        coll = _collision_prob_two_stage(ng, sizes, B)
        row = dict(
            generator="two-stage",
            label=f"twostage_{ng}x{cs}_B{B}",
            n=n,
            B=B,
            n_groups=ng,
            cell_size=cs,
            log10_ref_set=_log10_ref_two_stage(ng, sizes),
            median_time_s=med_t,
            min_time_s=min_t,
            max_time_s=max_t,
            median_peak_memory_MB=med_m / (1024 * 1024),
            collision_prob=coll,
            strategy=strat,
        )
        rows.append(row)
        print(
            f"  [{done:3d}/{len(combos)}] 2-stage "
            f"G={ng:2d}  cs={cs:<3d}  n={n:5d}  B={B:5d}  "
            f"strategy={strat:16s}  time={med_t:.4f}s"
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Chart generation
# ------------------------------------------------------------------ #


def _make_heatmap(df: pd.DataFrame, image_dir: Path) -> None:
    """Time heatmap as a function of (n, B) with log-scale colour (global only)."""
    gdf = df[df["generator"] == "global"]
    if gdf.empty:
        print("  Skipping heatmap — no global data")
        return
    pivot = gdf.pivot(index="n", columns="B", values="median_time_s")

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
    ax.set_title("Global Generator — Time (seconds, log scale)")

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
    """Line plot: time vs. B for selected n values (global only)."""
    gdf = df[df["generator"] == "global"]
    if gdf.empty:
        print("  Skipping time_vs_B — no global data")
        return
    selected_n = [n for n in [8, 10, 12, 15, 20, 50, 100, 500] if n in gdf["n"].values]

    fig, ax = plt.subplots(figsize=(10, 6))
    for n in selected_n:
        subset = gdf[gdf["n"] == n].sort_values("B")
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
    """Collision probability vs. log10(reference set) for all generators."""
    mask = (df["collision_prob"] > 0) & (~df["strategy"].str.contains("lehmer"))
    subset = df[mask].copy()

    if subset.empty:
        print("  Skipping collision_rate plot — no dedup-eligible rows")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    gen_markers = {"global": "o", "within": "s", "between": "^", "two-stage": "D"}
    gen_colors = {
        "global": "#4C72B0",
        "within": "#55A868",
        "between": "#DD8452",
        "two-stage": "#C44E52",
    }

    for gen in ["global", "within", "between", "two-stage"]:
        gen_df = subset[subset["generator"] == gen]
        if gen_df.empty:
            continue
        ax.scatter(
            gen_df["log10_ref_set"],
            gen_df["collision_prob"],
            label=gen,
            s=60,
            zorder=5,
            marker=gen_markers[gen],
            color=gen_colors[gen],
            alpha=0.7,
        )

    ax.axhline(
        y=1e-9,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Dedup threshold (1e-9)",
    )

    ax.set_xlabel("log₁₀(reference set size)")
    ax.set_ylabel("Collision probability")
    ax.set_yscale("log")
    ax.set_title("Birthday-Paradox Collision Probability — All Generators")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(image_dir / "collision_rate.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {image_dir / 'collision_rate.png'}")


def _make_strategy_breakdown(df: pd.DataFrame, image_dir: Path) -> None:
    """2×2 panel: strategy breakdown by generator."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Strategy Selection by Generator", fontsize=14)

    strategy_colors = {
        "lehmer": "#4C72B0",
        "dedup": "#DD8452",
        "vectorised": "#55A868",
        "vectorised-batch": "#8172B3",
        "hash-dedup": "#C44E52",
    }
    generators = ["global", "within", "between", "two-stage"]

    for ax, gen in zip(axes.flat, generators, strict=True):
        gen_df = df[df["generator"] == gen].copy()
        if gen_df.empty:
            ax.set_title(f"{gen} (no data)")
            continue

        if gen == "global":
            gen_df["config"] = gen_df["n"].astype(str)
            x_key = "config"
            ax.set_xlabel("n (n_samples)")
        elif gen in ("within", "two-stage"):
            gen_df["config"] = (
                gen_df["n_groups"].astype(str) + "×" + gen_df["cell_size"].astype(str)
            )
            x_key = "config"
            ax.set_xlabel("groups × cell_size")
        else:  # between
            gen_df["config"] = gen_df["n_groups"].astype(str)
            x_key = "config"
            ax.set_xlabel("G (groups)")

        # Aggregate: median time per config per strategy
        agg = (
            gen_df.groupby([x_key, "strategy"], sort=False)
            .agg(
                time=("median_time_s", "median"),
            )
            .reset_index()
        )

        configs = list(dict.fromkeys(gen_df[x_key]))  # preserve order
        x = np.arange(len(configs))

        for strat in sorted(agg["strategy"].unique()):
            heights = []
            for cfg in configs:
                row = agg[(agg[x_key] == cfg) & (agg["strategy"] == strat)]
                heights.append(float(row["time"].iloc[0]) if not row.empty else 0)
            ax.bar(
                x,
                heights,
                label=strat,
                color=strategy_colors.get(strat, "gray"),
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Median time (s)")
        ax.set_yscale("log")
        ax.set_title(gen)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(image_dir / "strategy_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {image_dir / 'strategy_breakdown.png'}")


def _make_within_cell_heatmap(df: pd.DataFrame, image_dir: Path) -> None:
    """Heatmap of within-cell generator time by (n_groups, cell_size)."""
    wdf = df[df["generator"] == "within"].copy()
    if wdf.empty:
        print("  Skipping within_cell_heatmap — no data")
        return

    agg = (
        wdf.groupby(["cell_size", "n_groups"])
        .agg(
            median_time_s=("median_time_s", "median"),
            strategy=("strategy", "first"),
        )
        .reset_index()
    )

    pivot = agg.pivot(index="cell_size", columns="n_groups", values="median_time_s")
    strat_pivot = agg.pivot(index="cell_size", columns="n_groups", values="strategy")

    fig, ax = plt.subplots(figsize=(10, 7))
    valid = pivot.values[~np.isnan(pivot.values)]
    if len(valid) == 0:
        plt.close(fig)
        return

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="YlOrRd",
        norm=LogNorm(
            vmin=max(valid[valid > 0].min(), 1e-6),
            vmax=valid.max(),
        ),
        origin="lower",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Number of groups")
    ax.set_ylabel("Cell size")
    ax.set_title("Within-Cell Generator — Time by (groups, cell_size)")

    abbrev = {"lehmer": "L", "vectorised-batch": "VB", "hash-dedup": "HD"}
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            strat = strat_pivot.values[i, j]
            if np.isnan(val):
                text = "—"
            elif val < 0.01:
                text = f"{val:.4f}"
            elif val < 1:
                text = f"{val:.3f}"
            else:
                text = f"{val:.2f}"
            sa = abbrev.get(str(strat), "") if not pd.isna(strat) else ""
            ax.text(
                j,
                i,
                f"{text}\n({sa})",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    fig.colorbar(im, ax=ax, label="Time (s)")
    fig.tight_layout()
    fig.savefig(image_dir / "within_cell_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {image_dir / 'within_cell_heatmap.png'}")


def _make_cell_generator_comparison(df: pd.DataFrame, image_dir: Path) -> None:
    """Bar chart comparing all 4 generators at a common B value."""
    for B_target in [1_000, 500, 100]:
        subset = df[df["B"] == B_target]
        if not subset.empty:
            break
    else:
        print("  Skipping cell_generator_comparison — no common B")
        return

    gen_colors = {
        "global": "#4C72B0",
        "within": "#55A868",
        "between": "#DD8452",
        "two-stage": "#C44E52",
    }
    generators = [
        g
        for g in ["global", "within", "between", "two-stage"]
        if g in subset["generator"].values
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_groups: list[tuple[str, float, str, str]] = []
    for gen in generators:
        gen_df = subset[subset["generator"] == gen].sort_values("n")
        for _, row in gen_df.iterrows():
            bar_groups.append(
                (row["label"], row["median_time_s"], gen, row["strategy"])
            )

    x = np.arange(len(bar_groups))
    labels = [bg[0] for bg in bar_groups]
    heights = [bg[1] for bg in bar_groups]
    colors = [gen_colors[bg[2]] for bg in bar_groups]
    strats = [bg[3] for bg in bar_groups]

    bars = ax.bar(
        x, heights, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5
    )

    for bar, strat in zip(bars, strats, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            strat[:3].upper(),
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=90,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=6)
    ax.set_ylabel("Median time (s)")
    ax.set_yscale("log")
    ax.set_title(f"All Generators at B = {B_target:,}")

    from matplotlib.patches import Patch

    legend_el = [Patch(facecolor=gen_colors[g], label=g) for g in generators]
    ax.legend(handles=legend_el, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(image_dir / "cell_generator_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {image_dir / 'cell_generator_comparison.png'}")


def _make_lehmer_crossover(df: pd.DataFrame, image_dir: Path) -> None:
    """Scatter: Lehmer vs non-Lehmer timing across all generators."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gen_markers = {"global": "o", "within": "s", "between": "^", "two-stage": "D"}

    for gen in ["global", "within", "between", "two-stage"]:
        gen_df = df[df["generator"] == gen]
        if gen_df.empty:
            continue
        is_lehmer = gen_df["strategy"].str.contains("lehmer")

        for is_l, colour, suffix in [
            (True, "#4C72B0", "Lehmer"),
            (False, "#DD8452", "other"),
        ]:
            sub = gen_df[is_lehmer] if is_l else gen_df[~is_lehmer]
            if sub.empty:
                continue
            kwargs: dict = dict(
                marker=gen_markers[gen],
                s=60,
                alpha=0.7,
                label=f"{gen} ({suffix})",
            )
            if is_l:
                kwargs["color"] = colour
            else:
                kwargs["facecolors"] = "none"
                kwargs["edgecolors"] = colour
            ax.scatter(sub["log10_ref_set"], sub["median_time_s"], **kwargs)

    ax.axvline(
        x=math.log10(50_000),
        color="red",
        linestyle="--",
        alpha=0.6,
        label="Lehmer threshold (50 K)",
    )
    ax.set_xlabel("log₁₀(reference set size)")
    ax.set_ylabel("Median time (s)")
    ax.set_yscale("log")
    ax.set_title("Lehmer vs. Other Strategies — Crossover Analysis")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(image_dir / "lehmer_crossover.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {image_dir / 'lehmer_crossover.png'}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile permutation generation across all generators",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced grid for quick smoke testing",
    )
    args = parser.parse_args()
    quick = args.quick

    print("=" * 65)
    print("Permutation Generation Profile — All Generators")
    print("=" * 65)
    print(f"  Platform:  {platform.platform()}")
    print(f"  Python:    {platform.python_version()}")
    print(f"  NumPy:     {np.__version__}")
    print(f"  CPU:       {platform.processor()}")
    print(f"  Mode:      {'quick' if quick else 'full'}")
    print(f"  Repeats:   {REPEATS}")
    print()

    # ---- Global ----
    print("--- Global generator ---")
    df_global = run_global_grid(
        N_VALUES_QUICK if quick else N_VALUES_FULL,
        B_VALUES_QUICK if quick else B_VALUES_FULL,
        REPEATS,
    )
    print()

    # ---- Within-cell ----
    print("--- Within-cell generator ---")
    df_within = run_within_grid(
        WITHIN_GRID_QUICK if quick else WITHIN_GRID_FULL,
        WITHIN_B_QUICK if quick else WITHIN_B_FULL,
        REPEATS,
    )
    print()

    # ---- Between-cell ----
    print("--- Between-cell generator ---")
    df_between = run_between_grid(
        BETWEEN_GRID_QUICK if quick else BETWEEN_GRID_FULL,
        BETWEEN_B_QUICK if quick else BETWEEN_B_FULL,
        REPEATS,
    )
    print()

    # ---- Two-stage ----
    print("--- Two-stage generator ---")
    df_two_stage = run_two_stage_grid(
        TWO_STAGE_GRID_QUICK if quick else TWO_STAGE_GRID_FULL,
        TWO_STAGE_B_QUICK if quick else TWO_STAGE_B_FULL,
        REPEATS,
    )
    print()

    # ---- Combine and save ----
    df = pd.concat(
        [df_global, df_within, df_between, df_two_stage],
        ignore_index=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "dedup_profile.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # ---- Charts ----
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating charts...")
    _make_heatmap(df, IMAGE_DIR)
    _make_time_vs_B(df, IMAGE_DIR)
    _make_collision_rate(df, IMAGE_DIR)
    _make_strategy_breakdown(df, IMAGE_DIR)
    _make_within_cell_heatmap(df, IMAGE_DIR)
    _make_cell_generator_comparison(df, IMAGE_DIR)
    _make_lehmer_crossover(df, IMAGE_DIR)

    # ---- Summary by generator ----
    print("\n" + "=" * 65)
    print("Summary by Generator")
    print("=" * 65)

    for gen in ["global", "within", "between", "two-stage"]:
        gen_df = df[df["generator"] == gen]
        if gen_df.empty:
            continue
        print(f"\n  {gen.upper()} ({len(gen_df)} scenarios)")
        for strat in sorted(gen_df["strategy"].unique()):
            sdf = gen_df[gen_df["strategy"] == strat]
            print(
                f"    {strat:16s}: {len(sdf):3d} combos, "
                f"time [{sdf['median_time_s'].min():.4f}s, "
                f"{sdf['median_time_s'].max():.4f}s]"
            )

    # ---- Threshold analysis ----
    print("\n" + "=" * 65)
    print("Threshold Analysis")
    print("=" * 65)

    gdf = df[df["generator"] == "global"]
    if not gdf.empty:
        leh_max = gdf.loc[gdf["strategy"] == "lehmer", "median_time_s"]
        ded_12 = gdf.loc[
            (gdf["strategy"] == "dedup") & (gdf["n"] == 12), "median_time_s"
        ]
        print(
            f"  Global Lehmer ↔ dedup: "
            f"max Lehmer={leh_max.max():.4f}s, "
            f"min dedup(n=12)={ded_12.min():.4f}s"
            if not ded_12.empty
            else f"  Global Lehmer max={leh_max.max():.4f}s"
        )

    wdf = df[df["generator"] == "within"]
    if not wdf.empty:
        for strat in ["lehmer", "vectorised-batch"]:
            sdf = wdf[wdf["strategy"] == strat]
            if not sdf.empty:
                print(
                    f"  Within {strat}: {len(sdf)} combos, "
                    f"max time={sdf['median_time_s'].max():.4f}s"
                )

    tdf = df[df["generator"] == "two-stage"]
    if not tdf.empty:
        for strat in ["lehmer", "hash-dedup"]:
            sdf = tdf[tdf["strategy"] == strat]
            if not sdf.empty:
                print(
                    f"  Two-stage {strat}: {len(sdf)} combos, "
                    f"max time={sdf['median_time_s'].max():.4f}s"
                )


if __name__ == "__main__":
    main()

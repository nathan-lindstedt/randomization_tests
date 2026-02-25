"""Real-world stress-test benchmarks for permutation generators.

Tests all four generators (global, within-cell, between-cell, two-stage)
against realistic experimental designs with unbalanced cell sizes,
varying cluster counts, and typical sample sizes seen in practice.

Every generated permutation matrix is validated for correctness:
  - Each row is a valid permutation of [0..n-1]
  - No duplicate rows (all unique)
  - Identity excluded

Usage::

    conda activate randomization-tests
    python benchmarks/profile_realworld.py          # full suite
    python benchmarks/profile_realworld.py --quick  # reduced B values

Outputs:
    benchmarks/results/realworld_profile.csv
    benchmarks/image/realworld-profile/time_by_scenario.png
    benchmarks/image/realworld-profile/generator_comparison.png
    benchmarks/image/realworld-profile/memory_by_scenario.png
    benchmarks/image/realworld-profile/validity_report.txt
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure the package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from randomization_tests.permutations import (  # noqa: E402
    generate_between_cell_permutations,
    generate_two_stage_permutations,
    generate_unique_permutations,
    generate_within_cell_permutations,
)

# ------------------------------------------------------------------ #
# Real-world scenario definitions
# ------------------------------------------------------------------ #
#
# Each scenario is a dict with:
#   name:        Human-readable label
#   description: What it models
#   cell_sizes:  List of per-cell observation counts (unbalanced OK)
#   generators:  Which generators to test


SCENARIOS: list[dict] = [
    # ---- Clinical trials -----------------------------------------
    {
        "name": "small_rct_3arm",
        "description": "3-arm RCT: placebo(22), low-dose(18), high-dose(20)",
        "cell_sizes": [22, 18, 20],
        "generators": ["global", "within", "between", "two-stage"],
    },
    {
        "name": "rct_5arm_unbalanced",
        "description": "5-arm dose-finding: 30, 25, 28, 15, 22",
        "cell_sizes": [30, 25, 28, 15, 22],
        "generators": ["global", "within", "between", "two-stage"],
    },
    # ---- Multi-site clinical -------------------------------------
    {
        "name": "multisite_8sites",
        "description": "8-site trial: sites of 20-40 patients",
        "cell_sizes": [35, 28, 40, 22, 33, 26, 38, 20],
        "generators": ["global", "within", "between", "two-stage"],
    },
    {
        "name": "multisite_balanced_8",
        "description": "8-site trial: balanced 30/site (reference)",
        "cell_sizes": [30] * 8,
        "generators": ["global", "within", "between", "two-stage"],
    },
    # ---- Education -----------------------------------------------
    {
        "name": "classroom_12",
        "description": "12 classrooms: 18-30 students each",
        "cell_sizes": [24, 28, 19, 30, 22, 26, 18, 25, 27, 21, 29, 23],
        "generators": ["global", "within", "between", "two-stage"],
    },
    {
        "name": "classroom_balanced_12",
        "description": "12 classrooms: balanced 25/class (reference)",
        "cell_sizes": [25] * 12,
        "generators": ["global", "within", "between", "two-stage"],
    },
    # ---- Large cluster RCT ---------------------------------------
    {
        "name": "cluster_rct_20clinics",
        "description": "20 clinics: 30-80 patients, high imbalance",
        "cell_sizes": [
            45,
            62,
            33,
            78,
            41,
            55,
            30,
            67,
            50,
            38,
            72,
            44,
            58,
            35,
            65,
            48,
            60,
            37,
            53,
            70,
        ],
        "generators": ["global", "within", "two-stage"],
    },
    {
        "name": "cluster_rct_balanced_20",
        "description": "20 clinics: balanced 50/clinic (reference)",
        "cell_sizes": [50] * 20,
        "generators": ["global", "within", "two-stage"],
    },
    # ---- Survey / stratified experiment --------------------------
    {
        "name": "survey_30strata",
        "description": "30 strata: 15-50 observations, heavy imbalance",
        "cell_sizes": [
            18,
            42,
            25,
            50,
            15,
            35,
            28,
            45,
            20,
            38,
            22,
            48,
            30,
            16,
            40,
            33,
            19,
            46,
            27,
            36,
            24,
            44,
            31,
            17,
            39,
            29,
            47,
            21,
            37,
            43,
        ],
        "generators": ["global", "within", "two-stage"],
    },
    # ---- Balanced multi-arm at scale -----------------------------
    {
        "name": "balanced_5x200",
        "description": "5 arms × 200 observations: large balanced",
        "cell_sizes": [200] * 5,
        "generators": ["global", "within", "between", "two-stage"],
    },
    {
        "name": "balanced_10x100",
        "description": "10 arms × 100 observations: moderate balanced",
        "cell_sizes": [100] * 10,
        "generators": ["global", "within", "between", "two-stage"],
    },
    # ---- Tiny cells, many groups ---------------------------------
    {
        "name": "paired_design_30",
        "description": "30 matched pairs (cell size 2): paired t-test",
        "cell_sizes": [2] * 30,
        "generators": ["within", "between", "two-stage"],
    },
    {
        "name": "triplet_design_20",
        "description": "20 matched triplets (cell size 3)",
        "cell_sizes": [3] * 20,
        "generators": ["within", "between", "two-stage"],
    },
    # ---- Mixed-size realistic ------------------------------------
    {
        "name": "mixed_hospital",
        "description": "6 hospitals: 2 large(150), 2 mid(80), 2 small(30)",
        "cell_sizes": [150, 150, 80, 80, 30, 30],
        "generators": ["global", "within", "between", "two-stage"],
    },
    {
        "name": "mixed_school_district",
        "description": "15 schools: 5 large(40), 5 mid(25), 5 small(12)",
        "cell_sizes": [40] * 5 + [25] * 5 + [12] * 5,
        "generators": ["global", "within", "between", "two-stage"],
    },
]

B_VALUES_FULL = [999, 4_999, 9_999]
B_VALUES_QUICK = [999, 4_999]

REPEATS = 3
SEED_BASE = 42

RESULTS_DIR = Path(__file__).resolve().parent / "results"
IMAGE_DIR = Path(__file__).resolve().parent / "image" / "realworld-profile"


# ------------------------------------------------------------------ #
# Budget / feasibility helpers
# ------------------------------------------------------------------ #


def _between_cell_total(cell_sizes: list[int]) -> int:
    """Product of factorials of same-size group counts."""
    counts = Counter(cell_sizes)
    total = 1
    for cnt in counts.values():
        total *= math.factorial(cnt)
    return total


def _within_total_capped(cell_sizes: list[int], cap: int) -> int:
    """Product of per-cell factorials, capped early."""
    total = 1
    for s in cell_sizes:
        total *= math.factorial(s)
        if total > cap:
            return cap
    return total


def _is_feasible(
    generator: str,
    cell_sizes: list[int],
    B: int,
) -> bool:
    """Check if the scenario can produce B unique permutations."""
    n = sum(cell_sizes)
    cap = B + 2

    if generator == "global":
        try:
            return math.factorial(n) - 1 >= B
        except (OverflowError, ValueError):
            return True  # n! is astronomical

    if generator == "within":
        total = _within_total_capped(cell_sizes, cap)
        return total - 1 >= B

    if generator == "between":
        total = _between_cell_total(cell_sizes)
        return total - 1 >= B

    if generator == "two-stage":
        bt = _between_cell_total(cell_sizes)
        wt = _within_total_capped(cell_sizes, cap)
        return min(bt * wt, cap) - 1 >= B

    return False


# ------------------------------------------------------------------ #
# Correctness validation
# ------------------------------------------------------------------ #


def _validate_result(
    result: np.ndarray,
    n: int,
    B_requested: int,
    generator: str,
    scenario_name: str,
) -> dict:
    """Validate a permutation matrix for correctness.

    Returns a dict with validation results.
    """
    issues: list[str] = []

    B_got = result.shape[0]
    if result.ndim != 2 or result.shape[1] != n:
        issues.append(f"Wrong shape: expected (*, {n}), got {result.shape}")

    # Check valid permutations
    identity = list(range(n))
    invalid_count = 0
    identity_found = False
    for i, row in enumerate(result):
        if sorted(row) != identity:
            invalid_count += 1
            if invalid_count <= 3:
                issues.append(f"Row {i} invalid: sorted={sorted(row)[:10]}...")
        if list(row) == identity:
            identity_found = True

    if invalid_count > 0:
        issues.append(f"Total invalid rows: {invalid_count}/{B_got}")

    if identity_found:
        issues.append("Identity permutation found in output")

    # Check uniqueness
    unique = len({tuple(row.tolist()) for row in result})
    if unique < B_got:
        issues.append(f"Duplicates: {B_got - unique} duplicate rows")

    return {
        "scenario": scenario_name,
        "generator": generator,
        "B_requested": B_requested,
        "B_got": B_got,
        "valid": len(issues) == 0,
        "invalid_rows": invalid_count,
        "duplicates": B_got - unique,
        "identity_present": identity_found,
        "issues": "; ".join(issues) if issues else "OK",
    }


# ------------------------------------------------------------------ #
# Benchmark runner
# ------------------------------------------------------------------ #


def _build_cells(cell_sizes: list[int]) -> np.ndarray:
    """Build a cell-label vector from a list of per-cell sizes."""
    return np.concatenate(
        [np.full(s, i, dtype=np.intp) for i, s in enumerate(cell_sizes)]
    )


def _run_single(
    generator: str,
    cell_sizes: list[int],
    B: int,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    """Run one generator call, return (result, elapsed, peak_bytes)."""
    n = sum(cell_sizes)
    cells = _build_cells(cell_sizes)

    tracemalloc.start()
    t0 = time.perf_counter()

    if generator == "global":
        result = generate_unique_permutations(
            n_samples=n,
            n_permutations=B,
            random_state=seed,
            exclude_identity=True,
        )
    elif generator == "within":
        result = generate_within_cell_permutations(
            n_samples=n,
            n_permutations=B,
            cells=cells,
            random_state=seed,
            exclude_identity=True,
        )
    elif generator == "between":
        result = generate_between_cell_permutations(
            n_samples=n,
            n_permutations=B,
            cells=cells,
            random_state=seed,
            exclude_identity=True,
        )
    elif generator == "two-stage":
        result = generate_two_stage_permutations(
            n_samples=n,
            n_permutations=B,
            cells=cells,
            random_state=seed,
            exclude_identity=True,
        )
    else:
        raise ValueError(f"Unknown generator: {generator}")

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, elapsed, peak


def run_benchmark(
    scenarios: list[dict],
    b_values: list[int],
    repeats: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run all scenarios × generators × B values.

    Returns (results_df, validations).
    """
    rows: list[dict] = []
    validations: list[dict] = []

    # Count total combos for progress
    total = 0
    for sc in scenarios:
        for gen in sc["generators"]:
            for B in b_values:
                if _is_feasible(gen, sc["cell_sizes"], B):
                    total += 1

    done = 0
    for sc in scenarios:
        cell_sizes = sc["cell_sizes"]
        n = sum(cell_sizes)
        G = len(cell_sizes)
        name = sc["name"]

        for gen in sc["generators"]:
            for B in b_values:
                if not _is_feasible(gen, cell_sizes, B):
                    print(f"  [skip] {name:30s}  {gen:10s}  B={B:5d}  (infeasible)")
                    continue

                done += 1
                times: list[float] = []
                mems: list[float] = []
                last_result = None

                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    for r in range(repeats):
                        result, elapsed, peak = _run_single(
                            gen,
                            cell_sizes,
                            B,
                            SEED_BASE + r,
                        )
                        times.append(elapsed)
                        mems.append(peak)
                        if r == 0:
                            last_result = result

                # Validate first-repeat result
                if last_result is not None:
                    v = _validate_result(
                        last_result,
                        n,
                        B,
                        gen,
                        name,
                    )
                    validations.append(v)

                med_t = float(np.median(times))
                min_t = min(times)
                max_t = max(times)
                med_m = float(np.median(mems))

                # Determine balance description
                unique_sizes = set(cell_sizes)
                if len(unique_sizes) == 1:
                    balance = "balanced"
                else:
                    ratio = max(cell_sizes) / max(min(cell_sizes), 1)
                    if ratio <= 1.5:
                        balance = "slight-imbalance"
                    elif ratio <= 3.0:
                        balance = "moderate-imbalance"
                    else:
                        balance = "high-imbalance"

                row = {
                    "scenario": name,
                    "description": sc["description"],
                    "generator": gen,
                    "n": n,
                    "G": G,
                    "B": B,
                    "min_cell": min(cell_sizes),
                    "max_cell": max(cell_sizes),
                    "balance": balance,
                    "median_time_s": med_t,
                    "min_time_s": min_t,
                    "max_time_s": max_t,
                    "median_peak_memory_MB": med_m / (1024 * 1024),
                    "B_delivered": last_result.shape[0]
                    if last_result is not None
                    else 0,
                    "valid": v["valid"] if last_result is not None else False,
                }
                rows.append(row)

                status = "OK" if row["valid"] else "FAIL"
                print(
                    f"  [{done:3d}/{total}] {name:30s}  {gen:10s}  "
                    f"n={n:5d}  G={G:2d}  B={B:5d}  "
                    f"time={med_t:8.4f}s  [{status}]"
                )

    return pd.DataFrame(rows), validations


# ------------------------------------------------------------------ #
# Charts
# ------------------------------------------------------------------ #


def _make_time_by_scenario(df: pd.DataFrame, image_dir: Path) -> None:
    """Grouped bar chart: time by scenario, coloured by generator."""
    # Use the largest B for each scenario/generator
    max_B = df["B"].max()
    sub = df[df["B"] == max_B].copy()
    if sub.empty:
        print("  Skipping time_by_scenario — no data at max B")
        return

    scenarios = sub["scenario"].unique()
    generators = sorted(sub["generator"].unique())
    colours = {
        "global": "#2196F3",
        "within": "#4CAF50",
        "between": "#FF9800",
        "two-stage": "#9C27B0",
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    bar_width = 0.18
    x = np.arange(len(scenarios))

    for i, gen in enumerate(generators):
        gen_data = sub[sub["generator"] == gen]
        times = []
        for sc in scenarios:
            match = gen_data[gen_data["scenario"] == sc]
            times.append(match["median_time_s"].values[0] if len(match) else 0)
        offset = (i - len(generators) / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            times,
            bar_width,
            label=gen,
            color=colours.get(gen, "#999"),
            edgecolor="white",
            linewidth=0.5,
        )
        # Annotate bars
        for bar, t in zip(bars, times, strict=True):
            if t > 0:
                label = f"{t:.3f}" if t < 1 else f"{t:.1f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Median time (s)")
    ax.set_title(f"Time by Scenario — B={max_B:,}")
    ax.set_yscale("log")
    ax.legend(title="Generator")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(image_dir / "time_by_scenario.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {image_dir / 'time_by_scenario.png'}")


def _make_generator_comparison(df: pd.DataFrame, image_dir: Path) -> None:
    """Scatter: time vs n for each generator, sized by B."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    generators = ["global", "within", "between", "two-stage"]
    colours = {
        "balanced": "#2196F3",
        "slight-imbalance": "#4CAF50",
        "moderate-imbalance": "#FF9800",
        "high-imbalance": "#F44336",
    }

    for ax, gen in zip(axes.flat, generators, strict=True):
        sub = df[df["generator"] == gen]
        if sub.empty:
            ax.set_title(f"{gen} (no data)")
            continue

        for balance, colour in colours.items():
            bdata = sub[sub["balance"] == balance]
            if bdata.empty:
                continue
            sizes = bdata["B"] / bdata["B"].max() * 200 + 20
            ax.scatter(
                bdata["n"],
                bdata["median_time_s"],
                s=sizes,
                c=colour,
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5,
                label=balance,
            )

        ax.set_xlabel("n (total observations)")
        ax.set_ylabel("Median time (s)")
        ax.set_title(gen)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7, title="Balance")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Generator Performance vs. Sample Size", fontsize=14)
    fig.tight_layout()
    fig.savefig(image_dir / "generator_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {image_dir / 'generator_comparison.png'}")


def _make_memory_chart(df: pd.DataFrame, image_dir: Path) -> None:
    """Bar chart: peak memory by scenario at largest B."""
    max_B = df["B"].max()
    sub = df[df["B"] == max_B].copy()
    if sub.empty:
        print("  Skipping memory chart — no data at max B")
        return

    scenarios = sub["scenario"].unique()
    generators = sorted(sub["generator"].unique())
    colours = {
        "global": "#2196F3",
        "within": "#4CAF50",
        "between": "#FF9800",
        "two-stage": "#9C27B0",
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    bar_width = 0.18
    x = np.arange(len(scenarios))

    for i, gen in enumerate(generators):
        gen_data = sub[sub["generator"] == gen]
        mems = []
        for sc in scenarios:
            match = gen_data[gen_data["scenario"] == sc]
            mems.append(match["median_peak_memory_MB"].values[0] if len(match) else 0)
        offset = (i - len(generators) / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            mems,
            bar_width,
            label=gen,
            color=colours.get(gen, "#999"),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title(f"Peak Memory by Scenario — B={max_B:,}")
    ax.legend(title="Generator")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(image_dir / "memory_by_scenario.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {image_dir / 'memory_by_scenario.png'}")


def _write_validity_report(
    validations: list[dict],
    image_dir: Path,
) -> None:
    """Write a plain-text validity report."""
    path = image_dir / "validity_report.txt"
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("PERMUTATION VALIDITY REPORT")
    lines.append("=" * 72)

    all_ok = all(v["valid"] for v in validations)
    lines.append(f"Total checks: {len(validations)}")
    lines.append(f"All valid: {all_ok}")
    lines.append("")

    if not all_ok:
        lines.append("FAILURES:")
        lines.append("-" * 72)
        for v in validations:
            if not v["valid"]:
                lines.append(
                    f"  {v['scenario']:30s}  {v['generator']:10s}  "
                    f"B={v['B_requested']:5d} → {v['B_got']:5d}  "
                    f"issues: {v['issues']}"
                )
        lines.append("")

    lines.append("SUMMARY:")
    lines.append("-" * 72)
    for v in validations:
        status = "OK" if v["valid"] else "FAIL"
        lines.append(
            f"  [{status:4s}] {v['scenario']:30s}  {v['generator']:10s}  "
            f"B={v['B_requested']:5d} → {v['B_got']:5d}"
        )

    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(f"  Saved {path}")

    if not all_ok:
        print("\n  *** VALIDITY FAILURES DETECTED — see report ***\n")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-world stress-test benchmarks",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with reduced B values for a quick smoke test",
    )
    args = parser.parse_args()

    b_values = B_VALUES_QUICK if args.quick else B_VALUES_FULL

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("REAL-WORLD STRESS-TEST BENCHMARKS")
    print("=" * 72)
    print(f"Python:   {sys.version}")
    print(f"NumPy:    {np.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"B values: {b_values}")
    print(f"Repeats:  {REPEATS}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print()

    # Print scenario summary
    print("Scenario overview:")
    print(f"  {'Name':30s}  {'n':>5s}  {'G':>3s}  {'Sizes':30s}  Generators")
    print("  " + "-" * 90)
    for sc in SCENARIOS:
        sizes = sc["cell_sizes"]
        n = sum(sizes)
        G = len(sizes)
        if G <= 6:
            size_str = str(sizes)
        else:
            size_str = f"[{min(sizes)}..{max(sizes)}] (G={G})"
        gens = ", ".join(sc["generators"])
        print(f"  {sc['name']:30s}  {n:5d}  {G:3d}  {size_str:30s}  {gens}")
    print()

    # Run benchmarks
    print("Running benchmarks...")
    print("-" * 72)
    t_start = time.perf_counter()
    df, validations = run_benchmark(SCENARIOS, b_values, REPEATS)
    t_total = time.perf_counter() - t_start

    print()
    print(f"Total benchmark time: {t_total:.1f}s")
    print(f"Total scenarios run:  {len(df)}")
    print()

    # Save CSV
    csv_path = RESULTS_DIR / "realworld_profile.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Saved results to {csv_path}")
    print()

    # Generate charts
    print("Generating charts...")
    _make_time_by_scenario(df, IMAGE_DIR)
    _make_generator_comparison(df, IMAGE_DIR)
    _make_memory_chart(df, IMAGE_DIR)
    _write_validity_report(validations, IMAGE_DIR)
    print()

    # Print summary table
    print("=" * 72)
    print("RESULTS SUMMARY (largest B)")
    print("=" * 72)
    max_B = max(b_values)
    summary = df[df["B"] == max_B].copy()
    if not summary.empty:
        print(
            f"  {'Scenario':30s}  {'Gen':10s}  {'n':>5s}  "
            f"{'B':>5s}  {'Time':>8s}  {'Mem MB':>7s}  {'OK':>3s}"
        )
        print("  " + "-" * 80)
        for _, row in summary.iterrows():
            status = "Y" if row["valid"] else "N"
            print(
                f"  {row['scenario']:30s}  {row['generator']:10s}  "
                f"{row['n']:5d}  {row['B']:5d}  "
                f"{row['median_time_s']:8.4f}  "
                f"{row['median_peak_memory_MB']:7.2f}  {status:>3s}"
            )

    # Final validity check
    all_valid = all(v["valid"] for v in validations)
    print()
    if all_valid:
        print("ALL PERMUTATIONS VALID")
    else:
        failures = [v for v in validations if not v["valid"]]
        print(f"VALIDITY FAILURES: {len(failures)}")
        for v in failures:
            print(f"  {v['scenario']}  {v['generator']}  {v['issues']}")

    print("=" * 72)


if __name__ == "__main__":
    main()

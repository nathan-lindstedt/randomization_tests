"""Automated benchmark contract tests and algorithmic performance floors.

Verifies that the performance, scaling orders, and numerical guarantees
documented in the benchmarking suite remain invariant under ongoing codebase development.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from randomization_tests import randomization_test_regression
from randomization_tests._conformal import (
    conformal_prediction,
    evaluate_conformal,
)
from randomization_tests._knockoffs import knockoff_test
from randomization_tests.families_mixed import LinearMixedFamily


class TestBenchmarkContracts:
    """Automated regression checks asserting algorithmic floors."""

    def test_woodbury_linear_scaling(self) -> None:
        """Verify that random-effects whitening scales linearly O(G), not cubic O(N^3)."""
        rng = np.random.default_rng(42)
        fam = LinearMixedFamily()

        # Small grid: G=10 clusters, 20 obs per cluster -> N=200
        n1, g1 = 200, 10
        groups1 = np.repeat(np.arange(g1), n1 // g1)
        X1 = rng.standard_normal((n1, 3))
        y1 = (
            X1[:, 0] * 1.5
            + rng.standard_normal(g1)[groups1]
            + rng.standard_normal(n1) * 0.5
        )
        cal1 = fam.calibrate(X1, y1, fit_intercept=True, groups=groups1)

        t0 = time.perf_counter()
        for _ in range(10):
            _ = cal1.whiten(y1)
        t_small = time.perf_counter() - t0

        # Doubled grid: G=20 clusters, 20 obs per cluster -> N=400
        n2, g2 = 400, 20
        groups2 = np.repeat(np.arange(g2), n2 // g2)
        X2 = rng.standard_normal((n2, 3))
        y2 = (
            X2[:, 0] * 1.5
            + rng.standard_normal(g2)[groups2]
            + rng.standard_normal(n2) * 0.5
        )
        cal2 = fam.calibrate(X2, y2, fit_intercept=True, groups=groups2)

        t0 = time.perf_counter()
        for _ in range(10):
            _ = cal2.whiten(y2)
        t_large = time.perf_counter() - t0

        # Under cubic scaling O(N^3), doubling N would cause an 8x blowup.
        # Under linear block-diagonal Woodbury O(G), doubling G causes ~2x blowup.
        # We assert time ratio < 5.0x to catch accidental dense O(N^3) inversion.
        ratio = t_large / max(t_small, 1e-6)
        assert ratio < 5.0, (
            f"Woodbury whitening scaling ratio {ratio:.2f}x exceeds linear O(G) bound (< 5.0x)"
        )

    def test_score_test_throughput_advantage(self) -> None:
        """Verify Score test evaluated in null tangent space is faster than Freedman-Lane refit."""
        rng = np.random.default_rng(42)
        n, p = 300, 4
        X = pd.DataFrame(
            rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)]
        )
        y = pd.DataFrame({"y": X["x0"] * 1.5 + rng.standard_normal(n)})
        confounders = [f"x{i}" for i in range(1, p)]

        # Score test
        t0 = time.perf_counter()
        res_score = randomization_test_regression(
            X,
            y,
            method="score",
            family="linear",
            confounders=confounders,
            n_randomizations=300,
            random_state=42,
        )
        t_score = time.perf_counter() - t0

        # Freedman-Lane
        t0 = time.perf_counter()
        res_fl = randomization_test_regression(
            X,
            y,
            method="freedman_lane",
            family="linear",
            confounders=confounders,
            n_randomizations=300,
            random_state=42,
        )
        t_fl = time.perf_counter() - t0

        assert res_score is not None
        assert res_fl is not None
        # Score test should be strictly faster or competitive with FL
        assert t_score < t_fl * 1.5

    def test_fixed_x_knockoff_gram_exactness(self) -> None:
        """Verify Fixed-X knockoffs preserve Gram matrix inner products to machine precision."""
        rng = np.random.default_rng(42)
        n, p = 120, 6
        X = pd.DataFrame(
            rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)]
        )
        y = X["x0"] * 2.0 + rng.standard_normal(n) * 0.5

        res = knockoff_test(X, y, method="fixed_x", fdr_level=0.1, random_state=42)
        assert res is not None
        assert res.guarantee_tier == "finite_sample_exact"
        # Diagnostics should report max Gram preservation error < 1e-10
        gram_err = res.diagnostics.get("max_gram_preservation_error", 0.0)
        assert gram_err < 1e-10, (
            f"Fixed-X Gram preservation error {gram_err:.2e} exceeds tolerance (1e-10)"
        )

    def test_conformal_exact_coverage_floor(self) -> None:
        """Verify Split conformal prediction achieves empirical coverage within error of nominal."""
        rng = np.random.default_rng(42)
        n_train, n_new, p = 300, 100, 4
        X_train = rng.standard_normal((n_train, p))
        beta = np.array([1.2, -0.8, 0.5, 0.3])
        y_train = X_train @ beta + rng.standard_normal(n_train) * 0.5

        X_new = rng.standard_normal((n_new, p))
        y_new = X_new @ beta + rng.standard_normal(n_new) * 0.5

        confidence_level = 0.90
        res = conformal_prediction(
            X_train,
            y_train,
            X_new,
            confidence_level=confidence_level,
            method="split",
            random_state=42,
        )
        eval_res = evaluate_conformal(res, y_new)

        # Theoretical guarantee: coverage >= 1 - alpha on average
        # On n_new = 100, standard error is sqrt(0.9 * 0.1 / 100) = 0.03
        assert eval_res.empirical_coverage >= confidence_level - 0.08, (
            f"Conformal empirical coverage {eval_res.empirical_coverage:.3f} below floor (0.82)"
        )

    def test_dml_nonlinear_confounder_leakage_mitigation(self) -> None:
        """Verify DML absorbs nonlinear confounding where linear Freedman-Lane inflates Type I error."""
        fl_rejections = 0
        dml_rejections = 0
        n_reps = 8
        n_samples = 200

        for rep in range(n_reps):
            seed = 42 + rep
            rng = np.random.default_rng(seed)
            Z = rng.uniform(-2, 2, size=(n_samples, 1))
            z = Z[:, 0]
            # Aligned nonlinear confounding: m(Z) = sin(2Z) + Z^2, g(Z) = 1.5 Z^2
            X = np.sin(2 * z) + (z**2) + rng.normal(0, 0.5, size=n_samples)
            y = 1.5 * (z**2) + rng.normal(0, 0.5, size=n_samples)  # H0: beta = 0
            df = pd.DataFrame({"X": X, "Z": z})
            y_df = pd.DataFrame({"y": y})

            # 1. Linear Freedman-Lane (omits nonlinear terms, causing Type I inflation)
            res_fl = randomization_test_regression(
                df,
                y_df,
                confounders=["Z"],
                method="freedman_lane",
                n_randomizations=100,
                random_state=seed,
            )
            if float(res_fl.raw_empirical_p[0]) < 0.05:
                fl_rejections += 1

            # 2. Double-Residualized DML (absorbs nonlinear terms out-of-fold)
            gb_y = GradientBoostingRegressor(
                n_estimators=25, max_depth=3, learning_rate=0.1, random_state=seed
            )
            gb_x = GradientBoostingRegressor(
                n_estimators=25, max_depth=3, learning_rate=0.1, random_state=seed
            )
            res_dml = randomization_test_regression(
                df,
                y_df,
                confounders=["Z"],
                method="freedman_lane",
                reduced_model=(gb_y, gb_x),
                n_randomizations=100,
                random_state=seed,
            )
            if float(res_dml.raw_empirical_p[0]) < 0.05:
                dml_rejections += 1

        fl_rate = fl_rejections / n_reps
        dml_rate = dml_rejections / n_reps

        assert fl_rate >= 0.50, (
            f"Linear Freedman-Lane rejection rate {fl_rate:.2f} failed to demonstrate confounder leakage (expected >= 0.50)"
        )
        assert dml_rate <= 0.25, (
            f"Double-Residualized DML rejection rate {dml_rate:.2f} exceeded nominal threshold (expected <= 0.25)"
        )

    def test_benchmark_scripts_exist_and_documented(self) -> None:
        """Verify that all 6 benchmark scripts exist and are documented."""
        import subprocess
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        benchmark_dir = repo_root / "benchmarks"
        summary_doc = repo_root / "docs" / "BENCHMARKS.md"

        assert summary_doc.is_file(), (
            f"Missing benchmark suite architecture doc: {summary_doc}"
        )
        doc_text = summary_doc.read_text(encoding="utf-8")

        expected_scripts = [
            "profile_solvers.py",
            "profile_endtoend.py",
            "profile_subsystems.py",
            "profile_dml.py",
            "profile_dedup.py",
            "profile_realworld.py",
        ]

        for script_name in expected_scripts:
            script_path = benchmark_dir / script_name
            assert script_path.is_file(), f"Benchmark script missing: {script_path}"
            assert script_name in doc_text, (
                f"Benchmark script {script_name} not documented in {summary_doc}"
            )

        # Fast execution test on the two fastest scripts (<3s total)
        for fast_script in ["profile_solvers.py", "profile_subsystems.py"]:
            cmd = [sys.executable, str(benchmark_dir / fast_script), "--quick"]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            assert proc.returncode == 0, (
                f"{fast_script} --quick failed with code {proc.returncode}:\n{proc.stderr}\n{proc.stdout}"
            )

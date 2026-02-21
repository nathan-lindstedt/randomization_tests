"""Tests for the diagnostics module."""

import math

import numpy as np
import pandas as pd
import pytest

from randomization_tests.diagnostics import (
    _runs_test,
    compute_all_diagnostics,
    compute_breusch_pagan,
    compute_cooks_distance,
    compute_deviance_residual_diagnostics,
    compute_divergence_flags,
    compute_exposure_r_squared,
    compute_monte_carlo_se,
    compute_permutation_coverage,
    compute_standardized_coefs,
    compute_vif,
)


# ── Fixtures ─────────────────────────────────────────────────────── #

def _make_linear_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "x3": rng.standard_normal(n),
    })
    y_values = 2.0 * X["x1"].values - 1.0 * X["x2"].values + rng.standard_normal(n) * 0.5
    return X, y_values


def _make_binary_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
    })
    logits = 2.0 * X["x1"].values + 0.0 * X["x2"].values
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_values = rng.binomial(1, probs).astype(float)
    return X, y_values


def _make_collinear_data(n=100, seed=42):
    """Create data where x2 ≈ x1, producing high VIF."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1,
        "x2": x1 + rng.standard_normal(n) * 0.01,  # nearly identical
    })
    y_values = 1.0 * x1 + rng.standard_normal(n) * 0.5
    return X, y_values


# ── Per-predictor diagnostics ────────────────────────────────────── #


class TestStandardizedCoefs:
    def test_linear_shape_matches_input(self):
        X, y = _make_linear_data()
        coefs = np.array([2.0, -1.0, 0.0])
        result = compute_standardized_coefs(X, y, coefs, is_binary=False)
        assert result.shape == (3,)

    def test_logistic_shape_matches_input(self):
        X, y = _make_binary_data()
        coefs = np.array([2.0, 0.0])
        result = compute_standardized_coefs(X, y, coefs, is_binary=True)
        assert result.shape == (2,)

    def test_linear_scales_by_sd_ratio(self):
        X, y = _make_linear_data()
        coefs = np.array([2.0, -1.0, 0.0])
        result = compute_standardized_coefs(X, y, coefs, is_binary=False)
        sd_x = np.std(X.values, axis=0, ddof=1)
        sd_y = np.std(y, ddof=1)
        expected = coefs * sd_x / sd_y
        np.testing.assert_allclose(result, expected)

    def test_logistic_scales_by_sd_x_only(self):
        X, y = _make_binary_data()
        coefs = np.array([2.0, 0.0])
        result = compute_standardized_coefs(X, y, coefs, is_binary=True)
        sd_x = np.std(X.values, axis=0, ddof=1)
        expected = coefs * sd_x
        np.testing.assert_allclose(result, expected)

    def test_zero_sd_y_returns_zeros(self):
        """If all y values are identical, standardized coefs should be 0."""
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        y = np.array([5.0, 5.0, 5.0])
        coefs = np.array([1.0])
        result = compute_standardized_coefs(X, y, coefs, is_binary=False)
        np.testing.assert_allclose(result, [0.0])


class TestVIF:
    def test_independent_predictors_low_vif(self):
        X, _ = _make_linear_data(n=500, seed=0)
        vifs = compute_vif(X)
        assert vifs.shape == (3,)
        # Independent predictors: VIF should be near 1
        for v in vifs:
            assert v < 2.0

    def test_collinear_predictors_high_vif(self):
        X, _ = _make_collinear_data()
        vifs = compute_vif(X)
        # x1 and x2 are nearly identical: VIF should be very high
        assert vifs[0] > 100
        assert vifs[1] > 100

    def test_single_predictor_vif_is_one(self):
        X = pd.DataFrame({"x1": np.arange(10, dtype=float)})
        vifs = compute_vif(X)
        assert len(vifs) == 1
        assert vifs[0] == 1.0


class TestMonteCarloSE:
    def test_shape_matches_input(self):
        raw_p = np.array([0.05, 0.5, 0.95])
        result = compute_monte_carlo_se(raw_p, n_permutations=1000)
        assert result.shape == (3,)

    def test_formula_correct(self):
        raw_p = np.array([0.05, 0.5])
        B = 999
        result = compute_monte_carlo_se(raw_p, n_permutations=B)
        expected = np.sqrt(raw_p * (1.0 - raw_p) / (B + 1))
        np.testing.assert_allclose(result, expected)

    def test_extreme_p_values_low_se(self):
        raw_p = np.array([0.001, 0.999])
        result = compute_monte_carlo_se(raw_p, n_permutations=1000)
        # Very extreme p-values should have low SE
        for se in result:
            assert se < 0.01


class TestDivergenceFlags:
    def test_no_divergence(self):
        emp = np.array([0.01, 0.06])
        cls = np.array([0.02, 0.08])
        flags = compute_divergence_flags(emp, cls, threshold=0.05)
        assert flags == ["", ""]

    def test_divergence_detected(self):
        emp = np.array([0.04, 0.06])
        cls = np.array([0.06, 0.04])
        flags = compute_divergence_flags(emp, cls, threshold=0.05)
        assert flags == ["DIVERGENT", "DIVERGENT"]

    def test_nan_produces_empty_flag(self):
        emp = np.array([np.nan, 0.04])
        cls = np.array([0.06, np.nan])
        flags = compute_divergence_flags(emp, cls, threshold=0.05)
        assert flags == ["", ""]


# ── Model-level diagnostics ──────────────────────────────────────── #


class TestBreuschPagan:
    def test_returns_expected_keys(self):
        X, y = _make_linear_data()
        result = compute_breusch_pagan(X, y)
        assert "lm_stat" in result
        assert "lm_p_value" in result
        assert "f_stat" in result
        assert "f_p_value" in result
        assert "warning" in result

    def test_homoscedastic_data_passes(self):
        # Well-behaved data with constant variance should NOT trigger warning
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x1": rng.standard_normal(200)})
        y = 1.0 * X["x1"].values + rng.standard_normal(200) * 1.0
        result = compute_breusch_pagan(X, y)
        # p-value should be relatively large (no heteroscedasticity)
        assert result["lm_p_value"] > 0.01

    def test_heteroscedastic_data_warns(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        # Error variance grows with x → heteroscedastic
        y = x + rng.standard_normal(200) * np.abs(x)
        X = pd.DataFrame({"x1": x})
        result = compute_breusch_pagan(X, y)
        assert result["lm_p_value"] < 0.05
        assert "heteroscedastic" in result["warning"]


class TestDevianceResiduals:
    def test_returns_expected_keys(self):
        X, y = _make_binary_data()
        result = compute_deviance_residual_diagnostics(X, y)
        assert "mean" in result
        assert "variance" in result
        assert "n_extreme" in result
        assert "runs_test_z" in result
        assert "runs_test_p" in result
        assert "warning" in result

    def test_well_specified_model(self):
        X, y = _make_binary_data(n=500)
        result = compute_deviance_residual_diagnostics(X, y)
        # For a well-specified logistic model, variance should be near 1
        assert 0.5 < result["variance"] < 2.0


class TestCooksDistance:
    def test_linear_returns_expected_keys(self):
        X, y = _make_linear_data()
        result = compute_cooks_distance(X, y, is_binary=False)
        assert "cooks_d" in result
        assert "n_influential" in result
        assert "threshold" in result
        assert "influential_indices" in result
        assert "warning" in result

    def test_logistic_returns_expected_keys(self):
        X, y = _make_binary_data()
        result = compute_cooks_distance(X, y, is_binary=True)
        assert "cooks_d" in result
        assert len(result["cooks_d"]) == len(y)

    def test_threshold_is_four_over_n(self):
        X, y = _make_linear_data(n=100)
        result = compute_cooks_distance(X, y, is_binary=False)
        assert result["threshold"] == pytest.approx(4.0 / 100)


class TestPermutationCoverage:
    def test_small_n(self):
        result = compute_permutation_coverage(n_samples=8, n_permutations=5000)
        assert result["n_factorial"] == math.factorial(8)
        assert result["coverage"] == pytest.approx(5000 / math.factorial(8))
        assert "%" in result["coverage_str"] or "<" in result["coverage_str"]

    def test_large_n_does_not_overflow(self):
        result = compute_permutation_coverage(n_samples=200, n_permutations=10000)
        assert result["coverage_str"]  # should produce a non-empty string
        assert result["coverage"] >= 0


class TestRunsTest:
    def test_alternating_sequence_many_runs(self):
        seq = np.array([0, 1] * 50)
        z, p = _runs_test(seq)
        # Perfectly alternating → too many runs → significant
        assert p < 0.05

    def test_all_same_returns_neutral(self):
        seq = np.zeros(50, dtype=int)
        z, p = _runs_test(seq)
        assert z == 0.0
        assert p == 1.0

    def test_short_sequence_handled(self):
        seq = np.array([1])
        z, p = _runs_test(seq)
        assert p == 1.0


# ── Aggregate helper ─────────────────────────────────────────────── #


class TestComputeAllDiagnostics:
    def test_linear_all_keys_present(self):
        X, y = _make_linear_data()
        coefs = np.array([2.0, -1.0, 0.0])
        raw_emp = np.array([0.01, 0.02, 0.5])
        raw_cls = np.array([0.01, 0.03, 0.55])
        result = compute_all_diagnostics(
            X, y, coefs, is_binary=False,
            raw_empirical_p=raw_emp, raw_classic_p=raw_cls,
            n_permutations=1000,
        )
        assert "standardized_coefs" in result
        assert "vif" in result
        assert "monte_carlo_se" in result
        assert "divergence_flags" in result
        assert "breusch_pagan" in result
        assert "cooks_distance" in result
        assert "permutation_coverage" in result
        # Linear should NOT have deviance_residuals
        assert "deviance_residuals" not in result

    def test_logistic_all_keys_present(self):
        X, y = _make_binary_data()
        coefs = np.array([2.0, 0.0])
        raw_emp = np.array([0.01, 0.5])
        raw_cls = np.array([0.01, 0.55])
        result = compute_all_diagnostics(
            X, y, coefs, is_binary=True,
            raw_empirical_p=raw_emp, raw_classic_p=raw_cls,
            n_permutations=1000,
        )
        assert "standardized_coefs" in result
        assert "vif" in result
        assert "deviance_residuals" in result
        assert "cooks_distance" in result
        # Logistic should NOT have breusch_pagan
        assert "breusch_pagan" not in result

    def test_per_predictor_lengths_match(self):
        X, y = _make_linear_data()
        coefs = np.array([2.0, -1.0, 0.0])
        raw_emp = np.array([0.01, 0.02, 0.5])
        raw_cls = np.array([0.01, 0.03, 0.55])
        result = compute_all_diagnostics(
            X, y, coefs, is_binary=False,
            raw_empirical_p=raw_emp, raw_classic_p=raw_cls,
            n_permutations=1000,
        )
        n_features = X.shape[1]
        assert len(result["standardized_coefs"]) == n_features
        assert len(result["vif"]) == n_features
        assert len(result["monte_carlo_se"]) == n_features
        assert len(result["divergence_flags"]) == n_features


# ── Integration with display ─────────────────────────────────────── #


class TestPrintDiagnosticsTable:
    def test_prints_without_error(self, capsys):
        from randomization_tests.display import print_diagnostics_table

        results = {
            "model_type": "linear",
            "extended_diagnostics": {
                "standardized_coefs": [0.8, -0.4],
                "vif": [1.1, 1.1],
                "monte_carlo_se": [0.006, 0.015],
                "divergence_flags": ["", "DIVERGENT"],
                "breusch_pagan": {
                    "lm_stat": 1.5,
                    "lm_p_value": 0.22,
                    "f_stat": 1.4,
                    "f_p_value": 0.24,
                    "warning": "",
                },
                "cooks_distance": {
                    "n_influential": 2,
                    "threshold": 0.04,
                    "warning": "2 observation(s) with Cook's D > 0.0400 (4/n).",
                },
                "permutation_coverage": {
                    "coverage_str": "5.2% of 3628800 possible",
                },
            },
        }
        print_diagnostics_table(results, ["x1", "x2"])
        captured = capsys.readouterr()
        assert "Per-predictor" in captured.out
        assert "Model-level" in captured.out
        assert "VIF" in captured.out
        assert "DIVERGENT" in captured.out
        assert "Breusch-Pagan" in captured.out
        assert "Cook" in captured.out

    def test_logistic_shows_deviance_residuals(self, capsys):
        from randomization_tests.display import print_diagnostics_table

        results = {
            "model_type": "logistic",
            "extended_diagnostics": {
                "standardized_coefs": [2.0, 0.1],
                "vif": [1.0, 1.0],
                "monte_carlo_se": [0.003, 0.015],
                "divergence_flags": ["", ""],
                "deviance_residuals": {
                    "mean": -0.01,
                    "variance": 1.02,
                    "n_extreme": 3,
                    "runs_test_z": -0.5,
                    "runs_test_p": 0.62,
                    "warning": "3 observation(s) with |deviance residual| > 2.",
                },
                "cooks_distance": {
                    "n_influential": 0,
                    "threshold": 0.02,
                    "warning": "",
                },
                "permutation_coverage": {
                    "coverage_str": "< 0.1% of 7.27e+18 possible",
                },
            },
        }
        print_diagnostics_table(results, ["x1", "x2"])
        captured = capsys.readouterr()
        assert "Deviance resid" in captured.out
        assert "Runs test" in captured.out
        # Should NOT have Breusch-Pagan for logistic
        assert "Breusch-Pagan" not in captured.out

    def test_no_extended_diagnostics_exits_silently(self, capsys):
        from randomization_tests.display import print_diagnostics_table

        results = {"model_type": "linear"}
        print_diagnostics_table(results, ["x1"])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_high_vif_warning_displayed(self, capsys):
        from randomization_tests.display import print_diagnostics_table

        results = {
            "model_type": "linear",
            "extended_diagnostics": {
                "standardized_coefs": [0.5, 0.5],
                "vif": [12.0, 6.5],
                "monte_carlo_se": [0.006, 0.015],
                "divergence_flags": ["", ""],
                "breusch_pagan": {
                    "lm_stat": 1.0,
                    "lm_p_value": 0.3,
                    "f_stat": 1.0,
                    "f_p_value": 0.3,
                    "warning": "",
                },
                "cooks_distance": {
                    "n_influential": 0,
                    "threshold": 0.04,
                    "warning": "",
                },
                "permutation_coverage": {
                    "coverage_str": "1.0% of 500000 possible",
                },
            },
        }
        print_diagnostics_table(results, ["x1", "x2"])
        captured = capsys.readouterr()
        assert "severe" in captured.out
        assert "moderate" in captured.out


# ── Exposure R² ──────────────────────────────────────────────────── #


class TestExposureRSquared:
    def test_no_confounders_returns_zeros(self):
        """With no confounders, exposure R² should be 0 for all features
        (the exposure model is just the mean)."""
        X, _ = _make_linear_data()
        r2 = compute_exposure_r_squared(X, confounders=[])
        assert len(r2) == 3
        for v in r2:
            assert v is not None
            assert v == pytest.approx(0.0, abs=1e-4)

    def test_confounders_are_none(self):
        """Confounders should have None in their position."""
        X, _ = _make_linear_data()
        r2 = compute_exposure_r_squared(X, confounders=["x3"])
        assert r2[2] is None  # x3 is the confounder
        # x1, x2 should be numeric
        assert isinstance(r2[0], float)
        assert isinstance(r2[1], float)

    def test_collinear_predictor_high_r2(self):
        """When a predictor is nearly identical to a confounder,
        exposure R² should be close to 1.0."""
        rng = np.random.default_rng(42)
        n = 100
        z = rng.standard_normal(n)
        X = pd.DataFrame({
            "x1": z + rng.standard_normal(n) * 0.01,  # nearly = z
            "z": z,
        })
        r2 = compute_exposure_r_squared(X, confounders=["z"])
        assert r2[0] > 0.99  # x1 is nearly collinear with z
        assert r2[1] is None  # z is a confounder

    def test_independent_predictor_low_r2(self):
        """An independent predictor should have low exposure R²."""
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame({
            "x1": rng.standard_normal(n),
            "z": rng.standard_normal(n),
        })
        r2 = compute_exposure_r_squared(X, confounders=["z"])
        assert r2[0] < 0.1  # x1 is independent of z

    def test_present_in_kennedy_diagnostics(self):
        """compute_all_diagnostics should include exposure_r_squared
        when method='kennedy'."""
        X, y = _make_linear_data()
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)
        coefs = np.ravel(model.coef_)
        raw_p = np.array([0.01, 0.5, 0.8])

        result = compute_all_diagnostics(
            X=X, y_values=y, model_coefs=coefs, is_binary=False,
            raw_empirical_p=raw_p, raw_classic_p=raw_p,
            n_permutations=100, method="kennedy", confounders=["x3"],
        )
        assert "exposure_r_squared" in result
        assert len(result["exposure_r_squared"]) == 3
        assert result["exposure_r_squared"][2] is None  # confounder

    def test_absent_for_ter_braak(self):
        """compute_all_diagnostics should NOT include exposure_r_squared
        for ter_braak."""
        X, y = _make_linear_data()
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)
        coefs = np.ravel(model.coef_)
        raw_p = np.array([0.01, 0.5, 0.8])

        result = compute_all_diagnostics(
            X=X, y_values=y, model_coefs=coefs, is_binary=False,
            raw_empirical_p=raw_p, raw_classic_p=raw_p,
            n_permutations=100, method="ter_braak",
        )
        assert "exposure_r_squared" not in result

    def test_absent_for_kennedy_without_confounders(self):
        """compute_all_diagnostics should NOT include exposure_r_squared
        for kennedy when no confounders are specified (all values would
        be trivially 0)."""
        X, y = _make_linear_data()
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)
        coefs = np.ravel(model.coef_)
        raw_p = np.array([0.01, 0.5, 0.8])

        result = compute_all_diagnostics(
            X=X, y_values=y, model_coefs=coefs, is_binary=False,
            raw_empirical_p=raw_p, raw_classic_p=raw_p,
            n_permutations=100, method="kennedy", confounders=[],
        )
        assert "exposure_r_squared" not in result

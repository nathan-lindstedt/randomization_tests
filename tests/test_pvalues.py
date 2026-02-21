"""Tests for the pvalues module."""

import numpy as np
import pandas as pd

from randomization_tests.pvalues import calculate_p_values


class TestCalculatePValues:
    """Tests for calculate_p_values."""

    def _make_linear_data(self, n=100, seed=42):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        })
        y = pd.DataFrame({"y": 3.0 * X["x1"] + 0.0 * X["x2"] + rng.standard_normal(n) * 0.5})
        return X, y

    def _make_binary_data(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        })
        logits = 2.0 * X["x1"] + 0.0 * X["x2"]
        probs = 1 / (1 + np.exp(-logits))
        y = pd.DataFrame({"y": rng.binomial(1, probs)})
        return X, y

    def test_returns_correct_length_linear(self):
        X, y = self._make_linear_data()
        model_coefs = np.array([3.0, 0.0])
        permuted_coefs = np.random.default_rng(0).standard_normal((500, 2))
        emp, asy, raw_emp, raw_asy = calculate_p_values(X, y, permuted_coefs, model_coefs)
        assert len(emp) == 2
        assert len(asy) == 2
        assert len(raw_emp) == 2
        assert len(raw_asy) == 2

    def test_returns_correct_length_binary(self):
        X, y = self._make_binary_data()
        model_coefs = np.array([2.0, 0.0])
        permuted_coefs = np.random.default_rng(0).standard_normal((500, 2))
        emp, asy, raw_emp, raw_asy = calculate_p_values(X, y, permuted_coefs, model_coefs)
        assert len(emp) == 2
        assert len(asy) == 2
        assert len(raw_emp) == 2
        assert len(raw_asy) == 2

    def test_significant_predictor_detected(self):
        X, y = self._make_linear_data(n=200)
        # Simulate: permuted coefs for x1 are all near 0 (null)
        rng = np.random.default_rng(0)
        permuted_coefs = rng.standard_normal((1000, 2)) * 0.3
        model_coefs = np.array([3.0, 0.01])
        emp, *_ = calculate_p_values(X, y, permuted_coefs, model_coefs)
        # x1 should be significant
        assert "(**)" in emp[0] or "(*)" in emp[0]

    def test_p_values_never_zero(self):
        """Phipson & Smyth correction ensures p > 0."""
        X, y = self._make_linear_data()
        model_coefs = np.array([100.0, 100.0])  # extreme
        permuted_coefs = np.zeros((500, 2))
        emp, *_ = calculate_p_values(X, y, permuted_coefs, model_coefs)
        for pv in emp:
            numeric = float(pv.split()[0])
            assert numeric > 0

    def test_formatting_contains_marker(self):
        X, y = self._make_linear_data()
        model_coefs = np.array([0.0, 0.0])
        permuted_coefs = np.random.default_rng(0).standard_normal((500, 2))
        emp, asy, *_ = calculate_p_values(X, y, permuted_coefs, model_coefs)
        for pv in emp + asy:
            assert "(*)" in pv or "(**)" in pv or "(ns)" in pv

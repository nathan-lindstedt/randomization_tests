"""Tests for the core permutation test engine."""

import numpy as np
import pandas as pd
import pytest

from randomization_tests.core import permutation_test_regression


def _make_linear_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "x3": rng.standard_normal(n),
    })
    y = pd.DataFrame({"y": 2.0 * X["x1"] - 1.0 * X["x2"] + rng.standard_normal(n) * 0.5})
    return X, y


def _make_binary_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
    })
    logits = 2.0 * X["x1"] + 0.0 * X["x2"]
    probs = 1 / (1 + np.exp(-logits))
    y = pd.DataFrame({"y": rng.binomial(1, probs)})
    return X, y


class TestTerBraakLinear:
    def test_returns_expected_keys(self):
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="ter_braak", random_state=42,
        )
        assert "model_coefs" in result
        assert "permuted_p_values" in result
        assert "classic_p_values" in result
        assert result["model_type"] == "linear"
        assert result["method"] == "ter_braak"

    def test_correct_number_of_coefs(self):
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="ter_braak", random_state=42,
        )
        assert len(result["model_coefs"]) == 3
        assert len(result["permuted_p_values"]) == 3
        assert len(result["classic_p_values"]) == 3

    def test_significant_predictor(self):
        X, y = _make_linear_data(n=200)
        result = permutation_test_regression(
            X, y, n_permutations=200, method="ter_braak", random_state=42,
        )
        # x1 has a strong effect — should be significant
        assert "(**)" in result["permuted_p_values"][0] or "(*)" in result["permuted_p_values"][0]


class TestTerBraakLogistic:
    def test_returns_logistic_type(self):
        X, y = _make_binary_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="ter_braak", random_state=42,
        )
        assert result["model_type"] == "logistic"

    def test_correct_number_of_coefs(self):
        X, y = _make_binary_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="ter_braak", random_state=42,
        )
        assert len(result["model_coefs"]) == 2


class TestKennedyIndividual:
    def test_linear_no_confounders(self):
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="kennedy", confounders=[], random_state=42,
        )
        assert result["method"] == "kennedy"
        assert len(result["model_coefs"]) == 3

    def test_linear_with_confounders(self):
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="kennedy", confounders=["x3"], random_state=42,
        )
        # x3 should be marked as confounder
        assert result["permuted_p_values"][2] == "N/A (confounder)"
        assert result["classic_p_values"][2] == "N/A (confounder)"

    def test_logistic_no_confounders(self):
        X, y = _make_binary_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="kennedy", confounders=[], random_state=42,
        )
        assert result["model_type"] == "logistic"


class TestKennedyJoint:
    def test_linear_returns_expected_keys(self):
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="kennedy_joint", confounders=[], random_state=42,
        )
        assert "observed_improvement" in result
        assert "p_value" in result
        assert "p_value_str" in result
        assert result["method"] == "kennedy_joint"

    def test_logistic_returns_expected_keys(self):
        X, y = _make_binary_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="kennedy_joint", confounders=[], random_state=42,
        )
        assert "observed_improvement" in result
        assert result["model_type"] == "logistic"

    def test_p_value_in_valid_range(self):
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="kennedy_joint", confounders=[], random_state=42,
        )
        assert 0 < result["p_value"] <= 1.0


class TestInvalidMethod:
    def test_raises_on_unknown_method(self):
        X, y = _make_linear_data()
        with pytest.raises(ValueError, match="Invalid method"):
            permutation_test_regression(X, y, method="not_a_method")


class TestDiagnostics:
    def test_linear_diagnostics(self):
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="ter_braak", random_state=42,
        )
        diag = result["diagnostics"]
        assert "r_squared" in diag
        assert "aic" in diag
        assert diag["n_observations"] == 100
        assert diag["n_features"] == 3

    def test_logistic_diagnostics(self):
        X, y = _make_binary_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="ter_braak", random_state=42,
        )
        diag = result["diagnostics"]
        assert "pseudo_r_squared" in diag
        assert "log_likelihood" in diag

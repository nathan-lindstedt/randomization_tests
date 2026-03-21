"""End-to-end tests for randomization_test_regression(randomization="sign_flip")."""

import numpy as np
import pandas as pd
import pytest

from randomization_tests import randomization_test_regression

# ------------------------------------------------------------------ #
# Test data factories
# ------------------------------------------------------------------ #


def _make_linear_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "x3": rng.standard_normal(n),
        }
    )
    y = pd.DataFrame(
        {"y": 2.0 * X["x1"] - 1.0 * X["x2"] + rng.standard_normal(n) * 0.5}
    )
    return X, y


def _make_binary_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    logits = 2.0 * X["x1"] + 0.0 * X["x2"]
    probs = 1 / (1 + np.exp(-logits))
    y = pd.DataFrame({"y": rng.binomial(1, probs)})
    return X, y


def _make_count_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    mu = np.exp(0.5 * X["x1"] + 0.0 * X["x2"])
    y = pd.DataFrame({"y": rng.poisson(mu)})
    return X, y


def _make_ordinal_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    latent = X["x1"] + rng.standard_normal(n)
    y = pd.DataFrame({"y": np.digitize(latent, [-1, 0, 1])})
    return X, y


# ------------------------------------------------------------------ #
# Linear family
# ------------------------------------------------------------------ #


class TestSignFlipLinear:
    """Sign-flip test with linear regression."""

    def test_returns_individual_result(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=50, random_state=42, randomization="sign_flip"
        )
        assert "model_coefs" in result
        assert "permuted_p_values" in result
        assert result.family.name == "linear"

    def test_correct_number_of_coefs(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=50, random_state=42, randomization="sign_flip"
        )
        assert len(result.model_coefs) == 3

    def test_significant_predictor_detected(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=200, random_state=42, randomization="sign_flip"
        )
        # x1 has coefficient 2.0 — should be significant.
        assert result.raw_empirical_p[0] < 0.05

    def test_null_predictor_not_significant(self):
        X, y = _make_linear_data()
        # x3 has coefficient 0.0 — should not be significant.
        result = randomization_test_regression(
            X, y, n_randomizations=200, random_state=42, randomization="sign_flip"
        )
        assert result.raw_empirical_p[2] > 0.05

    def test_p_values_in_unit_interval(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=100, random_state=42, randomization="sign_flip"
        )
        assert all(0 < p <= 1 for p in result.raw_empirical_p)

    def test_n_randomizations_matches(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=150, random_state=42, randomization="sign_flip"
        )
        assert result.n_randomizations == 150

    def test_feature_names_preserved(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=50, random_state=42, randomization="sign_flip"
        )
        assert result.feature_names == ["x1", "x2", "x3"]

    def test_target_name_preserved(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=50, random_state=42, randomization="sign_flip"
        )
        assert result.target_name == "y"


# ------------------------------------------------------------------ #
# Logistic family
# ------------------------------------------------------------------ #


class TestSignFlipLogistic:
    """Sign-flip test with logistic regression."""

    def test_logistic_runs(self):
        X, y = _make_binary_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            family="logistic",
            randomization="sign_flip",
        )
        assert result.family.name == "logistic"

    def test_logistic_p_values_valid(self):
        X, y = _make_binary_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=100,
            random_state=42,
            family="logistic",
            randomization="sign_flip",
        )
        assert all(0 < p <= 1 for p in result.raw_empirical_p)

    def test_logistic_auto_detect(self):
        X, y = _make_binary_data()
        result = randomization_test_regression(
            X, y, n_randomizations=50, random_state=42, randomization="sign_flip"
        )
        assert result.family.name == "logistic"


# ------------------------------------------------------------------ #
# Poisson family
# ------------------------------------------------------------------ #


class TestSignFlipPoisson:
    """Sign-flip test with Poisson regression."""

    def test_poisson_runs(self):
        X, y = _make_count_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            family="poisson",
            randomization="sign_flip",
        )
        assert result.family.name == "poisson"

    def test_poisson_p_values_valid(self):
        X, y = _make_count_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=100,
            random_state=42,
            family="poisson",
            randomization="sign_flip",
        )
        assert all(0 < p <= 1 for p in result.raw_empirical_p)


# ------------------------------------------------------------------ #
# Rejection for direct-permutation families
# ------------------------------------------------------------------ #


class TestSignFlipRejection:
    """Sign-flip rejects families without well-defined residuals."""

    def test_ordinal_raises(self):
        X, y = _make_ordinal_data()
        with pytest.raises(ValueError, match="well-defined residuals"):
            randomization_test_regression(
                X, y, n_randomizations=50, family="ordinal", randomization="sign_flip"
            )

    def test_multinomial_raises(self):
        X, y = _make_ordinal_data()
        # Multinomial needs >= 3 categories which our data has.
        with pytest.raises(ValueError, match="well-defined residuals"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                family="multinomial",
                randomization="sign_flip",
            )

    def test_score_exact_raises(self):
        """score_exact permutes Y directly and is incompatible with sign_flip."""
        import warnings as _w

        X, y = _make_linear_data()
        # score_exact on a non-mixed family emits UserWarning before the
        # sign_flip ValueError — suppress it so the ValueError propagates.
        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="score_exact"):
                randomization_test_regression(
                    X,
                    y,
                    n_randomizations=50,
                    method="score_exact",
                    randomization="sign_flip",
                )

    def test_manly_raises(self):
        """Manly uses direct Y randomization — no residuals to sign-flip."""
        import warnings as _w

        X, y = _make_linear_data()
        # Suppress the Manly-on-linear UserWarning so the ValueError propagates
        # (pytest converts all warnings → errors by default).
        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="Manly"):
                randomization_test_regression(
                    X, y, n_randomizations=50, method="manly", randomization="sign_flip"
                )


# ------------------------------------------------------------------ #
# Determinism
# ------------------------------------------------------------------ #


class TestSignFlipDeterminism:
    """Sign-flip results are deterministic under fixed seed."""

    def test_same_seed_produces_same_results(self):
        X, y = _make_linear_data()
        r1 = randomization_test_regression(
            X, y, n_randomizations=100, random_state=42, randomization="sign_flip"
        )
        r2 = randomization_test_regression(
            X, y, n_randomizations=100, random_state=42, randomization="sign_flip"
        )
        np.testing.assert_array_equal(r1.raw_empirical_p, r2.raw_empirical_p)
        np.testing.assert_array_equal(r1.model_coefs, r2.model_coefs)

    def test_different_seeds_differ(self):
        X, y = _make_linear_data()
        r1 = randomization_test_regression(
            X, y, n_randomizations=100, random_state=1, randomization="sign_flip"
        )
        r2 = randomization_test_regression(
            X, y, n_randomizations=100, random_state=2, randomization="sign_flip"
        )
        # Different seeds should give different permuted coefs.
        assert not np.array_equal(r1.permuted_coefs, r2.permuted_coefs)


# ------------------------------------------------------------------ #
# Confounders
# ------------------------------------------------------------------ #


class TestSignFlipConfounders:
    """Sign-flip test with confounders."""

    def test_confounder_masking(self):
        # Confounder masking ('"(confounder)"') only applies to kennedy method.
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="kennedy",
            confounders=["x3"],
            randomization="sign_flip",
        )
        # x3 should be masked as confounder.
        assert result.permuted_p_values[2] == "(confounder)"

    def test_confounder_not_found(self):
        X, y = _make_linear_data()
        with pytest.raises(ValueError, match="not found"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                confounders=["nonexistent"],
                randomization="sign_flip",
            )


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #


class TestSignFlipEdgeCases:
    """Edge cases for sign-flip randomization."""

    def test_single_feature(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x1": rng.standard_normal(50)})
        y = pd.DataFrame({"y": 2.0 * X["x1"] + rng.standard_normal(50)})
        result = randomization_test_regression(
            X, y, n_randomizations=50, random_state=42, randomization="sign_flip"
        )
        assert len(result.model_coefs) == 1

    def test_confidence_intervals_present(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=100, random_state=42, randomization="sign_flip"
        )
        assert "permutation_ci" in result.confidence_intervals
        assert "wald_ci" in result.confidence_intervals
        assert result.confidence_intervals["confidence_level"] == 0.95

    def test_diagnostics_present(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=50, random_state=42, randomization="sign_flip"
        )
        assert "n_observations" in result.diagnostics
        assert len(result.extended_diagnostics) > 0

    def test_to_dict_works(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=50, random_state=42, randomization="sign_flip"
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["family"] == "linear"

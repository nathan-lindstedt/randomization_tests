"""Tests for the core permutation test engine."""

import numpy as np
import pandas as pd
import pytest

from randomization_tests.core import permutation_test_regression


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


class TestTerBraakLinear:
    def test_returns_expected_keys(self):
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
        )
        assert "model_coefs" in result
        assert "permuted_p_values" in result
        assert "classic_p_values" in result
        assert result["model_type"] == "linear"
        assert result["method"] == "ter_braak"

    def test_correct_number_of_coefs(self):
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
        )
        assert len(result["model_coefs"]) == 3
        assert len(result["permuted_p_values"]) == 3
        assert len(result["classic_p_values"]) == 3

    def test_significant_predictor(self):
        X, y = _make_linear_data(n=200)
        result = permutation_test_regression(
            X,
            y,
            n_permutations=200,
            method="ter_braak",
            random_state=42,
        )
        # x1 has a strong effect — should be significant
        assert (
            "(**)" in result["permuted_p_values"][0]
            or "(*)" in result["permuted_p_values"][0]
        )


class TestTerBraakLogistic:
    def test_returns_logistic_type(self):
        X, y = _make_binary_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
        )
        assert result["model_type"] == "logistic"

    def test_correct_number_of_coefs(self):
        X, y = _make_binary_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
        )
        assert len(result["model_coefs"]) == 2


class TestKennedyIndividual:
    def test_linear_no_confounders(self):
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=50,
                method="kennedy",
                confounders=[],
                random_state=42,
            )
        assert result["method"] == "kennedy"
        assert len(result["model_coefs"]) == 3

    def test_linear_with_confounders(self):
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="kennedy",
            confounders=["x3"],
            random_state=42,
        )
        # x3 should be marked as confounder
        assert result["permuted_p_values"][2] == "N/A (confounder)"
        assert result["classic_p_values"][2] == "N/A (confounder)"

    def test_logistic_no_confounders(self):
        X, y = _make_binary_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=50,
                method="kennedy",
                confounders=[],
                random_state=42,
            )
        assert result["model_type"] == "logistic"


class TestKennedyJoint:
    def test_linear_returns_expected_keys(self):
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=50,
                method="kennedy_joint",
                confounders=[],
                random_state=42,
            )
        assert "observed_improvement" in result
        assert "p_value" in result
        assert "p_value_str" in result
        assert result["method"] == "kennedy_joint"

    def test_logistic_returns_expected_keys(self):
        X, y = _make_binary_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=50,
                method="kennedy_joint",
                confounders=[],
                random_state=42,
            )
        assert "observed_improvement" in result
        assert result["model_type"] == "logistic"

    def test_p_value_in_valid_range(self):
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=50,
                method="kennedy_joint",
                confounders=[],
                random_state=42,
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
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
        )
        diag = result["diagnostics"]
        assert "r_squared" in diag
        assert "aic" in diag
        assert diag["n_observations"] == 100
        assert diag["n_features"] == 3

    def test_logistic_diagnostics(self):
        X, y = _make_binary_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
        )
        diag = result["diagnostics"]
        assert "pseudo_r_squared" in diag
        assert "log_likelihood" in diag


class TestFitInterceptFalse:
    """Verify that fit_intercept=False runs without error and produces
    results in the same format as the default (fit_intercept=True)."""

    def test_ter_braak_linear_no_intercept(self):
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
            fit_intercept=False,
        )
        assert result["model_type"] == "linear"
        assert len(result["model_coefs"]) == 3
        assert len(result["permuted_p_values"]) == 3
        assert len(result["classic_p_values"]) == 3

    def test_ter_braak_logistic_no_intercept(self):
        X, y = _make_binary_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
            fit_intercept=False,
        )
        assert result["model_type"] == "logistic"
        assert len(result["model_coefs"]) == 2

    def test_kennedy_linear_no_intercept(self):
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="kennedy",
            confounders=["x3"],
            random_state=42,
            fit_intercept=False,
        )
        assert result["method"] == "kennedy"
        assert result["permuted_p_values"][2] == "N/A (confounder)"

    def test_kennedy_logistic_no_intercept(self):
        X, y = _make_binary_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=50,
                method="kennedy",
                confounders=[],
                random_state=42,
                fit_intercept=False,
            )
        assert result["model_type"] == "logistic"

    def test_kennedy_joint_linear_no_intercept(self):
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=50,
                method="kennedy_joint",
                confounders=[],
                random_state=42,
                fit_intercept=False,
            )
        assert "observed_improvement" in result
        assert 0 < result["p_value"] <= 1.0

    def test_kennedy_joint_logistic_no_intercept(self):
        X, y = _make_binary_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=50,
                method="kennedy_joint",
                confounders=[],
                random_state=42,
                fit_intercept=False,
            )
        assert result["model_type"] == "logistic"
        assert "p_value" in result

    def test_coefs_differ_from_intercept_model(self):
        """Coefficients with fit_intercept=False should generally
        differ from the default (True) fit."""
        X, y = _make_linear_data()
        res_with = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
            fit_intercept=True,
        )
        res_without = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
            fit_intercept=False,
        )
        # At least one coefficient should differ meaningfully
        coefs_with = np.array(res_with["model_coefs"])
        coefs_without = np.array(res_without["model_coefs"])
        assert not np.allclose(coefs_with, coefs_without, atol=1e-6)


class TestFamilyParameter:
    """Verify that the ``family`` parameter on
    ``permutation_test_regression`` controls model selection correctly
    and that ``validate_y`` is enforced for explicit families."""

    # ---- Explicit family matches data → success ----

    def test_explicit_linear_with_continuous_y(self):
        """Passing family='linear' with continuous Y should produce
        a linear result identical to auto-detection."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
            family="linear",
        )
        assert result["model_type"] == "linear"
        assert len(result["model_coefs"]) == 3

    def test_explicit_logistic_with_binary_y(self):
        """Passing family='logistic' with binary Y should produce
        a logistic result identical to auto-detection."""
        X, y = _make_binary_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
            family="logistic",
        )
        assert result["model_type"] == "logistic"
        assert len(result["model_coefs"]) == 2

    def test_auto_selects_linear_for_continuous(self):
        """Default family='auto' should resolve to linear for
        continuous Y."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
            family="auto",
        )
        assert result["model_type"] == "linear"

    def test_auto_selects_logistic_for_binary(self):
        """Default family='auto' should resolve to logistic for
        binary {0, 1} Y."""
        X, y = _make_binary_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
            family="auto",
        )
        assert result["model_type"] == "logistic"

    # ---- Explicit family vs data mismatch → validate_y error ----

    def test_logistic_family_with_continuous_y_raises(self):
        """Explicit family='logistic' with continuous Y should fail
        the family's validate_y check."""
        X, y = _make_linear_data()
        with pytest.raises(ValueError, match="binary"):
            permutation_test_regression(
                X,
                y,
                n_permutations=50,
                family="logistic",
            )

    def test_linear_family_with_constant_y_raises(self):
        """Explicit family='linear' with constant Y should fail
        the family's validate_y check (zero variance)."""
        X, _ = _make_linear_data()
        y_const = pd.DataFrame({"y": np.ones(len(X))})
        with pytest.raises(ValueError, match="non-constant"):
            permutation_test_regression(
                X,
                y_const,
                n_permutations=50,
                family="linear",
            )

    # ---- Unknown family string ----

    def test_unknown_family_raises(self):
        """An unrecognised family string should raise ValueError."""
        X, y = _make_linear_data()
        with pytest.raises(ValueError, match="Unknown family"):
            permutation_test_regression(
                X,
                y,
                n_permutations=50,
                family="gamma",
            )

    # ---- Explicit family matches auto result ----

    def test_explicit_linear_matches_auto(self):
        """Explicit family='linear' should produce the same
        coefficients as family='auto' on continuous data."""
        X, y = _make_linear_data()
        res_auto = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
            family="auto",
        )
        res_explicit = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
            family="linear",
        )
        np.testing.assert_array_equal(
            res_auto["model_coefs"], res_explicit["model_coefs"]
        )

    def test_explicit_logistic_matches_auto(self):
        """Explicit family='logistic' should produce the same
        coefficients as family='auto' on binary data."""
        X, y = _make_binary_data()
        res_auto = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
            family="auto",
        )
        res_explicit = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
            family="logistic",
        )
        np.testing.assert_array_equal(
            res_auto["model_coefs"], res_explicit["model_coefs"]
        )

    # ---- Kennedy methods with explicit family ----

    def test_kennedy_with_explicit_family(self):
        """Kennedy method should work with an explicit family."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="kennedy",
            confounders=["x3"],
            random_state=42,
            family="linear",
        )
        assert result["model_type"] == "linear"
        assert result["permuted_p_values"][2] == "N/A (confounder)"

    def test_kennedy_joint_with_explicit_family(self):
        """Kennedy joint method should work with an explicit family."""
        X, y = _make_binary_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=50,
                method="kennedy_joint",
                random_state=42,
                family="logistic",
            )
        assert result["model_type"] == "logistic"
        assert "observed_improvement" in result


class TestNJobs:
    """Tests for joblib-based parallel permutation fitting.

    Forces the NumPy backend so that ``n_jobs`` actually exercises
    joblib parallelism (the JAX backend uses vmap instead).
    """

    @pytest.fixture(autouse=True)
    def _use_numpy_backend(self):
        from randomization_tests import set_backend

        set_backend("numpy")
        yield
        set_backend("auto")

    def test_ter_braak_linear_n_jobs(self):
        """ter Braak linear with n_jobs=2 matches n_jobs=1."""
        X, y = _make_linear_data()
        r1 = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=0,
            n_jobs=1,
        )
        r2 = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=0,
            n_jobs=2,
        )
        np.testing.assert_allclose(
            r1["raw_empirical_p"],
            r2["raw_empirical_p"],
            rtol=1e-10,
        )

    def test_ter_braak_logistic_n_jobs(self):
        """ter Braak logistic with n_jobs=2 matches n_jobs=1."""
        X, y = _make_binary_data()
        r1 = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=0,
            n_jobs=1,
        )
        r2 = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=0,
            n_jobs=2,
        )
        np.testing.assert_allclose(
            r1["raw_empirical_p"],
            r2["raw_empirical_p"],
            rtol=1e-10,
        )

    def test_kennedy_linear_n_jobs(self):
        """Kennedy individual linear with n_jobs=2 matches n_jobs=1."""
        X, y = _make_linear_data()
        # x3 is noise — use as a confounder
        r1 = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="kennedy",
            confounders=["x3"],
            random_state=0,
            n_jobs=1,
        )
        r2 = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="kennedy",
            confounders=["x3"],
            random_state=0,
            n_jobs=2,
        )
        np.testing.assert_allclose(
            r1["raw_empirical_p"],
            r2["raw_empirical_p"],
            rtol=1e-10,
        )

    def test_kennedy_joint_n_jobs(self):
        """Kennedy joint with n_jobs=2 matches n_jobs=1."""
        X, y = _make_linear_data()
        r1 = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="kennedy_joint",
            confounders=["x3"],
            random_state=0,
            n_jobs=1,
        )
        r2 = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="kennedy_joint",
            confounders=["x3"],
            random_state=0,
            n_jobs=2,
        )
        np.testing.assert_allclose(
            r1["observed_improvement"],
            r2["observed_improvement"],
            rtol=1e-10,
        )
        assert r1["p_value"] == r2["p_value"]

    def test_n_jobs_minus_one_works(self):
        """n_jobs=-1 (all cores) should run without error."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=0,
            n_jobs=-1,
        )
        assert "permuted_p_values" in result

    def test_freedman_lane_linear_n_jobs(self):
        """Freedman–Lane individual with n_jobs=2 matches n_jobs=1."""
        X, y = _make_linear_data()
        r1 = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane",
            confounders=["x3"],
            random_state=0,
            n_jobs=1,
        )
        r2 = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane",
            confounders=["x3"],
            random_state=0,
            n_jobs=2,
        )
        np.testing.assert_allclose(
            r1["raw_empirical_p"],
            r2["raw_empirical_p"],
            rtol=1e-10,
        )

    def test_freedman_lane_joint_n_jobs(self):
        """Freedman–Lane joint with n_jobs=2 matches n_jobs=1."""
        X, y = _make_linear_data()
        r1 = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane_joint",
            confounders=["x3"],
            random_state=0,
            n_jobs=1,
        )
        r2 = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane_joint",
            confounders=["x3"],
            random_state=0,
            n_jobs=2,
        )
        np.testing.assert_allclose(
            r1["observed_improvement"],
            r2["observed_improvement"],
            rtol=1e-10,
        )
        assert r1["p_value"] == r2["p_value"]


# ------------------------------------------------------------------ #
# Freedman–Lane individual
# ------------------------------------------------------------------ #


class TestFreedmanLaneIndividual:
    def test_linear_with_confounders(self):
        """Freedman–Lane individual with confounders produces expected keys."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane",
            confounders=["x3"],
            random_state=42,
        )
        assert result["method"] == "freedman_lane"
        assert result["model_type"] == "linear"
        assert len(result["model_coefs"]) == 3
        # x3 is a confounder — its p-value should be N/A
        assert result["permuted_p_values"][2] == "N/A (confounder)"
        assert result["classic_p_values"][2] == "N/A (confounder)"
        assert np.isnan(result["raw_empirical_p"][2])

    def test_linear_no_confounders_warns(self):
        """Freedman–Lane without confounders issues a UserWarning."""
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=50,
                method="freedman_lane",
                confounders=[],
                random_state=42,
            )
        assert result["method"] == "freedman_lane"
        assert len(result["model_coefs"]) == 3

    def test_logistic_with_confounders(self):
        """Freedman–Lane individual with logistic family and confounders."""
        X, y = _make_binary_data()
        # Add a noise confounder column
        rng = np.random.default_rng(99)
        X = X.copy()
        X["z1"] = rng.standard_normal(len(X))
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane",
            confounders=["z1"],
            random_state=42,
        )
        assert result["model_type"] == "logistic"
        assert result["method"] == "freedman_lane"
        # z1 is a confounder
        z1_idx = list(X.columns).index("z1")
        assert result["permuted_p_values"][z1_idx] == "N/A (confounder)"

    def test_logistic_no_confounders_warns(self):
        """Freedman–Lane logistic without confounders warns."""
        X, y = _make_binary_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=50,
                method="freedman_lane",
                confounders=[],
                random_state=42,
            )
        assert result["model_type"] == "logistic"

    def test_p_values_in_valid_range(self):
        """All non-confounder p-values should be in (0, 1]."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane",
            confounders=["x3"],
            random_state=42,
        )
        for i, col in enumerate(X.columns):
            if col != "x3":
                p = result["raw_empirical_p"][i]
                assert 0 < p <= 1.0, f"p-value for {col} = {p}"

    def test_explicit_family_linear(self):
        """Explicit family='linear' works with Freedman–Lane."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane",
            confounders=["x3"],
            random_state=42,
            family="linear",
        )
        assert result["model_type"] == "linear"

    def test_explicit_family_logistic(self):
        """Explicit family='logistic' works with Freedman–Lane."""
        X, y = _make_binary_data()
        rng = np.random.default_rng(99)
        X = X.copy()
        X["z1"] = rng.standard_normal(len(X))
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane",
            confounders=["z1"],
            random_state=42,
            family="logistic",
        )
        assert result["model_type"] == "logistic"

    def test_fit_intercept_false(self):
        """Freedman–Lane with fit_intercept=False runs correctly."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane",
            confounders=["x3"],
            random_state=42,
            fit_intercept=False,
        )
        assert result["method"] == "freedman_lane"
        assert len(result["model_coefs"]) == 3

    def test_permuted_coefs_in_result(self):
        """Result dict should include permuted_coefs."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane",
            confounders=["x3"],
            random_state=42,
        )
        assert "permuted_coefs" in result
        coefs = np.array(result["permuted_coefs"])
        assert coefs.shape == (50, 3)


# ------------------------------------------------------------------ #
# Freedman–Lane joint
# ------------------------------------------------------------------ #


class TestFreedmanLaneJoint:
    def test_linear_returns_expected_keys(self):
        """Freedman–Lane joint linear returns all expected keys."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane_joint",
            confounders=["x3"],
            random_state=42,
        )
        assert "observed_improvement" in result
        assert "permuted_improvements" in result
        assert "p_value" in result
        assert "p_value_str" in result
        assert result["method"] == "freedman_lane_joint"
        assert result["metric_type"] == "RSS Reduction"

    def test_logistic_returns_expected_keys(self):
        """Freedman–Lane joint logistic returns all expected keys."""
        X, y = _make_binary_data()
        rng = np.random.default_rng(99)
        X = X.copy()
        X["z1"] = rng.standard_normal(len(X))
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane_joint",
            confounders=["z1"],
            random_state=42,
        )
        assert "observed_improvement" in result
        assert result["model_type"] == "logistic"
        assert result["metric_type"] == "Deviance Reduction"

    def test_p_value_in_valid_range(self):
        """Joint p-value should be in (0, 1]."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane_joint",
            confounders=["x3"],
            random_state=42,
        )
        assert 0 < result["p_value"] <= 1.0

    def test_no_confounders_warns(self):
        """Freedman–Lane joint without confounders warns."""
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=50,
                method="freedman_lane_joint",
                confounders=[],
                random_state=42,
            )
        assert result["method"] == "freedman_lane_joint"

    def test_features_tested(self):
        """features_tested should list only non-confounder columns."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane_joint",
            confounders=["x3"],
            random_state=42,
        )
        assert result["features_tested"] == ["x1", "x2"]
        assert result["confounders"] == ["x3"]

    def test_permuted_improvements_length(self):
        """permuted_improvements should have n_permutations entries."""
        X, y = _make_linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="freedman_lane_joint",
            confounders=["x3"],
            random_state=42,
        )
        assert len(result["permuted_improvements"]) == 50

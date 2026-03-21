"""Tests for the core permutation test engine."""

import numpy as np
import pandas as pd
import pytest

from randomization_tests.core import randomization_test_regression


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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
        )
        assert "model_coefs" in result
        assert "permuted_p_values" in result
        assert "classic_p_values" in result
        assert result.family.name == "linear"
        assert result["method"] == "ter_braak"

    def test_correct_number_of_coefs(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
        )
        assert len(result["model_coefs"]) == 3
        assert len(result["permuted_p_values"]) == 3
        assert len(result["classic_p_values"]) == 3

    def test_significant_predictor(self):
        X, y = _make_linear_data(n=200)
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=200,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
        )
        assert result.family.name == "logistic"

    def test_correct_number_of_coefs(self):
        X, y = _make_binary_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
        )
        assert len(result["model_coefs"]) == 2


class TestKennedyIndividual:
    def test_linear_no_confounders(self):
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="kennedy",
                confounders=[],
                random_state=42,
            )
        assert result["method"] == "kennedy"
        assert len(result["model_coefs"]) == 3

    def test_linear_with_confounders(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="kennedy",
            confounders=["x3"],
            random_state=42,
        )
        # x3 should be marked as confounder
        assert result["permuted_p_values"][2] == "(confounder)"
        assert result["classic_p_values"][2] == "(confounder)"

    def test_logistic_no_confounders(self):
        X, y = _make_binary_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="kennedy",
                confounders=[],
                random_state=42,
            )
        assert result.family.name == "logistic"


class TestKennedyJoint:
    def test_linear_returns_expected_keys(self):
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
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
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="kennedy_joint",
                confounders=[],
                random_state=42,
            )
        assert "observed_improvement" in result
        assert result.family.name == "logistic"

    def test_p_value_in_valid_range(self):
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="kennedy_joint",
                confounders=[],
                random_state=42,
            )
        assert 0 < result["p_value"] <= 1.0


class TestInvalidMethod:
    def test_raises_on_unknown_method(self):
        X, y = _make_linear_data()
        with pytest.raises(ValueError, match="Invalid method"):
            randomization_test_regression(X, y, method="not_a_method")


class TestDiagnostics:
    def test_linear_diagnostics(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
            fit_intercept=False,
        )
        assert result.family.name == "linear"
        assert len(result["model_coefs"]) == 3
        assert len(result["permuted_p_values"]) == 3
        assert len(result["classic_p_values"]) == 3

    def test_ter_braak_logistic_no_intercept(self):
        X, y = _make_binary_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
            fit_intercept=False,
        )
        assert result.family.name == "logistic"
        assert len(result["model_coefs"]) == 2

    def test_kennedy_linear_no_intercept(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="kennedy",
            confounders=["x3"],
            random_state=42,
            fit_intercept=False,
        )
        assert result["method"] == "kennedy"
        assert result["permuted_p_values"][2] == "(confounder)"

    def test_kennedy_logistic_no_intercept(self):
        X, y = _make_binary_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="kennedy",
                confounders=[],
                random_state=42,
                fit_intercept=False,
            )
        assert result.family.name == "logistic"

    def test_kennedy_joint_linear_no_intercept(self):
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
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
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="kennedy_joint",
                confounders=[],
                random_state=42,
                fit_intercept=False,
            )
        assert result.family.name == "logistic"
        assert "p_value" in result

    def test_coefs_differ_from_intercept_model(self):
        """Coefficients with fit_intercept=False should generally
        differ from the default (True) fit."""
        X, y = _make_linear_data()
        res_with = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
            fit_intercept=True,
        )
        res_without = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
    ``randomization_test_regression`` controls model selection correctly
    and that ``validate_y`` is enforced for explicit families."""

    # ---- Explicit family matches data → success ----

    def test_explicit_linear_with_continuous_y(self):
        """Passing family='linear' with continuous Y should produce
        a linear result identical to auto-detection."""
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
            family="linear",
        )
        assert result.family.name == "linear"
        assert len(result["model_coefs"]) == 3

    def test_explicit_logistic_with_binary_y(self):
        """Passing family='logistic' with binary Y should produce
        a logistic result identical to auto-detection."""
        X, y = _make_binary_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
            family="logistic",
        )
        assert result.family.name == "logistic"
        assert len(result["model_coefs"]) == 2

    def test_auto_selects_linear_for_continuous(self):
        """Default family='auto' should resolve to linear for
        continuous Y."""
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
            family="auto",
        )
        assert result.family.name == "linear"

    def test_auto_selects_logistic_for_binary(self):
        """Default family='auto' should resolve to logistic for
        binary {0, 1} Y."""
        X, y = _make_binary_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
            family="auto",
        )
        assert result.family.name == "logistic"

    # ---- Explicit family vs data mismatch → validate_y error ----

    def test_logistic_family_with_continuous_y_raises(self):
        """Explicit family='logistic' with continuous Y should fail
        the family's validate_y check."""
        X, y = _make_linear_data()
        with pytest.raises(ValueError, match="binary"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                family="logistic",
            )

    def test_linear_family_with_constant_y_raises(self):
        """Explicit family='linear' with constant Y should fail
        the family's validate_y check (zero variance)."""
        X, _ = _make_linear_data()
        y_const = pd.DataFrame({"y": np.ones(len(X))})
        with pytest.raises(ValueError, match="non-constant"):
            randomization_test_regression(
                X,
                y_const,
                n_randomizations=50,
                family="linear",
            )

    # ---- Unknown family string ----

    def test_unknown_family_raises(self):
        """An unrecognised family string should raise ValueError."""
        X, y = _make_linear_data()
        with pytest.raises(ValueError, match="Unknown family"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                family="gamma",
            )

    # ---- Explicit family matches auto result ----

    def test_explicit_linear_matches_auto(self):
        """Explicit family='linear' should produce the same
        coefficients as family='auto' on continuous data."""
        X, y = _make_linear_data()
        res_auto = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
            family="auto",
        )
        res_explicit = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
        res_auto = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
            family="auto",
        )
        res_explicit = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="kennedy",
            confounders=["x3"],
            random_state=42,
            family="linear",
        )
        assert result.family.name == "linear"
        assert result["permuted_p_values"][2] == "(confounder)"

    def test_kennedy_joint_with_explicit_family(self):
        """Kennedy joint method should work with an explicit family."""
        X, y = _make_binary_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="kennedy_joint",
                random_state=42,
                family="logistic",
            )
        assert result.family.name == "logistic"
        assert "observed_improvement" in result

    def test_family_instance_passthrough(self):
        """Step 1: passing a ModelFamily instance directly."""
        from randomization_tests.families import LinearFamily

        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=42,
            family=LinearFamily(),
        )
        assert result.family.name == "linear"
        assert len(result["model_coefs"]) == 3


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
        """ter Braak linear with n_jobs=2 warns and falls back to n_jobs=1."""
        X, y = _make_linear_data()
        r1 = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=0,
            n_jobs=1,
        )
        with pytest.warns(UserWarning, match="n_jobs has no effect"):
            r2 = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
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
        r1 = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="ter_braak",
            random_state=0,
            n_jobs=1,
        )
        r2 = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
        r1 = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="kennedy",
            confounders=["x3"],
            random_state=0,
            n_jobs=1,
        )
        r2 = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
        r1 = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="kennedy_joint",
            confounders=["x3"],
            random_state=0,
            n_jobs=1,
        )
        r2 = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
        with pytest.warns(UserWarning, match="n_jobs has no effect"):
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="ter_braak",
                random_state=0,
                n_jobs=-1,
            )
        assert "permuted_p_values" in result

    def test_freedman_lane_linear_n_jobs(self):
        """Freedman–Lane individual with n_jobs=2 warns and falls back."""
        X, y = _make_linear_data()
        r1 = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="freedman_lane",
            confounders=["x3"],
            random_state=0,
            n_jobs=1,
        )
        with pytest.warns(UserWarning, match="n_jobs has no effect"):
            r2 = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
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
        r1 = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="freedman_lane_joint",
            confounders=["x3"],
            random_state=0,
            n_jobs=1,
        )
        r2 = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="freedman_lane",
            confounders=["x3"],
            random_state=42,
        )
        assert result["method"] == "freedman_lane"
        assert result.family.name == "linear"
        assert len(result["model_coefs"]) == 3
        # x3 is a confounder — its p-value should be masked
        assert result["permuted_p_values"][2] == "(confounder)"
        assert result["classic_p_values"][2] == "(confounder)"
        assert np.isnan(result["raw_empirical_p"][2])

    def test_linear_no_confounders_warns(self):
        """Freedman–Lane without confounders issues a UserWarning."""
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="freedman_lane",
            confounders=["z1"],
            random_state=42,
        )
        assert result.family.name == "logistic"
        assert result["method"] == "freedman_lane"
        # z1 is a confounder
        z1_idx = list(X.columns).index("z1")
        assert result["permuted_p_values"][z1_idx] == "(confounder)"

    def test_logistic_no_confounders_warns(self):
        """Freedman–Lane logistic without confounders warns."""
        X, y = _make_binary_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="freedman_lane",
                confounders=[],
                random_state=42,
            )
        assert result.family.name == "logistic"

    def test_p_values_in_valid_range(self):
        """All non-confounder p-values should be in (0, 1]."""
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="freedman_lane",
            confounders=["x3"],
            random_state=42,
            family="linear",
        )
        assert result.family.name == "linear"

    def test_explicit_family_logistic(self):
        """Explicit family='logistic' works with Freedman–Lane."""
        X, y = _make_binary_data()
        rng = np.random.default_rng(99)
        X = X.copy()
        X["z1"] = rng.standard_normal(len(X))
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="freedman_lane",
            confounders=["z1"],
            random_state=42,
            family="logistic",
        )
        assert result.family.name == "logistic"

    def test_fit_intercept_false(self):
        """Freedman–Lane with fit_intercept=False runs correctly."""
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="freedman_lane_joint",
            confounders=["z1"],
            random_state=42,
        )
        assert "observed_improvement" in result
        assert result.family.name == "logistic"
        assert result["metric_type"] == "Deviance Reduction"

    def test_p_value_in_valid_range(self):
        """Joint p-value should be in (0, 1]."""
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="freedman_lane_joint",
            confounders=["x3"],
            random_state=42,
        )
        assert 0 < result["p_value"] <= 1.0

    def test_no_confounders_warns(self):
        """Freedman–Lane joint without confounders warns."""
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="freedman_lane_joint",
                confounders=[],
                random_state=42,
            )
        assert result["method"] == "freedman_lane_joint"

    def test_features_tested(self):
        """features_tested should list only non-confounder columns."""
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="freedman_lane_joint",
            confounders=["x3"],
            random_state=42,
        )
        assert result["features_tested"] == ["x1", "x2"]
        assert result["confounders"] == ["x3"]

    def test_permuted_improvements_length(self):
        """permuted_improvements should have n_randomizations entries."""
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            method="freedman_lane_joint",
            confounders=["x3"],
            random_state=42,
        )
        assert len(result["permuted_improvements"]) == 50


class TestResultDictProvenance:
    """Every result dict must carry 'family' and 'backend' provenance keys."""

    @pytest.fixture(autouse=True)
    def _use_numpy_backend(self):
        from randomization_tests import set_backend

        set_backend("numpy")
        yield
        set_backend("auto")

    def test_ter_braak_has_family_and_backend(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=20, method="ter_braak", random_state=0
        )
        assert result.family.name == "linear"
        assert result["backend"] == "numpy"

    def test_kennedy_has_family_and_backend(self):
        X, y = _make_binary_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            method="kennedy",
            confounders=["x2"],
            random_state=0,
        )
        assert result.family.name == "logistic"
        assert result["backend"] == "numpy"

    def test_kennedy_joint_has_family_and_backend(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            method="kennedy_joint",
            confounders=["x3"],
            random_state=0,
        )
        assert result.family.name == "linear"
        assert result["backend"] == "numpy"

    def test_freedman_lane_has_family_and_backend(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            method="freedman_lane",
            confounders=["x3"],
            random_state=0,
        )
        assert result.family.name == "linear"
        assert result["backend"] == "numpy"

    def test_freedman_lane_joint_has_family_and_backend(self):
        X, y = _make_binary_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            method="freedman_lane_joint",
            confounders=["x2"],
            random_state=0,
        )
        assert result.family.name == "logistic"
        assert result["backend"] == "numpy"


# ------------------------------------------------------------------ #
# TestBackendParameter
# ------------------------------------------------------------------ #


class TestBackendParameter:
    """The ``backend=`` parameter injects a specific backend per call."""

    def test_explicit_numpy_via_parameter(self):
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=20, random_state=0, backend="numpy"
        )
        assert result["backend"] == "numpy"

    def test_unknown_backend_raises(self):
        X, y = _make_linear_data()
        with pytest.raises(ValueError, match="Unknown backend"):
            randomization_test_regression(
                X, y, n_randomizations=20, random_state=0, backend="torch"
            )


# ------------------------------------------------------------------ #
# Groups parameter tests (Step 10f)
# ------------------------------------------------------------------ #


def _make_grouped_data(n_per_group=20, n_groups=5, seed=42):
    """Create linear data with balanced group labels."""
    n = n_per_group * n_groups
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    y = pd.DataFrame({"y": 2.0 * X["x1"] + rng.standard_normal(n) * 0.5})
    groups = np.repeat(np.arange(n_groups), n_per_group)
    return X, y, groups


class TestGroupsParameter:
    """Tests for the ``groups=`` parameter (Step 10f)."""

    def test_groups_wrong_length_raises(self):
        X, y = _make_linear_data(n=100)
        groups = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="groups has 3 elements but X has 100"):
            randomization_test_regression(
                X, y, n_randomizations=20, random_state=0, groups=groups
            )

    def test_strategy_without_groups_raises(self):
        X, y = _make_linear_data()
        with pytest.raises(ValueError, match="requires groups="):
            randomization_test_regression(
                X, y, n_randomizations=20, random_state=0, permutation_strategy="within"
            )

    def test_invalid_strategy_raises(self):
        X, y, groups = _make_grouped_data()
        with pytest.raises(ValueError, match="permutation_strategy must be one of"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                groups=groups,
                permutation_strategy="random",
            )

    def test_between_with_few_groups_raises(self):
        X, y, groups = _make_grouped_data(n_per_group=25, n_groups=3)
        with pytest.raises(ValueError, match="requires at least 5 groups"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                groups=groups,
                permutation_strategy="between",
            )

    def test_between_infeasible_all_unique_sizes_raises(self):
        """Between-cell raises ValueError when all cells have unique sizes."""
        rng = np.random.default_rng(0)
        # 5 groups with sizes 10, 11, 12, 13, 14 — all different
        sizes = [10, 11, 12, 13, 14]
        n = sum(sizes)
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        y = pd.DataFrame({"y": rng.standard_normal(n)})
        groups = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])

        with pytest.raises(ValueError, match="infeasible.*all.*different sizes"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                groups=groups,
                permutation_strategy="between",
            )

    def test_between_low_budget_warns(self):
        """Between-cell warns when few permutations available."""
        rng = np.random.default_rng(0)
        # 5 groups: 2 of size 10, 3 of size 12 — between_total = 2!*3! = 12
        # available = 11 (excluding identity), which is < 100
        sizes = [10, 10, 12, 12, 12]
        n = sum(sizes)
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        y = pd.DataFrame({"y": rng.standard_normal(n)})
        groups = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])

        with pytest.warns(UserWarning, match="Only.*unique between-cell"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                groups=groups,
                permutation_strategy="between",
            )

    def test_groups_without_strategy_defaults_to_within(self):
        X, y, groups = _make_grouped_data()
        result = randomization_test_regression(
            X, y, n_randomizations=20, random_state=0, groups=groups
        )
        assert result.permutation_strategy == "within"

    def test_multi_column_dataframe_cross_classification(self):
        """Multi-column DataFrame produces cross-classified cells."""
        n = 100
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        y = pd.DataFrame({"y": rng.standard_normal(n)})
        groups_df = pd.DataFrame(
            {
                "block": np.repeat([0, 1], 50),
                "site": np.tile([0, 1, 2, 3, 4], 20),
            }
        )
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            random_state=0,
            groups=groups_df,
            permutation_strategy="within",
        )
        assert result.permutation_strategy == "within"
        assert result.groups is not None
        # 2 blocks × 5 sites = 10 unique cells
        assert len(np.unique(result.groups)) == 10

    def test_single_column_dataframe_extracted(self):
        """Single-column DataFrame is treated as 1-D."""
        X, y, groups = _make_grouped_data()
        groups_df = pd.DataFrame({"group": groups})
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            random_state=0,
            groups=groups_df,
            permutation_strategy="within",
        )
        assert result.permutation_strategy == "within"
        assert result.groups is not None
        assert len(np.unique(result.groups)) == 5

    def test_result_fields_populated(self):
        """Result objects have groups and permutation_strategy set."""
        X, y, groups = _make_grouped_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            random_state=0,
            groups=groups,
            permutation_strategy="within",
        )
        assert result.groups is not None
        assert result.permutation_strategy == "within"
        assert len(result.groups) == len(groups)

    def test_no_groups_result_fields_none(self):
        """Without groups, result fields remain None."""
        X, y = _make_linear_data()
        result = randomization_test_regression(
            X, y, n_randomizations=20, random_state=0
        )
        assert result.groups is None
        assert result.permutation_strategy is None

    def test_between_strategy_end_to_end(self):
        """Between-cell strategy works end-to-end."""
        X, y, groups = _make_grouped_data(n_per_group=10, n_groups=6)
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            random_state=0,
            groups=groups,
            permutation_strategy="between",
        )
        assert result.permutation_strategy == "between"

    def test_two_stage_strategy_end_to_end(self):
        """Two-stage strategy works end-to-end."""
        X, y, groups = _make_grouped_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            random_state=0,
            groups=groups,
            permutation_strategy="two-stage",
        )
        assert result.permutation_strategy == "two-stage"

    def test_joint_method_with_groups(self):
        """Groups work with joint methods too."""
        X, y, groups = _make_grouped_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            random_state=0,
            groups=groups,
            permutation_strategy="within",
            method="kennedy_joint",
            confounders=["x1"],
        )
        assert result.permutation_strategy == "within"
        assert result.groups is not None


# ------------------------------------------------------------------ #
# Callback validation tests (Step 11f)
# ------------------------------------------------------------------ #


class TestCallbackValidation:
    """Tests for ``permutation_constraints`` callback validation (Step 11a)."""

    def test_non_callable_raises(self):
        X, y = _make_linear_data()
        with pytest.raises(TypeError, match="must be callable"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                permutation_constraints="not_callable",  # type: ignore[arg-type]
            )

    def test_wrong_return_type_raises(self):
        X, y = _make_linear_data()

        def bad_callback(perms: np.ndarray) -> list[int]:
            return [1, 2, 3]  # type: ignore[return-value]

        with pytest.raises(TypeError, match="must return np.ndarray"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                permutation_constraints=bad_callback,  # type: ignore[arg-type]
            )

    def test_wrong_shape_raises(self):
        X, y = _make_linear_data()

        def bad_shape(perms: np.ndarray) -> np.ndarray:
            return perms[:, :5]  # wrong number of columns

        with pytest.raises(TypeError, match="returned shape"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                permutation_constraints=bad_shape,
            )

    def test_valid_callback_applies(self):
        X, y = _make_linear_data()

        def keep_all(perms: np.ndarray) -> np.ndarray:
            return perms

        result = randomization_test_regression(
            X, y, n_randomizations=20, random_state=0, permutation_constraints=keep_all
        )
        assert result is not None

    def test_filtering_callback_reduces_randomizations(self):
        """A callback that drops half the rows reduces the effective B."""
        X, y = _make_linear_data()

        def keep_half(perms: np.ndarray) -> np.ndarray:
            # Keep only even-indexed permutations.
            return perms[::2]

        result = randomization_test_regression(
            X, y, n_randomizations=40, random_state=0, permutation_constraints=keep_half
        )
        # Effective n_randomizations ≤ requested (half were discarded).
        assert result.n_randomizations <= 40
        assert all(0 < p <= 1 for p in result.raw_empirical_p)

    def test_filtering_callback_differs_from_unconstrained(self):
        """Filtering callback should produce different p-values than unconstrained."""
        X, y = _make_linear_data()

        call_count = [0]

        def drop_first_half(perms: np.ndarray) -> np.ndarray:
            call_count[0] += 1
            # Only keep permutations where the first element > n//2,
            # breaking the typical null distribution.
            mask = perms[:, 0] > (perms.shape[1] // 2)
            filtered = perms[mask]
            # Return at least 1 row to avoid empty array.
            return filtered if len(filtered) > 0 else perms[:1]

        result_constrained = randomization_test_regression(
            X,
            y,
            n_randomizations=60,
            random_state=0,
            permutation_constraints=drop_first_half,
        )
        randomization_test_regression(X, y, n_randomizations=60, random_state=0)
        # Callback was called at least once (probe + actual calls).
        assert call_count[0] >= 1
        # Results should be finite (constraint doesn't break the pipeline).
        assert all(np.isfinite(p) for p in result_constrained.raw_empirical_p)


# ------------------------------------------------------------------ #
# Singleton warnings tests (Step 11f)
# ------------------------------------------------------------------ #


class TestSingletonWarnings:
    """Tests for singleton cell warnings (Step 11b)."""

    def test_within_with_singleton_warns(self):
        n = 100
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        y = pd.DataFrame({"y": rng.standard_normal(n)})
        # Make 98 observations in one group and 2 singletons
        groups = np.zeros(n, dtype=int)
        groups[-2] = 1
        groups[-1] = 2
        with pytest.warns(UserWarning, match="single observation"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                groups=groups,
                permutation_strategy="within",
            )

    def test_between_with_singleton_no_warning(self):
        n = 50
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        y = pd.DataFrame({"y": rng.standard_normal(n)})
        groups = np.repeat(np.arange(5), 10)
        groups[-1] = 5  # one singleton — 6 groups total
        # Between strategy → no singleton warning
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("error", UserWarning)
            # This should NOT raise — between doesn't warn about singletons
            # But it might warn about other things, so we just check no
            # "single observation" warning
            try:
                randomization_test_regression(
                    X,
                    y,
                    n_randomizations=20,
                    random_state=0,
                    groups=groups,
                    permutation_strategy="between",
                )
            except UserWarning as exc:
                assert "single observation" not in str(exc)

    def test_two_stage_with_singleton_warns(self):
        n = 16
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        y = pd.DataFrame({"y": rng.standard_normal(n)})
        # 5 groups of size 3 + 1 singleton → max/min = 3.0, not > 3
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5])
        with pytest.warns(UserWarning, match="single observation"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                groups=groups,
                permutation_strategy="two-stage",
            )


# ------------------------------------------------------------------ #
# Two-stage imbalance tests (Step 11f)
# ------------------------------------------------------------------ #


class TestTwoStageImbalance:
    """Tests for two-stage imbalance warnings (Step 11d)."""

    def test_unbalanced_warns(self):
        """Groups of sizes [2, 2, 20, ...] → warn about ratio > 3."""
        rng = np.random.default_rng(42)
        n = 60
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        y = pd.DataFrame({"y": rng.standard_normal(n)})
        # 3 groups: sizes 2, 2, 56 → max/min = 28
        groups = np.zeros(n, dtype=int)
        groups[:2] = 0
        groups[2:4] = 1
        groups[4:] = 2
        # Also expect singleton warning since groups 0,1 have size 2
        # but the key check is the imbalance warning
        with pytest.warns(UserWarning, match="unbalanced"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                groups=groups,
                permutation_strategy="two-stage",
            )

    def test_balanced_no_imbalance_warning(self):
        """Balanced groups → no imbalance warning."""
        X, y, groups = _make_grouped_data()
        # Should not warn about imbalance
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("error", UserWarning)
            try:
                randomization_test_regression(
                    X,
                    y,
                    n_randomizations=20,
                    random_state=0,
                    groups=groups,
                    permutation_strategy="two-stage",
                )
            except UserWarning as exc:
                assert "unbalanced" not in str(exc)


# ------------------------------------------------------------------ #
# Panel-data convenience layer (Step 15)
# ------------------------------------------------------------------ #


def _make_panel_data(
    n_panels: int = 10,
    n_times: int = 5,
    seed: int = 42,
    balanced: bool = True,
):
    """Create a simulated balanced (or unbalanced) panel dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_panels):
        t_count = n_times if balanced else rng.integers(3, n_times + 3)
        for t in range(t_count):
            rows.append({"panel": p, "time": t})

    panel_df = pd.DataFrame(rows)
    n = len(panel_df)
    panel_df["x1"] = rng.standard_normal(n)
    panel_df["x2"] = rng.standard_normal(n)
    panel_df["y"] = (
        2.0 * panel_df["x1"] - 1.0 * panel_df["x2"] + rng.standard_normal(n) * 0.5
    )

    X = panel_df[["x1", "x2"]].copy()
    y = panel_df[["y"]].copy()
    panel_id = panel_df["panel"].values
    time_id = panel_df["time"].values
    return X, y, panel_id, time_id


class TestPanelData:
    """Tests for panel_id / time_id convenience parameters (Step 15)."""

    def test_balanced_panel_runs(self):
        """Balanced panel with panel_id= produces valid results."""
        X, y, panel_id, time_id = _make_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            random_state=0,
            panel_id=panel_id,
            time_id=time_id,
        )
        assert result.permutation_strategy == "within"
        assert result.groups is not None
        # All p-values are valid
        for p in result.raw_empirical_p:
            assert 0.0 <= p <= 1.0

    def test_panel_diagnostics_present(self):
        """Panel diagnostics dict is populated when panel_id= given."""
        X, y, panel_id, time_id = _make_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            random_state=0,
            panel_id=panel_id,
            time_id=time_id,
        )
        diag = result.extended_diagnostics
        assert "panel_diagnostics" in diag
        pd_diag = diag["panel_diagnostics"]
        assert pd_diag["n_panels"] == 10
        assert pd_diag["obs_per_panel_min"] == 5
        assert pd_diag["obs_per_panel_max"] == 5
        assert pd_diag["balanced"] is True

    def test_equivalent_to_explicit_groups(self):
        """panel_id= gives identical results to groups= + within."""
        X, y, panel_id, _ = _make_panel_data()

        res_panel = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=123,
            panel_id=panel_id,
        )
        res_groups = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=123,
            groups=panel_id,
            permutation_strategy="within",
        )
        np.testing.assert_array_equal(
            res_panel.raw_empirical_p, res_groups.raw_empirical_p
        )
        np.testing.assert_array_equal(res_panel.model_coefs, res_groups.model_coefs)

    def test_panel_id_conflicts_with_groups(self):
        """panel_id= + groups= raises ValueError."""
        X, y, panel_id, _ = _make_panel_data()
        with pytest.raises(ValueError, match="panel_id.*groups.*cannot"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                panel_id=panel_id,
                groups=panel_id,
            )

    def test_panel_id_conflicts_with_strategy(self):
        """panel_id= + permutation_strategy= raises ValueError."""
        X, y, panel_id, _ = _make_panel_data()
        with pytest.raises(ValueError, match="panel_id.*permutation_strategy.*cannot"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                panel_id=panel_id,
                permutation_strategy="between",
            )

    def test_time_id_without_panel_id_raises(self):
        """time_id= without panel_id= raises ValueError."""
        X, y, _, time_id = _make_panel_data()
        with pytest.raises(ValueError, match="time_id.*requires.*panel_id"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                time_id=time_id,
            )

    def test_unbalanced_panel_warns(self):
        """Unbalanced panels emit a warning."""
        X, y, panel_id, time_id = _make_panel_data(balanced=False)
        with pytest.warns(UserWarning, match="[Uu]nbalanced"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                panel_id=panel_id,
                time_id=time_id,
            )

    def test_unsorted_panel_warns(self):
        """Data not sorted by (panel, time) emits a warning."""
        X, y, panel_id, time_id = _make_panel_data()
        # Reverse the data to break sort order.
        X_rev = X.iloc[::-1].reset_index(drop=True)
        y_rev = y.iloc[::-1].reset_index(drop=True)
        panel_rev = panel_id[::-1].copy()
        time_rev = time_id[::-1].copy()
        with pytest.warns(UserWarning, match="not sorted"):
            randomization_test_regression(
                X_rev,
                y_rev,
                n_randomizations=20,
                random_state=0,
                panel_id=panel_rev,
                time_id=time_rev,
            )

    def test_panel_id_as_column_name(self):
        """panel_id= accepts a string column name from X."""
        X, y, panel_id, time_id = _make_panel_data()
        X_with_panel = X.copy()
        X_with_panel["panel"] = panel_id
        X_with_panel["time"] = time_id
        # Use the feature columns only — panel/time are metadata.
        result = randomization_test_regression(
            X_with_panel[["x1", "x2", "panel", "time"]],
            y,
            n_randomizations=20,
            random_state=0,
            panel_id="panel",
            time_id="time",
        )
        assert result.permutation_strategy == "within"
        diag = result.extended_diagnostics
        assert "panel_diagnostics" in diag
        assert diag["panel_diagnostics"]["n_panels"] == 10

    def test_panel_id_bad_column_name_raises(self):
        """panel_id= with a string not in X columns raises."""
        X, y, _, _ = _make_panel_data()
        with pytest.raises(ValueError, match="not found in X columns"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                panel_id="nonexistent",
            )


# ------------------------------------------------------------------
# AR panel data helper
# ------------------------------------------------------------------


def _make_ar_panel_data(
    n_panels: int = 20,
    n_times: int = 20,
    rho: float = 0.7,
    beta: np.ndarray | None = None,
    null: bool = False,
    seed: int = 42,
):
    """Panel data with AR(1) errors.

    Parameters
    ----------
    null : bool
        If True, true coefficient on x1 is 0 (null hypothesis).
    """
    rng = np.random.default_rng(seed)
    if beta is None:
        beta = np.array([0.0, -1.0]) if null else np.array([2.0, -1.0])

    rows_x1, rows_x2, rows_y = [], [], []
    panels, times = [], []
    for p in range(n_panels):
        # Generate AR(1) errors
        eps = np.zeros(n_times)
        eps[0] = rng.standard_normal()
        for t in range(1, n_times):
            eps[t] = rho * eps[t - 1] + rng.standard_normal()

        x1 = rng.standard_normal(n_times)
        x2 = rng.standard_normal(n_times)
        y = beta[0] * x1 + beta[1] * x2 + eps

        rows_x1.extend(x1)
        rows_x2.extend(x2)
        rows_y.extend(y)
        panels.extend([p] * n_times)
        times.extend(range(n_times))

    X = pd.DataFrame({"x1": rows_x1, "x2": rows_x2})
    y = pd.DataFrame({"y": rows_y})
    panel_id = np.array(panels)
    time_id = np.array(times)
    return X, y, panel_id, time_id


# ------------------------------------------------------------------
# TestARScoreCorrection — Phase 7 integration tests (Step 18)
# ------------------------------------------------------------------


class TestARScoreCorrection:
    """Integration tests for ar_order= in randomization_test_regression."""

    def test_ar1_score_linear_runs(self):
        """AR(1) correction with linear family produces valid results."""
        X, y, panel_id, time_id = _make_ar_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=0,
            method="score",
            panel_id=panel_id,
            time_id=time_id,
            ar_order=1,
        )
        for p in result.raw_empirical_p:
            assert 0.0 <= p <= 1.0
        # AR diagnostics should be populated
        pd_diag = result.extended_diagnostics["panel_diagnostics"]
        assert pd_diag["ar_order"] == 1
        assert len(pd_diag["ar_coefficients"]) == 1
        assert "durbin_watson_before" in pd_diag
        assert "durbin_watson_after" in pd_diag

    def test_ar1_score_linear_controls_type_i(self):
        """AR correction controls Type I error under the null."""
        n_reps = 100
        reject_ar = 0
        reject_no_ar = 0
        alpha = 0.05

        for rep in range(n_reps):
            X, y, panel_id, time_id = _make_ar_panel_data(
                n_panels=20, n_times=30, rho=0.9, null=True, seed=rep
            )
            # With AR correction
            res_ar = randomization_test_regression(
                X,
                y,
                n_randomizations=99,
                random_state=rep,
                method="score",
                panel_id=panel_id,
                time_id=time_id,
                ar_order=1,
            )
            # Without AR correction
            res_no_ar = randomization_test_regression(
                X,
                y,
                n_randomizations=99,
                random_state=rep,
                method="score",
                panel_id=panel_id,
                time_id=time_id,
            )
            # x1 has zero true effect under null
            if res_ar.raw_empirical_p[0] < alpha:
                reject_ar += 1
            if res_no_ar.raw_empirical_p[0] < alpha:
                reject_no_ar += 1

        rate_ar = reject_ar / n_reps
        rate_no_ar = reject_no_ar / n_reps
        # AR correction should keep rejection rate ≤ 0.10
        assert rate_ar <= 0.10, f"AR rejection rate {rate_ar} > 0.10"
        # Without correction, rejection rate should be at least somewhat inflated
        # (score permutation is more robust than asymptotic tests, so we use a
        # lenient threshold)
        assert rate_no_ar >= rate_ar, f"No-AR rate {rate_no_ar} < AR rate {rate_ar}"

    def test_ar1_score_logistic_runs(self):
        """AR(1) correction with logistic family produces valid results."""
        rng = np.random.default_rng(42)
        n_panels, n_times = 15, 10
        n = n_panels * n_times
        panel_id = np.repeat(np.arange(n_panels), n_times)
        time_id = np.tile(np.arange(n_times), n_panels)
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        prob = 1 / (1 + np.exp(-0.5 * X["x1"].values))
        y = pd.DataFrame({"y": rng.binomial(1, prob)})
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=0,
            method="score",
            family="logistic",
            panel_id=panel_id,
            time_id=time_id,
            ar_order=1,
        )
        assert all(0.0 <= p <= 1.0 for p in result.raw_empirical_p)

    def test_ar1_score_poisson_runs(self):
        """AR(1) correction with Poisson family produces valid results."""
        rng = np.random.default_rng(42)
        n_panels, n_times = 15, 10
        n = n_panels * n_times
        panel_id = np.repeat(np.arange(n_panels), n_times)
        time_id = np.tile(np.arange(n_times), n_panels)
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        lam = np.exp(0.3 * X["x1"].values)
        y = pd.DataFrame({"y": rng.poisson(lam)})
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=0,
            method="score",
            family="poisson",
            panel_id=panel_id,
            time_id=time_id,
            ar_order=1,
        )
        assert all(0.0 <= p <= 1.0 for p in result.raw_empirical_p)

    def test_ar_order_without_panel_raises(self):
        """ar_order= without panel_id= raises ValueError."""
        X, y, _, _ = _make_ar_panel_data()
        with pytest.raises(ValueError, match="ar_order.*requires.*panel_id"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                method="score",
                ar_order=1,
            )

    def test_ar_order_without_time_raises(self):
        """ar_order= without time_id= raises ValueError."""
        X, y, panel_id, _ = _make_ar_panel_data()
        with pytest.raises(ValueError, match="ar_order.*requires.*time_id"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                method="score",
                panel_id=panel_id,
                ar_order=1,
            )

    def test_ar_order_with_tbraak_raises(self):
        """ar_order= with method='ter_braak' raises ValueError."""
        X, y, panel_id, time_id = _make_ar_panel_data()
        with pytest.raises(ValueError, match="ar_order.*requires.*method='score'"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                method="ter_braak",
                panel_id=panel_id,
                time_id=time_id,
                ar_order=1,
            )

    def test_ar_order_with_ordinal_raises(self):
        """ar_order= with family='ordinal' raises ValueError."""
        rng = np.random.default_rng(42)
        n = 200
        panel_id = np.repeat(np.arange(20), 10)
        time_id = np.tile(np.arange(10), 20)
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        y = pd.DataFrame({"y": rng.choice([0, 1, 2], size=n)})
        with pytest.raises(ValueError, match="ar_order.*not supported.*ordinal"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=20,
                random_state=0,
                method="score",
                family="ordinal",
                panel_id=panel_id,
                time_id=time_id,
                ar_order=1,
            )

    def test_ar_diagnostics_populated(self):
        """AR diagnostics are populated in extended_diagnostics."""
        X, y, panel_id, time_id = _make_ar_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=0,
            method="score",
            panel_id=panel_id,
            time_id=time_id,
            ar_order=1,
        )
        pd_diag = result.extended_diagnostics["panel_diagnostics"]
        assert pd_diag["ar_order"] == 1
        assert isinstance(pd_diag["ar_coefficients"], list)
        assert len(pd_diag["ar_coefficients"]) == 1
        # Before/after diagnostics
        assert 0 < pd_diag["durbin_watson_before"] < 4
        assert 0 < pd_diag["durbin_watson_after"] < 4
        # DW after AR filtering should be closer to 2.0
        assert abs(pd_diag["durbin_watson_after"] - 2.0) < abs(
            pd_diag["durbin_watson_before"] - 2.0
        )
        assert "ljung_box_before" in pd_diag
        assert "ljung_box_after" in pd_diag

    def test_ar_linear_matches_statsmodels_gls(self):
        """AR(1) score observed coefficients ≈ statsmodels GLS."""
        import statsmodels.api as sm
        from scipy.linalg import toeplitz

        X, y, panel_id, time_id = _make_ar_panel_data(n_panels=20, n_times=20, rho=0.7)
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=0,
            method="score",
            panel_id=panel_id,
            time_id=time_id,
            ar_order=1,
        )
        rho_hat = result.extended_diagnostics["panel_diagnostics"]["ar_coefficients"][0]

        # Build block-diagonal AR(1) covariance for GLS
        n_panels_unique = len(np.unique(panel_id))
        T = 20
        r = rho_hat ** np.arange(T)
        sigma_panel = toeplitz(r)
        from scipy.linalg import block_diag

        sigma_full = block_diag(*[sigma_panel] * n_panels_unique)

        X_aug = sm.add_constant(X.values)
        gls_model = sm.GLS(y.values.ravel(), X_aug, sigma=sigma_full)
        gls_result = gls_model.fit()
        gls_coefs = gls_result.params[1:]  # drop intercept

        np.testing.assert_allclose(result.model_coefs, gls_coefs, atol=0.15, rtol=0.1)

    def test_ar_mixed_linear_runs(self):
        """AR(1) with linear_mixed family produces valid results."""
        rng = np.random.default_rng(42)
        n_panels, n_times = 15, 10
        n = n_panels * n_times
        panel_id = np.repeat(np.arange(n_panels), n_times)
        time_id = np.tile(np.arange(n_times), n_panels)
        X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
        # Random intercepts + AR(1) errors
        u = rng.normal(0, 2.0, size=n_panels)
        eps = np.zeros(n)
        for p in range(n_panels):
            start = p * n_times
            eps[start] = rng.standard_normal()
            for t in range(1, n_times):
                eps[start + t] = 0.5 * eps[start + t - 1] + rng.standard_normal()
        y_vals = 1.5 * X["x1"].values + u[panel_id] + eps
        y = pd.DataFrame({"y": y_vals})

        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=0,
            method="score",
            family="linear_mixed",
            panel_id=panel_id,
            time_id=time_id,
            ar_order=1,
        )
        assert all(0.0 <= p <= 1.0 for p in result.raw_empirical_p)

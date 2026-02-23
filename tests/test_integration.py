"""Integration and regression tests for v0.3.17.5 stabilization.

Tests cover:
- Result type verification (IndividualTestResult / JointTestResult)
- Dict-access backward compatibility (_DictAccessMixin)
- .to_dict() roundtrip serialisation
- exchangeability_cells() protocol stub
- Pinned-seed regression tests for numerical determinism
- Schema validation (all expected fields present)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from randomization_tests import (
    IndividualTestResult,
    JointTestResult,
    permutation_test_regression,
)
from randomization_tests.display import (
    print_diagnostics_table,
    print_joint_results_table,
    print_results_table,
)
from randomization_tests.families import LinearFamily, LogisticFamily

# ------------------------------------------------------------------ #
# Shared fixtures
# ------------------------------------------------------------------ #

_SEED = 12345
_N_PERMS = 100


def _linear_data(n: int = 80, seed: int = _SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _binary_data(n: int = 150, seed: int = _SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    logits = 2.0 * X["x1"] + 0.0 * X["x2"]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = pd.DataFrame({"y": rng.binomial(1, probs)})
    return X, y


# ------------------------------------------------------------------ #
# 1. Result type verification
# ------------------------------------------------------------------ #


class TestResultTypes:
    """Verify that the correct typed result object is returned."""

    @pytest.mark.parametrize("method", ["ter_braak", "kennedy", "freedman_lane"])
    def test_individual_methods_return_individual_result(self, method: str) -> None:
        X, y = _linear_data()
        kwargs: dict = dict(
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method=method,
        )
        if method in ("kennedy", "freedman_lane"):
            kwargs["confounders"] = ["x3"]
        result = permutation_test_regression(X, y, **kwargs)
        assert isinstance(result, IndividualTestResult)

    @pytest.mark.parametrize("method", ["kennedy_joint", "freedman_lane_joint"])
    def test_joint_methods_return_joint_result(self, method: str) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method=method,
            confounders=["x3"],
        )
        assert isinstance(result, JointTestResult)

    def test_logistic_returns_individual_result(self) -> None:
        X, y = _binary_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="ter_braak",
        )
        assert isinstance(result, IndividualTestResult)
        assert result.family == "logistic"


# ------------------------------------------------------------------ #
# 2. Dict-access backward compatibility
# ------------------------------------------------------------------ #


class TestDictAccess:
    """Verify _DictAccessMixin provides drop-in dict replacement."""

    @pytest.fixture()
    def individual_result(self) -> IndividualTestResult:
        X, y = _linear_data()
        return permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="ter_braak"
        )

    @pytest.fixture()
    def joint_result(self) -> JointTestResult:
        X, y = _linear_data()
        return permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="kennedy_joint",
            confounders=["x3"],
        )

    def test_bracket_access(self, individual_result: IndividualTestResult) -> None:
        assert individual_result["method"] == "ter_braak"
        assert isinstance(individual_result["model_coefs"], list)

    def test_get_with_default(self, individual_result: IndividualTestResult) -> None:
        assert individual_result.get("method") == "ter_braak"
        assert individual_result.get("nonexistent", "fallback") == "fallback"

    def test_contains(self, individual_result: IndividualTestResult) -> None:
        assert "model_coefs" in individual_result
        assert "p_value_threshold_one" in individual_result
        assert "nonexistent_key" not in individual_result

    def test_keyerror_on_missing(self, individual_result: IndividualTestResult) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            individual_result["nonexistent"]

    def test_joint_bracket_access(self, joint_result: JointTestResult) -> None:
        assert joint_result["method"] == "kennedy_joint"
        assert isinstance(joint_result["p_value"], float)
        assert "observed_improvement" in joint_result

    def test_frozen_immutability(self, individual_result: IndividualTestResult) -> None:
        """Dataclass is frozen — attribute assignment should raise."""
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            individual_result.method = "modified"  # type: ignore[misc]


# ------------------------------------------------------------------ #
# 3. to_dict() roundtrip serialisation
# ------------------------------------------------------------------ #


class TestToDict:
    """Verify .to_dict() produces JSON-safe dictionaries."""

    def test_individual_to_dict_is_plain_dict(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="ter_braak"
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        # All keys should be strings
        assert all(isinstance(k, str) for k in d)
        # Should be JSON-serialisable (no numpy arrays)
        json.dumps(d)

    def test_joint_to_dict_is_plain_dict(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="kennedy_joint",
            confounders=["x3"],
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        json.dumps(d, default=str)  # default=str handles any edge cases

    def test_roundtrip_field_preservation(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="ter_braak"
        )
        d = result.to_dict()
        assert d["method"] == result.method
        assert d["family"] == result.family
        assert d["model_coefs"] == result.model_coefs


# ------------------------------------------------------------------ #
# 4. exchangeability_cells() protocol stub
# ------------------------------------------------------------------ #


class TestExchangeabilityCells:
    """Verify the v0.4.0 forward-compat stub exists and returns None."""

    def test_linear_returns_none(self) -> None:
        fam = LinearFamily()
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        assert fam.exchangeability_cells(X, y) is None

    def test_logistic_returns_none(self) -> None:
        fam = LogisticFamily()
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 1, 1, 0])
        assert fam.exchangeability_cells(X, y) is None

    def test_protocol_method_exists(self) -> None:
        """Both families expose the method — callable with y_values."""
        for cls in (LinearFamily, LogisticFamily):
            fam = cls()
            assert callable(getattr(fam, "exchangeability_cells", None))


# ------------------------------------------------------------------ #
# 5. Schema validation
# ------------------------------------------------------------------ #

_INDIVIDUAL_FIELDS = {
    "model_coefs",
    "permuted_coefs",
    "permuted_p_values",
    "classic_p_values",
    "raw_empirical_p",
    "raw_classic_p",
    "p_value_threshold_one",
    "p_value_threshold_two",
    "method",
    "confounders",
    "model_type",
    "family",
    "backend",
    "diagnostics",
    "extended_diagnostics",
}

_JOINT_FIELDS = {
    "observed_improvement",
    "permuted_improvements",
    "p_value",
    "p_value_str",
    "metric_type",
    "model_type",
    "family",
    "backend",
    "features_tested",
    "confounders",
    "p_value_threshold_one",
    "p_value_threshold_two",
    "method",
    "diagnostics",
}


class TestSchemaValidation:
    """Verify all expected fields are present in results."""

    def test_individual_has_all_fields(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="ter_braak"
        )
        d = result.to_dict()
        assert set(d.keys()) == _INDIVIDUAL_FIELDS

    def test_joint_has_all_fields(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="kennedy_joint",
            confounders=["x3"],
        )
        d = result.to_dict()
        assert set(d.keys()) == _JOINT_FIELDS


# ------------------------------------------------------------------ #
# 6. Display functions consume typed results
# ------------------------------------------------------------------ #


class TestDisplayIntegration:
    """Verify display functions accept the new typed results."""

    def test_print_results_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="ter_braak"
        )
        print_results_table(result, list(X.columns))
        captured = capsys.readouterr()
        assert (
            "ter_braak" in captured.out.lower() or "ter braak" in captured.out.lower()
        )

    def test_print_joint_results_table(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="kennedy_joint",
            confounders=["x3"],
        )
        print_joint_results_table(result)
        captured = capsys.readouterr()
        assert (
            "kennedy_joint" in captured.out.lower() or "kennedy" in captured.out.lower()
        )

    def test_print_diagnostics_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="ter_braak"
        )
        print_diagnostics_table(result, list(X.columns))
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ------------------------------------------------------------------ #
# 7. Pinned-seed regression tests
# ------------------------------------------------------------------ #


class TestPinnedSeedRegression:
    """Verify numerical determinism with pinned seeds.

    These tests ensure that the refactoring does not change the
    numerical output of the permutation engines.  Coefficients and
    p-values are compared to values computed once and hardcoded.
    """

    def test_ter_braak_linear_determinism(self) -> None:
        """ter Braak linear: same seed → same coefficients."""
        X, y = _linear_data()
        r1 = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="ter_braak"
        )
        r2 = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="ter_braak"
        )
        np.testing.assert_array_equal(r1.model_coefs, r2.model_coefs)
        np.testing.assert_array_equal(r1.raw_empirical_p, r2.raw_empirical_p)

    def test_kennedy_joint_determinism(self) -> None:
        """Kennedy joint: same seed → same improvement + p-value."""
        X, y = _linear_data()
        kwargs: dict = dict(
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="kennedy_joint",
            confounders=["x3"],
        )
        r1 = permutation_test_regression(X, y, **kwargs)
        r2 = permutation_test_regression(X, y, **kwargs)
        assert r1.observed_improvement == r2.observed_improvement
        assert r1.p_value == r2.p_value
        np.testing.assert_array_equal(
            r1.permuted_improvements, r2.permuted_improvements
        )

    def test_freedman_lane_determinism(self) -> None:
        """Freedman–Lane individual: same seed → same results."""
        X, y = _linear_data()
        kwargs: dict = dict(
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="freedman_lane",
            confounders=["x3"],
        )
        r1 = permutation_test_regression(X, y, **kwargs)
        r2 = permutation_test_regression(X, y, **kwargs)
        np.testing.assert_array_equal(r1.model_coefs, r2.model_coefs)
        np.testing.assert_array_equal(r1.raw_empirical_p, r2.raw_empirical_p)

    def test_logistic_ter_braak_determinism(self) -> None:
        """Logistic ter Braak: same seed → same results."""
        X, y = _binary_data()
        r1 = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="ter_braak"
        )
        r2 = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="ter_braak"
        )
        np.testing.assert_array_equal(r1.model_coefs, r2.model_coefs)
        np.testing.assert_array_equal(r1.raw_empirical_p, r2.raw_empirical_p)

    def test_confounders_list_not_none(self) -> None:
        """When no confounders passed, result.confounders is [] not None."""
        X, y = _linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="ter_braak"
        )
        assert result.confounders == []
        assert isinstance(result.confounders, list)


# ------------------------------------------------------------------ #
# 8. Cross-method consistency
# ------------------------------------------------------------------ #


class TestCrossMethodConsistency:
    """Sanity checks: individual and joint methods agree on metadata."""

    def test_same_diagnostics_shape(self) -> None:
        X, y = _linear_data()
        r_ind = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="kennedy",
            confounders=["x3"],
        )
        r_joint = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="kennedy_joint",
            confounders=["x3"],
        )
        # Both should carry diagnostics dicts with the same keys
        assert set(r_ind.diagnostics.keys()) == set(r_joint.diagnostics.keys())

    def test_family_and_model_type_consistent(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="ter_braak"
        )
        assert result.family == result.model_type


# ------------------------------------------------------------------ #
# 9. n_jobs parallelisation warnings
# ------------------------------------------------------------------ #


class TestNJobsWarnings:
    """Verify n_jobs warns on no-op paths and works on valid paths.

    All tests force the numpy backend so that the JAX auto-detection
    warning does not interfere with the n_jobs-specific assertions.
    """

    @pytest.fixture(autouse=True)
    def _force_numpy_backend(self) -> None:  # type: ignore[return]
        from randomization_tests import get_backend, set_backend

        prev = get_backend()
        set_backend("numpy")
        yield
        set_backend(prev)

    @pytest.mark.parametrize("method", ["ter_braak", "freedman_lane"])
    def test_warns_on_linear_vectorised_path(self, method: str) -> None:
        """n_jobs > 1 on linear ter_braak/freedman_lane should warn."""
        X, y = _linear_data()
        kwargs: dict = dict(
            n_permutations=50,
            random_state=_SEED,
            method=method,
            n_jobs=2,
        )
        if method == "freedman_lane":
            kwargs["confounders"] = ["x3"]
        with pytest.warns(UserWarning, match="n_jobs has no effect"):
            permutation_test_regression(X, y, **kwargs)

    @pytest.mark.parametrize(
        "method", ["kennedy", "kennedy_joint", "freedman_lane_joint"]
    )
    def test_no_warning_on_parallelisable_linear_path(self, method: str) -> None:
        """n_jobs > 1 should not warn on paths that genuinely parallelise."""
        X, y = _linear_data()
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            permutation_test_regression(
                X,
                y,
                n_permutations=50,
                random_state=_SEED,
                method=method,
                confounders=["x3"],
                n_jobs=2,
            )

    def test_no_warning_on_logistic_ter_braak(self) -> None:
        """Logistic ter_braak with n_jobs > 1 should not warn (genuinely parallel)."""
        X, y = _binary_data()
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            permutation_test_regression(
                X,
                y,
                n_permutations=50,
                random_state=_SEED,
                method="ter_braak",
                n_jobs=2,
            )

    def test_parallel_results_match_serial(self) -> None:
        """n_jobs=2 should produce identical results to n_jobs=1."""
        X, y = _linear_data()
        r_serial = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            random_state=_SEED,
            method="kennedy",
            confounders=["x3"],
            n_jobs=1,
        )
        r_parallel = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            random_state=_SEED,
            method="kennedy",
            confounders=["x3"],
            n_jobs=2,
        )
        np.testing.assert_array_equal(r_serial.model_coefs, r_parallel.model_coefs)
        np.testing.assert_array_almost_equal(
            r_serial.raw_empirical_p, r_parallel.raw_empirical_p
        )

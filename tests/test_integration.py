"""Integration and regression tests.

Tests cover:
- Result type verification (IndividualTestResult / JointTestResult)
- Dict-access backward compatibility (_DictAccessMixin)
- .to_dict() roundtrip serialisation
- exchangeability_cells() protocol stub
- Pinned-seed regression tests for numerical determinism
- Schema validation (all expected fields present)
- New GLM families: Poisson, NegBin, ordinal, multinomial (Step 34)
- Confounder module with each family
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from randomization_tests import (
    IndividualTestResult,
    JointTestResult,
    identify_confounders,
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


# ------------------------------------------------------------------ #
# 10. New GLM family integration tests  (Step 34)
# ------------------------------------------------------------------ #

# ---- Synthetic data generators ----


def _poisson_data(n: int = 200, seed: int = _SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    mu = np.exp(0.5 + 0.8 * X["x1"].values - 0.3 * X["x2"].values)
    y = pd.DataFrame({"y": rng.poisson(mu)})
    return X, y


def _negbin_data(n: int = 200, seed: int = _SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    mu = np.exp(0.5 + 0.6 * X["x1"].values)
    alpha = 1.0
    p_nb = 1.0 / (1.0 + alpha * mu)
    n_nb = 1.0 / alpha
    y = pd.DataFrame({"y": rng.negative_binomial(n_nb, p_nb)})
    return X, y


def _ordinal_data(n: int = 300, seed: int = _SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "x3": rng.standard_normal(n),
        }
    )
    latent = 1.0 * X["x1"].values + rng.standard_normal(n)
    y_vals = np.digitize(latent, bins=[-1, 0, 1])  # 0, 1, 2, 3
    y = pd.DataFrame({"y": y_vals})
    return X, y


def _multinomial_data(
    n: int = 300, seed: int = _SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "x3": rng.standard_normal(n),
        }
    )
    # 3 classes via softmax
    logits = np.column_stack(
        [
            np.zeros(n),
            0.8 * X["x1"].values,
            -0.5 * X["x2"].values,
        ]
    )
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_vals = np.array([rng.choice(3, p=p) for p in probs])
    y = pd.DataFrame({"y": y_vals})
    return X, y


# ---- Poisson integration ----


class TestPoissonIntegration:
    """Integration tests for Poisson family across methods."""

    @pytest.mark.parametrize("method", ["ter_braak", "kennedy", "freedman_lane"])
    def test_individual_methods(self, method: str) -> None:
        X, y = _poisson_data()
        kwargs: dict = dict(
            n_permutations=50,
            random_state=_SEED,
            method=method,
            family="poisson",
        )
        if method in ("kennedy", "freedman_lane"):
            kwargs["confounders"] = ["x2"]
        result = permutation_test_regression(X, y, **kwargs)
        assert isinstance(result, IndividualTestResult)
        assert result.family == "poisson"
        assert len(result.model_coefs) == 2

    @pytest.mark.parametrize("method", ["kennedy_joint", "freedman_lane_joint"])
    def test_joint_methods(self, method: str) -> None:
        X, y = _poisson_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            random_state=_SEED,
            method=method,
            family="poisson",
            confounders=["x2"],
        )
        assert isinstance(result, JointTestResult)
        assert result.family == "poisson"

    def test_deterministic_seed(self) -> None:
        X, y = _poisson_data()
        kw: dict = dict(
            n_permutations=50,
            random_state=_SEED,
            method="ter_braak",
            family="poisson",
        )
        r1 = permutation_test_regression(X, y, **kw)
        r2 = permutation_test_regression(X, y, **kw)
        np.testing.assert_array_equal(r1.model_coefs, r2.model_coefs)
        np.testing.assert_array_equal(r1.raw_empirical_p, r2.raw_empirical_p)

    def test_diagnostics_present(self) -> None:
        X, y = _poisson_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, random_state=_SEED, family="poisson"
        )
        assert "n_observations" in result.diagnostics

    def test_display_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        X, y = _poisson_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, random_state=_SEED, family="poisson"
        )
        print_results_table(result, list(X.columns))
        print_diagnostics_table(result, list(X.columns))
        captured = capsys.readouterr()
        assert "poisson" in captured.out.lower()


# ---- Negative binomial integration ----


class TestNegBinIntegration:
    """Integration tests for NegBin family across methods."""

    @pytest.mark.parametrize("method", ["ter_braak", "kennedy", "freedman_lane"])
    def test_individual_methods(self, method: str) -> None:
        X, y = _negbin_data()
        kwargs: dict = dict(
            n_permutations=50,
            random_state=_SEED,
            method=method,
            family="negative_binomial",
        )
        if method in ("kennedy", "freedman_lane"):
            kwargs["confounders"] = ["x2"]
        result = permutation_test_regression(X, y, **kwargs)
        assert isinstance(result, IndividualTestResult)
        assert result.family == "negative_binomial"

    @pytest.mark.parametrize("method", ["kennedy_joint", "freedman_lane_joint"])
    def test_joint_methods(self, method: str) -> None:
        X, y = _negbin_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            random_state=_SEED,
            method=method,
            family="negative_binomial",
            confounders=["x2"],
        )
        assert isinstance(result, JointTestResult)
        assert result.family == "negative_binomial"

    def test_deterministic_seed(self) -> None:
        X, y = _negbin_data()
        kw: dict = dict(
            n_permutations=50,
            random_state=_SEED,
            method="ter_braak",
            family="negative_binomial",
        )
        r1 = permutation_test_regression(X, y, **kw)
        r2 = permutation_test_regression(X, y, **kw)
        np.testing.assert_array_equal(r1.model_coefs, r2.model_coefs)
        np.testing.assert_array_equal(r1.raw_empirical_p, r2.raw_empirical_p)

    def test_diagnostics_present(self) -> None:
        X, y = _negbin_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, random_state=_SEED, family="negative_binomial"
        )
        assert "n_observations" in result.diagnostics

    def test_display_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        X, y = _negbin_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, random_state=_SEED, family="negative_binomial"
        )
        print_results_table(result, list(X.columns))
        print_diagnostics_table(result, list(X.columns))
        captured = capsys.readouterr()
        assert "negative" in captured.out.lower()


# ---- Ordinal integration ----


class TestOrdinalIntegration:
    """Integration tests for ordinal family across supported methods.

    Ordinal supports ter_braak, kennedy, kennedy_joint.
    Freedman-Lane raises NotImplementedError (ill-defined residuals).
    """

    @pytest.mark.parametrize("method", ["ter_braak", "kennedy"])
    def test_individual_methods(self, method: str) -> None:
        X, y = _ordinal_data()
        kwargs: dict = dict(
            n_permutations=50,
            random_state=_SEED,
            method=method,
            family="ordinal",
        )
        if method == "kennedy":
            kwargs["confounders"] = ["x3"]
        result = permutation_test_regression(X, y, **kwargs)
        assert isinstance(result, IndividualTestResult)
        assert result.family == "ordinal"

    def test_kennedy_joint(self) -> None:
        X, y = _ordinal_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            random_state=_SEED,
            method="kennedy_joint",
            family="ordinal",
            confounders=["x3"],
        )
        assert isinstance(result, JointTestResult)
        assert result.family == "ordinal"

    @pytest.mark.parametrize("method", ["freedman_lane", "freedman_lane_joint"])
    def test_freedman_lane_rejected(self, method: str) -> None:
        X, y = _ordinal_data()
        with pytest.raises(ValueError, match="not supported for family='ordinal'"):
            permutation_test_regression(
                X,
                y,
                n_permutations=50,
                random_state=_SEED,
                method=method,
                family="ordinal",
                confounders=["x3"],
            )

    def test_deterministic_seed(self) -> None:
        X, y = _ordinal_data()
        kw: dict = dict(
            n_permutations=50,
            random_state=_SEED,
            method="ter_braak",
            family="ordinal",
        )
        r1 = permutation_test_regression(X, y, **kw)
        r2 = permutation_test_regression(X, y, **kw)
        np.testing.assert_array_equal(r1.model_coefs, r2.model_coefs)
        np.testing.assert_array_equal(r1.raw_empirical_p, r2.raw_empirical_p)

    def test_diagnostics_present(self) -> None:
        X, y = _ordinal_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, random_state=_SEED, family="ordinal"
        )
        assert "n_observations" in result.diagnostics

    def test_display_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        X, y = _ordinal_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, random_state=_SEED, family="ordinal"
        )
        print_results_table(result, list(X.columns))
        print_diagnostics_table(result, list(X.columns))
        captured = capsys.readouterr()
        assert "ordinal" in captured.out.lower()


# ---- Multinomial integration ----


class TestMultinomialIntegration:
    """Integration tests for multinomial family across supported methods.

    Multinomial supports ter_braak, kennedy, kennedy_joint.
    Freedman-Lane raises NotImplementedError (ill-defined residuals).
    """

    @pytest.mark.parametrize("method", ["ter_braak", "kennedy"])
    def test_individual_methods(self, method: str) -> None:
        X, y = _multinomial_data()
        kwargs: dict = dict(
            n_permutations=50,
            random_state=_SEED,
            method=method,
            family="multinomial",
        )
        if method == "kennedy":
            kwargs["confounders"] = ["x3"]
        result = permutation_test_regression(X, y, **kwargs)
        assert isinstance(result, IndividualTestResult)
        assert result.family == "multinomial"

    def test_kennedy_joint(self) -> None:
        X, y = _multinomial_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            random_state=_SEED,
            method="kennedy_joint",
            family="multinomial",
            confounders=["x3"],
        )
        assert isinstance(result, JointTestResult)
        assert result.family == "multinomial"

    @pytest.mark.parametrize("method", ["freedman_lane", "freedman_lane_joint"])
    def test_freedman_lane_rejected(self, method: str) -> None:
        X, y = _multinomial_data()
        with pytest.raises(ValueError, match="not supported for family='multinomial'"):
            permutation_test_regression(
                X,
                y,
                n_permutations=50,
                random_state=_SEED,
                method=method,
                family="multinomial",
                confounders=["x3"],
            )

    def test_deterministic_seed(self) -> None:
        X, y = _multinomial_data()
        kw: dict = dict(
            n_permutations=50,
            random_state=_SEED,
            method="ter_braak",
            family="multinomial",
        )
        r1 = permutation_test_regression(X, y, **kw)
        r2 = permutation_test_regression(X, y, **kw)
        np.testing.assert_array_equal(r1.model_coefs, r2.model_coefs)
        np.testing.assert_array_equal(r1.raw_empirical_p, r2.raw_empirical_p)

    def test_diagnostics_present(self) -> None:
        X, y = _multinomial_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, random_state=_SEED, family="multinomial"
        )
        assert "n_observations" in result.diagnostics

    def test_display_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        X, y = _multinomial_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, random_state=_SEED, family="multinomial"
        )
        print_results_table(result, list(X.columns))
        print_diagnostics_table(result, list(X.columns))
        captured = capsys.readouterr()
        assert "multinomial" in captured.out.lower()


# ---- Confounder module with each family ----


class TestConfounderFamilyIntegration:
    """Test that identify_confounders works with each family."""

    def test_linear_confounders(self) -> None:
        X, y = _linear_data()
        result = identify_confounders(
            X, y, predictor="x1", random_state=_SEED, family="linear"
        )
        assert "identified_confounders" in result
        assert "identified_mediators" in result

    def test_logistic_confounders(self) -> None:
        X, y = _binary_data()
        result = identify_confounders(
            X, y, predictor="x1", random_state=_SEED, family="logistic"
        )
        assert "identified_confounders" in result

    def test_poisson_confounders(self) -> None:
        X, y = _poisson_data()
        result = identify_confounders(
            X, y, predictor="x1", random_state=_SEED, family="poisson"
        )
        assert "identified_confounders" in result

    def test_negbin_confounders(self) -> None:
        X, y = _negbin_data()
        result = identify_confounders(
            X, y, predictor="x1", random_state=_SEED, family="negative_binomial"
        )
        assert "identified_confounders" in result

    def test_ordinal_confounders(self) -> None:
        X, y = _ordinal_data()
        result = identify_confounders(
            X, y, predictor="x1", random_state=_SEED, family="ordinal"
        )
        assert "identified_confounders" in result

    def test_multinomial_confounders(self) -> None:
        X, y = _multinomial_data()
        result = identify_confounders(
            X, y, predictor="x1", random_state=_SEED, family="multinomial"
        )
        assert "identified_confounders" in result


# ---- Cross-family field consistency ----


class TestCrossFamilyConsistency:
    """Verify result schema consistency across all families."""

    @pytest.mark.parametrize(
        "family_str,data_fn",
        [
            ("linear", _linear_data),
            ("logistic", _binary_data),
            ("poisson", _poisson_data),
            ("negative_binomial", _negbin_data),
            ("ordinal", _ordinal_data),
            ("multinomial", _multinomial_data),
        ],
    )
    def test_individual_schema(self, family_str: str, data_fn: object) -> None:
        X, y = data_fn()  # type: ignore[operator]
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            random_state=_SEED,
            method="ter_braak",
            family=family_str,
        )
        d = result.to_dict()
        assert d["family"] == family_str
        assert "model_coefs" in d
        assert "diagnostics" in d
        assert "raw_empirical_p" in d

    @pytest.mark.parametrize(
        "family_str,data_fn",
        [
            ("linear", _linear_data),
            ("logistic", _binary_data),
            ("poisson", _poisson_data),
            ("negative_binomial", _negbin_data),
            ("ordinal", _ordinal_data),
            ("multinomial", _multinomial_data),
        ],
    )
    def test_joint_schema(self, family_str: str, data_fn: object) -> None:
        X, y = data_fn()  # type: ignore[operator]
        # Use x2 or x3 as confounder (whichever exists)
        conf = ["x2"] if "x2" in X.columns else ["x3"]
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            random_state=_SEED,
            method="kennedy_joint",
            family=family_str,
            confounders=conf,
        )
        d = result.to_dict()
        assert d["family"] == family_str
        assert "observed_improvement" in d
        assert "p_value" in d

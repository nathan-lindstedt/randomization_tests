"""Tests for src/randomization_tests/_results.py.

Covers:
- _DictAccessMixin API: __getitem__, get, __contains__, to_dict()
- _DictAccessMixin invariants: empty base _SERIALIZERS, fields()-based __contains__
- IndividualTestResult and JointTestResult _SERIALIZERS overrides
- KernelTestResult, ConformalResult, InvarianceResult, KnockoffResult:
  frozen enforcement, field access, dict-access mixin, to_dict() serialisation
- _numpy_to_python tuple/list behaviour and the ConformalResult serialiser
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import numpy as np
import pytest

from randomization_tests._results import (
    ConformalEvaluationResult,
    ConformalResult,
    FitResult,
    IndividualTestResult,
    InvarianceResult,
    JointTestResult,
    KernelTestResult,
    KnockoffResult,
    _DictAccessMixin,
    _numpy_to_python,
)
from randomization_tests.families import LinearFamily

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _kernel_result(**overrides) -> KernelTestResult:
    defaults = dict(
        statistic=0.42,
        p_value=0.03,
        null_distribution=np.array([0.1, 0.2, 0.3]),
        kernel_name="gaussian",
        n_randomizations=3,
        method="mmd",
    )
    defaults.update(overrides)
    return KernelTestResult(**defaults)


def _conformal_result_regression(**overrides) -> ConformalResult:
    defaults = dict(
        prediction_interval=(1.0, 3.0),
        prediction_set=None,
        nonconformity_scores=np.array([0.5, 0.6, 0.7]),
        confidence_level=0.95,
        method="split",
    )
    defaults.update(overrides)
    return ConformalResult(**defaults)


def _conformal_result_classification(**overrides) -> ConformalResult:
    defaults = dict(
        prediction_interval=None,
        prediction_set=np.array([0, 1]),
        nonconformity_scores=np.array([0.3, 0.4]),
        confidence_level=0.90,
        method="split",
    )
    defaults.update(overrides)
    return ConformalResult(**defaults)


def _conformal_eval_result(**overrides) -> ConformalEvaluationResult:
    defaults = dict(
        empirical_coverage=0.95,
        nominal_coverage=0.95,
        n_eval=50,
        covered=np.ones(50, dtype=bool),
        mean_interval_width=3.5,
    )
    defaults.update(overrides)
    return ConformalEvaluationResult(**defaults)


def _invariance_result(**overrides) -> InvarianceResult:
    defaults = dict(
        is_invariant=True,
        p_value=0.42,
        environment_coefficients={"env_0": 1.1, "env_1": 0.9},
        pooled_coefficient=1.0,
        max_deviation=0.2,
        null_distribution=np.array([0.1, 0.3, 0.5]),
    )
    defaults.update(overrides)
    return InvarianceResult(**defaults)


def _knockoff_result(**overrides) -> KnockoffResult:
    defaults = dict(
        selected_features=["x1", "x3"],
        knockoff_statistics=np.array([1.2, -0.3, 0.8]),
        fdr_level=0.1,
        threshold=0.5,
        feature_names=["x1", "x2", "x3"],
    )
    defaults.update(overrides)
    return KnockoffResult(**defaults)


def _fit_result(**overrides) -> FitResult:
    defaults = dict(
        params=np.array([1.5, 2.0, -1.0]),
        predictions=np.array([1.0, 2.0]),
        fitted_values=np.array([1.0, 2.0]),
        residuals=np.array([0.1, -0.1]),
        log_likelihood=-12.3,
        fisher_info=np.eye(3),
        se=np.array([0.1, 0.2, 0.3]),
        p_values=np.array([0.001, 0.01, 0.05]),
        fit_intercept=True,
        converged=True,
        df=47.0,
    )
    defaults.update(overrides)
    return FitResult(**defaults)


def _individual_result(**overrides) -> IndividualTestResult:
    defaults = dict(
        model_coefs=[1.0, 2.0],
        permuted_coefs=[[1.1, 1.9], [0.9, 2.1]],
        permuted_p_values=["0.01 (*)", "0.50 (ns)"],
        classic_p_values=["0.01 (*)", "0.45 (ns)"],
        raw_empirical_p=np.array([0.01, 0.50]),
        raw_classic_p=np.array([0.01, 0.45]),
        p_value_threshold_one=0.05,
        p_value_threshold_two=0.01,
        p_value_threshold_three=0.001,
        method="ter_braak",
        confounders=[],
        family=LinearFamily(),
        backend="numpy",
        feature_names=["x1", "x2"],
        target_name="y",
        n_randomizations=2,
        groups=None,
        permutation_strategy=None,
        diagnostics={"r_squared": 0.8},
        extended_diagnostics={},
        confidence_intervals={},
        guarantee_tier="asymptotically_exact",
    )
    defaults.update(overrides)
    return IndividualTestResult(**defaults)


def _joint_result(**overrides) -> JointTestResult:
    defaults = dict(
        observed_improvement=5.2,
        permuted_improvements=[4.1, 3.8],
        p_value=0.02,
        p_value_str="0.02 (*)",
        metric_type="RSS Reduction",
        family=LinearFamily(),
        backend="numpy",
        features_tested=["x1", "x2"],
        confounders=[],
        feature_names=["x1", "x2"],
        target_name="y",
        n_randomizations=2,
        groups=None,
        permutation_strategy=None,
        p_value_threshold_one=0.05,
        p_value_threshold_two=0.01,
        p_value_threshold_three=0.001,
        method="kennedy_joint",
        diagnostics={"r_squared": 0.8},
        extended_diagnostics={},
        guarantee_tier="asymptotically_exact",
    )
    defaults.update(overrides)
    return JointTestResult(**defaults)


# ------------------------------------------------------------------ #
# _DictAccessMixin fixes
# ------------------------------------------------------------------ #


class TestDictAccessMixinFixes:
    """Verify the dict-access mixin invariants."""

    def test_mixin_serializers_is_empty(self):
        """Base mixin _SERIALIZERS must be {} — no domain coupling."""
        assert _DictAccessMixin._SERIALIZERS == {}

    def test_contains_returns_false_for_methods(self):
        """to_dict, get, etc. are NOT data fields — 'in' must be False."""
        r = _kernel_result()
        assert "to_dict" not in r
        assert "get" not in r
        assert "__contains__" not in r
        assert "_SERIALIZERS" not in r

    def test_contains_returns_true_for_data_fields(self):
        r = _kernel_result()
        assert "statistic" in r
        assert "p_value" in r
        assert "method" in r

    def test_contains_returns_false_for_nonexistent_key(self):
        r = _kernel_result()
        assert "nonexistent" not in r

    def test_contains_returns_false_for_non_string(self):
        r = _kernel_result()
        assert 42 not in r  # type: ignore[operator]
        assert None not in r  # type: ignore[operator]


# ------------------------------------------------------------------ #
# Parameterized Result Dataclass Invariant Contracts
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "factory",
    [
        _kernel_result,
        _conformal_result_regression,
        _conformal_result_classification,
        _conformal_eval_result,
        _invariance_result,
        _knockoff_result,
        _fit_result,
        _individual_result,
        _joint_result,
    ],
)
class TestResultDataclassContracts:
    """Consolidated parameterized contract invariants across all result types."""

    def test_is_frozen(self, factory):
        r = factory()
        first_field = next(iter(fields(r))).name
        with pytest.raises(FrozenInstanceError):
            setattr(r, first_field, "mutated_val")

    def test_contains_data_fields_only(self, factory):
        r = factory()
        for f in fields(r):
            assert f.name in r
        assert "to_dict" not in r
        assert "get" not in r
        assert "nonexistent_key_xyz" not in r
        assert 9999 not in r  # type: ignore[operator]
        assert None not in r  # type: ignore[operator]

    def test_getitem_and_get(self, factory):
        r = factory()
        first_field = next(iter(fields(r))).name
        val_item = r[first_field]
        val_attr = getattr(r, first_field)
        if isinstance(val_item, np.ndarray):
            assert np.array_equal(val_item, val_attr)
            assert np.array_equal(r.get(first_field), val_attr)
        else:
            assert val_item == val_attr
            assert r.get(first_field) == val_attr
        assert r.get("nonexistent_key_xyz") is None
        assert r.get("nonexistent_key_xyz", "fallback") == "fallback"
        with pytest.raises(KeyError):
            _ = r["nonexistent_key_xyz"]

    def test_to_dict_plain_and_json_serialisable(self, factory):
        r = factory()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "context" not in d
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ------------------------------------------------------------------ #
# Domain-specific tests for FitResult
# ------------------------------------------------------------------ #


class TestFitResultDomain:
    """Domain-specific invariants for the unified FitResult container."""

    def test_conf_int_with_df_exact_student_t(self):
        """FitResult with df uses exact Student's t critical values."""
        from scipy.stats import t as t_dist

        r = _fit_result(
            df=10.0, se=np.array([1.0, 1.0, 1.0]), params=np.array([0.0, 0.0, 0.0])
        )
        ci = r.conf_int(alpha=0.05)
        t_crit = float(t_dist.ppf(0.975, df=10.0))
        np.testing.assert_allclose(ci[:, 0], -t_crit, atol=1e-12)
        np.testing.assert_allclose(ci[:, 1], t_crit, atol=1e-12)

    def test_conf_int_without_df_asymptotic_normal(self):
        """FitResult without df falls back to asymptotic standard normal."""
        from scipy.stats import norm

        r = _fit_result(
            df=None, se=np.array([1.0, 1.0, 1.0]), params=np.array([0.0, 0.0, 0.0])
        )
        ci = r.conf_int(alpha=0.05)
        z_crit = float(norm.ppf(0.975))
        np.testing.assert_allclose(ci[:, 0], -z_crit, atol=1e-12)
        np.testing.assert_allclose(ci[:, 1], z_crit, atol=1e-12)

    def test_coef_and_intercept_properties(self):
        """coef_ and intercept_ provide sklearn estimator duck typing."""
        r_intercept = _fit_result(params=np.array([5.0, 1.2, -0.8]), fit_intercept=True)
        assert r_intercept.intercept_ == 5.0
        np.testing.assert_allclose(r_intercept.coef_, [1.2, -0.8])

        r_no_intercept = _fit_result(params=np.array([1.2, -0.8]), fit_intercept=False)
        assert r_no_intercept.intercept_ == 0.0
        np.testing.assert_allclose(r_no_intercept.coef_, [1.2, -0.8])


# ------------------------------------------------------------------ #
# Domain-specific tests for KernelTestResult
# ------------------------------------------------------------------ #


class TestKernelTestResultDomain:
    def test_field_access(self):
        r = _kernel_result(statistic=1.23, method="hsic")
        assert r.statistic == 1.23
        assert r.method == "hsic"

    def test_empty_serializers(self):
        assert KernelTestResult._SERIALIZERS == {}

    def test_differing_scalar_produces_unequal_dicts(self):
        r1 = _kernel_result(statistic=1.0)
        r2 = _kernel_result(statistic=2.0)
        assert r1.to_dict() != r2.to_dict()


# ------------------------------------------------------------------ #
# Domain-specific tests for ConformalResult
# ------------------------------------------------------------------ #


class TestConformalResultDomain:
    def test_regression_fields(self):
        r = _conformal_result_regression()
        assert r.prediction_interval == (1.0, 3.0)
        assert r.prediction_set is None
        assert r.confidence_level == 0.95

    def test_classification_fields(self):
        r = _conformal_result_classification()
        assert r.prediction_interval is None
        assert isinstance(r.prediction_set, np.ndarray)

    def test_to_dict_prediction_interval_is_list(self):
        r = _conformal_result_regression()
        d = r.to_dict()
        assert isinstance(d["prediction_interval"], list)
        assert d["prediction_interval"] == [1.0, 3.0]

    def test_to_dict_none_stays_none(self):
        r_class = _conformal_result_classification()
        assert r_class.to_dict()["prediction_interval"] is None
        r_reg = _conformal_result_regression()
        assert r_reg.to_dict()["prediction_set"] is None

    def test_prediction_interval_serializer_registered(self):
        assert "prediction_interval" in ConformalResult._SERIALIZERS
        assert "prediction_set" in ConformalResult._SERIALIZERS
        assert "intervals" in ConformalResult._SERIALIZERS
        assert "prediction_sets" in ConformalResult._SERIALIZERS

    def test_intervals_and_prediction_sets_serialization(self):
        r = _conformal_result_regression(
            intervals=np.array([[1.0, 2.0], [3.0, 4.0]]),
            prediction_sets=[np.array([0, 1]), np.array([1])],
            point_predictions=np.array([1.5, 3.5]),
            probabilities=np.array([[0.8, 0.2], [0.1, 0.9]]),
        )
        d = r.to_dict()
        assert d["intervals"] == [[1.0, 2.0], [3.0, 4.0]]
        assert d["prediction_sets"] == [[0, 1], [1]]
        assert d["point_predictions"] == [1.5, 3.5]
        assert d["probabilities"] == [[0.8, 0.2], [0.1, 0.9]]


# ------------------------------------------------------------------ #
# Domain-specific tests for ConformalEvaluationResult
# ------------------------------------------------------------------ #


class TestConformalEvaluationResultDomain:
    def test_to_dict_and_json(self):
        r = ConformalEvaluationResult(
            empirical_coverage=0.94,
            nominal_coverage=0.95,
            n_eval=50,
            covered=np.ones(50, dtype=bool),
            mean_interval_width=3.5,
            median_interval_width=3.2,
            set_size_distribution={1: 45, 2: 5},
            singleton_rate=0.90,
            ambiguous_rate=0.10,
            empty_rate=0.0,
        )
        d = r.to_dict()
        assert isinstance(d["covered"], list)
        assert d["mean_interval_width"] == 3.5
        assert d["singleton_rate"] == 0.90
        json_str = json.dumps(d)
        assert "empirical_coverage" in json_str


# ------------------------------------------------------------------ #
# Domain-specific tests for InvarianceResult
# ------------------------------------------------------------------ #


class TestInvarianceResultDomain:
    def test_field_access(self):
        r = _invariance_result(is_invariant=False, p_value=0.01)
        assert r.is_invariant is False
        assert r.p_value == 0.01

    def test_to_dict_is_invariant_is_bool(self):
        d = _invariance_result(is_invariant=True).to_dict()
        assert isinstance(d["is_invariant"], (bool, int))

    def test_getitem_environment_coefficients(self):
        r = _invariance_result()
        ec = r["environment_coefficients"]
        assert isinstance(ec, dict)

    def test_serializers_registered(self):
        assert "null_distribution" in InvarianceResult._SERIALIZERS


# ------------------------------------------------------------------ #
# Domain-specific tests for KnockoffResult
# ------------------------------------------------------------------ #


class TestKnockoffResultDomain:
    def test_field_access(self):
        r = _knockoff_result()
        assert r.fdr_level == 0.1
        assert r.threshold == 0.5
        assert r.selected_features == ["x1", "x3"]
        assert r.method == "equicorrelated"
        assert r.guarantee_tier == "model_x_approximate"
        assert isinstance(r.diagnostics, dict)

    def test_empty_serializers(self):
        assert KnockoffResult._SERIALIZERS == {}


# ------------------------------------------------------------------ #
# _numpy_to_python tuple handling (used by ConformalResult serialiser)
# ------------------------------------------------------------------ #


class TestNumpyToPython:
    def test_tuple_preserved_as_tuple(self):
        """_numpy_to_python converts tuple contents but keeps type."""
        result = _numpy_to_python((np.float64(1.0), np.float64(3.0)))
        assert isinstance(result, tuple)
        assert result == (1.0, 3.0)

    def test_list_preserved_as_list(self):
        result = _numpy_to_python([np.float64(1.0)])
        assert isinstance(result, list)

    def test_conformal_serializer_converts_tuple_to_list(self):
        """The ConformalResult serializer explicitly converts to list."""
        serializer = ConformalResult._SERIALIZERS["prediction_interval"]
        assert serializer((1.0, 3.0)) == [1.0, 3.0]
        assert serializer(None) is None


# ------------------------------------------------------------------ #
# IndividualTestResult / JointTestResult serializer contracts
# ------------------------------------------------------------------ #


class TestIndividualResultSerializers:
    """Unit tests for IndividualTestResult._SERIALIZERS and _EXCLUDE_FROM_DICT.

    Uses SimpleNamespace to mock ModelFamily — no pipeline run needed.
    """

    def test_family_serializer_defined(self):
        assert "family" in IndividualTestResult._SERIALIZERS

    def test_family_serializer_calls_name(self):
        mock_family = SimpleNamespace(name="linear")
        result = IndividualTestResult._SERIALIZERS["family"](mock_family)
        assert result == "linear"

    def test_context_excluded_from_to_dict(self):
        assert "context" in IndividualTestResult._EXCLUDE_FROM_DICT

    def test_mixin_serializers_unchanged(self):
        """IndividualTestResult overrides _SERIALIZERS; mixin default stays empty."""
        assert _DictAccessMixin._SERIALIZERS == {}
        assert "family" in IndividualTestResult._SERIALIZERS


class TestJointResultSerializers:
    """Unit tests for JointTestResult._SERIALIZERS and _EXCLUDE_FROM_DICT."""

    def test_family_serializer_defined(self):
        assert "family" in JointTestResult._SERIALIZERS

    def test_family_serializer_calls_name(self):
        mock_family = SimpleNamespace(name="logistic")
        result = JointTestResult._SERIALIZERS["family"](mock_family)
        assert result == "logistic"

    def test_context_excluded_from_to_dict(self):
        assert "context" in JointTestResult._EXCLUDE_FROM_DICT

    def test_serializers_independent_of_individual(self):
        """Each class owns its own _SERIALIZERS dict — not a shared reference."""
        assert IndividualTestResult._SERIALIZERS is not JointTestResult._SERIALIZERS

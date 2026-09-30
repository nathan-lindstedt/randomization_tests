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
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from randomization_tests._results import (
    ConformalEvaluationResult,
    ConformalResult,
    IndividualTestResult,
    InvarianceResult,
    JointTestResult,
    KernelTestResult,
    KnockoffResult,
    _DictAccessMixin,
    _numpy_to_python,
)

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
# KernelTestResult
# ------------------------------------------------------------------ #


class TestKernelTestResult:
    def test_is_frozen(self):
        r = _kernel_result()
        with pytest.raises(FrozenInstanceError):
            r.statistic = 99.0  # type: ignore[misc]

    def test_field_access(self):
        r = _kernel_result(statistic=1.23, method="hsic")
        assert r.statistic == 1.23
        assert r.method == "hsic"

    def test_getitem(self):
        r = _kernel_result()
        assert r["statistic"] == r.statistic
        assert r["kernel_name"] == "gaussian"

    def test_getitem_missing_raises_key_error(self):
        r = _kernel_result()
        with pytest.raises(KeyError):
            _ = r["nonexistent"]

    def test_get_with_default(self):
        r = _kernel_result()
        assert r.get("statistic") == r.statistic
        assert r.get("nonexistent", "fallback") == "fallback"
        assert r.get("nonexistent") is None

    def test_to_dict_is_plain_dict(self):
        r = _kernel_result()
        d = r.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_json_serialisable(self):
        r = _kernel_result()
        json.dumps(r.to_dict())

    def test_to_dict_null_distribution_is_list(self):
        r = _kernel_result()
        d = r.to_dict()
        assert isinstance(d["null_distribution"], list)

    def test_to_dict_preserves_scalar_fields(self):
        r = _kernel_result(statistic=0.5, p_value=0.02, n_randomizations=100)
        d = r.to_dict()
        assert d["statistic"] == 0.5
        assert d["p_value"] == 0.02
        assert d["n_randomizations"] == 100

    def test_to_dict_has_all_fields(self):
        r = _kernel_result()
        d = r.to_dict()
        for fname in (
            "statistic",
            "p_value",
            "null_distribution",
            "kernel_name",
            "n_randomizations",
            "method",
        ):
            assert fname in d

    def test_empty_serializers(self):
        assert KernelTestResult._SERIALIZERS == {}

    def test_identical_instances_have_equal_dicts(self):
        """Two identically-constructed instances serialise identically."""
        assert _kernel_result().to_dict() == _kernel_result().to_dict()

    def test_differing_scalar_produces_unequal_dicts(self):
        r1 = _kernel_result(statistic=1.0)
        r2 = _kernel_result(statistic=2.0)
        assert r1.to_dict() != r2.to_dict()


# ------------------------------------------------------------------ #
# ConformalResult
# ------------------------------------------------------------------ #


class TestConformalResult:
    def test_is_frozen(self):
        r = _conformal_result_regression()
        with pytest.raises(FrozenInstanceError):
            r.confidence_level = 0.5  # type: ignore[misc]

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
        """tuple[float, float] must serialize to list, not tuple."""
        r = _conformal_result_regression()
        d = r.to_dict()
        assert isinstance(d["prediction_interval"], list)
        assert d["prediction_interval"] == [1.0, 3.0]

    def test_to_dict_none_prediction_interval_stays_none(self):
        r = _conformal_result_classification()
        d = r.to_dict()
        assert d["prediction_interval"] is None

    def test_to_dict_none_prediction_set_stays_none(self):
        r = _conformal_result_regression()
        d = r.to_dict()
        assert d["prediction_set"] is None

    def test_to_dict_prediction_set_is_list(self):
        r = _conformal_result_classification()
        d = r.to_dict()
        assert isinstance(d["prediction_set"], list)

    def test_to_dict_nonconformity_scores_is_list(self):
        r = _conformal_result_regression()
        d = r.to_dict()
        assert isinstance(d["nonconformity_scores"], list)

    def test_to_dict_json_serialisable_regression(self):
        json.dumps(_conformal_result_regression().to_dict())

    def test_to_dict_json_serialisable_classification(self):
        json.dumps(_conformal_result_classification().to_dict())

    def test_contains_fields(self):
        r = _conformal_result_regression()
        assert "prediction_interval" in r
        assert "confidence_level" in r
        assert "method" in r
        assert "coverage_guarantee" in r
        assert "groups" in r
        assert "diagnostics" in r

    def test_getitem(self):
        r = _conformal_result_regression()
        assert r["method"] == "split"
        assert r["confidence_level"] == 0.95
        assert r["coverage_guarantee"] == "finite_sample_exact"

    def test_prediction_interval_serializer_registered(self):
        assert "prediction_interval" in ConformalResult._SERIALIZERS
        assert "prediction_set" in ConformalResult._SERIALIZERS
        assert "intervals" in ConformalResult._SERIALIZERS
        assert "prediction_sets" in ConformalResult._SERIALIZERS
        assert "point_predictions" in ConformalResult._SERIALIZERS
        assert "probabilities" in ConformalResult._SERIALIZERS
        assert "nonconformity_scores" in ConformalResult._SERIALIZERS
        assert "groups" in ConformalResult._SERIALIZERS

    def test_intervals_and_prediction_sets_serialization(self):
        r = _conformal_result_regression(
            intervals=np.array([[1.0, 2.0], [3.0, 4.0]]),
            prediction_sets=[np.array([0, 1]), np.array([1])],
            point_predictions=np.array([1.5, 3.5]),
            probabilities=np.array([[0.8, 0.2], [0.1, 0.9]]),
        )
        d = r.to_dict()
        assert isinstance(d["intervals"], list)
        assert d["intervals"] == [[1.0, 2.0], [3.0, 4.0]]
        assert isinstance(d["prediction_sets"], list)
        assert d["prediction_sets"] == [[0, 1], [1]]
        assert isinstance(d["point_predictions"], list)
        assert d["point_predictions"] == [1.5, 3.5]
        assert isinstance(d["probabilities"], list)
        assert d["probabilities"] == [[0.8, 0.2], [0.1, 0.9]]
        json.dumps(d)


# ------------------------------------------------------------------ #
# ConformalEvaluationResult
# ------------------------------------------------------------------ #


class TestConformalEvaluationResult:
    def test_is_frozen(self):
        r = ConformalEvaluationResult(
            empirical_coverage=0.96,
            nominal_coverage=0.95,
            n_eval=100,
            covered=np.ones(100, dtype=bool),
            mean_interval_width=5.2,
        )
        with pytest.raises(FrozenInstanceError):
            r.empirical_coverage = 0.90  # type: ignore[misc]

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

    def test_dict_access_and_contains(self):
        r = ConformalEvaluationResult(
            empirical_coverage=0.95,
            nominal_coverage=0.95,
            n_eval=20,
            covered=np.ones(20, dtype=bool),
        )
        assert r["empirical_coverage"] == 0.95
        assert "nominal_coverage" in r
        assert "n_eval" in r


# ------------------------------------------------------------------ #
# InvarianceResult
# ------------------------------------------------------------------ #


class TestInvarianceResult:
    def test_is_frozen(self):
        r = _invariance_result()
        with pytest.raises(FrozenInstanceError):
            r.p_value = 0.99  # type: ignore[misc]

    def test_field_access(self):
        r = _invariance_result(is_invariant=False, p_value=0.01)
        assert r.is_invariant is False
        assert r.p_value == 0.01

    def test_to_dict_is_plain_dict(self):
        d = _invariance_result().to_dict()
        assert isinstance(d, dict)

    def test_to_dict_json_serialisable(self):
        json.dumps(_invariance_result().to_dict())

    def test_to_dict_null_distribution_is_list(self):
        d = _invariance_result().to_dict()
        assert isinstance(d["null_distribution"], list)

    def test_to_dict_is_invariant_is_bool(self):
        d = _invariance_result(is_invariant=True).to_dict()
        # Python bool, not numpy bool_
        assert isinstance(d["is_invariant"], (bool, int))

    def test_to_dict_has_all_fields(self):
        d = _invariance_result().to_dict()
        for fname in (
            "is_invariant",
            "p_value",
            "environment_coefficients",
            "pooled_coefficient",
            "max_deviation",
            "null_distribution",
            "tier",
            "guarantee_tier",
            "method",
            "diagnostics",
        ):
            assert fname in d

    def test_contains_fields(self):
        r = _invariance_result()
        assert "is_invariant" in r
        assert "max_deviation" in r
        assert "tier" in r
        assert "guarantee_tier" in r
        assert "method" in r

    def test_getitem_environment_coefficients(self):
        r = _invariance_result()
        ec = r["environment_coefficients"]
        assert isinstance(ec, dict)

    def test_serializers_registered(self):
        assert "null_distribution" in InvarianceResult._SERIALIZERS


# ------------------------------------------------------------------ #
# KnockoffResult
# ------------------------------------------------------------------ #


class TestKnockoffResult:
    def test_is_frozen(self):
        r = _knockoff_result()
        with pytest.raises(FrozenInstanceError):
            r.threshold = 99.0  # type: ignore[misc]

    def test_field_access(self):
        r = _knockoff_result()
        assert r.fdr_level == 0.1
        assert r.threshold == 0.5
        assert r.selected_features == ["x1", "x3"]
        assert r.method == "equicorrelated"
        assert r.guarantee_tier == "model_x_approximate"
        assert isinstance(r.diagnostics, dict)

    def test_to_dict_is_plain_dict(self):
        d = _knockoff_result().to_dict()
        assert isinstance(d, dict)

    def test_to_dict_json_serialisable(self):
        json.dumps(_knockoff_result().to_dict())

    def test_to_dict_knockoff_statistics_is_list(self):
        d = _knockoff_result().to_dict()
        assert isinstance(d["knockoff_statistics"], list)

    def test_to_dict_has_all_fields(self):
        d = _knockoff_result().to_dict()
        for fname in (
            "selected_features",
            "knockoff_statistics",
            "fdr_level",
            "threshold",
            "feature_names",
            "method",
            "guarantee_tier",
            "diagnostics",
        ):
            assert fname in d

    def test_contains_fields(self):
        r = _knockoff_result()
        assert "selected_features" in r
        assert "feature_names" in r

    def test_getitem(self):
        r = _knockoff_result()
        assert r["fdr_level"] == 0.1

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

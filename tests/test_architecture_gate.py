"""Architectural consistency and real-world multi-level gate tests.

Validates:
1. End-to-end multi-level hierarchical ExchangeabilityTree regression for both
   individual and joint tests, ensuring leaf clustering for statsmodels SEs
   and JSON serialization of the resulting test objects.
2. Protocol parity: all 9 permutation strategies accept forward-compatible
   keyword arguments without raising TypeError.
3. Model family method signature uniformity across all 9 families.
4. Modernized GLMM batch stub messages reflecting Kennedy availability.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from randomization_tests import (
    ExchangeabilityTree,
    randomization_test_regression,
)
from randomization_tests._strategies import (
    _STRATEGY_REGISTRY,
    _ensure_registry,
    resolve_strategy,
)
from randomization_tests.families import (
    LinearFamily,
    LogisticFamily,
    ModelFamily,
    MultinomialFamily,
    NegativeBinomialFamily,
    OrdinalFamily,
    PoissonFamily,
)
from randomization_tests.families_mixed import (
    LinearMixedFamily,
    LogisticMixedFamily,
    PoissonMixedFamily,
    _GLMMBatchStubMixin,
)


class TestExchangeabilityTreeRealWorld:
    """Verify that multi-level nested ExchangeabilityTree works end-to-end."""

    def test_nested_tree_individual_and_joint_regression(self) -> None:
        """Real-world educational hierarchy: District -> School -> Student."""
        n_students = 60
        rng = np.random.default_rng(42)
        districts = np.repeat([0, 1], 30)
        schools = np.repeat(np.arange(6), 10)

        tree = ExchangeabilityTree.from_labels(
            [districts, schools],
            strategies=["between", "within"],
        )
        assert tree.n_samples == n_students

        df = pd.DataFrame(
            {
                "study_hours": rng.uniform(2, 20, n_students),
                "parent_edu": rng.integers(1, 5, n_students).astype(float),
                "school_budget": rng.uniform(5000, 15000, n_students),
            }
        )
        df["score"] = (
            50.0
            + 2.5 * df["study_hours"]
            + 3.0 * df["parent_edu"]
            + 0.001 * df["school_budget"]
            + rng.normal(0, 5, n_students)
        )

        # 1. Individual test (Freedman-Lane with confounders)
        res_ind = randomization_test_regression(
            df[["study_hours", "parent_edu", "school_budget"]],
            df[["score"]],
            confounders=["school_budget"],
            groups=tree,
            method="freedman_lane",
            n_randomizations=50,
            random_state=42,
        )
        assert isinstance(res_ind.groups, ExchangeabilityTree)
        assert res_ind.context is not None
        assert isinstance(res_ind.context.groups, ExchangeabilityTree)
        assert len(res_ind.permuted_p_values) == 3
        assert res_ind.permuted_p_values[2] == "(confounder)"

        # Verify JSON serializability of IndividualTestResult with ExchangeabilityTree
        dict_ind = res_ind.to_dict()
        assert isinstance(dict_ind["groups"], list)
        assert len(dict_ind["groups"]) == n_students
        json_str_ind = json.dumps(dict_ind)
        assert len(json_str_ind) > 0

        # 2. Joint test
        res_joint = randomization_test_regression(
            df[["study_hours", "parent_edu", "school_budget"]],
            df[["score"]],
            confounders=["school_budget"],
            groups=tree,
            method="freedman_lane_joint",
            n_randomizations=50,
            random_state=42,
        )
        assert isinstance(res_joint.groups, ExchangeabilityTree)
        assert res_joint.context is not None
        assert isinstance(res_joint.context.groups, ExchangeabilityTree)
        assert 0.0 <= res_joint.p_value <= 1.0

        # Verify JSON serializability of JointTestResult with ExchangeabilityTree
        dict_joint = res_joint.to_dict()
        assert isinstance(dict_joint["groups"], list)
        assert len(dict_joint["groups"]) == n_students
        json_str_joint = json.dumps(dict_joint)
        assert len(json_str_joint) > 0


class TestStrategyParity:
    """Verify strategy execute() protocol parity and forward compatibility."""

    def test_all_strategies_accept_extra_kwargs(self) -> None:
        """Every strategy must accept **kwargs without raising TypeError."""
        _ensure_registry()
        fam = LinearFamily()
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([1.0, 2.0, 3.0, 4.0])
        perm = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])

        for method_name in _STRATEGY_REGISTRY:
            strat = resolve_strategy(method_name)
            # Invoke with arbitrary forward-compatible kwargs
            res = strat.execute(
                X,
                y,
                fam,
                perm,
                confounders=[],
                model_coefs=np.array([1.0]),
                fit_intercept=True,
                n_jobs=1,
                randomization="permute",
                extra_future_arg="forward_compatible",
            )
            assert res is not None


class TestFamilySignatureUniformity:
    """Verify that all 9 model families implement identical method signatures."""

    @pytest.fixture
    def all_families(self) -> list[ModelFamily]:
        return [
            LinearFamily(),
            LogisticFamily(),
            PoissonFamily(),
            NegativeBinomialFamily(),
            OrdinalFamily(),
            MultinomialFamily(),
            LinearMixedFamily(),
            LogisticMixedFamily(),
            PoissonMixedFamily(),
        ]

    def test_varying_x_signature_uniformity(
        self, all_families: list[ModelFamily]
    ) -> None:
        """batch_fit_varying_X must have identical signature across all 9 families."""
        expected_sig = None
        for fam in all_families:
            sig = inspect.signature(fam.batch_fit_varying_X)
            # fit_intercept parameter must NOT have a default value
            param = sig.parameters["fit_intercept"]
            assert param.default is inspect.Parameter.empty, (
                f"{fam.name} batch_fit_varying_X has default fit_intercept={param.default}"
            )
            if expected_sig is None:
                expected_sig = str(sig)
            assert str(sig) == expected_sig, (
                f"{fam.name} signature {sig} does not match {expected_sig}"
            )

    def test_varying_x_fit_and_score_signature_uniformity(
        self, all_families: list[ModelFamily]
    ) -> None:
        """batch_fit_and_score_varying_X must have identical signature across all 9 families."""
        expected_sig = None
        for fam in all_families:
            sig = inspect.signature(fam.batch_fit_and_score_varying_X)
            param = sig.parameters["fit_intercept"]
            assert param.default is inspect.Parameter.empty, (
                f"{fam.name} batch_fit_and_score_varying_X has default fit_intercept={param.default}"
            )
            if expected_sig is None:
                expected_sig = str(sig)
            assert str(sig) == expected_sig, (
                f"{fam.name} signature {sig} does not match {expected_sig}"
            )

    def test_glmm_stub_mixin_message(self) -> None:
        """_GLMMBatchStubMixin must mention score, score_joint, and kennedy."""
        stub = _GLMMBatchStubMixin()
        with pytest.raises(NotImplementedError) as exc_info:
            stub.batch_fit(np.zeros((2, 2)), np.zeros((2, 2)), fit_intercept=True)
        msg = str(exc_info.value)
        assert "kennedy" in msg
        assert "score" in msg

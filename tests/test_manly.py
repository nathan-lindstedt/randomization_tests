"""Tests for the Manly direct Y-permutation strategy.

Covers:
- Protocol conformance and registry wiring
- Individual test on ordinal / multinomial families
- Joint test on ordinal / multinomial families
- UserWarning when Manly is used on a residual-based family (linear)
- ValueError when sign_flip is requested with Manly (no residuals)
- Valid p-values (∈ (0, 1])
- Result type discrimination (Individual vs Joint)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from randomization_tests import (
    IndividualTestResult,
    JointTestResult,
    randomization_test_regression,
)
from randomization_tests._strategies import PermutationStrategy, resolve_strategy
from randomization_tests._strategies.manly import ManlyJointStrategy, ManlyStrategy

# ------------------------------------------------------------------ #
# Data factories
# ------------------------------------------------------------------ #


def _make_ordinal_data(
    n: int = 200, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    latent = 1.5 * X["x1"].values + rng.standard_normal(n)
    y = pd.DataFrame({"y": np.digitize(latent, bins=[-1, 0, 1]).astype(float)})
    return X, y


def _make_multinomial_data(
    n: int = 300, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    logits_1 = 1.0 * X["x1"].values
    logits_2 = -0.5 * X["x2"].values
    p0 = 1.0 / (1.0 + np.exp(logits_1) + np.exp(logits_2))
    p1 = np.exp(logits_1) * p0
    p2 = np.exp(logits_2) * p0
    probs = np.column_stack([p0, p1, p2])
    y_vals = np.array([rng.choice(3, p=probs[i]) for i in range(n)]).astype(float)
    y = pd.DataFrame({"y": y_vals})
    return X, y


def _make_linear_data(
    n: int = 100, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    y = pd.DataFrame({"y": 2.0 * X["x1"].values + rng.standard_normal(n)})
    return X, y


# ------------------------------------------------------------------ #
# Protocol conformance and registry wiring
# ------------------------------------------------------------------ #


class TestManlyProtocol:
    """ManlyStrategy and ManlyJointStrategy satisfy PermutationStrategy protocol."""

    def test_manly_individual_is_strategy(self):
        assert isinstance(ManlyStrategy(), PermutationStrategy)

    def test_manly_joint_is_strategy(self):
        assert isinstance(ManlyJointStrategy(), PermutationStrategy)

    def test_manly_individual_is_not_joint(self):
        assert ManlyStrategy().is_joint is False

    def test_manly_joint_is_joint(self):
        assert ManlyJointStrategy().is_joint is True

    def test_resolve_strategy_manly(self):
        strategy = resolve_strategy("manly")
        assert isinstance(strategy, ManlyStrategy)

    def test_resolve_strategy_manly_joint(self):
        strategy = resolve_strategy("manly_joint")
        assert isinstance(strategy, ManlyJointStrategy)


# ------------------------------------------------------------------ #
# Individual tests
# ------------------------------------------------------------------ #


class TestManlyIndividual:
    """Manly individual test returns IndividualTestResult with valid p-values."""

    def test_ordinal_returns_individual_result(self):
        X, y = _make_ordinal_data()
        result = randomization_test_regression(
            X, y, n_randomizations=50, random_state=42, method="manly", family="ordinal"
        )
        assert isinstance(result, IndividualTestResult)

    def test_ordinal_p_values_valid(self):
        X, y = _make_ordinal_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=100,
            random_state=42,
            method="manly",
            family="ordinal",
        )
        assert all(0 < p <= 1 for p in result.raw_empirical_p)

    def test_ordinal_correct_number_of_coefs(self):
        X, y = _make_ordinal_data()
        result = randomization_test_regression(
            X, y, n_randomizations=50, random_state=42, method="manly", family="ordinal"
        )
        assert len(result.model_coefs) == X.shape[1]

    def test_multinomial_returns_individual_result(self):
        X, y = _make_multinomial_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="manly",
            family="multinomial",
        )
        assert isinstance(result, IndividualTestResult)

    def test_multinomial_p_values_valid(self):
        X, y = _make_multinomial_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=100,
            random_state=42,
            method="manly",
            family="multinomial",
        )
        assert all(0 < p <= 1 for p in result.raw_empirical_p)

    def test_feature_names_preserved(self):
        X, y = _make_ordinal_data()
        result = randomization_test_regression(
            X, y, n_randomizations=50, random_state=42, method="manly", family="ordinal"
        )
        assert result.feature_names == ["x1", "x2"]

    def test_deterministic_under_fixed_seed(self):
        X, y = _make_ordinal_data()
        kwargs = dict(
            n_randomizations=50, random_state=0, method="manly", family="ordinal"
        )
        r1 = randomization_test_regression(X, y, **kwargs)
        r2 = randomization_test_regression(X, y, **kwargs)
        np.testing.assert_array_equal(r1.raw_empirical_p, r2.raw_empirical_p)


# ------------------------------------------------------------------ #
# Joint tests
# ------------------------------------------------------------------ #


class TestManlyJoint:
    """Manly joint test returns JointTestResult with valid p-value.

    manly_joint uses direct Y permutation and compares deviance reduction
    between the reduced model (confounders only) and the full model.  For
    ordinal and multinomial families, deviance is obtained from ``score()``
    (via ``model.llf``) rather than ``fit_metric()`` — so all families
    that support ``batch_fit_and_score`` are supported.
    """

    def test_linear_returns_joint_result(self):
        X, y = _make_linear_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Manly is marginal
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                random_state=42,
                method="manly_joint",
                family="linear",
            )
        assert isinstance(result, JointTestResult)

    def test_linear_joint_p_value_valid(self):
        X, y = _make_linear_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=100,
                random_state=42,
                method="manly_joint",
                family="linear",
            )
        assert 0 < result.p_value <= 1

    def test_ordinal_joint_returns_result(self):
        """manly_joint works for ordinal — uses score() instead of fit_metric()."""
        X, y = _make_ordinal_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="manly_joint",
            family="ordinal",
        )
        assert isinstance(result, JointTestResult)
        assert 0 < result.p_value <= 1

    def test_multinomial_joint_returns_result(self):
        """manly_joint works for multinomial — uses score() instead of fit_metric()."""
        X, y = _make_multinomial_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="manly_joint",
            family="multinomial",
        )
        assert isinstance(result, JointTestResult)
        assert 0 < result.p_value <= 1


# ------------------------------------------------------------------ #
# Guard: sign_flip rejected with Manly
# ------------------------------------------------------------------ #


class TestManlyRejectsSignFlip:
    """sign_flip is incompatible with Manly (no residuals to flip)."""

    def test_manly_individual_rejects_sign_flip(self):
        # Must use a non-direct-permutation family (linear) so the Manly guard
        # fires rather than the ordinal/multinomial family guard.  Suppress the
        # expected Manly-on-residual-family UserWarning so the ValueError is
        # raised (pytest converts all warnings → errors by default).
        X, y = _make_linear_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="Manly"):
                randomization_test_regression(
                    X,
                    y,
                    n_randomizations=50,
                    family="linear",
                    method="manly",
                    randomization="sign_flip",
                )

    def test_manly_joint_rejects_sign_flip(self):
        X, y = _make_linear_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="Manly"):
                randomization_test_regression(
                    X,
                    y,
                    n_randomizations=50,
                    family="linear",
                    method="manly_joint",
                    randomization="sign_flip",
                )


# ------------------------------------------------------------------ #
# Guard: UserWarning when Manly used on residual-based family
# ------------------------------------------------------------------ #


class TestManlyWarnsOnResidualFamily:
    """Manly on linear/logistic/etc. emits a UserWarning (marginal test)."""

    def test_manly_on_linear_warns(self):
        X, y = _make_linear_data()
        with pytest.warns(UserWarning, match="marginal"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                random_state=42,
                method="manly",
                family="linear",
            )

    def test_manly_on_linear_still_returns_result(self):
        X, y = _make_linear_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                random_state=42,
                method="manly",
                family="linear",
            )
        assert isinstance(result, IndividualTestResult)
        assert all(0 < p <= 1 for p in result.raw_empirical_p)

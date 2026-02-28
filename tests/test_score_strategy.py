"""Tests for the score projection strategy (Plan B).

Covers:
- Protocol conformance and registry wiring
- Linear individual & joint end-to-end
- LMM individual & joint end-to-end
- Score ≡ Freedman–Lane equivalence (LMM, bit-for-bit)
- Score ≡ ter Braak equivalence (linear, no confounders)
- Unsupported family rejection
- score_exact non-GLMM rejection
- Confounder masking
- n_jobs warning for score path
- Valid p-values (∈ [0, 1])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from randomization_tests import (
    IndividualTestResult,
    JointTestResult,
    permutation_test_regression,
)
from randomization_tests._strategies import (
    PermutationStrategy,
    resolve_strategy,
)
from randomization_tests._strategies.score import (
    ScoreExactStrategy,
    ScoreIndividualStrategy,
    ScoreJointStrategy,
)

# ------------------------------------------------------------------ #
# Shared helpers
# ------------------------------------------------------------------ #

_SEED = 12345
_N_PERMS = 200


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


def _grouped_data(
    n_per_group: int = 20,
    n_groups: int = 10,
    seed: int = _SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Clustered LMM data: y = Xβ + Zu + ε."""
    rng = np.random.default_rng(seed)
    n = n_per_group * n_groups
    groups = np.repeat(np.arange(n_groups), n_per_group)
    X = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    beta_true = np.array([2.0, -1.0])
    u = rng.normal(0, 2.0, size=n_groups)  # τ² = 4.0
    y_vals = X.values @ beta_true + u[groups] + rng.standard_normal(n)
    y = pd.DataFrame({"y": y_vals})
    return X, y, groups


# ------------------------------------------------------------------ #
# Protocol conformance & registry
# ------------------------------------------------------------------ #


class TestProtocolConformance:
    """Score strategies satisfy the PermutationStrategy protocol."""

    def test_individual_isinstance(self) -> None:
        assert isinstance(ScoreIndividualStrategy(), PermutationStrategy)

    def test_joint_isinstance(self) -> None:
        assert isinstance(ScoreJointStrategy(), PermutationStrategy)

    def test_exact_isinstance(self) -> None:
        assert isinstance(ScoreExactStrategy(), PermutationStrategy)

    def test_individual_is_not_joint(self) -> None:
        assert ScoreIndividualStrategy.is_joint is False

    def test_joint_is_joint(self) -> None:
        assert ScoreJointStrategy.is_joint is True

    def test_exact_is_not_joint(self) -> None:
        assert ScoreExactStrategy.is_joint is False


class TestRegistry:
    """Score strategies are reachable via resolve_strategy()."""

    def test_resolve_score(self) -> None:
        s = resolve_strategy("score")
        assert isinstance(s, ScoreIndividualStrategy)

    def test_resolve_score_joint(self) -> None:
        s = resolve_strategy("score_joint")
        assert isinstance(s, ScoreJointStrategy)

    def test_resolve_score_exact(self) -> None:
        s = resolve_strategy("score_exact")
        assert isinstance(s, ScoreExactStrategy)


# ------------------------------------------------------------------ #
# Linear individual
# ------------------------------------------------------------------ #


class TestScoreLinearIndividual:
    """method='score' with family='linear'."""

    def test_returns_individual_result(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score",
        )
        assert isinstance(result, IndividualTestResult)

    def test_p_values_in_range(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score",
        )
        assert all(0.0 <= p <= 1.0 for p in result.raw_empirical_p)

    def test_significant_feature_detected(self) -> None:
        """x1 (β=2) should have a small p-value."""
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score",
        )
        idx_x1 = list(X.columns).index("x1")
        assert result.raw_empirical_p[idx_x1] < 0.05

    def test_null_feature_not_rejected(self) -> None:
        """x3 (β=0) should have a large p-value."""
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score",
        )
        idx_x3 = list(X.columns).index("x3")
        assert result.raw_empirical_p[idx_x3] > 0.05


# ------------------------------------------------------------------ #
# Score ≡ ter Braak equivalence (linear, no confounders)
# ------------------------------------------------------------------ #


class TestScoreEqualsTerBraak:
    """For linear OLS without confounders, score ≡ ter Braak.

    Both compute pinv(X)[j] @ e_π — the score strategy via matmul,
    ter Braak via batch_fit + column extraction.  With the same
    permutation indices (same seed), the p-values must be identical.
    """

    def test_p_values_match(self) -> None:
        X, y = _linear_data()
        r_score = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score",
        )
        r_tb = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="ter_braak",
        )
        np.testing.assert_array_equal(
            r_score.raw_empirical_p,
            r_tb.raw_empirical_p,
        )


# ------------------------------------------------------------------ #
# LMM individual
# ------------------------------------------------------------------ #


class TestScoreLMMIndividual:
    """method='score' with family='linear_mixed'."""

    def test_returns_individual_result(self) -> None:
        X, y, groups = _grouped_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score",
            family="linear_mixed",
            groups=groups,
        )
        assert isinstance(result, IndividualTestResult)

    def test_p_values_in_range(self) -> None:
        X, y, groups = _grouped_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score",
            family="linear_mixed",
            groups=groups,
        )
        assert all(0.0 <= p <= 1.0 for p in result.raw_empirical_p)

    def test_significant_feature_detected(self) -> None:
        """x1 (β=2) should have a small p-value."""
        X, y, groups = _grouped_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score",
            family="linear_mixed",
            groups=groups,
        )
        idx_x1 = list(X.columns).index("x1")
        assert result.raw_empirical_p[idx_x1] < 0.05


# ------------------------------------------------------------------ #
# Score ≡ Freedman–Lane equivalence (LMM, bit-for-bit)
# ------------------------------------------------------------------ #


class TestScoreEqualsFreedmanLaneLMM:
    """For LMM, score ≡ Freedman–Lane (bit-for-bit identical).

    Both compute A @ Y* where A = (X'V⁻¹X)⁻¹X'V⁻¹ and
    Y* = ŷ_red + e_π.  The score strategy does this via one matmul
    per feature; Freedman–Lane does the full batch refit.  Same
    permutation indices → identical p-values.
    """

    def test_p_values_match(self) -> None:
        X, y, groups = _grouped_data()
        r_score = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score",
            family="linear_mixed",
            groups=groups,
        )
        r_fl = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="freedman_lane",
            family="linear_mixed",
            groups=groups,
            confounders=["x2"],
        )
        # Only compare the feature tested by both — x1.
        # FL with confounders=["x2"] tests x1; score tests all.
        idx_x1 = list(X.columns).index("x1")
        np.testing.assert_array_equal(
            r_score.raw_empirical_p[idx_x1],
            r_fl.raw_empirical_p[idx_x1],
        )

    def test_p_values_match_no_confounders(self) -> None:
        """Without confounders, every feature's p-value must match."""
        import warnings

        X, y, groups = _grouped_data()
        r_score = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score",
            family="linear_mixed",
            groups=groups,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_fl = permutation_test_regression(
                X,
                y,
                n_permutations=_N_PERMS,
                random_state=_SEED,
                method="freedman_lane",
                family="linear_mixed",
                groups=groups,
            )
        np.testing.assert_array_equal(
            r_score.raw_empirical_p,
            r_fl.raw_empirical_p,
        )


# ------------------------------------------------------------------ #
# Linear joint
# ------------------------------------------------------------------ #


class TestScoreLinearJoint:
    """method='score_joint' with family='linear'."""

    def test_returns_joint_result(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score_joint",
            confounders=["x3"],
        )
        assert isinstance(result, JointTestResult)

    def test_p_value_in_range(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score_joint",
            confounders=["x3"],
        )
        assert 0.0 <= result.p_value <= 1.0


# ------------------------------------------------------------------ #
# LMM joint
# ------------------------------------------------------------------ #


class TestScoreLMMJoint:
    """method='score_joint' with family='linear_mixed'."""

    def test_returns_joint_result(self) -> None:
        X, y, groups = _grouped_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score_joint",
            family="linear_mixed",
            groups=groups,
            confounders=["x2"],
        )
        assert isinstance(result, JointTestResult)

    def test_p_value_in_range(self) -> None:
        X, y, groups = _grouped_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score_joint",
            family="linear_mixed",
            groups=groups,
            confounders=["x2"],
        )
        assert 0.0 <= result.p_value <= 1.0


# ------------------------------------------------------------------ #
# Unsupported families → clear error
# ------------------------------------------------------------------ #


class TestUnsupportedFamily:
    """score methods with unsupported families raise ValueError."""

    @pytest.mark.parametrize("method", ["score", "score_joint", "score_exact"])
    def test_logistic_rejected(self, method: str) -> None:
        rng = np.random.default_rng(_SEED)
        n = 100
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        logits = 1.5 * X["x1"]
        y = pd.DataFrame({"y": rng.binomial(1, 1 / (1 + np.exp(-logits)))})
        with pytest.raises(ValueError, match="score_project"):
            permutation_test_regression(
                X,
                y,
                n_permutations=50,
                random_state=_SEED,
                method=method,
                family="logistic",
            )

    @pytest.mark.parametrize("method", ["score", "score_joint", "score_exact"])
    def test_poisson_rejected(self, method: str) -> None:
        rng = np.random.default_rng(_SEED)
        n = 100
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        y = pd.DataFrame({"y": rng.poisson(3, size=n)})
        with pytest.raises(ValueError, match="score_project"):
            permutation_test_regression(
                X,
                y,
                n_permutations=50,
                random_state=_SEED,
                method=method,
                family="poisson",
            )


# ------------------------------------------------------------------ #
# score_exact non-GLMM rejection
# ------------------------------------------------------------------ #


class TestScoreExactNonGLMM:
    """score_exact raises ValueError for non-GLMM families."""

    def test_linear_raises(self) -> None:
        """score_exact on linear rejects because linear has no log_chol."""
        X, y = _linear_data()
        with pytest.raises(ValueError, match="score_exact"):
            permutation_test_regression(
                X,
                y,
                n_permutations=50,
                random_state=_SEED,
                method="score_exact",
            )

    def test_lmm_raises(self) -> None:
        """score_exact on linear_mixed rejects — REML, not Laplace."""
        X, y, groups = _grouped_data()
        with pytest.raises(ValueError, match="score_exact"):
            permutation_test_regression(
                X,
                y,
                n_permutations=50,
                random_state=_SEED,
                method="score_exact",
                family="linear_mixed",
                groups=groups,
            )


# ------------------------------------------------------------------ #
# Confounder masking
# ------------------------------------------------------------------ #


class TestConfounderMasking:
    """Score individual masks confounder p-values as N/A."""

    def test_confounder_masked(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score",
            confounders=["x3"],
        )
        idx_x3 = list(X.columns).index("x3")
        assert np.isnan(result.raw_empirical_p[idx_x3])
        assert "N/A" in result.permuted_p_values[idx_x3]

    def test_non_confounder_not_masked(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score",
            confounders=["x3"],
        )
        idx_x1 = list(X.columns).index("x1")
        assert not np.isnan(result.raw_empirical_p[idx_x1])


# ------------------------------------------------------------------ #
# n_jobs warning
# ------------------------------------------------------------------ #


class TestNJobsWarning:
    """score with n_jobs > 1 on linear should warn (vectorised path)."""

    @pytest.fixture(autouse=True)
    def _force_numpy_backend(self) -> None:  # type: ignore[return]
        from randomization_tests import get_backend, set_backend

        prev = get_backend()
        set_backend("numpy")
        yield
        set_backend(prev)

    @pytest.mark.parametrize("method", ["score", "score_joint"])
    def test_warns_on_linear(self, method: str) -> None:
        X, y = _linear_data()
        kwargs: dict = dict(
            n_permutations=50,
            random_state=_SEED,
            method=method,
            n_jobs=2,
        )
        if method == "score_joint":
            kwargs["confounders"] = ["x3"]
        with pytest.warns(UserWarning, match="n_jobs has no effect"):
            permutation_test_regression(X, y, **kwargs)


# ------------------------------------------------------------------ #
# Determinism
# ------------------------------------------------------------------ #


class TestDeterminism:
    """Same seed → same results for score methods."""

    def test_score_deterministic(self) -> None:
        X, y = _linear_data()
        r1 = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="score"
        )
        r2 = permutation_test_regression(
            X, y, n_permutations=_N_PERMS, random_state=_SEED, method="score"
        )
        np.testing.assert_array_equal(r1.raw_empirical_p, r2.raw_empirical_p)

    def test_score_joint_deterministic(self) -> None:
        X, y = _linear_data()
        r1 = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score_joint",
            confounders=["x3"],
        )
        r2 = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="score_joint",
            confounders=["x3"],
        )
        assert r1.p_value == r2.p_value

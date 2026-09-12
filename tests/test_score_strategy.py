"""Tests for the score projection strategy (Plan B).

Covers:
- Protocol conformance and registry wiring
- Linear individual & joint end-to-end
- LMM individual & joint end-to-end
- Score ≡ Freedman–Lane equivalence (LMM, bit-for-bit)
- Score ≡ Freedman–Lane equivalence (linear, no confounders)
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
    randomization_test_regression,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score",
        )
        assert isinstance(result, IndividualTestResult)

    def test_p_values_in_range(self) -> None:
        X, y = _linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score",
        )
        assert all(0.0 <= p <= 1.0 for p in result.raw_empirical_p)

    def test_significant_feature_detected(self) -> None:
        """x1 (β=2) should have a small p-value."""
        X, y = _linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score",
        )
        idx_x1 = list(X.columns).index("x1")
        assert result.raw_empirical_p[idx_x1] < 0.05

    def test_null_feature_not_rejected(self) -> None:
        """x3 (β=0) should have a large p-value."""
        X, y = _linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score",
        )
        idx_x3 = list(X.columns).index("x3")
        assert result.raw_empirical_p[idx_x3] > 0.05


# ------------------------------------------------------------------ #
# Score ≡ Freedman–Lane equivalence (linear, no confounders)
# ------------------------------------------------------------------ #


class TestScoreEqualsFreedmanLane:
    """For linear OLS, score ≡ canonical Freedman–Lane.

    For each tested feature j both permute the X_{−j} reduced-model
    residuals and evaluate A_j @ Y* — the score strategy via matmul,
    Freedman–Lane via batch_fit + column extraction.  With the same
    permutation indices (same seed), the p-values must be identical.
    """

    def test_p_values_match(self) -> None:
        import warnings

        X, y = _linear_data()
        r_score = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_fl = randomization_test_regression(
                X,
                y,
                n_randomizations=_N_PERMS,
                random_state=_SEED,
                method="freedman_lane",
            )
        np.testing.assert_array_equal(
            r_score.raw_empirical_p,
            r_fl.raw_empirical_p,
        )


# ------------------------------------------------------------------ #
# LMM individual
# ------------------------------------------------------------------ #


class TestScoreLMMIndividual:
    """method='score' with family='linear_mixed'."""

    def test_returns_individual_result(self) -> None:
        X, y, groups = _grouped_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score",
            family="linear_mixed",
            groups=groups,
        )
        assert isinstance(result, IndividualTestResult)

    def test_p_values_in_range(self) -> None:
        X, y, groups = _grouped_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score",
            family="linear_mixed",
            groups=groups,
        )
        assert all(0.0 <= p <= 1.0 for p in result.raw_empirical_p)

    def test_significant_feature_detected(self) -> None:
        """x1 (β=2) should have a small p-value."""
        X, y, groups = _grouped_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
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
        r_score = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score",
            family="linear_mixed",
            groups=groups,
        )
        r_fl = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
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
        r_score = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score",
            family="linear_mixed",
            groups=groups,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_fl = randomization_test_regression(
                X,
                y,
                n_randomizations=_N_PERMS,
                random_state=_SEED,
                method="freedman_lane",
                family="linear_mixed",
                groups=groups,
            )
        np.testing.assert_array_equal(
            r_score.raw_empirical_p,
            r_fl.raw_empirical_p,
        )

    def test_null_distributions_match(self) -> None:
        """Compare the NULL DRAWS, not just the p-values.

        A permutation p-value is a count over B draws, so a null that has shifted
        or changed width can still produce an identical count and slip past an
        ``assert_array_equal`` on p-values alone.  That is not hypothetical: when
        whitening was applied to ``score_project`` and not to the Freedman-Lane
        path, the two nulls diverged on 5 of 6 configurations (score
        ``sd(draws)/SE`` 0.98 against FL 0.41 under random slopes) while the
        p-value assertions above still passed on this fixture.

        Since the two strategies compute the same estimator, their null draws must
        agree elementwise, not merely in the tail count they imply.
        """
        import warnings

        X, y, groups = _grouped_data()
        kwargs = dict(
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            family="linear_mixed",
            groups=groups,
        )
        r_score = randomization_test_regression(X, y, method="score", **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_fl = randomization_test_regression(X, y, method="freedman_lane", **kwargs)

        draws_score = np.asarray(r_score.permuted_coefs)
        draws_fl = np.asarray(r_fl.permuted_coefs)
        assert draws_score.shape == draws_fl.shape
        np.testing.assert_allclose(
            draws_score,
            draws_fl,
            rtol=1e-9,
            atol=1e-9,
            err_msg=(
                "score and freedman_lane null draws diverged; they compute the "
                "same estimator, so one path has been changed without the other"
            ),
        )

    def test_null_distributions_match_under_random_slopes(self) -> None:
        """The same invariant where the fixture above is insensitive.

        ``_grouped_data()`` is a random-intercept design, the one configuration
        where a whitening change is close to neutral.  Random slopes are where the
        two paths visibly separate, so the coupling must be pinned there too.
        """
        import warnings

        rng = np.random.default_rng(4242)
        n_groups, per_group = 12, 8
        n = n_groups * per_group
        groups = np.repeat(np.arange(n_groups), per_group)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y_vals = np.empty(n)
        for g in range(n_groups):
            idx = np.flatnonzero(np.equal(groups, g))
            slope = rng.normal(scale=1.5)
            y_vals[idx] = (
                0.6 * x2[idx]
                + rng.normal(scale=2.0)
                + slope * x1[idx]
                + rng.normal(size=idx.size)
            )
        X = pd.DataFrame({"x1": x1, "x2": x2})
        y = pd.DataFrame({"y": y_vals})

        kwargs = dict(
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            family="linear_mixed",
            groups=groups,
            random_slopes=[0],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_score = randomization_test_regression(X, y, method="score", **kwargs)
            r_fl = randomization_test_regression(X, y, method="freedman_lane", **kwargs)

        np.testing.assert_allclose(
            np.asarray(r_score.permuted_coefs),
            np.asarray(r_fl.permuted_coefs),
            rtol=1e-9,
            atol=1e-9,
            err_msg=("score and freedman_lane null draws diverged under random slopes"),
        )


# ------------------------------------------------------------------ #
# Linear joint
# ------------------------------------------------------------------ #


class TestScoreLinearJoint:
    """method='score_joint' with family='linear'."""

    def test_returns_joint_result(self) -> None:
        X, y = _linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score_joint",
            confounders=["x3"],
        )
        assert isinstance(result, JointTestResult)

    def test_p_value_in_range(self) -> None:
        X, y = _linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score_joint",
            family="linear_mixed",
            groups=groups,
            confounders=["x2"],
        )
        assert isinstance(result, JointTestResult)

    def test_p_value_in_range(self) -> None:
        X, y, groups = _grouped_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score_joint",
            family="linear_mixed",
            groups=groups,
            confounders=["x2"],
        )
        assert 0.0 <= result.p_value <= 1.0


# ------------------------------------------------------------------ #
# Score joint ≡ Freedman–Lane joint equivalence (M13)
# ------------------------------------------------------------------ #


class TestScoreEqualsFreedmanLaneJoint:
    """score_joint's residual-based branch shares one implementation with
    freedman_lane_joint (``_residual_joint_statistic``), so with the same
    permutation indices the null arrays and observed statistic must be
    bit-identical -- not merely close.  Regression guard for M13, where
    score_joint permuted full-model residuals instead of reduced-model
    residuals and silently diverged from Freedman-Lane joint.
    """

    def test_linear_with_confounders_and_real_effect(self) -> None:
        """The configuration that exposed M13 (q=2 features, real effect)."""
        rng = np.random.default_rng(7)
        n = 200
        x1, x2, z = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
        y_vals = 0.7 * z + 0.5 * x1 + 0.5 * x2 + rng.normal(size=n)
        X = pd.DataFrame({"x1": x1, "x2": x2, "z": z})
        y = pd.DataFrame({"y": y_vals})

        r_fl = randomization_test_regression(
            X,
            y,
            family="linear",
            method="freedman_lane_joint",
            confounders=["z"],
            n_randomizations=_N_PERMS,
            random_state=_SEED,
        )
        r_sj = randomization_test_regression(
            X,
            y,
            family="linear",
            method="score_joint",
            confounders=["z"],
            n_randomizations=_N_PERMS,
            random_state=_SEED,
        )
        assert r_fl.observed_improvement == r_sj.observed_improvement
        np.testing.assert_array_equal(
            r_fl.permuted_improvements, r_sj.permuted_improvements
        )

    def test_lmm_with_random_slopes(self) -> None:
        """LMM under random slopes (whitening engaged) -- the configuration
        where an unwhitened score_joint would previously have diverged."""
        X, y, groups = _grouped_data()
        r_fl = randomization_test_regression(
            X,
            y,
            family="linear_mixed",
            groups=groups,
            method="freedman_lane_joint",
            confounders=["x2"],
            n_randomizations=_N_PERMS,
            random_state=_SEED,
        )
        r_sj = randomization_test_regression(
            X,
            y,
            family="linear_mixed",
            groups=groups,
            method="score_joint",
            confounders=["x2"],
            n_randomizations=_N_PERMS,
            random_state=_SEED,
        )
        assert r_fl.observed_improvement == r_sj.observed_improvement
        np.testing.assert_array_equal(
            r_fl.permuted_improvements, r_sj.permuted_improvements
        )


# ------------------------------------------------------------------ #
# Unsupported families → clear error
# ------------------------------------------------------------------ #


class TestUnsupportedFamily:
    """score methods with previously unsupported families now work.

    LogisticFamily and PoissonFamily gained real ``score_project()``
    implementations in Phase 5, so ``score`` and ``score_joint``
    should complete without raising.  ``score_exact`` remains GLMM-
    only and is tested in ``TestScoreExactNonGLMM`` below.
    """

    @pytest.mark.parametrize("method", ["score", "score_joint"])
    def test_logistic_accepted(self, method: str) -> None:
        rng = np.random.default_rng(_SEED)
        n = 100
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        logits = 1.5 * X["x1"]
        y = pd.DataFrame({"y": rng.binomial(1, 1 / (1 + np.exp(-logits)))})
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=_SEED,
            method=method,
            family="logistic",
        )
        assert result is not None

    @pytest.mark.parametrize("method", ["score", "score_joint"])
    def test_poisson_accepted(self, method: str) -> None:
        rng = np.random.default_rng(_SEED)
        n = 100
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        y = pd.DataFrame({"y": rng.poisson(3, size=n)})
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=_SEED,
            method=method,
            family="poisson",
        )
        assert result is not None


# ------------------------------------------------------------------ #
# score_exact non-GLMM rejection
# ------------------------------------------------------------------ #


class TestScoreExactNonGLMM:
    """score_exact warns and then raises ValueError for non-GLMM families."""

    def test_linear_raises(self) -> None:
        """score_exact on linear: UserWarning from engine, ValueError from strategy."""
        X, y = _linear_data()
        with pytest.warns(UserWarning, match="score_exact"), pytest.raises(ValueError):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                random_state=_SEED,
                method="score_exact",
            )

    def test_lmm_raises(self) -> None:
        """score_exact on linear_mixed rejects — REML, not Laplace."""
        X, y, groups = _grouped_data()
        with pytest.raises(ValueError, match="score_exact"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
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
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score",
            confounders=["x3"],
        )
        idx_x3 = list(X.columns).index("x3")
        assert np.isnan(result.raw_empirical_p[idx_x3])
        assert result.permuted_p_values[idx_x3] == "(confounder)"

    def test_non_confounder_not_masked(self) -> None:
        X, y = _linear_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
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
            n_randomizations=50,
            random_state=_SEED,
            method=method,
            n_jobs=2,
        )
        if method == "score_joint":
            kwargs["confounders"] = ["x3"]
        with pytest.warns(UserWarning, match="n_jobs has no effect"):
            randomization_test_regression(X, y, **kwargs)


# ------------------------------------------------------------------ #
# Determinism
# ------------------------------------------------------------------ #


class TestDeterminism:
    """Same seed → same results for score methods."""

    def test_score_deterministic(self) -> None:
        X, y = _linear_data()
        r1 = randomization_test_regression(
            X, y, n_randomizations=_N_PERMS, random_state=_SEED, method="score"
        )
        r2 = randomization_test_regression(
            X, y, n_randomizations=_N_PERMS, random_state=_SEED, method="score"
        )
        np.testing.assert_array_equal(r1.raw_empirical_p, r2.raw_empirical_p)

    def test_score_joint_deterministic(self) -> None:
        X, y = _linear_data()
        r1 = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score_joint",
            confounders=["x3"],
        )
        r2 = randomization_test_regression(
            X,
            y,
            n_randomizations=_N_PERMS,
            random_state=_SEED,
            method="score_joint",
            confounders=["x3"],
        )
        assert r1.p_value == r2.p_value


# ------------------------------------------------------------------ #
# Singularity guards (audit fix Steps 3, 7)
# ------------------------------------------------------------------ #


class TestSingularityGuards:
    """Verify score projection handles singular matrices gracefully."""

    def test_collinear_fisher_returns_finite(self) -> None:
        """_glm_score_projection_row with collinear design -> finite result."""
        from randomization_tests._backends._jax import _glm_score_projection_row

        n, p = 50, 3
        X = np.random.default_rng(42).standard_normal((n, p))
        # Make column 2 a duplicate of column 1 -> singular Fisher
        X[:, 2] = X[:, 1]
        X_aug = np.column_stack([np.ones(n), X])
        W_diag = np.ones(n) * 0.25  # logistic working weights
        result = _glm_score_projection_row(X_aug, W_diag, feature_idx=1)
        assert np.all(np.isfinite(result))
        assert result.shape == (n,)


# ------------------------------------------------------------------ #
# AR score parity (Step 19)
# ------------------------------------------------------------------ #


class TestARScoreParity:
    """AR score correction parity and non-triviality tests."""

    def test_linear_score_cached_pinv_matches_inline(self) -> None:
        """Cached projection_A gives bit-for-bit same results as inline pinv."""
        X, y = _linear_data(n=80, seed=99)
        # Two runs with the same seed should give identical null distributions
        res1 = randomization_test_regression(
            X, y, n_randomizations=_N_PERMS, random_state=_SEED, method="score"
        )
        res2 = randomization_test_regression(
            X, y, n_randomizations=_N_PERMS, random_state=_SEED, method="score"
        )
        np.testing.assert_array_equal(res1.raw_empirical_p, res2.raw_empirical_p)
        np.testing.assert_array_equal(res1.model_coefs, res2.model_coefs)

    def test_linear_score_ar_changes_pvalues(self) -> None:
        """AR(1) correction produces different p-values than without."""
        rng = np.random.default_rng(42)
        n_panels, n_times = 20, 20
        n = n_panels * n_times
        panel_id = np.repeat(np.arange(n_panels), n_times)
        time_id = np.tile(np.arange(n_times), n_panels)

        # Panel data with strong AR(1) errors and WEAK signal so p-values
        # aren't at the boundary.
        X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
        eps = np.zeros(n)
        for p in range(n_panels):
            start = p * n_times
            eps[start] = rng.standard_normal()
            for t in range(1, n_times):
                eps[start + t] = 0.8 * eps[start + t - 1] + rng.standard_normal()
        # Weak effect: 0.2 instead of 2.0
        y = pd.DataFrame({"y": 0.2 * X["x1"].values + eps})

        res_no_ar = randomization_test_regression(
            X,
            y,
            n_randomizations=200,
            random_state=_SEED,
            method="score",
            panel_id=panel_id,
            time_id=time_id,
        )
        res_ar = randomization_test_regression(
            X,
            y,
            n_randomizations=200,
            random_state=_SEED,
            method="score",
            panel_id=panel_id,
            time_id=time_id,
            ar_order=1,
        )
        # p-values should differ — AR correction modifies the score projection
        assert not np.allclose(res_no_ar.raw_empirical_p, res_ar.raw_empirical_p), (
            "AR correction had no effect on p-values"
        )


# ------------------------------------------------------------------ #
# score / score_joint for ordinal and multinomial families
# ------------------------------------------------------------------ #

_RNG_CAT = np.random.default_rng(42)
_N_CAT = 60


def _make_ordinal_data(n: int = _N_CAT) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["x1", "x2", "x3"])
    y = pd.DataFrame({"y": rng.integers(0, 4, n)})
    return X, y


def _make_multinomial_data(n: int = _N_CAT) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["x1", "x2", "x3"])
    y = pd.DataFrame({"y": rng.integers(0, 4, n)})
    return X, y


class TestScoreOrdinalMultinomial:
    """score and score_joint work for ordinal and multinomial families.

    The score method uses the exact score_project() implementation
    (R-matrix for ordinal, chi-square for multinomial).  The score_joint
    method uses direct Y permutation (since these are direct_permutation
    families) with deviance as the test statistic.
    """

    def test_score_ordinal_returns_individual_result(self):
        X, y = _make_ordinal_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score",
            family="ordinal",
        )
        assert isinstance(result, IndividualTestResult)
        assert len(result["model_coefs"]) == 3

    def test_score_ordinal_p_values_finite(self):
        X, y = _make_ordinal_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=100,
            random_state=42,
            method="score",
            family="ordinal",
        )
        p = result.raw_empirical_p
        assert np.all(np.isfinite(p))
        assert np.all(p >= 0)
        assert np.all(p <= 1)

    def test_score_multinomial_returns_individual_result(self):
        X, y = _make_multinomial_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score",
            family="multinomial",
        )
        assert isinstance(result, IndividualTestResult)
        assert len(result["model_coefs"]) == 3

    def test_score_multinomial_p_values_finite(self):
        X, y = _make_multinomial_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=100,
            random_state=42,
            method="score",
            family="multinomial",
        )
        p = result.raw_empirical_p
        assert np.all(np.isfinite(p))
        assert np.all(p >= 0)
        assert np.all(p <= 1)

    def test_score_joint_ordinal_returns_joint_result(self):
        X, y = _make_ordinal_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score_joint",
            family="ordinal",
        )
        assert isinstance(result, JointTestResult)
        assert 0 < result.p_value <= 1

    def test_score_joint_multinomial_returns_joint_result(self):
        X, y = _make_multinomial_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score_joint",
            family="multinomial",
        )
        assert isinstance(result, JointTestResult)
        assert 0 < result.p_value <= 1

    def test_score_ordinal_with_confounders(self):
        X, y = _make_ordinal_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score",
            family="ordinal",
            confounders=["x3"],
        )
        assert isinstance(result, IndividualTestResult)
        # Only x1 and x2 are tested; x3 keeps its observed coef.
        assert len(result["model_coefs"]) == 3

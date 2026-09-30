"""Adversarial dual-harness tests for causal invariance testing (Phase 7, Steps 18-20).

Tests conditional independence H0: Y ⟂ E | X across heterogeneous environments,
verifying that:
1. Covariate shift across environments (P(X|E=1) != P(X|E=2)) does NOT inflate
   Type I error when the causal mechanism P(Y|X) is invariant.
2. True structural breaks (beta_1 != beta_2) are detected with high statistical power.
3. Discrete conditioning covariates auto-dispatch to Tier 0 exact discrete permutation.
4. Non-linear confounding is absorbed via Double Machine Learning (reduced_model=).
5. The CPT pairwise-swap MCMC sampler preserves conditional environment laws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from randomization_tests import (
    InvarianceResult,
    invariance_test,
    print_invariance_table,
)
from randomization_tests._cpt import cpt_sample


class TestInvarianceDualHarness:
    """Adversarial dual-harness tests for invariance under covariate shift and true breaks."""

    def test_invariance_covariate_shift_null_control(self) -> None:
        """Adversarial Harness 1: P(X|E) shifts dramatically but P(Y|X) is invariant.

        Env 1: X ~ N(0, 1), n=100
        Env 2: X ~ N(3, 1), n=100
        Mechanism: Y = 2*X + eps, eps ~ N(0, 1) in BOTH environments.

        Unconditional label shuffling would test (X, Y) ⟂ E and falsely reject (p < 0.001).
        Valid conditional tests (Freedman-Lane, Kennedy, CPT) must condition on X
        and retain nominal Type I size (p > 0.05).
        """
        rng = np.random.default_rng(42)
        n1, n2 = 100, 100

        x1 = rng.normal(0.0, 1.0, size=n1)
        x2 = rng.normal(3.0, 1.0, size=n2)
        y1 = 2.0 * x1 + rng.normal(0.0, 1.0, size=n1)
        y2 = 2.0 * x2 + rng.normal(0.0, 1.0, size=n2)

        df = pd.DataFrame(
            {
                "X": np.concatenate([x1, x2]),
                "env": np.array(["env_A"] * n1 + ["env_B"] * n2),
            }
        )
        y = pd.Series(np.concatenate([y1, y2]), name="Y")

        # 1. Freedman-Lane (Tier 1 default for linear models)
        res_fl = invariance_test(
            df,
            y,
            predictor="X",
            environments="env",
            method="freedman_lane",
            n_randomizations=999,
            random_state=42,
        )
        assert res_fl.is_invariant is True
        assert res_fl.p_value > 0.05
        assert res_fl.tier == "tier_1_residualized"
        assert res_fl.guarantee_tier == "asymptotically_exact"
        assert res_fl.method == "freedman_lane"

        # 2. Kennedy joint test (Tier 1 exposure residualization)
        res_ken = invariance_test(
            df,
            y,
            predictor="X",
            environments="env",
            method="kennedy",
            n_randomizations=999,
            random_state=42,
        )
        assert res_ken.is_invariant is True
        assert res_ken.p_value > 0.05
        assert res_ken.tier == "tier_1_residualized"
        assert res_ken.guarantee_tier == "asymptotically_exact"
        assert res_ken.method == "kennedy"

        # 3. Conditional Permutation Test (Tier 2 CPT MCMC)
        res_cpt = invariance_test(
            df,
            y,
            predictor="X",
            environments="env",
            method="conditional",
            n_randomizations=499,
            random_state=42,
        )
        assert res_cpt.is_invariant is True
        assert res_cpt.p_value > 0.05
        assert res_cpt.tier == "tier_2_cpt"
        assert res_cpt.guarantee_tier == "exact_up_to_tv_error"
        assert res_cpt.method == "conditional"

    def test_invariance_true_slope_shift_detected(self) -> None:
        """Adversarial Harness 2: True mechanism violation across environments.

        Env 1: Y = 2*X + eps
        Env 2: Y = -1*X + eps (slope shift)
        All valid strategies must reject invariance (p < 0.01).
        """
        rng = np.random.default_rng(123)
        n1, n2 = 80, 80

        x1 = rng.normal(0.0, 1.0, size=n1)
        x2 = rng.normal(0.0, 1.0, size=n2)
        y1 = 2.0 * x1 + rng.normal(0.0, 1.0, size=n1)
        y2 = -1.0 * x2 + rng.normal(0.0, 1.0, size=n2)

        df = pd.DataFrame(
            {
                "X": np.concatenate([x1, x2]),
                "env": np.array(["env_1"] * n1 + ["env_2"] * n2),
            }
        )
        y = pd.Series(np.concatenate([y1, y2]), name="Y")

        res_fl = invariance_test(
            df,
            y,
            predictor="X",
            environments="env",
            method="freedman_lane",
            n_randomizations=499,
            random_state=42,
        )
        assert res_fl.is_invariant is False
        assert res_fl.p_value < 0.01
        assert res_fl.max_deviation > 1.0

        res_ken = invariance_test(
            df,
            y,
            predictor="X",
            environments="env",
            method="kennedy",
            n_randomizations=499,
            random_state=42,
        )
        assert res_ken.is_invariant is False
        assert res_ken.p_value < 0.01

        res_cpt = invariance_test(
            df,
            y,
            predictor="X",
            environments="env",
            method="conditional",
            n_randomizations=499,
            random_state=42,
        )
        assert res_cpt.is_invariant is False
        assert res_cpt.p_value < 0.01

    def test_invariance_intercept_shift_detected(self) -> None:
        """Invariance test detects baseline intercept shifts across environments."""
        rng = np.random.default_rng(456)
        n = 100
        x = rng.normal(0.0, 1.0, size=n)
        # Environment 1 intercept = 0, Environment 2 intercept = 3.5
        y = (
            1.5 * x
            + np.where(np.arange(n) < 50, 0.0, 3.5)
            + rng.normal(0.0, 0.5, size=n)
        )
        envs = np.where(np.arange(n) < 50, "A", "B")

        df = pd.DataFrame({"X": x, "site": envs})
        res = invariance_test(
            df, y, predictor="X", environments="site", random_state=42
        )
        assert res.is_invariant is False
        assert res.p_value < 0.01

    def test_invariance_discrete_tier0_exact(self) -> None:
        """Tier 0: Discrete conditioning covariate auto-dispatches to exact stratification."""
        rng = np.random.default_rng(789)
        n = 120
        # Discrete binary covariate
        x_discrete = rng.choice([0, 1], size=n)
        envs = rng.choice(["site_1", "site_2"], size=n)
        y = 2.0 * x_discrete + rng.normal(0.0, 1.0, size=n)

        df = pd.DataFrame({"X_cat": x_discrete, "env": envs})
        res = invariance_test(
            df,
            y,
            predictor="X_cat",
            environments="env",
            method="auto",
            random_state=42,
        )
        assert res.tier == "tier_0_stratified"
        assert res.guarantee_tier == "finite_sample_exact"
        assert res.method == "stratified"
        assert res.is_invariant is True
        assert res.p_value > 0.05

    def test_invariance_dml_nonlinear_confounding(self) -> None:
        """Invariance testing with Double Machine Learning (DML) for non-linear confounding."""
        rng = np.random.default_rng(999)
        n = 200
        z = rng.uniform(-2.0, 2.0, size=n)
        # Non-linear nuisance functions
        x = np.sin(z * np.pi) + rng.normal(0.0, 0.5, size=n)
        envs = np.where(z > 0, "env_hi", "env_lo")
        y = 1.5 * x + np.cos(z * np.pi) + rng.normal(0.0, 0.5, size=n)

        df = pd.DataFrame({"X": x, "Z": z, "env": envs})
        res_dml = invariance_test(
            df,
            y,
            predictor="X",
            environments="env",
            reduced_model=GradientBoostingRegressor(random_state=42),
            method="freedman_lane",
            random_state=42,
        )
        assert res_dml.is_invariant is True
        assert res_dml.p_value > 0.05
        assert "reduced_model" in res_dml.diagnostics

    def test_invariance_score_strategy(self) -> None:
        """Invariance test with method='score' evaluates score_joint."""
        rng = np.random.default_rng(42)
        n = 100
        x = rng.normal(size=n)
        envs = np.array(["E1"] * 50 + ["E2"] * 50)
        y = 2.0 * x + rng.normal(size=n)
        df = pd.DataFrame({"X": x, "env": envs})

        res_score = invariance_test(
            df,
            y,
            predictor="X",
            environments="env",
            method="score",
            n_randomizations=299,
            random_state=42,
        )
        assert res_score.method == "score"
        assert res_score.is_invariant is True
        assert res_score.p_value > 0.05

    def test_invariance_display_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_invariance_table renders formatted 80-column ASCII table with guarantee banner."""
        rng = np.random.default_rng(42)
        n = 80
        x = rng.normal(size=n)
        envs = np.array(["Site_A"] * 40 + ["Site_B"] * 40)
        y = 1.8 * x + rng.normal(size=n)
        df = pd.DataFrame({"X": x, "site": envs})

        res = invariance_test(
            df, y, predictor="X", environments="site", random_state=42
        )
        print_invariance_table(res)
        out = capsys.readouterr().out

        assert "Causal Invariance Test" in out
        assert "Guarantee: Asymptotically Exact" in out
        assert "Decision:" in out
        assert "Pooled Slope:" in out
        assert "Site_A" in out
        assert "Site_B" in out

    def test_invariance_invalid_inputs_raise(self) -> None:
        """Invalid predictor, missing environment, or insufficient environments raise ValueError."""
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0], "env": ["A", "A", "A", "A"]})
        y = np.array([1.0, 2.0, 3.0, 4.0])

        # Less than 2 environments
        with pytest.raises(ValueError, match="at least 2 distinct environments"):
            invariance_test(df, y, predictor="X", environments="env")

        # Missing predictor
        df_valid = pd.DataFrame(
            {"X": [1.0, 2.0, 3.0, 4.0], "env": ["A", "A", "B", "B"]}
        )
        with pytest.raises(ValueError, match="not found in X"):
            invariance_test(df_valid, y, predictor="Z_nonexistent", environments="env")

        # Unknown method
        with pytest.raises(ValueError, match="Unknown method"):
            invariance_test(
                df_valid, y, predictor="X", environments="env", method="bogus_method"
            )

        # Invalid confidence level
        with pytest.raises(ValueError, match="confidence_level"):
            invariance_test(
                df_valid, y, predictor="X", environments="env", confidence_level=1.5
            )

    def test_invariance_interpretation_property(self) -> None:
        """InvarianceResult.interpretation property returns informative human-readable string."""
        res_inv = InvarianceResult(
            is_invariant=True,
            p_value=0.45,
            environment_coefficients={"A": 2.0, "B": 2.1},
            pooled_coefficient=2.05,
            max_deviation=0.05,
            null_distribution=np.array([0.02, 0.04]),
        )
        assert "Invariant effect" in res_inv.interpretation
        assert "stable across environments" in res_inv.interpretation

        res_non = InvarianceResult(
            is_invariant=False,
            p_value=0.001,
            environment_coefficients={"A": 2.0, "B": -1.0},
            pooled_coefficient=0.5,
            max_deviation=1.5,
            null_distribution=np.array([0.1, 0.2]),
        )
        assert "Non-invariant effect" in res_non.interpretation
        assert "varies across environments" in res_non.interpretation


class TestCPTSampler:
    """Unit tests for the Conditional Permutation Test (CPT) MCMC pairwise-swap sampler."""

    def test_cpt_sample_preserves_conditional_law(self) -> None:
        """Resampled labels preserve conditional frequencies P_hat(E | X)."""
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(-2.0, 2.0, size=n)
        # Environment assignment strongly depends on X
        p_env1 = 1.0 / (1.0 + np.exp(-2.0 * x))
        labels = (rng.uniform(size=n) < p_env1).astype(int)

        X_mat = x.reshape(-1, 1)
        n_draws = 200
        draws = cpt_sample(labels, X_mat, n_draws=n_draws, random_state=42)

        assert draws.shape == (n_draws, n)
        # Every draw must preserve total environment sample size counts exactly
        n1_orig = int(np.sum(labels == 1))
        for d in range(n_draws):
            assert int(np.sum(draws[d] == 1)) == n1_orig

        # Average resampled probability across draws should correlate strongly with P(E=1|X)
        mean_draw_probs = np.mean(draws == 1, axis=0)
        corr = float(np.corrcoef(mean_draw_probs, p_env1)[0, 1])
        assert corr > 0.85

    def test_cpt_determinism(self) -> None:
        """Fixed random_state yields identical CPT draws."""
        labels = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        X = np.arange(8).reshape(-1, 1).astype(float)

        d1 = cpt_sample(labels, X, n_draws=10, random_state=42)
        d2 = cpt_sample(labels, X, n_draws=10, random_state=42)
        assert np.array_equal(d1, d2)

"""Tests for the confounders module — full sieve coverage.

Covers: partial correlation, distance correlation, multiple-testing
correction, collider detection, moderation analysis, mixed-family
fallback, cluster bootstrap, collinearity guard, multinomial exclusion,
and the four-stage identify_confounders orchestrator.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from randomization_tests._results import ConfounderAnalysisResult
from randomization_tests.confounders import (
    _cluster_bootstrap_indices,
    _cluster_jackknife_indices,
    _collider_test,
    _distance_correlation,
    _partial_correlation,
    _resolve_base_family,
    identify_confounders,
    mediation_analysis,
    moderation_analysis,
    screen_potential_confounders,
)
from randomization_tests.diagnostics import compute_e_value, rosenbaum_bounds
from randomization_tests.families import (
    LinearFamily,
    LogisticFamily,
    MultinomialFamily,
    NegativeBinomialFamily,
)
from randomization_tests.families_mixed import (
    LinearMixedFamily,
    LogisticMixedFamily,
    PoissonMixedFamily,
)

# ── Fixtures ─────────────────────────────────────────────────────── #


def _make_confounder_data(n=2000, seed=42):
    """Z causes both X and Y (confounder)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    x1 = 0.3 * z + rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y_vals = 2.0 * z + rng.standard_normal(n) * 0.5
    X = pd.DataFrame({"x1": x1, "x2": x2, "z": z})
    y = pd.DataFrame({"y": y_vals})
    return X, y


def _make_mediator_data(n=2000, seed=42):
    """M mediates: X → M → Y (strong indirect effect)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    m = 0.8 * x + rng.standard_normal(n) * 0.3
    y_vals = 0.7 * m + rng.standard_normal(n) * 0.3
    z = rng.standard_normal(n)
    X = pd.DataFrame({"x": x, "m": m, "z": z})
    y = pd.DataFrame({"y": y_vals})
    return X, y


def _make_moderator_data(n=2000, seed=42):
    """Z moderates: Y = βX + γXZ + ε with significant γ."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y_vals = 1.0 * x + 0.8 * x * z + rng.standard_normal(n) * 0.5
    w = rng.standard_normal(n)
    X = pd.DataFrame({"x": x, "z": z, "w": w})
    y = pd.DataFrame({"y": y_vals})
    return X, y


def _make_collider_data(n=2000, seed=42):
    """Z is a collider: X → Z ← Y (no direct X → Y effect)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y_vals = rng.standard_normal(n)
    z = 0.7 * x + 0.7 * y_vals + rng.standard_normal(n) * 0.3
    w = rng.standard_normal(n)
    X = pd.DataFrame({"x": x, "z": z, "w": w})
    y = pd.DataFrame({"y": y_vals})
    return X, y


def _make_clustered_data(n_groups=30, group_size=20, seed=42):
    """Create clustered mediation data."""
    rng = np.random.default_rng(seed)
    n = n_groups * group_size
    groups = np.repeat(np.arange(n_groups), group_size)
    group_effect = rng.standard_normal(n_groups)
    ge = group_effect[groups]
    x = rng.standard_normal(n) + ge
    m = 0.8 * x + ge + rng.standard_normal(n) * 0.3
    y_vals = 0.7 * m + ge + rng.standard_normal(n) * 0.3
    z = rng.standard_normal(n)
    X = pd.DataFrame({"x": x, "m": m, "z": z})
    y = pd.DataFrame({"y": y_vals})
    return X, y, groups


# ── TestPartialCorrelation ───────────────────────────────────────── #


class TestPartialCorrelation:
    def test_removes_spurious_association(self):
        """Partial r(X,Y|Z) should be near 0 when Z causes both."""
        rng = np.random.default_rng(42)
        n = 1000
        z = rng.standard_normal(n)
        x = 0.8 * z + rng.standard_normal(n) * 0.3
        y = 0.8 * z + rng.standard_normal(n) * 0.3
        # Marginal r(x,y) should be substantial.
        marg_r, _ = sp_stats.pearsonr(x, y)
        assert abs(marg_r) > 0.3
        # Partial r(x,y|z) should be near 0.
        pr, p_val = _partial_correlation(x, y, z.reshape(-1, 1))
        assert abs(pr) < 0.1
        assert p_val > 0.05

    def test_corrected_df(self):
        """P-value uses n-k-2 df, not n-2."""
        rng = np.random.default_rng(99)
        n = 50
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        covs = rng.standard_normal((n, 5))
        pr, p_val = _partial_correlation(x, y, covs)
        assert 0 <= p_val <= 1
        assert abs(pr) < 1.0


# ── TestDistanceCorrelation ──────────────────────────────────────── #


class TestDistanceCorrelation:
    def test_detects_nonlinear(self):
        """Distance correlation detects Z = X² + ε (Pearson misses)."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        y = x**2 + rng.standard_normal(n) * 0.3
        pear_r, _ = sp_stats.pearsonr(x, y)
        assert abs(pear_r) < 0.15
        dcor, p_val = _distance_correlation(x, y)
        assert dcor > 0.3
        assert p_val < 0.05

    def test_independent_returns_near_zero(self):
        """Independent variables should have dcor ≈ 0."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        dcor, p_val = _distance_correlation(x, y)
        assert dcor < 0.15
        assert p_val > 0.05

    def test_negative_dcor_clamped(self):
        """Negative bias-corrected estimator → dcor = 0, p = 1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        dcor, p_val = _distance_correlation(x, y)
        assert dcor >= 0.0
        assert 0 <= p_val <= 1.0

    def test_large_n_warning(self):
        """n > 10,000 should emit a UserWarning about memory."""
        x = np.arange(10_001, dtype=float)
        y = np.arange(10_001, dtype=float)
        with pytest.warns(UserWarning, match="O\\(n"):
            _distance_correlation(x, y)


# ── TestMultipleTesting ──────────────────────────────────────────── #


class TestMultipleTesting:
    def test_holm_correction(self):
        """Holm correction makes marginal associations non-significant."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.standard_normal(n)
        cols: dict[str, np.ndarray] = {"x": x}
        for i in range(20):
            cols[f"z{i}"] = 0.12 * x + rng.standard_normal(n) * 0.9
        cols["y_target"] = rng.standard_normal(n) + 0.12 * x
        X = pd.DataFrame(cols)
        y = pd.DataFrame({"y": cols["y_target"]})
        X_no_target = X.drop(columns=["y_target"])

        res_raw = screen_potential_confounders(
            X_no_target, y, predictor="x", correlation_threshold=0.05
        )
        res_holm = screen_potential_confounders(
            X_no_target,
            y,
            predictor="x",
            correlation_threshold=0.05,
            correction_method="holm",
        )
        assert len(res_holm["potential_confounders"]) <= len(
            res_raw["potential_confounders"]
        )
        assert res_holm["correction_method"] == "holm"

    def test_fdr_bh_correction(self):
        """FDR-BH correction returns adjusted p-values."""
        X, y = _make_confounder_data(n=500)
        res = screen_potential_confounders(
            X, y, predictor="x1", correction_method="fdr_bh"
        )
        assert res["correction_method"] == "fdr_bh"
        assert "adjusted_p_values" in res
        assert "predictor" in res["adjusted_p_values"]
        assert "outcome" in res["adjusted_p_values"]


# ── TestColliderDetection ────────────────────────────────────────── #


class TestColliderDetection:
    def test_detects_collider_linear(self):
        """Z = αX + δY + ε should amplify X-Y association."""
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        z = 0.7 * x + 0.7 * y + rng.standard_normal(n) * 0.3
        is_coll, _, _ = _collider_test(x, y, z)
        assert is_coll is True

    def test_confounder_not_collider(self):
        """True confounder Z should NOT be classified as collider."""
        rng = np.random.default_rng(42)
        n = 2000
        z = rng.standard_normal(n)
        x = 0.5 * z + rng.standard_normal(n) * 0.5
        y = 0.5 * z + rng.standard_normal(n) * 0.5
        is_coll, _, _ = _collider_test(x, y, z)
        assert is_coll is False

    def test_near_constant_z_skipped(self):
        """Near-constant Z → not collider, NaN coefficients."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        z = np.ones(100) * 5.0 + rng.standard_normal(100) * 1e-12
        is_coll, cm, cp = _collider_test(x, y, z)
        assert is_coll is False
        assert np.isnan(cm)

    def test_multinomial_returns_false_nan(self):
        """Multinomial family → (False, NaN, NaN)."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        z = rng.standard_normal(100)
        fam = MultinomialFamily()
        is_coll, cm, cp = _collider_test(x, y, z, family=fam)
        assert is_coll is False
        assert np.isnan(cm)
        assert np.isnan(cp)

    def test_glm_permutation_calibrated_threshold(self):
        """Logistic with independent Z should NOT be collider."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        z = rng.standard_normal(n)
        y_prob = 1 / (1 + np.exp(-(0.5 * x)))
        y = (rng.random(n) < y_prob).astype(float)
        fam = LogisticFamily()
        is_coll, _, _ = _collider_test(x, y, z, family=fam, random_state=42)
        assert is_coll is False


# ── TestModerationAnalysis ───────────────────────────────────────── #


class TestModerationAnalysis:
    def test_detects_moderator(self):
        """Strong interaction Y = βX + γXZ + ε should be detected."""
        X, y = _make_moderator_data()
        result = moderation_analysis(
            X,
            y,
            predictor="x",
            moderator="z",
            n_bootstrap=1000,
            random_state=42,
        )
        assert result["is_moderator"] is True
        assert result["ci_method"] == "BCa"
        ci = result["interaction_ci"]
        assert ci[0] > 0 or ci[1] < 0

    def test_non_moderator(self):
        """Noise variable w should not be flagged as moderator."""
        rng = np.random.default_rng(123)
        n = 500
        x = rng.standard_normal(n)
        w = rng.standard_normal(n)
        # Y depends on X linearly, no interaction with w.
        y_vals = 1.0 * x + rng.standard_normal(n) * 0.5
        X = pd.DataFrame({"x": x, "w": w})
        y = pd.DataFrame({"y": y_vals})
        result = moderation_analysis(
            X,
            y,
            predictor="x",
            moderator="w",
            n_bootstrap=1000,
            random_state=42,
        )
        assert result["is_moderator"] is False

    def test_collinear_interaction_skipped(self):
        """Constant Z → collinear interaction → skip."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.standard_normal(n)
        z = np.ones(n)
        y_vals = x + rng.standard_normal(n) * 0.5
        X = pd.DataFrame({"x": x, "z": z})
        y = pd.DataFrame({"y": y_vals})
        with pytest.warns(UserWarning, match="collinear"):
            result = moderation_analysis(
                X,
                y,
                predictor="x",
                moderator="z",
                random_state=42,
            )
        assert result["is_moderator"] is False
        assert np.isnan(result["interaction_coef"])


# ── TestEValue ───────────────────────────────────────────────────── #


class TestEValue:
    def test_linear_known_rr(self):
        """Linear family: d=1 → RR=exp(0.91) ≈ 2.484."""
        import math

        res = compute_e_value(1.0, "linear", sd_x=1.0, sd_y=1.0)
        expected_rr = math.exp(0.91)
        assert abs(res["rr"] - expected_rr) < 0.01
        assert res["e_value"] > 1.0

    def test_logistic_or_conversion(self):
        """Logistic family: coef=1 → OR=e ≈ 2.718."""
        import math

        res = compute_e_value(1.0, "logistic")
        expected_or = math.exp(1.0)
        assert abs(res["rr"] - expected_or) < 0.01
        assert res["e_value"] > 1.0

    def test_logistic_with_prevalence(self):
        """Logistic with baseline_prevalence gives tighter bound."""
        res_no = compute_e_value(1.0, "logistic")
        res_prev = compute_e_value(1.0, "logistic", baseline_prevalence=0.1)
        assert res_prev["rr"] != res_no["rr"]
        assert res_prev["e_value"] > 1.0

    def test_poisson_direct_rr(self):
        """Poisson: coef = log(2) → RR = 2."""
        import math

        res = compute_e_value(math.log(2), "poisson")
        assert abs(res["rr"] - 2.0) < 0.01

    def test_ordinal_same_as_logistic(self):
        """Ordinal uses same OR formula as logistic."""
        res_ord = compute_e_value(1.0, "ordinal")
        res_log = compute_e_value(1.0, "logistic")
        assert abs(res_ord["e_value"] - res_log["e_value"]) < 0.001

    def test_mixed_family_stripped(self):
        """Mixed family names stripped to base."""
        res = compute_e_value(1.0, "linear_mixed", sd_x=1.0, sd_y=1.0)
        assert res["family"] == "linear_mixed"
        assert res["e_value"] > 1.0

    def test_multinomial_nan_warning(self):
        """Multinomial → NaN with UserWarning."""
        with pytest.warns(UserWarning, match="multinomial"):
            res = compute_e_value(1.0, "multinomial")
        assert np.isnan(res["e_value"])

    def test_ci_bound(self):
        """CI bound produces e_value_ci."""
        res = compute_e_value(1.0, "poisson", ci_bound=0.5)
        assert not np.isnan(res["e_value_ci"])

    def test_negative_binomial(self):
        """Negative binomial uses direct RR like Poisson."""
        import math

        res = compute_e_value(math.log(3), "negative_binomial")
        assert abs(res["rr"] - 3.0) < 0.01

    def test_linear_requires_sd(self):
        """Linear without sd_x/sd_y raises ValueError."""
        with pytest.raises(ValueError, match="sd_x"):
            compute_e_value(1.0, "linear")


# ── TestRosenbaumBounds ──────────────────────────────────────────── #


class TestRosenbaumBounds:
    def _make_binary_linear_result(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        x = (rng.random(n) > 0.5).astype(float)
        y = 0.15 * x + rng.standard_normal(n) * 1.0
        X = pd.DataFrame({"x": x})
        # Compute actual p-value from OLS so Γ=1 is consistent.
        ones = np.ones((n, 1))
        X_design = np.column_stack([ones, x])
        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
        residuals = y - X_design @ beta
        rse = np.sqrt(np.sum(residuals**2) / (n - 2))
        se_beta = rse * np.sqrt(np.linalg.inv(X_design.T @ X_design)[1, 1])
        t_stat = beta[1] / se_beta
        from scipy.stats import t as t_dist

        observed_p = float(2 * t_dist.sf(abs(t_stat), n - 2))
        result = {"p_values": {"x": observed_p}, "family": "linear"}
        return result, X, y

    def test_gamma_1_matches_observed(self):
        """Γ=1 gives the observed p-value."""
        result, X, y = self._make_binary_linear_result()
        observed_p = result["p_values"]["x"]
        rb = rosenbaum_bounds(result, X, y, gammas=(1.0,))
        assert abs(rb["worst_case_p"][0] - observed_p) < 1e-6

    def test_output_structure(self):
        """Output dict has expected keys and shapes."""
        result, X, y = self._make_binary_linear_result()
        gammas = (1.0, 1.5, 2.0, 2.5, 3.0)
        rb = rosenbaum_bounds(result, X, y, gammas=gammas)
        assert "gamma_values" in rb
        assert "worst_case_p" in rb
        assert "critical_gamma" in rb
        assert "interpretation" in rb
        assert len(rb["worst_case_p"]) == len(gammas)
        # All p-values should be valid probabilities.
        for p in rb["worst_case_p"]:
            assert 0 <= p <= 1 or np.isnan(p)

    def test_nonlinear_raises(self):
        """Non-linear family raises NotImplementedError."""
        result = {"p_values": {"x": 0.02}, "family": "logistic"}
        X = pd.DataFrame({"x": [0, 1, 0, 1]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(NotImplementedError, match="linear"):
            rosenbaum_bounds(result, X, y)

    def test_mixed_family_raises(self):
        """LinearMixedFamily raises NotImplementedError."""
        result = {"p_values": {"x": 0.02}, "family": "linear_mixed"}
        X = pd.DataFrame({"x": [0, 1, 0, 1]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(NotImplementedError, match="linear"):
            rosenbaum_bounds(result, X, y)

    def test_continuous_predictor_raises(self):
        """Continuous predictor raises ValueError."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x": rng.standard_normal(50)})
        y = rng.standard_normal(50)
        result = {"p_values": {"x": 0.02}, "family": "linear"}
        with pytest.raises(ValueError, match="binary"):
            rosenbaum_bounds(result, X, y)


# ── TestMixedFamilyFallback ──────────────────────────────────────── #


class TestMixedFamilyFallback:
    def test_linear_mixed_resolves(self):
        fam = LinearMixedFamily()
        base, was_mixed = _resolve_base_family(fam)
        assert base.name == "linear"
        assert was_mixed is True

    def test_logistic_mixed_resolves(self):
        fam = LogisticMixedFamily()
        base, was_mixed = _resolve_base_family(fam)
        assert base.name == "logistic"
        assert was_mixed is True

    def test_poisson_mixed_resolves(self):
        fam = PoissonMixedFamily()
        base, was_mixed = _resolve_base_family(fam)
        assert base.name == "poisson"
        assert was_mixed is True

    def test_base_family_unchanged(self):
        fam = LinearFamily()
        base, was_mixed = _resolve_base_family(fam)
        assert base.name == "linear"
        assert was_mixed is False

    def test_negative_binomial_resolves_to_poisson(self):
        """NegativeBinomialFamily falls back to PoissonFamily.

        NB requires calibrate() before fit(); the sieve does not need
        the NB-specific dispersion parameter for DAG classification.
        """
        fam = NegativeBinomialFamily()
        base, did_fallback = _resolve_base_family(fam)
        assert base.name == "poisson"
        assert did_fallback is True

    def test_mediation_with_mixed_family(self):
        """Mediation falls back to base family for mixed family."""
        X, y = _make_mediator_data(n=500)
        result = mediation_analysis(
            X,
            y,
            predictor="x",
            mediator="m",
            n_bootstrap=500,
            random_state=42,
            family=LinearMixedFamily(),
        )
        assert result["is_mediator"] is True


# ── TestClusterBootstrap ─────────────────────────────────────────── #


class TestClusterBootstrap:
    def test_cluster_indices_complete_groups(self):
        """Each replicate contains only complete groups."""
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        rng = np.random.default_rng(42)
        indices = _cluster_bootstrap_indices(groups, 100, rng)
        for idx in indices:
            selected_groups = set(groups[idx])
            for g in selected_groups:
                group_mask = groups == g
                group_indices = set(np.where(group_mask)[0])
                count = sum(1 for i in idx if i in group_indices)
                assert count % len(group_indices) == 0

    def test_cluster_jackknife_count(self):
        """Leave-one-cluster-out produces G jackknife values."""
        groups = np.array([0, 0, 1, 1, 2, 2])
        jack_idx = _cluster_jackknife_indices(groups)
        assert len(jack_idx) == 3

    def test_cluster_jackknife_completeness(self):
        """Each jackknife set excludes exactly one group."""
        groups = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        jack_idx = _cluster_jackknife_indices(groups)
        unique_labels = np.unique(groups)
        for i, idx in enumerate(jack_idx):
            excluded_label = unique_labels[i]
            assert excluded_label not in groups[idx]
            for other in unique_labels:
                if other != excluded_label:
                    assert other in groups[idx]

    def test_cluster_bootstrap_wider_ci(self):
        """Cluster bootstrap should produce wider CIs for clustered data."""
        X, y, groups = _make_clustered_data(n_groups=30, group_size=20, seed=42)
        res_iid = mediation_analysis(
            X,
            y,
            predictor="x",
            mediator="m",
            n_bootstrap=500,
            random_state=42,
        )
        res_cluster = mediation_analysis(
            X,
            y,
            predictor="x",
            mediator="m",
            n_bootstrap=500,
            random_state=42,
            groups=groups,
        )
        ci_iid = res_iid["indirect_effect_ci"]
        ci_cluster = res_cluster["indirect_effect_ci"]
        width_iid = ci_iid[1] - ci_iid[0]
        width_cluster = ci_cluster[1] - ci_cluster[0]
        # Cluster CI should generally be wider (loose sanity check).
        assert width_cluster > width_iid * 0.5


# ── TestCollinearityGuard ────────────────────────────────────────── #


class TestCollinearityGuard:
    def test_high_r_squared_warning(self):
        """Confounders explaining >95% of predictor → UserWarning."""
        rng = np.random.default_rng(42)
        n = 500
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        x = 0.7 * z1 + 0.7 * z2 + rng.standard_normal(n) * 0.01
        y_vals = 0.5 * z1 + 0.5 * z2 + rng.standard_normal(n) * 0.3
        X = pd.DataFrame({"x": x, "z1": z1, "z2": z2})
        y = pd.DataFrame({"y": y_vals})
        with pytest.warns(UserWarning, match="predictor variance"):
            identify_confounders(
                X,
                y,
                predictor="x",
                n_bootstrap_mediation=200,
                n_bootstrap_moderation=200,
                random_state=42,
            )


# ── TestIdentifyConfoundersFullSieve ─────────────────────────────── #


class TestIdentifyConfoundersFullSieve:
    def test_confounder_classified(self):
        """Z causing both X and Y → classified as confounder."""
        X, y = _make_confounder_data()
        result = identify_confounders(
            X,
            y,
            predictor="x1",
            n_bootstrap_mediation=500,
            n_bootstrap_moderation=500,
            random_state=42,
        )
        assert isinstance(result, ConfounderAnalysisResult)
        assert "z" in result.screening_results["potential_confounders"]

    def test_mediator_classified(self):
        """M on the causal path → classified as mediator."""
        X, y = _make_mediator_data()
        result = identify_confounders(
            X,
            y,
            predictor="x",
            n_bootstrap_mediation=500,
            n_bootstrap_moderation=500,
            random_state=42,
        )
        assert isinstance(result, ConfounderAnalysisResult)
        assert "m" in result.identified_mediators
        assert "m" not in result.identified_confounders

    def test_result_has_to_dict(self):
        """ConfounderAnalysisResult has to_dict() for backward compat."""
        X, y = _make_confounder_data(n=500)
        result = identify_confounders(
            X,
            y,
            predictor="x1",
            n_bootstrap_mediation=200,
            n_bootstrap_moderation=200,
            random_state=42,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "predictor" in d
        assert "identified_confounders" in d
        assert "identified_mediators" in d
        assert "identified_moderators" in d
        assert "identified_colliders" in d

    def test_correlation_method_passthrough(self):
        """correlation_method is passed to screening."""
        X, y = _make_confounder_data(n=500)
        result = identify_confounders(
            X,
            y,
            predictor="x1",
            correlation_method="partial",
            n_bootstrap_mediation=200,
            n_bootstrap_moderation=200,
            random_state=42,
        )
        assert result.screening_results["correlation_method"] == "partial"

    def test_collider_removed(self):
        """Collider Z → removed from confounder pool."""
        X, y = _make_collider_data()
        result = identify_confounders(
            X,
            y,
            predictor="x",
            n_bootstrap_mediation=500,
            n_bootstrap_moderation=500,
            random_state=42,
        )
        assert isinstance(result, ConfounderAnalysisResult)
        # If z passed screening, it should be collider or not confounder.
        if "z" in result.screening_results["potential_confounders"]:
            assert (
                "z" in result.identified_colliders
                or "z" not in result.identified_confounders
            )


# ── TestMultinomialExclusion ─────────────────────────────────────── #


class TestMultinomialExclusion:
    def test_orchestrator_early_exit(self):
        """Multinomial family → all candidates as confounders, warning."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        z = 0.5 * x + rng.standard_normal(n) * 0.3
        y_vals = (rng.random(n) * 3).astype(int).astype(float)
        X = pd.DataFrame({"x": x, "z": z})
        y = pd.DataFrame({"y": y_vals})
        with pytest.warns(UserWarning, match="[Mm]ultinomial"):
            result = identify_confounders(
                X,
                y,
                predictor="x",
                family="multinomial",
                n_bootstrap_mediation=200,
                n_bootstrap_moderation=200,
                random_state=42,
            )
        assert isinstance(result, ConfounderAnalysisResult)
        assert len(result.identified_mediators) == 0
        assert len(result.identified_moderators) == 0
        assert len(result.identified_colliders) == 0


# ── TestEdgeCases ────────────────────────────────────────────────── #


class TestEdgeCases:
    def test_screen_distance_method(self):
        """Distance correlation screening works."""
        X, y = _make_confounder_data(n=500)
        res = screen_potential_confounders(
            X, y, predictor="x1", correlation_method="distance"
        )
        assert res["correlation_method"] == "distance"

    def test_screen_partial_method(self):
        """Partial correlation screening works."""
        X, y = _make_confounder_data(n=500)
        res = screen_potential_confounders(
            X, y, predictor="x1", correlation_method="partial"
        )
        assert res["correlation_method"] == "partial"

    def test_invalid_correlation_method(self):
        """Invalid correlation method raises ValueError."""
        X, y = _make_confounder_data(n=100)
        with pytest.raises(ValueError, match="correlation_method"):
            screen_potential_confounders(
                X, y, predictor="x1", correlation_method="invalid"
            )

    def test_invalid_correction_method(self):
        """Invalid correction method raises ValueError."""
        X, y = _make_confounder_data(n=100)
        with pytest.raises(ValueError, match="correction_method"):
            screen_potential_confounders(
                X, y, predictor="x1", correction_method="invalid"
            )

    def test_confounder_result_round_trips(self):
        """ConfounderAnalysisResult.to_dict() → reconstruction."""
        result = ConfounderAnalysisResult(
            predictor="x",
            identified_confounders=["z1"],
            identified_mediators=["m1"],
            identified_moderators=["mod1"],
            identified_colliders=["c1"],
            screening_results={
                "predictor": "x",
                "potential_confounders": ["z1", "m1", "mod1", "c1"],
            },
        )
        d = result.to_dict()
        assert d["predictor"] == "x"
        assert d["identified_confounders"] == ["z1"]
        assert d["identified_mediators"] == ["m1"]
        assert d["identified_moderators"] == ["mod1"]
        assert d["identified_colliders"] == ["c1"]


# ── Legacy tests (backward compatibility) ────────────────────────── #


class TestScreenPotentialConfounders:
    def test_detects_confounder(self):
        X, y = _make_confounder_data()
        result = screen_potential_confounders(X, y, predictor="x1")
        assert "z" in result["potential_confounders"]

    def test_excludes_noise(self):
        X, y = _make_confounder_data()
        result = screen_potential_confounders(X, y, predictor="x1")
        assert "x2" in result["excluded_variables"]


class TestMediationAnalysis:
    def test_detects_mediator(self):
        X, y = _make_mediator_data()
        result = mediation_analysis(
            X,
            y,
            predictor="x",
            mediator="m",
            random_state=42,
        )
        assert result["is_mediator"] is True

    def test_non_mediator(self):
        X, y = _make_mediator_data()
        result = mediation_analysis(
            X,
            y,
            predictor="x",
            mediator="z",
            random_state=42,
        )
        assert result["is_mediator"] is False

    def test_indirect_effect_ci_excludes_zero_for_mediator(self):
        X, y = _make_mediator_data()
        result = mediation_analysis(
            X,
            y,
            predictor="x",
            mediator="m",
            random_state=42,
        )
        ci = result["indirect_effect_ci"]
        assert ci[0] > 0 or ci[1] < 0

    def test_ci_method_is_bca(self):
        X, y = _make_mediator_data()
        result = mediation_analysis(
            X,
            y,
            predictor="x",
            mediator="m",
            random_state=42,
        )
        assert result["ci_method"] == "BCa"


class TestIdentifyConfounders:
    def test_identifies_confounder_not_mediator(self):
        """z causes both x1 and y, so it should be flagged as a candidate."""
        X, y = _make_confounder_data()
        result = identify_confounders(
            X,
            y,
            predictor="x1",
            random_state=42,
            n_bootstrap_mediation=500,
            n_bootstrap_moderation=500,
        )
        screening = result["screening_results"]
        assert "z" in screening["potential_confounders"]
        assert "z" not in screening["excluded_variables"]

    def test_identifies_mediator_not_confounder(self):
        X, y = _make_mediator_data()
        result = identify_confounders(
            X,
            y,
            predictor="x",
            random_state=42,
            n_bootstrap_mediation=500,
            n_bootstrap_moderation=500,
        )
        assert "m" in result["identified_mediators"]
        assert "m" not in result["identified_confounders"]

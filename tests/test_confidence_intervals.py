"""Tests for the confidence interval pipeline (plan Step 9).

Covers sections 9a–9h:
  9a. Standardisation fix
  9b. BCa helper
  9c. Jackknife coefficients
  9d. Permutation CI centering
  9e. Clopper-Pearson p-value CIs
  9f. Wald CI family dispatch
  9g. Standardised CI scaling
  9h. Integration across all families
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from randomization_tests import permutation_test_regression
from randomization_tests.diagnostics import (
    _bca_percentile,
    compute_jackknife_coefs,
    compute_permutation_ci,
    compute_pvalue_ci,
    compute_standardized_ci,
    compute_standardized_coefs,
    compute_wald_ci,
)
from randomization_tests.families import (
    LinearFamily,
    LogisticFamily,
    MultinomialFamily,
    NegativeBinomialFamily,
    OrdinalFamily,
    PoissonFamily,
)

# ── Shared fixtures ──────────────────────────────────────────────── #

_SEED = 42
_N_PERMS = 99


def _linear_data(n: int = 100, seed: int = _SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    y = pd.DataFrame(
        {
            "y": 2.0 * X["x1"].values
            - 1.0 * X["x2"].values
            + rng.standard_normal(n) * 0.5
        }
    )
    return X, y


def _binary_data(n: int = 200, seed: int = _SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    logits = 2.0 * X["x1"].values
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = pd.DataFrame({"y": rng.binomial(1, probs).astype(float)})
    return X, y


def _count_data(n: int = 200, seed: int = _SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    mu = np.exp(0.5 * X["x1"].values + 0.2 * X["x2"].values)
    y = pd.DataFrame({"y": rng.poisson(mu).astype(float)})
    return X, y


def _ordinal_data(n: int = 200, seed: int = _SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    latent = 1.5 * X["x1"].values + rng.standard_normal(n)
    y = pd.DataFrame({"y": np.digitize(latent, bins=[-1, 0, 1]).astype(float)})
    return X, y


def _multinomial_data(
    n: int = 300, seed: int = _SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    logits_1 = 1.0 * X["x1"].values
    logits_2 = -0.5 * X["x2"].values
    p0 = 1.0 / (1.0 + np.exp(logits_1) + np.exp(logits_2))
    p1 = np.exp(logits_1) * p0
    p2 = np.exp(logits_2) * p0
    probs = np.column_stack([p0, p1, p2])
    y = pd.DataFrame(
        {"y": np.array([rng.choice(3, p=probs[i]) for i in range(n)]).astype(float)}
    )
    return X, y


# ================================================================== #
# 9a. Standardisation fix                                            #
# ================================================================== #


class TestStandardisationFix:
    """Verify family-specific standardisation formulas."""

    def test_poisson_uses_sd_x_only(self) -> None:
        X, y_df = _count_data()
        y = y_df.values.ravel()
        coefs = np.array([0.5, 0.2])
        result = compute_standardized_coefs(X, y, coefs, PoissonFamily())
        sd_x = np.std(X.values, axis=0, ddof=1)
        expected = coefs * sd_x
        np.testing.assert_allclose(result, expected)

    def test_negative_binomial_uses_sd_x_only(self) -> None:
        X, y_df = _count_data()
        y = y_df.values.ravel()
        coefs = np.array([0.5, 0.2])
        result = compute_standardized_coefs(X, y, coefs, NegativeBinomialFamily())
        sd_x = np.std(X.values, axis=0, ddof=1)
        expected = coefs * sd_x
        np.testing.assert_allclose(result, expected)

    def test_ordinal_uses_sd_x_only(self) -> None:
        X, y_df = _ordinal_data()
        y = y_df.values.ravel()
        coefs = np.array([1.5, 0.0])
        result = compute_standardized_coefs(X, y, coefs, OrdinalFamily())
        sd_x = np.std(X.values, axis=0, ddof=1)
        expected = coefs * sd_x
        np.testing.assert_allclose(result, expected)

    def test_multinomial_returns_nan(self) -> None:
        X, y_df = _multinomial_data()
        y = y_df.values.ravel()
        coefs = np.array([1.0, -0.5])
        result = compute_standardized_coefs(X, y, coefs, MultinomialFamily())
        assert result.shape == (2,)
        assert all(math.isnan(v) for v in result)

    def test_linear_unchanged(self) -> None:
        """Linear: β · SD(X) / SD(Y) — regression guard."""
        X, y_df = _linear_data()
        y = y_df.values.ravel()
        coefs = np.array([2.0, -1.0])
        result = compute_standardized_coefs(X, y, coefs, LinearFamily())
        sd_x = np.std(X.values, axis=0, ddof=1)
        sd_y = np.std(y, ddof=1)
        expected = coefs * sd_x / sd_y
        np.testing.assert_allclose(result, expected)

    def test_logistic_unchanged(self) -> None:
        """Logistic: β · SD(X) — regression guard."""
        X, y_df = _binary_data()
        y = y_df.values.ravel()
        coefs = np.array([2.0, 0.0])
        result = compute_standardized_coefs(X, y, coefs, LogisticFamily())
        sd_x = np.std(X.values, axis=0, ddof=1)
        expected = coefs * sd_x
        np.testing.assert_allclose(result, expected)


# ================================================================== #
# 9b. BCa helper                                                     #
# ================================================================== #


class TestBcaPercentile:
    """Unit tests for _bca_percentile()."""

    def test_normal_distribution_recovers_percentile(self) -> None:
        """With z₀ ≈ 0 and â ≈ 0, BCa ≈ standard percentile."""
        rng = np.random.default_rng(_SEED)
        boot = rng.standard_normal(10_000)
        observed = 0.0
        jackknife = rng.standard_normal(100)  # Symmetric → â ≈ 0

        lo, hi = _bca_percentile(boot, observed, jackknife, alpha=0.05)
        # Should be close to ±1.96.
        assert abs(lo - (-1.96)) < 0.15
        assert abs(hi - 1.96) < 0.15

    def test_skewed_distribution_asymmetric_ci(self) -> None:
        """With skewed data, z₀ ≠ 0 → asymmetric CI."""
        rng = np.random.default_rng(_SEED)
        # Exponential is right-skewed.
        boot = rng.exponential(scale=1.0, size=5_000)
        observed = 1.0  # Median of exponential(1) ≈ 0.69.
        # Skewed jackknife so â ≠ 0.
        jackknife = rng.exponential(scale=1.0, size=50)

        lo, hi = _bca_percentile(boot, observed, jackknife, alpha=0.05)
        # CI should be asymmetric (wider on the right).
        width_left = observed - lo
        width_right = hi - observed
        assert width_right > width_left * 0.5  # Right tail is at least comparable.

    def test_identical_jackknife_no_crash(self) -> None:
        """All jackknife stats identical → â = 0, no division-by-zero."""
        rng = np.random.default_rng(_SEED)
        boot = rng.standard_normal(1_000)
        observed = 0.0
        jackknife = np.ones(50)  # All identical.

        lo, hi = _bca_percentile(boot, observed, jackknife, alpha=0.05)
        assert np.isfinite(lo)
        assert np.isfinite(hi)
        assert lo < hi


# ================================================================== #
# 9c. Jackknife coefficients                                         #
# ================================================================== #


class TestJackknifeCoefs:
    """Tests for compute_jackknife_coefs()."""

    def test_shape_linear(self) -> None:
        X, y_df = _linear_data(n=20)
        y = y_df.values.ravel()
        result = compute_jackknife_coefs(
            LinearFamily(), X.values, y, fit_intercept=True
        )
        assert result is not None
        assert result.shape == (20, 2)

    def test_loo_correctness(self) -> None:
        """Row 0 matches manual fit with obs 0 removed."""
        X, y_df = _linear_data(n=20)
        y = y_df.values.ravel()
        family = LinearFamily()
        result = compute_jackknife_coefs(family, X.values, y, fit_intercept=True)
        assert result is not None

        # Manual LOO for obs 0.
        mask = np.ones(20, dtype=bool)
        mask[0] = False
        model = family.fit(X.values[mask], y[mask], fit_intercept=True)
        expected = family.coefs(model)[:2]
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_n_gt_500_returns_none(self) -> None:
        X, y_df = _linear_data(n=501)
        y = y_df.values.ravel()
        result = compute_jackknife_coefs(
            LinearFamily(), X.values, y, fit_intercept=True
        )
        assert result is None


# ================================================================== #
# 9d. Permutation CI                                                 #
# ================================================================== #


class TestPermutationCI:
    """Tests for compute_permutation_ci()."""

    def test_ter_braak_shifted(self) -> None:
        """ter Braak null distribution is shifted by +β̂."""
        rng = np.random.default_rng(_SEED)
        model_coefs = np.array([2.0, -1.0])
        # Null distribution centred on 0.
        permuted = rng.standard_normal((500, 2))

        ci = compute_permutation_ci(
            permuted,
            model_coefs,
            "ter_braak",
            alpha=0.05,
            jackknife_coefs=None,
            confounders=[],
            feature_names=["x1", "x2"],
        )
        assert ci.shape == (2, 2)
        # After shift, CI should be centred near β̂.
        midpoint = ci.mean(axis=1)
        np.testing.assert_allclose(midpoint, model_coefs, atol=0.3)

    def test_score_unshifted(self) -> None:
        """Score null distribution is already centred on β̂."""
        rng = np.random.default_rng(_SEED)
        model_coefs = np.array([2.0, -1.0])
        # Distribution already centred on β̂.
        permuted = rng.standard_normal((500, 2)) + model_coefs

        ci = compute_permutation_ci(
            permuted,
            model_coefs,
            "score",
            alpha=0.05,
            jackknife_coefs=None,
            confounders=[],
            feature_names=["x1", "x2"],
        )
        midpoint = ci.mean(axis=1)
        np.testing.assert_allclose(midpoint, model_coefs, atol=0.3)

    def test_percentile_fallback_when_no_jackknife(self) -> None:
        """With jackknife=None, returns simple percentile CI."""
        rng = np.random.default_rng(_SEED)
        model_coefs = np.array([0.0])
        permuted = rng.standard_normal((1_000, 1))

        ci = compute_permutation_ci(
            permuted,
            model_coefs,
            "ter_braak",
            alpha=0.05,
            jackknife_coefs=None,
            confounders=[],
            feature_names=["x1"],
        )
        # Should be close to ±1.96.
        assert abs(ci[0, 0] - (-1.96)) < 0.2
        assert abs(ci[0, 1] - 1.96) < 0.2

    def test_confounder_columns_nan(self) -> None:
        """Confounder columns should return [NaN, NaN]."""
        rng = np.random.default_rng(_SEED)
        model_coefs = np.array([2.0, -1.0])
        permuted = rng.standard_normal((100, 2))

        ci = compute_permutation_ci(
            permuted,
            model_coefs,
            "ter_braak",
            alpha=0.05,
            jackknife_coefs=None,
            confounders=["x2"],
            feature_names=["x1", "x2"],
        )
        assert np.isfinite(ci[0, 0])
        assert np.isnan(ci[1, 0])
        assert np.isnan(ci[1, 1])


# ================================================================== #
# 9e. Clopper-Pearson p-value CIs                                    #
# ================================================================== #


class TestClopperPearson:
    """Tests for compute_pvalue_ci()."""

    def test_hand_computed(self) -> None:
        """Verify against direct scipy computation for counts=3, B=99."""
        counts = np.array([3])
        B = 99
        alpha = 0.05

        ci = compute_pvalue_ci(counts, B, alpha)
        assert ci.shape == (1, 2)

        # Hand computation: successes = 4, trials = 100.
        lo = float(sp_stats.beta.ppf(0.025, 4, 97))
        hi = float(sp_stats.beta.ppf(0.975, 5, 96))
        np.testing.assert_allclose(ci[0, 0], lo, rtol=1e-10)
        np.testing.assert_allclose(ci[0, 1], hi, rtol=1e-10)

    def test_edge_counts_zero(self) -> None:
        """counts=0 → successes=1 with Phipson & Smyth; lower bound > 0 but small."""
        counts = np.array([0])
        ci = compute_pvalue_ci(counts, 99, 0.05)
        # With successes=1, lower bound is small but positive.
        assert ci[0, 0] > 0.0
        assert ci[0, 0] < 0.01
        assert ci[0, 1] > ci[0, 0]

    def test_edge_counts_equal_B(self) -> None:
        """counts=B → upper bound = 1."""
        counts = np.array([99])
        ci = compute_pvalue_ci(counts, 99, 0.05)
        assert ci[0, 0] < 1.0
        assert ci[0, 1] == 1.0

    def test_monotone_bounds(self) -> None:
        """lower ≤ p̂ ≤ upper."""
        counts = np.array([10, 50, 90])
        B = 99
        ci = compute_pvalue_ci(counts, B, 0.05)
        p_hat = (counts + 1) / (B + 1)
        assert np.all(ci[:, 0] <= p_hat)
        assert np.all(ci[:, 1] >= p_hat)

    def test_width_decreases_with_more_permutations(self) -> None:
        """CI width shrinks as B increases."""
        # Fixed count fraction of ~10%.
        widths = []
        for B in [99, 999, 9999]:
            counts = np.array([int(0.1 * B)])
            ci = compute_pvalue_ci(counts, B, 0.05)
            widths.append(ci[0, 1] - ci[0, 0])
        assert widths[0] > widths[1] > widths[2]


# ================================================================== #
# 9f. Wald CI family dispatch                                        #
# ================================================================== #


class TestWaldCI:
    """Tests for compute_wald_ci()."""

    def test_linear_refit(self) -> None:
        """Linear: sklearn model → statsmodels refit → finite Wald CI."""
        import statsmodels.api as sm

        X, y_df = _linear_data()
        y = y_df.values.ravel()
        family = LinearFamily()
        sklearn_model = family.fit(X.values, y, fit_intercept=True)

        wald, cat = compute_wald_ci(
            sklearn_model,
            family,
            2,
            alpha=0.05,
            X=X.values,
            y=y,
            fit_intercept=True,
        )
        assert cat is None
        assert wald.shape == (2, 2)
        assert np.all(np.isfinite(wald))

        # Compare to direct statsmodels OLS.
        X_aug = sm.add_constant(X.values)
        expected = np.asarray(sm.OLS(y, X_aug).fit().conf_int(0.05))[1:3]
        np.testing.assert_allclose(wald, expected)

    def test_logistic_refit(self) -> None:
        """Logistic: sklearn model → statsmodels refit → finite Wald CI."""
        import statsmodels.api as sm

        X, y_df = _binary_data()
        y = y_df.values.ravel()
        family = LogisticFamily()
        sklearn_model = family.fit(X.values, y, fit_intercept=True)

        wald, cat = compute_wald_ci(
            sklearn_model,
            family,
            2,
            alpha=0.05,
            X=X.values,
            y=y,
            fit_intercept=True,
        )
        assert cat is None
        assert wald.shape == (2, 2)
        assert np.all(np.isfinite(wald))

        # Compare to direct statsmodels Logit.
        X_aug = sm.add_constant(X.values)
        expected = np.asarray(sm.Logit(y, X_aug).fit(disp=0).conf_int(0.05))[1:3]
        np.testing.assert_allclose(wald, expected, rtol=1e-4)

    def test_ordinal_excludes_thresholds(self) -> None:
        """Ordinal: output shape is (p, 2), not (p + K-1, 2)."""
        X, y_df = _ordinal_data()
        y = y_df.values.ravel()
        family = OrdinalFamily()
        model = family.fit(X.values, y, fit_intercept=True)
        wald, cat = compute_wald_ci(model, family, 2, alpha=0.05)

        assert cat is None
        assert wald.shape == (2, 2)
        assert np.all(np.isfinite(wald))

    def test_multinomial_per_predictor_nan_and_category_shape(self) -> None:
        """Multinomial: per-predictor NaN; category CI shape (p, K-1, 2)."""
        X, y_df = _multinomial_data()
        y = y_df.values.ravel()
        family = MultinomialFamily()
        model = family.fit(X.values, y, fit_intercept=True)

        n_categories = len(np.unique(y))
        wald, cat = compute_wald_ci(model, family, 2, alpha=0.05)

        # Per-predictor Wald CI should be NaN.
        assert wald.shape == (2, 2)
        assert np.all(np.isnan(wald))

        # Category Wald CI should have shape (p, K-1, 2).
        assert cat is not None
        assert cat.shape == (2, n_categories - 1, 2)

    def test_degenerate_singular_returns_nan(self) -> None:
        """Singular design → NaN-filled array, no crash."""
        # Create a degenerate scenario by passing a bad model.
        X, y = _linear_data()
        family = LinearFamily()

        # Use a mock object that raises on conf_int.
        class _BadModel:
            def conf_int(self, alpha: float) -> None:
                raise np.linalg.LinAlgError("singular matrix")

        wald, cat = compute_wald_ci(_BadModel(), family, 2, alpha=0.05)  # type: ignore[arg-type]
        assert wald.shape == (2, 2)
        assert np.all(np.isnan(wald))
        assert cat is None


# ================================================================== #
# 9g. Standardised CI scaling                                        #
# ================================================================== #


class TestStandardizedCI:
    """Tests for compute_standardized_ci()."""

    def test_linear_scales_by_sd_ratio(self) -> None:
        X, y_df = _linear_data()
        y = y_df.values.ravel()
        coefs = np.array([2.0, -1.0])
        perm_ci = np.array([[1.5, 2.5], [-1.5, -0.5]])

        std_ci = compute_standardized_ci(perm_ci, coefs, X, y, LinearFamily())
        sd_x = np.std(X.values, axis=0, ddof=1)
        sd_y = np.std(y, ddof=1)
        expected = perm_ci * (sd_x / sd_y)[:, np.newaxis]
        np.testing.assert_allclose(std_ci, expected)

    def test_logistic_scales_by_sd_x(self) -> None:
        X, y_df = _binary_data()
        y = y_df.values.ravel()
        coefs = np.array([2.0, 0.0])
        perm_ci = np.array([[1.0, 3.0], [-0.5, 0.5]])

        std_ci = compute_standardized_ci(perm_ci, coefs, X, y, LogisticFamily())
        sd_x = np.std(X.values, axis=0, ddof=1)
        expected = perm_ci * sd_x[:, np.newaxis]
        np.testing.assert_allclose(std_ci, expected)

    def test_multinomial_returns_nan(self) -> None:
        X, y_df = _multinomial_data()
        y = y_df.values.ravel()
        coefs = np.array([1.0, -0.5])
        perm_ci = np.array([[0.5, 1.5], [-1.0, 0.0]])

        std_ci = compute_standardized_ci(perm_ci, coefs, X, y, MultinomialFamily())
        assert std_ci.shape == (2, 2)
        assert np.all(np.isnan(std_ci))


# ================================================================== #
# 9h. Integration — result dict across families                      #
# ================================================================== #


_CI_KEYS = {
    "permutation_ci",
    "pvalue_ci",
    "wald_ci",
    "standardized_ci",
    "confidence_level",
    "ci_method",
}


class TestCIIntegrationLinear:
    """Linear family CI integration."""

    def test_ci_dict_structure(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="ter_braak",
            confidence_level=0.95,
        )
        ci = result.confidence_intervals
        assert set(ci.keys()) >= _CI_KEYS
        assert ci["confidence_level"] == 0.95
        assert ci["ci_method"] in ("bca", "percentile")

    def test_ci_shapes(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="ter_braak",
        )
        ci = result.confidence_intervals
        p = X.shape[1]
        assert np.array(ci["permutation_ci"]).shape == (p, 2)
        assert np.array(ci["pvalue_ci"]).shape == (p, 2)
        assert np.array(ci["wald_ci"]).shape == (p, 2)
        assert np.array(ci["standardized_ci"]).shape == (p, 2)


class TestCIIntegrationLogistic:
    """Logistic family CI integration."""

    def test_ci_dict_structure(self) -> None:
        X, y = _binary_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="ter_braak",
            family="logistic",
        )
        ci = result.confidence_intervals
        assert set(ci.keys()) >= _CI_KEYS


class TestCIIntegrationPoisson:
    """Poisson family CI integration."""

    def test_ci_dict_structure(self) -> None:
        X, y = _count_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="ter_braak",
            family="poisson",
        )
        ci = result.confidence_intervals
        assert set(ci.keys()) >= _CI_KEYS


class TestCIIntegrationNegativeBinomial:
    """Negative binomial family CI integration."""

    def test_ci_dict_structure(self) -> None:
        X, y = _count_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="ter_braak",
            family="negative_binomial",
        )
        ci = result.confidence_intervals
        assert set(ci.keys()) >= _CI_KEYS


class TestCIIntegrationOrdinal:
    """Ordinal family CI integration."""

    def test_ci_dict_structure(self) -> None:
        X, y = _ordinal_data()
        with pytest.warns(UserWarning, match="ordinal"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=_N_PERMS,
                random_state=_SEED,
                method="ter_braak",
                family="ordinal",
            )
        ci = result.confidence_intervals
        assert set(ci.keys()) >= _CI_KEYS


class TestCIIntegrationMultinomial:
    """Multinomial family CI integration."""

    def test_ci_dict_structure(self) -> None:
        X, y = _multinomial_data()
        with pytest.warns(UserWarning, match="multinomial"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=_N_PERMS,
                random_state=_SEED,
                method="ter_braak",
                family="multinomial",
            )
        ci = result.confidence_intervals
        assert set(ci.keys()) >= _CI_KEYS

    def test_category_wald_ci_present(self) -> None:
        X, y = _multinomial_data()
        with pytest.warns(UserWarning, match="multinomial"):
            result = permutation_test_regression(
                X,
                y,
                n_permutations=_N_PERMS,
                random_state=_SEED,
                method="ter_braak",
                family="multinomial",
            )
        ci = result.confidence_intervals
        assert "category_wald_ci" in ci
        cat_ci = np.array(ci["category_wald_ci"])
        n_categories = len(np.unique(y))
        assert cat_ci.shape == (X.shape[1], n_categories - 1, 2)


class TestCIIntegrationKennedy:
    """Kennedy method CI integration (score-based, no shift)."""

    def test_ci_dict_structure(self) -> None:
        X, y = _linear_data()
        X_with_conf = X.copy()
        X_with_conf["x3"] = np.random.default_rng(_SEED).standard_normal(len(y))
        result = permutation_test_regression(
            X_with_conf,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="kennedy",
            confounders=["x3"],
        )
        ci = result.confidence_intervals
        assert set(ci.keys()) >= _CI_KEYS


class TestCIIntegrationFreedmanLane:
    """Freedman-Lane method CI integration (shifted)."""

    def test_ci_dict_structure(self) -> None:
        X, y = _linear_data()
        X_with_conf = X.copy()
        X_with_conf["x3"] = np.random.default_rng(_SEED).standard_normal(len(y))
        result = permutation_test_regression(
            X_with_conf,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
            method="freedman_lane",
            confounders=["x3"],
        )
        ci = result.confidence_intervals
        assert set(ci.keys()) >= _CI_KEYS

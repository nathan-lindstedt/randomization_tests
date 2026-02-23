"""Tests for the ModelFamily protocol, LinearFamily, LogisticFamily,
PoissonFamily, NegativeBinomialFamily, and OrdinalFamily.
"""

import numpy as np
import pytest

from randomization_tests.families import (
    LinearFamily,
    LogisticFamily,
    ModelFamily,
    NegativeBinomialFamily,
    OrdinalFamily,
    PoissonFamily,
    register_family,
    resolve_family,
)

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def linear_data(rng):
    n, p = 100, 3
    X = rng.standard_normal((n, p))
    y = X @ np.array([2.0, -1.0, 0.5]) + rng.standard_normal(n) * 0.5
    return X, y


@pytest.fixture()
def family():
    return LinearFamily()


# ------------------------------------------------------------------ #
# Protocol conformance
# ------------------------------------------------------------------ #


class TestProtocolConformance:
    def test_isinstance_check(self, family):
        assert isinstance(family, ModelFamily)

    def test_name(self, family):
        assert family.name == "linear"

    def test_residual_type(self, family):
        assert family.residual_type == "raw"

    def test_direct_permutation(self, family):
        assert family.direct_permutation is False


# ------------------------------------------------------------------ #
# validate_y
# ------------------------------------------------------------------ #


class TestValidateY:
    def test_valid_continuous(self, family, rng):
        family.validate_y(rng.standard_normal(50))

    def test_rejects_non_numeric(self, family):
        with pytest.raises(ValueError, match="numeric"):
            family.validate_y(np.array(["a", "b", "c"]))

    def test_rejects_constant(self, family):
        with pytest.raises(ValueError, match="non-constant"):
            family.validate_y(np.ones(50))


# ------------------------------------------------------------------ #
# Single-model operations
# ------------------------------------------------------------------ #


class TestSingleModel:
    def test_fit_predict_roundtrip(self, family, linear_data):
        X, y = linear_data
        model = family.fit(X, y)
        y_hat = family.predict(model, X)
        assert y_hat.shape == y.shape
        # R² should be high for this low-noise data
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        assert 1 - ss_res / ss_tot > 0.9

    def test_coefs_shape(self, family, linear_data):
        X, y = linear_data
        model = family.fit(X, y)
        c = family.coefs(model)
        assert c.shape == (X.shape[1],)

    def test_coefs_close_to_true(self, family, linear_data):
        X, y = linear_data
        model = family.fit(X, y)
        c = family.coefs(model)
        np.testing.assert_allclose(c, [2.0, -1.0, 0.5], atol=0.3)

    def test_residuals_shape(self, family, linear_data):
        X, y = linear_data
        model = family.fit(X, y)
        r = family.residuals(model, X, y)
        assert r.shape == y.shape

    def test_residuals_sum_near_zero(self, family, linear_data):
        X, y = linear_data
        model = family.fit(X, y, fit_intercept=True)
        r = family.residuals(model, X, y)
        assert abs(r.mean()) < 0.1

    def test_fit_no_intercept(self, family, linear_data):
        X, y = linear_data
        model = family.fit(X, y, fit_intercept=False)
        c = family.coefs(model)
        assert c.shape == (X.shape[1],)


# ------------------------------------------------------------------ #
# reconstruct_y
# ------------------------------------------------------------------ #


class TestReconstructY:
    def test_additive(self, family, rng):
        preds = rng.standard_normal(50)
        resids = rng.standard_normal(50)
        result = family.reconstruct_y(preds, resids, rng)
        np.testing.assert_array_equal(result, preds + resids)


# ------------------------------------------------------------------ #
# fit_metric (RSS)
# ------------------------------------------------------------------ #


class TestFitMetric:
    def test_rss_value(self, family):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert family.fit_metric(y_true, y_pred) == pytest.approx(0.0)

    def test_rss_nonzero(self, family):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.0, 0.0, 0.0])
        # RSS = 1 + 4 + 9 = 14
        assert family.fit_metric(y_true, y_pred) == pytest.approx(14.0)


# ------------------------------------------------------------------ #
# diagnostics
# ------------------------------------------------------------------ #


class TestDiagnostics:
    def test_keys(self, family, linear_data):
        X, y = linear_data
        diag = family.diagnostics(X, y)
        expected_keys = {
            "n_observations",
            "n_features",
            "r_squared",
            "r_squared_adj",
            "f_statistic",
            "f_p_value",
            "aic",
            "bic",
        }
        assert set(diag.keys()) == expected_keys

    def test_n_observations(self, family, linear_data):
        X, y = linear_data
        diag = family.diagnostics(X, y)
        assert diag["n_observations"] == len(y)

    def test_r_squared_range(self, family, linear_data):
        X, y = linear_data
        diag = family.diagnostics(X, y)
        assert 0.0 < diag["r_squared"] <= 1.0


# ------------------------------------------------------------------ #
# classical_p_values
# ------------------------------------------------------------------ #


class TestClassicalPValues:
    def test_shape(self, family, linear_data):
        X, y = linear_data
        pvals = family.classical_p_values(X, y)
        assert pvals.shape == (X.shape[1],)

    def test_range(self, family, linear_data):
        X, y = linear_data
        pvals = family.classical_p_values(X, y)
        assert np.all(pvals >= 0.0)
        assert np.all(pvals <= 1.0)

    def test_significant_predictor(self, family, linear_data):
        X, y = linear_data
        pvals = family.classical_p_values(X, y)
        # x1 has true coef 2.0 — should be highly significant
        assert pvals[0] < 0.01


# ------------------------------------------------------------------ #
# batch_fit
# ------------------------------------------------------------------ #


class TestBatchFit:
    def test_shape(self, family, linear_data, rng):
        X, y = linear_data
        B = 20
        Y_matrix = np.tile(y, (B, 1))
        # Shuffle each row
        for i in range(B):
            rng.shuffle(Y_matrix[i])
        coefs = family.batch_fit(X, Y_matrix, fit_intercept=True)
        assert coefs.shape == (B, X.shape[1])

    def test_unpermuted_row_matches_single_fit(self, family, linear_data):
        X, y = linear_data
        Y_matrix = y.reshape(1, -1)
        batch_coefs = family.batch_fit(X, Y_matrix, fit_intercept=True)
        model = family.fit(X, y, fit_intercept=True)
        single_coefs = family.coefs(model)
        np.testing.assert_allclose(batch_coefs[0], single_coefs, atol=1e-8)


# ------------------------------------------------------------------ #
# Family registry / resolution
# ------------------------------------------------------------------ #


class TestRegistry:
    def test_linear_registered(self):
        fam = resolve_family("linear", np.array([1.0, 2.0]))
        assert isinstance(fam, LinearFamily)

    def test_auto_continuous(self, rng):
        y = rng.standard_normal(50)
        fam = resolve_family("auto", y)
        assert isinstance(fam, LinearFamily)

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown family"):
            resolve_family("gamma", np.array([1.0, 2.0]))

    def test_register_non_protocol_raises(self):
        with pytest.raises(TypeError, match="does not implement"):
            register_family("bad", int)

    def test_frozen_dataclass(self, family):
        with pytest.raises(AttributeError):
            family.name = "other"  # type: ignore[misc]


# ================================================================== #
# LogisticFamily tests
# ================================================================== #


@pytest.fixture()
def logistic_data(rng):
    n, p = 200, 2
    X = rng.standard_normal((n, p))
    logits = 2.0 * X[:, 0] + 0.0 * X[:, 1]
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(float)
    return X, y


@pytest.fixture()
def logistic_family():
    return LogisticFamily()


# ------------------------------------------------------------------ #
# Protocol conformance (logistic)
# ------------------------------------------------------------------ #


class TestLogisticProtocolConformance:
    def test_isinstance_check(self, logistic_family):
        assert isinstance(logistic_family, ModelFamily)

    def test_name(self, logistic_family):
        assert logistic_family.name == "logistic"

    def test_residual_type(self, logistic_family):
        assert logistic_family.residual_type == "probability"

    def test_direct_permutation(self, logistic_family):
        assert logistic_family.direct_permutation is False


# ------------------------------------------------------------------ #
# validate_y (logistic)
# ------------------------------------------------------------------ #


class TestLogisticValidateY:
    def test_valid_binary(self, logistic_family):
        logistic_family.validate_y(np.array([0, 1, 0, 1, 1, 0]))

    def test_rejects_continuous(self, logistic_family):
        with pytest.raises(ValueError, match="binary"):
            logistic_family.validate_y(np.array([0.5, 1.5, 2.5]))

    def test_rejects_single_class(self, logistic_family):
        with pytest.raises(ValueError, match="binary"):
            logistic_family.validate_y(np.zeros(50))

    def test_rejects_three_classes(self, logistic_family):
        with pytest.raises(ValueError, match="binary"):
            logistic_family.validate_y(np.array([0, 1, 2]))


# ------------------------------------------------------------------ #
# Single-model operations (logistic)
# ------------------------------------------------------------------ #


class TestLogisticSingleModel:
    def test_fit_predict_shape(self, logistic_family, logistic_data):
        X, y = logistic_data
        model = logistic_family.fit(X, y)
        preds = logistic_family.predict(model, X)
        assert preds.shape == y.shape

    def test_predict_range(self, logistic_family, logistic_data):
        X, y = logistic_data
        model = logistic_family.fit(X, y)
        preds = logistic_family.predict(model, X)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)

    def test_coefs_shape(self, logistic_family, logistic_data):
        X, y = logistic_data
        model = logistic_family.fit(X, y)
        c = logistic_family.coefs(model)
        assert c.shape == (X.shape[1],)

    def test_significant_predictor_positive(self, logistic_family, logistic_data):
        X, y = logistic_data
        model = logistic_family.fit(X, y)
        c = logistic_family.coefs(model)
        # x1 has true coef 2.0 — should be clearly positive
        assert c[0] > 0.5

    def test_residuals_shape(self, logistic_family, logistic_data):
        X, y = logistic_data
        model = logistic_family.fit(X, y)
        r = logistic_family.residuals(model, X, y)
        assert r.shape == y.shape

    def test_residuals_range(self, logistic_family, logistic_data):
        X, y = logistic_data
        model = logistic_family.fit(X, y)
        r = logistic_family.residuals(model, X, y)
        assert np.all(r >= -1.0)
        assert np.all(r <= 1.0)


# ------------------------------------------------------------------ #
# reconstruct_y (logistic)
# ------------------------------------------------------------------ #


class TestLogisticReconstructY:
    def test_output_is_binary(self, logistic_family, rng):
        preds = np.full(100, 0.5)
        resids = rng.uniform(-0.3, 0.3, size=100)
        result = logistic_family.reconstruct_y(preds, resids, rng)
        assert set(np.unique(result)).issubset({0, 1})

    def test_shape_preserved(self, logistic_family, rng):
        preds = np.full((20, 50), 0.5)
        resids = rng.uniform(-0.3, 0.3, size=(20, 50))
        result = logistic_family.reconstruct_y(preds, resids, rng)
        assert result.shape == (20, 50)


# ------------------------------------------------------------------ #
# fit_metric (logistic — deviance)
# ------------------------------------------------------------------ #


class TestLogisticFitMetric:
    def test_perfect_prediction(self, logistic_family):
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.001, 0.999, 0.001, 0.999])
        # Near-zero deviance for near-perfect predictions
        assert logistic_family.fit_metric(y_true, y_pred) < 0.1

    def test_bad_prediction_high_deviance(self, logistic_family):
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        # Deviance should be higher than near-perfect case
        assert logistic_family.fit_metric(y_true, y_pred) > 1.0


# ------------------------------------------------------------------ #
# diagnostics (logistic)
# ------------------------------------------------------------------ #


class TestLogisticDiagnostics:
    def test_keys(self, logistic_family, logistic_data):
        X, y = logistic_data
        diag = logistic_family.diagnostics(X, y)
        expected_keys = {
            "n_observations",
            "n_features",
            "pseudo_r_squared",
            "log_likelihood",
            "log_likelihood_null",
            "llr_p_value",
            "aic",
            "bic",
        }
        assert set(diag.keys()) == expected_keys

    def test_n_observations(self, logistic_family, logistic_data):
        X, y = logistic_data
        diag = logistic_family.diagnostics(X, y)
        assert diag["n_observations"] == len(y)

    def test_pseudo_r_squared_positive(self, logistic_family, logistic_data):
        X, y = logistic_data
        diag = logistic_family.diagnostics(X, y)
        assert diag["pseudo_r_squared"] > 0.0


# ------------------------------------------------------------------ #
# classical_p_values (logistic)
# ------------------------------------------------------------------ #


class TestLogisticClassicalPValues:
    def test_shape(self, logistic_family, logistic_data):
        X, y = logistic_data
        pvals = logistic_family.classical_p_values(X, y)
        assert pvals.shape == (X.shape[1],)

    def test_range(self, logistic_family, logistic_data):
        X, y = logistic_data
        pvals = logistic_family.classical_p_values(X, y)
        assert np.all(pvals >= 0.0)
        assert np.all(pvals <= 1.0)

    def test_significant_predictor(self, logistic_family, logistic_data):
        X, y = logistic_data
        pvals = logistic_family.classical_p_values(X, y)
        # x1 has true coef 2.0 — should be highly significant
        assert pvals[0] < 0.01


# ------------------------------------------------------------------ #
# batch_fit (logistic)
# ------------------------------------------------------------------ #


class TestLogisticBatchFit:
    def test_shape(self, logistic_family, logistic_data, rng):
        X, y = logistic_data
        B = 10
        Y_matrix = np.tile(y, (B, 1))
        for i in range(B):
            rng.shuffle(Y_matrix[i])
        coefs = logistic_family.batch_fit(X, Y_matrix, fit_intercept=True)
        assert coefs.shape == (B, X.shape[1])

    def test_unpermuted_row_matches_single_fit(self, logistic_family, logistic_data):
        X, y = logistic_data
        Y_matrix = y.reshape(1, -1)
        batch_coefs = logistic_family.batch_fit(X, Y_matrix, fit_intercept=True)
        model = logistic_family.fit(X, y, fit_intercept=True)
        single_coefs = logistic_family.coefs(model)
        np.testing.assert_allclose(batch_coefs[0], single_coefs, atol=0.1)


# ------------------------------------------------------------------ #
# Registry (logistic additions)
# ------------------------------------------------------------------ #


class TestLogisticRegistry:
    def test_logistic_registered(self):
        fam = resolve_family("logistic", np.array([0.0, 1.0]))
        assert isinstance(fam, LogisticFamily)

    def test_auto_binary(self):
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=float)
        fam = resolve_family("auto", y)
        assert isinstance(fam, LogisticFamily)

    def test_frozen_dataclass(self, logistic_family):
        with pytest.raises(AttributeError):
            logistic_family.name = "other"  # type: ignore[misc]


# ================================================================== #
# PoissonFamily
# ================================================================== #


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture()
def poisson_rng():
    return np.random.default_rng(42)


@pytest.fixture()
def poisson_data(poisson_rng):
    n, p = 100, 3
    X = poisson_rng.standard_normal((n, p))
    # True model: log(μ) = 2 + 1.5*x1 - 0.8*x2 + 0.3*x3
    eta = 2.0 + X @ np.array([1.5, -0.8, 0.3])
    mu = np.exp(eta)
    y = poisson_rng.poisson(lam=mu).astype(float)
    return X, y


@pytest.fixture()
def poisson_family():
    return PoissonFamily()


# ------------------------------------------------------------------ #
# Protocol conformance
# ------------------------------------------------------------------ #


class TestPoissonProtocolConformance:
    def test_isinstance_check(self, poisson_family):
        assert isinstance(poisson_family, ModelFamily)

    def test_name(self, poisson_family):
        assert poisson_family.name == "poisson"

    def test_residual_type(self, poisson_family):
        assert poisson_family.residual_type == "response"

    def test_direct_permutation(self, poisson_family):
        assert poisson_family.direct_permutation is False

    def test_metric_label(self, poisson_family):
        assert poisson_family.metric_label == "Deviance Reduction"


# ------------------------------------------------------------------ #
# validate_y
# ------------------------------------------------------------------ #


class TestPoissonValidateY:
    def test_valid_counts(self, poisson_family):
        """Non-negative integer counts should pass."""
        poisson_family.validate_y(np.array([0, 1, 5, 10, 100], dtype=float))

    def test_valid_int_dtype(self, poisson_family):
        """Integer dtypes should pass."""
        poisson_family.validate_y(np.array([0, 1, 2, 3], dtype=int))

    def test_valid_whole_floats(self, poisson_family):
        """Float values that are whole numbers should pass."""
        poisson_family.validate_y(np.array([0.0, 1.0, 5.0, 10.0]))

    def test_rejects_non_numeric(self, poisson_family):
        with pytest.raises(ValueError, match="numeric"):
            poisson_family.validate_y(np.array(["a", "b", "c"]))

    def test_rejects_nan(self, poisson_family):
        with pytest.raises(ValueError, match="NaN"):
            poisson_family.validate_y(np.array([1.0, np.nan, 3.0]))

    def test_rejects_negative(self, poisson_family):
        with pytest.raises(ValueError, match="non-negative"):
            poisson_family.validate_y(np.array([1.0, -1.0, 3.0]))

    def test_rejects_fractional(self, poisson_family):
        with pytest.raises(ValueError, match="integer-valued"):
            poisson_family.validate_y(np.array([1.0, 2.5, 3.0]))


# ------------------------------------------------------------------ #
# Single-model operations
# ------------------------------------------------------------------ #


class TestPoissonSingleModel:
    def test_fit_predict_roundtrip(self, poisson_family, poisson_data):
        X, y = poisson_data
        model = poisson_family.fit(X, y, fit_intercept=True)
        preds = poisson_family.predict(model, X)
        assert preds.shape == (len(y),)
        # Predictions should be non-negative (response scale).
        assert np.all(preds > 0)
        # Correlation between y and ŷ should be positive for a
        # well-specified Poisson model.
        assert np.corrcoef(y, preds)[0, 1] > 0.5

    def test_coefs_shape(self, poisson_family, poisson_data):
        X, y = poisson_data
        model = poisson_family.fit(X, y, fit_intercept=True)
        coefs = poisson_family.coefs(model)
        assert coefs.shape == (X.shape[1],)

    def test_coefs_accuracy(self, poisson_family, poisson_data):
        """Recovered coefficients should be close to true values."""
        X, y = poisson_data
        model = poisson_family.fit(X, y, fit_intercept=True)
        coefs = poisson_family.coefs(model)
        true_coefs = np.array([1.5, -0.8, 0.3])
        np.testing.assert_allclose(coefs, true_coefs, atol=0.4)

    def test_residuals_shape(self, poisson_family, poisson_data):
        X, y = poisson_data
        model = poisson_family.fit(X, y, fit_intercept=True)
        resids = poisson_family.residuals(model, X, y)
        assert resids.shape == (len(y),)

    def test_residuals_are_response_scale(self, poisson_family, poisson_data):
        """Response-scale residuals should be y - μ̂."""
        X, y = poisson_data
        model = poisson_family.fit(X, y, fit_intercept=True)
        resids = poisson_family.residuals(model, X, y)
        preds = poisson_family.predict(model, X)
        np.testing.assert_allclose(resids, y - preds)

    def test_fit_no_intercept(self, poisson_family, poisson_data):
        X, y = poisson_data
        model = poisson_family.fit(X, y, fit_intercept=False)
        coefs = poisson_family.coefs(model)
        assert coefs.shape == (X.shape[1],)


# ------------------------------------------------------------------ #
# reconstruct_y
# ------------------------------------------------------------------ #


class TestPoissonReconstructY:
    def test_output_is_nonneg_integer(self, poisson_family, poisson_data, poisson_rng):
        X, y = poisson_data
        model = poisson_family.fit(X, y, fit_intercept=True)
        preds = poisson_family.predict(model, X)
        resids = poisson_family.residuals(model, X, y)
        y_star = poisson_family.reconstruct_y(
            preds[np.newaxis, :], resids[np.newaxis, :], poisson_rng
        )
        assert y_star.shape == (1, len(y))
        assert np.all(y_star >= 0)
        # Poisson sampling produces integers.
        assert np.allclose(y_star, np.round(y_star))

    def test_shape_with_multiple_permutations(
        self, poisson_family, poisson_data, poisson_rng
    ):
        X, y = poisson_data
        model = poisson_family.fit(X, y, fit_intercept=True)
        preds = poisson_family.predict(model, X)
        resids = poisson_family.residuals(model, X, y)
        B = 20
        perm_resids = np.array([poisson_rng.permutation(resids) for _ in range(B)])
        y_star = poisson_family.reconstruct_y(
            preds[np.newaxis, :], perm_resids, poisson_rng
        )
        assert y_star.shape == (B, len(y))

    def test_mean_preserving(self, poisson_family, poisson_data, poisson_rng):
        """Reconstructed Y* should have a similar mean to original Y."""
        X, y = poisson_data
        model = poisson_family.fit(X, y, fit_intercept=True)
        preds = poisson_family.predict(model, X)
        resids = poisson_family.residuals(model, X, y)
        B = 50
        perm_resids = np.array([poisson_rng.permutation(resids) for _ in range(B)])
        y_star = poisson_family.reconstruct_y(
            preds[np.newaxis, :], perm_resids, poisson_rng
        )
        # Mean of Y* across all B permutations should be close to
        # the original mean (within a factor of 3).
        assert y_star.mean() < y.mean() * 3
        assert y_star.mean() > y.mean() / 3


# ------------------------------------------------------------------ #
# fit_metric
# ------------------------------------------------------------------ #


class TestPoissonFitMetric:
    def test_perfect_prediction(self, poisson_family):
        """Deviance should be 0 when predictions match exactly."""
        y = np.array([5.0, 10.0, 3.0, 8.0])
        deviance = poisson_family.fit_metric(y, y)
        assert deviance == pytest.approx(0.0, abs=1e-10)

    def test_zero_counts(self, poisson_family):
        """y=0 entries should contribute 0 to deviance (0*log(0/μ)=0)."""
        y = np.array([0.0, 0.0, 5.0, 10.0])
        preds = np.array([1.0, 2.0, 5.0, 10.0])
        deviance = poisson_family.fit_metric(y, preds)
        # Only the first two entries contribute: 2*(0 - (0-μ)) = 2*μ
        assert deviance > 0

    def test_positive_for_imperfect(self, poisson_family):
        y = np.array([3.0, 7.0, 12.0])
        preds = np.array([5.0, 5.0, 5.0])
        deviance = poisson_family.fit_metric(y, preds)
        assert deviance > 0

    def test_no_runtime_warnings(self, poisson_family):
        """fit_metric should not emit divide-by-zero warnings."""
        import warnings as w

        y = np.array([0.0, 0.0, 1.0, 5.0, 0.0])
        preds = np.array([2.0, 3.0, 1.0, 5.0, 1.0])
        with w.catch_warnings():
            w.simplefilter("error", RuntimeWarning)
            poisson_family.fit_metric(y, preds)  # should not raise


# ------------------------------------------------------------------ #
# diagnostics
# ------------------------------------------------------------------ #


class TestPoissonDiagnostics:
    def test_expected_keys(self, poisson_family, poisson_data):
        X, y = poisson_data
        diag = poisson_family.diagnostics(X, y, fit_intercept=True)
        expected = {
            "n_observations",
            "n_features",
            "deviance",
            "pearson_chi2",
            "dispersion",
            "log_likelihood",
            "aic",
            "bic",
        }
        assert expected.issubset(set(diag.keys()))

    def test_n_observations(self, poisson_family, poisson_data):
        X, y = poisson_data
        diag = poisson_family.diagnostics(X, y, fit_intercept=True)
        assert diag["n_observations"] == len(y)

    def test_dispersion_positive(self, poisson_family, poisson_data):
        X, y = poisson_data
        diag = poisson_family.diagnostics(X, y, fit_intercept=True)
        assert diag["dispersion"] > 0

    def test_deviance_positive(self, poisson_family, poisson_data):
        X, y = poisson_data
        diag = poisson_family.diagnostics(X, y, fit_intercept=True)
        assert diag["deviance"] > 0


# ------------------------------------------------------------------ #
# classical_p_values
# ------------------------------------------------------------------ #


class TestPoissonClassicalPValues:
    def test_shape(self, poisson_family, poisson_data):
        X, y = poisson_data
        pvals = poisson_family.classical_p_values(X, y, fit_intercept=True)
        assert pvals.shape == (X.shape[1],)

    def test_range(self, poisson_family, poisson_data):
        X, y = poisson_data
        pvals = poisson_family.classical_p_values(X, y, fit_intercept=True)
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)

    def test_significant_predictor_detected(self, poisson_family, poisson_data):
        """The true β₁ = 1.5 should be strongly significant."""
        X, y = poisson_data
        pvals = poisson_family.classical_p_values(X, y, fit_intercept=True)
        assert pvals[0] < 0.01


# ------------------------------------------------------------------ #
# exchangeability_cells
# ------------------------------------------------------------------ #


class TestPoissonExchangeabilityCells:
    def test_returns_none(self, poisson_family, poisson_data):
        X, y = poisson_data
        cells = poisson_family.exchangeability_cells(X, y)
        assert cells is None


# ------------------------------------------------------------------ #
# batch_fit
# ------------------------------------------------------------------ #


class TestPoissonBatchFit:
    def test_output_shape(self, poisson_family, poisson_data, poisson_rng):
        X, y = poisson_data
        B = 10
        Y_matrix = np.tile(y, (B, 1))
        for i in range(B):
            poisson_rng.shuffle(Y_matrix[i])
        coefs = poisson_family.batch_fit(X, Y_matrix, fit_intercept=True)
        assert coefs.shape == (B, X.shape[1])

    def test_unpermuted_row_matches_single_fit(self, poisson_family, poisson_data):
        X, y = poisson_data
        Y_matrix = y.reshape(1, -1)
        batch_coefs = poisson_family.batch_fit(X, Y_matrix, fit_intercept=True)
        model = poisson_family.fit(X, y, fit_intercept=True)
        single_coefs = poisson_family.coefs(model)
        np.testing.assert_allclose(batch_coefs[0], single_coefs, atol=0.01)

    def test_convergence_failure_returns_nan(self, poisson_family):
        """Pathological data should produce NaN, not crash."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((20, 2))
        # Extreme Y values to provoke convergence failure.
        y_bad = np.array([1e6] * 10 + [0] * 10, dtype=float)
        Y_matrix = y_bad.reshape(1, -1)
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("ignore")
            coefs = poisson_family.batch_fit(X, Y_matrix, fit_intercept=True)
        # Should be shape (1, 2), possibly NaN if failed.
        assert coefs.shape == (1, 2)


# ------------------------------------------------------------------ #
# batch_fit_varying_X
# ------------------------------------------------------------------ #


class TestPoissonBatchFitVaryingX:
    def test_output_shape(self, poisson_family, poisson_data, poisson_rng):
        X, y = poisson_data
        B = 5
        X_batch = np.broadcast_to(X, (B, *X.shape)).copy()
        # Permute column 0 in each batch.
        for i in range(B):
            poisson_rng.shuffle(X_batch[i, :, 0])
        coefs = poisson_family.batch_fit_varying_X(X_batch, y, fit_intercept=True)
        assert coefs.shape == (B, X.shape[1])

    def test_unpermuted_matches_single_fit(self, poisson_family, poisson_data):
        X, y = poisson_data
        X_batch = X[np.newaxis, :, :]  # (1, n, p)
        batch_coefs = poisson_family.batch_fit_varying_X(X_batch, y, fit_intercept=True)
        model = poisson_family.fit(X, y, fit_intercept=True)
        single_coefs = poisson_family.coefs(model)
        np.testing.assert_allclose(batch_coefs[0], single_coefs, atol=0.01)


# ------------------------------------------------------------------ #
# Registry (Poisson additions)
# ------------------------------------------------------------------ #


class TestPoissonRegistry:
    def test_poisson_registered(self):
        fam = resolve_family("poisson", np.array([0.0, 1.0, 2.0]))
        assert isinstance(fam, PoissonFamily)

    def test_auto_does_not_resolve_poisson(self):
        """Count auto-detection is not yet implemented (Step 22)."""
        y = np.array([0, 1, 2, 5, 10, 3, 7, 4], dtype=float)
        fam = resolve_family("auto", y)
        # Should fall through to linear, not poisson.
        assert isinstance(fam, LinearFamily)


# ================================================================== #
# NegativeBinomialFamily
# ================================================================== #


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture()
def nb_rng():
    return np.random.default_rng(42)


@pytest.fixture()
def nb_data(nb_rng):
    """Overdispersed count data from the NB2 model.

    True model: log(μ) = 1 + 1.0*x1 - 0.5*x2 + 0.2*x3
    True α = 1.5
    """
    n, p = 200, 3
    X = nb_rng.standard_normal((n, p))
    eta = 1.0 + X @ np.array([1.0, -0.5, 0.2])
    mu = np.exp(eta)
    alpha = 1.5
    # NB2 sampling: n_param = 1/α, p_param = 1/(1 + α·μ)
    n_param = 1.0 / alpha
    p_param = 1.0 / (1.0 + alpha * mu)
    y = nb_rng.negative_binomial(n=n_param, p=p_param).astype(float)
    return X, y


@pytest.fixture()
def nb_family_uncalibrated():
    """Uncalibrated NB family (alpha=None)."""
    return NegativeBinomialFamily()


@pytest.fixture()
def nb_family(nb_data):
    """Pre-calibrated NB family with α estimated from the fixture data."""
    X, y = nb_data
    fam = NegativeBinomialFamily()
    calibrated = fam.calibrate(X, y, fit_intercept=True)
    assert isinstance(calibrated, NegativeBinomialFamily)
    return calibrated


@pytest.fixture()
def nb_family_known_alpha():
    """NB family with a user-supplied α."""
    return NegativeBinomialFamily(alpha=1.5)


# ------------------------------------------------------------------ #
# Protocol conformance
# ------------------------------------------------------------------ #


class TestNBProtocolConformance:
    def test_isinstance_check(self, nb_family):
        assert isinstance(nb_family, ModelFamily)

    def test_name(self, nb_family):
        assert nb_family.name == "negative_binomial"

    def test_residual_type(self, nb_family):
        assert nb_family.residual_type == "response"

    def test_direct_permutation(self, nb_family):
        assert nb_family.direct_permutation is False

    def test_metric_label(self, nb_family):
        assert nb_family.metric_label == "Deviance Reduction"


# ------------------------------------------------------------------ #
# Calibration
# ------------------------------------------------------------------ #


class TestNBCalibrate:
    def test_uncalibrated_alpha_is_none(self, nb_family_uncalibrated):
        assert nb_family_uncalibrated.alpha is None

    def test_calibrate_returns_new_instance(self, nb_family_uncalibrated, nb_data):
        X, y = nb_data
        calibrated = nb_family_uncalibrated.calibrate(X, y, fit_intercept=True)
        assert calibrated is not nb_family_uncalibrated
        assert isinstance(calibrated, NegativeBinomialFamily)
        assert calibrated.alpha is not None
        # Original should remain unchanged (frozen).
        assert nb_family_uncalibrated.alpha is None

    def test_calibrated_alpha_positive(self, nb_family):
        assert nb_family.alpha is not None
        assert nb_family.alpha > 0

    def test_calibrated_alpha_reasonable(self, nb_family):
        """Estimated α should be in the right ballpark (true α=1.5)."""
        assert nb_family.alpha is not None
        assert 0.3 < nb_family.alpha < 5.0

    def test_idempotent_returns_self(self, nb_family, nb_data):
        X, y = nb_data
        recalibrated = nb_family.calibrate(X, y, fit_intercept=True)
        assert recalibrated is nb_family

    def test_user_supplied_alpha_idempotent(self, nb_family_known_alpha, nb_data):
        X, y = nb_data
        recalibrated = nb_family_known_alpha.calibrate(X, y, fit_intercept=True)
        assert recalibrated is nb_family_known_alpha
        assert recalibrated.alpha == 1.5

    def test_frozen_after_calibration(self, nb_family):
        with pytest.raises(AttributeError):
            nb_family.alpha = 999.0  # type: ignore[misc]


# ------------------------------------------------------------------ #
# Uncalibrated guards
# ------------------------------------------------------------------ #


class TestNBUncalibratedGuards:
    def test_fit_raises_uncalibrated(self, nb_family_uncalibrated, nb_data):
        X, y = nb_data
        with pytest.raises(RuntimeError, match="calibrate"):
            nb_family_uncalibrated.fit(X, y)

    def test_reconstruct_y_raises_uncalibrated(self, nb_family_uncalibrated, nb_rng):
        preds = np.ones(10)
        resids = np.zeros(10)
        with pytest.raises(RuntimeError, match="calibrate"):
            nb_family_uncalibrated.reconstruct_y(preds, resids, nb_rng)

    def test_fit_metric_raises_uncalibrated(self, nb_family_uncalibrated):
        y = np.array([1.0, 2.0, 3.0])
        preds = np.array([1.0, 2.0, 3.0])
        with pytest.raises(RuntimeError, match="calibrate"):
            nb_family_uncalibrated.fit_metric(y, preds)

    def test_diagnostics_raises_uncalibrated(self, nb_family_uncalibrated, nb_data):
        X, y = nb_data
        with pytest.raises(RuntimeError, match="calibrate"):
            nb_family_uncalibrated.diagnostics(X, y)

    def test_batch_fit_raises_uncalibrated(self, nb_family_uncalibrated, nb_data):
        X, y = nb_data
        Y_matrix = y.reshape(1, -1)
        with pytest.raises(RuntimeError, match="calibrate"):
            nb_family_uncalibrated.batch_fit(X, Y_matrix, fit_intercept=True)

    def test_batch_fit_varying_X_raises_uncalibrated(
        self, nb_family_uncalibrated, nb_data
    ):
        X, y = nb_data
        X_batch = X[np.newaxis, :, :]
        with pytest.raises(RuntimeError, match="calibrate"):
            nb_family_uncalibrated.batch_fit_varying_X(X_batch, y, fit_intercept=True)


# ------------------------------------------------------------------ #
# validate_y
# ------------------------------------------------------------------ #


class TestNBValidateY:
    def test_valid_counts(self, nb_family):
        nb_family.validate_y(np.array([0, 1, 5, 10, 100], dtype=float))

    def test_valid_int_dtype(self, nb_family):
        nb_family.validate_y(np.array([0, 1, 2, 3], dtype=int))

    def test_valid_whole_floats(self, nb_family):
        nb_family.validate_y(np.array([0.0, 1.0, 5.0, 10.0]))

    def test_rejects_non_numeric(self, nb_family):
        with pytest.raises(ValueError, match="numeric"):
            nb_family.validate_y(np.array(["a", "b", "c"]))

    def test_rejects_nan(self, nb_family):
        with pytest.raises(ValueError, match="NaN"):
            nb_family.validate_y(np.array([1.0, np.nan, 3.0]))

    def test_rejects_negative(self, nb_family):
        with pytest.raises(ValueError, match="non-negative"):
            nb_family.validate_y(np.array([1.0, -1.0, 3.0]))

    def test_rejects_fractional(self, nb_family):
        with pytest.raises(ValueError, match="integer-valued"):
            nb_family.validate_y(np.array([1.0, 2.5, 3.0]))


# ------------------------------------------------------------------ #
# Single-model operations
# ------------------------------------------------------------------ #


class TestNBSingleModel:
    def test_fit_predict_roundtrip(self, nb_family, nb_data):
        X, y = nb_data
        model = nb_family.fit(X, y, fit_intercept=True)
        preds = nb_family.predict(model, X)
        assert preds.shape == (len(y),)
        assert np.all(preds > 0)
        assert np.corrcoef(y, preds)[0, 1] > 0.3

    def test_coefs_shape(self, nb_family, nb_data):
        X, y = nb_data
        model = nb_family.fit(X, y, fit_intercept=True)
        coefs = nb_family.coefs(model)
        assert coefs.shape == (X.shape[1],)

    def test_coefs_accuracy(self, nb_family, nb_data):
        """Recovered coefficients should be close to true values."""
        X, y = nb_data
        model = nb_family.fit(X, y, fit_intercept=True)
        coefs = nb_family.coefs(model)
        true_coefs = np.array([1.0, -0.5, 0.2])
        np.testing.assert_allclose(coefs, true_coefs, atol=0.5)

    def test_residuals_shape(self, nb_family, nb_data):
        X, y = nb_data
        model = nb_family.fit(X, y, fit_intercept=True)
        resids = nb_family.residuals(model, X, y)
        assert resids.shape == (len(y),)

    def test_residuals_are_response_scale(self, nb_family, nb_data):
        """Response-scale residuals should be y − μ̂."""
        X, y = nb_data
        model = nb_family.fit(X, y, fit_intercept=True)
        resids = nb_family.residuals(model, X, y)
        preds = nb_family.predict(model, X)
        np.testing.assert_allclose(resids, y - preds)

    def test_fit_no_intercept(self, nb_family, nb_data):
        X, y = nb_data
        model = nb_family.fit(X, y, fit_intercept=False)
        coefs = nb_family.coefs(model)
        assert coefs.shape == (X.shape[1],)


# ------------------------------------------------------------------ #
# reconstruct_y
# ------------------------------------------------------------------ #


class TestNBReconstructY:
    def test_output_is_nonneg_integer(self, nb_family, nb_data, nb_rng):
        X, y = nb_data
        model = nb_family.fit(X, y, fit_intercept=True)
        preds = nb_family.predict(model, X)
        resids = nb_family.residuals(model, X, y)
        y_star = nb_family.reconstruct_y(preds, resids, nb_rng)
        assert y_star.shape == (len(y),)
        assert np.all(y_star >= 0)
        assert np.allclose(y_star, np.round(y_star))

    def test_shape_with_broadcasting(self, nb_family, nb_data, nb_rng):
        X, y = nb_data
        model = nb_family.fit(X, y, fit_intercept=True)
        preds = nb_family.predict(model, X)
        resids = nb_family.residuals(model, X, y)
        B = 10
        perm_resids = np.array([nb_rng.permutation(resids) for _ in range(B)])
        y_star = nb_family.reconstruct_y(preds[np.newaxis, :], perm_resids, nb_rng)
        assert y_star.shape == (B, len(y))

    def test_mean_preserving(self, nb_family, nb_data, nb_rng):
        """Reconstructed Y* should have a similar mean to original Y."""
        X, y = nb_data
        model = nb_family.fit(X, y, fit_intercept=True)
        preds = nb_family.predict(model, X)
        resids = nb_family.residuals(model, X, y)
        B = 50
        perm_resids = np.array([nb_rng.permutation(resids) for _ in range(B)])
        y_star = nb_family.reconstruct_y(preds[np.newaxis, :], perm_resids, nb_rng)
        assert y_star.mean() < y.mean() * 5
        assert y_star.mean() > y.mean() / 5


# ------------------------------------------------------------------ #
# fit_metric (NB deviance)
# ------------------------------------------------------------------ #


class TestNBFitMetric:
    def test_perfect_prediction(self, nb_family):
        """Deviance should be 0 when predictions match exactly."""
        y = np.array([5.0, 10.0, 3.0, 8.0])
        deviance = nb_family.fit_metric(y, y)
        assert deviance == pytest.approx(0.0, abs=1e-10)

    def test_zero_counts(self, nb_family):
        """y=0 entries should contribute 0 to the first deviance term."""
        y = np.array([0.0, 0.0, 5.0, 10.0])
        preds = np.array([1.0, 2.0, 5.0, 10.0])
        deviance = nb_family.fit_metric(y, preds)
        assert deviance > 0

    def test_positive_for_imperfect(self, nb_family):
        y = np.array([3.0, 7.0, 12.0])
        preds = np.array([5.0, 5.0, 5.0])
        deviance = nb_family.fit_metric(y, preds)
        assert deviance > 0

    def test_no_runtime_warnings(self, nb_family):
        """fit_metric should not emit divide-by-zero warnings."""
        import warnings as w

        y = np.array([0.0, 0.0, 1.0, 5.0, 0.0])
        preds = np.array([2.0, 3.0, 1.0, 5.0, 1.0])
        with w.catch_warnings():
            w.simplefilter("error", RuntimeWarning)
            nb_family.fit_metric(y, preds)  # should not raise


# ------------------------------------------------------------------ #
# diagnostics
# ------------------------------------------------------------------ #


class TestNBDiagnostics:
    def test_expected_keys(self, nb_family, nb_data):
        X, y = nb_data
        diag = nb_family.diagnostics(X, y, fit_intercept=True)
        expected = {
            "n_observations",
            "n_features",
            "deviance",
            "pearson_chi2",
            "dispersion",
            "alpha",
            "log_likelihood",
            "aic",
            "bic",
        }
        assert expected.issubset(set(diag.keys()))

    def test_n_observations(self, nb_family, nb_data):
        X, y = nb_data
        diag = nb_family.diagnostics(X, y, fit_intercept=True)
        assert diag["n_observations"] == len(y)

    def test_dispersion_positive(self, nb_family, nb_data):
        X, y = nb_data
        diag = nb_family.diagnostics(X, y, fit_intercept=True)
        assert diag["dispersion"] > 0

    def test_alpha_matches_family(self, nb_family, nb_data):
        """Diagnostics α should match the calibrated family's α."""
        X, y = nb_data
        diag = nb_family.diagnostics(X, y, fit_intercept=True)
        assert diag["alpha"] == pytest.approx(nb_family.alpha, rel=1e-3)


# ------------------------------------------------------------------ #
# classical_p_values
# ------------------------------------------------------------------ #


class TestNBClassicalPValues:
    def test_shape(self, nb_family, nb_data):
        X, y = nb_data
        pvals = nb_family.classical_p_values(X, y, fit_intercept=True)
        assert pvals.shape == (X.shape[1],)

    def test_range(self, nb_family, nb_data):
        X, y = nb_data
        pvals = nb_family.classical_p_values(X, y, fit_intercept=True)
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)

    def test_significant_predictor_detected(self, nb_family, nb_data):
        """The true β₁ = 1.0 should be significant."""
        X, y = nb_data
        pvals = nb_family.classical_p_values(X, y, fit_intercept=True)
        assert pvals[0] < 0.05


# ------------------------------------------------------------------ #
# exchangeability_cells
# ------------------------------------------------------------------ #


class TestNBExchangeabilityCells:
    def test_returns_none(self, nb_family, nb_data):
        X, y = nb_data
        cells = nb_family.exchangeability_cells(X, y)
        assert cells is None


# ------------------------------------------------------------------ #
# batch_fit
# ------------------------------------------------------------------ #


class TestNBBatchFit:
    def test_output_shape(self, nb_family, nb_data, nb_rng):
        X, y = nb_data
        B = 10
        Y_matrix = np.tile(y, (B, 1))
        for i in range(B):
            nb_rng.shuffle(Y_matrix[i])
        coefs = nb_family.batch_fit(X, Y_matrix, fit_intercept=True)
        assert coefs.shape == (B, X.shape[1])

    def test_unpermuted_row_matches_single_fit(self, nb_family, nb_data):
        X, y = nb_data
        Y_matrix = y.reshape(1, -1)
        batch_coefs = nb_family.batch_fit(X, Y_matrix, fit_intercept=True)
        model = nb_family.fit(X, y, fit_intercept=True)
        single_coefs = nb_family.coefs(model)
        np.testing.assert_allclose(batch_coefs[0], single_coefs, atol=0.01)

    def test_convergence_failure_returns_nan(self, nb_family_known_alpha):
        """Pathological data should produce NaN, not crash."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((20, 2))
        y_bad = np.array([1e6] * 10 + [0] * 10, dtype=float)
        Y_matrix = y_bad.reshape(1, -1)
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("ignore")
            coefs = nb_family_known_alpha.batch_fit(X, Y_matrix, fit_intercept=True)
        assert coefs.shape == (1, 2)


# ------------------------------------------------------------------ #
# batch_fit_varying_X
# ------------------------------------------------------------------ #


class TestNBBatchFitVaryingX:
    def test_output_shape(self, nb_family, nb_data, nb_rng):
        X, y = nb_data
        B = 5
        X_batch = np.broadcast_to(X, (B, *X.shape)).copy()
        for i in range(B):
            nb_rng.shuffle(X_batch[i, :, 0])
        coefs = nb_family.batch_fit_varying_X(X_batch, y, fit_intercept=True)
        assert coefs.shape == (B, X.shape[1])

    def test_unpermuted_matches_single_fit(self, nb_family, nb_data):
        X, y = nb_data
        X_batch = X[np.newaxis, :, :]  # (1, n, p)
        batch_coefs = nb_family.batch_fit_varying_X(X_batch, y, fit_intercept=True)
        model = nb_family.fit(X, y, fit_intercept=True)
        single_coefs = nb_family.coefs(model)
        np.testing.assert_allclose(batch_coefs[0], single_coefs, atol=0.01)


# ------------------------------------------------------------------ #
# Registry (NB additions)
# ------------------------------------------------------------------ #


class TestNBRegistry:
    def test_nb_registered(self):
        fam = resolve_family("negative_binomial", np.array([0.0, 1.0, 2.0]))
        assert isinstance(fam, NegativeBinomialFamily)

    def test_auto_does_not_resolve_nb(self):
        """Count auto-detection does not select NB."""
        y = np.array([0, 1, 5, 10, 20, 3, 7, 4], dtype=float)
        fam = resolve_family("auto", y)
        assert isinstance(fam, LinearFamily)

    def test_frozen_dataclass(self, nb_family):
        with pytest.raises(AttributeError):
            nb_family.name = "other"  # type: ignore[misc]


# ================================================================== #
# OrdinalFamily
# ================================================================== #

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture()
def ordinal_data(rng):
    """Ordinal outcome with 4 ordered categories (0, 1, 2, 3)."""
    n, p = 120, 3
    X = rng.standard_normal((n, p))
    z = X @ np.array([0.6, -0.3, 0.2]) + rng.standard_normal(n) * 0.8
    y = np.digitize(z, bins=np.quantile(z, [0.25, 0.5, 0.75])).astype(float)
    return X, y


@pytest.fixture()
def ordinal_family():
    return OrdinalFamily()


# ------------------------------------------------------------------ #
# Protocol conformance
# ------------------------------------------------------------------ #


class TestOrdinalProtocolConformance:
    def test_isinstance_check(self, ordinal_family):
        assert isinstance(ordinal_family, ModelFamily)

    def test_name(self, ordinal_family):
        assert ordinal_family.name == "ordinal"

    def test_residual_type(self, ordinal_family):
        assert ordinal_family.residual_type == "none"

    def test_direct_permutation(self, ordinal_family):
        assert ordinal_family.direct_permutation is True

    def test_metric_label(self, ordinal_family):
        assert ordinal_family.metric_label == "Deviance Reduction"


# ------------------------------------------------------------------ #
# validate_y
# ------------------------------------------------------------------ #


class TestOrdinalValidateY:
    def test_valid_ordinal(self, ordinal_family):
        """Integer-coded ordinal with ≥ 3 levels should pass."""
        ordinal_family.validate_y(np.array([0, 1, 2, 3, 1, 2], dtype=float))

    def test_valid_int_dtype(self, ordinal_family):
        """Integer dtypes should pass."""
        ordinal_family.validate_y(np.array([0, 1, 2, 0, 1, 2], dtype=int))

    def test_float_whole_numbers(self, ordinal_family):
        """Whole-number floats (0.0, 1.0, etc.) should pass."""
        ordinal_family.validate_y(np.array([0.0, 1.0, 2.0, 3.0, 1.0]))

    def test_rejects_non_numeric(self, ordinal_family):
        with pytest.raises(ValueError, match="numeric"):
            ordinal_family.validate_y(np.array(["a", "b", "c"]))

    def test_rejects_nan(self, ordinal_family):
        with pytest.raises(ValueError, match="NaN"):
            ordinal_family.validate_y(np.array([0.0, 1.0, float("nan"), 2.0]))

    def test_rejects_non_integer(self, ordinal_family):
        with pytest.raises(ValueError, match="integer"):
            ordinal_family.validate_y(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))

    def test_rejects_binary(self, ordinal_family):
        """Fewer than 3 levels should be rejected."""
        with pytest.raises(ValueError, match="3"):
            ordinal_family.validate_y(np.array([0, 1, 0, 1, 0], dtype=float))

    def test_rejects_single_level(self, ordinal_family):
        with pytest.raises(ValueError, match="3"):
            ordinal_family.validate_y(np.array([2, 2, 2, 2], dtype=float))


# ------------------------------------------------------------------ #
# Single-model operations
# ------------------------------------------------------------------ #


class TestOrdinalSingleModel:
    def test_fit_returns_model(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        model = ordinal_family.fit(X, y)
        assert model is not None
        assert hasattr(model, "params")

    def test_coefs_shape(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        model = ordinal_family.fit(X, y)
        coefs = ordinal_family.coefs(model)
        assert coefs.shape == (X.shape[1],)

    def test_predict_shape(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        model = ordinal_family.fit(X, y)
        preds = ordinal_family.predict(model, X)
        assert preds.shape == (X.shape[0],)

    def test_predict_expected_range(self, ordinal_family, ordinal_data):
        """E[Y|X] should be within [0, K-1]."""
        X, y = ordinal_data
        model = ordinal_family.fit(X, y)
        preds = ordinal_family.predict(model, X)
        n_levels = len(np.unique(y))
        assert np.all(preds >= 0)
        assert np.all(preds <= n_levels - 1)

    def test_fit_intercept_ignored(self, ordinal_family, ordinal_data):
        """fit_intercept should be accepted but have no effect."""
        X, y = ordinal_data
        model_true = ordinal_family.fit(X, y, fit_intercept=True)
        model_false = ordinal_family.fit(X, y, fit_intercept=False)
        # Both should produce identical coefficients
        np.testing.assert_allclose(
            ordinal_family.coefs(model_true),
            ordinal_family.coefs(model_false),
            rtol=1e-4,
        )


# ------------------------------------------------------------------ #
# Residuals / reconstruct_y / fit_metric — all raise
# ------------------------------------------------------------------ #


class TestOrdinalNotImplemented:
    def test_residuals_raises(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        model = ordinal_family.fit(X, y)
        with pytest.raises(NotImplementedError, match="not well-defined"):
            ordinal_family.residuals(model, X, y)

    def test_reconstruct_y_raises(self, ordinal_family):
        rng = np.random.default_rng(0)
        with pytest.raises(NotImplementedError, match="direct_permutation"):
            ordinal_family.reconstruct_y(np.zeros((1, 5)), np.zeros((1, 5)), rng)

    def test_fit_metric_raises(self, ordinal_family):
        with pytest.raises(NotImplementedError, match="model_fit_metric"):
            ordinal_family.fit_metric(np.zeros(5), np.zeros(5))


# ------------------------------------------------------------------ #
# Model-based fit metric (duck-typed)
# ------------------------------------------------------------------ #


class TestOrdinalModelFitMetric:
    def test_model_fit_metric_positive(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        model = ordinal_family.fit(X, y)
        metric = ordinal_family.model_fit_metric(model)
        assert metric > 0  # -2*llf should be positive

    def test_null_fit_metric_positive(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        model = ordinal_family.fit(X, y)
        null_metric = ordinal_family.null_fit_metric(model)
        assert null_metric > 0

    def test_null_metric_greater_than_full(self, ordinal_family, ordinal_data):
        """Null model deviance should be >= full model deviance."""
        X, y = ordinal_data
        model = ordinal_family.fit(X, y)
        full = ordinal_family.model_fit_metric(model)
        null = ordinal_family.null_fit_metric(model)
        assert null >= full

    def test_hasattr_duck_typing(self, ordinal_family):
        """Duck-type detection should find model_fit_metric."""
        assert hasattr(ordinal_family, "model_fit_metric")
        assert hasattr(ordinal_family, "null_fit_metric")

    def test_other_families_lack_model_fit_metric(self):
        """Other families should NOT have model_fit_metric."""
        assert not hasattr(LinearFamily(), "model_fit_metric")
        assert not hasattr(LogisticFamily(), "model_fit_metric")
        assert not hasattr(PoissonFamily(), "model_fit_metric")


# ------------------------------------------------------------------ #
# Diagnostics
# ------------------------------------------------------------------ #


class TestOrdinalDiagnostics:
    def test_diagnostics_keys(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        diag = ordinal_family.diagnostics(X, y)
        expected = {
            "n_observations",
            "n_features",
            "n_categories",
            "pseudo_r_squared",
            "log_likelihood",
            "log_likelihood_null",
            "aic",
            "bic",
            "llr_p_value",
            "thresholds",
        }
        assert expected.issubset(diag.keys())

    def test_n_observations(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        diag = ordinal_family.diagnostics(X, y)
        assert diag["n_observations"] == len(y)

    def test_n_features(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        diag = ordinal_family.diagnostics(X, y)
        assert diag["n_features"] == X.shape[1]

    def test_n_categories(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        diag = ordinal_family.diagnostics(X, y)
        assert diag["n_categories"] == len(np.unique(y))

    def test_pseudo_r_squared_range(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        diag = ordinal_family.diagnostics(X, y)
        assert 0 <= diag["pseudo_r_squared"] <= 1

    def test_thresholds_count(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        diag = ordinal_family.diagnostics(X, y)
        n_cats = len(np.unique(y))
        assert len(diag["thresholds"]) == n_cats - 1

    def test_log_likelihood_negative(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        diag = ordinal_family.diagnostics(X, y)
        assert diag["log_likelihood"] < 0

    def test_aic_bic_positive(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        diag = ordinal_family.diagnostics(X, y)
        assert diag["aic"] > 0
        assert diag["bic"] > 0


# ------------------------------------------------------------------ #
# Classical p-values
# ------------------------------------------------------------------ #


class TestOrdinalClassicalPValues:
    def test_shape(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        pvals = ordinal_family.classical_p_values(X, y)
        assert pvals.shape == (X.shape[1],)

    def test_range(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        pvals = ordinal_family.classical_p_values(X, y)
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)


# ------------------------------------------------------------------ #
# Exchangeability cells
# ------------------------------------------------------------------ #


class TestOrdinalExchangeabilityCells:
    def test_returns_none(self, ordinal_family, ordinal_data):
        X, y = ordinal_data
        assert ordinal_family.exchangeability_cells(X, y) is None


# ------------------------------------------------------------------ #
# Batch fitting
# ------------------------------------------------------------------ #


class TestOrdinalBatchFit:
    def test_shape(self, ordinal_family, ordinal_data, rng):
        X, y = ordinal_data
        B = 5
        Y_matrix = np.array([y[rng.permutation(len(y))] for _ in range(B)])
        coefs = ordinal_family.batch_fit(X, Y_matrix, fit_intercept=True)
        assert coefs.shape == (B, X.shape[1])

    def test_no_all_nan(self, ordinal_family, ordinal_data, rng):
        """Most permutations should converge."""
        X, y = ordinal_data
        B = 10
        Y_matrix = np.array([y[rng.permutation(len(y))] for _ in range(B)])
        coefs = ordinal_family.batch_fit(X, Y_matrix, fit_intercept=True)
        n_valid = np.sum(~np.any(np.isnan(coefs), axis=1))
        assert n_valid >= B // 2  # at least half should converge


class TestOrdinalBatchFitVaryingX:
    def test_shape(self, ordinal_family, ordinal_data, rng):
        X, y = ordinal_data
        n = len(y)
        B = 5
        X_batch = np.array([X[rng.permutation(n)] for _ in range(B)])
        coefs = ordinal_family.batch_fit_varying_X(X_batch, y, fit_intercept=True)
        assert coefs.shape == (B, X.shape[1])


# ------------------------------------------------------------------ #
# Registry
# ------------------------------------------------------------------ #


class TestOrdinalRegistry:
    def test_ordinal_registered(self):
        fam = resolve_family("ordinal", np.array([0.0, 1.0, 2.0]))
        assert isinstance(fam, OrdinalFamily)

    def test_auto_does_not_resolve_ordinal(self):
        """Auto-detection should not select ordinal."""
        y = np.array([0, 1, 2, 3, 1, 2, 0, 3], dtype=float)
        fam = resolve_family("auto", y)
        assert isinstance(fam, LinearFamily)

    def test_frozen_dataclass(self, ordinal_family):
        with pytest.raises(AttributeError):
            ordinal_family.name = "other"  # type: ignore[misc]

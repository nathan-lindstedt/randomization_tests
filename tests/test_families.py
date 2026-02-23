"""Tests for the ModelFamily protocol, LinearFamily, and LogisticFamily."""

import numpy as np
import pytest

from randomization_tests.families import (
    LinearFamily,
    LogisticFamily,
    ModelFamily,
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

"""Tests for the ModelFamily protocol and LinearFamily implementation."""

import numpy as np
import pytest

from randomization_tests.families import (
    LinearFamily,
    ModelFamily,
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
            resolve_family("poisson", np.array([1.0, 2.0]))

    def test_register_non_protocol_raises(self):
        with pytest.raises(TypeError, match="does not implement"):
            register_family("bad", int)

    def test_frozen_dataclass(self, family):
        with pytest.raises(AttributeError):
            family.name = "other"  # type: ignore[misc]

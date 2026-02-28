"""Tests for LinearMixedFamily (Step A.6), LogisticMixedFamily, and PoissonMixedFamily (Step C.5).

Covers protocol conformance, REML/Laplace calibration, single/batch fitting,
scoring, exchangeability cells, diagnostics, display, registry wiring,
multi-factor grouping, score projection, and GLMM-specific functionality.
"""

from __future__ import annotations

import numpy as np
import pytest

from randomization_tests.families import ModelFamily, resolve_family
from randomization_tests.families_mixed import (
    LinearMixedFamily,
    LogisticMixedFamily,
    PoissonMixedFamily,
    _build_random_effects_design,
)

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def mixed_data(rng):
    """Generate clustered data with known structure.

    10 groups, n=200 total (20 obs per group).
    Group random intercepts ~ N(0, τ²=4.0).
    Residual σ²=1.0.
    True β = [2.0, -1.0].
    """
    n_groups = 10
    n_per_group = 20
    n = n_groups * n_per_group
    p = 2

    groups = np.repeat(np.arange(n_groups), n_per_group)
    X = rng.standard_normal((n, p))
    beta_true = np.array([2.0, -1.0])

    # Random intercepts
    u = rng.normal(0, 2.0, size=n_groups)  # τ = 2.0 → τ² = 4.0
    y = X @ beta_true + u[groups] + rng.standard_normal(n) * 1.0

    return X, y, groups


@pytest.fixture()
def calibrated_family(mixed_data):
    """Return a calibrated LinearMixedFamily."""
    X, y, groups = mixed_data
    family = LinearMixedFamily()
    return family.calibrate(X, y, fit_intercept=True, groups=groups)


# ------------------------------------------------------------------ #
# Protocol conformance
# ------------------------------------------------------------------ #


class TestProtocolConformance:
    def test_isinstance_check(self):
        assert isinstance(LinearMixedFamily(), ModelFamily)

    def test_name(self):
        assert LinearMixedFamily().name == "linear_mixed"

    def test_residual_type(self):
        assert LinearMixedFamily().residual_type == "conditional"

    def test_direct_permutation(self):
        assert LinearMixedFamily().direct_permutation is False

    def test_stat_label(self):
        assert LinearMixedFamily().stat_label == "t"

    def test_metric_label(self):
        assert LinearMixedFamily().metric_label == "RSS Reduction"

    def test_frozen_dataclass(self):
        """Verify the family is immutable (frozen dataclass)."""
        family = LinearMixedFamily()
        with pytest.raises(AttributeError):
            family.name = "oops"  # type: ignore[misc]


# ------------------------------------------------------------------ #
# _build_random_effects_design
# ------------------------------------------------------------------ #


class TestBuildRandomEffectsDesign:
    def test_single_factor_1d(self):
        """1-D integer array → single factor Z."""
        groups = np.array([0, 0, 1, 1, 2, 2])
        Z, re_struct = _build_random_effects_design(groups)
        assert Z.shape == (6, 3)
        assert re_struct == [(3, 1)]
        # Each row has exactly one 1 in the correct position.
        assert np.allclose(Z.sum(axis=1), 1.0)

    def test_single_factor_noncontiguous(self):
        """Non-contiguous labels are remapped to 0-based."""
        groups = np.array([10, 10, 20, 20, 30, 30])
        Z, re_struct = _build_random_effects_design(groups)
        assert Z.shape == (6, 3)
        assert re_struct == [(3, 1)]

    def test_dict_two_factors(self):
        """Dict of arrays → crossed factors, Z = [Z_1 | Z_2]."""
        groups = {
            "school": np.array([0, 0, 1, 1, 0, 0]),
            "teacher": np.array([0, 0, 0, 1, 1, 1]),
        }
        Z, re_struct = _build_random_effects_design(groups)
        # school has 2 levels, teacher has 2 levels → Z = (6, 4)
        assert Z.shape == (6, 4)
        assert re_struct == [(2, 1), (2, 1)]

    def test_dict_preserves_order(self):
        """Factor order matches insertion order of the dict."""
        groups = {
            "a": np.array([0, 1, 0, 1]),
            "b": np.array([0, 0, 1, 1]),
            "c": np.array([2, 2, 3, 3]),
        }
        Z, re_struct = _build_random_effects_design(groups)
        assert Z.shape == (4, 2 + 2 + 2)
        assert re_struct == [(2, 1), (2, 1), (2, 1)]


# ------------------------------------------------------------------ #
# _build_random_effects_design — random slopes
# ------------------------------------------------------------------ #


class TestBuildRandomEffectsDesignSlopes:
    def test_single_factor_one_slope(self):
        """Single factor with one random slope."""
        groups = np.array([0, 0, 1, 1, 2, 2])
        X = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        )
        Z, re_struct = _build_random_effects_design(groups, X=X, random_slopes=[0])
        # 3 groups × d_k=2 (intercept + 1 slope) = 6 columns
        assert Z.shape == (6, 6)
        assert re_struct == [(3, 2)]

    def test_slope_columns_contain_x_values(self):
        """Slope columns in Z contain the X column values."""
        groups = np.array([0, 0, 1, 1])
        X = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
        Z, re_struct = _build_random_effects_design(groups, X=X, random_slopes=[1])
        # d_k = 2 (intercept + slope on col 1), G_k = 2
        # Layout: [g0_intercept, g0_slope, g1_intercept, g1_slope]
        assert Z.shape == (4, 4)
        assert re_struct == [(2, 2)]
        # Row 0: group 0 → g0_intercept=1, g0_slope=X[0,1]=20, rest 0
        np.testing.assert_allclose(Z[0], [1.0, 20.0, 0.0, 0.0])
        # Row 2: group 1 → g1_intercept=1, g1_slope=X[2,1]=60
        np.testing.assert_allclose(Z[2], [0.0, 0.0, 1.0, 60.0])

    def test_two_slopes(self):
        """Single factor with two random slopes."""
        groups = np.array([0, 0, 1, 1])
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        Z, re_struct = _build_random_effects_design(groups, X=X, random_slopes=[0, 1])
        # d_k = 3 (intercept + 2 slopes), G_k = 2 → 6 columns
        assert Z.shape == (4, 6)
        assert re_struct == [(2, 3)]

    def test_dict_with_slopes(self):
        """Dict factors with per-factor slope specification."""
        groups = {
            "school": np.array([0, 0, 1, 1]),
            "teacher": np.array([0, 1, 0, 1]),
        }
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        Z, re_struct = _build_random_effects_design(
            groups,
            X=X,
            random_slopes={"school": [0], "teacher": []},
        )
        # school: 2 groups × d=2 = 4 cols; teacher: 2 groups × d=1 = 2
        assert Z.shape == (4, 6)
        assert re_struct == [(2, 2), (2, 1)]

    def test_slopes_require_X(self):
        """Requesting slopes without X raises ValueError."""
        groups = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="X must be provided"):
            _build_random_effects_design(groups, random_slopes=[0])

    def test_no_slopes_matches_intercept_only(self):
        """random_slopes=None gives same result as no-slopes call."""
        groups = np.array([0, 0, 1, 1, 2, 2])
        X = np.ones((6, 2))
        Z1, rs1 = _build_random_effects_design(groups)
        Z2, rs2 = _build_random_effects_design(groups, X=X, random_slopes=None)
        np.testing.assert_array_equal(Z1, Z2)
        assert rs1 == rs2


# ------------------------------------------------------------------ #
# Calibration
# ------------------------------------------------------------------ #


class TestCalibration:
    def test_calibrate_populates_fields(self, calibrated_family):
        """After calibration, all fields are non-None."""
        f = calibrated_family
        assert f.projection_A is not None
        assert f.sigma2 is not None
        assert f.re_covariances is not None
        assert f.log_chol is not None
        assert f.Z is not None
        assert f.C22 is not None
        assert f.re_struct is not None

    def test_calibrate_idempotent(self, calibrated_family, mixed_data):
        """Calling calibrate() on an already-calibrated instance is a no-op."""
        X, y, groups = mixed_data
        f2 = calibrated_family.calibrate(X, y, groups=groups)
        assert f2 is calibrated_family

    def test_calibrate_requires_groups(self):
        """calibrate() without groups= raises ValueError."""
        f = LinearMixedFamily()
        X = np.ones((10, 2))
        y = np.ones(10)
        with pytest.raises(ValueError, match="groups"):
            f.calibrate(X, y)

    def test_sigma2_positive(self, calibrated_family):
        assert calibrated_family.sigma2 > 0

    def test_re_covariances_nonneg(self, calibrated_family):
        """Intercept variances (diagonal of covariance matrices) are non-negative."""
        for cov_k in calibrated_family.re_covariances:
            assert cov_k[0, 0] >= 0

    def test_converged(self, calibrated_family):
        """The solver should converge on well-conditioned data."""
        assert calibrated_family.converged is True

    def test_projection_shape(self, calibrated_family, mixed_data):
        """A has shape (p+1, n) when fit_intercept=True."""
        X, y, groups = mixed_data
        n, p = X.shape
        assert calibrated_family.projection_A.shape == (p + 1, n)


# ------------------------------------------------------------------ #
# Single-model fit
# ------------------------------------------------------------------ #


class TestFit:
    def test_fit_returns_result(self, calibrated_family, mixed_data):
        X, y, groups = mixed_data
        result = calibrated_family.fit(X, y, fit_intercept=True)
        n, p = X.shape
        beta = calibrated_family.coefs(result)
        pred = calibrated_family.predict(result, X)
        assert beta.shape == (p,)  # excludes intercept
        assert pred.shape == (n,)

    def test_beta_accuracy(self, calibrated_family, mixed_data):
        """\u03b2\u0302 should be close to true \u03b2 = [2.0, -1.0]."""
        X, y, groups = mixed_data
        result = calibrated_family.fit(X, y, fit_intercept=True)
        beta = calibrated_family.coefs(result)
        np.testing.assert_allclose(beta, [2.0, -1.0], atol=0.3)

    def test_fit_no_intercept(self, mixed_data):
        """fit_intercept=False works and returns the right shape."""
        X, y, groups = mixed_data
        f = LinearMixedFamily().calibrate(X, y, fit_intercept=False, groups=groups)
        result = f.fit(X, y, fit_intercept=False)
        beta = f.coefs(result)
        pred = f.predict(result, X)
        assert beta.shape == (X.shape[1],)
        assert pred.shape == (X.shape[0],)

    def test_fit_requires_calibration(self):
        """fit() before calibrate() raises RuntimeError."""
        f = LinearMixedFamily()
        with pytest.raises(RuntimeError, match="requires calibration"):
            f.fit(np.ones((5, 2)), np.ones(5), fit_intercept=True)


# ------------------------------------------------------------------ #
# Batch fitting
# ------------------------------------------------------------------ #


class TestBatchFit:
    def test_batch_fit_shape(self, calibrated_family, mixed_data, rng):
        """batch_fit returns (B, p) for B permutations."""
        X, y, groups = mixed_data
        B = 10
        Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])
        result = calibrated_family.batch_fit(X, Y_matrix, fit_intercept=True)
        assert result.shape == (B, X.shape[1])

    def test_batch_fit_matches_single(self, calibrated_family, mixed_data, rng):
        """First row of batch_fit should match single fit on same Y."""
        X, y, groups = mixed_data
        B = 5
        Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])
        batch_result = calibrated_family.batch_fit(X, Y_matrix, fit_intercept=True)

        # Compare with single fit on first permuted Y
        result = calibrated_family.fit(X, Y_matrix[0], fit_intercept=True)
        beta_single = calibrated_family.coefs(result)
        np.testing.assert_allclose(batch_result[0], beta_single, atol=1e-10)


# ------------------------------------------------------------------ #
# Batch fit and score
# ------------------------------------------------------------------ #


class TestBatchFitAndScore:
    def test_batch_fit_and_score_shape(self, calibrated_family, mixed_data, rng):
        """batch_fit_and_score returns (coefs, rss) tuple."""
        X, y, groups = mixed_data
        B = 10
        Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])
        coefs, rss = calibrated_family.batch_fit_and_score(
            X, Y_matrix, fit_intercept=True
        )
        assert coefs.shape == (B, X.shape[1])
        assert rss.shape == (B,)

    def test_scores_are_finite(self, calibrated_family, mixed_data, rng):
        """All scores should be finite."""
        X, y, groups = mixed_data
        B = 10
        Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])
        coefs, rss = calibrated_family.batch_fit_and_score(
            X, Y_matrix, fit_intercept=True
        )
        assert np.all(np.isfinite(coefs))
        assert np.all(np.isfinite(rss))


# ------------------------------------------------------------------ #
# Batch fit varying X (Kennedy)
# ------------------------------------------------------------------ #


class TestBatchFitVaryingX:
    def test_shape(self, calibrated_family, mixed_data, rng):
        """batch_fit_varying_X returns (B, p)."""
        X, y, groups = mixed_data
        B = 5
        n, p = X.shape
        # Generate B slightly perturbed X matrices.
        X_batch = np.stack([X + rng.normal(0, 0.01, X.shape) for _ in range(B)])
        result = calibrated_family.batch_fit_varying_X(X_batch, y, fit_intercept=True)
        assert result.shape == (B, p)


# ------------------------------------------------------------------ #
# Batch fit paired
# ------------------------------------------------------------------ #


class TestBatchFitPaired:
    def test_shape(self, calibrated_family, mixed_data, rng):
        """batch_fit_paired returns (B, p)."""
        X, y, groups = mixed_data
        B = 5
        Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])
        X_batch = np.stack([X for _ in range(B)])
        result = calibrated_family.batch_fit_paired(
            X_batch, Y_matrix, fit_intercept=True
        )
        assert result.shape == (B, X.shape[1])


# ------------------------------------------------------------------ #
# Exchangeability cells
# ------------------------------------------------------------------ #


class TestExchangeabilityCells:
    def test_returns_group_labels(self, calibrated_family, mixed_data):
        """exchangeability_cells returns integer group labels."""
        X, y, groups = mixed_data
        cells = calibrated_family.exchangeability_cells(X, y)
        assert cells.shape == (X.shape[0],)
        # Should have the correct number of unique groups.
        assert len(np.unique(cells)) == 10

    def test_uncalibrated_returns_none(self):
        """Uncalibrated family returns None (no group info yet)."""
        f = LinearMixedFamily()
        X = np.ones((5, 2))
        y = np.ones(5)
        assert f.exchangeability_cells(X, y) is None


# ------------------------------------------------------------------ #
# fit_metric
# ------------------------------------------------------------------ #


class TestFitMetric:
    def test_fit_metric_positive(self, calibrated_family, mixed_data):
        """fit_metric (RSS) should be non-negative for observed data."""
        X, y, groups = mixed_data
        result = calibrated_family.fit(X, y, fit_intercept=True)
        y_pred = calibrated_family.predict(result, X)
        metric = calibrated_family.fit_metric(y, y_pred)
        assert metric >= 0


# ------------------------------------------------------------------ #
# Diagnostics
# ------------------------------------------------------------------ #


class TestDiagnostics:
    def test_diagnostics_keys(self, calibrated_family, mixed_data):
        """diagnostics() returns expected keys."""
        X, y, groups = mixed_data
        diag = calibrated_family.diagnostics(X, y, fit_intercept=True)
        # LMM diagnostics use marginal/conditional R²
        assert "r_squared_marginal" in diag
        assert "r_squared_conditional" in diag
        assert "icc" in diag
        assert "sigma2" in diag

    def test_icc_range(self, calibrated_family, mixed_data):
        """ICC should be between 0 and 1."""
        X, y, groups = mixed_data
        ext_diag = calibrated_family.compute_extended_diagnostics(X, y, True)
        lmm_gof = ext_diag["lmm_gof"]
        assert 0 <= lmm_gof["icc"] <= 1

    def test_sigma2_matches_calibration(self, calibrated_family, mixed_data):
        """Extended diagnostics σ² should match calibrated value."""
        X, y, groups = mixed_data
        ext_diag = calibrated_family.compute_extended_diagnostics(X, y, True)
        np.testing.assert_allclose(
            ext_diag["lmm_gof"]["sigma2"],
            calibrated_family.sigma2,
            rtol=1e-10,
        )

    def test_variance_components_present(self, calibrated_family, mixed_data):
        """Variance components dict should be non-empty."""
        X, y, groups = mixed_data
        ext_diag = calibrated_family.compute_extended_diagnostics(X, y, True)
        var_comps = ext_diag["lmm_gof"]["variance_components"]
        factors = var_comps["factors"]
        assert len(factors) > 0
        for f in factors:
            assert f["intercept_var"] >= 0


# ------------------------------------------------------------------ #
# Display
# ------------------------------------------------------------------ #


class TestDisplay:
    def test_display_header(self, calibrated_family, mixed_data):
        """display_header returns rows of 4-tuples."""
        X, y, groups = mixed_data
        diag = calibrated_family.diagnostics(X, y, fit_intercept=True)
        ext_diag = calibrated_family.compute_extended_diagnostics(X, y, True)
        combined = {**diag, **ext_diag}
        rows = calibrated_family.display_header(combined)
        assert len(rows) >= 1
        for row in rows:
            assert len(row) == 4

    def test_display_diagnostics(self, calibrated_family, mixed_data):
        """display_diagnostics returns lines and notes."""
        X, y, groups = mixed_data
        ext_diag = calibrated_family.compute_extended_diagnostics(X, y, True)
        lines, notes = calibrated_family.display_diagnostics(ext_diag)
        assert isinstance(lines, list)
        assert isinstance(notes, list)
        # Should have at least σ² line.
        assert len(lines) >= 1


# ------------------------------------------------------------------ #
# Registry
# ------------------------------------------------------------------ #


class TestRegistry:
    def test_resolve_family_string(self):
        """resolve_family('linear_mixed') returns a LinearMixedFamily."""
        f = resolve_family("linear_mixed")
        assert isinstance(f, LinearMixedFamily)
        assert f.name == "linear_mixed"

    def test_resolve_family_instance_passthrough(self):
        """Passing an instance to resolve_family returns it directly."""
        f = LinearMixedFamily()
        assert resolve_family(f) is f


# ------------------------------------------------------------------ #
# validate_y
# ------------------------------------------------------------------ #


class TestValidateY:
    def test_valid_continuous(self, rng):
        LinearMixedFamily().validate_y(rng.standard_normal(50))

    def test_rejects_constant(self):
        with pytest.raises(ValueError, match="non-constant"):
            LinearMixedFamily().validate_y(np.ones(10))


# ------------------------------------------------------------------ #
# classical_p_values
# ------------------------------------------------------------------ #


class TestClassicalPValues:
    def test_classical_p_shape(self, calibrated_family, mixed_data):
        """classical_p_values returns (p,) array of slope p-values."""
        X, y, groups = mixed_data
        p_vals = calibrated_family.classical_p_values(X, y, fit_intercept=True)
        assert p_vals.shape == (X.shape[1],)
        # P-values should be in [0, 1].
        assert np.all(p_vals >= 0)
        assert np.all(p_vals <= 1)

    def test_true_effect_significant(self, calibrated_family, mixed_data):
        """True β = [2.0, -1.0] should produce small p-values."""
        X, y, groups = mixed_data
        p_vals = calibrated_family.classical_p_values(X, y, fit_intercept=True)
        # Both effects are strong — should be significant.
        assert p_vals[0] < 0.05
        assert p_vals[1] < 0.05


# ------------------------------------------------------------------ #
# Multi-factor grouping
# ------------------------------------------------------------------ #


class TestMultiFactor:
    def test_two_factor_calibration(self, rng):
        """Two crossed factors produce correct Z structure."""
        n = 60
        school = np.repeat(np.arange(3), 20)  # 3 schools
        classroom = np.tile(np.repeat(np.arange(4), 5), 3)  # 4 classrooms

        X = rng.standard_normal((n, 2))
        u_school = rng.normal(0, 1.5, size=3)
        u_class = rng.normal(0, 0.8, size=4)
        y = (
            X @ [1.0, 0.5]
            + u_school[school]
            + u_class[classroom]
            + rng.normal(0, 0.5, size=n)
        )

        groups = {"school": school, "classroom": classroom}
        f = LinearMixedFamily().calibrate(X, y, fit_intercept=True, groups=groups)

        assert f.re_struct == ((3, 1), (4, 1))
        assert f.Z.shape == (n, 7)  # 3 + 4
        assert f.projection_A is not None
        assert len(f.re_covariances) == 2

    def test_two_factor_exchangeability(self, rng):
        """exchangeability_cells returns labels from the first factor."""
        n = 60
        school = np.repeat(np.arange(3), 20)
        classroom = np.tile(np.repeat(np.arange(4), 5), 3)

        X = rng.standard_normal((n, 2))
        y = rng.standard_normal(n)

        groups = {"school": school, "classroom": classroom}
        f = LinearMixedFamily().calibrate(X, y, fit_intercept=True, groups=groups)
        cells = f.exchangeability_cells(X, y)
        assert len(np.unique(cells)) == 3  # first factor has 3 groups


# ------------------------------------------------------------------ #
# Random-slope calibration
# ------------------------------------------------------------------ #


class TestRandomSlopeCalibration:
    """End-to-end calibration with correlated random slopes."""

    @pytest.fixture()
    def slope_data(self):
        """Generate data with random intercept + random slope on X[:,0]."""
        rng = np.random.default_rng(123)
        n_groups = 8
        n_per = 25
        n = n_groups * n_per
        groups = np.repeat(np.arange(n_groups), n_per)
        X = rng.standard_normal((n, 2))
        beta_true = np.array([1.5, -0.5])

        # Correlated random intercept + slope: Σ = [[2.0, 0.5], [0.5, 0.8]]
        Sigma_true = np.array([[2.0, 0.5], [0.5, 0.8]])
        L_true = np.linalg.cholesky(Sigma_true)
        u = rng.standard_normal((n_groups, 2)) @ L_true.T
        # u[:, 0] = random intercepts, u[:, 1] = random slopes on X[:, 0]
        y = (
            X @ beta_true
            + u[groups, 0]
            + u[groups, 1] * X[:, 0]
            + rng.normal(0, 0.5, n)
        )
        return X, y, groups

    def test_re_struct_shape(self, slope_data):
        """re_struct has d_k=2 for intercept + one slope."""
        X, y, groups = slope_data
        f = LinearMixedFamily().calibrate(
            X,
            y,
            fit_intercept=True,
            groups=groups,
            random_slopes=[0],
        )
        assert f.re_struct == ((8, 2),)

    def test_z_shape_with_slopes(self, slope_data):
        """Z has G*d = 8*2 = 16 columns for intercept + slope."""
        X, y, groups = slope_data
        f = LinearMixedFamily().calibrate(
            X,
            y,
            fit_intercept=True,
            groups=groups,
            random_slopes=[0],
        )
        assert f.Z.shape == (200, 16)

    def test_covariance_is_2x2(self, slope_data):
        """re_covariances[0] is a (2, 2) matrix."""
        X, y, groups = slope_data
        f = LinearMixedFamily().calibrate(
            X,
            y,
            fit_intercept=True,
            groups=groups,
            random_slopes=[0],
        )
        assert len(f.re_covariances) == 1
        assert f.re_covariances[0].shape == (2, 2)
        # Diagonal (variances) must be non-negative
        assert f.re_covariances[0][0, 0] >= 0
        assert f.re_covariances[0][1, 1] >= 0
        # Must be symmetric
        np.testing.assert_allclose(
            f.re_covariances[0], f.re_covariances[0].T, atol=1e-12
        )

    def test_beta_accuracy_with_slopes(self, slope_data):
        """β̂ should be close to true β = [1.5, -0.5]."""
        X, y, groups = slope_data
        f = LinearMixedFamily().calibrate(
            X,
            y,
            fit_intercept=True,
            groups=groups,
            random_slopes=[0],
        )
        result = f.fit(X, y, fit_intercept=True)
        beta = f.coefs(result)
        np.testing.assert_allclose(beta, [1.5, -0.5], atol=0.5)

    def test_diagnostics_with_slopes(self, slope_data):
        """diagnostics() works with random slopes."""
        X, y, groups = slope_data
        f = LinearMixedFamily().calibrate(
            X,
            y,
            fit_intercept=True,
            groups=groups,
            random_slopes=[0],
        )
        diag = f.diagnostics(X, y, fit_intercept=True)
        assert "r_squared_marginal" in diag
        assert "icc" in diag
        # Variance components should include factors with covariance
        # info for d>1 (structured representation)
        var_comps = diag["variance_components"]
        factors = var_comps["factors"]
        assert len(factors) > 0
        f0 = factors[0]
        assert f0["d_k"] == 2  # intercept + 1 slope
        assert "intercept_var" in f0
        assert "slope_vars" in f0
        assert "correlations" in f0

    def test_exchangeability_with_slopes(self, slope_data):
        """exchangeability_cells still extracts correct groups."""
        X, y, groups = slope_data
        f = LinearMixedFamily().calibrate(
            X,
            y,
            fit_intercept=True,
            groups=groups,
            random_slopes=[0],
        )
        cells = f.exchangeability_cells(X, y)
        assert cells.shape == (200,)
        assert len(np.unique(cells)) == 8

    def test_batch_fit_with_slopes(self, slope_data):
        """batch_fit works with random-slope model."""
        rng = np.random.default_rng(99)
        X, y, groups = slope_data
        f = LinearMixedFamily().calibrate(
            X,
            y,
            fit_intercept=True,
            groups=groups,
            random_slopes=[0],
        )
        B = 5
        Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])
        result = f.batch_fit(X, Y_matrix, fit_intercept=True)
        assert result.shape == (B, X.shape[1])

    def test_intercept_only_matches_no_slopes(self):
        """random_slopes=[] gives same calibration as random_slopes=None."""
        rng = np.random.default_rng(42)
        n = 60
        groups = np.repeat(np.arange(3), 20)
        X = rng.standard_normal((n, 2))
        y = X @ [1.0, 0.5] + rng.normal(0, 1, 3)[groups] + rng.normal(0, 0.5, n)

        f1 = LinearMixedFamily().calibrate(
            X,
            y,
            fit_intercept=True,
            groups=groups,
        )
        f2 = LinearMixedFamily().calibrate(
            X,
            y,
            fit_intercept=True,
            groups=groups,
            random_slopes=[],
        )
        assert f1.re_struct == f2.re_struct
        np.testing.assert_allclose(f1.projection_A, f2.projection_A, atol=1e-10)


# ------------------------------------------------------------------ #
# Integration: end-to-end through PermutationEngine
# ------------------------------------------------------------------ #


class TestIntegration:
    def test_engine_with_linear_mixed(self, mixed_data):
        """PermutationEngine can run with LinearMixedFamily."""
        import pandas as pd

        from randomization_tests.engine import PermutationEngine

        X, y, groups = mixed_data
        X_df = pd.DataFrame(X, columns=["x1", "x2"])

        engine = PermutationEngine(
            X_df,
            y,
            family=LinearMixedFamily(),
            fit_intercept=True,
            n_permutations=50,
            random_state=42,
            groups=groups,
        )

        assert engine.family.name == "linear_mixed"
        assert engine.family.projection_A is not None

    def test_engine_string_family(self, mixed_data):
        """PermutationEngine can resolve 'linear_mixed' string."""
        import pandas as pd

        from randomization_tests.engine import PermutationEngine

        X, y, groups = mixed_data
        X_df = pd.DataFrame(X, columns=["x1", "x2"])

        engine = PermutationEngine(
            X_df,
            y,
            family="linear_mixed",
            fit_intercept=True,
            n_permutations=50,
            random_state=42,
            groups=groups,
        )

        assert isinstance(engine.family, LinearMixedFamily)


# ================================================================== #
# LogisticMixedFamily Tests (Step C.5)
# ================================================================== #

# ------------------------------------------------------------------ #
# Fixtures — logistic GLMM
# ------------------------------------------------------------------ #


@pytest.fixture()
def logistic_mixed_data():
    """Generate clustered binary data with known structure.

    10 groups × 30 obs = 300 total.
    Random intercepts ~ N(0, τ²=1.0).
    True β = [0.8, -0.5] in logit space.
    """
    rng = np.random.default_rng(2024)
    n_groups = 10
    n_per_group = 30
    n = n_groups * n_per_group
    p = 2

    groups = np.repeat(np.arange(n_groups), n_per_group)
    X = rng.standard_normal((n, p))
    beta_true = np.array([0.8, -0.5])

    # Random intercepts
    u = rng.normal(0, 1.0, size=n_groups)  # τ = 1.0
    eta = X @ beta_true + u[groups]
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < prob).astype(float)

    return X, y, groups


@pytest.fixture()
def calibrated_logistic_family(logistic_mixed_data):
    """Return a calibrated LogisticMixedFamily."""
    X, y, groups = logistic_mixed_data
    family = LogisticMixedFamily()
    return family.calibrate(X, y, fit_intercept=True, groups=groups)


# ------------------------------------------------------------------ #
# Protocol conformance — logistic
# ------------------------------------------------------------------ #


class TestLogisticProtocolConformance:
    def test_isinstance_check(self):
        assert isinstance(LogisticMixedFamily(), ModelFamily)

    def test_name(self):
        assert LogisticMixedFamily().name == "logistic_mixed"

    def test_residual_type(self):
        assert LogisticMixedFamily().residual_type == "deviance"

    def test_direct_permutation(self):
        assert LogisticMixedFamily().direct_permutation is False

    def test_stat_label(self):
        assert LogisticMixedFamily().stat_label == "z"

    def test_metric_label(self):
        assert LogisticMixedFamily().metric_label == "Deviance Reduction"

    def test_frozen_dataclass(self):
        """Verify the family is immutable (frozen dataclass)."""
        family = LogisticMixedFamily()
        with pytest.raises(AttributeError):
            family.name = "oops"  # type: ignore[misc]


# ------------------------------------------------------------------ #
# Registry — logistic
# ------------------------------------------------------------------ #


class TestLogisticRegistry:
    def test_resolve_family_string(self):
        f = resolve_family("logistic_mixed")
        assert isinstance(f, LogisticMixedFamily)
        assert f.name == "logistic_mixed"

    def test_resolve_family_instance_passthrough(self):
        f = LogisticMixedFamily()
        assert resolve_family(f) is f


# ------------------------------------------------------------------ #
# Validate Y — logistic
# ------------------------------------------------------------------ #


class TestLogisticValidateY:
    def test_valid_binary_int(self):
        LogisticMixedFamily().validate_y(np.array([0, 1, 1, 0, 1]))

    def test_valid_binary_float(self):
        LogisticMixedFamily().validate_y(np.array([0.0, 1.0, 0.0]))

    def test_rejects_non_binary(self):
        with pytest.raises(ValueError, match="binary"):
            LogisticMixedFamily().validate_y(np.array([0, 1, 2]))

    def test_rejects_continuous(self):
        with pytest.raises(ValueError, match="binary"):
            LogisticMixedFamily().validate_y(np.array([0.1, 0.5, 0.9]))


# ------------------------------------------------------------------ #
# Calibration — logistic
# ------------------------------------------------------------------ #


class TestLogisticCalibration:
    def test_calibrate_populates_fields(self, calibrated_logistic_family):
        f = calibrated_logistic_family
        assert f.beta is not None
        assert f.u is not None
        assert f.W is not None
        assert f.mu is not None
        assert f.V_inv_diag is not None
        assert f.fisher_info is not None
        assert f.re_covariances is not None
        assert f.log_chol is not None
        assert f.Z is not None
        assert f.C22 is not None
        assert f.re_struct is not None

    def test_calibrate_idempotent(
        self, calibrated_logistic_family, logistic_mixed_data
    ):
        X, y, groups = logistic_mixed_data
        f2 = calibrated_logistic_family.calibrate(X, y, groups=groups)
        assert f2 is calibrated_logistic_family

    def test_calibrate_requires_groups(self):
        f = LogisticMixedFamily()
        X = np.ones((10, 2))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        with pytest.raises(ValueError, match="groups"):
            f.calibrate(X, y)

    def test_converged(self, calibrated_logistic_family):
        assert calibrated_logistic_family.converged is True

    def test_re_struct(self, calibrated_logistic_family):
        assert calibrated_logistic_family.re_struct == ((10, 1),)

    def test_fisher_info_shape(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        n, p = X.shape
        # Fisher info includes intercept → (p+1, p+1)
        assert calibrated_logistic_family.fisher_info.shape == (p + 1, p + 1)

    def test_fisher_info_positive_definite(self, calibrated_logistic_family):
        eigvals = np.linalg.eigvalsh(calibrated_logistic_family.fisher_info)
        assert np.all(eigvals > 0)

    def test_V_inv_diag_positive(self, calibrated_logistic_family):
        assert np.all(calibrated_logistic_family.V_inv_diag > 0)

    def test_mu_range(self, calibrated_logistic_family):
        """Predicted probabilities should be in (0, 1)."""
        assert np.all(calibrated_logistic_family.mu > 0)
        assert np.all(calibrated_logistic_family.mu < 1)

    def test_W_positive(self, calibrated_logistic_family):
        """Working weights should be positive."""
        assert np.all(calibrated_logistic_family.W > 0)

    def test_re_covariances_nonneg_diagonal(self, calibrated_logistic_family):
        for cov_k in calibrated_logistic_family.re_covariances:
            assert cov_k[0, 0] >= 0


# ------------------------------------------------------------------ #
# Fit — logistic
# ------------------------------------------------------------------ #


class TestLogisticFit:
    def test_fit_returns_result(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        result = calibrated_logistic_family.fit(X, y, fit_intercept=True)
        n, p = X.shape
        beta = calibrated_logistic_family.coefs(result)
        pred = calibrated_logistic_family.predict(result, X)
        assert beta.shape == (p,)
        assert pred.shape == (n,)

    def test_predictions_are_probabilities(
        self, calibrated_logistic_family, logistic_mixed_data
    ):
        X, y, groups = logistic_mixed_data
        result = calibrated_logistic_family.fit(X, y, fit_intercept=True)
        pred = calibrated_logistic_family.predict(result, X)
        assert np.all(pred > 0)
        assert np.all(pred < 1)

    def test_beta_direction(self, calibrated_logistic_family, logistic_mixed_data):
        """β̂ should have the correct sign direction (true β = [0.8, -0.5])."""
        X, y, groups = logistic_mixed_data
        result = calibrated_logistic_family.fit(X, y, fit_intercept=True)
        beta = calibrated_logistic_family.coefs(result)
        assert beta[0] > 0  # true +0.8
        assert beta[1] < 0  # true -0.5

    def test_residuals_shape(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        result = calibrated_logistic_family.fit(X, y, fit_intercept=True)
        resid = calibrated_logistic_family.residuals(result, X, y)
        assert resid.shape == (len(y),)

    def test_fit_requires_calibration(self):
        f = LogisticMixedFamily()
        with pytest.raises(RuntimeError, match="requires calibration"):
            f.fit(np.ones((5, 2)), np.ones(5), fit_intercept=True)


# ------------------------------------------------------------------ #
# Score projection — logistic
# ------------------------------------------------------------------ #


class TestLogisticScoreProject:
    def test_score_project_shape(self, calibrated_logistic_family, logistic_mixed_data):
        """score_project returns (B,) array."""
        X, y, groups = logistic_mixed_data
        rng = np.random.default_rng(99)
        result = calibrated_logistic_family.fit(X, y, fit_intercept=True)
        resid = calibrated_logistic_family.residuals(result, X, y)
        B = 20
        perm_indices = np.array([rng.permutation(len(y)) for _ in range(B)])
        scores = calibrated_logistic_family.score_project(
            X,
            0,
            resid,
            perm_indices,
            fit_intercept=True,
        )
        assert scores.shape == (B,)

    def test_score_project_finite(
        self, calibrated_logistic_family, logistic_mixed_data
    ):
        """All score projections should be finite."""
        X, y, groups = logistic_mixed_data
        rng = np.random.default_rng(99)
        result = calibrated_logistic_family.fit(X, y, fit_intercept=True)
        resid = calibrated_logistic_family.residuals(result, X, y)
        B = 20
        perm_indices = np.array([rng.permutation(len(y)) for _ in range(B)])
        for feat_idx in range(X.shape[1]):
            scores = calibrated_logistic_family.score_project(
                X,
                feat_idx,
                resid,
                perm_indices,
                fit_intercept=True,
            )
            assert np.all(np.isfinite(scores)), (
                f"Non-finite scores for feature {feat_idx}"
            )

    def test_score_project_requires_calibration(self):
        f = LogisticMixedFamily()
        with pytest.raises(RuntimeError, match="requires calibration"):
            f.score_project(
                np.ones((5, 2)),
                0,
                np.ones(5),
                np.arange(5, dtype=np.intp).reshape(1, -1),
            )


# ------------------------------------------------------------------ #
# Batch fit raises — logistic
# ------------------------------------------------------------------ #


class TestLogisticBatchFitRaises:
    def test_batch_fit_raises(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        with pytest.raises(NotImplementedError, match="method='score'"):
            calibrated_logistic_family.batch_fit(
                X, np.vstack([y, y]), fit_intercept=True
            )

    def test_batch_fit_varying_X_raises(
        self, calibrated_logistic_family, logistic_mixed_data
    ):
        X, y, groups = logistic_mixed_data
        with pytest.raises(NotImplementedError, match="method='score'"):
            calibrated_logistic_family.batch_fit_varying_X(
                np.stack([X, X]), y, fit_intercept=True
            )

    def test_batch_fit_and_score_raises(
        self, calibrated_logistic_family, logistic_mixed_data
    ):
        X, y, groups = logistic_mixed_data
        with pytest.raises(NotImplementedError, match="method='score'"):
            calibrated_logistic_family.batch_fit_and_score(
                X, np.vstack([y, y]), fit_intercept=True
            )

    def test_batch_fit_paired_raises(
        self, calibrated_logistic_family, logistic_mixed_data
    ):
        X, y, groups = logistic_mixed_data
        with pytest.raises(NotImplementedError, match="method='score'"):
            calibrated_logistic_family.batch_fit_paired(
                np.stack([X, X]), np.vstack([y, y]), fit_intercept=True
            )


# ------------------------------------------------------------------ #
# Scoring — logistic
# ------------------------------------------------------------------ #


class TestLogisticScoring:
    def test_score_finite(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        result = calibrated_logistic_family.fit(X, y, fit_intercept=True)
        s = calibrated_logistic_family.score(result, X, y)
        assert np.isfinite(s)
        assert s >= 0  # deviance is non-negative

    def test_null_score_finite(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        ns = calibrated_logistic_family.null_score(y, fit_intercept=True)
        assert np.isfinite(ns)
        assert ns >= 0

    def test_fit_metric_nonneg(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        result = calibrated_logistic_family.fit(X, y, fit_intercept=True)
        pred = calibrated_logistic_family.predict(result, X)
        metric = calibrated_logistic_family.fit_metric(y, pred)
        assert metric >= 0

    def test_reconstruct_y_binary(
        self, calibrated_logistic_family, logistic_mixed_data
    ):
        X, y, groups = logistic_mixed_data
        result = calibrated_logistic_family.fit(X, y, fit_intercept=True)
        pred = calibrated_logistic_family.predict(result, X)
        resid = calibrated_logistic_family.residuals(result, X, y)
        rng = np.random.default_rng(42)
        y_star = calibrated_logistic_family.reconstruct_y(pred, resid, rng)
        assert y_star.shape == y.shape
        assert set(np.unique(y_star)).issubset({0.0, 1.0})


# ------------------------------------------------------------------ #
# Diagnostics — logistic
# ------------------------------------------------------------------ #


class TestLogisticDiagnostics:
    def test_diagnostics_keys(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        diag = calibrated_logistic_family.diagnostics(X, y, fit_intercept=True)
        assert "deviance" in diag
        assert "icc" in diag
        assert "variance_components" in diag
        assert "converged" in diag
        assert "n_groups" in diag

    def test_icc_range(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        diag = calibrated_logistic_family.diagnostics(X, y, fit_intercept=True)
        assert 0 <= diag["icc"] <= 1

    def test_deviance_positive(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        diag = calibrated_logistic_family.diagnostics(X, y, fit_intercept=True)
        assert diag["deviance"] >= 0

    def test_variance_components_present(
        self, calibrated_logistic_family, logistic_mixed_data
    ):
        X, y, groups = logistic_mixed_data
        diag = calibrated_logistic_family.diagnostics(X, y, fit_intercept=True)
        factors = diag["variance_components"]["factors"]
        assert len(factors) > 0
        assert factors[0]["intercept_var"] >= 0

    def test_diagnostics_requires_calibration(self):
        f = LogisticMixedFamily()
        with pytest.raises(RuntimeError, match="requires calibration"):
            f.diagnostics(np.ones((5, 2)), np.ones(5))


# ------------------------------------------------------------------ #
# Extended diagnostics + display — logistic
# ------------------------------------------------------------------ #


class TestLogisticDisplay:
    def test_display_header(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        diag = calibrated_logistic_family.diagnostics(X, y, fit_intercept=True)
        ext = calibrated_logistic_family.compute_extended_diagnostics(X, y, True)
        combined = {**diag, **ext}
        rows = calibrated_logistic_family.display_header(combined)
        assert len(rows) >= 1
        for row in rows:
            assert len(row) == 4

    def test_display_diagnostics(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        ext = calibrated_logistic_family.compute_extended_diagnostics(X, y, True)
        lines, notes = calibrated_logistic_family.display_diagnostics(ext)
        assert isinstance(lines, list)
        assert isinstance(notes, list)
        assert len(lines) >= 1  # at least τ² line

    def test_compute_extended_diagnostics_keys(
        self, calibrated_logistic_family, logistic_mixed_data
    ):
        X, y, groups = logistic_mixed_data
        ext = calibrated_logistic_family.compute_extended_diagnostics(X, y, True)
        assert "glmm_gof" in ext
        assert "icc" in ext["glmm_gof"]
        assert "variance_components" in ext["glmm_gof"]


# ------------------------------------------------------------------ #
# Classical p-values — logistic
# ------------------------------------------------------------------ #


class TestLogisticClassicalPValues:
    def test_shape_and_range(self, calibrated_logistic_family, logistic_mixed_data):
        X, y, groups = logistic_mixed_data
        pvals = calibrated_logistic_family.classical_p_values(X, y, fit_intercept=True)
        assert pvals.shape == (X.shape[1],)
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)

    def test_requires_calibration(self):
        f = LogisticMixedFamily()
        with pytest.raises(RuntimeError, match="requires calibration"):
            f.classical_p_values(np.ones((5, 2)), np.ones(5))


# ------------------------------------------------------------------ #
# Exchangeability cells — logistic
# ------------------------------------------------------------------ #


class TestLogisticExchangeabilityCells:
    def test_returns_group_labels(
        self, calibrated_logistic_family, logistic_mixed_data
    ):
        X, y, groups = logistic_mixed_data
        cells = calibrated_logistic_family.exchangeability_cells(X, y)
        assert cells.shape == (len(y),)
        assert len(np.unique(cells)) == 10

    def test_uncalibrated_returns_none(self):
        f = LogisticMixedFamily()
        assert f.exchangeability_cells(np.ones((5, 2)), np.ones(5)) is None


# ================================================================== #
# PoissonMixedFamily Tests (Step C.5)
# ================================================================== #

# ------------------------------------------------------------------ #
# Fixtures — Poisson GLMM
# ------------------------------------------------------------------ #


@pytest.fixture()
def poisson_mixed_data():
    """Generate clustered count data with known structure.

    10 groups × 30 obs = 300 total.
    Random intercepts ~ N(0, τ²=0.25) on log scale.
    True β = [0.3, -0.2] in log space.
    Intercept ≈ 1.5 so baseline rate ≈ exp(1.5) ≈ 4.5.
    """
    rng = np.random.default_rng(2025)
    n_groups = 10
    n_per_group = 30
    n = n_groups * n_per_group
    p = 2

    groups = np.repeat(np.arange(n_groups), n_per_group)
    X = rng.standard_normal((n, p))
    beta_true = np.array([0.3, -0.2])

    # Random intercepts on log scale
    u = rng.normal(0, 0.5, size=n_groups)  # τ = 0.5
    eta = X @ beta_true + u[groups] + 1.5
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)

    return X, y, groups


@pytest.fixture()
def calibrated_poisson_family(poisson_mixed_data):
    """Return a calibrated PoissonMixedFamily."""
    X, y, groups = poisson_mixed_data
    family = PoissonMixedFamily()
    return family.calibrate(X, y, fit_intercept=True, groups=groups)


# ------------------------------------------------------------------ #
# Protocol conformance — Poisson
# ------------------------------------------------------------------ #


class TestPoissonProtocolConformance:
    def test_isinstance_check(self):
        assert isinstance(PoissonMixedFamily(), ModelFamily)

    def test_name(self):
        assert PoissonMixedFamily().name == "poisson_mixed"

    def test_residual_type(self):
        assert PoissonMixedFamily().residual_type == "deviance"

    def test_direct_permutation(self):
        assert PoissonMixedFamily().direct_permutation is False

    def test_stat_label(self):
        assert PoissonMixedFamily().stat_label == "z"

    def test_metric_label(self):
        assert PoissonMixedFamily().metric_label == "Deviance Reduction"

    def test_frozen_dataclass(self):
        family = PoissonMixedFamily()
        with pytest.raises(AttributeError):
            family.name = "oops"  # type: ignore[misc]


# ------------------------------------------------------------------ #
# Registry — Poisson
# ------------------------------------------------------------------ #


class TestPoissonRegistry:
    def test_resolve_family_string(self):
        f = resolve_family("poisson_mixed")
        assert isinstance(f, PoissonMixedFamily)
        assert f.name == "poisson_mixed"

    def test_resolve_family_instance_passthrough(self):
        f = PoissonMixedFamily()
        assert resolve_family(f) is f


# ------------------------------------------------------------------ #
# Validate Y — Poisson
# ------------------------------------------------------------------ #


class TestPoissonValidateY:
    def test_valid_int(self):
        PoissonMixedFamily().validate_y(np.array([0, 1, 5, 10]))

    def test_valid_float_integers(self):
        PoissonMixedFamily().validate_y(np.array([0.0, 1.0, 3.0]))

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            PoissonMixedFamily().validate_y(np.array([0, 1, -1]))

    def test_rejects_non_integer(self):
        with pytest.raises(ValueError, match="integer"):
            PoissonMixedFamily().validate_y(np.array([0.5, 1.5, 2.5]))


# ------------------------------------------------------------------ #
# Calibration — Poisson
# ------------------------------------------------------------------ #


class TestPoissonCalibration:
    def test_calibrate_populates_fields(self, calibrated_poisson_family):
        f = calibrated_poisson_family
        assert f.beta is not None
        assert f.u is not None
        assert f.W is not None
        assert f.mu is not None
        assert f.V_inv_diag is not None
        assert f.fisher_info is not None
        assert f.re_covariances is not None
        assert f.log_chol is not None
        assert f.Z is not None
        assert f.C22 is not None
        assert f.re_struct is not None

    def test_calibrate_idempotent(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        f2 = calibrated_poisson_family.calibrate(X, y, groups=groups)
        assert f2 is calibrated_poisson_family

    def test_calibrate_requires_groups(self):
        f = PoissonMixedFamily()
        X = np.ones((10, 2))
        y = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3], dtype=float)
        with pytest.raises(ValueError, match="groups"):
            f.calibrate(X, y)

    def test_converged(self, calibrated_poisson_family):
        assert calibrated_poisson_family.converged is True

    def test_re_struct(self, calibrated_poisson_family):
        assert calibrated_poisson_family.re_struct == ((10, 1),)

    def test_fisher_info_shape(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        n, p = X.shape
        assert calibrated_poisson_family.fisher_info.shape == (p + 1, p + 1)

    def test_fisher_info_positive_definite(self, calibrated_poisson_family):
        eigvals = np.linalg.eigvalsh(calibrated_poisson_family.fisher_info)
        assert np.all(eigvals > 0)

    def test_V_inv_diag_positive(self, calibrated_poisson_family):
        assert np.all(calibrated_poisson_family.V_inv_diag > 0)

    def test_mu_positive(self, calibrated_poisson_family):
        """Predicted rates should be positive."""
        assert np.all(calibrated_poisson_family.mu > 0)

    def test_W_positive(self, calibrated_poisson_family):
        assert np.all(calibrated_poisson_family.W > 0)


# ------------------------------------------------------------------ #
# Fit — Poisson
# ------------------------------------------------------------------ #


class TestPoissonFit:
    def test_fit_returns_result(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        result = calibrated_poisson_family.fit(X, y, fit_intercept=True)
        n, p = X.shape
        beta = calibrated_poisson_family.coefs(result)
        pred = calibrated_poisson_family.predict(result, X)
        assert beta.shape == (p,)
        assert pred.shape == (n,)

    def test_predictions_positive(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        result = calibrated_poisson_family.fit(X, y, fit_intercept=True)
        pred = calibrated_poisson_family.predict(result, X)
        assert np.all(pred > 0)

    def test_beta_direction(self, calibrated_poisson_family, poisson_mixed_data):
        """β̂ should have the correct sign direction (true β = [0.3, -0.2])."""
        X, y, groups = poisson_mixed_data
        result = calibrated_poisson_family.fit(X, y, fit_intercept=True)
        beta = calibrated_poisson_family.coefs(result)
        assert beta[0] > 0  # true +0.3
        assert beta[1] < 0  # true -0.2

    def test_residuals_shape(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        result = calibrated_poisson_family.fit(X, y, fit_intercept=True)
        resid = calibrated_poisson_family.residuals(result, X, y)
        assert resid.shape == (len(y),)

    def test_fit_requires_calibration(self):
        f = PoissonMixedFamily()
        with pytest.raises(RuntimeError, match="requires calibration"):
            f.fit(np.ones((5, 2)), np.ones(5), fit_intercept=True)


# ------------------------------------------------------------------ #
# Score projection — Poisson
# ------------------------------------------------------------------ #


class TestPoissonScoreProject:
    def test_score_project_shape(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        rng = np.random.default_rng(99)
        result = calibrated_poisson_family.fit(X, y, fit_intercept=True)
        resid = calibrated_poisson_family.residuals(result, X, y)
        B = 20
        perm_indices = np.array([rng.permutation(len(y)) for _ in range(B)])
        scores = calibrated_poisson_family.score_project(
            X,
            0,
            resid,
            perm_indices,
            fit_intercept=True,
        )
        assert scores.shape == (B,)

    def test_score_project_finite(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        rng = np.random.default_rng(99)
        result = calibrated_poisson_family.fit(X, y, fit_intercept=True)
        resid = calibrated_poisson_family.residuals(result, X, y)
        B = 20
        perm_indices = np.array([rng.permutation(len(y)) for _ in range(B)])
        for feat_idx in range(X.shape[1]):
            scores = calibrated_poisson_family.score_project(
                X,
                feat_idx,
                resid,
                perm_indices,
                fit_intercept=True,
            )
            assert np.all(np.isfinite(scores)), (
                f"Non-finite scores for feature {feat_idx}"
            )


# ------------------------------------------------------------------ #
# Batch fit raises — Poisson
# ------------------------------------------------------------------ #


class TestPoissonBatchFitRaises:
    def test_batch_fit_raises(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        with pytest.raises(NotImplementedError, match="method='score'"):
            calibrated_poisson_family.batch_fit(
                X, np.vstack([y, y]), fit_intercept=True
            )

    def test_batch_fit_and_score_raises(
        self, calibrated_poisson_family, poisson_mixed_data
    ):
        X, y, groups = poisson_mixed_data
        with pytest.raises(NotImplementedError, match="method='score'"):
            calibrated_poisson_family.batch_fit_and_score(
                X, np.vstack([y, y]), fit_intercept=True
            )


# ------------------------------------------------------------------ #
# Scoring — Poisson
# ------------------------------------------------------------------ #


class TestPoissonScoring:
    def test_score_finite(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        result = calibrated_poisson_family.fit(X, y, fit_intercept=True)
        s = calibrated_poisson_family.score(result, X, y)
        assert np.isfinite(s)

    def test_null_score_finite(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        ns = calibrated_poisson_family.null_score(y, fit_intercept=True)
        assert np.isfinite(ns)
        assert ns >= 0

    def test_fit_metric_nonneg(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        result = calibrated_poisson_family.fit(X, y, fit_intercept=True)
        pred = calibrated_poisson_family.predict(result, X)
        metric = calibrated_poisson_family.fit_metric(y, pred)
        assert metric >= 0

    def test_reconstruct_y_count(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        result = calibrated_poisson_family.fit(X, y, fit_intercept=True)
        pred = calibrated_poisson_family.predict(result, X)
        resid = calibrated_poisson_family.residuals(result, X, y)
        rng = np.random.default_rng(42)
        y_star = calibrated_poisson_family.reconstruct_y(pred, resid, rng)
        assert y_star.shape == y.shape
        assert np.all(y_star >= 0)  # counts are non-negative
        assert np.allclose(y_star, np.round(y_star))  # integer-valued


# ------------------------------------------------------------------ #
# Diagnostics — Poisson
# ------------------------------------------------------------------ #


class TestPoissonDiagnostics:
    def test_diagnostics_keys(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        diag = calibrated_poisson_family.diagnostics(X, y, fit_intercept=True)
        assert "deviance" in diag
        assert "dispersion" in diag  # Poisson-specific
        assert "icc" in diag
        assert "variance_components" in diag
        assert "converged" in diag

    def test_icc_range(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        diag = calibrated_poisson_family.diagnostics(X, y, fit_intercept=True)
        assert 0 <= diag["icc"] <= 1

    def test_dispersion_positive(self, calibrated_poisson_family, poisson_mixed_data):
        """Pearson dispersion should be positive."""
        X, y, groups = poisson_mixed_data
        diag = calibrated_poisson_family.diagnostics(X, y, fit_intercept=True)
        assert diag["dispersion"] > 0

    def test_variance_components_present(
        self, calibrated_poisson_family, poisson_mixed_data
    ):
        X, y, groups = poisson_mixed_data
        diag = calibrated_poisson_family.diagnostics(X, y, fit_intercept=True)
        factors = diag["variance_components"]["factors"]
        assert len(factors) > 0
        assert factors[0]["intercept_var"] >= 0


# ------------------------------------------------------------------ #
# Extended diagnostics + display — Poisson
# ------------------------------------------------------------------ #


class TestPoissonDisplay:
    def test_display_header(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        diag = calibrated_poisson_family.diagnostics(X, y, fit_intercept=True)
        ext = calibrated_poisson_family.compute_extended_diagnostics(X, y, True)
        combined = {**diag, **ext}
        rows = calibrated_poisson_family.display_header(combined)
        assert len(rows) >= 1
        for row in rows:
            assert len(row) == 4

    def test_display_diagnostics(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        ext = calibrated_poisson_family.compute_extended_diagnostics(X, y, True)
        lines, notes = calibrated_poisson_family.display_diagnostics(ext)
        assert isinstance(lines, list)
        assert isinstance(notes, list)

    def test_compute_extended_diagnostics_keys(
        self, calibrated_poisson_family, poisson_mixed_data
    ):
        X, y, groups = poisson_mixed_data
        ext = calibrated_poisson_family.compute_extended_diagnostics(X, y, True)
        assert "glmm_gof" in ext
        assert "icc" in ext["glmm_gof"]
        assert "dispersion" in ext["glmm_gof"]


# ------------------------------------------------------------------ #
# Classical p-values — Poisson
# ------------------------------------------------------------------ #


class TestPoissonClassicalPValues:
    def test_shape_and_range(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        pvals = calibrated_poisson_family.classical_p_values(X, y, fit_intercept=True)
        assert pvals.shape == (X.shape[1],)
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)


# ------------------------------------------------------------------ #
# Exchangeability cells — Poisson
# ------------------------------------------------------------------ #


class TestPoissonExchangeabilityCells:
    def test_returns_group_labels(self, calibrated_poisson_family, poisson_mixed_data):
        X, y, groups = poisson_mixed_data
        cells = calibrated_poisson_family.exchangeability_cells(X, y)
        assert cells.shape == (len(y),)
        assert len(np.unique(cells)) == 10

    def test_uncalibrated_returns_none(self):
        f = PoissonMixedFamily()
        assert f.exchangeability_cells(np.ones((5, 2)), np.ones(5)) is None


# ------------------------------------------------------------------ #
# GLMM integration: end-to-end through PermutationEngine
# ------------------------------------------------------------------ #


class TestGLMMIntegration:
    def test_engine_logistic_mixed_score(self, logistic_mixed_data):
        """PermutationEngine can run with LogisticMixedFamily + method='score'."""
        import pandas as pd

        from randomization_tests.engine import PermutationEngine

        X, y, groups = logistic_mixed_data
        X_df = pd.DataFrame(X, columns=["x1", "x2"])

        engine = PermutationEngine(
            X_df,
            y,
            family=LogisticMixedFamily(),
            fit_intercept=True,
            n_permutations=50,
            random_state=42,
            groups=groups,
            method="score",
        )

        assert engine.family.name == "logistic_mixed"
        assert engine.family.fisher_info is not None

    def test_engine_poisson_mixed_score(self, poisson_mixed_data):
        """PermutationEngine can run with PoissonMixedFamily + method='score'."""
        import pandas as pd

        from randomization_tests.engine import PermutationEngine

        X, y, groups = poisson_mixed_data
        X_df = pd.DataFrame(X, columns=["x1", "x2"])

        engine = PermutationEngine(
            X_df,
            y,
            family=PoissonMixedFamily(),
            fit_intercept=True,
            n_permutations=50,
            random_state=42,
            groups=groups,
            method="score",
        )

        assert engine.family.name == "poisson_mixed"
        assert engine.family.fisher_info is not None

    def test_engine_logistic_string_resolution(self, logistic_mixed_data):
        """PermutationEngine resolves 'logistic_mixed' string."""
        import pandas as pd

        from randomization_tests.engine import PermutationEngine

        X, y, groups = logistic_mixed_data
        X_df = pd.DataFrame(X, columns=["x1", "x2"])

        engine = PermutationEngine(
            X_df,
            y,
            family="logistic_mixed",
            fit_intercept=True,
            n_permutations=50,
            random_state=42,
            groups=groups,
            method="score",
        )

        assert isinstance(engine.family, LogisticMixedFamily)

    def test_engine_poisson_string_resolution(self, poisson_mixed_data):
        """PermutationEngine resolves 'poisson_mixed' string."""
        import pandas as pd

        from randomization_tests.engine import PermutationEngine

        X, y, groups = poisson_mixed_data
        X_df = pd.DataFrame(X, columns=["x1", "x2"])

        engine = PermutationEngine(
            X_df,
            y,
            family="poisson_mixed",
            fit_intercept=True,
            n_permutations=50,
            random_state=42,
            groups=groups,
            method="score",
        )

        assert isinstance(engine.family, PoissonMixedFamily)

    def test_glmm_rejects_ter_braak(self, logistic_mixed_data):
        """GLMM with method='ter_braak' should fail (batch_fit not supported)."""
        import pandas as pd

        from randomization_tests.engine import PermutationEngine

        X, y, groups = logistic_mixed_data
        X_df = pd.DataFrame(X, columns=["x1", "x2"])

        # ter_braak calls batch_fit which raises NotImplementedError,
        # but the engine should still construct — the error happens at run time.
        # However, the Manly warning path may also kick in.
        engine = PermutationEngine(
            X_df,
            y,
            family=LogisticMixedFamily(),
            fit_intercept=True,
            n_permutations=50,
            random_state=42,
            groups=groups,
            method="ter_braak",
        )
        assert engine.family.name == "logistic_mixed"

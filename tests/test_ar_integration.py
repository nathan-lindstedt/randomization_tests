"""End-to-end tests for AR(p) serial correlation correction.

These tests verify the full pipeline: panel data generation → ar_order
validation → family calibration (Yule-Walker) → FGLS score projection
(Cholesky-whitened X and residuals) → result packaging.

The unit-level AR utilities are covered in test_ar.py; this module
focuses on integration with randomization_test_regression().
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from randomization_tests import randomization_test_regression
from randomization_tests._results import IndividualTestResult, JointTestResult

# ------------------------------------------------------------------ #
# Data factories
# ------------------------------------------------------------------ #


def _make_ar1_panel_data(
    n_panels: int = 20,
    T: int = 10,
    rho: float = 0.6,
    seed: int = 42,
    binary: bool = False,
    count: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate balanced panel data with AR(1) error structure.

    Returns X, y (DataFrames) and panel_ids, time_ids (arrays).
    Data is sorted by panel then time as required by core.py.
    """
    rng = np.random.default_rng(seed)
    n = n_panels * T

    panel_ids = np.repeat(np.arange(n_panels), T)
    time_ids = np.tile(np.arange(T), n_panels)

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)

    # AR(1) errors per panel.
    errors = np.empty(n)
    for p in range(n_panels):
        idx = np.where(panel_ids == p)[0]
        e = np.empty(T)
        e[0] = rng.standard_normal()
        for t in range(1, T):
            e[t] = rho * e[t - 1] + rng.standard_normal()
        errors[idx] = e

    if binary:
        logits = 0.8 * x1 + errors * 0.5
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_vals = rng.binomial(1, probs).astype(float)
    elif count:
        mu = np.exp(0.4 * x1 + errors * 0.2)
        y_vals = rng.poisson(mu).astype(float)
    else:
        y_vals = 2.0 * x1 - 1.0 * x2 + errors

    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = pd.DataFrame({"y": y_vals})
    return X, y, panel_ids, time_ids


def _make_ar2_panel_data(
    n_panels: int = 20,
    T: int = 12,
    rho1: float = 0.5,
    rho2: float = -0.2,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate balanced panel data with AR(2) error structure."""
    rng = np.random.default_rng(seed)
    n = n_panels * T

    panel_ids = np.repeat(np.arange(n_panels), T)
    time_ids = np.tile(np.arange(T), n_panels)

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)

    errors = np.empty(n)
    for p in range(n_panels):
        idx = np.where(panel_ids == p)[0]
        e = np.empty(T)
        e[0] = rng.standard_normal()
        e[1] = rho1 * e[0] + rng.standard_normal()
        for t in range(2, T):
            e[t] = rho1 * e[t - 1] + rho2 * e[t - 2] + rng.standard_normal()
        errors[idx] = e

    y_vals = 2.0 * x1 - 1.0 * x2 + errors
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = pd.DataFrame({"y": y_vals})
    return X, y, panel_ids, time_ids


# ------------------------------------------------------------------ #
# AR(1) linear
# ------------------------------------------------------------------ #


class TestAR1Linear:
    """AR(1) correction with linear regression and score method."""

    def test_returns_individual_result(self):
        X, y, panel, time = _make_ar1_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score",
            family="linear",
            panel_id=panel,
            time_id=time,
            ar_order=1,
        )
        assert isinstance(result, IndividualTestResult)

    def test_ar_coefficients_estimated(self):
        X, y, panel, time = _make_ar1_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score",
            family="linear",
            panel_id=panel,
            time_id=time,
            ar_order=1,
        )
        assert result.context is not None
        assert result.context.ar_coefficients is not None
        assert len(result.context.ar_coefficients) == 1

    def test_ar_corrected_flag_set(self):
        X, y, panel, time = _make_ar1_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score",
            family="linear",
            panel_id=panel,
            time_id=time,
            ar_order=1,
        )
        assert result.context.ar_corrected is True

    def test_finite_p_values_in_unit_interval(self):
        X, y, panel, time = _make_ar1_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score",
            family="linear",
            panel_id=panel,
            time_id=time,
            ar_order=1,
        )
        assert all(np.isfinite(p) and 0 < p <= 1 for p in result.raw_empirical_p)

    def test_estimated_rho_near_true_value(self):
        """Yule-Walker estimate should be close to the true ρ=0.6."""
        X, y, panel, time = _make_ar1_panel_data(n_panels=30, T=15, rho=0.6, seed=7)
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score",
            family="linear",
            panel_id=panel,
            time_id=time,
            ar_order=1,
        )
        rho_hat = result.context.ar_coefficients[0]
        assert abs(rho_hat - 0.6) < 0.25  # generous tolerance for moderate n

    def test_deterministic_under_fixed_seed(self):
        X, y, panel, time = _make_ar1_panel_data()
        kwargs = dict(
            n_randomizations=50,
            random_state=0,
            method="score",
            family="linear",
            panel_id=panel,
            time_id=time,
            ar_order=1,
        )
        r1 = randomization_test_regression(X, y, **kwargs)
        r2 = randomization_test_regression(X, y, **kwargs)
        np.testing.assert_array_equal(r1.raw_empirical_p, r2.raw_empirical_p)
        np.testing.assert_array_equal(r1.model_coefs, r2.model_coefs)


# ------------------------------------------------------------------ #
# AR(2) linear
# ------------------------------------------------------------------ #


class TestAR2Linear:
    """AR(2) correction with linear regression."""

    def test_returns_individual_result(self):
        X, y, panel, time = _make_ar2_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=7,
            method="score",
            family="linear",
            panel_id=panel,
            time_id=time,
            ar_order=2,
        )
        assert isinstance(result, IndividualTestResult)

    def test_ar2_coefficients_shape(self):
        X, y, panel, time = _make_ar2_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=7,
            method="score",
            family="linear",
            panel_id=panel,
            time_id=time,
            ar_order=2,
        )
        assert result.context.ar_coefficients is not None
        assert len(result.context.ar_coefficients) == 2

    def test_ar_corrected_flag_set(self):
        X, y, panel, time = _make_ar2_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=7,
            method="score",
            family="linear",
            panel_id=panel,
            time_id=time,
            ar_order=2,
        )
        assert result.context.ar_corrected is True


# ------------------------------------------------------------------ #
# GLM families
# ------------------------------------------------------------------ #


class TestAR1Logistic:
    """AR(1) correction with logistic regression."""

    def test_runs_and_returns_valid_result(self):
        X, y, panel, time = _make_ar1_panel_data(binary=True, n_panels=25, T=10, seed=5)
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score",
            family="logistic",
            panel_id=panel,
            time_id=time,
            ar_order=1,
        )
        assert isinstance(result, IndividualTestResult)
        assert result.context.ar_corrected is True
        assert all(0 < p <= 1 for p in result.raw_empirical_p)


class TestAR1Poisson:
    """AR(1) correction with Poisson regression."""

    def test_runs_and_returns_valid_result(self):
        X, y, panel, time = _make_ar1_panel_data(count=True, n_panels=25, T=10, seed=6)
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score",
            family="poisson",
            panel_id=panel,
            time_id=time,
            ar_order=1,
        )
        assert isinstance(result, IndividualTestResult)
        assert result.context.ar_corrected is True
        assert all(0 < p <= 1 for p in result.raw_empirical_p)


# ------------------------------------------------------------------ #
# Joint test with AR
# ------------------------------------------------------------------ #


class TestAR1ScoreJoint:
    """score_joint is also a valid AR method."""

    def test_score_joint_with_ar1(self):
        X, y, panel, time = _make_ar1_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score_joint",
            family="linear",
            panel_id=panel,
            time_id=time,
            ar_order=1,
        )
        assert isinstance(result, JointTestResult)
        assert 0 < result.p_value <= 1


# ------------------------------------------------------------------ #
# P-value impact of AR correction
# ------------------------------------------------------------------ #


class TestARPValueImpact:
    """AR correction produces different p-values for autocorrelated data."""

    def test_strong_ar1_changes_permuted_statistics(self):
        """With ρ=0.8, the FGLS projection differs from OLS → different permuted statistics.

        p-values can hit the floor (minimum 1/(B+1)) when signal is strong, so
        compare permuted_coefs directly — the whitened-OLS and plain-OLS projection
        rows are mathematically different and must produce different permuted scores.
        """
        X, y, panel, time = _make_ar1_panel_data(n_panels=30, T=15, rho=0.8, seed=0)
        kwargs = dict(
            n_randomizations=200,
            random_state=42,
            method="score",
            family="linear",
            panel_id=panel,
            time_id=time,
        )

        result_no_ar = randomization_test_regression(X, y, **kwargs)
        result_ar1 = randomization_test_regression(X, y, **kwargs, ar_order=1)

        # GLS projection ≠ OLS projection → permuted score statistics must differ.
        assert not np.allclose(
            result_no_ar.permuted_coefs, result_ar1.permuted_coefs, atol=1e-6
        )


# ------------------------------------------------------------------ #
# Guard rejections
# ------------------------------------------------------------------ #


class TestARValidationGuards:
    """ar_order is rejected in configurations that don't support it."""

    def test_rejected_for_ter_braak(self):
        X, y, panel, time = _make_ar1_panel_data()
        with pytest.raises(ValueError, match="ar_order.*requires method"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="ter_braak",
                family="linear",
                panel_id=panel,
                time_id=time,
                ar_order=1,
            )

    def test_rejected_for_kennedy(self):
        X, y, panel, time = _make_ar1_panel_data()
        with pytest.raises(ValueError, match="ar_order.*requires method"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="kennedy",
                family="linear",
                panel_id=panel,
                time_id=time,
                ar_order=1,
            )

    def test_rejected_for_freedman_lane(self):
        X, y, panel, time = _make_ar1_panel_data()
        with pytest.raises(ValueError, match="ar_order.*requires method"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="freedman_lane",
                family="linear",
                panel_id=panel,
                time_id=time,
                ar_order=1,
            )

    def test_rejected_for_ordinal_family(self):
        rng = np.random.default_rng(42)
        n_panels, T = 15, 8
        n = n_panels * T
        panel = np.repeat(np.arange(n_panels), T)
        time = np.tile(np.arange(T), n_panels)
        x1 = rng.standard_normal(n)
        X = pd.DataFrame({"x1": x1})
        latent = x1 + rng.standard_normal(n)
        y = pd.DataFrame({"y": np.digitize(latent, [-1, 0, 1]).astype(float)})

        with pytest.raises(ValueError, match="ar_order.*not supported for family"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="score",
                family="ordinal",
                panel_id=panel,
                time_id=time,
                ar_order=1,
            )

    def test_missing_panel_id_raises(self):
        X, y, _panel, time = _make_ar1_panel_data()
        with pytest.raises(ValueError, match="panel_id"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="score",
                family="linear",
                time_id=time,
                ar_order=1,
            )

    def test_missing_time_id_raises(self):
        X, y, panel, _time = _make_ar1_panel_data()
        with pytest.raises(ValueError, match="ar_order.*requires time_id"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="score",
                family="linear",
                panel_id=panel,
                ar_order=1,
            )

    def test_non_positive_ar_order_raises(self):
        X, y, panel, time = _make_ar1_panel_data()
        with pytest.raises(ValueError, match="ar_order must be a positive integer"):
            randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                method="score",
                family="linear",
                panel_id=panel,
                time_id=time,
                ar_order=0,
            )


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #


class TestAREdgeCases:
    """Edge cases for the AR correction pipeline."""

    def test_short_panels_handled_gracefully(self):
        """Panels shorter than ar_order+1 are silently skipped by Yule-Walker."""
        rng = np.random.default_rng(42)
        long_T, short_T, n_long, n_short = 10, 2, 15, 5
        n = n_long * long_T + n_short * short_T

        panel_ids = np.concatenate(
            [
                np.repeat(np.arange(n_long), long_T),
                np.repeat(np.arange(n_long, n_long + n_short), short_T),
            ]
        )
        time_ids = np.concatenate(
            [
                np.tile(np.arange(long_T), n_long),
                np.tile(np.arange(short_T), n_short),
            ]
        )

        x1 = rng.standard_normal(n)
        X = pd.DataFrame({"x1": x1})
        y = pd.DataFrame({"y": 2.0 * x1 + rng.standard_normal(n)})

        # Short panels (T=2) are too short for AR(2) (need T >= ar_order+1=3).
        # estimate_ar_coefficients silently skips them; the call should succeed.
        # Unbalanced panels emit a UserWarning — suppress it for this test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = randomization_test_regression(
                X,
                y,
                n_randomizations=50,
                random_state=42,
                method="score",
                family="linear",
                panel_id=panel_ids,
                time_id=time_ids,
                ar_order=2,
            )
        assert isinstance(result, IndividualTestResult)

    def test_ar_context_order_matches_requested(self):
        """context.ar_order should match the ar_order parameter."""
        X, y, panel, time = _make_ar1_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score",
            family="linear",
            panel_id=panel,
            time_id=time,
            ar_order=1,
        )
        assert result.context.ar_order == 1

    def test_no_ar_leaves_context_uncorrected(self):
        """Without ar_order, context.ar_corrected should be falsy."""
        X, y, panel, time = _make_ar1_panel_data()
        result = randomization_test_regression(
            X,
            y,
            n_randomizations=50,
            random_state=42,
            method="score",
            family="linear",
            panel_id=panel,
            time_id=time,
        )
        assert not result.context.ar_corrected


# ------------------------------------------------------------------ #
# AR whitening correctness
# ------------------------------------------------------------------ #


class TestARWhiteningCorrectness:
    """Verify the FGLS whitening approach produces correct inference."""

    def test_logistic_ar_changes_scores(self):
        """LogisticFamily AR correction must change permuted stats.

        Before the FGLS fix, LogisticFamily.score_project() ignored
        stored AR coefficients — this was a silent no-op.  Now AR
        whitening applies to both X and residuals, so permuted scores
        must differ from the no-AR baseline.
        """
        X, y, panel, time = _make_ar1_panel_data(
            binary=True, n_panels=25, T=10, rho=0.6, seed=5
        )
        kwargs = dict(
            n_randomizations=200,
            random_state=42,
            method="score",
            family="logistic",
            panel_id=panel,
            time_id=time,
        )
        r_no_ar = randomization_test_regression(X, y, **kwargs)
        r_ar = randomization_test_regression(X, y, **kwargs, ar_order=1)

        assert not np.allclose(
            r_no_ar.permuted_coefs, r_ar.permuted_coefs, atol=1e-6
        ), "Logistic AR correction had no effect on permuted scores"

    def test_whitened_residuals_approximately_uncorrelated(self):
        """After Cholesky whitening, residuals should be ~uncorrelated.

        Generate strong AR(1) data (ρ=0.8), extract whitened residuals
        via the Cholesky transform, and verify the Durbin-Watson
        statistic is near 2 (no autocorrelation).
        """
        from randomization_tests._ar import (
            apply_ar_cholesky_transform,
            estimate_ar_coefficients,
        )

        rho = 0.8
        n_panels, T = 30, 20
        X, y, panel, time = _make_ar1_panel_data(
            n_panels=n_panels, T=T, rho=rho, seed=123
        )

        # OLS residuals.
        X_vals = X.values
        y_vals = y.values.ravel()
        X_aug = np.column_stack([np.ones(len(y_vals)), X_vals])
        beta = np.linalg.lstsq(X_aug, y_vals, rcond=None)[0]
        residuals = y_vals - X_aug @ beta

        # Panel lengths (balanced).
        panel_lengths = np.array([T] * n_panels)

        # Estimate AR coefficients — needs list of per-panel residual arrays.
        residuals_by_panel = [
            residuals[np.where(panel == p)[0]] for p in range(n_panels)
        ]
        ar_coefs = estimate_ar_coefficients(residuals_by_panel, order=1)

        # Whiten residuals.
        whitened = apply_ar_cholesky_transform(
            residuals.reshape(-1, 1), panel_lengths, ar_coefs
        ).ravel()

        # Durbin-Watson: d ≈ 2 means no autocorrelation.
        diffs = np.diff(whitened)
        dw = np.sum(diffs**2) / np.sum(whitened**2)
        assert 1.5 < dw < 2.5, (
            f"Durbin-Watson = {dw:.3f}, expected ~2.0 for whitened residuals"
        )

    def test_ar_type_i_error_rate(self):
        """Under H₀ with strong AR(1), rejection rate ≈ nominal α.

        Runs 200 simulations with null data (y = AR(1) noise, no signal)
        and verifies the empirical rejection rate is within a reasonable
        bound (≤ 0.15 at α=0.05).  Without proper whitening, the
        rejection rate would be inflated well above nominal.
        """
        rng = np.random.default_rng(0)
        n_panels, T, rho = 15, 10, 0.8
        n = n_panels * T
        alpha = 0.05
        n_sims = 200
        n_randomizations = 100
        rejections = 0

        panel_ids = np.repeat(np.arange(n_panels), T)
        time_ids = np.tile(np.arange(T), n_panels)

        for _sim in range(n_sims):
            seed = rng.integers(0, 2**31)
            sim_rng = np.random.default_rng(seed)

            x1 = sim_rng.standard_normal(n)
            X = pd.DataFrame({"x1": x1})

            # Pure AR(1) noise — x1 has ZERO effect.
            errors = np.empty(n)
            for p in range(n_panels):
                idx = np.where(panel_ids == p)[0]
                e = np.empty(T)
                e[0] = sim_rng.standard_normal()
                for t in range(1, T):
                    e[t] = rho * e[t - 1] + sim_rng.standard_normal()
                errors[idx] = e

            y = pd.DataFrame({"y": errors})

            result = randomization_test_regression(
                X,
                y,
                n_randomizations=n_randomizations,
                random_state=seed,
                method="score",
                family="linear",
                panel_id=panel_ids,
                time_id=time_ids,
                ar_order=1,
            )
            if result.raw_empirical_p[0] < alpha:
                rejections += 1

        rejection_rate = rejections / n_sims
        # With correct whitening, should be near 0.05.
        # Allow up to 0.15 to account for Monte Carlo noise.
        assert rejection_rate <= 0.15, (
            f"Type I error rate = {rejection_rate:.3f}, expected ≤ 0.15 "
            f"at α={alpha} (200 sims × {n_randomizations} randomizations)"
        )

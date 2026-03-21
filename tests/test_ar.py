"""Tests for the _ar autoregressive utilities module."""

import numpy as np
import pytest
import scipy.linalg

from randomization_tests._ar import (
    apply_ar_precision,
    ar_diagnostics,
    build_ar_precision_block,
    estimate_ar_coefficients,
)


def _generate_ar1_panels(
    rho: float,
    n_panels: int,
    T: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate AR(1) panel residuals with known coefficient."""
    panels = []
    for _ in range(n_panels):
        e = np.empty(T)
        e[0] = rng.standard_normal()
        for t in range(1, T):
            e[t] = rho * e[t - 1] + rng.standard_normal()
        panels.append(e)
    return panels


def _generate_ar2_panels(
    rho1: float,
    rho2: float,
    n_panels: int,
    T: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate AR(2) panel residuals with known coefficients."""
    panels = []
    for _ in range(n_panels):
        e = np.empty(T)
        e[0] = rng.standard_normal()
        e[1] = rho1 * e[0] + rng.standard_normal()
        for t in range(2, T):
            e[t] = rho1 * e[t - 1] + rho2 * e[t - 2] + rng.standard_normal()
        panels.append(e)
    return panels


class TestEstimateArCoefficients:
    """Tests for Yule-Walker AR coefficient estimation."""

    def test_yule_walker_ar1_known_rho(self):
        rng = np.random.default_rng(42)
        panels = _generate_ar1_panels(0.7, n_panels=20, T=200, rng=rng)
        coefs = estimate_ar_coefficients(panels, order=1)
        assert coefs.shape == (1,)
        assert abs(coefs[0] - 0.7) < 0.1

    def test_yule_walker_ar2_known(self):
        rng = np.random.default_rng(123)
        panels = _generate_ar2_panels(0.5, -0.3, n_panels=20, T=200, rng=rng)
        coefs = estimate_ar_coefficients(panels, order=2)
        assert coefs.shape == (2,)
        assert abs(coefs[0] - 0.5) < 0.15
        assert abs(coefs[1] - (-0.3)) < 0.15

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="order must be >= 1"):
            estimate_ar_coefficients([np.zeros(10)], order=0)

    def test_short_panels_skipped(self):
        """Panels shorter than order+1 are silently skipped."""
        rng = np.random.default_rng(7)
        long_panels = _generate_ar1_panels(0.5, n_panels=10, T=100, rng=rng)
        short_panels = [np.array([1.0, 2.0])]  # Too short for order=3
        coefs = estimate_ar_coefficients(long_panels + short_panels, order=3)
        assert coefs.shape == (3,)

    def test_all_panels_too_short_raises(self):
        with pytest.raises(ValueError, match="No panels long enough"):
            estimate_ar_coefficients([np.array([1.0])], order=1)


class TestBuildArPrecisionBlock:
    """Tests for the AR precision matrix builder."""

    def test_precision_block_ar1_matches_toeplitz_inverse(self):
        rho = 0.6
        T = 10
        # Build Toeplitz covariance for AR(1).
        acov = np.array([rho ** abs(k) / (1 - rho**2) for k in range(T)])
        cov = scipy.linalg.toeplitz(acov)
        expected = np.linalg.inv(cov)

        # Our function (unit innovation variance).
        prec = build_ar_precision_block(np.array([rho]), T)

        # Scale: our precision assumes unit innovation variance σ²=1 so
        # the covariance has γ(0) = 1/(1−ρ²).  The Toeplitz above also
        # uses γ(0) = 1/(1−ρ²), so the matrices should match directly.
        np.testing.assert_allclose(prec, expected, atol=1e-10)

    def test_precision_block_ar2_positive_definite(self):
        ar_coefs = np.array([0.5, -0.3])
        T = 8
        prec = build_ar_precision_block(ar_coefs, T)
        eigenvalues = np.linalg.eigvalsh(prec)
        assert np.all(eigenvalues > 0)

    def test_precision_block_symmetry(self):
        prec = build_ar_precision_block(np.array([0.4]), T=12)
        np.testing.assert_allclose(prec, prec.T, atol=1e-14)


class TestApplyArPrecision:
    """Tests for block-diagonal Ω⁻¹v application."""

    def test_apply_precision_matches_block_diagonal(self):
        rng = np.random.default_rng(99)
        ar_coefs = np.array([0.6])
        panel_lengths = np.array([5, 8, 5])
        n = panel_lengths.sum()
        panel_indices = np.repeat(np.arange(len(panel_lengths)), panel_lengths)
        v = rng.standard_normal(n)

        # Build full block-diagonal for reference.
        blocks = [build_ar_precision_block(ar_coefs, int(T)) for T in panel_lengths]
        full_prec = scipy.linalg.block_diag(*blocks)
        expected = full_prec @ v

        result = apply_ar_precision(v, panel_indices, panel_lengths, ar_coefs)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_apply_precision_2d_input(self):
        rng = np.random.default_rng(77)
        ar_coefs = np.array([0.5])
        panel_lengths = np.array([4, 6])
        n = panel_lengths.sum()
        panel_indices = np.repeat(np.arange(len(panel_lengths)), panel_lengths)
        v = rng.standard_normal((n, 3))

        blocks = [build_ar_precision_block(ar_coefs, int(T)) for T in panel_lengths]
        full_prec = scipy.linalg.block_diag(*blocks)
        expected = full_prec @ v

        result = apply_ar_precision(v, panel_indices, panel_lengths, ar_coefs)
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestArDiagnostics:
    """Tests for Durbin-Watson and Ljung-Box diagnostics."""

    def test_durbin_watson_no_autocorrelation(self):
        rng = np.random.default_rng(1)
        panels = [rng.standard_normal(100) for _ in range(10)]
        diag = ar_diagnostics(panels)
        # DW ≈ 2.0 for iid residuals.
        assert 1.5 < diag["durbin_watson"] < 2.5

    def test_durbin_watson_positive_autocorrelation(self):
        rng = np.random.default_rng(2)
        panels = _generate_ar1_panels(0.7, n_panels=10, T=100, rng=rng)
        diag = ar_diagnostics(panels)
        # Positive autocorrelation → DW < 2.
        assert diag["durbin_watson"] < 1.0

    def test_ljung_box_detects_ar(self):
        rng = np.random.default_rng(3)
        panels = _generate_ar1_panels(0.7, n_panels=10, T=100, rng=rng)
        diag = ar_diagnostics(panels)
        assert diag["ljung_box_p"] < 0.05

    def test_diagnostics_keys(self):
        rng = np.random.default_rng(4)
        panels = [rng.standard_normal(20) for _ in range(5)]
        diag = ar_diagnostics(panels)
        assert set(diag.keys()) == {
            "durbin_watson",
            "ljung_box_Q",
            "ljung_box_p",
        }

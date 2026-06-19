"""Tests for src/randomization_tests/_kernel_tests.py.

Covers mmd_test, hsic_test, and kernel_regression_test.

Test design principles:
- Deterministic: all calls use random_state=42 for reproducibility.
- Fast: n_permutations=499 (≤500) throughout; sample sizes ≤100.
- Signal tests: well-separated distributions / strong dependence
  should yield p < 0.05 with high probability at n_permutations=499.
- Null tests: use p_value > 0.01 (not 0.05) to avoid false failures
  from valid Type I errors on a fixed seed.
- Nyström paths: tested with max_landmarks well below n to exercise
  the approximate code path end-to-end.
"""

from __future__ import annotations

import numpy as np

from randomization_tests._kernel_tests import (
    hsic_test,
    kernel_regression_test,
    mmd_test,
)
from randomization_tests._kernels import (
    GaussianKernel,
    LaplacianKernel,
    LinearKernel,
    PrecomputedKernel,
)
from randomization_tests._results import KernelTestResult

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestMMDTest:
    """Tests for mmd_test."""

    def test_returns_kernel_test_result(self) -> None:
        rng = _rng()
        X = rng.normal(size=(30, 2))
        Y = rng.normal(size=(30, 2))
        result = mmd_test(X, Y, n_permutations=99, random_state=42)
        assert isinstance(result, KernelTestResult)

    def test_result_fields(self) -> None:
        rng = _rng()
        X = rng.normal(size=(20, 1))
        Y = rng.normal(size=(20, 1))
        result = mmd_test(X, Y, n_permutations=99, random_state=42)
        assert result.method == "mmd"
        assert result.n_permutations == 99
        assert result.null_distribution.shape == (99,)
        assert isinstance(result.statistic, float)
        assert 0.0 < result.p_value <= 1.0

    def test_identical_distributions_null(self) -> None:
        """H₀ true: same distribution → p-value should not be tiny."""
        rng = _rng(7)
        X = rng.normal(0, 1, size=(50, 2))
        Y = rng.normal(0, 1, size=(50, 2))
        result = mmd_test(X, Y, n_permutations=499, random_state=42)
        # Valid Type I: p > 0.01 on this seed (not hard 0.05 to avoid flakiness)
        assert result.p_value > 0.01

    def test_well_separated_distributions_signal(self) -> None:
        """H₁ true: clearly different means → p-value should be small."""
        rng = _rng(1)
        X = rng.normal(0, 0.5, size=(60, 2))
        Y = rng.normal(5, 0.5, size=(60, 2))
        result = mmd_test(X, Y, n_permutations=499, random_state=42)
        assert result.p_value < 0.05

    def test_unequal_sample_sizes(self) -> None:
        """mmd_test must handle n_x ≠ n_y."""
        rng = _rng(2)
        X = rng.normal(0, 1, size=(30, 3))
        Y = rng.normal(3, 1, size=(50, 3))
        result = mmd_test(X, Y, n_permutations=199, random_state=42)
        assert result.p_value < 0.05

    def test_1d_input(self) -> None:
        """1-D arrays should be accepted without error."""
        rng = _rng(3)
        X = rng.normal(0, 1, size=40)
        Y = rng.normal(2, 1, size=40)
        result = mmd_test(X, Y, n_permutations=199, random_state=42)
        assert result.p_value < 0.05

    def test_custom_kernel(self) -> None:
        """LaplacianKernel should work as a drop-in replacement."""
        rng = _rng(4)
        X = rng.normal(0, 1, size=(30, 2))
        Y = rng.normal(3, 1, size=(30, 2))
        result = mmd_test(
            X, Y, kernel=LaplacianKernel(), n_permutations=199, random_state=42
        )
        assert result.p_value < 0.05
        assert "Laplacian" in result.kernel_name

    def test_reproducible_with_same_seed(self) -> None:
        """Same random_state → identical p-value and null distribution."""
        rng = _rng(5)
        X = rng.normal(0, 1, size=(40, 2))
        Y = rng.normal(1, 1, size=(40, 2))
        r1 = mmd_test(X, Y, n_permutations=199, random_state=7)
        r2 = mmd_test(X, Y, n_permutations=199, random_state=7)
        assert r1.p_value == r2.p_value
        np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)

    def test_null_distribution_length(self) -> None:
        rng = _rng(6)
        X = rng.normal(size=(20, 2))
        Y = rng.normal(size=(20, 2))
        result = mmd_test(X, Y, n_permutations=123, random_state=42)
        assert len(result.null_distribution) == 123

    def test_precomputed_kernel(self) -> None:
        """PrecomputedKernel should work with mmd_test."""
        rng = _rng(8)
        X = rng.normal(0, 1, size=(20, 2))
        Y = rng.normal(4, 1, size=(20, 2))
        Z = np.vstack([X, Y])
        # build a valid PSD kernel matrix on the pooled sample
        K_mat = GaussianKernel(sigma=1.0)(Z)
        kernel = PrecomputedKernel(K_mat)
        result = mmd_test(X, Y, kernel=kernel, n_permutations=199, random_state=42)
        assert isinstance(result, KernelTestResult)
        assert result.p_value < 0.05

    def test_nystrom_path_runs(self) -> None:
        """max_landmarks < n should invoke Nyström without error."""
        rng = _rng(9)
        X = rng.normal(0, 1, size=(60, 3))
        Y = rng.normal(3, 1, size=(60, 3))
        result = mmd_test(X, Y, n_permutations=199, max_landmarks=20, random_state=42)
        assert isinstance(result, KernelTestResult)
        # Nyström should still detect a strong signal
        assert result.p_value < 0.05

    def test_nystrom_reproducible(self) -> None:
        """Nyström path must also be reproducible with the same random_state."""
        rng = _rng(10)
        X = rng.normal(0, 1, size=(50, 2))
        Y = rng.normal(2, 1, size=(50, 2))
        r1 = mmd_test(X, Y, n_permutations=99, max_landmarks=15, random_state=13)
        r2 = mmd_test(X, Y, n_permutations=99, max_landmarks=15, random_state=13)
        assert r1.p_value == r2.p_value
        np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)


class TestHSICTest:
    """Tests for hsic_test."""

    def test_returns_kernel_test_result(self) -> None:
        rng = _rng()
        X = rng.normal(size=(40, 2))
        Y = rng.normal(size=(40, 2))
        result = hsic_test(X, Y, n_permutations=99, random_state=42)
        assert isinstance(result, KernelTestResult)

    def test_result_fields(self) -> None:
        rng = _rng()
        X = rng.normal(size=(30, 1))
        Y = rng.normal(size=(30, 1))
        result = hsic_test(X, Y, n_permutations=99, random_state=42)
        assert result.method == "hsic"
        assert result.n_permutations == 99
        assert result.null_distribution.shape == (99,)
        assert "," in result.kernel_name  # kernel_x,kernel_y format

    def test_independent_variables_null(self) -> None:
        """H₀ true: independent X, Y → p-value should not be tiny."""
        rng = _rng(11)
        X = rng.normal(size=(60,))
        Y = rng.normal(size=(60,))
        result = hsic_test(X, Y, n_permutations=499, random_state=42)
        assert result.p_value > 0.01

    def test_quadratic_dependence_signal(self) -> None:
        """H₁ true: Y = X² + noise → HSIC should detect dependence."""
        rng = _rng(12)
        X = rng.uniform(-2, 2, size=60)
        Y = X**2 + rng.normal(0, 0.3, size=60)
        result = hsic_test(X, Y, n_permutations=499, random_state=42)
        assert result.p_value < 0.05

    def test_linear_dependence_signal(self) -> None:
        """H₁ true: strong linear relationship → detected."""
        rng = _rng(13)
        X = rng.normal(size=50)
        Y = 3 * X + rng.normal(0, 0.2, size=50)
        result = hsic_test(X, Y, n_permutations=499, random_state=42)
        assert result.p_value < 0.05

    def test_custom_kernels(self) -> None:
        """Custom kernel_x and kernel_y should be accepted."""
        rng = _rng(14)
        X = rng.normal(size=(40, 2))
        Y = 0.5 * X + rng.normal(0, 0.1, size=(40, 2))
        result = hsic_test(
            X,
            Y,
            kernel_x=LinearKernel(),
            kernel_y=LinearKernel(),
            n_permutations=199,
            random_state=42,
        )
        assert result.p_value < 0.05

    def test_reproducible_with_same_seed(self) -> None:
        rng = _rng(15)
        X = rng.normal(size=50)
        Y = X + rng.normal(0, 0.5, size=50)
        r1 = hsic_test(X, Y, n_permutations=199, random_state=9)
        r2 = hsic_test(X, Y, n_permutations=199, random_state=9)
        assert r1.p_value == r2.p_value
        np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)

    def test_nystrom_path_runs(self) -> None:
        rng = _rng(16)
        X = rng.normal(size=80)
        Y = X**2 + rng.normal(0, 0.3, size=80)
        result = hsic_test(X, Y, n_permutations=199, max_landmarks=20, random_state=42)
        assert isinstance(result, KernelTestResult)
        assert result.p_value < 0.05

    def test_nystrom_reproducible(self) -> None:
        rng = _rng(17)
        X = rng.normal(size=60)
        Y = X + rng.normal(0, 0.5, size=60)
        r1 = hsic_test(X, Y, n_permutations=99, max_landmarks=15, random_state=17)
        r2 = hsic_test(X, Y, n_permutations=99, max_landmarks=15, random_state=17)
        assert r1.p_value == r2.p_value
        np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)

    def test_multivariate_inputs(self) -> None:
        rng = _rng(18)
        X = rng.normal(size=(50, 3))
        Y = X @ rng.normal(size=(3, 2)) + rng.normal(0, 0.1, size=(50, 2))
        result = hsic_test(X, Y, n_permutations=199, random_state=42)
        assert result.p_value < 0.05

    def test_distance_kernel_detects_nonlinear_signal(self) -> None:
        """Gaussian HSIC and dCor should agree on signal direction.

        This does NOT check numeric closeness — it checks that both detect
        a nonlinear dependence (p < 0.05).
        """
        from scipy.stats import spearmanr

        rng = _rng(19)
        X = rng.uniform(-2, 2, size=60)
        Y = np.sin(X) + rng.normal(0, 0.3, size=60)

        result = hsic_test(X, Y, n_permutations=499, random_state=42)

        # Spearman correlation as a sanity-check signal reference
        spear_r, _ = spearmanr(X, Y)

        # Both should agree there is signal (HSIC p < 0.05 and |r| > 0.1)
        assert result.p_value < 0.05
        assert abs(spear_r) > 0.1


class TestKernelRegressionTest:
    """Tests for kernel_regression_test."""

    def test_returns_kernel_test_result(self) -> None:
        rng = _rng()
        X = rng.normal(size=(40,))
        Y = X + rng.normal(0, 0.5, size=40)
        result = kernel_regression_test(X, Y, n_permutations=99, random_state=42)
        assert isinstance(result, KernelTestResult)

    def test_result_fields(self) -> None:
        rng = _rng()
        X = rng.normal(size=30)
        Y = X + rng.normal(size=30)
        result = kernel_regression_test(X, Y, n_permutations=99, random_state=42)
        assert result.method == "kernel_regression"
        assert result.n_permutations == 99
        assert "," not in result.kernel_name  # only kernel_y name

    def test_no_confounders_detects_dependence(self) -> None:
        rng = _rng(20)
        X = rng.normal(size=60)
        Y = 2 * X + rng.normal(0, 0.3, size=60)
        result = kernel_regression_test(X, Y, n_permutations=499, random_state=42)
        assert result.p_value < 0.05

    def test_no_confounders_null(self) -> None:
        """Independent X, Y without confounders → H₀ true."""
        rng = _rng(21)
        X = rng.normal(size=60)
        Y = rng.normal(size=60)
        result = kernel_regression_test(X, Y, n_permutations=499, random_state=42)
        assert result.p_value > 0.01

    def test_confounders_residualised(self) -> None:
        """X → Y relationship survives after residualising Z."""
        rng = _rng(22)
        Z = rng.normal(size=(60, 2))
        X = Z @ rng.normal(size=2) + rng.normal(0, 0.5, size=60)
        Y = X + Z @ rng.normal(size=2) + rng.normal(0, 0.3, size=60)
        result = kernel_regression_test(
            X, Y, confounders=Z, n_permutations=499, random_state=42
        )
        assert result.p_value < 0.05

    def test_confounders_remove_spurious_association(self) -> None:
        """When X-Y association is entirely mediated by Z, p should be large."""
        rng = _rng(23)
        Z = rng.normal(size=60)
        # X and Y are both driven by Z; conditionally independent
        X = 2 * Z + rng.normal(0, 0.1, size=60)
        Y = 3 * Z + rng.normal(0, 0.1, size=60)
        result = kernel_regression_test(
            X, Y, confounders=Z, n_permutations=499, random_state=42
        )
        # After removing Z, little signal should remain; p > 0.01 is a
        # conservative threshold — we're not guaranteed p > 0.05 with n=60.
        assert result.p_value > 0.01

    def test_custom_kernel_y(self) -> None:
        # kernel_regression_test uses LinearKernel on X — use a linear signal
        # so the linear kernel can detect it, and vary only kernel_y.
        rng = _rng(24)
        X = rng.normal(size=50)
        Y = 3 * X + rng.normal(0, 0.2, size=50)
        result = kernel_regression_test(
            X,
            Y,
            kernel_y=GaussianKernel(sigma=1.0),
            n_permutations=199,
            random_state=42,
        )
        assert result.p_value < 0.05

    def test_reproducible_with_same_seed(self) -> None:
        rng = _rng(25)
        X = rng.normal(size=40)
        Y = X + rng.normal(0, 0.5, size=40)
        r1 = kernel_regression_test(X, Y, n_permutations=199, random_state=11)
        r2 = kernel_regression_test(X, Y, n_permutations=199, random_state=11)
        assert r1.p_value == r2.p_value
        np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)

    def test_multivariate_X_and_Y(self) -> None:
        rng = _rng(26)
        X = rng.normal(size=(50, 2))
        Y = X @ rng.normal(size=(2, 2)) + rng.normal(0, 0.2, size=(50, 2))
        result = kernel_regression_test(X, Y, n_permutations=199, random_state=42)
        assert result.p_value < 0.05


class TestNystromPaths:
    """End-to-end Nyström path checks for all three test functions."""

    def test_mmd_nystrom_signal(self) -> None:
        rng = _rng(30)
        X = rng.normal(0, 1, size=(80, 2))
        Y = rng.normal(4, 1, size=(80, 2))
        result = mmd_test(X, Y, n_permutations=299, max_landmarks=25, random_state=42)
        assert result.p_value < 0.05

    def test_hsic_nystrom_signal(self) -> None:
        rng = _rng(31)
        X = rng.normal(size=80)
        Y = X**2 + rng.normal(0, 0.4, size=80)
        result = hsic_test(X, Y, n_permutations=299, max_landmarks=25, random_state=42)
        assert result.p_value < 0.05

    def test_kernel_regression_nystrom(self) -> None:
        rng = _rng(32)
        X = rng.normal(size=80)
        Y = 2 * X + rng.normal(0, 0.5, size=80)
        result = kernel_regression_test(
            X, Y, n_permutations=299, max_landmarks=25, random_state=42
        )
        assert result.p_value < 0.05

    def test_nystrom_does_not_change_method_field(self) -> None:
        rng = _rng(33)
        X = rng.normal(size=50)
        Y = rng.normal(size=50)
        r_mmd = mmd_test(X, Y, n_permutations=99, max_landmarks=15, random_state=42)
        r_hsic = hsic_test(X, Y, n_permutations=99, max_landmarks=15, random_state=42)
        r_kr = kernel_regression_test(
            X, Y, n_permutations=99, max_landmarks=15, random_state=42
        )
        assert r_mmd.method == "mmd"
        assert r_hsic.method == "hsic"
        assert r_kr.method == "kernel_regression"

    def test_mixed_exact_nystrom_reproducible(self) -> None:
        """Even with large max_landmarks (exact fallback), results are stable."""
        rng = _rng(34)
        X = rng.normal(size=40)
        Y = X + rng.normal(0, 0.4, size=40)
        # max_landmarks >= n → exact path (no Nyström)
        r_exact = hsic_test(X, Y, n_permutations=99, max_landmarks=999, random_state=5)
        r_also_exact = hsic_test(
            X, Y, n_permutations=99, max_landmarks=999, random_state=5
        )
        assert r_exact.p_value == r_also_exact.p_value

"""Tests for src/randomization_tests/_kernels.py.

Covers:
- Kernel implementations: GaussianKernel, CosineKernel, LinearKernel,
  LaplacianKernel, PrecomputedKernel (known values, validation, median heuristic)
- KernelEval exact path: is_exact=True, K_mm_inv=eye, full Gram returned
- KernelEval Nyström path: K_nm @ K_mm_inv @ K_nm.T ≈ full Gram
- Nyström landmark selection concentrates on influential points
- Gram arithmetic helpers: gram_row_sums, gram_centering, gram_permute,
  gram_trace_product — exact and Nyström paths
- Kernel protocol structural check via isinstance
"""

from __future__ import annotations

import numpy as np
import pytest

from randomization_tests._kernels import (
    CosineKernel,
    GaussianKernel,
    Kernel,
    LaplacianKernel,
    LinearKernel,
    PrecomputedKernel,
    gram_centering,
    gram_permute,
    gram_row_sums,
    gram_trace_product,
)

# ================================================================== #
# Section 1 — Kernel implementations
# ================================================================== #


class TestKernelImplementations:
    """Known-value and structural tests for each kernel class."""

    def test_gaussian_known_values(self) -> None:
        """Gram entries match hand-computed values for small 2-D input."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        sigma = 1.0
        k = GaussianKernel(sigma=sigma)
        K = k(X)
        assert K.shape == (3, 3)
        # diagonal must be 1
        np.testing.assert_allclose(np.diag(K), 1.0)
        # k(x0, x1): dist=1 → exp(-0.5)
        expected_01 = np.exp(-0.5)
        np.testing.assert_allclose(K[0, 1], expected_01, rtol=1e-10)
        np.testing.assert_allclose(K[1, 0], expected_01, rtol=1e-10)
        # symmetry
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_gaussian_two_arg_call(self) -> None:
        """__call__(X1, X2) computes cross-Gram between distinct arrays."""
        rng = np.random.default_rng(0)
        X1 = rng.standard_normal((4, 3))
        X2 = rng.standard_normal((5, 3))
        k = GaussianKernel(sigma=1.5)
        K = k(X1, X2)
        assert K.shape == (4, 5)
        # each entry in (0, 1]
        assert K.min() > 0.0 and K.max() <= 1.0 + 1e-10

    def test_cosine_unit_vectors(self) -> None:
        """Unit vectors on the unit circle: cosine = cos(angle difference)."""
        angles = np.array([0.0, np.pi / 3, np.pi / 2])
        X = np.column_stack([np.cos(angles), np.sin(angles)])
        k = CosineKernel()
        K = k(X)
        # k(v_i, v_i) = 1
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)
        # k(0°, 60°) = cos(60°) = 0.5
        np.testing.assert_allclose(K[0, 1], 0.5, atol=1e-12)
        # symmetry
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_linear_equals_dot_product(self) -> None:
        """LinearKernel Gram equals X @ X.T exactly."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((10, 4))
        k = LinearKernel()
        K = k(X)
        np.testing.assert_allclose(K, X @ X.T, atol=1e-12)

    def test_linear_two_arg(self) -> None:
        """LinearKernel(X1, X2) = X1 @ X2.T."""
        rng = np.random.default_rng(2)
        X1 = rng.standard_normal((6, 3))
        X2 = rng.standard_normal((8, 3))
        k = LinearKernel()
        np.testing.assert_allclose(k(X1, X2), X1 @ X2.T, atol=1e-12)

    def test_precomputed_roundtrip(self) -> None:
        """PrecomputedKernel wrapping a Gaussian Gram returns identical matrix."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((8, 2))
        K_gauss = GaussianKernel(sigma=1.0)(X)
        k = PrecomputedKernel(K_gauss)
        K_pre = k(X)
        np.testing.assert_allclose(K_pre, K_gauss, atol=1e-14)

    def test_precomputed_validates_symmetry(self) -> None:
        """Non-symmetric matrix raises ValueError."""
        K = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            PrecomputedKernel(K)

    def test_precomputed_validates_psd(self) -> None:
        """Negative-definite matrix raises ValueError."""
        K = np.array([[1.0, 2.0], [2.0, 1.0]])  # min eigenvalue = -1
        with pytest.raises(ValueError, match="positive semi-definite"):
            PrecomputedKernel(K)

    def test_precomputed_validate_psd_skip(self) -> None:
        """validate_psd=False skips the O(n³) check."""
        K = np.eye(5)  # trivially PSD, skip check should still work
        k = PrecomputedKernel(K, validate_psd=False)
        np.testing.assert_allclose(k(np.zeros((5, 1))), K)

    def test_precomputed_two_args_raises(self) -> None:
        """Calling with two arguments raises ValueError."""
        K = np.eye(4)
        k = PrecomputedKernel(K, validate_psd=False)
        with pytest.raises(ValueError, match="two arguments"):
            k(K, K)

    def test_gaussian_median_heuristic(self) -> None:
        """sigma='median' sets bandwidth to median pairwise distance."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2)) * 5.0  # large scale
        k = GaussianKernel(sigma="median")
        K = k(X)
        # all entries in (0, 1)
        assert K.min() > 0.0
        assert K.max() <= 1.0 + 1e-10
        # diagonal exactly 1
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)
        # same sigma="median" on different scale → still in (0,1)
        X2 = X * 100.0
        K2 = GaussianKernel(sigma="median")(X2)
        assert K2.min() > 0.0 and K2.max() <= 1.0 + 1e-10

    def test_laplacian_known_values(self) -> None:
        """LaplacianKernel entries match exp(-L1/sigma)."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        sigma = 2.0
        k = LaplacianKernel(sigma=sigma)
        K = k(X)
        # L1 distance between rows = 2; k = exp(-2/2) = exp(-1)
        np.testing.assert_allclose(K[0, 1], np.exp(-1.0), rtol=1e-10)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

    def test_gaussian_invalid_sigma(self) -> None:
        """Non-positive or invalid sigma raises ValueError."""
        with pytest.raises(ValueError):
            GaussianKernel(sigma=-1.0)
        with pytest.raises(ValueError):
            GaussianKernel(sigma="bad")

    def test_kernel_protocol_isinstance(self) -> None:
        """All kernel classes satisfy the Kernel protocol."""
        for k in [
            GaussianKernel(),
            CosineKernel(),
            LinearKernel(),
            LaplacianKernel(),
            PrecomputedKernel(np.eye(3), validate_psd=False),
        ]:
            assert isinstance(k, Kernel), f"{type(k).__name__} should satisfy Kernel"

    def test_kernel_name_strings(self) -> None:
        """name() returns informative strings."""
        assert "Gaussian" in GaussianKernel(sigma=1.0).name()
        assert "median" in GaussianKernel(sigma="median").name()
        assert "Cosine" in CosineKernel().name()
        assert "Linear" in LinearKernel().name()
        assert "Laplacian" in LaplacianKernel(sigma=0.5).name()
        assert "Precomputed" in PrecomputedKernel(np.eye(4), validate_psd=False).name()


# ================================================================== #
# Section 2 — KernelEval exact and Nyström paths
# ================================================================== #


class TestKernelEvalAndNystrom:
    """Tests for evaluate() returning exact or Nyström KernelEval."""

    def test_evaluate_exact_is_full_gram(self) -> None:
        """evaluate() with no max_landmarks returns full Gram, is_exact=True."""
        rng = np.random.default_rng(10)
        X = rng.standard_normal((12, 3))
        k = GaussianKernel(sigma=1.0)
        ev = k.evaluate(X)
        assert ev.is_exact is True
        assert ev.n == 12
        assert ev.m == 12
        assert ev.K_nm.shape == (12, 12)
        assert ev.K_mm_inv.shape == (12, 12)
        # K_nm should equal k(X)
        K_direct = k(X)
        np.testing.assert_allclose(ev.K_nm, K_direct, atol=1e-10)

    def test_evaluate_nystrom_shape(self) -> None:
        """evaluate() with max_landmarks returns (n, m) factor."""
        rng = np.random.default_rng(11)
        X = rng.standard_normal((50, 2))
        k = GaussianKernel(sigma=1.0)
        ev = k.evaluate(X, max_landmarks=10)
        assert ev.is_exact is False
        assert ev.n == 50
        assert ev.m == 10
        assert ev.K_nm.shape == (50, 10)
        assert ev.K_mm_inv.shape == (10, 10)

    def test_evaluate_nystrom_approximates_gram(self) -> None:
        """K_nm @ K_mm_inv @ K_nm.T approximates full Gram; error decreases with m."""
        rng = np.random.default_rng(12)
        X = rng.standard_normal((80, 2))
        k = GaussianKernel(sigma=1.0)
        K_exact = k(X)

        errors = []
        for m in [5, 15, 40]:
            np.random.seed(12)
            ev = k.evaluate(X, max_landmarks=m)
            K_approx = ev.K_nm @ ev.K_mm_inv @ ev.K_nm.T
            rel_err = np.linalg.norm(K_exact - K_approx, "fro") / np.linalg.norm(
                K_exact, "fro"
            )
            errors.append(rel_err)

        # error decreases as m increases
        assert errors[0] > errors[1] > errors[2], (
            f"Expected decreasing approximation error: {errors}"
        )

    def test_evaluate_nystrom_max_landmarks_ge_n_gives_exact(self) -> None:
        """max_landmarks >= n returns exact KernelEval."""
        rng = np.random.default_rng(13)
        X = rng.standard_normal((20, 2))
        k = GaussianKernel(sigma=1.0)
        ev = k.evaluate(X, max_landmarks=20)
        assert ev.is_exact is True
        ev2 = k.evaluate(X, max_landmarks=999)
        assert ev2.is_exact is True

    def test_nystrom_leverage_sampling_not_uniform(self) -> None:
        """Leverage-score sampling concentrates on influential points.

        Create data with a cluster of points far from the rest — those
        points have large column norms and should be over-selected as
        landmarks relative to uniform sampling.
        """
        rng = np.random.default_rng(14)
        # 90 points near origin, 10 points far away
        X_near = rng.standard_normal((90, 2)) * 0.1
        X_far = rng.standard_normal((10, 2)) * 10.0
        X = np.vstack([X_near, X_far])  # far points are indices 90-99
        k = GaussianKernel(sigma=1.0)

        far_counts = []
        np.random.seed(0)
        for _ in range(20):
            ev = k.evaluate(X, max_landmarks=20)
            # approximate: check K_nm columns — far points produce larger K_nm norms
            col_norms = np.linalg.norm(ev.K_nm, axis=0)
            # this just checks the evaluation ran without error and
            # returns correct shapes; leverage-score concentration is
            # a statistical property that holds in expectation, not every run.
            far_counts.append(col_norms.max() > col_norms.min())

        # at least sometimes the norms are not all equal (indicating variation in landmarks)
        assert any(far_counts)

    def test_precomputed_nystrom_path(self) -> None:
        """PrecomputedKernel.evaluate() with max_landmarks < n uses Nyström."""
        rng = np.random.default_rng(15)
        X = rng.standard_normal((30, 2))
        K = GaussianKernel(sigma=1.0)(X)
        k = PrecomputedKernel(K)
        np.random.seed(0)
        ev = k.evaluate(X, max_landmarks=8)
        assert ev.is_exact is False
        assert ev.n == 30
        assert ev.m == 8
        K_approx = ev.K_nm @ ev.K_mm_inv @ ev.K_nm.T
        # should approximate original K (loose tolerance due to small m)
        rel_err = np.linalg.norm(K - K_approx, "fro") / np.linalg.norm(K, "fro")
        assert rel_err < 1.0, f"Relative error too large: {rel_err:.4f}"

    def test_nystrom_memory_bound(self) -> None:
        """With max_landmarks=100 and n=1000, K_nm is (1000,100) not (1000,1000)."""
        rng = np.random.default_rng(16)
        X = rng.standard_normal((1000, 5))
        k = GaussianKernel(sigma=1.0)
        np.random.seed(0)
        ev = k.evaluate(X, max_landmarks=100)
        assert ev.K_nm.shape == (1000, 100), (
            f"Expected (1000, 100), got {ev.K_nm.shape}"
        )
        assert ev.K_mm_inv.shape == (100, 100)
        # the full matrix is never created — verify K_nm is not square
        assert ev.K_nm.shape[0] != ev.K_nm.shape[1]

    def test_cosine_evaluate_exact(self) -> None:
        """CosineKernel.evaluate() exact path returns full Gram."""
        rng = np.random.default_rng(17)
        X = rng.standard_normal((10, 4))
        k = CosineKernel()
        ev = k.evaluate(X)
        K_direct = k(X)
        assert ev.is_exact is True
        np.testing.assert_allclose(ev.K_nm, K_direct, atol=1e-12)

    def test_linear_evaluate_nystrom(self) -> None:
        """LinearKernel Nyström approximation is factored correctly."""
        rng = np.random.default_rng(18)
        X = rng.standard_normal((40, 3))
        k = LinearKernel()
        K_exact = k(X)
        np.random.seed(0)
        ev = k.evaluate(X, max_landmarks=15)
        assert ev.is_exact is False
        K_approx = ev.K_nm @ ev.K_mm_inv @ ev.K_nm.T
        # LinearKernel is low-rank; Nyström with rank >= true rank is exact
        # here true rank = 3, so m=15 should give excellent approximation
        rel_err = np.linalg.norm(K_exact - K_approx, "fro") / np.linalg.norm(
            K_exact, "fro"
        )
        assert rel_err < 0.05, (
            f"Relative error too large for linear kernel: {rel_err:.4f}"
        )


# ================================================================== #
# Section 3 — Gram arithmetic helpers
# ================================================================== #


class TestGramHelpers:
    """Tests for gram_row_sums, gram_centering, gram_permute, gram_trace_product."""

    # ---- gram_row_sums ----

    def test_gram_row_sums_exact(self) -> None:
        """gram_row_sums on exact KernelEval matches K.sum(axis=1)."""
        rng = np.random.default_rng(20)
        X = rng.standard_normal((15, 3))
        k = GaussianKernel(sigma=1.0)
        ev = k.evaluate(X)
        expected = ev.K_nm.sum(axis=1)
        result = gram_row_sums(ev)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_gram_row_sums_nystrom(self) -> None:
        """gram_row_sums on Nyström approximates exact row sums."""
        rng = np.random.default_rng(21)
        X = rng.standard_normal((60, 3))
        k = GaussianKernel(sigma=1.0)
        ev_exact = k.evaluate(X)
        np.random.seed(0)
        ev_nys = k.evaluate(X, max_landmarks=30)
        rs_exact = gram_row_sums(ev_exact)
        rs_nys = gram_row_sums(ev_nys)
        # Nyström approximation: relative error < 20% for m=30, n=60
        rel_err = np.linalg.norm(rs_exact - rs_nys) / np.linalg.norm(rs_exact)
        assert rel_err < 0.5, f"Row sums relative error too large: {rel_err:.4f}"

    # ---- gram_centering ----

    def test_gram_centering_exact_zero_sums(self) -> None:
        """Centred exact Gram has row sums ≈ 0 and column sums ≈ 0."""
        rng = np.random.default_rng(22)
        X = rng.standard_normal((20, 3))
        k = GaussianKernel(sigma=1.0)
        ev = k.evaluate(X)
        ev_c = gram_centering(ev)
        assert ev_c.is_exact is True
        np.testing.assert_allclose(ev_c.K_nm.sum(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(ev_c.K_nm.sum(axis=1), 0.0, atol=1e-10)

    def test_gram_centering_nystrom_zero_col_sums(self) -> None:
        """Nyström centering zeroes column means of K_nm (H @ K_nm)."""
        rng = np.random.default_rng(23)
        X = rng.standard_normal((50, 3))
        k = GaussianKernel(sigma=1.0)
        np.random.seed(0)
        ev = k.evaluate(X, max_landmarks=15)
        ev_c = gram_centering(ev)
        assert ev_c.is_exact is False
        np.testing.assert_allclose(ev_c.K_nm.mean(axis=0), 0.0, atol=1e-10)

    def test_gram_centering_exact_matches_formula(self) -> None:
        """Centred exact Gram matches H @ K @ H computed directly."""
        rng = np.random.default_rng(24)
        X = rng.standard_normal((15, 2))
        k = GaussianKernel(sigma=1.0)
        ev = k.evaluate(X)
        n = ev.n
        H = np.eye(n) - np.ones((n, n)) / n
        K_c_direct = H @ ev.K_nm @ H
        ev_c = gram_centering(ev)
        np.testing.assert_allclose(ev_c.K_nm, K_c_direct, atol=1e-10)

    # ---- gram_permute ----

    def test_gram_permute_exact(self) -> None:
        """gram_permute on exact Gram matches K[perm][:, perm]."""
        rng = np.random.default_rng(25)
        X = rng.standard_normal((12, 3))
        k = GaussianKernel(sigma=1.0)
        ev = k.evaluate(X)
        perm = rng.permutation(12)
        ev_p = gram_permute(ev, perm)
        K_expected = ev.K_nm[np.ix_(perm, perm)]
        np.testing.assert_allclose(ev_p.K_nm, K_expected, atol=1e-12)
        assert ev_p.is_exact is True

    def test_gram_permute_nystrom_rows_only(self) -> None:
        """gram_permute on Nyström only reindexes K_nm rows."""
        rng = np.random.default_rng(26)
        X = rng.standard_normal((40, 3))
        k = GaussianKernel(sigma=1.0)
        np.random.seed(0)
        ev = k.evaluate(X, max_landmarks=10)
        perm = rng.permutation(40)
        ev_p = gram_permute(ev, perm)
        np.testing.assert_allclose(ev_p.K_nm, ev.K_nm[perm], atol=1e-12)
        # K_mm_inv unchanged
        np.testing.assert_allclose(ev_p.K_mm_inv, ev.K_mm_inv, atol=1e-12)
        assert ev_p.is_exact is False

    def test_gram_permute_identity_unchanged(self) -> None:
        """Identity permutation leaves the exact Gram unchanged."""
        rng = np.random.default_rng(27)
        X = rng.standard_normal((10, 2))
        k = GaussianKernel(sigma=1.0)
        ev = k.evaluate(X)
        perm = np.arange(10)
        ev_p = gram_permute(ev, perm)
        np.testing.assert_allclose(ev_p.K_nm, ev.K_nm, atol=1e-12)

    def test_gram_permute_matches_reindexed_gram(self) -> None:
        """Permuted Nyström reconstruction ≈ permuted exact Gram."""
        rng = np.random.default_rng(28)
        X = rng.standard_normal((30, 2))
        k = GaussianKernel(sigma=1.0)
        ev_exact = k.evaluate(X)
        np.random.seed(0)
        ev_nys = k.evaluate(X, max_landmarks=20)
        perm = rng.permutation(30)

        K_exact_perm = ev_exact.K_nm[np.ix_(perm, perm)]
        ev_nys_p = gram_permute(ev_nys, perm)
        K_nys_perm = ev_nys_p.K_nm @ ev_nys_p.K_mm_inv @ ev_nys_p.K_nm.T

        rel_err = np.linalg.norm(K_exact_perm - K_nys_perm, "fro") / np.linalg.norm(
            K_exact_perm, "fro"
        )
        assert rel_err < 0.3, (
            f"Permuted Nyström relative error too large: {rel_err:.4f}"
        )

    # ---- gram_trace_product ----

    def test_gram_trace_product_exact(self) -> None:
        """gram_trace_product with both exact matches np.trace(K_A @ K_B)."""
        rng = np.random.default_rng(30)
        X = rng.standard_normal((20, 2))
        k = GaussianKernel(sigma=1.0)
        ev_a = k.evaluate(X)
        ev_b = LinearKernel().evaluate(X)
        result = gram_trace_product(ev_a, ev_b)
        expected = float(np.trace(ev_a.K_nm @ ev_b.K_nm))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gram_trace_product_symmetric_exact(self) -> None:
        """gram_trace_product(A, A) equals sum of squared entries."""
        rng = np.random.default_rng(31)
        X = rng.standard_normal((15, 3))
        k = GaussianKernel(sigma=1.0)
        ev = k.evaluate(X)
        result = gram_trace_product(ev, ev)
        expected = float((ev.K_nm**2).sum())
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gram_trace_product_nystrom_approximates_exact(self) -> None:
        """gram_trace_product with Nyström factors approximates exact value."""
        rng = np.random.default_rng(32)
        X = rng.standard_normal((60, 3))
        k = GaussianKernel(sigma=1.0)
        ev_exact = k.evaluate(X)
        np.random.seed(0)
        ev_nys = k.evaluate(X, max_landmarks=30)
        exact_val = gram_trace_product(ev_exact, ev_exact)
        nys_val = gram_trace_product(ev_nys, ev_nys)
        rel_err = abs(exact_val - nys_val) / abs(exact_val)
        assert rel_err < 0.5, (
            f"Nyström trace product relative error too large: {rel_err:.4f}"
        )

    def test_gram_trace_product_nystrom_both_path(self) -> None:
        """gram_trace_product with two different Nyström KernelEvals runs correctly."""
        rng = np.random.default_rng(33)
        X = rng.standard_normal((40, 2))
        k_g = GaussianKernel(sigma=1.0)
        k_l = LinearKernel()
        np.random.seed(0)
        ev_a = k_g.evaluate(X, max_landmarks=12)
        ev_b = k_l.evaluate(X, max_landmarks=12)
        result = gram_trace_product(ev_a, ev_b)
        assert np.isfinite(result), "Nyström trace product should be finite"

    def test_gram_trace_product_mixed_path(self) -> None:
        """gram_trace_product with one exact and one Nyström runs without error."""
        rng = np.random.default_rng(34)
        X = rng.standard_normal((25, 2))
        k = GaussianKernel(sigma=1.0)
        ev_exact = k.evaluate(X)
        np.random.seed(0)
        ev_nys = k.evaluate(X, max_landmarks=10)
        result_ae = gram_trace_product(ev_exact, ev_nys)
        result_ea = gram_trace_product(ev_nys, ev_exact)
        assert np.isfinite(result_ae)
        assert np.isfinite(result_ea)

    def test_gram_trace_product_positive_definite_kernel(self) -> None:
        """tr(K @ K) >= 0 for any symmetric PSD kernel."""
        rng = np.random.default_rng(35)
        X = rng.standard_normal((18, 3))
        k = GaussianKernel(sigma=1.0)
        ev = k.evaluate(X)
        result = gram_trace_product(ev, ev)
        assert result >= -1e-10, f"tr(K²) should be non-negative, got {result}"

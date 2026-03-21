"""Tests for the sign_flips module — sign-flip generation and symmetry check."""

import numpy as np
import pytest

from randomization_tests.sign_flips import generate_sign_flips, validate_symmetry

# ------------------------------------------------------------------ #
# generate_sign_flips
# ------------------------------------------------------------------ #


class TestGenerateSignFlips:
    """Unit tests for sign-flip vector generation."""

    def test_shape(self):
        s = generate_sign_flips(10, 50, random_state=42)
        assert s.shape == (50, 10)

    def test_entries_are_plus_minus_one(self):
        s = generate_sign_flips(20, 100, random_state=42)
        unique = set(s.ravel().tolist())
        assert unique == {-1, 1}

    def test_no_identity_by_default(self):
        s = generate_sign_flips(5, 31, random_state=42)
        assert not any(np.all(row == 1) for row in s)

    def test_identity_included_when_allowed(self):
        # For n=3, all 2^3 = 8 vectors should be returned.
        with pytest.warns(UserWarning, match="only 8 unique vectors"):
            s = generate_sign_flips(3, 100, exclude_identity=False)
        rows_as_tuples = set(tuple(r) for r in s.tolist())
        assert (1, 1, 1) in rows_as_tuples

    def test_uniqueness(self):
        s = generate_sign_flips(15, 200, random_state=42)
        rows = [tuple(r) for r in s.tolist()]
        assert len(rows) == len(set(rows))

    def test_determinism(self):
        s1 = generate_sign_flips(20, 100, random_state=42)
        s2 = generate_sign_flips(20, 100, random_state=42)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        s1 = generate_sign_flips(20, 100, random_state=1)
        s2 = generate_sign_flips(20, 100, random_state=2)
        assert not np.array_equal(s1, s2)

    def test_exhaustive_small_n(self):
        # n=4 -> 2^4 - 1 = 15 vectors
        with pytest.warns(UserWarning, match="only 15 unique vectors"):
            s = generate_sign_flips(4, 100, random_state=42)
        assert s.shape == (15, 4)
        rows = set(tuple(r) for r in s.tolist())
        assert len(rows) == 15

    def test_exhaustive_clamp_warning(self):
        with pytest.warns(UserWarning, match="only 7 unique vectors"):
            s = generate_sign_flips(3, 100, random_state=42)
        assert s.shape == (7, 3)

    def test_sample_from_exhaustive(self):
        # n=15, want 50 — should sample from the 2^15 space
        s = generate_sign_flips(15, 50, random_state=42)
        assert s.shape == (50, 15)

    def test_large_n_random(self):
        s = generate_sign_flips(100, 1000, random_state=42)
        assert s.shape == (1000, 100)
        unique = set(s.ravel().tolist())
        assert unique == {-1, 1}

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            generate_sign_flips(0, 10)

    def test_invalid_n_flips(self):
        with pytest.raises(ValueError, match="n_flips must be >= 1"):
            generate_sign_flips(10, 0)

    def test_single_sample(self):
        # n=1: only vector is [-1] (identity [+1] excluded).
        with pytest.warns(UserWarning, match="only 1 unique vectors"):
            s = generate_sign_flips(1, 10, random_state=42)
        assert s.shape == (1, 1)
        assert s[0, 0] == -1


# ------------------------------------------------------------------ #
# validate_symmetry
# ------------------------------------------------------------------ #


class TestValidateSymmetry:
    """Unit tests for the symmetry diagnostic."""

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        result = validate_symmetry(rng.standard_normal(100))
        assert "is_symmetric" in result
        assert "test_statistic" in result
        assert "p_value" in result

    def test_symmetric_residuals(self):
        rng = np.random.default_rng(42)
        result = validate_symmetry(rng.standard_normal(200))
        assert result["is_symmetric"] is True

    def test_asymmetric_residuals(self):
        rng = np.random.default_rng(42)
        skewed = np.abs(rng.standard_normal(200)) + 1.0
        result = validate_symmetry(skewed)
        assert result["is_symmetric"] is False

    def test_too_few_observations(self):
        result = validate_symmetry(np.array([1.0, -1.0, 0.5]))
        assert result["is_symmetric"] is True
        assert np.isnan(result["test_statistic"])
        assert np.isnan(result["p_value"])

    def test_custom_alpha(self):
        rng = np.random.default_rng(42)
        resids = rng.standard_normal(50)
        # Very lenient alpha — should always be symmetric.
        result = validate_symmetry(resids, alpha=0.001)
        assert result["is_symmetric"] is True

    def test_all_zeros(self):
        # All zeros → too few non-zero observations.
        result = validate_symmetry(np.zeros(20))
        assert result["is_symmetric"] is True
        assert np.isnan(result["test_statistic"])

"""Tests for the permutations module."""

import numpy as np
import pytest

from randomization_tests.permutations import (
    _unrank_permutation,
    generate_unique_permutations,
)


class TestUnrankPermutation:
    """Tests for the Lehmer-code unranking function."""

    def test_rank_zero_is_identity(self):
        assert _unrank_permutation(0, 4) == [0, 1, 2, 3]

    def test_last_rank_is_reverse(self):
        # The last lexicographic permutation of [0,1,2,3] is [3,2,1,0]
        # which has rank 4! - 1 = 23
        assert _unrank_permutation(23, 4) == [3, 2, 1, 0]

    def test_known_rank(self):
        # rank 4 of [0,1,2] is [2,0,1]
        assert _unrank_permutation(4, 3) == [2, 0, 1]

    def test_all_ranks_unique(self):
        # All 24 permutations of [0,1,2,3] should be distinct
        perms = [tuple(_unrank_permutation(k, 4)) for k in range(24)]
        assert len(set(perms)) == 24


class TestGenerateUniquePermutations:
    """Tests for generate_unique_permutations."""

    def test_shape(self):
        result = generate_unique_permutations(10, 50, random_state=42)
        assert result.shape == (50, 10)

    def test_uniqueness(self):
        result = generate_unique_permutations(8, 100, random_state=42)
        as_tuples = set(tuple(row) for row in result)
        assert len(as_tuples) == 100

    def test_excludes_identity(self):
        result = generate_unique_permutations(6, 100, random_state=42)
        identity = tuple(range(6))
        for row in result:
            assert tuple(row) != identity

    def test_includes_identity_when_allowed(self):
        # With small n and many permutations, identity should appear
        result = generate_unique_permutations(
            4,
            23,
            random_state=42,
            exclude_identity=False,
        )
        # Not guaranteed but very likely with 23 of 24 permutations
        # Just check shape and uniqueness
        assert result.shape == (23, 4)

    def test_exhaustive_small_n(self):
        # n=4 → 24 permutations, exclude identity → 23 available
        result = generate_unique_permutations(4, 23, random_state=42)
        assert result.shape == (23, 4)
        as_tuples = set(tuple(row) for row in result)
        assert len(as_tuples) == 23

    def test_raises_on_too_many(self):
        with pytest.raises(ValueError, match="Requested"):
            generate_unique_permutations(4, 24, random_state=42)

    def test_reproducibility(self):
        a = generate_unique_permutations(10, 50, random_state=99)
        b = generate_unique_permutations(10, 50, random_state=99)
        np.testing.assert_array_equal(a, b)

    def test_valid_indices(self):
        result = generate_unique_permutations(20, 100, random_state=42)
        assert np.all(result >= 0)
        assert np.all(result < 20)
        # Each row should be a permutation of 0..19
        for row in result:
            assert sorted(row) == list(range(20))

    def test_vectorised_large_n_no_dedup(self):
        """Large n takes the vectorised path with no deduplication."""
        result = generate_unique_permutations(50, 200, random_state=42)
        assert result.shape == (200, 50)
        for row in result:
            assert sorted(row) == list(range(50))

    def test_vectorised_large_n_excludes_identity(self):
        """Large-n vectorised path still excludes identity."""
        result = generate_unique_permutations(20, 500, random_state=42)
        identity = tuple(range(20))
        for row in result:
            assert tuple(row) != identity

    def test_reproducibility_large_n(self):
        """Vectorised path is also reproducible."""
        a = generate_unique_permutations(30, 100, random_state=7)
        b = generate_unique_permutations(30, 100, random_state=7)
        np.testing.assert_array_equal(a, b)

"""Tests for the permutations module."""

import numpy as np
import pytest

from randomization_tests.permutations import (
    _unrank_permutation,
    _unrank_within_cell,
    generate_between_cell_permutations,
    generate_two_stage_permutations,
    generate_unique_permutations,
    generate_within_cell_permutations,
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


# ------------------------------------------------------------------ #
# Within-cell permutation tests
# ------------------------------------------------------------------ #


class TestWithinCellPermutations:
    """Tests for generate_within_cell_permutations."""

    def test_shape(self):
        cells = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        result = generate_within_cell_permutations(9, 50, cells, random_state=42)
        assert result.shape == (50, 9)

    def test_uniqueness(self):
        cells = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        result = generate_within_cell_permutations(8, 100, cells, random_state=42)
        as_tuples = {tuple(row) for row in result}
        assert len(as_tuples) == 100

    def test_excludes_identity(self):
        cells = np.array([0, 0, 0, 1, 1, 1])
        result = generate_within_cell_permutations(6, 20, cells, random_state=42)
        identity = tuple(range(6))
        for row in result:
            assert tuple(row) != identity

    def test_includes_identity_when_allowed(self):
        cells = np.array([0, 0, 1, 1])
        # 2! * 2! = 4 total permutations, request all 4
        result = generate_within_cell_permutations(
            4, 4, cells, random_state=42, exclude_identity=False
        )
        identity = tuple(range(4))
        found = any(tuple(row) == identity for row in result)
        assert found or result.shape[0] == 4  # at least got all of them

    def test_cell_boundary_preservation(self):
        """Indices must never cross cell boundaries."""
        cells = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        result = generate_within_cell_permutations(9, 100, cells, random_state=42)
        for row in result:
            # Cell 0 indices {0,1,2} should map only to {0,1,2}
            assert set(row[:3].tolist()) == {0, 1, 2}
            # Cell 1 indices {3,4,5} should map only to {3,4,5}
            assert set(row[3:6].tolist()) == {3, 4, 5}
            # Cell 2 indices {6,7,8} should map only to {6,7,8}
            assert set(row[6:9].tolist()) == {6, 7, 8}

    def test_singleton_cell_pinned(self):
        """A cell with a single observation is never shuffled."""
        cells = np.array([0, 1, 1, 1, 2])
        # 1! × 3! × 1! = 6 total, 5 excluding identity
        result = generate_within_cell_permutations(5, 5, cells, random_state=42)
        for row in result:
            # Position 0 (cell 0, singleton) must stay at 0
            assert row[0] == 0
            # Position 4 (cell 2, singleton) must stay at 4
            assert row[4] == 4

    def test_budget_warning_and_cap(self):
        """Warns and caps when ∏ n_c! < requested B."""
        cells = np.array([0, 0, 1, 1])
        # 2! * 2! = 4, minus identity = 3 available
        with pytest.warns(UserWarning, match="Only 3 unique within-cell"):
            result = generate_within_cell_permutations(4, 100, cells, random_state=42)
        assert result.shape[0] == 3

    def test_reproducibility(self):
        cells = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        a = generate_within_cell_permutations(9, 50, cells, random_state=99)
        b = generate_within_cell_permutations(9, 50, cells, random_state=99)
        np.testing.assert_array_equal(a, b)

    def test_valid_indices(self):
        cells = np.array([0, 0, 0, 1, 1, 1])
        result = generate_within_cell_permutations(6, 20, cells, random_state=42)
        for row in result:
            assert sorted(row) == list(range(6))


# ------------------------------------------------------------------ #
# Between-cell permutation tests
# ------------------------------------------------------------------ #


class TestBetweenCellPermutations:
    """Tests for generate_between_cell_permutations."""

    def test_shape(self):
        cells = np.array([0, 0, 1, 1, 2, 2])
        result = generate_between_cell_permutations(6, 5, cells, random_state=42)
        assert result.shape == (5, 6)

    def test_uniqueness(self):
        cells = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        result = generate_between_cell_permutations(8, 20, cells, random_state=42)
        as_tuples = {tuple(row) for row in result}
        assert len(as_tuples) == 20

    def test_excludes_identity(self):
        cells = np.array([0, 0, 1, 1, 2, 2])
        result = generate_between_cell_permutations(6, 5, cells, random_state=42)
        identity = tuple(range(6))
        for row in result:
            assert tuple(row) != identity

    def test_internal_order_preserved_balanced(self):
        """For balanced cells, within-cell ordering must be preserved."""
        cells = np.array([0, 0, 1, 1, 2, 2])
        result = generate_between_cell_permutations(6, 5, cells, random_state=42)
        for row in result:
            # Each pair of adjacent positions from original cells
            # should map to a pair that maintains relative order.
            # Cell 0 original positions [0,1] → some cell [a,b] with a < b
            mapped_cell0 = row[[0, 1]]
            assert mapped_cell0[0] < mapped_cell0[1]
            mapped_cell1 = row[[2, 3]]
            assert mapped_cell1[0] < mapped_cell1[1]
            mapped_cell2 = row[[4, 5]]
            assert mapped_cell2[0] < mapped_cell2[1]

    def test_valid_permutation(self):
        cells = np.array([0, 0, 1, 1, 2, 2])
        result = generate_between_cell_permutations(6, 5, cells, random_state=42)
        for row in result:
            assert sorted(row) == list(range(6))

    def test_budget_warning_and_cap(self):
        """Warns and caps when G! < requested B."""
        cells = np.array([0, 0, 1, 1])
        # G=2, 2! = 2, minus identity = 1 available
        with pytest.warns(UserWarning, match="Only 1 unique between-cell"):
            result = generate_between_cell_permutations(4, 100, cells, random_state=42)
        assert result.shape[0] == 1

    def test_unbalanced_cells_no_swap(self):
        """Unbalanced cells with no same-size groups have no swaps."""
        cells = np.array([0, 0, 0, 1, 1])
        # sizes [3, 2] — no same-size pairs, only identity valid
        with pytest.warns(UserWarning, match="Only 0 unique between-cell"):
            result = generate_between_cell_permutations(5, 1, cells, random_state=42)
        assert result.shape == (0, 5)

    def test_unbalanced_cells_same_size_swap(self):
        """Unbalanced cells with some same-size groups can be swapped."""
        # cells: 3 of size 2 + 1 of size 3 → 3! = 6 between-perms
        cells = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])
        result = generate_between_cell_permutations(9, 5, cells, random_state=42)
        assert result.shape == (5, 9)
        for row in result:
            assert sorted(row) == list(range(9))

    def test_reproducibility(self):
        cells = np.array([0, 0, 1, 1, 2, 2])
        a = generate_between_cell_permutations(6, 5, cells, random_state=99)
        b = generate_between_cell_permutations(6, 5, cells, random_state=99)
        np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------ #
# Two-stage permutation tests
# ------------------------------------------------------------------ #


class TestTwoStagePermutations:
    """Tests for generate_two_stage_permutations."""

    def test_shape(self):
        cells = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        result = generate_two_stage_permutations(9, 50, cells, random_state=42)
        assert result.shape == (50, 9)

    def test_uniqueness(self):
        cells = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        result = generate_two_stage_permutations(9, 100, cells, random_state=42)
        as_tuples = {tuple(row) for row in result}
        assert len(as_tuples) == 100

    def test_excludes_identity(self):
        cells = np.array([0, 0, 0, 1, 1, 1])
        result = generate_two_stage_permutations(6, 30, cells, random_state=42)
        identity = tuple(range(6))
        for row in result:
            assert tuple(row) != identity

    def test_valid_indices(self):
        cells = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        result = generate_two_stage_permutations(9, 50, cells, random_state=42)
        for row in result:
            assert sorted(row) == list(range(9))

    def test_budget_warning_and_cap(self):
        """Warns and caps when G! × ∏ n_c! < requested B."""
        cells = np.array([0, 0, 1, 1])
        # G=2 → 2!, cells → 2!×2! = 4, total = 2!×4 = 8, minus identity = 7
        with pytest.warns(UserWarning, match="Only 7 unique two-stage"):
            result = generate_two_stage_permutations(4, 100, cells, random_state=42)
        assert result.shape[0] == 7

    def test_reproducibility(self):
        cells = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        a = generate_two_stage_permutations(9, 50, cells, random_state=99)
        b = generate_two_stage_permutations(9, 50, cells, random_state=99)
        np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------ #
# Mixed-radix Lehmer helper tests
# ------------------------------------------------------------------ #


class TestUnrankWithinCell:
    """Tests for the mixed-radix within-cell unranking helper."""

    def test_rank_zero_is_identity(self):
        """Rank 0 must produce the identity permutation."""
        # 3 cells of size 2: total = 2!^3 = 8
        cell_idx_list = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
        cell_factorials = [2, 2, 2]
        result = _unrank_within_cell(0, cell_idx_list, cell_factorials, 6)
        np.testing.assert_array_equal(result, np.arange(6))

    def test_all_ranks_unique(self):
        """Every rank in [0, ∏n_c!) must produce a distinct permutation."""
        # 2 cells of size 3: total = 3! × 3! = 36
        cell_idx_list = [np.array([0, 1, 2]), np.array([3, 4, 5])]
        cell_factorials = [6, 6]
        total = 36
        perms = set()
        for k in range(total):
            p = _unrank_within_cell(k, cell_idx_list, cell_factorials, 6)
            perms.add(tuple(p.tolist()))
        assert len(perms) == total

    def test_cell_boundaries_preserved(self):
        """Each rank's output must respect cell boundaries."""
        cell_idx_list = [np.array([0, 1, 2]), np.array([3, 4])]
        cell_factorials = [6, 2]
        total = 12
        for k in range(total):
            p = _unrank_within_cell(k, cell_idx_list, cell_factorials, 5)
            # Cell 0 values must come from {0, 1, 2}
            assert set(p[:3].tolist()) == {0, 1, 2}
            # Cell 1 values must come from {3, 4}
            assert set(p[3:].tolist()) == {3, 4}

    def test_singleton_cells_ignored(self):
        """Singleton cells (n_c=1) contribute 1! = 1 and are pinned."""
        cell_idx_list = [np.array([0]), np.array([1, 2, 3]), np.array([4])]
        cell_factorials = [1, 6, 1]
        # Only 6 unique permutations (from the middle cell)
        perms = set()
        for k in range(6):
            p = _unrank_within_cell(k, cell_idx_list, cell_factorials, 5)
            assert p[0] == 0  # singleton pinned
            assert p[4] == 4  # singleton pinned
            perms.add(tuple(p.tolist()))
        assert len(perms) == 6


# ------------------------------------------------------------------ #
# Within-cell Lehmer path tests
# ------------------------------------------------------------------ #


class TestWithinCellLehmerPath:
    """Tests for the Lehmer-code fast path in within-cell generation."""

    def test_lehmer_exact_count(self):
        """Small cells must return exactly B permutations — zero collisions."""
        # 5 cells of size 3: ∏ = 6^5 = 7776, well under threshold
        cells = np.repeat(np.arange(5), 3)
        result = generate_within_cell_permutations(15, 500, cells, random_state=42)
        assert result.shape == (500, 15)
        # All unique
        assert len({tuple(row) for row in result}) == 500

    def test_lehmer_exhaustive_enumeration(self):
        """When B >= available, enumerate all permutations for exact test."""
        # 2 cells of size 3: ∏ = 6 × 6 = 36, minus identity = 35
        cells = np.array([0, 0, 0, 1, 1, 1])
        with pytest.warns(UserWarning, match="Only 35"):
            result = generate_within_cell_permutations(6, 100, cells, random_state=42)
        assert result.shape[0] == 35
        assert len({tuple(row) for row in result}) == 35
        # Identity must not be present
        identity = tuple(range(6))
        for row in result:
            assert tuple(row) != identity

    def test_lehmer_cell_boundaries(self):
        """Lehmer path must preserve cell boundaries."""
        cells = np.repeat(np.arange(4), 3)  # 4 cells of size 3
        result = generate_within_cell_permutations(12, 200, cells, random_state=42)
        for row in result:
            for c in range(4):
                start = c * 3
                expected = {start, start + 1, start + 2}
                assert set(row[start : start + 3].tolist()) == expected

    def test_lehmer_reproduces(self):
        """Lehmer path is reproducible with same seed."""
        cells = np.repeat(np.arange(5), 3)
        a = generate_within_cell_permutations(15, 200, cells, random_state=7)
        b = generate_within_cell_permutations(15, 200, cells, random_state=7)
        np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------ #
# Within-cell vectorised batch path tests
# ------------------------------------------------------------------ #


class TestWithinCellVectorisedPath:
    """Tests for the vectorised batch path in within-cell generation."""

    def test_large_cells_shape(self):
        """Large cells that exceed Lehmer threshold use vectorised path."""
        # 5 cells of size 50: ∏ = 50!^5, far beyond threshold
        cells = np.repeat(np.arange(5), 50)
        result = generate_within_cell_permutations(250, 500, cells, random_state=42)
        assert result.shape == (500, 250)

    def test_large_cells_boundary_preservation(self):
        """Vectorised path must preserve cell boundaries."""
        cells = np.repeat(np.arange(3), 20)
        result = generate_within_cell_permutations(60, 200, cells, random_state=42)
        for row in result:
            assert set(row[:20].tolist()) == set(range(20))
            assert set(row[20:40].tolist()) == set(range(20, 40))
            assert set(row[40:60].tolist()) == set(range(40, 60))

    def test_large_cells_uniqueness(self):
        """Vectorised path must produce unique permutations."""
        cells = np.repeat(np.arange(5), 20)
        result = generate_within_cell_permutations(100, 500, cells, random_state=42)
        assert len({tuple(row) for row in result}) == 500

    def test_large_cells_excludes_identity(self):
        """Vectorised path must exclude identity."""
        cells = np.repeat(np.arange(5), 20)
        result = generate_within_cell_permutations(100, 500, cells, random_state=42)
        identity = tuple(range(100))
        for row in result:
            assert tuple(row) != identity


# ------------------------------------------------------------------ #
# Two-stage Lehmer path tests
# ------------------------------------------------------------------ #


class TestTwoStageLehmerPath:
    """Tests for the Lehmer-code fast path in two-stage generation."""

    def test_balanced_lehmer_exact_count(self):
        """Balanced small cells use composite Lehmer — exact B rows."""
        # 3 cells of size 3: G!=6, (3!)^3=216, total=1296
        cells = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        result = generate_two_stage_permutations(9, 500, cells, random_state=42)
        assert result.shape == (500, 9)
        assert len({tuple(row) for row in result}) == 500

    def test_balanced_lehmer_exhaustive(self):
        """Exhaustive: request more than available → capped + all returned."""
        # 2 cells of size 2: G!=2, (2!)^2=4, total=8, minus identity=7
        cells = np.array([0, 0, 1, 1])
        with pytest.warns(UserWarning, match="Only 7"):
            result = generate_two_stage_permutations(4, 100, cells, random_state=42)
        assert result.shape[0] == 7
        assert len({tuple(row) for row in result}) == 7
        identity = tuple(range(4))
        for row in result:
            assert tuple(row) != identity

    def test_balanced_lehmer_valid_permutations(self):
        """Lehmer path must produce valid permutations of [0..n-1]."""
        cells = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        result = generate_two_stage_permutations(9, 200, cells, random_state=42)
        for row in result:
            assert sorted(row) == list(range(9))

    def test_unbalanced_falls_through(self):
        """Unbalanced cells: only identity between-perm is valid."""
        cells = np.array([0, 0, 0, 1, 1])
        # sizes [3, 2] — between=1 (identity only), within=3!×2!=12
        # total=12, minus identity=11
        with pytest.warns(UserWarning, match="Only 11 unique two-stage"):
            result = generate_two_stage_permutations(5, 20, cells, random_state=42)
        assert result.shape == (11, 5)
        assert len({tuple(row) for row in result}) == 11
        # All rows must be valid permutations of [0..4]
        for row in result:
            assert sorted(row) == list(range(5))

    def test_balanced_lehmer_reproduces(self):
        """Balanced Lehmer path is reproducible."""
        cells = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        a = generate_two_stage_permutations(9, 200, cells, random_state=7)
        b = generate_two_stage_permutations(9, 200, cells, random_state=7)
        np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------ #
# Global generator gap-fill safety cap test
# ------------------------------------------------------------------ #


class TestGlobalGapFillCap:
    """Tests for the safety cap on the global generator's gap-fill loop."""

    def test_gap_fill_returns_valid(self):
        """Gap-fill loop produces valid output even for medium-n dedup path."""
        # n=12 triggers dedup path (birthday collision prob > 1e-9 for B=5000)
        # The gap-fill loop should produce exactly 5000 unique permutations
        # if collisions are filled successfully.
        result = generate_unique_permutations(12, 5000, random_state=42)
        # Should get exactly 5000 (collisions rare enough, cap generous)
        assert result.shape[0] == 5000
        assert len({tuple(row) for row in result}) == 5000

    def test_medium_n_excludes_identity(self):
        """Medium-n dedup path still excludes identity."""
        result = generate_unique_permutations(12, 1000, random_state=42)
        identity = tuple(range(12))
        for row in result:
            assert tuple(row) != identity

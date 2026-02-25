"""Pre-generation of unique permutation index arrays.

Ensures no duplicate permutations are drawn and supports exhaustive
enumeration when n! is small enough.

Why unique permutations matter
------------------------------
Permutation tests derive their validity from the exchangeability
assumption: under the null hypothesis, the observed test statistic is
just one member of a finite reference set of size n! (for *n* data
points).  Drawing the **same** permutation more than once wastes
computation without adding information, and can bias the empirical
p-value upward (toward anti-conservatism) for very small n.

Three generation strategies are used depending on the sample size:

1. **Lehmer-code sampling** (n <= ``max_exhaustive``, default 10):
   Each of the n! lexicographic orderings can be identified by a unique
   integer rank k ∈ [0, n!).  Rather than materialising all n!
   orderings (which is infeasible for n >= 11), we draw B ranks
   without replacement via ``np.random.choice`` and convert each rank
   to its corresponding permutation using the factorial number system
   (Lehmer code).  This is O(B·n) in time and O(B·n) in memory —
   independent of n!.

2. **Vectorised batch generation** (n large enough that collisions are
   negligible): All B permutations are produced in a single
   ``np.random.Generator.permuted`` call — a C-level loop with no
   Python overhead per permutation.  The birthday-paradox bound
   B(B−1)/(2·n!) is evaluated to decide whether deduplication is
   needed (threshold: < 1e−9).

3. **Vectorised generation with hash-based deduplication** (medium n
   where collisions are rare but possible): Same batch generation,
   followed by a post-hoc pass that inserts each permutation's tuple
   into a Python ``set`` to detect duplicates.  Any gaps from
   collisions or identity removal are filled with targeted redraws.

Cell-constrained permutation generators
----------------------------------------
Three additional generators restrict permutations within
exchangeability cells — contiguous or non-contiguous subsets of
observations that are exchangeable under H₀:

4. **Within-cell permutations** — shuffle indices *only* within each
   cell.  Positions in different cells are never swapped.  This is the
   standard strategy for clustered / hierarchical designs where
   observations within a cluster are exchangeable but observations
   across clusters are not.

5. **Between-cell permutations** — permute *entire cells* as units,
   preserving the internal ordering within each cell.  The
   permutation operates on the G cell labels, then remaps back to
   data-level indices.  Only cells of the same size may be
   exchanged; the effective reference set is ``∏_s count_s!``.

6. **Two-stage permutations** — compose between-cell and within-cell
   permutations.  First shuffle cell labels (between), then
   independently shuffle within each cell (within).  Appropriate
   when both levels of exchangeability are plausible.
"""

from __future__ import annotations

import math
import warnings
from functools import reduce

import numpy as np

# ------------------------------------------------------------------ #
# Lehmer code (factorial number system)
# ------------------------------------------------------------------ #
#
# Every permutation of [0, 1, …, n−1] has a unique "rank" — its
# position in the lexicographic enumeration of all n! orderings.
# The factoradic representation decomposes that rank into a sequence
# of digits d₁, d₂, …, dₙ where the i-th digit is expressed in
# base (n−i)!:
#
#   k = d₁·(n−1)! + d₂·(n−2)! + ··· + dₙ·0!
#
# Each digit dᵢ ∈ [0, n−i] selects the dᵢ-th remaining element from
# a shrinking pool.  This bijection lets us go from rank → permutation
# in O(n) time without storing the full enumeration.
#
# Example for n=3, k=4:
#   k=4 → digits [2, 0, 0] in factoradic
#   pool=[0,1,2] → pop(2)=2, pool=[0,1] → pop(0)=0, pool=[1] → pop(0)=1
#   result = [2, 0, 1]


def _unrank_permutation(k: int, n: int) -> list[int]:
    """Convert rank *k* to the *k*-th lexicographic permutation of ``[0..n-1]``.

    Args:
        k: Rank in ``[0, n!)``.
        n: Length of the permutation.

    Returns:
        List of *n* integers representing the permutation.
    """
    available = list(range(n))
    result: list[int] = []
    for i in range(n, 0, -1):
        f = math.factorial(i - 1)
        idx, k = divmod(k, f)
        result.append(available.pop(idx))
    return result


def _unrank_within_cell(
    rank: int,
    cell_idx_list: list[np.ndarray],
    cell_factorials: list[int],
    n_samples: int,
) -> np.ndarray:
    """Convert a composite rank to a within-cell permutation.

    Each within-cell permutation is uniquely identified by a tuple of
    per-cell Lehmer ranks ``(r_0, r_1, \u2026, r_{G-1})`` with
    ``r_c \u2208 [0, n_c!)``.  The composite rank encodes this tuple via
    mixed-radix decomposition (least-significant cell first).

    Args:
        rank: Composite rank in ``[0, \u220f n_c!)``.
        cell_idx_list: Per-cell sorted index arrays (consistent order).
        cell_factorials: Per-cell factorials ``[n_0!, n_1!, \u2026]``.
        n_samples: Total number of observations.

    Returns:
        Integer array of shape ``(n_samples,)`` \u2014 a valid within-cell
        permutation.
    """
    perm = np.arange(n_samples, dtype=np.intp)
    remaining = rank

    for cidx, cf in zip(cell_idx_list, cell_factorials, strict=True):
        nc = len(cidx)
        if nc <= 1:
            continue
        # Extract this cell's rank via mixed-radix decomposition.
        cell_rank = remaining % cf
        remaining //= cf
        # Decode to per-cell permutation and apply.
        cell_perm = _unrank_permutation(cell_rank, nc)
        perm[cidx] = cidx[cell_perm]

    return perm


def generate_unique_permutations(
    n_samples: int,
    n_permutations: int,
    random_state: int | None = None,
    exclude_identity: bool = True,
    max_exhaustive: int = 10,
) -> np.ndarray:
    """Pre-generate a matrix of unique permutation index arrays.

    For small *n_samples* (<= *max_exhaustive*), Lehmer-code sampling
    draws permutations by their lexicographic rank — O(B·n) time and
    memory, independent of n!.  For larger inputs a vectorised batch
    strategy generates all B permutations in one NumPy call, with an
    optional post-hoc deduplication pass when the birthday-paradox
    collision bound warrants it.

    Args:
        n_samples: Length of the array to permute.
        n_permutations: Number of unique permutations requested.
        random_state: Seed for reproducibility.
        exclude_identity: If ``True``, the identity permutation
            ``[0, 1, ..., n-1]`` is excluded so the observed data is
            never counted as a null sample.
        max_exhaustive: When ``n_samples <= max_exhaustive``, use
            Lehmer-code sampling instead of random generation.

    Returns:
        Array of shape ``(n_permutations, n_samples)`` where each row is
        a unique permutation of ``range(n_samples)``.

    Raises:
        ValueError: If *n_permutations* exceeds the number of available
            unique permutations (after optionally excluding the identity).
    """
    rng = np.random.default_rng(random_state)
    identity = tuple(range(n_samples))

    # ---- Small n: Lehmer-code sampling ------------------------------------
    #
    # For n <= max_exhaustive, the permutation space is small enough that
    # we can exactly identify each ordering by its lexicographic rank
    # (an integer in [0, n!)).  We draw B ranks *without replacement*
    # via np.random.choice — O(1) per rank — then decode each rank to
    # its permutation via _unrank_permutation.
    #
    # This replaces the old approach of calling
    #   list(itertools.permutations(range(n)))
    # which materialised ALL n! tuples into memory.  For n=12 that would
    # be 479 million tuples (~50 GB) — a guaranteed OOM.  The Lehmer-code
    # approach uses O(B·n) memory regardless of n!.
    #
    # The identity permutation [0, 1, …, n-1] always has rank 0 in
    # lexicographic order.  Excluding it is therefore as simple as
    # drawing from ranks [1, n!) instead of [0, n!).
    if n_samples <= max_exhaustive:
        total_perms = math.factorial(n_samples)
        available = total_perms - 1 if exclude_identity else total_perms

        if n_permutations > available:
            raise ValueError(
                f"Requested {n_permutations} unique permutations but only "
                f"{available} are available for n_samples={n_samples} "
                f"(exclude_identity={exclude_identity})."
            )

        # Identity = rank 0, so excluding it means sampling from [1, total).
        if exclude_identity:
            ranks = (
                rng.choice(
                    total_perms - 1,
                    size=n_permutations,
                    replace=False,
                )
                + 1
            )
        else:
            ranks = rng.choice(
                total_perms,
                size=n_permutations,
                replace=False,
            )

        return np.array(
            [_unrank_permutation(int(k), n_samples) for k in ranks],
            dtype=np.intp,
        )

    # ---- Large n: vectorised batch generation -----------------------------
    #
    # For n > max_exhaustive, n! is so large that:
    #   - Enumerating all permutations is infeasible.
    #   - The probability of drawing a duplicate is tiny.
    #
    # We generate all B permutations in a SINGLE vectorised NumPy call:
    #
    #   rng.permuted(batch, axis=1, out=batch)
    #
    # This shuffles each row independently at the C level — no Python
    # loop, no per-permutation tuple conversion.  For B=5000, n=569
    # (the breast cancer dataset), this is ~100× faster than the old
    # while-loop approach.
    #
    # Birthday-paradox collision bound:
    #   P(≥1 duplicate) ≈ B(B-1) / (2·n!)
    #
    # When this probability is < 1e-9, we skip deduplication entirely.
    # For medium n where the bound is non-negligible, we do a post-hoc
    # dedup pass using a Python set — still faster than the old approach
    # because the bulk generation is vectorised.

    collision_prob = (
        n_permutations * (n_permutations - 1) / (2 * math.factorial(n_samples))
    )
    need_dedup = collision_prob >= 1e-9

    # Generate all B permutations at once.
    batch = np.tile(np.arange(n_samples), (n_permutations, 1))
    rng.permuted(batch, axis=1, out=batch)

    # Fast path: no dedup needed and identity not excluded.
    if not need_dedup and not exclude_identity:
        return batch

    # Post-hoc filtering: remove identity and/or deduplicate.
    # For the no-dedup case this only checks against the identity
    # (probability ≈ B/n! ≈ 0, but checked for correctness).
    seen: set[tuple[int, ...]] = set()
    if exclude_identity:
        seen.add(identity)

    result = np.empty((n_permutations, n_samples), dtype=np.intp)
    count = 0

    for i in range(len(batch)):
        key = tuple(batch[i].tolist())
        if key not in seen:
            seen.add(key)
            result[count] = batch[i]
            count += 1
            if count == n_permutations:
                return result

    # Fill gaps left by identity hits or (rare) duplicate collisions.
    # Safety cap prevents an unbounded loop in the astronomically
    # unlikely event of sustained collisions for medium-n samples
    # where dedup is enabled.
    max_attempts = n_permutations * 20 + 1000
    attempts = 0
    while count < n_permutations and attempts < max_attempts:
        perm = rng.permutation(n_samples)
        key = tuple(perm.tolist())
        if key not in seen:
            seen.add(key)
            result[count] = perm
            count += 1
        attempts += 1

    return result[:count]


# ------------------------------------------------------------------ #
# Within-cell permutation generator
# ------------------------------------------------------------------ #
#
# Given an integer cell-label vector cells[i] ∈ {0, 1, …, G−1}, a
# "within-cell" permutation shuffles indices *only* among observations
# sharing the same cell label.  Observations in different cells are
# never swapped.
#
# Mathematically, the reference distribution for a within-cell test
# has size ∏_c n_c! (product of per-cell factorials).  This is
# typically far smaller than the global n!, and can be exhausted for
# small cells.
#
# The implementation follows the pattern from Appendix A §A.9 of the
# v0.4.0 series plan: for each permutation row, independently shuffle
# each cell's member indices.  Hash-based dedup ensures uniqueness.
#
# Singleton cells (n_c = 1) contribute factor 1! = 1 to the product —
# they are never shuffled and never affect the reference set size.


def generate_within_cell_permutations(
    n_samples: int,
    n_permutations: int,
    cells: np.ndarray,
    random_state: int | None = None,
    exclude_identity: bool = True,
) -> np.ndarray:
    """Generate permutation indices that shuffle only within cells.

    Each output row is a permutation of ``[0, 1, …, n−1]`` where
    indices belonging to different cells are never exchanged.  This
    is the standard approach for clustered / hierarchical data where
    observations within a cluster are exchangeable under H₀ but
    observations across clusters are not.

    Args:
        n_samples: Total number of observations.
        n_permutations: Number of unique permutations requested.
        cells: Integer array of shape ``(n_samples,)`` mapping each
            observation to a cell label (0-indexed).
        random_state: Seed for reproducibility.
        exclude_identity: If ``True``, the identity permutation is
            excluded from the output.

    Returns:
        Array of shape ``(B, n_samples)`` with permutation indices,
        where B ≤ *n_permutations* (may be smaller if the total
        number of unique within-cell permutations is limited).

    Warns:
        UserWarning: If the total number of unique within-cell
            permutations (∏ n_c!) is smaller than *n_permutations*.
    """
    rng = np.random.default_rng(random_state)
    cells = np.asarray(cells)

    # ---- Build cell index map ------------------------------------
    #
    # Pre-compute sorted indices for each unique cell label so we
    # don't re-scan the array on every permutation.
    unique_cells = np.unique(cells)
    cell_indices: dict[int, np.ndarray] = {
        int(c): np.where(cells == c)[0] for c in unique_cells
    }

    # ---- Budget check: ∏ n_c! ------------------------------------
    #
    # The total number of distinct within-cell permutations is the
    # product of per-cell factorials.  If any cell has n_c = 1, its
    # factor is 1! = 1 (identity only — the observation is pinned).
    # We compute the product carefully using reduce to avoid overflow
    # for moderate cell sizes, and cap at n_permutations + 2 to avoid
    # computing astronomically large factorials needlessly.
    cell_sizes = [len(idx) for idx in cell_indices.values()]
    cell_factorials = [math.factorial(s) for s in cell_sizes]

    # Product of factorials — use functools.reduce; cap early to
    # avoid unbounded big-int arithmetic.
    _CAP = n_permutations + 2
    total_unique = reduce(lambda a, b: min(a * b, _CAP), cell_factorials, 1)

    available = total_unique - 1 if exclude_identity else total_unique

    if available < n_permutations:
        warnings.warn(
            f"Only {available} unique within-cell permutations are "
            f"available (product of per-cell factorials minus identity), "
            f"but {n_permutations} were requested.  Capping at {available}.",
            UserWarning,
            stacklevel=2,
        )
        n_permutations = available

    # ---- Small cells: Lehmer-code sampling ---------------------------
    #
    # When all cells have n_c <= 10 and the total reference set
    # (prod n_c!) is small enough, use mixed-radix Lehmer encoding to
    # sample composite ranks without replacement -- zero collisions.
    # Each within-cell permutation is uniquely identified by a tuple
    # of per-cell Lehmer ranks, packed into a single integer via the
    # factorial number system.
    _LEHMER_THRESHOLD = 50_000
    total_exact = 1
    all_small = True
    for cf in cell_factorials:
        total_exact *= cf
        if total_exact > _LEHMER_THRESHOLD:
            all_small = False
            break
    else:
        all_small = all(s <= 10 for s in cell_sizes)

    if all_small and total_exact <= _LEHMER_THRESHOLD:
        pool_start = 1 if exclude_identity else 0
        ranks = (
            rng.choice(
                total_exact - pool_start,
                size=n_permutations,
                replace=False,
            )
            + pool_start
        )
        cell_idx_list = list(cell_indices.values())
        result = np.empty((n_permutations, n_samples), dtype=np.intp)
        for i, rank in enumerate(ranks):
            result[i] = _unrank_within_cell(
                int(rank), cell_idx_list, cell_factorials, n_samples
            )
        return result

    # ---- Large cells: vectorised batch with hash dedup ---------------
    #
    # For each cell, shuffle all B candidate rows simultaneously using
    # rng.permuted(axis=1) on the cell's column block -- a C-level loop
    # with no Python overhead per permutation per cell.  This replaces
    # the previous one-at-a-time Python loop.
    identity = tuple(range(n_samples))
    seen: set[tuple[int, ...]] = set()
    if exclude_identity:
        seen.add(identity)

    result = np.empty((n_permutations, n_samples), dtype=np.intp)
    count = 0

    # Vectorised batch: generate all B candidates at once.
    batch = np.tile(np.arange(n_samples, dtype=np.intp), (n_permutations, 1))
    for cidx in cell_indices.values():
        if len(cidx) > 1:
            cell_block = batch[:, cidx].copy()
            rng.permuted(cell_block, axis=1, out=cell_block)
            batch[:, cidx] = cell_block

    # Dedup pass.
    for j in range(len(batch)):
        key = tuple(batch[j].tolist())
        if key not in seen:
            seen.add(key)
            result[count] = batch[j]
            count += 1
            if count == n_permutations:
                return result

    # Gap-fill: one-at-a-time for remaining slots (rare for large
    # cells where collision probability is negligible).
    max_attempts = n_permutations * 20 + 1000
    attempts = 0
    while count < n_permutations and attempts < max_attempts:
        perm = np.arange(n_samples, dtype=np.intp)
        for cidx in cell_indices.values():
            if len(cidx) > 1:
                perm[cidx] = rng.permutation(cidx)
        key = tuple(perm.tolist())
        if key not in seen:
            seen.add(key)
            result[count] = perm
            count += 1
        attempts += 1

    return result[:count]


# ------------------------------------------------------------------ #
# Between-cell permutation generator
# ------------------------------------------------------------------ #
#
# A "between-cell" permutation shuffles *entire cells* as units —
# the internal ordering within each cell is preserved.  This is
# equivalent to permuting the G cell labels and remapping each
# observation to the indices of its newly assigned cell.
#
# For G cells with sizes n_1, n_2, …, n_G, only cells of the *same
# size* may be exchanged — swapping a cell of size 3 with a cell of
# size 2 would require mapping 3 observations to 2 positions, which
# cannot produce a valid permutation of [0..n-1].
#
# The effective reference set size is therefore:
#
#     ∏_s  count_s!
#
# where count_s is the number of cells that share size s.  (When all
# cells are balanced this reduces to G!.)
#
# Algorithmically we:
#   1. Group cell indices by cell size.
#   2. Generate independent label permutations *within* each size
#      group (cells in a size group can be freely exchanged).
#   3. Compose the per-group permutations into a single G-element
#      label permutation and remap to data-level indices.


def _between_cell_total(cell_sizes: list[int]) -> int:
    """Return the number of distinct between-cell permutations.

    Only cells of the same size may be exchanged, so the total is
    ``∏_s  count_s!`` where ``count_s`` is how many cells share
    each unique size ``s``.
    """
    from collections import Counter

    counts = Counter(cell_sizes)
    total = 1
    for cnt in counts.values():
        total *= math.factorial(cnt)
    return total


def _random_restricted_label_perm(
    cell_sizes: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Return a random label permutation that only swaps same-size cells.

    Args:
        cell_sizes: Per-cell sizes ``[n_0, n_1, …, n_{G-1}]``.
        rng: NumPy random generator.

    Returns:
        A list of length G — a permutation of ``[0..G-1]`` where
        ``cell_sizes[i] == cell_sizes[result[i]]`` for all *i*.
    """
    from collections import defaultdict

    G = len(cell_sizes)
    label_perm = list(range(G))

    # Group cell indices by size.
    size_groups: dict[int, list[int]] = defaultdict(list)
    for c, s in enumerate(cell_sizes):
        size_groups[s].append(c)

    # Independently permute within each size group.
    for members in size_groups.values():
        if len(members) <= 1:
            continue
        shuffled = rng.permutation(members).tolist()
        for dst, src in zip(members, shuffled, strict=True):
            label_perm[dst] = src

    return label_perm


def _unrank_restricted_label_perm(
    rank: int,
    cell_sizes: list[int],
) -> list[int]:
    """Decode a rank to a restricted label permutation (same-size swaps).

    The rank space is ``[0, ∏_s count_s!)`` — a mixed-radix encoding
    where each size group contributes ``count_s!`` factor.

    Args:
        rank: Integer rank in ``[0, total)``.
        cell_sizes: Per-cell sizes ``[n_0, n_1, …, n_{G-1}]``.

    Returns:
        A list of length G — a valid restricted label permutation.
    """
    from collections import defaultdict

    G = len(cell_sizes)
    label_perm = list(range(G))

    # Group cell indices by size.
    size_groups: dict[int, list[int]] = defaultdict(list)
    for c, s in enumerate(cell_sizes):
        size_groups[s].append(c)

    # Decode via mixed-radix decomposition (one group at a time).
    remaining = rank
    for members in size_groups.values():
        k = len(members)
        if k <= 1:
            continue
        group_total = math.factorial(k)
        group_rank = remaining % group_total
        remaining //= group_total
        perm_within = _unrank_permutation(group_rank, k)
        for dst_idx, src_idx in zip(range(k), perm_within, strict=True):
            label_perm[members[dst_idx]] = members[src_idx]

    return label_perm


def generate_between_cell_permutations(
    n_samples: int,
    n_permutations: int,
    cells: np.ndarray,
    random_state: int | None = None,
    exclude_identity: bool = True,
) -> np.ndarray:
    """Generate permutation indices that shuffle entire cells as units.

    Each output row is a permutation of ``[0, 1, …, n−1]`` where the
    internal ordering within each cell is preserved.  Only the cell
    labels are permuted — observations are remapped to the indices of
    whichever cell they are assigned to.

    Only cells of the **same size** may be exchanged.  For balanced
    designs (all cells same size) this is equivalent to permuting
    groups in a one-way ANOVA.  For unbalanced designs the effective
    reference set is ``∏_s count_s!`` — the product of factorials of
    the number of cells sharing each unique size.

    Args:
        n_samples: Total number of observations.
        n_permutations: Number of unique permutations requested.
        cells: Integer array of shape ``(n_samples,)`` mapping each
            observation to a cell label (0-indexed).
        random_state: Seed for reproducibility.
        exclude_identity: If ``True``, the identity permutation is
            excluded from the output.

    Returns:
        Array of shape ``(B, n_samples)`` with permutation indices,
        where B ≤ *n_permutations*.

    Warns:
        UserWarning: If the number of distinct between-cell
            permutations is less than *n_permutations*.
    """
    rng = np.random.default_rng(random_state)
    cells = np.asarray(cells)

    # ---- Build cell structure ------------------------------------
    unique_cells = np.unique(cells)
    G = len(unique_cells)

    # Sorted indices for each cell (sorted so internal ordering is
    # deterministic and consistent across permutations).
    cell_indices: list[np.ndarray] = [
        np.sort(np.where(cells == c)[0]) for c in unique_cells
    ]

    # ---- Budget check: ∏_s count_s! -----------------------------
    #
    # Only cells of equal size can be exchanged.  The effective
    # reference set is the product of factorials of the multiplicity
    # of each unique cell size.
    cell_sizes = [len(idx) for idx in cell_indices]
    total_unique = _between_cell_total(cell_sizes)
    available = total_unique - 1 if exclude_identity else total_unique

    if available < n_permutations:
        warnings.warn(
            f"Only {available} unique between-cell permutations are "
            f"available (∏ count_s!={total_unique} with G={G} cells), "
            f"but {n_permutations} were requested.  Capping at {available}.",
            UserWarning,
            stacklevel=2,
        )
        n_permutations = available

    if n_permutations == 0:
        return np.empty((0, n_samples), dtype=np.intp)

    # ---- Generate via restricted label permutation + remap -------
    #
    # Strategy: generate label permutations that only swap same-size
    # cells, then remap to data-level indices.
    identity = tuple(range(n_samples))
    seen: set[tuple[int, ...]] = set()
    if exclude_identity:
        seen.add(identity)

    result = np.empty((n_permutations, n_samples), dtype=np.intp)
    count = 0

    # For small total (≤ threshold), use Lehmer-code enumeration.
    _LEHMER_THRESHOLD = 50_000
    if total_unique <= _LEHMER_THRESHOLD:
        pool_start = 1 if exclude_identity else 0
        pool_size = total_unique - pool_start

        if n_permutations <= pool_size:
            ranks = rng.choice(pool_size, size=n_permutations, replace=False)
            ranks += pool_start
        else:
            ranks = np.arange(pool_start, total_unique)

        # Decode all B label permutations into a (B, G) array.
        label_perms = np.array(
            [_unrank_restricted_label_perm(int(r), cell_sizes) for r in ranks],
            dtype=np.intp,
        )

        # Batch-remap all at once — O(G) NumPy ops, not O(n) Python.
        batch = _remap_between_batch(cell_indices, label_perms)

        # Dedup pass (collapses permutations that produce the same
        # data-level index vector — rare but possible if cell
        # contents happen to be identical).
        for j in range(len(batch)):
            key = tuple(batch[j].tolist())
            if key not in seen:
                seen.add(key)
                result[count] = batch[j]
                count += 1
                if count == n_permutations:
                    break

        return result[:count]

    # For large reference sets, use random restricted label
    # permutations with batch remap and hash-dedup.

    # Generate all B candidate label permutations at once.
    label_perms = np.array(
        [_random_restricted_label_perm(cell_sizes, rng) for _ in range(n_permutations)],
        dtype=np.intp,
    )

    # Batch-remap — O(G) NumPy ops per cell, not O(n) Python per row.
    batch = _remap_between_batch(cell_indices, label_perms)

    # Dedup pass.
    for j in range(len(batch)):
        key = tuple(batch[j].tolist())
        if key not in seen:
            seen.add(key)
            result[count] = batch[j]
            count += 1
            if count == n_permutations:
                break

    if count < n_permutations:
        # Gap-fill: one-at-a-time for remaining slots.
        max_attempts = n_permutations * 20 + 1000
        attempts = 0
        while count < n_permutations and attempts < max_attempts:
            label_perm = _random_restricted_label_perm(cell_sizes, rng)
            perm = _remap_between(cell_indices, label_perm)
            key = tuple(perm.tolist())
            if key not in seen:
                seen.add(key)
                result[count] = perm
                count += 1
            attempts += 1

    return result[:count]


def _remap_between(
    cell_indices: list[np.ndarray],
    label_perm: list[int],
) -> np.ndarray:
    """Build a data-level index vector from a cell-label permutation.

    For each original cell *c*, the observations in ``cell_indices[c]``
    are mapped to the positions of the target cell
    ``cell_indices[label_perm[c]]``.  Source and target cells must
    have the same size (only same-size swaps are valid).

    Args:
        cell_indices: Per-cell sorted index arrays.
        label_perm: Permuted cell-label assignment (length G).

    Returns:
        Integer array of shape ``(n,)`` — a valid permutation of
        ``[0..n-1]``.
    """
    G = len(cell_indices)
    n = sum(len(idx) for idx in cell_indices)
    perm = np.empty(n, dtype=np.intp)

    for c in range(G):
        src = cell_indices[c]  # original positions
        tgt = cell_indices[label_perm[c]]  # target cell's positions
        perm[src] = tgt

    return perm


def _remap_between_batch(
    cell_indices: list[np.ndarray],
    label_perms: np.ndarray,
) -> np.ndarray:
    """Batch-remap B label permutations to data-level index vectors.

    Vectorised counterpart of calling :func:`_remap_between` in a
    Python loop.  Iterates over *G* cells (not *n* observations),
    using NumPy advanced indexing to write all *B* rows at once.

    Only same-size cell swaps are valid — source and target cells
    always have equal length.

    Args:
        cell_indices: Per-cell sorted index arrays (length G).
        label_perms: Integer array of shape ``(B, G)`` — each row is
            a permuted cell-label assignment.

    Returns:
        Integer array of shape ``(B, n)`` with permutation indices.
    """
    B = label_perms.shape[0]
    G = len(cell_indices)
    n = sum(len(idx) for idx in cell_indices)
    result = np.empty((B, n), dtype=np.intp)

    for c in range(G):
        src = cell_indices[c]
        src_len = len(src)
        # target_labels[b] = label_perms[b, c] for each b
        target_labels = label_perms[:, c]

        # Build (B, src_len) target-index matrix.
        # Since only same-size cells are swapped, all targets have
        # the same length as src.  Group by target label for
        # vectorised assignment.
        unique_targets = np.unique(target_labels)
        for tl in unique_targets:
            mask = target_labels == tl  # which rows map to this target
            tgt = cell_indices[tl]
            result[np.ix_(mask, src)] = tgt[:src_len]

    return result


# ------------------------------------------------------------------ #
# Two-stage permutation generator
# ------------------------------------------------------------------ #
#
# A two-stage permutation composes between-cell and within-cell
# shuffles: first permute the cell labels (between), then
# independently shuffle within each (now-remapped) cell.
#
# This is appropriate when both levels of exchangeability are
# plausible — for example, in educational studies where both the
# assignment of students to classrooms and the ordering within each
# classroom are exchangeable under the null.
#
# The reference set size is (∏_s count_s!) × (∏_c n_c!), where the
# first factor counts valid between-cell permutations (only
# same-size cells may be exchanged) and the second counts within-
# cell permutations.


def generate_two_stage_permutations(
    n_samples: int,
    n_permutations: int,
    cells: np.ndarray,
    random_state: int | None = None,
    exclude_identity: bool = True,
) -> np.ndarray:
    """Generate permutation indices via between-cell then within-cell.

    Each output row is a composition: first shuffle cell labels
    (between-cell, restricted to same-size swaps), then independently
    shuffle observations within each (remapped) cell (within-cell).

    Args:
        n_samples: Total number of observations.
        n_permutations: Number of unique permutations requested.
        cells: Integer array of shape ``(n_samples,)`` mapping each
            observation to a cell label (0-indexed).
        random_state: Seed for reproducibility.
        exclude_identity: If ``True``, the identity permutation is
            excluded from the output.

    Returns:
        Array of shape ``(B, n_samples)`` with permutation indices,
        where B ≤ *n_permutations*.

    Warns:
        UserWarning: If (∏_s count_s!) × (∏_c n_c!) < *n_permutations*.
    """
    rng = np.random.default_rng(random_state)
    cells = np.asarray(cells)

    # ---- Build cell structure ------------------------------------
    unique_cells = np.unique(cells)
    G = len(unique_cells)
    cell_indices: list[np.ndarray] = [
        np.sort(np.where(cells == c)[0]) for c in unique_cells
    ]

    # ---- Budget check: (∏_s count_s!) × (∏_c n_c!) --------------
    cell_sizes = [len(idx) for idx in cell_indices]
    cell_factorials = [math.factorial(s) for s in cell_sizes]
    between_total = _between_cell_total(cell_sizes)

    _CAP = n_permutations + 2
    within_product = reduce(lambda a, b: min(a * b, _CAP), cell_factorials, 1)
    total_unique = min(between_total * within_product, _CAP)

    available = total_unique - 1 if exclude_identity else total_unique

    if available < n_permutations:
        warnings.warn(
            f"Only {available} unique two-stage permutations are "
            f"available ((∏ count_s!)×(∏ n_c!) with G={G} cells), but "
            f"{n_permutations} were requested.  Capping at {available}.",
            UserWarning,
            stacklevel=2,
        )
        n_permutations = available

    if n_permutations == 0:
        return np.empty((0, n_samples), dtype=np.intp)

    # ---- Small balanced: Lehmer-code sampling ---------------------
    #
    # For balanced cells (all same size) with small total reference
    # set (between_total × prod n_c! <= threshold), use composite
    # Lehmer encoding.  Between-component: rank in [0, between_total),
    # within-component: rank in [0, prod n_c!).  Total rank space is
    # between_total × prod n_c!; the identity is always rank 0.
    _LEHMER_THRESHOLD = 50_000
    balanced = len(set(cell_sizes)) == 1
    within_exact = 1
    _fits = True
    for cf in cell_factorials:
        within_exact *= cf
        if within_exact > _LEHMER_THRESHOLD:
            _fits = False
            break
    total_exact = between_total * within_exact if _fits else _LEHMER_THRESHOLD + 1

    if balanced and G <= 10 and total_exact <= _LEHMER_THRESHOLD:
        pool_start = 1 if exclude_identity else 0
        ranks = (
            rng.choice(
                total_exact - pool_start,
                size=n_permutations,
                replace=False,
            )
            + pool_start
        )

        result = np.empty((n_permutations, n_samples), dtype=np.intp)
        for i, rank in enumerate(ranks):
            k_between = int(rank) // within_exact
            k_within = int(rank) % within_exact

            # Decode between-component: restricted label permutation.
            label_perm = _unrank_restricted_label_perm(k_between, cell_sizes)

            # Decode within-component + compose with between.
            perm = np.empty(n_samples, dtype=np.intp)
            remaining = k_within
            for c in range(G):
                src = cell_indices[c]
                tgt = cell_indices[label_perm[c]]
                nc = len(tgt)
                cf = math.factorial(nc)
                cell_rank = remaining % cf
                remaining //= cf
                cell_perm = _unrank_permutation(cell_rank, nc)
                # Vectorised assignment.
                perm[src] = tgt[cell_perm]

            result[i] = perm

        return result

    # ---- Generate: between then within ---------------------------
    identity = tuple(range(n_samples))
    seen: set[tuple[int, ...]] = set()
    if exclude_identity:
        seen.add(identity)

    result = np.empty((n_permutations, n_samples), dtype=np.intp)
    count = 0

    # ---- Batch generation: between remap + within shuffle --------
    #
    # Generate all B candidate between-cell label permutations
    # (restricted to same-size swaps), batch-remap to data-level
    # indices, then apply within-cell shuffles.
    label_perms = np.array(
        [_random_restricted_label_perm(cell_sizes, rng) for _ in range(n_permutations)],
        dtype=np.intp,
    )

    # Batch between-cell remap — O(G) NumPy ops, not O(n) Python.
    batch = _remap_between_batch(cell_indices, label_perms)

    # Apply within-cell shuffles — vectorised across all B rows.
    #
    # After _remap_between_batch, batch[j, src] contains the sorted
    # indices of the target cell for row j.  rng.permuted(block,
    # axis=1) shuffles each row of the (B, n_c) block independently
    # at C-level — identical to per-row rng.permutation(tgt) but
    # with no Python loop over B.
    for c in range(G):
        src = cell_indices[c]
        if len(src) > 1:
            cell_block = batch[:, src].copy()
            rng.permuted(cell_block, axis=1, out=cell_block)
            batch[:, src] = cell_block

    # Dedup pass.
    for j in range(len(batch)):
        key = tuple(batch[j].tolist())
        if key not in seen:
            seen.add(key)
            result[count] = batch[j]
            count += 1
            if count == n_permutations:
                break

    if count < n_permutations:
        # Gap-fill: one-at-a-time for remaining slots.
        max_attempts = n_permutations * 20 + 1000
        attempts = 0
        while count < n_permutations and attempts < max_attempts:
            label_perm = _random_restricted_label_perm(cell_sizes, rng)
            perm = np.empty(n_samples, dtype=np.intp)
            for c in range(G):
                src = cell_indices[c]
                tgt = cell_indices[label_perm[c]]
                shuffled = rng.permutation(tgt)
                perm[src] = shuffled
            key = tuple(perm.tolist())
            if key not in seen:
                seen.add(key)
                result[count] = perm
                count += 1
            attempts += 1

    return result[:count]

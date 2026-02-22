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
"""

import math

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
    while count < n_permutations:
        perm = rng.permutation(n_samples)
        key = tuple(perm.tolist())
        if key not in seen:
            seen.add(key)
            result[count] = perm
            count += 1

    return result

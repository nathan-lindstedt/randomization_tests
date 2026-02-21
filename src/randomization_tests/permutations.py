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

Two strategies are used depending on the sample size:

1. **Exhaustive enumeration** (n <= ``max_exhaustive``, default 12):
   Materialise all n! orderings via ``itertools.permutations`` and draw
   a random subset without replacement.  12! = 479 001 600, which fits
   comfortably in memory as a list of tuples.

2. **Hash-based sampling** (n > 12): Each random permutation is
   converted to a tuple and inserted into a Python ``set``.  A new
   draw is rejected if it collides with an existing member.  Because
   n! grows super-exponentially, the collision probability for typical
   sample sizes (n >= 20, B = 5 000) is negligible — on the order of
   B² / (2 * n!), which is < 1e-30 for n = 20.
"""

import itertools
import math
import warnings

import numpy as np


def generate_unique_permutations(
    n_samples: int,
    n_permutations: int,
    random_state: int | None = None,
    exclude_identity: bool = True,
    max_exhaustive: int = 12,
) -> np.ndarray:
    """Pre-generate a matrix of unique permutation index arrays.

    For small *n_samples* (<= *max_exhaustive*), all n! permutations are
    enumerated and a random subset is drawn.  For larger inputs a
    hash-based sampling strategy avoids duplicates.

    Args:
        n_samples: Length of the array to permute.
        n_permutations: Number of unique permutations requested.
        random_state: Seed for reproducibility.
        exclude_identity: If ``True``, the identity permutation
            ``[0, 1, ..., n-1]`` is excluded so the observed data is
            never counted as a null sample.
        max_exhaustive: When ``n_samples <= max_exhaustive``, enumerate
            all n! permutations exhaustively instead of sampling.

    Returns:
        Array of shape ``(n_permutations, n_samples)`` where each row is
        a unique permutation of ``range(n_samples)``.

    Raises:
        ValueError: If *n_permutations* exceeds the number of available
            unique permutations (after optionally excluding the identity).
    """
    rng = np.random.default_rng(random_state)

    # For large n_samples, n! is astronomically large — e.g. 20! ≈ 2.4e18.
    # Computing math.factorial(n_samples) is itself O(n) in arbitrary-
    # precision integers, and the probability of drawing a duplicate in B
    # trials is vanishingly small (birthday-paradox bound: B²/(2·n!)).
    # We therefore skip all factorial-based bookkeeping and go straight to
    # the hash-based sampling path where collisions are essentially impossible.
    if n_samples <= max_exhaustive:
        total_perms = math.factorial(n_samples)
        # If we exclude the identity permutation [0, 1, …, n-1], one
        # ordering is removed from the candidate pool.  The identity is
        # excluded by default because it corresponds to the *observed* data
        # arrangement — including it would count the observed statistic
        # twice in the reference distribution.
        available = total_perms - 1 if exclude_identity else total_perms

        if n_permutations > available:
            raise ValueError(
                f"Requested {n_permutations} unique permutations but only "
                f"{available} are available for n_samples={n_samples} "
                f"(exclude_identity={exclude_identity})."
            )

        if n_permutations > available * 0.5:
            warnings.warn(
                f"Requesting {n_permutations} of {available} available "
                f"permutations ({n_permutations / available:.0%}). This may "
                f"be slow for large n_samples due to collision avoidance.",
                stacklevel=2,
            )

    identity = tuple(range(n_samples))

    # --- Exhaustive path ---------------------------------------------------
    # Materialise every possible ordering of (0, 1, …, n-1) via the
    # standard lexicographic generator.  For n=12 this produces 479 001 600
    # tuples — large but tractable.  We then sample `n_permutations` of
    # them without replacement, giving an exact draw from the combinatorial
    # reference set.
    if n_samples <= max_exhaustive:
        all_perms = list(itertools.permutations(range(n_samples)))
        if exclude_identity:
            all_perms = [p for p in all_perms if p != identity]
        chosen_idx = rng.choice(len(all_perms), size=n_permutations, replace=False)
        return np.array([all_perms[i] for i in chosen_idx], dtype=np.intp)

    # --- Sampling path with hash-based deduplication -----------------------
    # For n > max_exhaustive the full permutation space is far too large to
    # enumerate.  Instead, we draw random permutations one at a time and
    # store each as a hashable tuple in a set.  If a draw collides with an
    # existing member it is simply discarded and redrawn.
    #
    # The expected number of collisions before collecting B unique samples
    # from an n!-sized universe is approximately B*(B-1)/(2*n!) — the
    # birthday-paradox bound.  For n=20 and B=5000, this is ~5e-12, so in
    # practice every draw succeeds on the first attempt.
    seen: set = set()
    if exclude_identity:
        seen.add(identity)

    result = np.empty((n_permutations, n_samples), dtype=np.intp)
    count = 0

    while count < n_permutations:
        perm = tuple(rng.permutation(n_samples).tolist())
        if perm not in seen:
            seen.add(perm)
            result[count] = perm
            count += 1

    return result

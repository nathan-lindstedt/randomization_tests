"""Pre-generation of unique permutation index arrays.

Ensures no duplicate permutations are drawn and supports exhaustive
enumeration when n! is small enough.
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

    # For large n_samples, n! is astronomically large — skip the
    # factorial-based checks entirely and go straight to hash-based
    # sampling where collisions are essentially impossible.
    if n_samples <= max_exhaustive:
        total_perms = math.factorial(n_samples)
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

    # --- Exhaustive path ---
    if n_samples <= max_exhaustive:
        all_perms = list(itertools.permutations(range(n_samples)))
        if exclude_identity:
            all_perms = [p for p in all_perms if p != identity]
        chosen_idx = rng.choice(len(all_perms), size=n_permutations, replace=False)
        return np.array([all_perms[i] for i in chosen_idx], dtype=np.intp)

    # --- Sampling path with hash-based deduplication ---
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

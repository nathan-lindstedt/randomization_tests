"""Pre-generation of unique sign-flip arrays.

Sign-flipping is the randomization analogue of permutation for designs
where residuals are symmetric about zero under the null.  Instead of
permuting the residual vector, each residual is independently
multiplied by a Rademacher variable (+1 or −1).

Why sign-flips matter
---------------------
Permutation tests rely on **exchangeability**: under H₀, any
reordering of the residual vector is equally likely.  Sign-flip tests
rely on the strictly weaker assumption of **symmetry**: under H₀, the
distribution of each residual is symmetric about zero, so flipping its
sign is equally likely.

The reference set has size 2ⁿ rather than n!, and the generating
mechanism is element-wise multiplication rather than reordering.

Three generation strategies are used depending on the sample size:

1. **Exhaustive enumeration** (n ≤ ``max_exhaustive``, default 20):
   All 2ⁿ sign-flip vectors are enumerated via binary counting.
   When 2ⁿ > ``n_flips``, a random subset is drawn without
   replacement.

2. **Random sampling with negligible collision probability**
   (n large enough that 2ⁿ >> n_flips²):  All B vectors are drawn
   independently and collision risk is below 1e−9.

3. **Random sampling with hash-based deduplication** (medium n):
   Same random draw, followed by a post-hoc uniqueness pass using
   Python ``set`` on byte representations.

The ``validate_symmetry`` function provides a diagnostic check
via the Wilcoxon signed-rank test to help users assess whether the
symmetry assumption is plausible for their data.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

# ------------------------------------------------------------------ #
# Sign-flip generation
# ------------------------------------------------------------------ #


def generate_sign_flips(
    n_samples: int,
    n_flips: int = 5_000,
    random_state: int | None = None,
    exclude_identity: bool = True,
    max_exhaustive: int = 20,
    cells: np.ndarray | None = None,
    strategy: str | None = None,
    tree: ExchangeabilityTree | None = None,
) -> np.ndarray:
    """Generate unique sign-flip vectors.

    Returns a matrix of shape ``(B, n_samples)`` where every entry
    is +1 or −1.  The matrix contains no duplicate rows.

    When *tree* is given, delegates to
    :func:`generate_nested_sign_flips` for hierarchical
    exchangeability constraints.  When *cells* and *strategy* are
    given, dispatches to the appropriate flat block-constrained
    generator.

    Args:
        n_samples: Number of observations (length of each sign vector).
        n_flips: Desired number of sign-flip vectors.
        random_state: Seed for reproducibility.
        exclude_identity: If ``True`` (default), the all-+1 vector
            (identity flip) is excluded from the output because it
            reproduces the observed statistic exactly.
        max_exhaustive: Maximum n for exhaustive 2ⁿ enumeration.
            For n ≤ this value and when 2ⁿ − 1 ≤ n_flips (after
            identity exclusion), all possible sign-flip vectors are
            returned.
        cells: Optional integer array of shape ``(n_samples,)``
            mapping observations to cell labels (0-indexed).
        strategy: Cell-level strategy: ``"within"``, ``"between"``,
            or ``"two-stage"``.  Only used when *cells* is given.
        tree: An :class:`ExchangeabilityTree` for multi-level nested
            constraints.  Takes precedence over *cells*/*strategy*.

    Returns:
        Array of shape ``(B, n_samples)`` with entries in {−1, +1}.

    Raises:
        ValueError: If ``n_samples < 1`` or ``n_flips < 1``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    if n_flips < 1:
        raise ValueError(f"n_flips must be >= 1, got {n_flips}.")

    # ---- Hierarchical dispatch -----------------------------------
    if tree is not None:
        return generate_nested_sign_flips(
            n_samples, n_flips, tree, random_state, exclude_identity
        )

    # ---- Flat cell-constrained dispatch --------------------------
    if cells is not None:
        resolved = strategy or "within"
        if resolved == "between":
            return generate_between_cell_sign_flips(
                n_samples, n_flips, cells, random_state, exclude_identity
            )
        if resolved == "within":
            return generate_within_cell_sign_flips(
                n_samples,
                n_flips,
                cells,
                random_state,
                exclude_identity,
                max_exhaustive,
            )
        if resolved == "two-stage":
            return generate_two_stage_sign_flips(
                n_samples,
                n_flips,
                cells,
                random_state,
                exclude_identity,
                max_exhaustive,
            )

    # ---- Global (unconstrained) path -----------------------------

    rng = np.random.default_rng(random_state)

    # Total number of possible sign-flip vectors.
    total = 1 << n_samples  # 2^n
    available = total - 1 if exclude_identity else total

    if available < 1:
        raise ValueError(
            f"No sign-flip vectors available for n_samples={n_samples} "
            f"with exclude_identity={exclude_identity}."
        )

    # Clamp requested count to what is available.
    if n_flips > available:
        warnings.warn(
            f"Requested {n_flips} sign-flips but only {available} unique "
            f"vectors exist for n_samples={n_samples} "
            f"(exclude_identity={exclude_identity}).  "
            f"Returning all {available} vectors.",
            UserWarning,
            stacklevel=2,
        )
        n_flips = available

    # ---- Exhaustive path -----------------------------------------
    if n_samples <= max_exhaustive and available <= n_flips:
        return _exhaustive_sign_flips(n_samples, exclude_identity)

    # ---- Lehmer-style sampling for small n -----------------------
    # When n is small enough for exact enumeration but we only want
    # a subset, sample ranks without replacement.
    if n_samples <= max_exhaustive:
        return _sample_from_exhaustive(n_samples, n_flips, rng, exclude_identity)

    # ---- Random sampling (large n) -------------------------------
    return _random_sign_flips(n_samples, n_flips, rng, exclude_identity)


# ------------------------------------------------------------------ #
# Internal generators
# ------------------------------------------------------------------ #


def _exhaustive_sign_flips(
    n_samples: int,
    exclude_identity: bool,
) -> np.ndarray:
    """Enumerate all 2ⁿ sign-flip vectors via binary decomposition.

    Each integer k ∈ [0, 2ⁿ) maps to a sign vector:
    bit i of k → sign[i] = +1 if bit is 0, −1 if bit is 1.
    """
    total = 1 << n_samples
    ranks = np.arange(total, dtype=np.int64)

    if exclude_identity:
        # The all-+1 vector corresponds to rank 0 (all bits zero).
        ranks = ranks[1:]  # type: ignore[assignment]

    return _ranks_to_signs(ranks, n_samples)


def _sample_from_exhaustive(
    n_samples: int,
    n_flips: int,
    rng: np.random.Generator,
    exclude_identity: bool,
) -> np.ndarray:
    """Sample a subset of sign-flip vectors without replacement.

    Used when n is small enough to enumerate but we want fewer than
    2ⁿ vectors.
    """
    total = 1 << n_samples

    if exclude_identity:
        # Sample from [1, 2^n) — rank 0 is the identity.
        ranks = rng.choice(total - 1, size=n_flips, replace=False) + 1
    else:
        ranks = rng.choice(total, size=n_flips, replace=False)

    return _ranks_to_signs(ranks, n_samples)


def _ranks_to_signs(ranks: np.ndarray, n_samples: int) -> np.ndarray:
    """Convert integer ranks to ±1 sign vectors via bit decomposition.

    Bit i of rank k → sign[i]: 0 → +1, 1 → −1.
    """
    # Bit positions [0, 1, …, n−1] as column vector for broadcasting.
    bits = np.arange(n_samples, dtype=np.int64)
    # Extract bit i from each rank: (ranks[:, None] >> bits) & 1
    # gives a (B, n) binary matrix.
    binary = (ranks[:, np.newaxis] >> bits[np.newaxis, :]) & 1
    # Map 0 → +1, 1 → −1.
    return 1 - 2 * binary.astype(np.int8)  # type: ignore[no-any-return]


def _random_sign_flips(
    n_samples: int,
    n_flips: int,
    rng: np.random.Generator,
    exclude_identity: bool,
) -> np.ndarray:
    """Generate random sign-flip vectors with hash-based dedup.

    For large n, the birthday-paradox collision bound
    B(B−1) / (2 · 2ⁿ) is extremely small, so deduplication is
    rarely triggered.
    """
    # Initial batch — draw all at once.
    signs = rng.choice(
        np.array([-1, 1], dtype=np.int8),
        size=(n_flips, n_samples),
    )

    # Remove identity (all +1) rows if requested.
    if exclude_identity:
        identity_mask = np.all(signs == 1, axis=1)
        signs = signs[~identity_mask]

    # Hash-based deduplication.
    seen: set[bytes] = set()
    unique_rows: list[int] = []
    for i in range(signs.shape[0]):
        key = signs[i].tobytes()
        if key not in seen:
            seen.add(key)
            unique_rows.append(i)

    signs = signs[unique_rows]

    # Back-fill if we lost rows to collisions or identity removal.
    max_attempts = 50
    attempt = 0
    while signs.shape[0] < n_flips and attempt < max_attempts:
        attempt += 1
        gap = n_flips - signs.shape[0]
        extra = rng.choice(
            np.array([-1, 1], dtype=np.int8),
            size=(gap * 2, n_samples),
        )
        if exclude_identity:
            identity_mask = np.all(extra == 1, axis=1)
            extra = extra[~identity_mask]

        for i in range(extra.shape[0]):
            key = extra[i].tobytes()
            if key not in seen:
                seen.add(key)
                signs = np.vstack([signs, extra[i : i + 1]])
                if signs.shape[0] >= n_flips:
                    break

    return signs[:n_flips]


# ------------------------------------------------------------------ #
# Flat block-constrained sign-flip generators
# ------------------------------------------------------------------ #


def generate_within_cell_sign_flips(
    n_samples: int,
    n_flips: int,
    cells: np.ndarray,
    random_state: int | None = None,
    exclude_identity: bool = True,
    max_exhaustive: int = 20,
) -> np.ndarray:
    """Generate sign-flip vectors that flip independently within cells.

    Provided for API symmetry with
    :func:`~randomization_tests.permutations.generate_within_cell_permutations`.

    .. note::

        Within-cell sign-flip is **statistically equivalent** to
        unconstrained sign-flip.  Each observation receives an
        independent Rademacher variable regardless of cell membership,
        because constraining signs "within" a cell still assigns
        independent ±1 to every observation — the cell boundary has
        no effect on an element-wise operation.  This function
        therefore delegates directly to :func:`generate_sign_flips`
        (ignoring *cells*).

    Args:
        n_samples: Total number of observations.
        n_flips: Desired number of sign-flip vectors.
        cells: Integer array of shape ``(n_samples,)`` mapping each
            observation to a cell label (present for API consistency,
            but **ignored** since the result is identical to the
            unconstrained case).
        random_state: Seed for reproducibility.
        exclude_identity: If ``True``, the all-+1 vector is excluded.
        max_exhaustive: Passed through to :func:`generate_sign_flips`.

    Returns:
        Array of shape ``(B, n_samples)`` with entries in {−1, +1}.
    """
    # Within-cell sign-flip == unconstrained (Rademacher × ±1 = Rademacher).
    return generate_sign_flips(
        n_samples,
        n_flips,
        random_state=random_state,
        exclude_identity=exclude_identity,
        max_exhaustive=max_exhaustive,
    )


def generate_between_cell_sign_flips(
    n_samples: int,
    n_flips: int,
    cells: np.ndarray,
    random_state: int | None = None,
    exclude_identity: bool = True,
) -> np.ndarray:
    """Generate sign-flip vectors that assign one sign per cell.

    Every observation in a cell receives the **same** ±1 sign.
    The reference set has size 2^G (G = number of cells).

    This is the sign-flip analogue of
    :func:`~randomization_tests.permutations.generate_between_cell_permutations`:
    it tests whether the *direction* of the group-level effect
    differs from zero, while holding within-group structure fixed.

    Args:
        n_samples: Total number of observations.
        n_flips: Desired number of sign-flip vectors.
        cells: Integer array of shape ``(n_samples,)`` mapping each
            observation to a cell label (0-indexed).
        random_state: Seed for reproducibility.
        exclude_identity: If ``True``, the all-+1 vector is excluded.

    Returns:
        Array of shape ``(B, n_samples)`` with entries in {−1, +1}.

    Warns:
        UserWarning: If 2^G − 1 < *n_flips*.
    """
    rng = np.random.default_rng(random_state)
    cells = np.asarray(cells)

    unique_cells = np.unique(cells)
    G = len(unique_cells)

    # Build cell → observation index map.
    cell_indices: list[np.ndarray] = [np.where(cells == c)[0] for c in unique_cells]

    # ---- Budget check: 2^G --------------------------------------
    total = 1 << G
    available = total - 1 if exclude_identity else total

    if available < 1:
        raise ValueError(
            f"No between-cell sign-flip vectors available for G={G} "
            f"cells with exclude_identity={exclude_identity}."
        )

    if n_flips > available:
        warnings.warn(
            f"Only {available} unique between-cell sign-flip vectors "
            f"exist (2^{G} - 1 = {available} with "
            f"exclude_identity={exclude_identity}), but {n_flips} were "
            f"requested.  Returning all {available}.",
            UserWarning,
            stacklevel=2,
        )
        n_flips = available

    # ---- Exhaustive path (small G) -------------------------------
    if G <= 20 and available <= n_flips:
        ranks = np.arange(1 if exclude_identity else 0, total, dtype=np.int64)
        bits = np.arange(G, dtype=np.int64)
        binary = (ranks[:, np.newaxis] >> bits[np.newaxis, :]) & 1
        cell_signs = 1 - 2 * binary.astype(np.int8)  # (available, G)

        out = np.ones((available, n_samples), dtype=np.int8)
        for g in range(G):
            out[:, cell_indices[g]] = cell_signs[:, g : g + 1]
        return out

    # ---- Lehmer-style sampling (small G, subset requested) -------
    if G <= 20:
        pool_start = 1 if exclude_identity else 0
        ranks = rng.choice(total - pool_start, size=n_flips, replace=False) + pool_start  # type: ignore[assignment]
        bits = np.arange(G, dtype=np.int64)
        binary = (ranks[:, np.newaxis] >> bits[np.newaxis, :]) & 1
        cell_signs = 1 - 2 * binary.astype(np.int8)

        out = np.ones((n_flips, n_samples), dtype=np.int8)
        for g in range(G):
            out[:, cell_indices[g]] = cell_signs[:, g : g + 1]
        return out

    # ---- Random sampling with dedup (large G) --------------------
    _CHOICES = np.array([-1, 1], dtype=np.int8)
    identity = bytes(np.ones(G, dtype=np.int8))
    seen: set[bytes] = set()
    if exclude_identity:
        seen.add(identity)

    result = np.empty((n_flips, n_samples), dtype=np.int8)
    count = 0
    max_attempts = n_flips * 20 + 1000

    for _ in range(max_attempts):
        if count >= n_flips:
            break
        cell_signs = rng.choice(_CHOICES, size=G)
        key = cell_signs.tobytes()
        if key not in seen:
            seen.add(key)
            row = np.ones(n_samples, dtype=np.int8)
            for g in range(G):
                row[cell_indices[g]] = cell_signs[g]
            result[count] = row
            count += 1

    if count < n_flips:
        warnings.warn(
            f"Could only generate {count} unique between-cell "
            f"sign-flip vectors after {max_attempts} attempts "
            f"(requested {n_flips}).",
            UserWarning,
            stacklevel=2,
        )

    return result[:count]


def generate_two_stage_sign_flips(
    n_samples: int,
    n_flips: int,
    cells: np.ndarray,
    random_state: int | None = None,
    exclude_identity: bool = True,
    max_exhaustive: int = 20,
) -> np.ndarray:
    """Generate sign-flip vectors via block sign × within-cell sign.

    Provided for API symmetry with
    :func:`~randomization_tests.permutations.generate_two_stage_permutations`.

    .. note::

        Two-stage sign-flip is **statistically equivalent** to
        unconstrained sign-flip.  The two-stage procedure multiplies
        a uniform block sign (±1) by an independent within-cell
        Rademacher variable for each observation.  Because
        ``Rademacher × ±1 = Rademacher`` (the product of two
        independent symmetric ±1 variables is again symmetric ±1),
        the resulting distribution is identical to assigning
        independent Rademacher signs globally.  This function
        therefore delegates directly to :func:`generate_sign_flips`
        (ignoring *cells*).

    Args:
        n_samples: Total number of observations.
        n_flips: Desired number of sign-flip vectors.
        cells: Integer array of shape ``(n_samples,)`` mapping each
            observation to a cell label (present for API consistency,
            but **ignored** since the result is identical to the
            unconstrained case).
        random_state: Seed for reproducibility.
        exclude_identity: If ``True``, the all-+1 vector is excluded.
        max_exhaustive: Passed through to :func:`generate_sign_flips`.

    Returns:
        Array of shape ``(B, n_samples)`` with entries in {−1, +1}.
    """
    # Two-stage sign-flip == unconstrained (block × Rademacher = Rademacher).
    return generate_sign_flips(
        n_samples,
        n_flips,
        random_state=random_state,
        exclude_identity=exclude_identity,
        max_exhaustive=max_exhaustive,
    )


# ------------------------------------------------------------------ #
# Symmetry diagnostic
# ------------------------------------------------------------------ #


def validate_symmetry(
    residuals: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Check whether residuals are approximately symmetric about zero.

    Runs a Wilcoxon signed-rank test (two-sided) on the residuals.
    This is a **diagnostic**, not a hard gate — the user decides
    whether to trust the sign-flip assumption based on the result.

    Args:
        residuals: Residual vector of shape ``(n,)``.
        alpha: Significance level for the symmetry verdict.

    Returns:
        Dict with keys:

        * ``"is_symmetric"`` — ``True`` if the Wilcoxon test does
          *not* reject symmetry at level *alpha*.
        * ``"test_statistic"`` — Wilcoxon W statistic.
        * ``"p_value"`` — two-sided p-value from the Wilcoxon test.
    """
    residuals = np.asarray(residuals, dtype=float).ravel()

    # Drop exact zeros — Wilcoxon signed-rank discards them.
    nonzero = residuals[residuals != 0.0]

    if len(nonzero) < 10:
        # Too few observations for a meaningful symmetry test.
        return {
            "is_symmetric": True,
            "test_statistic": float("nan"),
            "p_value": float("nan"),
        }

    result = stats.wilcoxon(nonzero, alternative="two-sided")

    return {
        "is_symmetric": bool(result.pvalue >= alpha),
        "test_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


# ------------------------------------------------------------------ #
# Nested (multi-level) sign-flip generator
# ------------------------------------------------------------------ #
#
# Parallel to generate_nested_permutations() in permutations.py.
# Walks an ExchangeabilityTree recursively, composing per-level
# sign-flip vectors according to the Winkler et al. (2015) PALM
# algorithm adapted for sign-flip randomization.

from .exchangeability import ExchangeabilityNode, ExchangeabilityTree  # noqa: E402


def generate_nested_sign_flips(
    n_samples: int,
    n_flips: int,
    tree: ExchangeabilityTree,
    random_state: int | None = None,
    exclude_identity: bool = True,
) -> np.ndarray:
    """Generate sign-flip vectors respecting nested exchangeability.

    Each output row is a vector of ±1 values that respects the
    per-level strategies encoded in *tree*.  The algorithm
    recursively composes per-level sign vectors from root to leaves.

    Strategy semantics for sign-flip:

    * ``"within"`` — independent signs within each block.
    * ``"between"`` — one uniform sign per child block, composed
      with recursively generated sub-vectors.
    * ``"two-stage"`` — uniform block sign × independent within-block
      signs.
    * ``"whole"`` — independent signs for all observations, ignoring
      block structure.

    Args:
        n_samples: Total number of observations.
        n_flips: Desired number of sign-flip vectors.
        tree: An :class:`ExchangeabilityTree` encoding the nesting
            and per-level strategies.
        random_state: Seed for reproducibility.
        exclude_identity: If ``True``, the all-+1 vector is excluded.

    Returns:
        Array of shape ``(B, n_samples)`` with entries in {−1, +1},
        where B ≤ *n_flips*.

    Warns:
        UserWarning: If the reference set is smaller than *n_flips*.
    """
    rng = np.random.default_rng(random_state)

    # ---- Budget check --------------------------------------------
    ref_size = tree.reference_set_size("sign_flip", cap=n_flips + 2)
    available = ref_size - 1 if exclude_identity else ref_size

    if available < n_flips:
        warnings.warn(
            f"Only {available} unique nested sign-flip vectors are "
            f"available but {n_flips} were requested.  "
            f"Capping at {available}.",
            UserWarning,
            stacklevel=2,
        )
        n_flips = available

    if n_flips == 0:
        return np.empty((0, n_samples), dtype=np.int8)

    # ---- Generate with hash-based dedup --------------------------
    identity = bytes(np.ones(n_samples, dtype=np.int8))
    seen: set[bytes] = set()
    if exclude_identity:
        seen.add(identity)

    result = np.empty((n_flips, n_samples), dtype=np.int8)
    count = 0

    max_attempts = n_flips * 20 + 1000
    attempts = 0

    while count < n_flips and attempts < max_attempts:
        signs = _nested_flip_once(tree.root, 0, tree.strategies, rng)
        key = signs.tobytes()
        if key not in seen:
            seen.add(key)
            result[count] = signs
            count += 1
        attempts += 1

    if count < n_flips:
        warnings.warn(
            f"Could only generate {count} unique nested "
            f"sign-flip vectors after {max_attempts} attempts "
            f"(requested {n_flips}).",
            UserWarning,
            stacklevel=2,
        )

    return result[:count]


def _nested_flip_once(
    node: ExchangeabilityNode,
    level: int,
    strategies: tuple[str, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a single nested sign-flip vector for a subtree.

    Returns an int8 array of length ``node.size`` with entries ±1,
    ordered by the node's sorted indices.
    """
    _CHOICES = np.array([-1, 1], dtype=np.int8)

    # Leaf or past deepest strategy: independent signs.
    if node.is_leaf or level >= len(strategies):
        return rng.choice(_CHOICES, size=node.size)

    strategy = strategies[level]

    if strategy == "whole":
        return rng.choice(_CHOICES, size=node.size)

    children = node.children

    # Build position map (same as _nested_perm_once).
    child_positions: list[np.ndarray] = []
    for child in children:
        positions = np.searchsorted(node.indices, child.indices)
        child_positions.append(positions)

    out = np.empty(node.size, dtype=np.int8)

    if strategy == "within":
        # Independent signs within each child block.
        for i, child in enumerate(children):
            if child.is_leaf:
                sub_signs = rng.choice(_CHOICES, size=child.size)
            else:
                sub_signs = _nested_flip_once(child, level + 1, strategies, rng)
            out[child_positions[i]] = sub_signs
        return out

    if strategy == "between":
        # One uniform sign per child, composed with recursion.
        for i, child in enumerate(children):
            block_sign = rng.choice(_CHOICES)
            if child.is_leaf:
                sub_signs = rng.choice(_CHOICES, size=child.size)
            else:
                sub_signs = _nested_flip_once(child, level + 1, strategies, rng)
            out[child_positions[i]] = block_sign * sub_signs
        return out

    # strategy == "two-stage"
    # Uniform block sign × independent within-block signs.
    for i, child in enumerate(children):
        block_sign = rng.choice(_CHOICES)
        within_signs = rng.choice(_CHOICES, size=child.size)
        out[child_positions[i]] = block_sign * within_signs
    return out

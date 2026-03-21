"""Hierarchical exchangeability structures (Winkler et al. 2015 PALM).

This module defines the :class:`ExchangeabilityTree` data structure for
representing multi-level nested exchangeability constraints.  A tree
encodes which observations may be permuted or sign-flipped together
at each level of a hierarchy.

Single-level exchangeability (flat cells) is handled by the existing
generators in :mod:`permutations`.  The tree extends this to nested
designs such as students within classrooms within schools, where
different levels require different permutation strategies.

**Nested vs. crossed factors:**

*  **Nested** — classroom ⊂ school.  Each inner group belongs to
   exactly one outer group.  Represented by this tree.
*  **Crossed** — school × gender.  Independent factors whose
   combinations form cells.  Handled by ``_cross_classify()`` in
   :mod:`core` and produce flat cell arrays.

If a design has both nested and crossed elements, the crossed factors
should be pre-cross-classified into the labels at the appropriate
tree level before calling :meth:`ExchangeabilityTree.from_labels`.

Strategy semantics at each level
--------------------------------

The strategy at level *k* determines what happens among the children
(sub-blocks) of each node at that level:

``"within"``
    Preserve block boundaries.  At the **leaf** level this means free
    permutation of observations within each block (contribution
    ∏ n_c!).  At a **non-leaf** level this means "don't swap blocks
    at this level — recurse into children and let deeper strategies
    control permutation" (contribution 1 × recursion).

``"between"``
    Swap children as whole units (same-size constraint).  Contribution
    ∏_s count_s!  where count_s is the number of children sharing each
    unique size.  Then recurse into children for deeper permutation.

``"two-stage"``
    Compose between-block swaps and within-block free permutation at
    this level.  Contribution (∏_s count_s!) × (∏ n_c!).  This
    ignores sub-structure within children (children are treated as
    structureless blocks), matching the single-level semantics.

``"whole"``
    All observations at this level are freely permutable, ignoring
    block boundaries entirely.  Contribution n!.

Example
-------
>>> import numpy as np
>>> schools = np.array([0, 0, 0, 0, 1, 1, 1, 1])
>>> classes = np.array([0, 0, 1, 1, 2, 2, 3, 3])
>>> tree = ExchangeabilityTree.from_labels(
...     [schools, classes],
...     strategies=["between", "within"],
... )
>>> tree.n_levels
2
>>> tree.to_flat_cells()   # leaf-level cell labels
array([0, 0, 1, 1, 2, 2, 3, 3])
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

_VALID_STRATEGIES = frozenset({"within", "between", "two-stage", "whole"})


# ------------------------------------------------------------------ #
# Tree node
# ------------------------------------------------------------------ #


@dataclass(frozen=True, eq=False)
class ExchangeabilityNode:
    """A node in the exchangeability tree.

    Each node represents a block of observations at some level of
    the hierarchy.  Internal nodes have children (sub-blocks); leaf
    nodes have no children and represent the finest partition.

    Parameters
    ----------
    indices : np.ndarray
        Sorted integer indices of observations in this subtree.
    children : tuple[ExchangeabilityNode, ...]
        Child nodes at the next nesting level (empty for leaves).
    """

    indices: np.ndarray
    children: tuple[ExchangeabilityNode, ...] = ()

    @property
    def size(self) -> int:
        """Number of observations in this subtree."""
        return len(self.indices)

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children."""
        return len(self.children) == 0


# ------------------------------------------------------------------ #
# Tree container
# ------------------------------------------------------------------ #


@dataclass(frozen=True, eq=False)
class ExchangeabilityTree:
    """Hierarchical exchangeability structure (Winkler et al. 2015).

    Represents a nested design where observations are grouped at
    multiple levels, and each level has an associated permutation
    or sign-flip strategy.  Strategies are per-level (all nodes at
    the same depth share the same strategy), matching the standard
    PALM formulation.

    Parameters
    ----------
    root : ExchangeabilityNode
        Root of the tree (covers all observations).
    strategies : tuple[str, ...]
        One strategy per nesting level (outermost first).  Valid
        values: ``"within"``, ``"between"``, ``"two-stage"``,
        ``"whole"``.
    level_labels : tuple[str, ...] or None
        Optional human-readable names for each level
        (e.g. ``("school", "classroom")``).
    """

    root: ExchangeabilityNode
    strategies: tuple[str, ...]
    level_labels: tuple[str, ...] | None = None

    # ---- Properties ----------------------------------------------

    @property
    def n_levels(self) -> int:
        """Number of nesting levels (= number of strategies)."""
        return len(self.strategies)

    @property
    def n_samples(self) -> int:
        """Total number of observations."""
        return self.root.size

    # ---- Construction --------------------------------------------

    @classmethod
    def from_labels(
        cls,
        label_arrays: list[np.ndarray],
        strategies: list[str] | None = None,
        level_labels: list[str] | None = None,
    ) -> ExchangeabilityTree:
        """Build a tree from nesting-level label arrays.

        Parameters
        ----------
        label_arrays : list[np.ndarray]
            One array per nesting level, outermost first.  Each
            array has shape ``(n,)`` and maps observations to group
            labels at that level.

            Labels need not be globally unique — they are resolved
            *within* each parent group.  For example classroom IDs
            ``[1, 2]`` in school A and ``[1, 2]`` in school B
            produce four distinct leaf blocks.

        strategies : list[str] or None
            Per-level permutation strategy.  If ``None``, defaults
            to ``"between"`` for all outer levels and ``"within"``
            for the innermost level.

        level_labels : list[str] or None
            Optional human-readable level names.

        Returns
        -------
        ExchangeabilityTree

        Raises
        ------
        ValueError
            If label arrays have inconsistent lengths, labels are
            not properly nested, or strategies are invalid.
        """
        if not label_arrays:
            raise ValueError("At least one label array is required.")

        n_samples = len(label_arrays[0])
        for i, arr in enumerate(label_arrays):
            if len(arr) != n_samples:
                raise ValueError(
                    f"Label array at index {i} has length {len(arr)}, "
                    f"expected {n_samples}."
                )

        # Default strategies: between for outer, within for innermost.
        if strategies is None:
            if len(label_arrays) == 1:
                strategies = ["within"]
            else:
                strategies = ["between"] * (len(label_arrays) - 1) + ["within"]

        if len(strategies) != len(label_arrays):
            raise ValueError(
                f"Got {len(strategies)} strategies for "
                f"{len(label_arrays)} nesting levels."
            )

        for s in strategies:
            if s not in _VALID_STRATEGIES:
                raise ValueError(
                    f"Invalid strategy {s!r}.  Valid: {sorted(_VALID_STRATEGIES)}."
                )

        if level_labels is not None and len(level_labels) != len(label_arrays):
            raise ValueError(
                f"Got {len(level_labels)} level labels for "
                f"{len(label_arrays)} nesting levels."
            )

        # Convert label arrays to integer codes for fast grouping.
        int_labels = [_to_codes(arr) for arr in label_arrays]

        all_indices = np.arange(n_samples, dtype=np.intp)
        root = _build_subtree(all_indices, int_labels, level=0, depth=len(int_labels))

        tree = cls(
            root=root,
            strategies=tuple(strategies),
            level_labels=tuple(level_labels) if level_labels is not None else None,
        )
        tree.validate()
        return tree

    # ---- Backward-compatible flattening --------------------------

    def to_flat_cells(self) -> np.ndarray:
        """Collapse tree to innermost-level cell labels.

        Returns an integer array of shape ``(n,)`` where each
        observation is labelled by its leaf-level group.  This is
        the finest partition defined by the nesting hierarchy —
        **not** a cross-classification of all levels.

        For backward compatibility, this output can be passed
        directly to the existing single-level generators in
        :mod:`permutations`.
        """
        cells = np.empty(self.n_samples, dtype=np.intp)
        for cell_id, leaf in enumerate(_iter_leaves(self.root)):
            cells[leaf.indices] = cell_id
        return cells

    # ---- Reference set size --------------------------------------

    def reference_set_size(
        self,
        randomization: str = "permute",
        cap: int | None = None,
    ) -> int:
        """Compute the total reference set size for the tree.

        The total is the product of per-level reference set sizes
        across all nodes, matching the budget checks in the existing
        single-level generators.

        For sign-flip randomization, this is a conservative upper bound
        because the composition of between-level and within-level
        sign-flips can produce redundant sign vectors.  The actual
        deduplication pass in the generators handles this.

        Parameters
        ----------
        randomization : str
            ``"permute"`` for permutation reference set,
            ``"sign_flip"`` for sign-flip reference set.
        cap : int or None
            Early-stopping bound to avoid unbounded big-int
            arithmetic.  Defaults to 10^18.

        Returns
        -------
        int
            Total reference set size (capped if *cap* is set).
        """
        if cap is None:
            cap = 10**18

        if randomization not in ("permute", "sign_flip"):
            raise ValueError(
                f"Invalid randomization {randomization!r}.  "
                f"Expected 'permute' or 'sign_flip'."
            )

        return self._node_ref_size(
            self.root, level=0, randomization=randomization, cap=cap
        )

    def _node_ref_size(
        self,
        node: ExchangeabilityNode,
        level: int,
        randomization: str,
        cap: int,
    ) -> int:
        """Recursive reference set size for a subtree."""
        if node.is_leaf or level >= self.n_levels:
            return 1

        children = node.children
        child_sizes = [c.size for c in children]
        n_children = len(children)

        if randomization == "permute":
            return self._permute_ref(
                node, level, children, child_sizes, n_children, cap
            )
        return self._sign_flip_ref(node, level, children, child_sizes, n_children, cap)

    def _permute_ref(
        self,
        node: ExchangeabilityNode,
        level: int,
        children: tuple[ExchangeabilityNode, ...],
        child_sizes: list[int],
        n_children: int,
        cap: int,
    ) -> int:
        """Permutation reference set size at one node."""
        strategy = self.strategies[level]

        if strategy == "within":
            # Leaf children: free permutation (n_c! each).
            # Non-leaf children: recurse (deeper levels decide).
            total = 1
            for child in children:
                ct = (
                    math.factorial(child.size)
                    if child.is_leaf
                    else self._node_ref_size(child, level + 1, "permute", cap)
                )
                total *= ct
                if total >= cap:
                    return cap
            return total

        if strategy == "between":
            # Swap children as whole units (same-size constraint).
            counts = Counter(child_sizes)
            total = 1
            for cnt in counts.values():
                total *= math.factorial(cnt)
                if total >= cap:
                    return cap
            # Recurse into children for deeper permutation.
            for child in children:
                ct = self._node_ref_size(child, level + 1, "permute", cap)
                total *= ct
                if total >= cap:
                    return cap
            return total

        if strategy == "two-stage":
            # Between × free-within at this level (ignores sub-structure).
            counts = Counter(child_sizes)
            between = 1
            for cnt in counts.values():
                between *= math.factorial(cnt)
                if between >= cap:
                    return cap
            within = 1
            for s in child_sizes:
                within *= math.factorial(s)
                if within >= cap:
                    within = cap
                    break
            return min(between * within, cap)

        # strategy == "whole"
        return min(math.factorial(node.size), cap)

    def _sign_flip_ref(
        self,
        node: ExchangeabilityNode,
        level: int,
        children: tuple[ExchangeabilityNode, ...],
        child_sizes: list[int],
        n_children: int,
        cap: int,
    ) -> int:
        """Sign-flip reference set size at one node (upper bound)."""
        strategy = self.strategies[level]

        if strategy == "within":
            total = 1
            for child in children:
                ct = (
                    _safe_pow2(child.size, cap)
                    if child.is_leaf
                    else self._node_ref_size(child, level + 1, "sign_flip", cap)
                )
                total *= ct
                if total >= cap:
                    return cap
            return total

        if strategy == "between":
            # Uniform sign per child, then recurse.
            total = _safe_pow2(n_children, cap)
            for child in children:
                ct = self._node_ref_size(child, level + 1, "sign_flip", cap)
                total *= ct
                if total >= cap:
                    return cap
            return total

        if strategy == "two-stage":
            # Between-signs compose with within-signs; effective
            # set ≤ 2^(total observations).
            total_obs = sum(child_sizes)
            return _safe_pow2(total_obs, cap)

        # strategy == "whole"
        return _safe_pow2(node.size, cap)

    # ---- Validation ----------------------------------------------

    def validate(self) -> None:
        """Check structural consistency of the tree.

        Verifies:

        * Tree depth equals the number of strategies.
        * Children partition parent indices exactly (no gaps,
          no overlaps, children ⊆ parent).
        * All branches have uniform depth.
        * Level-label count (if provided) matches level count.

        Raises
        ------
        ValueError
            If any consistency check fails.
        """
        # Uniform depth check (also returns the depth).
        actual_depth = _check_uniform_depth(self.root)
        if actual_depth != self.n_levels:
            raise ValueError(
                f"Tree has depth {actual_depth} but "
                f"{self.n_levels} strategies were provided."
            )

        if self.level_labels is not None and len(self.level_labels) != self.n_levels:
            raise ValueError(
                f"Got {len(self.level_labels)} level labels for {self.n_levels} levels."
            )

        # Nesting: children partition parent.
        _validate_node(self.root)


# ------------------------------------------------------------------ #
# Private helpers
# ------------------------------------------------------------------ #


def _to_codes(arr: np.ndarray) -> np.ndarray:
    """Convert an arbitrary label array to contiguous integer codes."""
    arr = np.asarray(arr)
    _, codes = np.unique(arr, return_inverse=True)
    return codes  # type: ignore[no-any-return]


def _build_subtree(
    indices: np.ndarray,
    labels: list[np.ndarray],
    level: int,
    depth: int,
) -> ExchangeabilityNode:
    """Recursively construct the tree."""
    if level >= depth:
        # Leaf level — finest partition.
        return ExchangeabilityNode(indices=np.sort(indices))

    # Group by the current level's labels.
    level_labels = labels[level][indices]
    unique_vals = np.unique(level_labels)

    children: list[ExchangeabilityNode] = []
    for val in unique_vals:
        mask = level_labels == val
        child_indices = indices[mask]
        child = _build_subtree(child_indices, labels, level + 1, depth)
        children.append(child)

    return ExchangeabilityNode(
        indices=np.sort(indices),
        children=tuple(children),
    )


def _iter_leaves(node: ExchangeabilityNode) -> Iterator[ExchangeabilityNode]:
    """Yield all leaf nodes in depth-first order."""
    if node.is_leaf:
        yield node
    else:
        for child in node.children:
            yield from _iter_leaves(child)


def _check_uniform_depth(node: ExchangeabilityNode) -> int:
    """Return the depth of the subtree, raising if non-uniform."""
    if node.is_leaf:
        return 0

    depths: set[int] = set()
    for child in node.children:
        depths.add(_check_uniform_depth(child))

    if len(depths) > 1:
        raise ValueError(
            f"Non-uniform tree depth: children have depths "
            f"{sorted(depths)}.  Ensure labels define a properly "
            f"nested hierarchy."
        )

    return 1 + depths.pop()


def _validate_node(node: ExchangeabilityNode) -> None:
    """Recursively validate nesting consistency."""
    if node.is_leaf:
        return

    parent_set = set(node.indices.tolist())

    # Collect all child indices and check subset + partition.
    union: list[int] = []
    for child in node.children:
        child_set = set(child.indices.tolist())
        if not child_set.issubset(parent_set):
            raise ValueError(
                f"Child indices {sorted(child_set)} are not a subset "
                f"of parent indices {sorted(parent_set)}."
            )
        union.extend(child.indices.tolist())
        _validate_node(child)

    if sorted(union) != sorted(parent_set):
        raise ValueError(
            f"Children do not partition parent.  Parent has "
            f"{sorted(parent_set)}, children cover {sorted(union)}."
        )


def _safe_pow2(n: int, cap: int) -> int:
    """Return min(2 ** n, cap) without allocating huge integers."""
    if n >= 63:
        return cap
    val = 1 << n
    return min(val, cap)

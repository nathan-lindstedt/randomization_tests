"""Kernel protocol, KernelEval dataclass, Nyström approximation, and gram helpers.

Provides a unified representation for kernel evaluations — both exact (full Gram)
and approximate (Nyström factorisation) — and a set of arithmetic helpers that
MMD and HSIC use to compute traces/sums without ever materialising the full n×n
Gram matrix when landmarks m < n.

Key mathematical insight
------------------------
In a permutation test, Type I control is exact for ANY test statistic — it comes
from the permutation, not from the statistic.  This means approximation errors in
MMD/HSIC affect **power only, never validity**.

This lets us design a unified representation where the full Gram matrix is the
special case m = n of a low-rank Nyström factorisation:

    K ≈ K_nm @ K_mm_inv @ K_nm.T

When m = n (exact):  K_nm IS the Gram K;  K_mm_inv is unused.
When m < n (Nyström): K_nm is (n, m);  K_mm_inv = pinv(K_mm).

All MMD/HSIC arithmetic is expressed in terms of :func:`gram_trace_product`,
:func:`gram_row_sums`, :func:`gram_centering`, and :func:`gram_permute` —
none of which materialise the full matrix when m < n.

Nyström landmark selection
--------------------------
Leverage-score sampling (approximate):

1. Compute a pilot sample of column norms to proxy leverage scores.
2. Sample m landmarks proportional to those scores.
3. Compute K_nm (n×m) and K_mm (m×m).
4. K_mm_inv = pinv(K_mm) with rcond=1e-10.

With m ~ O(√n) this achieves the minimax-optimal MMD separation rate
(Chatalic et al. 2025).

Why Nyström over Random Fourier Features (RFF)
----------------------------------------------
RFF requires stationary (shift-invariant) kernels — it cannot handle
CosineKernel or PrecomputedKernel.  Nyström works with ALL kernels, making it
the principled universal default.  Both are minimax optimal for MMD (Feb 2025).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# ------------------------------------------------------------------ #
# KernelEval — unified exact / Nyström representation
# ------------------------------------------------------------------ #


@dataclass
class KernelEval:
    """Result of a kernel evaluation — full Gram or Nyström factorisation.

    Unified representation:  K ≈ K_nm @ K_mm_inv @ K_nm.T

    When ``is_exact=True`` (m = n):
        ``K_nm`` **is** the full n×n Gram matrix.
        ``K_mm_inv`` is a sentinel ``np.eye(n)`` that arithmetic helpers
        never use (they branch on ``is_exact``).

    When ``is_exact=False`` (m < n, Nyström):
        ``K_nm`` is the (n, m) left factor.
        ``K_mm_inv`` is the (m, m) pseudo-inverse of the landmark Gram.

    Attributes
    ----------
    K_nm:
        Shape (n, m).  Full Gram when ``is_exact=True``; left Nyström
        factor when ``is_exact=False``.
    K_mm_inv:
        Shape (m, m).  Sentinel ``eye(n)`` when exact; ``pinv(K_mm)``
        when Nyström.
    is_exact:
        ``True`` iff m == n and no approximation was applied.
    n:
        Number of observations.
    m:
        Number of landmarks (equals n when exact).
    """

    K_nm: np.ndarray
    K_mm_inv: np.ndarray
    is_exact: bool
    n: int
    m: int


# ------------------------------------------------------------------ #
# Kernel Protocol
# ------------------------------------------------------------------ #


@runtime_checkable
class Kernel(Protocol):
    """Protocol for kernel functions used in MMD and HSIC tests.

    Implementations must supply three methods:

    * ``__call__`` — compute a Gram matrix from two data arrays.
    * ``evaluate`` — evaluate on a dataset, returning a :class:`KernelEval`
      (full or Nyström depending on *max_landmarks*).
    * ``name`` — human-readable identifier used in :class:`KernelTestResult`.
    """

    def __call__(
        self,
        X1: np.ndarray,
        X2: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute Gram matrix K[i, j] = k(X1[i], X2[j]).

        When *X2* is ``None`` the symmetric matrix K[i, j] = k(X1[i], X1[j])
        is returned.
        """
        ...

    def evaluate(
        self,
        X: np.ndarray,
        *,
        max_landmarks: int | None = None,
        random_state: int | None = None,
    ) -> KernelEval:
        """Evaluate the kernel on *X*, returning a :class:`KernelEval`.

        When *max_landmarks* is ``None`` or ``>= n``: full Gram (exact).
        When *max_landmarks* ``< n``: Nyström approximation with leverage-score
        landmark sampling.  Type I control is exact either way.
        """
        ...

    def name(self) -> str:
        """Human-readable kernel name, stored in :class:`KernelTestResult`."""
        ...


# ------------------------------------------------------------------ #
# Shared Nyström helper
# ------------------------------------------------------------------ #


def _nystrom_evaluate(
    gram_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    max_landmarks: int,
    rng: np.random.Generator,
) -> KernelEval:
    """Compute a Nyström :class:`KernelEval` for *X* using leverage-score sampling.

    Parameters
    ----------
    gram_fn:
        Callable ``(X1, X2) -> np.ndarray`` that computes the Gram block.
        Must accept two 2-D arrays of shapes ``(n, d)`` and ``(m, d)``.
    X:
        Data array of shape ``(n, d)`` or ``(n,)`` (1-D is reshaped).
    max_landmarks:
        Number of landmark points m.  Must satisfy ``1 <= max_landmarks < n``.
    rng:
        NumPy random Generator for reproducible landmark selection.

    Returns
    -------
    KernelEval
        Low-rank factorisation with ``is_exact=False``.
    """
    X = np.atleast_2d(X) if X.ndim == 1 else X
    n = X.shape[0]
    m = max_landmarks

    # --- approximate leverage scores via pilot column norms -----------
    pilot_size = max(1, min(50, m, n))
    pilot_idx = rng.choice(n, size=pilot_size, replace=False)
    K_pilot = gram_fn(X, X[pilot_idx])
    scores = np.linalg.norm(K_pilot, axis=1)

    # guard against all-zero scores (degenerate kernels)
    score_sum = scores.sum()
    if score_sum == 0.0:
        probs = np.full(n, 1.0 / n)
    else:
        probs = scores / score_sum

    # sample 2*m candidates with replacement, unique-ify, take first m
    candidates = rng.choice(n, size=min(2 * m, n), replace=True, p=probs)
    landmarks = np.unique(candidates)[:m]
    # if unique candidates < m, pad with uniform random draws
    if len(landmarks) < m:
        remaining = np.setdiff1d(np.arange(n), landmarks)
        extra = rng.choice(remaining, size=m - len(landmarks), replace=False)
        landmarks = np.concatenate([landmarks, extra])

    K_nm = gram_fn(X, X[landmarks])  # (n, m)
    K_mm = gram_fn(X[landmarks], X[landmarks])  # (m, m)
    K_mm_inv = np.linalg.pinv(K_mm, rcond=1e-10)

    return KernelEval(K_nm=K_nm, K_mm_inv=K_mm_inv, is_exact=False, n=n, m=m)


# ------------------------------------------------------------------ #
# GaussianKernel
# ------------------------------------------------------------------ #


class GaussianKernel:
    """Gaussian (RBF) kernel: k(x, y) = exp(-||x - y||² / 2σ²).

    Parameters
    ----------
    sigma:
        Bandwidth parameter.  Pass a positive float, or ``"median"`` to use
        the median heuristic (median pairwise distance), which adapts to the
        data scale and gives good power across settings.
    """

    def __init__(self, sigma: float | str = "median") -> None:
        if sigma != "median" and not (isinstance(sigma, (int, float)) and sigma > 0):
            raise ValueError(
                f"sigma must be a positive float or 'median', got {sigma!r}"
            )
        self._sigma = sigma

    def _resolve_sigma(
        self, X: np.ndarray, X2: np.ndarray | None, rng: np.random.Generator
    ) -> float:
        if self._sigma != "median":
            return float(self._sigma)
        # median pairwise distance over a pilot sample (at most 500 points)
        pool = X if X2 is None else np.vstack([X, X2])
        pool = np.atleast_2d(pool)
        idx = rng.choice(pool.shape[0], size=min(500, pool.shape[0]), replace=False)
        sample = pool[idx]
        dists = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
        upper = dists[np.triu_indices_from(dists, k=1)]
        med = float(np.median(upper)) if upper.size > 0 else 1.0
        return med if med > 0.0 else 1.0

    def _gram(self, X1: np.ndarray, X2: np.ndarray, sigma: float) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        # squared Euclidean distances via identity ||x-y||² = ||x||² + ||y||² - 2x·y
        sq1 = np.sum(X1**2, axis=1, keepdims=True)
        sq2 = np.sum(X2**2, axis=1, keepdims=True)
        sq_dist = sq1 + sq2.T - 2.0 * X1 @ X2.T
        sq_dist = np.maximum(sq_dist, 0.0)  # numerical guard
        return np.asarray(np.exp(-sq_dist / (2.0 * sigma**2)))

    def __call__(
        self,
        X1: np.ndarray,
        X2: np.ndarray | None = None,
    ) -> np.ndarray:
        sigma = self._resolve_sigma(X1, X2, np.random.default_rng())
        X2_ = X1 if X2 is None else X2
        return self._gram(X1, X2_, sigma)

    def evaluate(
        self,
        X: np.ndarray,
        *,
        max_landmarks: int | None = None,
        random_state: int | None = None,
    ) -> KernelEval:
        X = np.atleast_2d(X) if X.ndim == 1 else X
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        sigma = self._resolve_sigma(X, None, rng)

        if max_landmarks is None or max_landmarks >= n:
            K = self._gram(X, X, sigma)
            return KernelEval(K_nm=K, K_mm_inv=np.eye(n), is_exact=True, n=n, m=n)

        return _nystrom_evaluate(
            lambda a, b: self._gram(a, b, sigma), X, max_landmarks, rng
        )

    def name(self) -> str:
        sigma_str = "median" if self._sigma == "median" else f"{self._sigma:.4g}"
        return f"GaussianKernel(sigma={sigma_str})"


# ------------------------------------------------------------------ #
# LaplacianKernel
# ------------------------------------------------------------------ #


class LaplacianKernel:
    """Laplacian kernel: k(x, y) = exp(-||x - y||₁ / σ).

    Robust to outliers due to the L1 norm.  Accepts ``sigma="median"``
    (median of pairwise L1 distances).
    """

    def __init__(self, sigma: float | str = "median") -> None:
        if sigma != "median" and not (isinstance(sigma, (int, float)) and sigma > 0):
            raise ValueError(
                f"sigma must be a positive float or 'median', got {sigma!r}"
            )
        self._sigma = sigma

    def _resolve_sigma(
        self, X: np.ndarray, X2: np.ndarray | None, rng: np.random.Generator
    ) -> float:
        if self._sigma != "median":
            return float(self._sigma)
        pool = X if X2 is None else np.vstack([X, X2])
        pool = np.atleast_2d(pool)
        idx = rng.choice(pool.shape[0], size=min(500, pool.shape[0]), replace=False)
        sample = pool[idx]
        dists = np.sum(np.abs(sample[:, None, :] - sample[None, :, :]), axis=-1)
        upper = dists[np.triu_indices_from(dists, k=1)]
        med = float(np.median(upper)) if upper.size > 0 else 1.0
        return med if med > 0.0 else 1.0

    def _gram(self, X1: np.ndarray, X2: np.ndarray, sigma: float) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        l1_dist = np.sum(np.abs(X1[:, None, :] - X2[None, :, :]), axis=-1)
        return np.asarray(np.exp(-l1_dist / sigma))

    def __call__(
        self,
        X1: np.ndarray,
        X2: np.ndarray | None = None,
    ) -> np.ndarray:
        sigma = self._resolve_sigma(X1, X2, np.random.default_rng())
        X2_ = X1 if X2 is None else X2
        return self._gram(X1, X2_, sigma)

    def evaluate(
        self,
        X: np.ndarray,
        *,
        max_landmarks: int | None = None,
        random_state: int | None = None,
    ) -> KernelEval:
        X = np.atleast_2d(X) if X.ndim == 1 else X
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        sigma = self._resolve_sigma(X, None, rng)

        if max_landmarks is None or max_landmarks >= n:
            K = self._gram(X, X, sigma)
            return KernelEval(K_nm=K, K_mm_inv=np.eye(n), is_exact=True, n=n, m=n)

        return _nystrom_evaluate(
            lambda a, b: self._gram(a, b, sigma), X, max_landmarks, rng
        )

    def name(self) -> str:
        sigma_str = "median" if self._sigma == "median" else f"{self._sigma:.4g}"
        return f"LaplacianKernel(sigma={sigma_str})"


# ------------------------------------------------------------------ #
# CosineKernel
# ------------------------------------------------------------------ #


class CosineKernel:
    """Cosine kernel: k(x, y) = (x · y) / (||x|| · ||y||).

    Suitable for document/text vectors (e.g. TF-IDF matrices).
    Zero vectors are handled by assigning zero similarity.
    """

    def _gram(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        norms1 = np.linalg.norm(X1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(X2, axis=1, keepdims=True)
        # avoid division by zero
        norms1 = np.where(norms1 == 0.0, 1.0, norms1)
        norms2 = np.where(norms2 == 0.0, 1.0, norms2)
        X1_norm = X1 / norms1
        X2_norm = X2 / norms2
        return np.asarray(X1_norm @ X2_norm.T)

    def __call__(
        self,
        X1: np.ndarray,
        X2: np.ndarray | None = None,
    ) -> np.ndarray:
        X2_ = X1 if X2 is None else X2
        return self._gram(X1, X2_)

    def evaluate(
        self,
        X: np.ndarray,
        *,
        max_landmarks: int | None = None,
        random_state: int | None = None,
    ) -> KernelEval:
        X = np.atleast_2d(X) if X.ndim == 1 else X
        n = X.shape[0]
        rng = np.random.default_rng(random_state)

        if max_landmarks is None or max_landmarks >= n:
            K = self._gram(X, X)
            return KernelEval(K_nm=K, K_mm_inv=np.eye(n), is_exact=True, n=n, m=n)

        return _nystrom_evaluate(self._gram, X, max_landmarks, rng)

    def name(self) -> str:
        return "CosineKernel()"


# ------------------------------------------------------------------ #
# LinearKernel
# ------------------------------------------------------------------ #


class LinearKernel:
    """Linear kernel: k(x, y) = x · y.

    Equivalent to PERMANOVA when used with MMD.
    """

    def _gram(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        return np.asarray(X1 @ X2.T)

    def __call__(
        self,
        X1: np.ndarray,
        X2: np.ndarray | None = None,
    ) -> np.ndarray:
        X2_ = X1 if X2 is None else X2
        return self._gram(X1, X2_)

    def evaluate(
        self,
        X: np.ndarray,
        *,
        max_landmarks: int | None = None,
        random_state: int | None = None,
    ) -> KernelEval:
        X = np.atleast_2d(X) if X.ndim == 1 else X
        n = X.shape[0]
        rng = np.random.default_rng(random_state)

        if max_landmarks is None or max_landmarks >= n:
            K = self._gram(X, X)
            return KernelEval(K_nm=K, K_mm_inv=np.eye(n), is_exact=True, n=n, m=n)

        return _nystrom_evaluate(self._gram, X, max_landmarks, rng)

    def name(self) -> str:
        return "LinearKernel()"


# ------------------------------------------------------------------ #
# PrecomputedKernel
# ------------------------------------------------------------------ #


class PrecomputedKernel:
    """Wrapper around a user-supplied Gram matrix.

    Validates on construction that the matrix is square, symmetric, and
    positive semi-definite.  ``evaluate()`` with ``max_landmarks < n``
    performs Nyström directly on the stored matrix using column-indexing.

    Parameters
    ----------
    K:
        The pre-computed n×n Gram matrix.
    validate_psd:
        When ``True`` (default), check that all eigenvalues are ≥ −1e-6.
        Pass ``False`` to skip the O(n³) check for large trusted matrices.
    """

    def __init__(self, K: np.ndarray, *, validate_psd: bool = True) -> None:
        K = np.asarray(K, dtype=float)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError(
                f"PrecomputedKernel requires a square 2-D matrix; got shape {K.shape}"
            )
        if not np.allclose(K, K.T, atol=1e-6):
            raise ValueError("PrecomputedKernel requires a symmetric matrix.")
        if validate_psd:
            min_ev = np.linalg.eigvalsh(K).min()
            if min_ev < -1e-6:
                raise ValueError(
                    f"PrecomputedKernel requires a positive semi-definite matrix; "
                    f"minimum eigenvalue is {min_ev:.6g}."
                )
        self._K = K

    def __call__(
        self,
        X1: np.ndarray,
        X2: np.ndarray | None = None,
    ) -> np.ndarray:
        if X2 is not None:
            raise ValueError(
                "PrecomputedKernel does not support __call__ with two arguments. "
                "Use evaluate() to access Nyström factors."
            )
        return self._K

    def evaluate(
        self,
        X: np.ndarray,
        *,
        max_landmarks: int | None = None,
        random_state: int | None = None,
    ) -> KernelEval:
        n = self._K.shape[0]

        if max_landmarks is None or max_landmarks >= n:
            K = self._K
            return KernelEval(K_nm=K, K_mm_inv=np.eye(n), is_exact=True, n=n, m=n)

        # Nyström on the precomputed matrix via column-indexing
        def _gram_fn(a_ignored: np.ndarray, b_idx_rows: np.ndarray) -> np.ndarray:
            # b_idx_rows contains the actual rows of X[landmarks] — we use
            # them to identify landmark indices by matching to X rows.
            # Since PrecomputedKernel ignores X (it has the full K), we
            # identify landmark indices by the row indices passed in.
            # The _nystrom_evaluate helper calls gram_fn(X, X[landmarks]),
            # so b_idx_rows IS X[landmarks].  We recover landmark indices
            # by finding which rows of X match b_idx_rows.
            # For PrecomputedKernel we store the landmark indices on first call.
            raise RuntimeError("Use _precomputed_nystrom instead")

        return _precomputed_nystrom(
            self._K, max_landmarks, np.random.default_rng(random_state)
        )

    def name(self) -> str:
        return f"PrecomputedKernel(n={self._K.shape[0]})"


def _precomputed_nystrom(
    K: np.ndarray, max_landmarks: int, rng: np.random.Generator
) -> KernelEval:
    """Nyström factorisation of a precomputed Gram matrix K.

    Landmark selection uses column-norm leverage-score sampling directly
    on K (no data array needed).
    """
    n = K.shape[0]
    m = max_landmarks

    # leverage scores from column norms of K
    scores = np.linalg.norm(K, axis=0)
    score_sum = scores.sum()
    probs = scores / score_sum if score_sum > 0.0 else np.full(n, 1.0 / n)

    candidates = rng.choice(n, size=min(2 * m, n), replace=True, p=probs)
    landmarks = np.unique(candidates)[:m]
    if len(landmarks) < m:
        remaining = np.setdiff1d(np.arange(n), landmarks)
        extra = rng.choice(remaining, size=m - len(landmarks), replace=False)
        landmarks = np.concatenate([landmarks, extra])

    K_nm = K[:, landmarks]  # (n, m)
    K_mm = K[np.ix_(landmarks, landmarks)]  # (m, m)
    K_mm_inv = np.linalg.pinv(K_mm, rcond=1e-10)

    return KernelEval(K_nm=K_nm, K_mm_inv=K_mm_inv, is_exact=False, n=n, m=m)


# ------------------------------------------------------------------ #
# Gram arithmetic helpers (Step 5)
# ------------------------------------------------------------------ #


def gram_row_sums(A: KernelEval) -> np.ndarray:
    """Compute K @ 1 (row sums of the kernel matrix) in O(nm).

    Parameters
    ----------
    A:
        A :class:`KernelEval` (exact or Nyström).

    Returns
    -------
    np.ndarray
        1-D array of shape ``(n,)`` containing row sums of K.
    """
    if A.is_exact:
        return np.asarray(A.K_nm.sum(axis=1))
    # Nyström: K ≈ K_nm @ K_mm_inv @ K_nm.T
    # row sums = K @ 1 = K_nm @ (K_mm_inv @ (K_nm.T @ 1))
    col_sums = A.K_nm.sum(axis=0)  # (m,)
    mid: np.ndarray = A.K_mm_inv @ col_sums  # (m,)
    return np.asarray(A.K_nm @ mid)  # (n,)


def gram_centering(A: KernelEval) -> KernelEval:
    """Return H @ K @ H (doubly-centred Gram) in factored form.

    H = I - 11ᵀ/n is the centering matrix.

    For an exact :class:`KernelEval`, computes the full centred matrix.
    For a Nyström :class:`KernelEval`, exploits the identity:

        H @ (K_nm @ K_mm_inv @ K_nm.T) @ H
        = (H @ K_nm) @ K_mm_inv @ (H @ K_nm).T

    so centering reduces to row-centering K_nm.

    Parameters
    ----------
    A:
        A :class:`KernelEval` (exact or Nyström).

    Returns
    -------
    KernelEval
        Centred representation.  ``is_exact`` is preserved.
    """
    if A.is_exact:
        K = A.K_nm
        n = A.n
        row_means = K.mean(axis=1, keepdims=True)
        col_means = K.mean(axis=0, keepdims=True)
        grand_mean = K.mean()
        K_c = K - row_means - col_means + grand_mean
        return KernelEval(K_nm=K_c, K_mm_inv=np.eye(n), is_exact=True, n=n, m=n)

    # Nyström: H @ K_nm centres each column of K_nm
    K_nm_c = A.K_nm - A.K_nm.mean(axis=0, keepdims=True)
    return KernelEval(K_nm=K_nm_c, K_mm_inv=A.K_mm_inv, is_exact=False, n=A.n, m=A.m)


def gram_permute(A: KernelEval, perm: np.ndarray) -> KernelEval:
    """Permute the rows (and columns for exact) of a :class:`KernelEval`.

    For an exact Gram K, both rows and columns are permuted:
    ``K[perm][:, perm]``.

    For a Nyström factorisation, only K_nm rows are reindexed; the
    landmark set (and hence K_mm_inv) is unchanged:
    ``KernelEval(K_nm[perm], K_mm_inv, ...)``.

    Parameters
    ----------
    A:
        A :class:`KernelEval` (exact or Nyström).
    perm:
        Integer permutation array of length n.

    Returns
    -------
    KernelEval
        Permuted representation.  Same ``is_exact`` as input.
    """
    if A.is_exact:
        K_perm = A.K_nm[np.ix_(perm, perm)]
        return KernelEval(K_nm=K_perm, K_mm_inv=A.K_mm_inv, is_exact=True, n=A.n, m=A.m)

    K_nm_perm = A.K_nm[perm]
    return KernelEval(K_nm=K_nm_perm, K_mm_inv=A.K_mm_inv, is_exact=False, n=A.n, m=A.m)


def gram_trace_product(A: KernelEval, B: KernelEval) -> float:
    """Compute tr(K_A @ K_B) without materialising full matrices.

    Three cases:

    **Both exact** (m = n):
        tr(K_A @ K_B) = Σ_ij K_A[i,j] · K_B[j,i]
        = Σ_ij K_A[i,j] · K_B[i,j]   (since both symmetric)
        = (K_A * K_B).sum()   — O(n²).

    **Both Nyström** (m < n):
        tr((A_nm A⁻¹_mm A_nm.T)(B_nm B⁻¹_mm B_nm.T))
        = tr(A_nm.T B_nm B⁻¹_mm B_nm.T A_nm A⁻¹_mm)   (cyclic trace)
        Computed as:
            M  = A_nm.T @ B_nm             (m_A × m_B)
            P  = A⁻¹_mm @ M               (m_A × m_B)
            Q  = B⁻¹_mm @ M.T             (m_B × m_A)
            tr = (P * Q.T).sum()           (scalar)
        Cost: O(n · m_A · m_B).

    **Mixed** (one exact, one Nyström):
        Materialise the Nyström approximation for the Nyström factor and
        fall back to element-wise product.  Correct but O(n²) — this
        path is only hit by pathological callers mixing representations.

    Parameters
    ----------
    A, B:
        :class:`KernelEval` objects (may have different m).

    Returns
    -------
    float
        tr(K_A @ K_B).
    """
    if A.is_exact and B.is_exact:
        return float((A.K_nm * B.K_nm).sum())

    if not A.is_exact and not B.is_exact:
        M = A.K_nm.T @ B.K_nm  # (m_A, m_B)
        P = A.K_mm_inv @ M  # (m_A, m_B)
        Q = B.K_mm_inv @ M.T  # (m_B, m_A)
        return float((P * Q.T).sum())

    # mixed: materialise the Nyström side
    if A.is_exact:
        # A is exact (full n×n), B is Nyström
        K_B = B.K_nm @ B.K_mm_inv @ B.K_nm.T
        return float((A.K_nm * K_B).sum())
    else:
        # A is Nyström, B is exact (full n×n)
        K_A = A.K_nm @ A.K_mm_inv @ A.K_nm.T
        return float((K_A * B.K_nm).sum())

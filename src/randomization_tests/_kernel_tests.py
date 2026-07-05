"""Kernel-based permutation tests: MMD, HSIC, and kernel regression.

Three test functions are provided:

:func:`mmd_test`
    Maximum Mean Discrepancy two-sample test.  Tests H₀: P = Q given
    samples X ~ P and Y ~ Q.  Uses the unbiased U-statistic estimator
    of MMD²:

        MMD²_u(X, Y) = (1/n_x(n_x-1)) Σ_{i≠j} k(xᵢ, xⱼ)
                     + (1/n_y(n_y-1)) Σ_{i≠j} k(yᵢ, yⱼ)
                     - (2/(n_x n_y))  Σᵢⱼ k(xᵢ, yⱼ)

    Permutation validity: relabelling the pooled sample Z = [X; Y] and
    extracting the first n_x points as the permuted "X" group is an
    exchangeable operation under H₀, so the permutation test is
    **finite-sample exact** — given two hypotheses: (i) observations
    are exchangeable under H₀ (e.g. iid sampling), and (ii) all
    data-dependent kernel tuning (median-heuristic bandwidth, Nyström
    landmarks) is computed ONCE from label-free pooled data, never
    per-group or per-permutation.  This module satisfies (ii) by
    construction — preserve that invariant when modifying it.

:func:`hsic_test`
    Hilbert-Schmidt Independence Criterion independence test.  Tests
    H₀: X ⊥ Y given paired observations (Xᵢ, Yᵢ).  The biased HSIC
    estimator is:

        HSIC(X, Y) = (1/n²) tr(K_c L_c)

    where K_c = H K H and L_c = H L H are the doubly-centred kernel
    matrices (H = I - 11ᵀ/n).

    Permutation validity: permuting the rows of Y (breaking the
    pairing) is exchangeable under H₀, so the test is **finite-sample
    exact** under the same two hypotheses as :func:`mmd_test`
    (pair-exchangeability under H₀; label-free one-time kernel tuning
    per variable).

:func:`kernel_regression_test`
    Partial independence test after residualising out confounders Z.
    Computes OLS residuals e_X = X - Z(ZᵀZ)⁻¹ZᵀX and
    e_Y = Y - Z(ZᵀZ)⁻¹ZᵀY, then runs :func:`hsic_test` on
    (e_X, e_Y) with a linear kernel on e_X.

    Unlike the two tests above, this one is residual-based: it is
    **asymptotically exact** only, and OLS removes only the LINEAR
    component of confounding (see the function docstring).

All three functions:

* Use :func:`~randomization_tests.permutations.generate_unique_permutations`
  for the null distribution — consistent with the permutation infrastructure
  used throughout the package.
* Apply the Phipson–Smyth corrected p-value: (b + 1) / (B + 1) where
  b is the count of permuted statistics ≥ the observed statistic.
* Return a :class:`~randomization_tests._results.KernelTestResult`.
* Support Nyström approximation for large n via the *max_landmarks*
  parameter (passed through to :meth:`Kernel.evaluate`).

References
----------
Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B., & Smola, A. (2012).
    A kernel two-sample test. *Journal of Machine Learning Research*, 13,
    723–773.

Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B., & Smola,
    A. J. (2008). A kernel statistical test of independence. *Advances in
    Neural Information Processing Systems*, 20.

Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be
    zero: calculating exact p-values when permutations are randomly drawn.
    *Statistical Applications in Genetics and Molecular Biology*, 9(1).
"""

from __future__ import annotations

import numpy as np

from ._kernels import (
    GaussianKernel,
    Kernel,
    KernelEval,
    LinearKernel,
    gram_centering,
    gram_permute,
    gram_trace_product,
)
from ._results import KernelTestResult
from .permutations import generate_unique_permutations

# ------------------------------------------------------------------ #
# Internal helper — unbiased MMD² from a pooled KernelEval
# ------------------------------------------------------------------ #


def _mmd2_unbiased(
    K: KernelEval,
    idx_x: np.ndarray,
    idx_y: np.ndarray,
) -> float:
    """Compute the unbiased U-statistic estimator of MMD²(X, Y).

    Parameters
    ----------
    K:
        :class:`~._kernels.KernelEval` of the pooled sample Z = [X; Y].
    idx_x:
        Integer indices of the X group within Z.
    idx_y:
        Integer indices of the Y group within Z.

    Returns
    -------
    float
        Unbiased MMD² estimate.  May be negative (unbiasedness is more
        important than non-negativity for permutation tests).
    """
    n_x = len(idx_x)
    n_y = len(idx_y)

    if K.is_exact:
        # exact case: direct sub-matrix slicing
        K_mat = K.K_nm  # full n×n Gram

        # XX diagonal and sum
        K_xx = K_mat[np.ix_(idx_x, idx_x)]
        diag_xx = np.trace(K_xx)
        sum_xx = K_xx.sum() - diag_xx  # off-diagonal sum

        # YY diagonal and sum
        K_yy = K_mat[np.ix_(idx_y, idx_y)]
        diag_yy = np.trace(K_yy)
        sum_yy = K_yy.sum() - diag_yy

        # XY cross sum
        cross_xy = K_mat[np.ix_(idx_x, idx_y)].sum()
    else:
        # Nyström case: K ≈ K_nm K_mm_inv K_nm.T
        # sub-blocks: A_nm = K.K_nm[idx_a], etc.
        A_nm = K.K_nm[idx_x]  # (n_x, m)
        B_nm = K.K_nm[idx_y]  # (n_y, m)
        Minv = K.K_mm_inv  # (m, m)

        # tr(K_xx) — diagonal of A_nm @ Minv @ A_nm.T
        # = sum_i (A_nm[i] . (Minv @ A_nm[i]))
        MiA = Minv @ A_nm.T  # (m, n_x)
        diag_xx = float(np.sum(A_nm * MiA.T))
        sum_xx = float(np.sum(A_nm @ MiA)) - diag_xx

        MiB = Minv @ B_nm.T  # (m, n_y)
        diag_yy = float(np.sum(B_nm * MiB.T))
        sum_yy = float(np.sum(B_nm @ MiB)) - diag_yy

        # cross: tr(A_nm @ Minv @ B_nm.T) = (A_nm @ MiB).sum()
        cross_xy = float(np.sum(A_nm @ MiB))

    xx_term = sum_xx / (n_x * (n_x - 1)) if n_x > 1 else 0.0
    yy_term = sum_yy / (n_y * (n_y - 1)) if n_y > 1 else 0.0
    xy_term = cross_xy / (n_x * n_y)

    return float(xx_term + yy_term - 2.0 * xy_term)


# ------------------------------------------------------------------ #
# mmd_test
# ------------------------------------------------------------------ #


def mmd_test(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    kernel: Kernel | None = None,
    n_permutations: int = 5000,
    max_landmarks: int | None = None,
    random_state: int | None = None,
) -> KernelTestResult:
    """Maximum Mean Discrepancy (MMD) two-sample permutation test.

    Tests the null hypothesis H₀: P = Q given i.i.d. samples X ~ P
    and Y ~ Q.

    Parameters
    ----------
    X:
        Observations from distribution P.  Shape ``(n_x,)`` or
        ``(n_x, d)``.
    Y:
        Observations from distribution Q.  Shape ``(n_y,)`` or
        ``(n_y, d)``.
    kernel:
        Kernel to use.  Defaults to :class:`~._kernels.GaussianKernel`
        with the median bandwidth heuristic.
    n_permutations:
        Number of random relabellings.  Default 5000.
    max_landmarks:
        If not ``None``, use Nyström approximation with this many
        landmark points.  Reduces memory from O(n²) to O(nm).
    random_state:
        Seed for reproducible permutations and Nyström landmark
        selection.  ``None`` gives non-deterministic results.

    Returns
    -------
    KernelTestResult
        ``method="mmd"``.  ``statistic`` is the unbiased MMD² estimate.
    """
    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    X_2d = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr
    Y_2d = Y_arr.reshape(-1, 1) if Y_arr.ndim == 1 else Y_arr

    n_x = X_2d.shape[0]
    n_y = Y_2d.shape[0]
    n = n_x + n_y

    if kernel is None:
        kernel = GaussianKernel(sigma="median")

    # Pool samples; evaluate kernel once
    Z = np.vstack([X_2d, Y_2d])
    K = kernel.evaluate(Z, max_landmarks=max_landmarks, random_state=random_state)

    # Fixed label masks for the observed statistic
    idx_x = np.arange(n_x)
    idx_y = np.arange(n_x, n)

    observed = _mmd2_unbiased(K, idx_x, idx_y)

    # Null distribution via label permutation
    perm_indices = generate_unique_permutations(
        n, n_permutations, random_state=random_state
    )

    null = np.empty(n_permutations, dtype=float)
    for b, perm in enumerate(perm_indices):
        new_x = perm[:n_x]
        new_y = perm[n_x:]
        null[b] = _mmd2_unbiased(K, new_x, new_y)

    # Phipson–Smyth corrected p-value
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))

    return KernelTestResult(
        statistic=observed,
        p_value=p_value,
        null_distribution=null,
        kernel_name=kernel.name(),
        n_permutations=n_permutations,
        method="mmd",
    )


# ------------------------------------------------------------------ #
# hsic_test
# ------------------------------------------------------------------ #


def hsic_test(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    kernel_x: Kernel | None = None,
    kernel_y: Kernel | None = None,
    n_permutations: int = 5000,
    max_landmarks: int | None = None,
    random_state: int | None = None,
) -> KernelTestResult:
    """Hilbert-Schmidt Independence Criterion (HSIC) independence test.

    Tests the null hypothesis H₀: X ⊥ Y given paired observations
    (Xᵢ, Yᵢ), i = 1, …, n.

    Parameters
    ----------
    X:
        First variable.  Shape ``(n,)`` or ``(n, d_x)``.
    Y:
        Second variable.  Shape ``(n,)`` or ``(n, d_y)``.
    kernel_x:
        Kernel for X.  Defaults to :class:`~._kernels.GaussianKernel`.
    kernel_y:
        Kernel for Y.  Defaults to :class:`~._kernels.GaussianKernel`.
    n_permutations:
        Number of random row permutations of Y.  Default 5000.
    max_landmarks:
        Nyström approximation landmark count (applied to both kernels).
    random_state:
        Seed for reproducible permutations and Nyström.

    Returns
    -------
    KernelTestResult
        ``method="hsic"``.  ``statistic`` is the biased HSIC estimate.
    """
    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    X_2d = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr
    Y_2d = Y_arr.reshape(-1, 1) if Y_arr.ndim == 1 else Y_arr

    n = X_2d.shape[0]

    if kernel_x is None:
        kernel_x = GaussianKernel(sigma="median")
    if kernel_y is None:
        kernel_y = GaussianKernel(sigma="median")

    # Evaluate both kernels once on the full datasets
    K = kernel_x.evaluate(X_2d, max_landmarks=max_landmarks, random_state=random_state)
    L = kernel_y.evaluate(Y_2d, max_landmarks=max_landmarks, random_state=random_state)

    # Doubly-centre both Gram matrices
    K_c = gram_centering(K)
    L_c = gram_centering(L)

    observed = gram_trace_product(K_c, L_c) / (n**2)

    # Null distribution: permute rows of Y (breaks pairing under H₀)
    perm_indices = generate_unique_permutations(
        n, n_permutations, random_state=random_state
    )

    null = np.empty(n_permutations, dtype=float)
    for b, perm in enumerate(perm_indices):
        L_perm = gram_permute(L_c, perm)
        null[b] = gram_trace_product(K_c, L_perm) / (n**2)

    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))

    kernel_name = f"{kernel_x.name()},{kernel_y.name()}"

    return KernelTestResult(
        statistic=observed,
        p_value=p_value,
        null_distribution=null,
        kernel_name=kernel_name,
        n_permutations=n_permutations,
        method="hsic",
    )


# ------------------------------------------------------------------ #
# kernel_regression_test
# ------------------------------------------------------------------ #


def kernel_regression_test(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    kernel_y: Kernel | None = None,
    confounders: np.ndarray | None = None,
    n_permutations: int = 5000,
    max_landmarks: int | None = None,
    random_state: int | None = None,
) -> KernelTestResult:
    """Kernel partial independence test after OLS residualisation.

    Tests H₀: X ⊥ Y | Z, where Z is a matrix of confounders.  The
    procedure residualises both X and Y on Z using ordinary least
    squares, then applies :func:`hsic_test` on the residuals with a
    linear kernel for e_X.

    When *confounders* is ``None``, this reduces to an HSIC test with
    a linear kernel on X.

    Parameters
    ----------
    X:
        Treatment / predictor variable.  Shape ``(n,)`` or ``(n, d_x)``.
    Y:
        Outcome variable.  Shape ``(n,)`` or ``(n, d_y)``.
    kernel_y:
        Kernel for the outcome residuals.  Defaults to
        :class:`~._kernels.GaussianKernel`.
    confounders:
        Confounder matrix.  Shape ``(n, q)``.  When supplied, both X
        and Y are residualised on Z via the OLS hat matrix
        H_Z = Z(ZᵀZ)⁻¹Zᵀ.
    n_permutations:
        Number of permutations.  Default 5000.
    max_landmarks:
        Nyström approximation landmark count.
    random_state:
        Seed for reproducible permutations and Nyström.

    Returns
    -------
    KernelTestResult
        ``method="kernel_regression"``.  ``kernel_name`` reflects the
        kernel used for the Y residuals only (X always uses the linear
        kernel).

    Notes
    -----
    Guarantee: this is a residual-based test.  OLS
    residuals are correlated through the hat matrix and only
    approximately exchangeable, so Type I control is **asymptotically
    exact**, not finite-sample exact.  Moreover, OLS residualisation
    removes only the LINEAR component of confounding: nonlinear
    confounding leaks into the residuals and can produce false
    positives — residual independence e_X ⊥ e_Y is a proxy for
    X ⊥ Y | Z that is exact only under additive linear confounding.
    For nonlinear confounders, use a flexible cross-fitted reduced
    model (``reduced_model=`` arrives with the DML integration) or a
    fully kernelised conditional test (``conditional_hsic_test``,
    v0.5.2).
    """
    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    X_2d = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr
    Y_2d = Y_arr.reshape(-1, 1) if Y_arr.ndim == 1 else Y_arr

    if kernel_y is None:
        kernel_y = GaussianKernel(sigma="median")

    if confounders is not None:
        Z = np.atleast_2d(np.asarray(confounders, dtype=float))
        if Z.shape[0] == 1 and confounders.ndim == 1:
            Z = Z.T
        # OLS hat projection: H_Z = Z pinv(Z); residuals = (I - H_Z) @ A
        Z_pinv = np.linalg.pinv(Z)  # (q, n)
        hat = Z @ Z_pinv  # (n, n)
        e_X = X_2d - hat @ X_2d
        e_Y = Y_2d - hat @ Y_2d
    else:
        e_X = X_2d
        e_Y = Y_2d

    result = hsic_test(
        e_X,
        e_Y,
        kernel_x=LinearKernel(),
        kernel_y=kernel_y,
        n_permutations=n_permutations,
        max_landmarks=max_landmarks,
        random_state=random_state,
    )

    return KernelTestResult(
        statistic=result.statistic,
        p_value=result.p_value,
        null_distribution=result.null_distribution,
        kernel_name=kernel_y.name(),
        n_permutations=n_permutations,
        method="kernel_regression",
    )

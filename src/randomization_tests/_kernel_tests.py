r"""Kernel-based permutation tests for two-sample, independence, and partial independence.

Provides non-parametric hypothesis testing in Reproducing Kernel Hilbert Spaces (RKHS)
using Maximum Mean Discrepancy (MMD), the Hilbert-Schmidt Independence Criterion (HSIC),
and residualized kernel partial regression.

Primary Components
------------------
* :func:`mmd_test`: Maximum Mean Discrepancy two-sample test ($H_0: P = Q$).
* :func:`hsic_test`: Hilbert-Schmidt Independence Criterion test ($H_0: X \perp Y$).
* :func:`kernel_regression_test`: Kernel partial independence test ($H_0: X \perp Y \mid Z$)
  with support for Double/Debiased Machine Learning (DML) non-linear confounding absorption.
* :class:`~._results.KernelTestResult`: Typed frozen dataclass container storing observed
  statistic, permutation p-value, and permuted null distribution.

Mathematical Foundations & Inferential Guarantees
-------------------------------------------------
1. **Maximum Mean Discrepancy (MMD)**:
   Tests $H_0: P = Q$ given samples $X \sim P$ and $Y \sim Q$. Uses the unbiased
   $U$-statistic estimator:

   .. math::
       \mathrm{MMD}^2_u(X, Y) = \frac{1}{n_x(n_x-1)} \sum_{i \neq j} k(x_i, x_j)
                              + \frac{1}{n_y(n_y-1)} \sum_{i \neq j} k(y_i, y_j)
                              - \frac{2}{n_x n_y} \sum_{i, j} k(x_i, y_j)

   [ Guarantee: Finite-Sample Exact ]
   Relabelling the pooled sample is exchangeable under $H_0$, providing finite-sample
   exact Type I error control under label-free kernel bandwidth estimation.

2. **Hilbert-Schmidt Independence Criterion (HSIC)**:
   Tests $H_0: X \perp Y$ given paired observations $(X_i, Y_i)_{i=1}^n$. Uses the
   doubly-centred $V$-statistic:

   .. math::
       \mathrm{HSIC}(X, Y) = \frac{1}{n^2} \mathrm{Tr}(K_c L_c)

   where $K_c = H K H$ and $L_c = H L H$ with centering projection $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$.
   The permutation null distribution has analytical expectation:

   .. math::
       \mathbb{E}_\pi[\mathrm{HSIC}_b] = \frac{\mathrm{Tr}(K_c)\mathrm{Tr}(L_c)}{n^2(n-1)}

   [ Guarantee: Finite-Sample Exact ]

3. **Residualized Kernel Regression**:
   Tests conditional independence $H_0: X \perp Y \mid Z$. When ``reduced_model=None``,
   computes OLS residuals $e_X$ and $e_Y$. When ``reduced_model`` is supplied,
   cross-fits non-linear nuisance functions out-of-fold via Double/Debiased Machine
   Learning (DML; Chernozhukov et al. 2018), eliminating non-linear confounder leakage.
   [ Guarantee: Asymptotically Exact (DML) ]

Cross-Module Integration
------------------------
* Inbound: Public user calls, text analysis in :func:`~._text.text_mmd_test`, and
  knockoff swap diagnostics in :mod:`~._knockoffs`.
* Outbound: Evaluates kernel Gram matrices and Nyström approximations via :mod:`~._kernels`,
  draws permutations via :func:`~.permutations.generate_unique_permutations`, and packages
  into :class:`~._results.KernelTestResult`.

References
----------
Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B., & Smola, A. (2012).
    A kernel two-sample test. *Journal of Machine Learning Research*, 13, 723–773.

Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B., & Smola,
    A. J. (2008). A kernel statistical test of independence. *Advances in
    Neural Information Processing Systems*, 20, 585–592.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.,
    & Robins, J. (2018). Double/debiased machine learning for treatment and
    structural parameters. *The Econometrics Journal*, 21(1), C1–C68.

Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be
    zero: calculating exact p-values when permutations are randomly drawn.
    *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.
"""

from __future__ import annotations

from typing import Any

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
from .families import _cross_fit_residualize
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
    n_randomizations: int = 5000,
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
    n_randomizations:
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
        n, n_randomizations, random_state=random_state
    )

    null = np.empty(n_randomizations, dtype=float)
    for b, perm in enumerate(perm_indices):
        new_x = perm[:n_x]
        new_y = perm[n_x:]
        null[b] = _mmd2_unbiased(K, new_x, new_y)

    # Phipson–Smyth corrected p-value
    p_value = float((np.sum(null >= observed) + 1) / (n_randomizations + 1))

    return KernelTestResult(
        statistic=observed,
        p_value=p_value,
        null_distribution=null,
        kernel_name=kernel.name(),
        n_randomizations=n_randomizations,
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
    n_randomizations: int = 5000,
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
    n_randomizations:
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
        n, n_randomizations, random_state=random_state
    )

    null = np.empty(n_randomizations, dtype=float)
    for b, perm in enumerate(perm_indices):
        L_perm = gram_permute(L_c, perm)
        null[b] = gram_trace_product(K_c, L_perm) / (n**2)

    p_value = float((np.sum(null >= observed) + 1) / (n_randomizations + 1))

    kernel_name = f"{kernel_x.name()},{kernel_y.name()}"

    return KernelTestResult(
        statistic=observed,
        p_value=p_value,
        null_distribution=null,
        kernel_name=kernel_name,
        n_randomizations=n_randomizations,
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
    reduced_model: Any | None = None,
    n_randomizations: int = 5000,
    max_landmarks: int | None = None,
    random_state: int | None = None,
) -> KernelTestResult:
    """Kernel partial independence test after OLS or cross-fitted residualisation.

    Tests H₀: X ⊥ Y | Z, where Z is a matrix of confounders.  When
    *reduced_model* is ``None`` (default), the procedure residualises both X
    and Y on Z using ordinary least squares (OLS), then applies :func:`hsic_test`
    on the residuals with a linear kernel for e_X.

    When *reduced_model* is provided, nonlinear confounding is purged by
    cross-fitting nuisance regressions E[Y|Z] and E[X|Z] out-of-fold via
    :func:`~randomization_tests.families._cross_fit_residualize` (Double Machine
    Learning), eliminating bias from nonlinear confounding.

    When *confounders* is ``None``, this reduces to an unconditional HSIC test
    with a linear kernel on X.

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
        Confounder matrix.  Shape ``(n, q)``.  When supplied without
        *reduced_model*, both X and Y are residualised on Z via the OLS hat
        matrix H_Z = Z(ZᵀZ)⁻¹Zᵀ.
    reduced_model:
        Optional flexible machine learning estimator implementing scikit-learn's
        fit/predict interface. When supplied with *confounders*, both X and Y
        are cross-fit residualised out-of-fold against Z, preventing nonlinear
        confounder leakage.
    n_randomizations:
        Number of randomizations.  Default 5000.
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
    Guarantee: this is a residual-based test.  Residual-based kernel conditional
    independence testing is **asymptotically valid** under consistent estimation
    of E[X|Z] and E[Y|Z]; it is not finite-sample exact.  OLS residualisation
    removes only the linear component of confounding; for nonlinear confounders,
    pass a flexible ``reduced_model`` (e.g. a gradient-boosting or kernel-ridge
    regressor with scikit-learn API) or use a fully kernelised conditional test.
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
        if reduced_model is not None:
            # Cross-fitted DML residualization (Double-Residualized)
            e_Y, e_X = _cross_fit_residualize(
                Z,
                Y_2d,
                X=X_2d,
                reduced_model=reduced_model,
                random_state=random_state,
            )
            e_X = e_X.reshape(-1, 1) if e_X.ndim == 1 else e_X
            e_Y = e_Y.reshape(-1, 1) if e_Y.ndim == 1 else e_Y
        else:
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
        n_randomizations=n_randomizations,
        max_landmarks=max_landmarks,
        random_state=random_state,
    )

    return KernelTestResult(
        statistic=result.statistic,
        p_value=result.p_value,
        null_distribution=result.null_distribution,
        kernel_name=kernel_y.name(),
        n_randomizations=n_randomizations,
        method="kernel_regression",
    )

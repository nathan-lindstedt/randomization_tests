"""Autoregressive (AR) utilities for longitudinal panel data.

Stateless, pure-NumPy/SciPy functions for:

- Yule-Walker AR(p) coefficient estimation pooled across panels
- AR(p) precision-matrix construction and block-diagonal application
- Durbin-Watson and Ljung-Box residual diagnostics

These functions are consumed by family ``calibrate()`` and
``score_project()`` methods to fold working-correlation structure
into the score projection matrix.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.stats


def estimate_ar_coefficients(
    residuals_by_panel: list[np.ndarray],
    order: int,
) -> np.ndarray:
    r"""Estimate AR(p) coefficients via pooled Yule-Walker.

    Pools within-panel autocovariances across all panels (weighted by
    :math:`T_i - p`) and solves the Yule-Walker system using
    :func:`scipy.linalg.solve_toeplitz`.

    For ``order=1`` this reduces to the pooled within-panel lag-1
    autocorrelation:

    .. math::

        \hat{\rho} = \frac{\sum_i \sum_t e_{it} e_{i,t-1}}
                          {\sum_i \sum_t e_{it}^2}

    Parameters
    ----------
    residuals_by_panel : list[np.ndarray]
        Each element is a 1-D array of within-panel residuals sorted by
        time.  Panels shorter than ``order + 1`` are silently skipped.
    order : int
        AR order *p* (must be ≥ 1).

    Returns
    -------
    np.ndarray
        Shape ``(order,)`` — estimated AR coefficients
        :math:`[\hat{\rho}_1, \dots, \hat{\rho}_p]`.
    """
    if order < 1:
        msg = f"order must be >= 1, got {order}"
        raise ValueError(msg)

    # Accumulate autocovariance sums across panels.
    # We need lags 0 .. order.
    gamma = np.zeros(order + 1)
    total_weight = 0.0

    for resid in residuals_by_panel:
        T = len(resid)
        if order + 1 > T:
            continue
        weight = T - order
        total_weight += weight
        # Lag-k autocovariance (not normalised) for k = 0 .. order.
        for k in range(order + 1):
            gamma[k] += np.dot(resid[: T - k], resid[k:T])

    if total_weight == 0.0:
        msg = f"No panels long enough for AR estimation (need T >= {order + 1})"
        raise ValueError(msg)

    # Normalise to autocovariance estimates.
    gamma /= total_weight

    # Solve Toeplitz system: Γ ρ = γ  where Γ_{ij} = γ(|i-j|).
    # First column of the Toeplitz matrix is γ(0), γ(1), …, γ(p-1).
    ar_coefs: np.ndarray = scipy.linalg.solve_toeplitz(
        gamma[:order], gamma[1 : order + 1]
    )
    return ar_coefs


def build_ar_precision_block(
    ar_coefs: np.ndarray,
    T: int,
) -> np.ndarray:
    r"""Build the :math:`T \times T` precision matrix for an AR(p) process.

    For AR(1) with coefficient :math:`\rho`, the precision is the
    tri-diagonal matrix scaled by :math:`1/(1-\rho^2)`:

    .. math::

        \Omega_T^{-1} = \frac{1}{1-\rho^2}
        \begin{pmatrix}
        1        & -\rho   &         &        \\
        -\rho    & 1+\rho^2 & -\rho   &        \\
                 & \ddots  & \ddots  & \ddots \\
                 &         & -\rho   & 1
        \end{pmatrix}

    For AR(p) with :math:`p \ge 2`, the Toeplitz covariance is formed
    and inverted via Cholesky decomposition.

    Parameters
    ----------
    ar_coefs : np.ndarray
        Shape ``(p,)`` — AR coefficients.
    T : int
        Block dimension (panel length).

    Returns
    -------
    np.ndarray
        Shape ``(T, T)`` — precision matrix :math:`\Omega_T^{-1}`.
    """
    p = len(ar_coefs)

    if p == 1:
        rho = ar_coefs[0]
        # The precision of AR(1) with unit innovation variance is the
        # tridiagonal matrix with 1 at corners, (1+ρ²) on the interior
        # diagonal, and −ρ on the off-diagonals — no additional scaling.
        prec = np.zeros((T, T))
        for i in range(T):
            if i == 0 or i == T - 1:
                prec[i, i] = 1.0
            else:
                prec[i, i] = 1.0 + rho**2
            if i > 0:
                prec[i, i - 1] = -rho
                prec[i - 1, i] = -rho
        return prec

    # General AR(p): build Toeplitz covariance, invert via Cholesky.
    # Autocovariances γ(0), γ(1), …, γ(T−1) from the AR coefficients.
    acov = _ar_autocovariance(ar_coefs, T)
    cov = scipy.linalg.toeplitz(acov)
    L = np.linalg.cholesky(cov)
    L_inv = scipy.linalg.solve_triangular(L, np.eye(T), lower=True)
    return L_inv.T @ L_inv  # type: ignore[no-any-return]


def build_ar_cholesky_factor(
    ar_coefs: np.ndarray,
    T: int,
) -> np.ndarray:
    r"""Build the Cholesky factor *L* of the AR(p) precision matrix.

    Returns the lower-triangular *L* such that
    :math:`\Omega_T^{-1} = L^\top L`.  Left-multiplying a vector by
    *L* performs the "whitening" transformation used by FGLS.

    Parameters
    ----------
    ar_coefs : np.ndarray
        Shape ``(p,)`` — AR coefficients.
    T : int
        Block dimension (panel length).

    Returns
    -------
    np.ndarray
        Shape ``(T, T)`` — lower-triangular Cholesky factor *L*.
    """
    # Build Toeplitz covariance → Cholesky → invert to get the
    # whitening factor.  Works for any AR order including AR(1).
    acov = _ar_autocovariance(ar_coefs, T)
    cov = scipy.linalg.toeplitz(acov)
    L_cov = np.linalg.cholesky(cov)
    L_inv = scipy.linalg.solve_triangular(L_cov, np.eye(T), lower=True)
    return L_inv  # type: ignore[no-any-return]


def apply_ar_cholesky_transform(
    v: np.ndarray,
    panel_lengths: np.ndarray,
    ar_coefs: np.ndarray,
) -> np.ndarray:
    r"""Apply the block-diagonal Cholesky whitening :math:`L v`.

    Transforms *v* so that OLS on the transformed data is equivalent
    to GLS on the original — the FGLS (Feasible Generalised Least
    Squares) approach.

    Parameters
    ----------
    v : np.ndarray
        Shape ``(n,)`` or ``(n, k)``.
    panel_lengths : np.ndarray
        Shape ``(n_panels,)`` — length of each panel.
    ar_coefs : np.ndarray
        Shape ``(p,)`` — AR coefficients.

    Returns
    -------
    np.ndarray
        Same shape as *v* — whitened data.
    """
    result = np.empty_like(v)
    chol_cache: dict[int, np.ndarray] = {}
    start = 0

    for length in panel_lengths:
        T = int(length)
        end = start + T

        if T not in chol_cache:
            chol_cache[T] = build_ar_cholesky_factor(ar_coefs, T)

        L = chol_cache[T]
        if v.ndim == 1:
            result[start:end] = L @ v[start:end]
        else:
            result[start:end] = L @ v[start:end]

        start = end

    return result


def apply_ar_precision(
    v: np.ndarray,
    panel_indices: np.ndarray,
    panel_lengths: np.ndarray,
    ar_coefs: np.ndarray,
) -> np.ndarray:
    r"""Compute :math:`\Omega^{-1} v` via block-diagonal application.

    Iterates over panels, applying the per-panel precision block
    :math:`\Omega_{T_i}^{-1}` to the corresponding slice of *v*.
    Avoids materialising the full :math:`n \times n` block-diagonal.

    Parameters
    ----------
    v : np.ndarray
        Shape ``(n,)`` or ``(n, k)`` — vector(s) to transform.
    panel_indices : np.ndarray
        Shape ``(n,)`` — integer panel membership for each observation,
        **assumed to be sorted** so that each panel forms a contiguous
        block.
    panel_lengths : np.ndarray
        Shape ``(n_panels,)`` — length of each panel.
    ar_coefs : np.ndarray
        Shape ``(p,)`` — AR coefficients.

    Returns
    -------
    np.ndarray
        Same shape as *v* — result of :math:`\Omega^{-1} v`.
    """
    result = np.empty_like(v)
    start = 0

    # Cache precision blocks by panel length to avoid redundant builds.
    prec_cache: dict[int, np.ndarray] = {}

    for length in panel_lengths:
        T = int(length)
        end = start + T

        if T not in prec_cache:
            prec_cache[T] = build_ar_precision_block(ar_coefs, T)

        prec = prec_cache[T]
        if v.ndim == 1:
            result[start:end] = prec @ v[start:end]
        else:
            result[start:end] = prec @ v[start:end]

        start = end

    return result


def ar_diagnostics(
    residuals_by_panel: list[np.ndarray],
) -> dict[str, float]:
    """Compute pooled Durbin-Watson and Ljung-Box diagnostics.

    Parameters
    ----------
    residuals_by_panel : list[np.ndarray]
        Each element is a 1-D array of within-panel residuals sorted by
        time.

    Returns
    -------
    dict[str, float]
        ``"durbin_watson"`` — pooled DW statistic (≈ 2.0 under no
        autocorrelation, < 2 for positive, > 2 for negative).
        ``"ljung_box_Q"`` — Ljung-Box Q statistic.
        ``"ljung_box_p"`` — p-value for the Q statistic.
    """
    # --- Pooled Durbin-Watson ---
    num_dw = 0.0
    den_dw = 0.0
    for resid in residuals_by_panel:
        if len(resid) < 2:
            continue
        diff = np.diff(resid)
        num_dw += np.dot(diff, diff)
        den_dw += np.dot(resid, resid)

    durbin_watson = float(num_dw / den_dw) if den_dw > 0 else np.nan

    # --- Pooled Ljung-Box ---
    min_T = min(len(r) for r in residuals_by_panel if len(r) >= 2)
    max_lag = min(10, min_T - 1)

    # Pool autocorrelations across panels.
    n_total = sum(len(r) for r in residuals_by_panel if len(r) >= 2)
    rho_hat = np.zeros(max_lag)
    total_var = 0.0

    for resid in residuals_by_panel:
        if len(resid) < 2:
            continue
        T = len(resid)
        mean_r = np.mean(resid)
        centred = resid - mean_r
        var_panel = np.dot(centred, centred)
        total_var += var_panel
        for k in range(1, max_lag + 1):
            if k < T:
                rho_hat[k - 1] += np.dot(centred[: T - k], centred[k:T])

    if total_var > 0:
        rho_hat /= total_var

    # Q = n_total * (n_total + 2) * Σ ρ̂(k)² / (n_total - k)
    Q = 0.0
    for k in range(1, max_lag + 1):
        Q += rho_hat[k - 1] ** 2 / (n_total - k)
    Q *= n_total * (n_total + 2)

    p_value = float(scipy.stats.chi2.sf(Q, df=max_lag))

    return {
        "durbin_watson": durbin_watson,
        "ljung_box_Q": float(Q),
        "ljung_box_p": p_value,
    }


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _ar_autocovariance(ar_coefs: np.ndarray, n_lags: int) -> np.ndarray:
    """Compute theoretical autocovariances γ(0)…γ(n_lags−1) of an AR(p).

    Uses the Yule-Walker recursion with unit innovation variance.
    """
    p = len(ar_coefs)
    gamma = np.zeros(n_lags)

    # Seed: solve the Yule-Walker system for γ(0)…γ(p−1).
    # Γ ρ = γ  →  γ(k) = Σ_{j=1}^p ρ_j γ(|k−j|) + σ² δ_{k,0}
    # with σ² = 1.  The first p+1 autocovariances satisfy:
    #   γ(0) = Σ ρ_j γ(j) + 1
    #   γ(k) = Σ ρ_j γ(|k−j|)   for k = 1 … p
    # This is a (p+1)×(p+1) linear system.
    A = np.zeros((p + 1, p + 1))
    b = np.zeros(p + 1)
    b[0] = 1.0  # innovation variance

    # Row 0: γ(0) - Σ ρ_j γ(j) = 1
    A[0, 0] = 1.0
    for j in range(p):
        A[0, j + 1] = -ar_coefs[j]

    # Rows 1..p: γ(k) - Σ ρ_j γ(|k-j|) = 0
    for k in range(1, p + 1):
        for col in range(p + 1):
            # coefficient of γ(col) in the equation for γ(k)
            coeff = 0.0
            if col == k:
                coeff += 1.0
            for j in range(p):
                if abs(k - (j + 1)) == col:
                    coeff -= ar_coefs[j]
            A[k, col] = coeff

    seed = np.linalg.solve(A, b)
    gamma[: min(p + 1, n_lags)] = seed[: min(p + 1, n_lags)]

    # Recursion for k > p: γ(k) = Σ_{j=1}^p ρ_j γ(k-j).
    for k in range(p + 1, n_lags):
        gamma[k] = sum(ar_coefs[j] * gamma[k - j - 1] for j in range(p))

    return gamma

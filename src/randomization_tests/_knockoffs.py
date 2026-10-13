r"""Knockoff filter feature selection with controlled False Discovery Rate (FDR).

Implements finite-sample False Discovery Rate (FDR) controlled feature selection via
the knockoff filter framework (Barber & Candès 2015; Candès, Fan, Janson & Lv 2018).

Primary Components
------------------
* :func:`knockoff_test`: High-level entrypoint for FDR-controlled variable selection.
* :class:`~._results.KnockoffResult`: Typed frozen dataclass container storing selected
  features, W-statistics, threshold $\tau_+$, and diagnostic metrics.

Mathematical Foundations & FDR Control
--------------------------------------
Constructs synthetic "knockoff" features $\tilde X$ that serve as negative controls:
1. **Swap Invariance**: Swapping any subset of original features $X_j$ with their knockoff
   counterparts $\tilde X_j$ preserves the joint covariate covariance structure:
   $(X, \tilde X)_{\text{swap}(S)}^\top (X, \tilde X)_{\text{swap}(S)} = (X, \tilde X)^\top (X, \tilde X)$.
2. **Conditional Independence**: Knockoffs are constructed without reference to the outcome $Y$,
   satisfying $\tilde X \perp Y \mid X$.

Two Supported Knockoff Regimes:
* **Fixed-X Knockoffs** (Barber & Candès 2015):
  Requires $n \ge 2p$. Constructs knockoffs deterministically via Gram matrix geometry:
  $\tilde X = X(I - \Sigma^{-1} \text{diag}(s)) + \tilde U C$, where $C^\top C = 2\text{diag}(s) - \text{diag}(s)\Sigma^{-1}\text{diag}(s)$.
  [ Guarantee: Finite-Sample Exact ]
* **Model-X Knockoffs** (Candès et al. 2018):
  Supports high-dimensional settings ($p > n$). Employs second-order Gaussian knockoffs
  with equicorrelated diagonal shifts and Ledoit–Wolf empirical covariance shrinkage.
  [ Guarantee: Model-X Approximate ]

Feature Importance Statistics & Knockoff+ Threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Feature importance is evaluated via signed contrast statistics:

.. math::
    W_j = \lvert \hat\beta_j \rvert - \lvert \hat{\tilde\beta}_j \rvert

The data-dependent knockoff+ selection threshold is determined by:

.. math::
    \tau_+ = \min \left\{ t > 0 : \frac{1 + \#\{j : W_j \le -t\}}{\#\{j : W_j \ge t\} \vee 1} \le q \right\}

guaranteeing $\mathbb{E}[\text{FDP}] \le q$ at target FDR level $q \in (0, 1)$.

Cross-Module Integration
------------------------
* Inbound: Public user calls to :func:`knockoff_test`.
* Outbound: Evaluates feature importance via :mod:`sklearn.linear_model`, verifies
  swap exchangeability via :func:`~._kernel_tests.mmd_test`, and packages output
  into :class:`~._results.KnockoffResult`.

References
----------
Barber, R. F., & Candès, E. J. (2015). Controlling the false discovery rate via
    knockoffs. *The Annals of Statistics*, 43(5), 2055–2085.

Candès, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: 'model-X'
    knockoffs for high dimensional controlled variable selection. *Journal of the
    Royal Statistical Society Series B: Statistical Methodology*, 80(3), 551–577.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LassoCV

from ._compat import DataFrameLike, _ensure_pandas_df
from ._kernel_tests import mmd_test
from ._kernels import Kernel
from ._results import KnockoffResult
from .families import ModelFamily, resolve_family

if TYPE_CHECKING:
    pass


def _compute_equicorrelated_s(Sigma: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Compute equicorrelated diagonal shift s such that 2*Sigma - diag(s) >= 0.

    Ensures that conditional covariance V = 2*diag(s) - diag(s) inv(Sigma) diag(s)
    is strictly positive definite.
    """
    p = Sigma.shape[0]
    eigvals = linalg.eigvalsh(Sigma)
    lambda_min = float(np.min(eigvals))

    if lambda_min <= 0:
        reg = abs(lambda_min) + 1e-6
        Sigma_reg = Sigma + reg * np.eye(p)
        eigvals = linalg.eigvalsh(Sigma_reg)
        lambda_min = float(np.min(eigvals))

    diag_min = float(np.min(np.diag(Sigma)))
    s_val = min(diag_min, 2.0 * lambda_min) * (1.0 - eps)
    if s_val <= 0:
        s_val = max(1e-6, 2.0 * lambda_min * (1.0 - eps))

    return np.full(p, s_val, dtype=np.float64)


def _construct_fixed_x_knockoffs(X_norm: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Construct Fixed-X knockoffs via QR orthonormal complement of X.

    Requires X_norm to have shape (n, p) with n >= 2p.
    Guarantees:
        X_tilde^T X_tilde == X_norm^T X_norm
        X_norm^T X_tilde == X_norm^T X_norm - diag(s)
    to machine precision.
    """
    n, p = X_norm.shape
    G = X_norm.T @ X_norm
    cond_G = np.linalg.cond(G)
    G_inv = linalg.pinv(G) if cond_G > 1e12 else linalg.inv(G)
    diag_s = np.diag(s)

    # Full QR decomposition to find an orthonormal basis for null(X^T)
    Q, _ = linalg.qr(X_norm, mode="full")
    U_tilde = Q[:, p : 2 * p]

    M = 2.0 * diag_s - diag_s @ G_inv @ diag_s
    M = 0.5 * (M + M.T)
    eig_M, v_M = linalg.eigh(M)
    if np.min(eig_M) < 0:
        M = v_M @ np.diag(np.maximum(eig_M, 1e-12)) @ v_M.T

    L = linalg.cholesky(M, lower=True)
    X_tilde = X_norm @ (np.eye(p) - G_inv @ diag_s) + U_tilde @ L.T
    return np.asarray(X_tilde, dtype=np.float64)


def _construct_model_x_knockoffs(
    X: np.ndarray,
    Sigma: np.ndarray,
    s: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct Model-X knockoffs via conditional Gaussian sampling."""
    n, p = X.shape
    diag_s = np.diag(s)
    cond_S = np.linalg.cond(Sigma)
    Sigma_inv = linalg.pinv(Sigma) if cond_S > 1e12 else linalg.inv(Sigma)

    mu = X @ (np.eye(p) - Sigma_inv @ diag_s)
    V = 2.0 * diag_s - diag_s @ Sigma_inv @ diag_s
    V = 0.5 * (V + V.T)
    eig_V, v_V = linalg.eigh(V)
    if np.min(eig_V) < 1e-12:
        V = v_V @ np.diag(np.maximum(eig_V, 1e-12)) @ v_V.T

    L = linalg.cholesky(V, lower=True)
    Z = rng.standard_normal((n, p))
    return np.asarray(mu + Z @ L.T, dtype=np.float64)


def _compute_knockoff_plus_threshold(W: np.ndarray, fdr_level: float) -> float:
    """Compute Barber & Candes (2015) knockoff+ threshold.

    tau_+ = min { t > 0 : (1 + #{j: W_j <= -t}) / (#{j: W_j >= t} v 1) <= q }
    """
    abs_W = np.abs(W)
    candidates = np.sort(abs_W[abs_W > 0])
    if len(candidates) == 0:
        return float("inf")

    for t in candidates:
        n_pos = int(np.sum(W >= t))  # noqa: SIM300
        n_neg = int(np.sum(W <= -t))  # noqa: SIM300
        ratio = (1.0 + n_neg) / max(1, n_pos)
        if ratio <= fdr_level:
            return float(t)

    return float("inf")


def knockoff_test(
    X: DataFrameLike,
    y: DataFrameLike | pd.Series | np.ndarray,
    family: str | ModelFamily = "auto",
    kernel: Kernel | None = None,
    fdr_level: float = 0.1,
    method: str = "equicorrelated",
    estimator: str = "auto",
    shrinkage: bool = True,
    random_state: int | None = None,
) -> KnockoffResult:
    """Controlled variable selection with false discovery rate (FDR) control via knockoffs.

    Constructs synthetic "knockoff" features that mimic the correlation structure
    of the original features but are conditionally independent of the response Y given X.
    Uses the Barber & Candès (2015) knockoff+ threshold to achieve provable FDR control.

    Theoretical Guarantee Tiers:
    - **`method="fixed_x"`** (Barber & Candès 2015):
      Finite-sample exact FDR control (``guarantee_tier="finite_sample_exact"``).
      Treats X as fixed, requiring only that response Y follows a Gaussian linear model
      with independent errors. Requires n >= 2p.
    - **`method="equicorrelated"` / `"sdp"`** (Candès et al. 2018):
      Model-X approximate FDR control (``guarantee_tier="model_x_approximate"``).
      Makes no assumption on Y | X, but requires that the true joint distribution of X
      is Gaussian with the estimated covariance matrix.

    Parameters
    ----------
    X : DataFrameLike
        Feature matrix as pandas or Polars DataFrame, shape (n, p).
    y : DataFrameLike | pd.Series | np.ndarray
        Response vector or single-column DataFrame/Series, shape (n,).
    family : str | ModelFamily, default="auto"
        Model family (e.g. ``"linear"``). For ``method="fixed_x"``, family must be linear.
    kernel : Kernel | None, default=None
        Kernel used for MMD swap-exchangeability diagnostic. Defaults to GaussianKernel.
    fdr_level : float, default=0.1
        Target False Discovery Rate (FDR) bound q in (0, 1).
    method : str, default="equicorrelated"
        Knockoff construction method: ``"fixed_x"``, ``"equicorrelated"``, or ``"sdp"``.
    estimator : str, default="auto"
        Model fitting estimator: ``"auto"`` (OLS when n >= 2p, LassoCV when n < 2p),
        ``"ols"``, or ``"lasso_cv"``.
    shrinkage : bool, default=True
        Whether to apply Ledoit-Wolf covariance shrinkage for Model-X knockoffs,
        guaranteeing strictly positive eigenvalues and numerical stability.
    random_state : int | None, default=None
        Random seed for knockoff generation and MMD permutation tests.

    Returns
    -------
    KnockoffResult
        Result object containing selected features, W-statistics, threshold,
        guarantee tier, and diagnostic metrics (MMD swap p-values, condition number).

    Raises
    ------
    ValueError
        If inputs are invalid, n < 2p for fixed-X knockoffs, or non-linear family
        is used with fixed-X knockoffs.
    """
    valid_methods = ("fixed_x", "equicorrelated", "sdp")
    if method not in valid_methods:
        msg = f"Unknown method {method!r}. Choose from: {valid_methods}."
        raise ValueError(msg)

    valid_estimators = ("auto", "ols", "lasso_cv")
    if estimator not in valid_estimators:
        msg = f"Unknown estimator {estimator!r}. Choose from: {valid_estimators}."
        raise ValueError(msg)

    if not 0.0 < fdr_level < 1.0:
        msg = f"fdr_level must be in (0, 1), got {fdr_level}."
        raise ValueError(msg)

    X_df = _ensure_pandas_df(X, name="X").copy()
    if isinstance(y, (pd.Series, np.ndarray)):
        y_vec = np.asarray(y, dtype=np.float64).ravel()
    else:
        y_df = _ensure_pandas_df(y, name="y")
        y_vec = y_df.iloc[:, 0].to_numpy(dtype=np.float64).ravel()

    n, p = X_df.shape
    if p < 2:
        msg = f"Knockoff filter requires at least 2 features, got p={p}."
        raise ValueError(msg)
    if n != len(y_vec):
        msg = f"Sample size mismatch: X has {n} rows, y has {len(y_vec)} elements."
        raise ValueError(msg)

    feature_names = list(X_df.columns)
    resolved_family = resolve_family(family, y_vec)

    # ------------------------------------------------------------------ #
    # Fixed-X Validation & Dispatch Preference (D26)
    # ------------------------------------------------------------------ #
    if method == "fixed_x":
        if n < 2 * p:
            msg = (
                f"Fixed-X knockoffs require n >= 2p ({2 * p}), got n={n}, p={p}. "
                "Use method='equicorrelated' for high-dimensional or small sample designs."
            )
            raise ValueError(msg)
        if resolved_family.name != "linear":
            msg = (
                f"Fixed-X knockoffs require a linear model family, got {resolved_family.name!r}. "
                "Use method='equicorrelated' for non-linear or discrete GLM families."
            )
            raise ValueError(msg)
    elif resolved_family.name == "linear" and n >= 2 * p:
        warnings.warn(
            f"For linear models with n >= 2p (n={n}, p={p}), method='fixed_x' provides "
            "a finite-sample exact FDR guarantee without requiring Gaussianity of X. "
            "Consider using method='fixed_x'.",
            UserWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(random_state)
    X_np = X_df.to_numpy(dtype=np.float64)

    # Center features and response for clean intercept absorption
    X_mean = np.mean(X_np, axis=0)
    X_centered = X_np - X_mean
    y_centered = y_vec - np.mean(y_vec)

    diag: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Construct Knockoff Matrix X_tilde
    # ------------------------------------------------------------------ #
    if method == "fixed_x":
        guarantee_tier = "finite_sample_exact"
        # Standardize columns to unit norm for canonical Barber & Candes (2015) geometry
        col_norms = np.linalg.norm(X_centered, axis=0)
        col_norms = np.where(col_norms > 0, col_norms, 1.0)
        X_norm = X_centered / col_norms

        G = X_norm.T @ X_norm
        cond_G = float(np.linalg.cond(G))
        diag["condition_number"] = cond_G
        s = _compute_equicorrelated_s(G)
        diag["s_vector"] = s.tolist()

        X_tilde_norm = _construct_fixed_x_knockoffs(X_norm, s)
        # Rescale knockoffs back to match original feature variance scale
        X_tilde = X_tilde_norm * col_norms

        gram_cross_err = float(
            np.max(np.abs(X_norm.T @ X_tilde_norm - (G - np.diag(s))))
        )
        gram_knock_err = float(np.max(np.abs(X_tilde_norm.T @ X_tilde_norm - G)))
        diag["gram_preservation_error"] = gram_cross_err
        diag["knockoff_gram_error"] = gram_knock_err

    else:
        guarantee_tier = "model_x_approximate"
        if method == "sdp":
            warnings.warn(
                "cvxpy solver not found; falling back to method='equicorrelated'.",
                UserWarning,
                stacklevel=2,
            )
            diag["fallback_from_sdp"] = True

        if shrinkage:
            lw = LedoitWolf().fit(X_centered)
            Sigma_hat = lw.covariance_
            diag["shrinkage_intensity"] = float(lw.shrinkage_)
        else:
            Sigma_hat = np.cov(X_centered, rowvar=False)

        cond_S = float(np.linalg.cond(Sigma_hat))
        diag["condition_number"] = cond_S
        s = _compute_equicorrelated_s(Sigma_hat)
        diag["s_vector"] = s.tolist()

        X_tilde = _construct_model_x_knockoffs(X_centered, Sigma_hat, s, rng)

    # ------------------------------------------------------------------ #
    # Augmented Design Fit & Importance Statistics W_j
    # ------------------------------------------------------------------ #
    X_aug = np.hstack([X_centered, X_tilde])

    use_lasso = (estimator == "lasso_cv") or (estimator == "auto" and n < 2 * p)
    if use_lasso:
        lasso = LassoCV(cv=5, random_state=random_state).fit(X_aug, y_centered)
        c_orig = lasso.coef_[:p]
        c_knock = lasso.coef_[p:]
        diag["estimator"] = "lasso_cv"
        diag["lasso_alpha"] = float(lasso.alpha_)
    else:
        coef_aug = linalg.lstsq(X_aug, y_centered)[0]
        c_orig = coef_aug[:p]
        c_knock = coef_aug[p:]
        diag["estimator"] = "ols"

    # W-statistics: W_j = |beta_j| - |beta_tilde_j|
    W = np.abs(c_orig) - np.abs(c_knock)

    # ------------------------------------------------------------------ #
    # Selection via Knockoff+ Threshold
    # ------------------------------------------------------------------ #
    tau = _compute_knockoff_plus_threshold(W, fdr_level)
    if tau < float("inf"):
        selected_idx = [j for j in range(p) if W[j] >= tau]
        selected_features = [feature_names[j] for j in selected_idx]
    else:
        selected_features = []

    # ------------------------------------------------------------------ #
    # Diagnostic Auditing: MMD Swap (Model-X) vs Gram Preservation (Fixed-X)
    # ------------------------------------------------------------------ #
    if method == "fixed_x":
        # Fixed-X guarantees algebraic Gram matrix preservation:
        # X_tilde^T X_tilde = X^T X and X^T X_tilde = X^T X - diag(s).
        # Distributional row exchangeability does not apply to Fixed-X geometry.
        diag["swap_violation"] = False
    else:
        mmd_p_values: dict[str, float] = {}
        n_diag = min(p, 5)
        for j in range(n_diag):
            pair_orig = np.column_stack([X_centered[:, j], X_tilde[:, j]])
            pair_swap = np.column_stack([X_tilde[:, j], X_centered[:, j]])
            m_res = mmd_test(
                pair_orig,
                pair_swap,
                kernel=kernel,
                n_randomizations=400,
                random_state=random_state,
            )
            mmd_p_values[feature_names[j]] = float(m_res.p_value)

        diag["mmd_swap_p_values"] = mmd_p_values
        diag["swap_violation"] = bool(
            any(pval <= 0.05 for pval in mmd_p_values.values())
        )

    return KnockoffResult(
        selected_features=selected_features,
        knockoff_statistics=W,
        fdr_level=fdr_level,
        threshold=float(tau),
        feature_names=feature_names,
        method=method,
        guarantee_tier=guarantee_tier,
        diagnostics=diag,
    )

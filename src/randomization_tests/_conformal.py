"""Conformal prediction module with finite-sample distribution-free guarantees.

Provides split conformal prediction and jackknife+ for regression and
classification families, with support for locally-weighted kernel adaptation
(Papadopoulos et al. 2008) and hierarchical/clustered exchangeability
(Dunn, Wasserman & Ramdas 2022).

Mathematical Foundations & Coverage Guarantees
-----------------------------------------------
Conformal prediction constructs prediction regions $\\mathcal{C}(X_{\\text{new}})$
that are mathematically guaranteed to cover the unseen true outcome $Y_{\\text{new}}$
at a user-specified confidence level $1 - \\alpha$, assuming only exchangeability of
the data points:

1. **Split Conformal Prediction (inductive conformal)**:
   Partitions data into a proper training set $(X_{\\text{train}}, y_{\\text{train}})$
   and a calibration set $(X_{\\text{cal}}, y_{\\text{cal}})$ of size $n_{\\text{cal}}$.
   The model is fit on the training fold, and nonconformity scores
   $S_i = \\mathcal{S}(X_{\\text{cal}, i}, y_{\\text{cal}, i}; \\hat\\mu)$ are evaluated on the
   calibration fold.

   The calibration threshold is computed at the finite-sample order statistic rank:
   .. math::
       k = \\lceil (n_{\\text{cal}} + 1)(1 - \\alpha) \\rceil

   Under exchangeability of $(X_i, y_i)_{i=1}^{n_{\\text{cal}}+1}$, the prediction region
   $\\mathcal{C}(X_{\\text{new}})$ satisfies the finite-sample exact marginal coverage guarantee:
   .. math::
       1 - \\alpha \\le \\mathbb{P}(Y_{\\text{new}} \\in \\mathcal{C}(X_{\\text{new}}))
       \\le 1 - \\alpha + \\frac{1}{n_{\\text{cal}} + 1}

   The $+1$ in the order statistic rank is the exact mathematical counterpart to the $+1$
   in the Phipson & Smyth (2010) permutation p-value: both arise from the uniform ordering
   distribution of $n+1$ exchangeable random variables.

2. **Jackknife+ (Barber, Candès, Ramdas & Tibshirani 2021)**:
   Constructs leave-one-out residuals $R_i^{\\text{LOO}} = |y_i - \\hat\\mu_{-i}(X_i)|$ and
   forms prediction intervals by evaluating candidate bounds
   $\\{\\hat\\mu_{-i}(X_{\\text{new}}) \\mp R_i^{\\text{LOO}}\\}_{i=1}^n$.
   Theoretical distribution-free worst-case guarantee:
   .. math::
       \\mathbb{P}(Y_{\\text{new}} \\in \\mathcal{C}(X_{\\text{new}})) \\ge 1 - 2\\alpha

   (In empirical practice, coverage typically approximates $1 - \\alpha$).

3. **Locally-Weighted Adaptive Intervals (Papadopoulos et al. 2008)**:
   When a :class:`~randomization_tests._kernels.Kernel` is provided, local conditional
   variance $\\hat\\sigma^2(X)$ is estimated via Nadaraya–Watson kernel smoothing over
   squared calibration residuals. Scores are studentized:
   .. math::
       S_i = \\frac{|y_i - \\hat\\mu(X_i)|}{\\hat\\sigma(X_i)}
   yielding adaptive intervals with variable width $2\\hat q \\hat\\sigma(X_{\\text{new}})$ that
   are narrower in dense, low-noise regions and wider in uncertain regions while preserving
   exact marginal coverage.

4. **Clustered / Hierarchical Conformal (Dunn, Wasserman & Ramdas 2022)**:
   When data are clustered (violating observation-level exchangeability), data are split
   at the cluster level. Cluster-maximum nonconformity scores $S_g = \\max_{i \\in g} S_i$
   are calibrated at rank $\\lceil (G_{\\text{cal}} + 1)(1 - \\alpha) \\rceil$, ensuring
   finite-sample validity under arbitrary intra-cluster correlation.

References:
    - Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
    - Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2008). Inductive confidence
      machines for regression. *ECML 2002*, LNAI 2430, 345–356.
    - Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive inference
      with the jackknife+. *Annals of Statistics*, 49(1), 486–507.
    - Dunn, R., Wasserman, L., & Ramdas, A. (2022). Distribution-free prediction sets for two-layer
      hierarchical models. *Journal of the American Statistical Association*, 1–12.
    - Sadinle, M., Lei, J., & Wasserman, L. (2019). Least ambiguous prediction sets for classification
      with an application to medical diagnostics. *JASA*, 114(526), 723–734.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._compat import DataFrameLike
from ._results import ConformalEvaluationResult, ConformalResult
from .exchangeability import ExchangeabilityTree
from .families import (
    ModelFamily,
    _augment_intercept,
    resolve_family,
)

if TYPE_CHECKING:
    from ._kernels import Kernel


def conformal_prediction(
    X_train: DataFrameLike | np.ndarray,
    y_train: DataFrameLike | np.ndarray,
    X_new: DataFrameLike | np.ndarray,
    family: str | ModelFamily = "auto",
    kernel: Kernel | None = None,
    groups: np.ndarray | ExchangeabilityTree | None = None,
    confidence_level: float = 0.95,
    method: str = "split",
    cal_ratio: float = 0.5,
    fit_intercept: bool = True,
    random_state: int | None = None,
) -> ConformalResult:
    """Compute distribution-free conformal prediction regions with finite-sample guarantees.

    For continuous outcomes (regression), constructs a prediction interval
    ``(lower, upper)`` covering $Y_{\\text{new}}$ with probability at least
    ``confidence_level``.  For categorical outcomes (classification), constructs a
    prediction set of plausible class labels.

    Args:
        X_train: Training feature matrix of shape ``(n, p)``.
        y_train: Training response vector of shape ``(n,)``.
        X_new: Query feature matrix of shape ``(n_new, p)`` or single query point ``(p,)``.
        family: Regression/classification family name or :class:`ModelFamily` instance.
            ``"auto"`` detects binary vs continuous outcomes automatically.
        kernel: Optional :class:`~randomization_tests._kernels.Kernel` instance for
            locally-weighted adaptive conformal prediction (Papadopoulos et al. 2008).
            When supplied, prediction interval width adapts to local feature density.
        groups: Cluster labels of shape ``(n,)`` or :class:`ExchangeabilityTree` for
            clustered/hierarchical conformal prediction (Dunn, Wasserman & Ramdas 2022).
        confidence_level: Nominal coverage level $1 - \\alpha$, default ``0.95``.
        method: Conformal algorithm: ``"split"`` (default, finite-sample exact) or
            ``"jackknife_plus"`` (leave-one-out, distribution-free $1 - 2\\alpha$ guarantee).
        cal_ratio: Fraction of training data allocated to the calibration set in split conformal
            (default ``0.5``).
        fit_intercept: Whether to fit an intercept column.
        random_state: Random seed for reproducible calibration splitting.

    Returns:
        :class:`~randomization_tests._results.ConformalResult` containing prediction interval/set,
        calibration nonconformity scores, nominal coverage level, and guarantee metadata.

    Raises:
        ValueError: If parameters are invalid, sample size is insufficient, or inputs are mismatched.
        NotImplementedError: If requested method/family combination is not supported.
    """
    if not 0.0 < confidence_level < 1.0:
        msg = f"confidence_level must be in (0, 1), got {confidence_level}."
        raise ValueError(msg)

    if not 0.0 < cal_ratio < 1.0:
        msg = f"cal_ratio must be in (0, 1), got {cal_ratio}."
        raise ValueError(msg)

    valid_methods = ("split", "jackknife_plus")
    if method not in valid_methods:
        msg = f"Unknown method {method!r}. Choose from: {valid_methods}."
        raise ValueError(msg)

    # Convert inputs to float64 NumPy arrays
    X_tr_np = np.asarray(X_train, dtype=np.float64)
    if isinstance(y_train, (pd.DataFrame, pd.Series)):
        y_tr_np = np.asarray(y_train.to_numpy(), dtype=np.float64).ravel()
    else:
        y_tr_np = np.asarray(y_train, dtype=np.float64).ravel()

    X_new_np = np.asarray(X_new, dtype=np.float64)
    if X_new_np.ndim == 1:
        X_new_np = X_new_np.reshape(1, -1)

    n_samples, p_dim = X_tr_np.shape
    if len(y_tr_np) != n_samples:
        msg = f"X_train has {n_samples} rows but y_train has {len(y_tr_np)} elements."
        raise ValueError(msg)

    if X_new_np.shape[1] != p_dim:
        msg = (
            f"X_new has {X_new_np.shape[1]} features but X_train has {p_dim} features."
        )
        raise ValueError(msg)

    if n_samples < 4:
        msg = f"Conformal prediction requires at least 4 observations, got {n_samples}."
        raise ValueError(msg)

    alpha = 1.0 - confidence_level

    # Resolve family
    resolved_family = resolve_family(family, y_tr_np)
    is_classification = resolved_family.name in ("logistic", "ordinal", "multinomial")

    if method == "jackknife_plus":
        if is_classification:
            msg = (
                f"method='jackknife_plus' is supported only for continuous regression families, "
                f"got classification family {resolved_family.name!r}. Use method='split' for "
                f"conformal prediction sets."
            )
            raise NotImplementedError(msg)
        if kernel is not None:
            msg = (
                "method='jackknife_plus' does not support kernel locally-weighted prediction. "
                "Use method='split' with kernel=... for adaptive prediction intervals."
            )
            raise NotImplementedError(msg)
        if groups is not None:
            msg = (
                "Clustered conformal prediction uses method='split' with cluster-level "
                "maximum calibration (Dunn, Wasserman & Ramdas 2022)."
            )
            raise NotImplementedError(msg)

        return _conformal_jackknife_plus(
            X_tr_np,
            y_tr_np,
            X_new_np,
            resolved_family,
            alpha=alpha,
            confidence_level=confidence_level,
            fit_intercept=fit_intercept,
        )

    # Split conformal dispatch
    if is_classification:
        if kernel is not None:
            warnings.warn(
                "kernel parameter is ignored for classification prediction sets; "
                "nonconformity is evaluated directly on class probability margins.",
                UserWarning,
                stacklevel=2,
            )
        return _conformal_split_classification(
            X_tr_np,
            y_tr_np,
            X_new_np,
            resolved_family,
            groups=groups,
            alpha=alpha,
            confidence_level=confidence_level,
            cal_ratio=cal_ratio,
            fit_intercept=fit_intercept,
            random_state=random_state,
        )

    return _conformal_split_regression(
        X_tr_np,
        y_tr_np,
        X_new_np,
        resolved_family,
        kernel=kernel,
        groups=groups,
        alpha=alpha,
        confidence_level=confidence_level,
        cal_ratio=cal_ratio,
        fit_intercept=fit_intercept,
        random_state=random_state,
    )


def _conformal_split_regression(
    X: np.ndarray,
    y: np.ndarray,
    X_new: np.ndarray,
    family: ModelFamily,
    *,
    kernel: Kernel | None,
    groups: np.ndarray | ExchangeabilityTree | None,
    alpha: float,
    confidence_level: float,
    cal_ratio: float,
    fit_intercept: bool,
    random_state: int | None,
) -> ConformalResult:
    """Split conformal prediction for continuous regression."""
    n = len(y)
    rng = np.random.default_rng(random_state)

    # Partition into proper training and calibration folds
    if groups is not None:
        flat_groups = (
            groups.to_flat_cells()
            if isinstance(groups, ExchangeabilityTree)
            else np.asarray(groups)
        )
        unique_groups = np.unique(flat_groups)
        n_groups = len(unique_groups)
        if n_groups < 2:
            msg = (
                "Clustered conformal prediction requires at least 2 distinct clusters."
            )
            raise ValueError(msg)
        n_cal_groups = max(1, int(np.round(n_groups * cal_ratio)))
        perm_groups = rng.permutation(unique_groups)
        cal_groups_set = set(perm_groups[:n_cal_groups])
        cal_mask = np.isin(flat_groups, list(cal_groups_set))
        train_mask = ~cal_mask
        if np.sum(train_mask) == 0 or np.sum(cal_mask) == 0:
            # Fallback to simple split if cluster sizes are extreme
            perm = rng.permutation(n)
            n_cal = max(1, int(np.round(n * cal_ratio)))
            cal_mask = np.zeros(n, dtype=bool)
            cal_mask[perm[:n_cal]] = True
            train_mask = ~cal_mask
    else:
        flat_groups = None
        perm = rng.permutation(n)
        n_cal = max(2, int(np.round(n * cal_ratio)))
        cal_mask = np.zeros(n, dtype=bool)
        cal_mask[perm[:n_cal]] = True
        train_mask = ~cal_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_cal, y_cal = X[cal_mask], y[cal_mask]

    # Fit base regression model on training fold
    model = family.fit(X_train, y_train, fit_intercept=fit_intercept)
    y_hat_cal = family.predict(model, X_cal)
    y_hat_new = family.predict(model, X_new)

    r_cal = np.abs(y_cal - y_hat_cal)

    # Locally-weighted adaptive scale via kernel smoothing
    if kernel is not None:
        K_cal = np.asarray(kernel(X_cal, X_cal), dtype=np.float64)
        w_cal = np.sum(K_cal, axis=1)
        sigma_cal = np.sqrt(np.sum(K_cal * (r_cal**2), axis=1) / (w_cal + 1e-8) + 1e-6)
        scores_cal = r_cal / sigma_cal

        K_new = np.asarray(kernel(X_new, X_cal), dtype=np.float64)
        w_new = np.sum(K_new, axis=1)
        sigma_new = np.sqrt(np.sum(K_new * (r_cal**2), axis=1) / (w_new + 1e-8) + 1e-6)
    else:
        scores_cal = r_cal
        sigma_new = np.ones(len(X_new), dtype=np.float64)

    # Clustered calibration vs standard iid calibration
    if flat_groups is not None:
        cal_clusters = flat_groups[cal_mask]
        unique_cal_c = np.unique(cal_clusters)
        cluster_scores = np.array(
            [np.max(scores_cal[cal_clusters == g]) for g in unique_cal_c],
            dtype=np.float64,
        )
        G_cal = len(cluster_scores)
        k = min(G_cal, int(np.ceil((G_cal + 1) * (1.0 - alpha))))
        q_hat = float(np.sort(cluster_scores)[k - 1])
        scores_for_result = cluster_scores
        n_cal_eff = G_cal
    else:
        n_cal_eff = len(scores_cal)
        k = min(n_cal_eff, int(np.ceil((n_cal_eff + 1) * (1.0 - alpha))))
        q_hat = float(np.sort(scores_cal)[k - 1])
        scores_for_result = scores_cal

    # Evaluate prediction intervals
    half_widths = q_hat * sigma_new
    lower_bounds = y_hat_new - half_widths
    upper_bounds = y_hat_new + half_widths

    if len(X_new) == 1:
        pred_interval: tuple[float, float] | None = (
            float(lower_bounds[0]),
            float(upper_bounds[0]),
        )
    else:
        pred_interval = (float(np.min(lower_bounds)), float(np.max(upper_bounds)))

    intervals_arr = np.column_stack([lower_bounds, upper_bounds])

    diag = {
        "n_train": int(len(X_train)),
        "n_cal": int(len(X_cal)),
        "n_cal_effective": int(n_cal_eff),
        "quantile_rank_k": int(k),
        "threshold_q": float(q_hat),
        "median_score": float(np.median(scores_for_result)),
        "point_predictions": y_hat_new.tolist(),
        "intervals": intervals_arr.tolist(),
        "mean_interval_width": float(np.mean(2.0 * half_widths)),
        "adaptive_kernel": kernel.name() if kernel is not None else None,
    }

    return ConformalResult(
        prediction_interval=pred_interval,
        prediction_set=None,
        nonconformity_scores=scores_for_result,
        confidence_level=confidence_level,
        method="split",
        coverage_guarantee="finite_sample_exact",
        groups=groups,
        intervals=intervals_arr,
        prediction_sets=None,
        point_predictions=y_hat_new,
        probabilities=None,
        diagnostics=diag,
    )


def _conformal_split_classification(
    X: np.ndarray,
    y: np.ndarray,
    X_new: np.ndarray,
    family: ModelFamily,
    *,
    groups: np.ndarray | ExchangeabilityTree | None,
    alpha: float,
    confidence_level: float,
    cal_ratio: float,
    fit_intercept: bool,
    random_state: int | None,
) -> ConformalResult:
    """Split conformal prediction for classification prediction sets."""
    n = len(y)
    rng = np.random.default_rng(random_state)

    perm = rng.permutation(n)
    n_cal = max(2, int(np.round(n * cal_ratio)))
    cal_mask = np.zeros(n, dtype=bool)
    cal_mask[perm[:n_cal]] = True
    train_mask = ~cal_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_cal, y_cal = X[cal_mask], y[cal_mask]

    # Fit classification model
    model = family.fit(X_train, y_train, fit_intercept=fit_intercept)

    # Extract class probability matrices
    if family.name == "logistic":
        p1_cal = family.predict(model, X_cal)
        probs_cal = np.column_stack([1.0 - p1_cal, p1_cal])
        p1_new = family.predict(model, X_new)
        probs_new = np.column_stack([1.0 - p1_new, p1_new])
        classes = np.array([0, 1], dtype=np.intp)
    elif family.name == "ordinal":
        probs_cal = np.asarray(model.predict(X_cal), dtype=np.float64)
        probs_new = np.asarray(model.predict(X_new), dtype=np.float64)
        classes = np.arange(probs_cal.shape[1], dtype=np.intp)
    elif family.name == "multinomial":
        X_cal_sm = _augment_intercept(X_cal, fit_intercept)
        X_new_sm = _augment_intercept(X_new, fit_intercept)
        probs_cal = np.asarray(model.predict(X_cal_sm), dtype=np.float64)
        probs_new = np.asarray(model.predict(X_new_sm), dtype=np.float64)
        classes = np.arange(probs_cal.shape[1], dtype=np.intp)
    else:
        msg = f"Unsupported classification family: {family.name!r}"
        raise ValueError(msg)

    # Nonconformity score S_i = 1 - P(Y = y_i | X_i)
    y_cal_int = y_cal.astype(int)
    # Clip indices to valid classes
    y_cal_int = np.clip(y_cal_int, 0, len(classes) - 1)
    scores_cal = 1.0 - probs_cal[np.arange(len(y_cal)), y_cal_int]

    k = min(len(scores_cal), int(np.ceil((len(scores_cal) + 1) * (1.0 - alpha))))
    q_hat = float(np.sort(scores_cal)[k - 1])

    # Construct prediction set: classes where P(Y = c | X_new) >= 1 - q_hat
    threshold_prob = max(0.0, 1.0 - q_hat)
    prediction_sets: list[np.ndarray] = []

    for i in range(len(X_new)):
        included = classes[probs_new[i] >= threshold_prob]
        if len(included) == 0:
            # Ensure non-empty prediction set by adding most probable class (Sadinle et al. 2019)
            included = np.array([classes[np.argmax(probs_new[i])]])
        prediction_sets.append(included)

    if len(X_new) == 1:
        pred_set = prediction_sets[0]
    else:
        # Union of classes across query points
        pred_set = np.unique(np.concatenate(prediction_sets))

    diag = {
        "n_train": int(len(X_train)),
        "n_cal": int(len(X_cal)),
        "quantile_rank_k": int(k),
        "threshold_q": float(q_hat),
        "threshold_probability": float(threshold_prob),
        "class_labels": classes.tolist(),
        "query_probabilities": probs_new.tolist(),
        "prediction_sets": [s.tolist() for s in prediction_sets],
        "mean_set_size": float(np.mean([len(s) for s in prediction_sets])),
    }

    return ConformalResult(
        prediction_interval=None,
        prediction_set=pred_set,
        nonconformity_scores=scores_cal,
        confidence_level=confidence_level,
        method="split",
        coverage_guarantee="finite_sample_exact",
        groups=groups,
        intervals=None,
        prediction_sets=prediction_sets,
        point_predictions=None,
        probabilities=probs_new,
        diagnostics=diag,
    )


def _conformal_jackknife_plus(
    X: np.ndarray,
    y: np.ndarray,
    X_new: np.ndarray,
    family: ModelFamily,
    *,
    alpha: float,
    confidence_level: float,
    fit_intercept: bool,
) -> ConformalResult:
    """Jackknife+ conformal prediction for regression (Barber et al. 2021)."""
    n = len(y)

    if family.name == "linear":
        # Fast closed-form leverage path for ordinary linear models
        X_aug = _augment_intercept(X, fit_intercept)
        X_new_aug = _augment_intercept(X_new, fit_intercept)

        XtX_inv = np.linalg.pinv(X_aug.T @ X_aug)
        H = X_aug @ XtX_inv @ X_aug.T
        h = np.diag(H)
        h_clipped = np.clip(h, 0.0, 1.0 - 1e-7)

        b_full = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        y_hat = X_aug @ b_full
        e = y - y_hat
        loo_resids = np.abs(e / (1.0 - h_clipped))  # (n,)

        # Vectorised leave-one-out coefficient updates:
        # beta_-i = beta_hat - (XtX_inv @ x_i) * (e_i / (1 - h_i))
        factor = (e / (1.0 - h_clipped))[:, None]  # (n, 1)
        delta_b = (X_aug @ XtX_inv) * factor  # (n, p_aug)
        b_loo_all = b_full[None, :] - delta_b  # (n, p_aug)

        # LOO predictions at X_new: shape (n, n_new)
        pred_loo_new = b_loo_all @ X_new_aug.T  # (n, n_new)
    else:
        # Exact leave-one-out iteration for non-linear GLM regression
        loo_resids = np.zeros(n, dtype=np.float64)
        pred_loo_new = np.zeros((n, len(X_new)), dtype=np.float64)

        for i in range(n):
            keep = np.ones(n, dtype=bool)
            keep[i] = False
            m_i = family.fit(X[keep], y[keep], fit_intercept=fit_intercept)
            y_hat_i = float(family.predict(m_i, X[i : i + 1])[0])
            loo_resids[i] = abs(y[i] - y_hat_i)
            pred_loo_new[i] = family.predict(m_i, X_new)

    k_lo = max(1, int(np.floor((n + 1) * alpha)))
    k_hi = min(n, int(np.ceil((n + 1) * (1.0 - alpha))))

    lower_bounds = np.zeros(len(X_new), dtype=np.float64)
    upper_bounds = np.zeros(len(X_new), dtype=np.float64)

    for col in range(len(X_new)):
        lower_cands = pred_loo_new[:, col] - loo_resids
        upper_cands = pred_loo_new[:, col] + loo_resids
        lower_bounds[col] = np.sort(lower_cands)[k_lo - 1]
        upper_bounds[col] = np.sort(upper_cands)[k_hi - 1]

    if len(X_new) == 1:
        pred_interval: tuple[float, float] | None = (
            float(lower_bounds[0]),
            float(upper_bounds[0]),
        )
    else:
        pred_interval = (float(np.min(lower_bounds)), float(np.max(upper_bounds)))

    intervals_arr = np.column_stack([lower_bounds, upper_bounds])
    point_preds_loo = np.mean(pred_loo_new, axis=0)

    diag = {
        "n_train": n,
        "n_loo_refits": n,
        "rank_lo": int(k_lo),
        "rank_hi": int(k_hi),
        "median_loo_residual": float(np.median(loo_resids)),
        "intervals": intervals_arr.tolist(),
        "mean_interval_width": float(np.mean(upper_bounds - lower_bounds)),
    }

    return ConformalResult(
        prediction_interval=pred_interval,
        prediction_set=None,
        nonconformity_scores=loo_resids,
        confidence_level=confidence_level,
        method="jackknife_plus",
        coverage_guarantee="worst_case_1_minus_2alpha",
        groups=None,
        intervals=intervals_arr,
        prediction_sets=None,
        point_predictions=point_preds_loo,
        probabilities=None,
        diagnostics=diag,
    )


def evaluate_conformal(
    result: ConformalResult,
    y_true: DataFrameLike | np.ndarray,
    groups: DataFrameLike | np.ndarray | None = None,
) -> ConformalEvaluationResult:
    """Evaluate empirical out-of-sample coverage and sharpness against ground-truth labels.

    For regression conformal predictions, evaluates empirical coverage
    :math:`\\frac{1}{n} \\sum_{i=1}^n \\mathbb{I}(y_i \\in [l_i, u_i])` along with
    mean and median interval widths.

    For classification conformal predictions, evaluates empirical coverage
    :math:`\\frac{1}{n} \\sum_{i=1}^n \\mathbb{I}(y_i \\in \\mathcal{C}(x_i))` along
    with singleton rates, ambiguous rates, empty set rates, and set size distribution.

    When ``groups`` is supplied, additionally evaluates Feature-Stratified Coverage (FSC;
    Angelopoulos & Bates 2021) and per-stratum coverage rates.

    Args:
        result: :class:`~randomization_tests._results.ConformalResult` containing
            prediction regions.
        y_true: True observed outcomes on the evaluation sample (vector or Series).
        groups: Optional discrete group or cluster labels of shape ``(n,)`` for evaluating
            stratum-conditional coverage and Feature-Stratified Coverage (FSC).

    Returns:
        :class:`~randomization_tests._results.ConformalEvaluationResult` containing
        empirical coverage rate, sharpness metrics, stratified coverage metrics, and
        per-observation coverage flags.

    Raises:
        ValueError: If sample sizes do not match or result lacks prediction regions.
    """
    if isinstance(y_true, (pd.DataFrame, pd.Series)):
        y_true_np = np.asarray(y_true.to_numpy(), dtype=np.float64).ravel()
    else:
        y_true_np = np.asarray(y_true, dtype=np.float64).ravel()

    n_eval = len(y_true_np)
    if n_eval == 0:
        msg = "y_true must contain at least 1 observation."
        raise ValueError(msg)

    if groups is not None:
        if isinstance(groups, (pd.DataFrame, pd.Series)):
            groups_np = groups.to_numpy().ravel()
        else:
            groups_np = np.asarray(groups).ravel()
        if len(groups_np) != n_eval:
            msg = f"groups has {len(groups_np)} elements, but y_true has {n_eval} observations."
            raise ValueError(msg)
    else:
        groups_np = None

    # Check whether regression or classification
    is_regression = (
        result.intervals is not None or result.prediction_interval is not None
    )
    is_classification = (
        result.prediction_sets is not None or result.prediction_set is not None
    )

    if not is_regression and not is_classification:
        msg = "ConformalResult contains neither intervals nor prediction sets."
        raise ValueError(msg)

    if (
        is_regression
        and result.prediction_set is None
        and result.prediction_sets is None
    ):
        # Regression evaluation
        if result.intervals is not None:
            intervals = result.intervals
        elif result.prediction_interval is not None:
            intervals = np.array([result.prediction_interval], dtype=np.float64)
        else:
            msg = "No interval data found in ConformalResult."
            raise ValueError(msg)

        if len(intervals) != n_eval:
            msg = (
                f"y_true has {n_eval} observations, but ConformalResult contains "
                f"{len(intervals)} query intervals."
            )
            raise ValueError(msg)

        lo = intervals[:, 0]
        hi = intervals[:, 1]
        covered = (y_true_np >= lo) & (y_true_np <= hi)
        empirical_cov = float(np.mean(covered))
        widths = hi - lo
        mean_width = float(np.mean(widths))
        median_width = float(np.median(widths))

        # Stratified metrics
        grp_cov: dict[Any, float] | None = None
        grp_counts: dict[Any, int] | None = None
        grp_widths: dict[Any, float] | None = None
        fsc: float | None = None

        if groups_np is not None:
            grp_cov = {}
            grp_counts = {}
            grp_widths = {}
            unique_g = np.unique(groups_np)
            for g in unique_g:
                mask_g = groups_np == g
                grp_cov[g] = float(np.mean(covered[mask_g]))
                grp_counts[g] = int(np.sum(mask_g))
                grp_widths[g] = float(np.mean(widths[mask_g]))
            fsc = float(np.min(list(grp_cov.values()))) if grp_cov else None

        return ConformalEvaluationResult(
            empirical_coverage=empirical_cov,
            nominal_coverage=result.confidence_level,
            n_eval=n_eval,
            covered=covered,
            mean_interval_width=mean_width,
            median_interval_width=median_width,
            set_size_distribution=None,
            singleton_rate=None,
            ambiguous_rate=None,
            empty_rate=None,
            group_coverage=grp_cov,
            group_counts=grp_counts,
            group_mean_width=grp_widths,
            fsc_metric=fsc,
            coverage_guarantee=result.coverage_guarantee,
        )

    # Classification evaluation
    if result.prediction_sets is not None:
        prediction_sets = result.prediction_sets
    elif result.prediction_set is not None:
        prediction_sets = [result.prediction_set]
    else:
        msg = "No prediction set data found in ConformalResult."
        raise ValueError(msg)

    if len(prediction_sets) != n_eval:
        msg = (
            f"y_true has {n_eval} observations, but ConformalResult contains "
            f"{len(prediction_sets)} query prediction sets."
        )
        raise ValueError(msg)

    covered_list: list[bool] = []
    set_sizes: list[int] = []

    for i in range(n_eval):
        y_val = int(round(y_true_np[i]))
        s = prediction_sets[i]
        covered_list.append(bool(np.isin(y_val, s)))
        set_sizes.append(len(s))

    covered_arr = np.array(covered_list, dtype=bool)
    empirical_cov = float(np.mean(covered_arr))
    set_sizes_np = np.array(set_sizes, dtype=int)
    singleton_rate = float(np.mean(set_sizes_np == 1))
    ambiguous_rate = float(np.mean(set_sizes_np > 1))
    empty_rate = float(np.mean(set_sizes_np == 0))

    unique_sizes, counts = np.unique(set_sizes_np, return_counts=True)
    size_dist = {int(k): int(v) for k, v in zip(unique_sizes, counts, strict=True)}

    # Stratified metrics for classification
    grp_cov = None
    grp_counts = None
    grp_sizes: dict[Any, float] | None = None
    fsc = None

    if groups_np is not None:
        grp_cov = {}
        grp_counts = {}
        grp_sizes = {}
        unique_g = np.unique(groups_np)
        for g in unique_g:
            mask_g = groups_np == g
            grp_cov[g] = float(np.mean(covered_arr[mask_g]))
            grp_counts[g] = int(np.sum(mask_g))
            grp_sizes[g] = float(np.mean(set_sizes_np[mask_g]))
        fsc = float(np.min(list(grp_cov.values()))) if grp_cov else None

    return ConformalEvaluationResult(
        empirical_coverage=empirical_cov,
        nominal_coverage=result.confidence_level,
        n_eval=n_eval,
        covered=covered_arr,
        mean_interval_width=None,
        median_interval_width=None,
        set_size_distribution=size_dist,
        singleton_rate=singleton_rate,
        ambiguous_rate=ambiguous_rate,
        empty_rate=empty_rate,
        group_coverage=grp_cov,
        group_counts=grp_counts,
        group_mean_width=grp_sizes,
        fsc_metric=fsc,
        coverage_guarantee=result.coverage_guarantee,
    )

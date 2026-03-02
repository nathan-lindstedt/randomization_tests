"""Confounder sieve via correlation screening, mediation, moderation & collider analysis.

Controlling for the right variables is critical in observational studies.
The confounder sieve identifies true confounders by systematically
pulling out variables that play other causal roles — **colliders**,
**mediators**, and **moderators** — from the set of candidates that
pass dual-correlation screening.

Causal roles:

* **Confounder** (X ← Z → Y): causes both X and Y, creating a
  spurious association.  Should be controlled for.
* **Mediator** (X → M → Y): lies on the causal pathway.  Controlling
  for it removes part of the *real* effect.
* **Moderator**: changes the *strength* of the X → Y relationship
  (interaction effect).  Informational — does not invalidate the
  Kennedy conditioning set.
* **Collider** (X → Z ← Y): both X and Y cause Z.  Conditioning on a
  collider **creates** spurious association — must *not* be controlled.

Workflow:
    1. :func:`screen_potential_confounders` – find variables correlated
       with both X and Y.  Supports Pearson, partial, and distance
       correlation with optional multiple-testing correction.
    2. :func:`identify_confounders` – orchestrates the four-stage sieve
       (collider → mediator → moderator → confounder), returning a
       :class:`~randomization_tests._results.ConfounderAnalysisResult`.

The sieve is an **exploratory** tool for data-driven confounder
selection.  For guaranteed Type I error control, specify
``confounders=`` based on domain knowledge or a pre-registered
analysis plan.  The permutation p-value is exact conditional on the
selected conditioning set.

References:
    Baron, R. M. & Kenny, D. A. (1986). The moderator–mediator
    variable distinction. *Journal of Personality and Social
    Psychology*, 51(6), 1173–1182.

    Preacher, K. J. & Hayes, A. F. (2004). SPSS and SAS procedures for
    estimating indirect effects in simple mediation models.  *Behavior
    Research Methods*, 36(4), 717–731.

    Preacher, K. J. & Hayes, A. F. (2008). Asymptotic and resampling
    strategies for assessing and comparing indirect effects in multiple
    mediator models.  *Behavior Research Methods*, 40(3), 879–891.

    Efron, B. (1987). Better bootstrap confidence intervals.  *Journal
    of the American Statistical Association*, 82(397), 171–185.

    Székely, G. J. & Rizzo, M. L. (2007). Measuring and testing
    dependence by correlation of distances.  *The Annals of
    Statistics*, 35(6), 2769–2794.

    Székely, G. J. & Rizzo, M. L. (2013). The distance correlation
    t-test of independence in high dimension.  *Journal of
    Multivariate Analysis*, 117, 193–213.

    VanderWeele, T. J. & Ding, P. (2017). Sensitivity analysis in
    observational research: introducing the E-value.  *Annals of
    Internal Medicine*, 167(4), 268–274.

    Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2008).
    Bootstrap-based improvements for inference with clustered errors.
    *Review of Economics and Statistics*, 90(3), 414–427.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from ._compat import DataFrameLike, _ensure_pandas_df
from .families import (
    ModelFamily,
    _augment_intercept,
    _suppress_sm_warnings,
    resolve_family,
)

if TYPE_CHECKING:
    from ._results import ConfounderAnalysisResult

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Family fallback for the confounder sieve
# ------------------------------------------------------------------ #
#
# The confounder sieve classifies causal roles (confounder, mediator,
# moderator, collider) based on structural relationships in the DAG,
# not distributional details.  Two categories of families need
# remapping before the sieve can call ``fam.fit()``:
#
# 1. **Mixed families** (LinearMixedFamily, etc.) carry random-effects
#    structure that is irrelevant for mediation/moderation decomposition
#    of fixed-effect pathways.  Resolving to the base family prevents
#    ``fam.fit()`` from requiring group labels that the confounder
#    module does not manage.
#
# 2. **Calibration-required families** (NegativeBinomialFamily) need a
#    nuisance parameter (dispersion α) estimated via ``calibrate()``
#    before ``fit()`` works.  Falling back to the closest
#    calibration-free family avoids that requirement.  Whether path
#    coefficients are estimated via NB2 or Poisson does not change
#    whether a variable is a confounder vs. mediator — that is a
#    structural question about the DAG.

_MIXED_TO_BASE: dict[str, str] = {
    "linear_mixed": "linear",
    "logistic_mixed": "logistic",
    "poisson_mixed": "poisson",
}

_CALIBRATION_FALLBACK: dict[str, str] = {
    "negative_binomial": "poisson",
}


def _resolve_base_family(
    fam: ModelFamily,
) -> tuple[ModelFamily, bool]:
    """Resolve a family that the sieve cannot use directly.

    Handles two cases:

    1. Mixed families → base families (drops random-effects structure).
    2. Calibration-required families → calibration-free counterparts.

    Returns:
        Tuple of (resolved_family, did_fallback).  *did_fallback* is
        ``True`` when a remapping occurred so callers can log it.
    """
    base_name = _MIXED_TO_BASE.get(fam.name)
    if base_name is not None:
        logger.debug(
            "Falling back from mixed family %r to base family %r "
            "(mediation/moderation operates on fixed-effect pathways only)",
            fam.name,
            base_name,
        )
        return resolve_family(base_name), True

    cal_name = _CALIBRATION_FALLBACK.get(fam.name)
    if cal_name is not None:
        logger.debug(
            "Falling back from %r to %r "
            "(confounder sieve does not require calibration-dependent "
            "link function; structural DAG classification is invariant)",
            fam.name,
            cal_name,
        )
        return resolve_family(cal_name), True

    return fam, False


# ------------------------------------------------------------------ #
# Cluster bootstrap helper
# ------------------------------------------------------------------ #


def _cluster_bootstrap_indices(
    groups: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    *,
    mode: str = "cluster",
) -> list[np.ndarray]:
    """Generate bootstrap index arrays that respect group structure.

    Two modes are available:

    ``mode="cluster"`` (whole-cluster resampling)
        Resample *G* cluster IDs with replacement, then concatenate
        all observations from the selected clusters.  This is the
        classic cluster bootstrap (Cameron, Gelbach & Miller, 2008)
        and is appropriate when cluster-level variance estimation is
        the goal.  Produces **ragged** index arrays when clusters are
        unbalanced — callers must iterate sequentially rather than
        batching with ``batch_fit_paired()``.

    ``mode="stratified"`` (within-cluster resampling)
        For each cluster *g*, resample *n_g* observation indices with
        replacement **within** that cluster.  Every replicate has
        exactly *n* observations, producing a rectangular ``(B, n)``
        index array that is directly compatible with
        ``batch_fit_paired()``.  Preserves hierarchical structure
        while enabling JAX / NumPy vectorised fitting.  Appropriate
        for the confounder sieve, where the question is "is this path
        coefficient nonzero?" rather than "what is the between-cluster
        variance?".

    Args:
        groups: 1-D array of group labels, length *n*.
        n_bootstrap: Number of bootstrap replicates.
        rng: NumPy random generator.
        mode: Resampling strategy — ``"cluster"`` (default) or
            ``"stratified"``.  See above.

    Returns:
        List of *n_bootstrap* 1-D index arrays.  Each array
        contains the observation indices for one replicate.
        For ``mode="stratified"`` all arrays have length *n*;
        for ``mode="cluster"`` lengths may vary when clusters
        are unbalanced.
    """
    unique_labels = np.unique(groups)
    n_groups = len(unique_labels)
    # Pre-compute index sets for each group label.
    group_indices: dict[object, np.ndarray] = {
        label: np.where(groups == label)[0] for label in unique_labels
    }
    indices: list[np.ndarray] = []

    if mode == "stratified":
        # Within-cluster resampling: resample n_g observations with
        # replacement inside each cluster, then concatenate.  Every
        # replicate has exactly n observations → rectangular output.
        for _ in range(n_bootstrap):
            parts: list[np.ndarray] = []
            for label in unique_labels:
                g_idx = group_indices[label]
                parts.append(rng.choice(g_idx, size=len(g_idx), replace=True))
            indices.append(np.concatenate(parts))
    elif mode == "cluster":
        # Whole-cluster resampling: resample G cluster IDs with
        # replacement, concatenate all observations from the selected
        # clusters.  Ragged when clusters are unbalanced.
        for _ in range(n_bootstrap):
            chosen = rng.choice(unique_labels, size=n_groups, replace=True)
            indices.append(np.concatenate([group_indices[lab] for lab in chosen]))
    else:
        raise ValueError(
            f"Unknown cluster bootstrap mode {mode!r}. "
            "Expected 'cluster' or 'stratified'."
        )

    return indices


def _cluster_jackknife_indices(
    groups: np.ndarray,
) -> list[np.ndarray]:
    """Generate leave-one-cluster-out index arrays.

    Returns *G* index arrays (one per group), each containing all
    observations except those in the left-out group.
    """
    unique_labels = np.unique(groups)
    all_idx = np.arange(len(groups))
    return [all_idx[groups != label] for label in unique_labels]


# ------------------------------------------------------------------ #
# Partial correlation helper
# ------------------------------------------------------------------ #


def _partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray,
) -> tuple[float, float]:
    """Partial correlation between *x* and *y* controlling for *covariates*.

    Regresses both *x* and *y* on *covariates* via OLS, then computes
    Pearson *r* between the residuals.

    The p-value uses df = n − k − 2 (not scipy's built-in n − 2),
    which is the correct degrees of freedom for residualised data
    where *k* covariates have been partialled out.

    .. note::

        P-values assume homoscedastic residuals.  For binary or count
        outcomes, p-values are approximate.  Use
        ``correlation_method='distance'`` for a distribution-free
        alternative.

    Args:
        x: 1-D array, shape ``(n,)``.
        y: 1-D array, shape ``(n,)``.
        covariates: 2-D array, shape ``(n, k)``.

    Returns:
        ``(partial_r, p_value)`` tuple.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    covariates = np.atleast_2d(covariates)
    if covariates.shape[0] == 1 and covariates.shape[1] == len(x):
        covariates = covariates.T  # fix (1, n) -> (n, 1)
    n = len(x)
    k = covariates.shape[1]

    # Add intercept column for OLS.
    C = _augment_intercept(covariates)

    # Residualise x and y.
    res_x = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    res_y = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]

    # Pearson r between residuals.
    r_val: float = float(np.corrcoef(res_x, res_y)[0, 1])

    # Manual t-test with corrected df = n - k - 2.
    df = n - k - 2
    if df <= 0 or abs(r_val) >= 1.0:
        return r_val, 0.0 if abs(r_val) >= 1.0 else 1.0
    t_stat = r_val * np.sqrt(df / (1.0 - r_val**2))
    p_val: float = float(2.0 * stats.t.sf(abs(t_stat), df))
    return r_val, p_val


# ------------------------------------------------------------------ #
# Distance correlation helper
# ------------------------------------------------------------------ #


def _distance_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Bias-corrected distance correlation with asymptotic t-test.

    Implements the Székely & Rizzo (2007) doubly-centered distance
    matrix formula with the bias-corrected estimator from Székely &
    Rizzo (2013).  The p-value uses the asymptotic t-test with
    df = n(n − 3)/2 − 1, avoiding an expensive permutation loop.

    Complexity is O(n²) in memory and time due to the pairwise
    distance matrices.

    Args:
        x: 1-D array, shape ``(n,)``.
        y: 1-D array, shape ``(n,)``.

    Returns:
        ``(dcor, p_value)`` tuple.  *dcor* is clamped to 0.0 when the
        bias-corrected estimator is negative (returns p = 1.0).

    Raises:
        UserWarning: When n > 10,000 (O(n²) memory).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)

    if n > 10_000:
        warnings.warn(
            f"Distance correlation with n={n:,} requires O(n²) memory "
            f"({n * n * 8 / 1e9:.1f} GB for each distance matrix). "
            "Consider down-sampling or using correlation_method='pearson'.",
            UserWarning,
            stacklevel=2,
        )

    # Pairwise Euclidean distance matrices.
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])

    # Bias-corrected doubly-centered distances (Székely & Rizzo 2013).
    # For a distance matrix d, the U-centered version is:
    #   Ã_ij = a_ij - (1/(n-2)) * sum_k a_ik - (1/(n-2)) * sum_l a_lj
    #          + (1/((n-1)(n-2))) * sum_{k,l} a_kl  for i != j
    #   Ã_ii = 0  (diagonal is set to zero)
    a_row = a.sum(axis=1)
    a_total = a_row.sum()
    b_row = b.sum(axis=1)
    b_total = b_row.sum()

    # U-centered matrices.
    A = (
        a
        - a_row[:, None] / (n - 2)
        - a_row[None, :] / (n - 2)
        + a_total / ((n - 1) * (n - 2))
    )
    B = (
        b
        - b_row[:, None] / (n - 2)
        - b_row[None, :] / (n - 2)
        + b_total / ((n - 1) * (n - 2))
    )
    np.fill_diagonal(A, 0.0)
    np.fill_diagonal(B, 0.0)

    # Bias-corrected covariance and variances.
    factor = 1.0 / (n * (n - 3))
    dcov2 = factor * (A * B).sum()
    dvar_x = factor * (A * A).sum()
    dvar_y = factor * (B * B).sum()

    # Distance correlation.
    denom = np.sqrt(dvar_x * dvar_y)
    if denom < 1e-15 or dcov2 < 0:
        return 0.0, 1.0

    dcor_sq = dcov2 / denom
    if dcor_sq < 0:
        return 0.0, 1.0
    dcor = float(np.sqrt(max(dcor_sq, 0.0)))

    # Asymptotic t-test (Székely & Rizzo 2013, Theorem 3).
    M = n * (n - 3) / 2
    df = M - 1
    if df <= 0:
        return dcor, 1.0
    t_stat = np.sqrt(df) * dcov2 / np.sqrt(max(dvar_x * dvar_y - dcov2**2, 1e-300))
    p_val = float(stats.t.sf(abs(t_stat), df) * 2)
    return dcor, p_val


# ------------------------------------------------------------------ #
# Step 1 – Correlation screening (Pearson r)
# ------------------------------------------------------------------ #
#
# The Pearson correlation coefficient measures the linear association
# between two variables:
#
#   r(Z, X) = Σ_i (z_i - z̄)(x_i - x̄)
#             ─────────────────────────────────────
#             √[Σ_i (z_i - z̄)²] √[Σ_i (x_i - x̄)²]
#
# r ranges from -1 (perfect negative linear relationship) to +1
# (perfect positive linear relationship).  A candidate confounder Z
# must be linearly associated with BOTH the predictor X and the
# outcome Y beyond minimum thresholds of |r| and p-value.
#
# This is a necessary condition for confounding (Z must be associated
# with both X and Y), but not sufficient — correlation alone cannot
# distinguish a confounder (X ← Z → Y) from a mediator (X → Z → Y).
# Step 2 (mediation analysis) resolves this ambiguity.


def screen_potential_confounders(
    X: DataFrameLike,
    y: DataFrameLike,
    predictor: str,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
    correlation_method: str = "pearson",
    correction_method: str | None = None,
) -> dict:
    """Screen for variables correlated with both predictor and outcome.

    A potential confounder *Z* satisfies ``|r(Z, X)| >= threshold``
    **and** ``|r(Z, Y)| >= threshold``, both with ``p < p_value_threshold``.

    Args:
        X: Feature matrix.  Accepts pandas or Polars DataFrames.
        y: Target variable.  Accepts pandas or Polars DataFrames.
        predictor: Name of the predictor of interest.
        correlation_threshold: Minimum absolute Pearson *r* (or
            distance correlation) to flag a variable.
        p_value_threshold: Maximum p-value for significance.
        correlation_method: ``"pearson"`` (default), ``"partial"``,
            or ``"distance"``.  When ``"partial"``, the Z-Y leg
            computes ``partial_r(Z, Y | X)`` (partials out only the
            predictor to avoid masking co-occurring confounders),
            while Z-X keeps marginal Pearson *r*.  When ``"distance"``,
            Székely & Rizzo's bias-corrected distance correlation is
            used for both legs.
        correction_method: ``None`` (default), ``"holm"``, or
            ``"fdr_bh"``.  Multiple-testing correction applied
            **per-leg** (not pooled) via
            :func:`statsmodels.stats.multitest.multipletests`.
            With *K* candidates, the *K* Z-X p-values are corrected
            together and the *K* Z-Y p-values separately, then the
            conjunction (both adjusted p-values < threshold) decides.

    Returns:
        Dictionary with keys ``predictor``, ``potential_confounders``,
        ``correlations_with_predictor``, ``correlations_with_outcome``,
        ``excluded_variables``, ``correlation_method``,
        ``correction_method``, and ``adjusted_p_values``.
    """
    if correlation_method not in ("pearson", "partial", "distance"):
        raise ValueError(
            f"correlation_method must be 'pearson', 'partial', or 'distance', "
            f"got {correlation_method!r}"
        )
    if correction_method is not None and correction_method not in ("holm", "fdr_bh"):
        raise ValueError(
            f"correction_method must be None, 'holm', or 'fdr_bh', "
            f"got {correction_method!r}"
        )

    X = _ensure_pandas_df(X, name="X")
    y = _ensure_pandas_df(y, name="y")

    y_values = np.ravel(y)
    other_features = [c for c in X.columns if c != predictor]
    predictor_values = X[predictor].values

    # --- First pass: compute raw correlations and p-values ---
    raw_pred: list[tuple[float, float]] = []  # (r, p) for Z-X
    raw_out: list[tuple[float, float]] = []  # (r, p) for Z-Y

    for feature in other_features:
        fv = X[feature].values

        # Z-X leg: always marginal Pearson r (even for "partial").
        if correlation_method == "distance":
            corr_pred, p_pred = _distance_correlation(fv, predictor_values)
        else:
            corr_pred, p_pred = stats.pearsonr(fv, predictor_values)

        # Z-Y leg: partial or distance depending on method.
        if correlation_method == "partial":
            corr_out, p_out = _partial_correlation(
                fv, y_values, predictor_values.reshape(-1, 1)
            )
        elif correlation_method == "distance":
            corr_out, p_out = _distance_correlation(fv, y_values)
        else:
            corr_out, p_out = stats.pearsonr(fv, y_values)

        raw_pred.append((float(corr_pred), float(p_pred)))
        raw_out.append((float(corr_out), float(p_out)))

    # --- Multiple-testing correction (per-leg) ---
    adjusted_p_pred = [p for _, p in raw_pred]
    adjusted_p_out = [p for _, p in raw_out]
    adj_p_pred_dict: dict[str, float] = {}
    adj_p_out_dict: dict[str, float] = {}

    if correction_method is not None and len(other_features) > 0:
        from statsmodels.stats.multitest import multipletests

        _, adj_pred_arr, _, _ = multipletests(
            adjusted_p_pred, alpha=p_value_threshold, method=correction_method
        )
        _, adj_out_arr, _, _ = multipletests(
            adjusted_p_out, alpha=p_value_threshold, method=correction_method
        )
        adjusted_p_pred = list(adj_pred_arr)
        adjusted_p_out = list(adj_out_arr)

    for i, feature in enumerate(other_features):
        adj_p_pred_dict[feature] = adjusted_p_pred[i]
        adj_p_out_dict[feature] = adjusted_p_out[i]

    # --- Second pass: apply thresholds ---
    potential_confounders: list[str] = []
    correlations_with_predictor: dict = {}
    correlations_with_outcome: dict = {}
    excluded_variables: list[str] = []

    for i, feature in enumerate(other_features):
        corr_pred, _ = raw_pred[i]
        corr_out, _ = raw_out[i]
        p_pred = adjusted_p_pred[i]
        p_out = adjusted_p_out[i]

        sig_pred = (abs(corr_pred) >= correlation_threshold) and (
            p_pred < p_value_threshold
        )
        sig_out = (abs(corr_out) >= correlation_threshold) and (
            p_out < p_value_threshold
        )

        if sig_pred and sig_out:
            potential_confounders.append(feature)
            correlations_with_predictor[feature] = {
                "r": corr_pred,
                "p": raw_pred[i][1],
                "p_adjusted": p_pred,
            }
            correlations_with_outcome[feature] = {
                "r": corr_out,
                "p": raw_out[i][1],
                "p_adjusted": p_out,
            }
        else:
            excluded_variables.append(feature)

    return {
        "predictor": predictor,
        "potential_confounders": potential_confounders,
        "correlations_with_predictor": correlations_with_predictor,
        "correlations_with_outcome": correlations_with_outcome,
        "excluded_variables": excluded_variables,
        "correlation_method": correlation_method,
        "correction_method": correction_method,
        "adjusted_p_values": {
            "predictor": adj_p_pred_dict,
            "outcome": adj_p_out_dict,
        },
    }


# ------------------------------------------------------------------ #
# Collider detection
# ------------------------------------------------------------------ #
#
# A **collider** Z is caused by both X and Y: X → Z ← Y.
# Conditioning on a collider *creates* a spurious association between
# X and Y (Berkson's paradox).  This is the most dangerous
# misclassification — hence colliders are tested first in the sieve.
#
# Detection relies on a two-part test:
#   (1) **Significance**: Z has a significant relationship with Y
#       after controlling for X.
#   (2) **Amplification**: conditioning on Z *increases* the X-Y
#       association magnitude (|partial| > |marginal|).
#
# Attenuation (|partial| < |marginal|) → confounder or mediator
# (distinguished later by the mediation test).  Only amplification
# signals a collider.
#
# **Suppressor pre-filtering**: true suppressors (correlated with X
# but not Y) fail the Z-Y dual-threshold screen in Step 4 and never
# reach the collider test, mitigating the conditioning-test
# conflation with suppressors.
#
# **Suppressive confounder limitation**: a confounder whose confounding
# direction *opposes* the true causal direction produces amplification
# indistinguishable from a collider signal.  This miss is conservative
# — biases toward the null, not away from it.  Full causal direction
# testing (LiNGAM/ANM) deferred to v0.7.0.


def _collider_test(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    family: ModelFamily | None = None,
    *,
    n_perm_collider: int = 200,
    random_state: int | None = None,
) -> tuple[bool, float, float]:
    """Family-aware collider detection (significance + amplification).

    Args:
        x: Predictor values, 1-D array.
        y: Outcome values, 1-D array.
        z: Candidate collider values, 1-D array.
        family: Model family.  ``None`` or linear → OLS; GLM families
            use ``fam.fit()``/``fam.coefs()``.  Multinomial is
            rejected (returns ``(False, NaN, NaN)``).
        n_perm_collider: Permutations for the non-collapsibility guard
            (GLM families only).  Default 200.
        random_state: Seed for the permutation test.

    Returns:
        ``(is_collider, coef_marginal, coef_partial)`` tuple.  For
        GLMs, coefficients are on the link scale.  For linear, they
        are Pearson *r* values.

    Notes:
        **Suppressor pre-filtering**: true suppressors (correlated
        with X but not Y) are caught by the dual-threshold screen
        and never reach this function.

        **Suppressive confounder limitation**: a confounder whose
        confounding direction opposes the true causal direction
        produces amplification indistinguishable from a collider
        signal.  Such variables get classified as colliders and
        removed from the confounder pool, when they should stay in.
        This miss is conservative: failing to control for a
        suppressive confounder biases X → Y toward the null
        (underestimation, not false positive).  Full causal direction
        testing (LiNGAM / ANM) is deferred to v0.7.0.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    # --- Guard: near-constant Z ---
    if np.var(z) < 1e-10:
        return False, np.nan, np.nan

    # --- Guard: multinomial family ---
    is_glm = family is not None and family.name != "linear"
    if family is not None and family.name == "multinomial":
        return False, np.nan, np.nan

    n = len(x)

    if is_glm:
        assert family is not None  # for type narrowing
        # Marginal coefficient: Y ~ X
        x_2d = x.reshape(-1, 1)
        model_marginal = family.fit(x_2d, y, fit_intercept=True)
        coef_marginal = float(family.coefs(model_marginal)[0])

        # Partial coefficient: Y ~ X + Z
        xz = np.column_stack([x, z])
        model_partial = family.fit(xz, y, fit_intercept=True)
        coef_partial = float(family.coefs(model_partial)[0])

        # Significance test: is beta_Z significant?
        beta_z = float(family.coefs(model_partial)[1])

        # Wald z-test for beta_Z significance.
        # Estimate se(beta_Z) via bootstrap (small B=100, just for
        # significance decision).
        try:
            # Try to get p-value from the model summary if available.
            p_z = float(model_partial.pvalues[2])  # intercept, X, Z
        except (AttributeError, IndexError, TypeError):
            # Fallback: Wald z-test from coefficient/se estimate.
            try:
                se_z = float(model_partial.bse[2])
                z_stat = beta_z / se_z
                p_z = float(2.0 * stats.norm.sf(abs(z_stat)))
            except (AttributeError, IndexError, TypeError):
                # If we can't get standard errors, use a liberal threshold.
                p_z = 0.01 if abs(beta_z) > 1e-6 else 1.0

        if p_z >= 0.05:
            return False, coef_marginal, coef_partial

        # Amplification check with permutation-calibrated threshold.
        abs_marginal = abs(coef_marginal)
        abs_partial = abs(coef_partial)

        if abs_marginal < 1e-15:
            # Cannot compute ratio when marginal coefficient is zero.
            return False, coef_marginal, coef_partial

        observed_delta = (abs_partial - abs_marginal) / abs_marginal

        if observed_delta <= 0:
            # No amplification — not a collider.
            return False, coef_marginal, coef_partial

        # Permutation null: permute Z to break the X→Z←Y structure.
        # Any inflation under permuted Z is pure non-collapsibility.
        rng = np.random.default_rng(random_state)

        _can_batch = hasattr(family, "batch_fit_varying_X")
        if _can_batch:
            # --- Vectorised path: single batch_fit_varying_X call ---
            B = n_perm_collider
            z_perms = np.stack([rng.permutation(z) for _ in range(B)])  # (B, n)
            x_tiled = np.broadcast_to(x.reshape(1, -1), (B, n))  # (B, n)
            X_batch = np.stack([x_tiled, z_perms], axis=-1)  # (B, n, 2)
            coefs_perm = family.batch_fit_varying_X(
                X_batch, y, fit_intercept=True
            )  # (B, p)
            abs_perms = np.abs(coefs_perm[:, 0])  # x-coef
            null_deltas = (abs_perms - abs_marginal) / abs_marginal
        else:
            # --- Sequential fallback ---
            null_deltas = np.empty(n_perm_collider)
            for i in range(n_perm_collider):
                z_perm = rng.permutation(z)
                xz_perm = np.column_stack([x, z_perm])
                model_perm = family.fit(xz_perm, y, fit_intercept=True)
                coef_perm = float(family.coefs(model_perm)[0])
                abs_perm = abs(coef_perm)
                null_deltas[i] = (abs_perm - abs_marginal) / abs_marginal

        threshold = float(np.percentile(null_deltas, 95))
        is_collider = observed_delta > threshold
        return is_collider, coef_marginal, coef_partial

    else:
        # Linear: OLS t-test + Pearson r amplification comparison.

        # Significance: t-test on beta_Z in Y ~ X + Z.
        ones = np.ones((n, 1))
        design = np.column_stack([ones, x, z])
        beta, residuals, _, _ = np.linalg.lstsq(design, y, rcond=None)
        beta_z = beta[2]

        # Estimate residual standard error.
        if len(residuals) > 0:
            rss = float(residuals[0])
        else:
            rss = float(np.sum((y - design @ beta) ** 2))
        df = n - 3  # intercept + x + z
        if df <= 0:
            return False, np.nan, np.nan
        mse = rss / df
        # Standard error of beta_z via (X'X)^{-1}.
        try:
            cov_beta = mse * np.linalg.inv(design.T @ design)
            se_z = np.sqrt(cov_beta[2, 2])
        except np.linalg.LinAlgError:
            return False, np.nan, np.nan
        t_stat = beta_z / se_z
        p_z = float(2.0 * stats.t.sf(abs(t_stat), df))

        if p_z >= 0.05:
            r_marginal = float(np.corrcoef(x, y)[0, 1])
            return False, r_marginal, r_marginal  # no amplification

        # Amplification: compare |r(X,Y)| vs |r(X,Y|Z)|.
        r_marginal, _ = stats.pearsonr(x, y)
        r_partial, _ = _partial_correlation(x, y, z.reshape(-1, 1))
        r_marginal = float(r_marginal)
        r_partial = float(r_partial)

        is_collider = abs(r_partial) > abs(r_marginal)
        return is_collider, r_marginal, r_partial


# ------------------------------------------------------------------ #
# Step 2 – Mediation analysis (Preacher & Hayes bootstrap)
# ------------------------------------------------------------------ #
#
# Mediation analysis decomposes the relationship between a predictor X
# and an outcome Y through a candidate mediator M into three pathways:
#
#   Total effect (c path):    Y = c·X + e₁
#     The unconditional effect of X on Y ignoring M.
#
#   a path:                   M = a·X + e₂
#     The effect of X on the candidate mediator.
#
#   b path + direct (c′):    Y = c′·X + b·M + e₃
#     The direct effect of X on Y (c′) after controlling for M, and
#     the effect of M on Y (b) after controlling for X.
#
# The **indirect effect** is the product a·b — the portion of X's
# influence on Y that is transmitted through M.  The total effect
# decomposes as:
#
#   c = c′ + a·b
#       ↑     ↑
#     direct  indirect
#
# Baron & Kenny (1986) proposed a four-step "causal steps" approach
# that required the total effect c to be significant as a prerequisite.
# Preacher & Hayes (2004, 2008) showed this is unnecessary — the
# bootstrap confidence interval of the indirect effect (a·b) is the
# sole criterion.  A significant total effect is neither necessary
# (inconsistent mediation can suppress c) nor sufficient for mediation.
#
# The bootstrap works as follows:
#   1. Resample the data with replacement B times (typically B >= 5000).
#   2. For each resample, compute a* and b* via OLS and form a*·b*.
#   3. Build the empirical distribution of the B indirect effects.
#   4. Construct a confidence interval.  If it excludes zero, the
#      indirect effect is statistically significant → mediation.
#
# We use bias-corrected and accelerated (BCa) intervals (Efron, 1987)
# rather than simple percentile intervals.  BCa adjusts for two
# problems:
#   • **Bias** – the bootstrap distribution's median may not equal the
#     point estimate (measured by z₀, the normal quantile of the
#     proportion of bootstrap values below the observed statistic).
#   • **Skewness** – the bootstrap distribution may be asymmetric
#     (measured by the acceleration â, estimated via jackknife).
#
# Together, z₀ and â shift the percentile cutoffs so the resulting
# interval has better coverage properties than unadjusted percentiles.


def mediation_analysis(
    X: DataFrameLike,
    y: DataFrameLike,
    predictor: str,
    mediator: str,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    precision: int = 4,
    random_state: int | None = None,
    family: str | ModelFamily = "auto",
    groups: np.ndarray | None = None,
) -> dict:
    """Preacher & Hayes (2004, 2008) bootstrap test of the indirect effect.

    Decomposes the total effect of *predictor* on *y* into a direct
    effect (c′) and an indirect effect through *mediator* (a × b).
    The indirect effect is tested via bias-corrected and accelerated
    (BCa) bootstrap confidence intervals, which correct for both
    median bias and skewness in the bootstrap distribution (Efron,
    1987).  Unlike the Baron & Kenny (1986) causal-steps approach,
    this method does **not** require the total effect to be significant
    as a prerequisite — the bootstrap CI of the indirect effect is the
    sole criterion for mediation.

    The bootstrap is **vectorised**: all *n_bootstrap* index arrays are
    pre-generated and the a-path / b-path regressions are batched via
    :func:`numpy.linalg.lstsq`.  When *groups* is provided, the
    bootstrap resamples entire clusters to preserve within-cluster
    correlation (Cameron, Gelbach & Miller, 2008).

    .. note::

        The ``proportion_mediated`` ratio ``indirect / total`` relies
        on the decomposition $c = c' + ab$, which holds exactly for
        linear models but is biased for non-linear models due to
        non-collapsibility.  The ``is_mediator`` binary decision
        (CI excludes zero) is unaffected — only the proportion
        scalar is approximate for GLMs.

    Args:
        X: Feature matrix.  Accepts pandas or Polars DataFrames.
        y: Target variable.  Accepts pandas or Polars DataFrames.
        predictor: Predictor (X in X → M → Y).
        mediator: Potential mediator (M).
        n_bootstrap: Number of bootstrap samples.  Preacher & Hayes
            recommend ≥ 5 000 for BCa intervals.
        confidence_level: Confidence-interval level.
        precision: Decimal places for rounding.
        random_state: Seed for reproducibility.
        family: Outcome family.  Mixed families are automatically
            resolved to their base family (mediation/moderation
            decompose fixed-effect pathways).
        groups: Optional 1-D array of group labels for cluster
            bootstrap.  Should match the group labels used for
            mixed-model estimation.

    Returns:
        Dictionary containing the mediation decomposition, BCa
        bootstrap CI, and a textual interpretation.

    References:
        Preacher, K. J. & Hayes, A. F. (2004). SPSS and SAS procedures
        for estimating indirect effects in simple mediation models.
        *Behavior Research Methods*, 36(4), 717–731.

        Preacher, K. J. & Hayes, A. F. (2008). Asymptotic and
        resampling strategies for assessing and comparing indirect
        effects in multiple mediator models. *Behavior Research
        Methods*, 40(3), 879–891.

        Efron, B. (1987). Better bootstrap confidence intervals.
        *Journal of the American Statistical Association*, 82(397),
        171–185.
    """
    X = _ensure_pandas_df(X, name="X")
    y = _ensure_pandas_df(y, name="y")

    y_values = np.ravel(y)
    x_vals = X[predictor].values.reshape(-1, 1)
    m_vals = X[mediator].values.reshape(-1, 1)
    n = len(y_values)

    # Resolve the outcome family.  Mixed families are resolved to
    # their base family — mediation operates on fixed-effect pathways.
    fam = resolve_family(family, y_values)
    fam, _was_mixed = _resolve_base_family(fam)
    _use_family = fam.name != "linear"
    logger.debug(
        "mediation_analysis: resolved family=%r, use_family=%s",
        fam.name,
        _use_family,
    )

    # Cluster bootstrap support.
    _use_cluster = groups is not None
    if _use_cluster:
        groups = np.asarray(groups).ravel()
        if len(groups) != n:
            raise ValueError(
                f"groups length ({len(groups)}) must match sample size ({n})"
            )

    # --- Observed paths (point estimates from the full sample) ---

    # Step 1: Total effect (c path) — regress Y on X alone.
    # This captures X's entire influence on Y, both direct and
    # any portion routed through M.
    #   Y = c·X + e₁
    if _use_family:
        model_total_f = fam.fit(x_vals, y_values.astype(float), fit_intercept=True)
        c_total = float(fam.coefs(model_total_f)[0])
    else:
        model_total = LinearRegression().fit(x_vals, y_values)
        c_total = model_total.coef_[0]

    # Step 2: a path — regress M on X.
    # Measures how much a one-unit change in X shifts M.
    # Always linear OLS — the mediator is continuous by assumption.
    #   M = a·X + e₂
    model_a = LinearRegression().fit(x_vals, m_vals.ravel())
    a_path = model_a.coef_[0]

    # Step 3: b path + direct effect (c′) — regress Y on X and M jointly.
    # After partialling out M, the coefficient on X is the direct
    # effect c′.  The coefficient on M is the b path: the effect of M
    # on Y holding X constant.
    #   Y = c′·X + b·M + e₃
    xm = np.hstack([x_vals, m_vals])
    if _use_family:
        model_full_f = fam.fit(xm, y_values.astype(float), fit_intercept=True)
        full_coefs = fam.coefs(model_full_f)
        c_prime = float(full_coefs[0])  # direct effect
        b_path = float(full_coefs[1])  # M → Y controlling for X
    else:
        model_full = LinearRegression().fit(xm, y_values)
        c_prime = model_full.coef_[0]  # direct effect
        b_path = model_full.coef_[1]  # M → Y controlling for X

    # The indirect effect: a·b
    # This is the product of X→M and M→Y|X, representing the portion
    # of X's influence on Y that is channelled through M.
    indirect_effect = a_path * b_path

    # --- Bootstrap for the indirect effect (a*b) ---
    #
    # For each of B bootstrap iterations we:
    #   1. Resample n observations WITH replacement (or whole clusters).
    #   2. Recompute the a path (M on X) and b path (Y on X+M).
    #   3. Record the product a*·b* as one draw from the bootstrap
    #      distribution of the indirect effect.
    rng = np.random.default_rng(random_state)

    if _use_cluster:
        assert groups is not None
        # Stratified within-cluster resampling produces rectangular
        # (B, n) index arrays, enabling batch fitting via
        # batch_fit_paired().  See _cluster_bootstrap_indices() for
        # the distinction between "cluster" and "stratified" modes.
        boot_idx_list = _cluster_bootstrap_indices(
            groups, n_bootstrap, rng, mode="stratified"
        )
    else:
        boot_idx_list = [
            rng.choice(n, size=n, replace=True) for _ in range(n_bootstrap)
        ]

    # Build the design matrix with an explicit intercept column for lstsq.
    ones = np.ones((n, 1))
    X_a_design = np.hstack([ones, x_vals])  # (n, 2)

    # --- Compute bootstrap indirect effects ---
    #
    # When the family supports ``batch_fit_paired()`` (e.g. ordinal,
    # logistic, Poisson via JAX vmap), the b-path regressions are
    # batched into a single vectorised call — typically 30-50× faster
    # than sequential ``fam.fit()`` loops.
    _can_batch = _use_family and hasattr(fam, "batch_fit_paired")
    boot_idx_arr = np.array(boot_idx_list)  # (B, n) — always rectangular

    if _can_batch and boot_idx_arr.ndim == 2:
        # --- Vectorised path: batch a-path + batch b-path -----------
        # a-path: vectorised OLS via np.linalg.lstsq on stacked matrices
        Xa_batch = X_a_design[boot_idx_arr]  # (B, n, 2)
        M_batch = m_vals[boot_idx_arr].squeeze(-1)  # (B, n)
        # Solve all a-paths at once: (B, n, 2) @ (2,) -> need per-row lstsq
        # Use normal equations: a = (X'X)^{-1} X' M for each replicate
        XtX = np.einsum("bij,bik->bjk", Xa_batch, Xa_batch)  # (B, 2, 2)
        XtM = np.einsum("bij,bi->bj", Xa_batch, M_batch)  # (B, 2)
        # solve expects (B, 2, 2) @ (B, 2, 1) → (B, 2, 1)
        a_stars = np.linalg.solve(XtX, XtM[..., np.newaxis]).squeeze(-1)[:, 1]

        # b-path: batch_fit_paired — all resampled (X, y) at once
        Xb_batch = np.stack(
            [
                np.column_stack([x_vals[idx].ravel(), m_vals[idx].ravel()])
                for idx in boot_idx_list
            ]
        )  # (B, n, 2)
        Yb_batch = y_values[boot_idx_arr].astype(float)  # (B, n)

        with _suppress_sm_warnings(hessian=True):
            b_coefs = fam.batch_fit_paired(
                Xb_batch, Yb_batch, fit_intercept=True
            )  # (B, 2)
        b_stars = b_coefs[:, 1]  # M→Y coefficient

        bootstrap_indirect = a_stars * b_stars
    else:
        # --- Sequential fallback (linear family or ragged clusters) -
        bootstrap_indirect = np.empty(n_bootstrap)
        for b_i, idx in enumerate(boot_idx_list):
            # a-path: M ~ X (always linear OLS)
            Xa_b = X_a_design[idx]
            M_b = m_vals[idx].ravel()
            coef_a, _, _, _ = np.linalg.lstsq(Xa_b, M_b, rcond=None)
            a_star = coef_a[1]

            # b-path: Y ~ X + M
            if _use_family:
                xm_b = np.column_stack([x_vals[idx].ravel(), m_vals[idx].ravel()])
                y_b = y_values[idx].astype(float)
                try:
                    model_b = fam.fit(xm_b, y_b, fit_intercept=True)
                    b_star = float(fam.coefs(model_b)[1])
                except Exception:
                    b_star = np.nan
            else:
                Xm_b = np.column_stack(
                    [np.ones(len(idx)), x_vals[idx].ravel(), m_vals[idx].ravel()]
                )
                y_b = y_values[idx]
                coef_b, _, _, _ = np.linalg.lstsq(Xm_b, y_b, rcond=None)
                b_star = coef_b[2]

            bootstrap_indirect[b_i] = a_star * b_star

    # --- BCa confidence interval (Efron, 1987) ---
    # The BCa interval adjusts the percentile endpoints of the bootstrap
    # distribution to correct for bias and skewness, yielding better
    # coverage than simple percentile or normal-approximation intervals.
    # See the _bca_ci helper for the mathematical details.
    ci_lower, ci_upper = _bca_ci(
        bootstrap_indirect,
        indirect_effect,
        n,
        x_vals,
        m_vals,
        y_values,
        confidence_level,
        family=fam if _use_family else None,
        groups=groups,
    )

    # Decision criterion (Preacher & Hayes):
    # If the BCa CI for the indirect effect (a·b) does NOT contain
    # zero, the indirect effect is statistically significant at the
    # chosen confidence level → the candidate is a mediator.
    # Unlike Baron & Kenny's causal-steps approach, we do NOT require
    # the total effect (c) to be significant.
    is_mediator = (ci_lower > 0) or (ci_upper < 0)

    if abs(c_total) > 1e-10:
        proportion_mediated = indirect_effect / c_total
    else:
        proportion_mediated = np.nan

    if is_mediator:
        if abs(c_prime) < abs(c_total) * 0.1:
            interpretation = (
                f"'{mediator}' fully mediates the effect of '{predictor}' "
                f"on the outcome (BCa CI excludes zero; direct effect "
                f"< 10% of total). Do not control for it as a confounder."
            )
        else:
            interpretation = (
                f"'{mediator}' partially mediates the effect of "
                f"'{predictor}' on the outcome (BCa CI excludes zero). "
                f"Consider whether to control for it based on the "
                f"research question."
            )
    else:
        interpretation = (
            f"'{mediator}' is not a significant mediator (BCa CI "
            f"includes zero). It may be a confounder if correlated "
            f"with both '{predictor}' and the outcome. Consider "
            f"controlling for it."
        )

    return {
        "predictor": predictor,
        "mediator": mediator,
        "total_effect": np.round(c_total, precision),
        "direct_effect": np.round(c_prime, precision),
        "indirect_effect": np.round(indirect_effect, precision),
        "a_path": np.round(a_path, precision),
        "b_path": np.round(b_path, precision),
        "indirect_effect_ci": (
            np.round(ci_lower, precision),
            np.round(ci_upper, precision),
        ),
        "ci_method": "BCa",
        "proportion_mediated": (
            np.round(proportion_mediated, precision)
            if not np.isnan(proportion_mediated)
            else np.nan
        ),
        "is_mediator": is_mediator,
        "interpretation": interpretation,
    }


def _bca_ci(
    bootstrap_dist: np.ndarray,
    observed_stat: float,
    n: int,
    x_vals: np.ndarray,
    m_vals: np.ndarray,
    y_values: np.ndarray,
    confidence_level: float,
    *,
    family: ModelFamily | None = None,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap CI for mediation.

    Computes the mediation-specific jackknife (a-path × b-path) and
    delegates the generic BCa percentile adjustment to
    :func:`diagnostics._bca_percentile`.  When *groups* is provided,
    the jackknife is leave-one-cluster-out instead of
    leave-one-observation-out.

    Args:
        bootstrap_dist: Array of bootstrap indirect-effect replicates.
        observed_stat: Observed indirect effect (a × b).
        n: Sample size.
        x_vals: Predictor values, shape ``(n, 1)``.
        m_vals: Mediator values, shape ``(n, 1)``.
        y_values: Outcome values, shape ``(n,)``.
        confidence_level: Nominal coverage (e.g. 0.95).
        family: Optional non-linear family for the b-path.
        groups: Optional group labels for cluster jackknife.

    Returns:
        ``(ci_lower, ci_upper)`` tuple.
    """
    from .diagnostics import _bca_percentile

    alpha = 1 - confidence_level

    # --- Jackknife: leave-one-out or leave-one-cluster-out ---
    if groups is not None:
        jack_idx_list = _cluster_jackknife_indices(groups)
    else:
        jack_idx_list = [
            np.concatenate([np.arange(i), np.arange(i + 1, n)]) for i in range(n)
        ]

    n_jack = len(jack_idx_list)
    ones_full = np.ones((n, 1))
    X_a_full = np.hstack([ones_full, x_vals])  # (n, 2)

    # Batch the jackknife fits when the family supports batch_fit_paired.
    _can_batch = family is not None and hasattr(family, "batch_fit_paired")
    jack_idx_arr = np.array(jack_idx_list) if groups is None else None

    if _can_batch and jack_idx_arr is not None and jack_idx_arr.ndim == 2:
        # --- Vectorised jackknife: batch a-path + batch b-path ------

        # a-path: vectorised normal equations
        Xa_jack = X_a_full[jack_idx_arr]  # (J, n-1, 2)
        M_jack = m_vals[jack_idx_arr].squeeze(-1)  # (J, n-1)
        XtX = np.einsum("bij,bik->bjk", Xa_jack, Xa_jack)  # (J, 2, 2)
        XtM = np.einsum("bij,bi->bj", Xa_jack, M_jack)  # (J, 2)
        a_jacks = np.linalg.solve(XtX, XtM[..., np.newaxis]).squeeze(-1)[:, 1]  # (J,)

        # b-path: batch_fit_paired
        Xb_jack = np.stack(
            [
                np.column_stack([x_vals[jidx].ravel(), m_vals[jidx].ravel()])
                for jidx in jack_idx_list
            ]
        )  # (J, n-1, 2)
        Yb_jack = y_values[jack_idx_arr].astype(float)  # (J, n-1)

        with _suppress_sm_warnings(hessian=True):
            assert family is not None  # narrowed by _can_batch
            b_coefs = family.batch_fit_paired(
                Xb_jack, Yb_jack, fit_intercept=True
            )  # (J, 2)
        b_jacks = b_coefs[:, 1]

        jackknife_indirect = a_jacks * b_jacks
    else:
        # --- Sequential fallback ------------------------------------
        jackknife_indirect = np.empty(n_jack)
        for j, jidx in enumerate(jack_idx_list):
            # a-path: always linear OLS
            Xa_j = X_a_full[jidx]
            M_j = m_vals[jidx].ravel()
            coef_a, _, _, _ = np.linalg.lstsq(Xa_j, M_j, rcond=None)
            a_j = coef_a[1]

            # b-path: family-appropriate model when non-linear
            if family is not None:
                xm_j = np.column_stack([x_vals[jidx].ravel(), m_vals[jidx].ravel()])
                y_j = y_values[jidx].astype(float)
                try:
                    model_j = family.fit(xm_j, y_j, fit_intercept=True)
                    b_j = float(family.coefs(model_j)[1])
                except Exception:
                    b_j = np.nan
            else:
                Xm_j = np.column_stack(
                    [np.ones(len(jidx)), x_vals[jidx].ravel(), m_vals[jidx].ravel()]
                )
                y_j = y_values[jidx]
                coef_b, _, _, _ = np.linalg.lstsq(Xm_j, y_j, rcond=None)
                b_j = coef_b[2]

            jackknife_indirect[j] = a_j * b_j

    return _bca_percentile(bootstrap_dist, observed_stat, jackknife_indirect, alpha)


# ------------------------------------------------------------------ #
# Moderation analysis
# ------------------------------------------------------------------ #
#
# A **moderator** Z changes the *strength* of the X → Y relationship
# (interaction effect).  The statistical test fits:
#
#   Y = β₁·X_c + β₂·Z_c + β₃·(X_c × Z_c) + ε
#
# where X_c and Z_c are mean-centered (to reduce multicollinearity).
# If the BCa CI for β₃ (the interaction coefficient) excludes zero,
# Z is a moderator.
#
# Moderator labeling is **non-exclusive**: a variable can be both
# moderator AND confounder.  The moderator flag is informational —
# it tells the user to consider adding X × Z as a predictor, but
# the variable stays in the confounder list for `confounders=`.


def moderation_analysis(
    X: DataFrameLike,
    y: DataFrameLike,
    predictor: str,
    moderator: str,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    precision: int = 4,
    random_state: int | None = None,
    family: str | ModelFamily = "auto",
    groups: np.ndarray | None = None,
) -> dict:
    """Bootstrap test for moderation (interaction effect).

    Mean-centers *predictor* and *moderator* **per resample** (no
    information leakage), constructs the interaction term
    X_c × Z_c, and fits Y ~ X_c + Z_c + X_c × Z_c.  The interaction
    coefficient is tested via BCa bootstrap CIs.

    Args:
        X: Feature matrix.  Accepts pandas or Polars DataFrames.
        y: Target variable.  Accepts pandas or Polars DataFrames.
        predictor: Predictor (X).
        moderator: Candidate moderator (Z).
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence-interval level.
        precision: Decimal places for rounding.
        random_state: Seed for reproducibility.
        family: Outcome family.  Mixed families are automatically
            resolved to their base family.
        groups: Optional 1-D array of group labels for cluster
            bootstrap.

    Returns:
        Dictionary with keys ``predictor``, ``moderator``,
        ``x_coef``, ``z_coef``, ``interaction_coef``,
        ``interaction_ci``, ``ci_method``, ``is_moderator``,
        and ``interpretation``.
    """
    from .diagnostics import _bca_percentile

    X = _ensure_pandas_df(X, name="X")
    y = _ensure_pandas_df(y, name="y")

    y_values = np.ravel(y)
    x_raw = X[predictor].values.astype(float)
    z_raw = X[moderator].values.astype(float)
    n = len(y_values)

    # Resolve family (mixed → base).
    fam = resolve_family(family, y_values)
    fam, _was_mixed = _resolve_base_family(fam)
    _use_family = fam.name != "linear"

    # Cluster bootstrap support.
    _use_cluster = groups is not None
    if _use_cluster:
        groups = np.asarray(groups).ravel()
        if len(groups) != n:
            raise ValueError(
                f"groups length ({len(groups)}) must match sample size ({n})"
            )

    # --- Observed interaction coefficient (full sample) ---
    x_c = x_raw - x_raw.mean()
    z_c = z_raw - z_raw.mean()
    xz = x_c * z_c

    # Check for collinearity of interaction term with main effects.
    design_check = np.column_stack([x_c, z_c, xz])
    rank = np.linalg.matrix_rank(design_check)
    if rank < 3:
        warnings.warn(
            f"Interaction term X×Z is collinear with main effects "
            f"(design rank {rank} < 3). Skipping moderation test "
            f"for '{moderator}'.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "predictor": predictor,
            "moderator": moderator,
            "x_coef": np.nan,
            "z_coef": np.nan,
            "interaction_coef": np.nan,
            "interaction_ci": (np.nan, np.nan),
            "ci_method": "BCa",
            "is_moderator": False,
            "interpretation": (
                f"Moderation test skipped for '{moderator}': "
                f"interaction term is collinear with main effects."
            ),
        }

    if _use_family:
        design_obs = np.column_stack([x_c, z_c, xz])
        model_obs = fam.fit(design_obs, y_values.astype(float), fit_intercept=True)
        obs_coefs = fam.coefs(model_obs)
        x_coef = float(obs_coefs[0])
        z_coef = float(obs_coefs[1])
        interaction_coef = float(obs_coefs[2])
    else:
        ones = np.ones(n)
        design_obs = np.column_stack([ones, x_c, z_c, xz])
        beta, _, _, _ = np.linalg.lstsq(design_obs, y_values, rcond=None)
        x_coef = float(beta[1])
        z_coef = float(beta[2])
        interaction_coef = float(beta[3])

    # --- Bootstrap for the interaction coefficient ---
    rng = np.random.default_rng(random_state)

    if _use_cluster:
        assert groups is not None
        # Stratified within-cluster resampling: rectangular (B, n)
        # arrays compatible with batch_fit_paired().
        boot_idx_list = _cluster_bootstrap_indices(
            groups, n_bootstrap, rng, mode="stratified"
        )
    else:
        boot_idx_list = [
            rng.choice(n, size=n, replace=True) for _ in range(n_bootstrap)
        ]

    boot_interaction = np.empty(n_bootstrap)
    n_filtered = 0

    # Batch the bootstrap fits when the family supports batch_fit_paired.
    _can_batch = _use_family and hasattr(fam, "batch_fit_paired")
    boot_idx_arr = np.array(boot_idx_list)  # (B, n) — always rectangular

    if _can_batch and boot_idx_arr.ndim == 2:
        # --- Vectorised moderation bootstrap ------------------------
        # Build per-resample design matrices with per-resample centering.
        n_boot = boot_idx_arr.shape[0]
        n_per = boot_idx_arr.shape[1]
        Xmod_batch = np.empty((n_boot, n_per, 3))
        Ymod_batch = np.empty((n_boot, n_per))

        for b_i in range(n_boot):
            idx = boot_idx_arr[b_i]
            xb = x_raw[idx]
            zb = z_raw[idx]
            xb_c = xb - xb.mean()
            zb_c = zb - zb.mean()
            Xmod_batch[b_i] = np.column_stack([xb_c, zb_c, xb_c * zb_c])
            Ymod_batch[b_i] = y_values[idx].astype(float)

        with _suppress_sm_warnings(hessian=True):
            batch_coefs = fam.batch_fit_paired(
                Xmod_batch, Ymod_batch, fit_intercept=True
            )  # (B, 3)
        raw_interaction = batch_coefs[:, 2]

        # Quasi-separation guard
        bad_mask = np.isnan(raw_interaction) | (np.abs(raw_interaction) > 100)
        n_filtered = int(np.sum(bad_mask))
        boot_interaction = np.where(  # type: ignore[assignment]
            bad_mask, np.nan, raw_interaction
        )
    else:
        # --- Sequential fallback ------------------------------------
        for b_i, idx in enumerate(boot_idx_list):
            # Mean-center per resample to avoid information leakage.
            xb = x_raw[idx]
            zb = z_raw[idx]
            xb_c = xb - xb.mean()
            zb_c = zb - zb.mean()
            xzb = xb_c * zb_c

            if _use_family:
                design_b = np.column_stack([xb_c, zb_c, xzb])
                y_b = y_values[idx].astype(float)
                try:
                    model_b = fam.fit(design_b, y_b, fit_intercept=True)
                    coef_int = float(fam.coefs(model_b)[2])
                    # Quasi-separation guard.
                    if np.isnan(coef_int) or abs(coef_int) > 100:
                        boot_interaction[b_i] = np.nan
                        n_filtered += 1
                    else:
                        boot_interaction[b_i] = coef_int
                except Exception:
                    boot_interaction[b_i] = np.nan
                    n_filtered += 1
            else:
                ones_b = np.ones(len(idx))
                design_b = np.column_stack([ones_b, xb_c, zb_c, xzb])
                y_b = y_values[idx]
                coef_b, _, _, _ = np.linalg.lstsq(design_b, y_b, rcond=None)
                boot_interaction[b_i] = coef_b[3]

    # Quasi-separation warning.
    if n_filtered > 0:
        pct_filtered = n_filtered / n_bootstrap * 100
        if pct_filtered > 5:
            warnings.warn(
                f"{pct_filtered:.1f}% of bootstrap replicates were filtered "
                f"due to quasi-complete separation or extreme coefficients "
                f"in moderation analysis for '{moderator}'.",
                UserWarning,
                stacklevel=2,
            )

    # Drop NaN replicates for BCa computation.
    valid_mask = ~np.isnan(boot_interaction)
    boot_valid = boot_interaction[valid_mask]

    if len(boot_valid) < 100:
        warnings.warn(
            f"Only {len(boot_valid)} valid bootstrap replicates for "
            f"moderation analysis of '{moderator}'. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "predictor": predictor,
            "moderator": moderator,
            "x_coef": np.round(x_coef, precision),
            "z_coef": np.round(z_coef, precision),
            "interaction_coef": np.round(interaction_coef, precision),
            "interaction_ci": (np.nan, np.nan),
            "ci_method": "BCa",
            "is_moderator": False,
            "interpretation": (
                f"Moderation test unreliable for '{moderator}': "
                f"too few valid bootstrap replicates ({len(boot_valid)})."
            ),
        }

    # --- Jackknife for BCa acceleration ---
    if _use_cluster:
        assert groups is not None
        jack_idx_list = _cluster_jackknife_indices(groups)
    else:
        jack_idx_list = [
            np.concatenate([np.arange(i), np.arange(i + 1, n)]) for i in range(n)
        ]

    n_jack = len(jack_idx_list)
    jack_interaction = np.empty(n_jack)

    # Batch the jackknife fits when the family supports batch_fit_paired.
    jack_idx_arr = np.array(jack_idx_list) if not _use_cluster else None

    if _can_batch and jack_idx_arr is not None and jack_idx_arr.ndim == 2:
        # --- Vectorised moderation jackknife ------------------------
        n_j = jack_idx_arr.shape[1]
        Xmod_jack = np.empty((n_jack, n_j, 3))
        Ymod_jack = np.empty((n_jack, n_j))

        for j in range(n_jack):
            jidx = jack_idx_arr[j]
            xj = x_raw[jidx]
            zj = z_raw[jidx]
            xj_c = xj - xj.mean()
            zj_c = zj - zj.mean()
            Xmod_jack[j] = np.column_stack([xj_c, zj_c, xj_c * zj_c])
            Ymod_jack[j] = y_values[jidx].astype(float)

        with _suppress_sm_warnings(hessian=True):
            jack_coefs = fam.batch_fit_paired(
                Xmod_jack, Ymod_jack, fit_intercept=True
            )  # (J, 3)
        jack_interaction = jack_coefs[:, 2]  # type: ignore[assignment]
    else:
        # --- Sequential fallback ------------------------------------
        for j, jidx in enumerate(jack_idx_list):
            xj = x_raw[jidx]
            zj = z_raw[jidx]
            xj_c = xj - xj.mean()
            zj_c = zj - zj.mean()
            xzj = xj_c * zj_c

            if _use_family:
                design_j = np.column_stack([xj_c, zj_c, xzj])
                y_j = y_values[jidx].astype(float)
                try:
                    model_j = fam.fit(design_j, y_j, fit_intercept=True)
                    jack_interaction[j] = float(fam.coefs(model_j)[2])
                except Exception:
                    jack_interaction[j] = np.nan
            else:
                ones_j = np.ones(len(jidx))
                design_j = np.column_stack([ones_j, xj_c, zj_c, xzj])
                y_j = y_values[jidx]
                coef_j, _, _, _ = np.linalg.lstsq(design_j, y_j, rcond=None)
                jack_interaction[j] = coef_j[3]

    # Drop NaN jackknife values.
    jack_valid = jack_interaction[~np.isnan(jack_interaction)]
    if len(jack_valid) < 3:
        ci_lower_val = float(np.nanpercentile(boot_valid, 2.5))
        ci_upper_val = float(np.nanpercentile(boot_valid, 97.5))
    else:
        alpha = 1 - confidence_level
        ci_lower_val, ci_upper_val = _bca_percentile(
            boot_valid, interaction_coef, jack_valid, alpha
        )

    is_moderator = (ci_lower_val > 0) or (ci_upper_val < 0)

    if is_moderator:
        interpretation = (
            f"'{moderator}' moderates the effect of '{predictor}' "
            f"on the outcome (interaction BCa CI excludes zero). "
            f"Consider including the interaction term X×Z as a predictor."
        )
    else:
        interpretation = (
            f"'{moderator}' does not significantly moderate the effect "
            f"of '{predictor}' (interaction BCa CI includes zero)."
        )

    return {
        "predictor": predictor,
        "moderator": moderator,
        "x_coef": np.round(x_coef, precision),
        "z_coef": np.round(z_coef, precision),
        "interaction_coef": np.round(interaction_coef, precision),
        "interaction_ci": (
            np.round(ci_lower_val, precision),
            np.round(ci_upper_val, precision),
        ),
        "ci_method": "BCa",
        "is_moderator": is_moderator,
        "interpretation": interpretation,
    }


# ------------------------------------------------------------------ #
# Four-stage confounder sieve
# ------------------------------------------------------------------ #
#
# The sieve identifies true confounders by removing variables that
# play other causal roles from the candidate pool.  Stages run in
# priority order with sequential removal:
#
#   1. **Screen** — dual-correlation (Pearson / partial / distance)
#      with optional multiple-testing correction.
#   2. **Collider test** — significance + amplification.  Colliders
#      are tested first because conditioning on a collider *creates*
#      bias (most dangerous misclassification).  Removed from pool.
#   3. **Mediator test** — Preacher & Hayes indirect effect bootstrap.
#      Mediators are removed from pool (controlling for them removes
#      part of the real effect).
#   4. **Moderator test** — interaction-term bootstrap.  Moderators
#      are NOT removed from pool (informational label only).
#
# What's left after stages 2–4 → identified confounders.
#
# **Post-selection inference caveat**: the sieve selects the
# conditioning set from the same data used for the subsequent
# permutation test.  The permutation p-value is exact *conditional on
# the selected conditioning set*, but the conditioning set itself is
# data-adaptive.  False collider classification (removing a true
# confounder) can inflate Type I error.  False confounder
# classification (including noise) is conservative — reduces power
# but does not inflate Type I error in Kennedy/Freedman-Lane.
#
# For guaranteed Type I error control, specify ``confounders=`` based
# on domain knowledge or a pre-registered analysis plan.
#
# **No multiple-testing correction within the sieve**: each sieve
# stage tests a different hypothesis type (is Z a collider? a
# mediator? a moderator?) — these are not repeated tests of the same
# null.  Correction across different hypothesis types is contested
# (Rothman, 1990).  The screening stage does offer optional correction
# because those are repeated tests of the same type.


def identify_confounders(
    X: DataFrameLike,
    y: DataFrameLike,
    predictor: str,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
    n_bootstrap_mediation: int = 1000,
    n_bootstrap_moderation: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
    family: str | ModelFamily = "auto",
    correlation_method: str = "pearson",
    correction_method: str | None = None,
    groups: np.ndarray | None = None,
) -> ConfounderAnalysisResult:
    """Four-stage confounder sieve.

    Classifies candidate variables as colliders, mediators, moderators,
    or true confounders via sequential testing:

    1. **Screen** — dual-correlation with both predictor and outcome.
    2. **Collider test** — removes colliders (X → Z ← Y).
    3. **Mediator test** — removes mediators (X → M → Y).
    4. **Moderator test** — labels moderators (informational; stays
       in confounder pool).

    The sieve is an **exploratory** tool for data-driven confounder
    selection.  For guaranteed Type I error control, specify
    ``confounders=`` based on domain knowledge or a pre-registered
    analysis plan.  The permutation p-value is exact conditional on the
    selected conditioning set.

    Args:
        X: Feature matrix.  Accepts pandas or Polars DataFrames.
        y: Target variable.  Accepts pandas or Polars DataFrames.
        predictor: Predictor of interest.
        correlation_threshold: Minimum absolute correlation to flag.
        p_value_threshold: Significance cutoff for screening.
        n_bootstrap_mediation: Bootstrap iterations for mediation.
        n_bootstrap_moderation: Bootstrap iterations for moderation.
        confidence_level: Confidence-interval level.
        random_state: Seed for reproducibility.
        family: Outcome family.  Mixed families resolved to base.
        correlation_method: ``"pearson"``, ``"partial"``, or
            ``"distance"`` (passed to screening).
        correction_method: ``None``, ``"holm"``, or ``"fdr_bh"``
            (passed to screening).
        groups: Optional group labels for cluster bootstrap
            (passed to mediation/moderation).

    Returns:
        :class:`ConfounderAnalysisResult` with classified candidates,
        screening results, and per-candidate analysis details.
    """
    from ._results import ConfounderAnalysisResult

    X = _ensure_pandas_df(X, name="X")
    y = _ensure_pandas_df(y, name="y")
    y_values = np.ravel(y)

    # Resolve family once (mixed → base).
    resolved_family = resolve_family(family, y_values)
    resolved_family, _was_mixed = _resolve_base_family(resolved_family)

    # --- Stage 1: Screen ---
    screening = screen_potential_confounders(
        X,
        y,
        predictor,
        correlation_threshold=correlation_threshold,
        p_value_threshold=p_value_threshold,
        correlation_method=correlation_method,
        correction_method=correction_method,
    )
    candidates = list(screening["potential_confounders"])

    # --- Multinomial early exit ---
    if resolved_family.name == "multinomial":
        warnings.warn(
            "Multinomial outcomes produce multi-class Wald χ² statistics, "
            "not scalar coefficient slopes. Mediation, moderation, and "
            "collider analysis require directional scalar effects and are "
            "not supported for multinomial families. All screened candidates "
            "are reported as confounders.",
            UserWarning,
            stacklevel=2,
        )
        return ConfounderAnalysisResult(
            predictor=predictor,
            identified_confounders=candidates,
            identified_mediators=[],
            identified_moderators=[],
            identified_colliders=[],
            screening_results=screening,
        )

    # --- Stage 2: Collider test ---
    identified_colliders: list[str] = []
    collider_results: dict = {}
    remaining = list(candidates)
    predictor_values = X[predictor].values

    for candidate in candidates:
        is_coll, coef_marg, coef_part = _collider_test(
            predictor_values,
            y_values,
            X[candidate].values,
            family=resolved_family if resolved_family.name != "linear" else None,
            random_state=random_state,
        )
        collider_results[candidate] = {
            "is_collider": is_coll,
            "coef_marginal": float(coef_marg) if not np.isnan(coef_marg) else np.nan,
            "coef_partial": float(coef_part) if not np.isnan(coef_part) else np.nan,
        }
        if is_coll:
            identified_colliders.append(candidate)
            remaining.remove(candidate)

    # --- Stage 3: Mediator test ---
    identified_mediators: list[str] = []
    mediation_results_dict: dict = {}

    for candidate in list(remaining):
        med = mediation_analysis(
            X,
            y,
            predictor,
            candidate,
            n_bootstrap=n_bootstrap_mediation,
            confidence_level=confidence_level,
            random_state=random_state,
            family=resolved_family,
            groups=groups,
        )
        mediation_results_dict[candidate] = med
        if med["is_mediator"]:
            identified_mediators.append(candidate)
            remaining.remove(candidate)

    # --- Stage 4: Moderator test ---
    identified_moderators: list[str] = []
    moderation_results_dict: dict = {}

    for candidate in list(remaining):
        mod = moderation_analysis(
            X,
            y,
            predictor,
            candidate,
            n_bootstrap=n_bootstrap_moderation,
            confidence_level=confidence_level,
            random_state=random_state,
            family=resolved_family,
            groups=groups,
        )
        moderation_results_dict[candidate] = mod
        if mod["is_moderator"]:
            identified_moderators.append(candidate)
            # Do NOT remove from remaining — moderator label is non-exclusive.

    # What's left → confounders.
    identified_confounders = list(remaining)

    # --- Collinearity guard (Step 10) ---
    if len(identified_confounders) >= 2:
        try:
            confounder_vals = X[identified_confounders].values
            pred_vals = X[predictor].values
            # Regress predictor on identified confounders.
            ones = np.ones((len(pred_vals), 1))
            design = np.column_stack([ones, confounder_vals])
            beta, _, _, _ = np.linalg.lstsq(design, pred_vals, rcond=None)
            pred_hat = design @ beta
            ss_res = np.sum((pred_vals - pred_hat) ** 2)
            ss_tot = np.sum((pred_vals - pred_vals.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if r_squared > 0.95:
                warnings.warn(
                    f"Identified confounders explain {r_squared:.1%} of "
                    f"predictor variance. Kennedy/Freedman-Lane test may "
                    f"have low power. Consider reducing the confounder set.",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception:
            pass  # Non-critical diagnostic — silently skip on error.

    return ConfounderAnalysisResult(
        predictor=predictor,
        identified_confounders=identified_confounders,
        identified_mediators=identified_mediators,
        identified_moderators=identified_moderators,
        identified_colliders=identified_colliders,
        screening_results=screening,
        mediation_results=mediation_results_dict,
        moderation_results=moderation_results_dict,
        collider_results=collider_results,
    )

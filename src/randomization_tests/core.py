"""Core permutation testing engine.

Permutation tests assess statistical significance without relying on
distributional assumptions (normality, homoscedasticity, etc.) that
underpin classical parametric tests.  The core idea is simple:

    Under the null hypothesis H₀: β_j = 0, the observed value of Y
    has no systematic relationship with predictor X_j.  Therefore,
    any rearrangement of the data that breaks the X-Y link but
    preserves nuisance structure should yield a coefficient β*_j
    drawn from the same null distribution.

By repeating this rearrangement B times, we build an empirical null
distribution (the **reference set**) and compare the observed β_j to
it.  The fraction of null values at least as extreme as the observed
one gives the p-value.

This module implements three permutation strategies:

1. **ter Braak (1992)** – Residual permutation under the reduced model.
   For each predictor X_j, fit the model *without* X_j to get
   predicted values ŷ₋ⱼ and residuals e₋ⱼ = Y - ŷ₋ⱼ.  Permute
   the residuals and add them back to ŷ₋ⱼ to form Y* = ŷ₋ⱼ + π(e₋ⱼ).
   Refit the full model on (X, Y*) to get β*_j.

   Intuition: Under H₀ the residuals from the reduced model are
   exchangeable — each ordering is equally likely.  Permuting them
   preserves the variance structure of Y while destroying any real
   association with X_j.  Because the reduced model conditions on
   *all other* predictors, each test is implicitly adjusted for the
   remaining features.

2. **Kennedy (1995) individual** – Partial out confounders via an
   exposure model, then permute the exposure residuals.

   For each predictor X_j, regress X_j on a matrix of confounders Z
   to get predicted values X̂_j and residuals eₓⱼ = X_j - X̂_j.
   Under H₀, eₓⱼ carries no information about Y.  Permute eₓⱼ and
   reconstruct X*_j = X̂_j + π(eₓⱼ).  Refit the full model on
   (X with X*_j, Y) to get β*_j.

   Intuition: The exposure model removes the linear influence of
   confounders on X_j.  The residuals represent the part of X_j
   that is *not* explained by Z — i.e., the "clean" variation.
   Permuting only this clean variation tests whether X_j has any
   additional predictive value for Y beyond what Z already explains.

3. **Kennedy (1995) joint** – Test whether a group of predictors
   collectively improves fit beyond confounders alone.  The test
   statistic is the improvement in fit:
     • Linear:   RSS_reduced − RSS_full (residual sum of squares)
     • Logistic:  Deviance_reduced − Deviance_full

   The exposure model is applied to all non-confounder predictors
   simultaneously, and residuals are permuted **row-wise** so that
   the correlation structure among the predictors is preserved.

Vectorisation strategy
~~~~~~~~~~~~~~~~~~~~~~
* **OLS (linear)** paths use batch matrix algebra.  The pseudoinverse
  (X'X)⁻¹X' is computed once and multiplied against all B permuted
  Y (or X) vectors simultaneously:

      β̂ = pinv(X) @ Y_matrix'   →  shape (p, B)  →  transpose to (B, p)

  This turns the inner permutation loop into a single ``(p × n) @ (n × B)``
  matrix product — orders of magnitude faster than fitting B separate
  models.

* **Logistic** paths have no closed-form solution and require
  iterative fitting.  When JAX is available, a Newton–Raphson solver
  (gradient and Hessian via ``jax.grad``) is vectorised across all B
  permutations using ``jax.vmap``, executing on GPU or TPU if present.
  Otherwise, an sklearn ``LogisticRegression`` loop serves as a
  transparent fallback.

References:
    ter Braak, C. J. F. (1992). Permutation versus bootstrap
    significance tests in multiple regression and ANOVA. In
    J.-P. Dijkstra (Ed.), *Statistics in Applied Science* (pp. 79–86).

    Kennedy, P. E. (1995). Randomization tests in econometrics.
    *Journal of Business & Economic Statistics*, 13(1), 85–94.

    Phipson, B. & Smyth, G. K. (2010). Permutation p-values should
    never be zero. *Stat. Appl. Genet. Mol. Biol.*, 9(1), Article 39.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning as SmConvergenceWarning,
)
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationWarning,
)

from ._compat import DataFrameLike, _ensure_pandas_df

# ------------------------------------------------------------------ #
# JAX-accelerated logistic regression
# ------------------------------------------------------------------ #
#
# Logistic regression permutation paths have no closed-form solution
# and require iterative fitting (Newton–Raphson) for each of the B
# permutations.  JAX dramatically accelerates this via:
#
#   1. **Automatic differentiation** — jax.grad computes exact
#      gradients of the logistic negative log-likelihood.  No hand-
#      coded gradient formulas, and numerically stable by construction.
#
#   2. **vmap (vectorised map)** — transforms a single-permutation
#      solver into a batched solver that processes all B permutations
#      in one XLA kernel launch.  Combined with jit compilation, this
#      eliminates Python-level dispatch overhead entirely.
#
# If JAX is not installed, the code below still imports successfully —
# _jax.py handles the try/except internally and the two fit functions
# simply won't be called (they only execute behind ``if _use_jax()``
# guards).  The sklearn LogisticRegression loop serves as a correct
# but slower fallback.
#
# The decision of whether to USE JAX is made at call time via
# ``_use_jax()``, which combines two checks:
#   - Is JAX importable?  (determined once at module load)
#   - Does the runtime policy say "jax"?  (get_backend() resolution;
#     see _config.py for the programmatic / env-var / auto cascade)
#
# Prior to v0.2.0, all JAX code lived inline in this file.  Extracting
# it into _jax.py keeps core.py focused on the permutation algorithms
# and provides a clean extraction point for v0.3.0's _backends/ package.
from .diagnostics import compute_all_diagnostics
from .families import ModelFamily, resolve_family
from .permutations import generate_unique_permutations
from .pvalues import calculate_p_values

# ------------------------------------------------------------------ #
# Batch OLS via pseudoinverse (numpy)
# ------------------------------------------------------------------ #
#
# The OLS estimator for a single Y vector is:
#
#   β̂ = (X'X)⁻¹ X'Y = pinv(X) @ Y
#
# where pinv(X) = (X'X)⁻¹X' is the Moore-Penrose pseudoinverse.
# Crucially, pinv(X) depends only on X, which is the SAME across all
# B permutations (we permute Y, not X, in the ter Braak method).
#
# By stacking all B permuted Y vectors into a matrix Y_matrix of
# shape (B, n), we compute ALL B coefficient vectors in one shot:
#
#   coefs = pinv(X) @ Y_matrix'   →  shape (p, B)
#   coefs.T                        →  shape (B, p)
#
# This replaces B separate lstsq calls with a single (p × n) @ (n × B)
# matrix multiply — a dramatic speedup for typical permutation counts
# (B = 5 000) because BLAS-level matrix multiplication is highly
# optimised for modern CPUs (cache-blocking, SIMD, multi-threading).


def _batch_ols_coefs(
    X: np.ndarray,
    Y_matrix: np.ndarray,
    fit_intercept: bool = True,
) -> np.ndarray:
    """Compute OLS coefficients for many Y vectors via one matrix multiply.

    When *fit_intercept* is True (default), an intercept column is
    prepended automatically so that the returned slope coefficients
    match what sklearn's ``LinearRegression(fit_intercept=True)``
    produces.  The intercept coefficients are stripped before
    returning.

    When *fit_intercept* is False, the raw design matrix is used and
    no intercept is estimated (matching ``fit_intercept=False``).

    Args:
        X: Design matrix of shape ``(n, p)`` — **without** an
            intercept column.
        Y_matrix: Matrix of shape ``(B, n)`` where each row is a
            permuted response vector.
        fit_intercept: Whether to include an intercept term.

    Returns:
        Coefficient matrix of shape ``(B, p)`` where
        ``coefs[b]`` contains the slope coefficients (intercept
        excluded when *fit_intercept* is True) for Y_matrix[b].
    """
    if fit_intercept:
        # Prepend a column of ones so the pseudoinverse yields both an
        # intercept term (column 0) and p slope terms (columns 1:).
        X_aug = np.column_stack([np.ones(X.shape[0]), X])  # (n, p+1)
        pinv = np.linalg.pinv(X_aug)  # (p+1, n)
        result: np.ndarray = (pinv @ Y_matrix.T).T  # (B, p+1)
        return result[:, 1:]  # drop intercept column
    else:
        pinv = np.linalg.pinv(X)  # (p, n)
        result = (pinv @ Y_matrix.T).T  # (B, p)
        return result


# ------------------------------------------------------------------ #
# Diagnostics (statsmodels)
# ------------------------------------------------------------------ #
#
# Model diagnostics provide context for interpreting the permutation
# results.  They are computed once on the *observed* (unpermuted) data
# via statsmodels, which provides a rich set of summary statistics.
#
# Key metrics:
#   Linear:
#     R²      — fraction of Y variance explained by X.
#     Adj. R² — R² penalised for model complexity.
#     F-stat  — joint test that all coefficients are zero.
#     AIC/BIC — information criteria for model comparison:
#       AIC = -2ℓ + 2k          (Akaike, 1974)
#       BIC = -2ℓ + k·ln(n)     (Schwarz, 1978)
#       where ℓ is log-likelihood and k the number of parameters.
#
#   Logistic:
#     Pseudo R² — McFadden's R² = 1 - ℓ(model)/ℓ(null).
#     Log-likelihood — ℓ(model) and ℓ(null) for the intercept-only.
#     LLR p-value — likelihood-ratio test for the full model.


def _compute_diagnostics(
    X: pd.DataFrame,
    y_values: np.ndarray,
    model_type: str,
    fit_intercept: bool = True,
) -> dict:
    """Compute model diagnostics via statsmodels.

    Fits the full model (with all features) using statsmodels and
    extracts summary statistics for display in the results table.

    Args:
        X: Feature matrix.
        y_values: Response vector.
        model_type: Model family name (e.g. ``"linear"``,
            ``"logistic"``).  Controls which statsmodels estimator
            is used.
        fit_intercept: Whether to include an intercept term.  When
            True (default), ``sm.add_constant(X)`` is used.  When
            False, the raw design matrix is passed directly.

    Returns:
        Dictionary of diagnostic statistics.  Keys differ by model
        type (e.g. ``r_squared`` for OLS, ``pseudo_r_squared`` for
        logistic).
    """
    n_obs = len(y_values)
    n_features = X.shape[1]
    X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)

    if model_type == "logistic":
        sm_model = sm.Logit(y_values, X_sm).fit(disp=0)
        return {
            "n_observations": n_obs,
            "n_features": n_features,
            "pseudo_r_squared": np.round(sm_model.prsquared, 4),
            "log_likelihood": np.round(sm_model.llf, 4),
            "log_likelihood_null": np.round(sm_model.llnull, 4),
            "llr_p_value": sm_model.llr_pvalue,
            "aic": np.round(sm_model.aic, 4),
            "bic": np.round(sm_model.bic, 4),
        }
    else:
        sm_model = sm.OLS(y_values, X_sm).fit()
        return {
            "n_observations": n_obs,
            "n_features": n_features,
            "r_squared": np.round(sm_model.rsquared, 4),
            "r_squared_adj": np.round(sm_model.rsquared_adj, 4),
            "f_statistic": np.round(sm_model.fvalue, 4),
            "f_p_value": sm_model.f_pvalue,
            "aic": np.round(sm_model.aic, 4),
            "bic": np.round(sm_model.bic, 4),
        }


# ------------------------------------------------------------------ #
# ter Braak (1992) — Residual permutation under the reduced model
# ------------------------------------------------------------------ #
#
# Algorithm (for each feature X_j):
#
#   1. Fit the **reduced** OLS model  Y ~ X_{-j}  (all features except
#      X_j) to obtain predicted values ŷ_{-j} and residuals:
#
#        e_{-j} = Y - ŷ_{-j}
#
#      Under H₀: β_j = 0, X_j contributes nothing, so Y is fully
#      described by X_{-j} plus exchangeable noise.
#
#   2. For each permutation b = 1, …, B, randomly reorder the
#      residuals to get π_b(e_{-j}), then construct:
#
#        Y*_b = ŷ_{-j} + π_b(e_{-j})
#
#      This preserves the systematic component (explained by X_{-j})
#      while destroying any real association between Y and X_j.
#
#   3. Refit the **full** model  Y*_b ~ X  and extract β*_{b,j}.
#
# The B values of β*_{b,j} form the empirical null distribution for
# the j-th coefficient.  Comparing |β_j| to this distribution yields
# the permutation p-value.
#
# Importantly, one set of permutations is generated ONCE and reused
# across all features.  The vectorised implementation computes all B
# full-model refits in a single matrix multiply via _batch_ols_coefs.


def _ter_braak_generic(
    X: pd.DataFrame,
    y_values: np.ndarray,
    family: ModelFamily,
    perm_indices: np.ndarray,
    fit_intercept: bool = True,
    n_jobs: int = 1,
) -> np.ndarray:
    """Family-generic ter Braak residual-permutation engine.

    Replaces the former ``_ter_braak_linear`` and ``_ter_braak_logistic``
    with a single code path that delegates all model-specific operations
    to the ``ModelFamily`` protocol:

    For each feature *j*:

    1. **Reduced model** — ``family.fit(X_{-j}, y)`` with column *j*
       dropped.  Produces predictions via ``family.predict()`` and
       residuals via ``family.residuals()``.

    2. **Permute residuals** — fancy-index the residual vector with
       ``perm_indices`` to obtain B permuted residual vectors at once.

    3. **Reconstruct Y*** — ``family.reconstruct_y(preds, perm_resids, rng)``.
       For linear families this is additive (Y* = ŷ + π(e)).
       For logistic families it clips to [0.001, 0.999] then draws
       Bernoulli(p*).

    4. **Batch refit** — ``family.batch_fit(X, Y_perm, fit_intercept)``
       computes all B full-model coefficient vectors in one call,
       dispatching to the active backend (NumPy pseudoinverse for OLS,
       JAX vmap'd Newton–Raphson for logistic).

    This design means adding a new family (Poisson, ordinal, etc.)
    requires zero changes here — only the family's protocol methods.

    Args:
        X: Feature matrix as a pandas DataFrame.
        y_values: Response vector of shape ``(n,)``.
        family: Resolved ``ModelFamily`` instance.
        perm_indices: Pre-generated permutation indices, shape
            ``(B, n)``.
        fit_intercept: Whether to include an intercept in both the
            reduced and full model refits.  Must match the observed
            model so that coefficients are comparable.

    Returns:
        Array of shape ``(B, n_features)`` with permuted coefficients.
    """
    X_np = X.values.astype(float)
    n_perm, n = perm_indices.shape
    n_features = X_np.shape[1]
    result = np.zeros((n_perm, n_features))

    # Derive a deterministic RNG from the permutation indices so that
    # any stochastic reconstruction step (e.g. Bernoulli sampling for
    # logistic) is reproducible given the same permutations.
    rng = np.random.default_rng(int(perm_indices[0, 0]))

    for j in range(n_features):
        # Step 1: Fit the reduced model Y ~ X_{-j}.
        # Drop column j from the design matrix.  The family's fit()
        # handles all model-specific details (solver, link function,
        # regularisation, etc.).
        X_red = np.delete(X_np, j, axis=1)
        reduced_model = family.fit(X_red, y_values, fit_intercept)

        # Predictions and residuals from the reduced model.
        # The residual type depends on the family:
        #   - Linear: raw residuals e = Y - ŷ
        #   - Logistic: probability-scale residuals e = Y - P̂(Y=1)
        preds_red = family.predict(reduced_model, X_red)  # shape: (n,)
        resids_red = family.residuals(reduced_model, X_red, y_values)  # (n,)

        # Step 2: Build all B permuted residual vectors simultaneously.
        # perm_indices has shape (B, n); fancy-indexing resids_red
        # gives a (B, n) matrix where each row is a permuted residual
        # vector.
        permuted_resids = resids_red[perm_indices]  # (B, n)

        # Step 3: Reconstruct permuted response vectors.
        # The family's reconstruct_y() handles the model-specific
        # transformation:
        #   - Linear: Y* = ŷ_{-j} + π(e_{-j})  (deterministic)
        #   - Logistic: Y* ~ Bernoulli(clip(ŷ + π(e)))  (stochastic)
        # Broadcasting: preds_red is (n,), permuted_resids is (B, n);
        # reconstruct_y produces (B, n).
        Y_perm = family.reconstruct_y(
            preds_red[np.newaxis, :],  # (1, n) for broadcasting
            permuted_resids,  # (B, n)
            rng,
        )  # (B, n)

        # Step 4: Batch-refit the full model on all B permuted Y vectors.
        # family.batch_fit() dispatches to the active backend:
        #   - Linear: NumPy pseudoinverse (single matrix multiply)
        #   - Logistic: JAX vmap'd Newton solver or sklearn fallback
        all_coefs = family.batch_fit(
            X_np, Y_perm, fit_intercept, n_jobs=n_jobs
        )  # (B, p)
        result[:, j] = all_coefs[:, j]

    return result


# ------------------------------------------------------------------ #
# Kennedy (1995) individual — exposure-model residual permutation
# ------------------------------------------------------------------ #
#
# Unlike ter Braak (which permutes Y-residuals), Kennedy's approach
# permutes X-residuals — the part of a predictor that is NOT explained
# by a set of known confounders Z.
#
# Algorithm (for each feature X_j that is NOT a confounder):
#
#   1. Fit an **exposure model**  X_j = Z·γ + eₓⱼ  where Z is the
#      matrix of confounder columns.  The predicted values X̂_j = Z·γ̂
#      capture X_j's linear dependence on Z, and the residuals
#      eₓⱼ = X_j - X̂_j represent the "clean" variation in X_j that
#      is orthogonal to the confounders.
#
#      If there are no confounders, X̂_j is just the column mean of
#      X_j, so the residuals are mean-centred values and permuting
#      them is equivalent to shuffling X_j directly.
#
#   2. Under H₀: β_j = 0, the clean variation eₓⱼ carries no
#      information about Y.  Permute eₓⱼ to get π_b(eₓⱼ) and
#      reconstruct:
#
#        X*_j = X̂_j + π_b(eₓⱼ)
#
#      This preserves X_j's relationship with Z while destroying any
#      real association between X_j and Y.
#
#   3. Replace column X_j with X*_j in the design matrix and refit
#      the full model  Y ~ X*  to get β*_{b,j}.
#
# The key difference from ter Braak: Kennedy conditions on SPECIFIED
# confounders only, not on all other predictors.  This makes it
# possible to test X_j while controlling for a known subset of Z,
# rather than implicitly conditioning on everything else in the model.


def _kennedy_individual_generic(
    X: pd.DataFrame,
    y_values: np.ndarray,
    family: ModelFamily,
    confounders: list[str],
    perm_indices: np.ndarray,
    model_coefs: np.ndarray,
    fit_intercept: bool = True,
    n_jobs: int = 1,
) -> np.ndarray:
    """Family-generic Kennedy individual exposure-residual permutation.

    Replaces the former ``_kennedy_individual_linear`` and
    ``_kennedy_individual_logistic`` with a single code path.

    The exposure model (X_j regressed on confounders Z) is always
    linear regardless of the outcome family.  The only family-specific
    step is the **outcome refit** (Y ~ X*), which is dispatched via
    ``family.batch_fit_varying_X(X_batch, y, fit_intercept)`` to the
    active backend.

    Algorithm for each non-confounder feature X_j:

    1. **Exposure model** — Regress X_j on Z via OLS to get predicted
       values X̂_j and residuals eₓⱼ = X_j − X̂_j.
    2. **Permute** — Shuffle eₓⱼ using ``perm_indices`` to get B
       permuted residual vectors π_b(eₓⱼ).
    3. **Reconstruct** — X*_j = X̂_j + π_b(eₓⱼ) for each permutation.
    4. **Batch refit** — Replace column j in X with X*_j for all B
       permutations, then call ``family.batch_fit_varying_X()`` which
       dispatches to the appropriate backend solver.

    Args:
        X: Feature matrix as a pandas DataFrame.
        y_values: Response vector of shape ``(n,)``.
        family: Resolved ``ModelFamily`` instance.
        confounders: List of confounder column names.
        perm_indices: Pre-generated permutation indices ``(B, n)``.
        model_coefs: Observed model coefficients ``(p,)`` — used to
            fill confounder slots (they are not being tested).
        fit_intercept: Whether to include an intercept.

    Returns:
        Array of shape ``(B, n_features)`` with permuted coefficients.
    """
    X_np = X.values.astype(float)
    n_perm, n = perm_indices.shape
    n_features = X_np.shape[1]
    result = np.zeros((n_perm, n_features))

    features_to_test = [c for c in X.columns if c not in confounders]

    if confounders:
        Z = X[confounders].values
    else:
        Z = np.zeros((n, 0))

    # Confounder columns are not being tested — fill their permutation
    # coefficients with the observed value so they pass through unchanged.
    for i, col in enumerate(X.columns):
        if col in confounders:
            result[:, i] = model_coefs[i]

    for feature in features_to_test:
        feat_idx = X.columns.get_loc(feature)
        x_target = X[[feature]].values  # (n, 1)

        # Step 1: Exposure model — regress X_j on confounders Z.
        # The exposure model is always linear because X_j is continuous
        # regardless of whether Y is binary.
        if Z.shape[1] > 0:
            if fit_intercept:
                Z_aug = np.column_stack([np.ones(n), Z])
            else:
                Z_aug = Z
            pinv_z = np.linalg.pinv(Z_aug)
            x_hat = Z_aug @ (pinv_z @ x_target)
        else:
            if fit_intercept:
                x_hat = np.full_like(x_target, x_target.mean())
            else:
                x_hat = np.zeros_like(x_target)
        x_resids = (x_target - x_hat).ravel()  # (n,)

        # Step 2: Permute exposure residuals for all B permutations.
        shuffled = x_resids[perm_indices]  # (B, n)

        # Step 3: Reconstruct X*_j = X̂_j + π(eₓⱼ) and build the
        # batch of design matrices.
        X_perm_all = np.broadcast_to(X_np, (n_perm, n, n_features)).copy()
        X_perm_all[:, :, feat_idx] = x_hat.ravel()[np.newaxis, :] + shuffled

        # Step 4: Batch-refit the outcome model on all B permuted
        # design matrices.  family.batch_fit_varying_X() dispatches to
        # the appropriate backend:
        #   - Linear: numpy lstsq loop or JAX vmap'd lstsq
        #   - Logistic: JAX vmap'd Newton solver or sklearn loop
        all_coefs = family.batch_fit_varying_X(
            X_perm_all, y_values, fit_intercept, n_jobs=n_jobs
        )
        result[:, feat_idx] = all_coefs[:, feat_idx]

    return result


# ------------------------------------------------------------------ #
# Kennedy (1995) joint — collective predictive improvement test
# ------------------------------------------------------------------ #
#
# Instead of testing each predictor individually, this tests whether a
# GROUP of non-confounder predictors collectively contributes
# significant predictive information beyond the confounders alone.
#
# The test statistic measures the improvement in model fit when adding
# the predictors of interest to a confounders-only (reduced) model:
#
#   Linear:   T = RSS_reduced − RSS_full
#             where RSS = Σᵢ(yᵢ − ŷᵢ)² = MSE × n.
#             A larger drop means the predictors explain more variance.
#
#   Logistic: T = Deviance_reduced − Deviance_full
#             where Deviance = 2·Σᵢ[−yᵢ·log(pᵢ) − (1−yᵢ)·log(1−pᵢ)].
#             A larger drop means the predictors improve classification.
#
# The null distribution is built by permuting the exposure-model
# residuals of ALL non-confounder predictors simultaneously.  Crucially,
# permutations are done **row-wise** (the same index shuffle applied to
# all predictor residuals for a given permutation), which preserves
# inter-predictor correlations.  If the predictors were shuffled
# independently, their correlation structure would be destroyed, leading
# to an anti-conservative test.
#
# Algorithm:
#   1. Fit exposure models: X_target = Z·Γ + E, where X_target is the
#      matrix of non-confounder columns and E is the residual matrix.
#   2. For each permutation, shuffle the ROWS of E (same permutation
#      for all columns), reconstruct X* = X̂ + π(E), and refit:
#      T*_b = metric(reduced) − metric(full on X*).
#   3. The p-value is the fraction of T*_b >= T_obs, with the
#      Phipson & Smyth (2010) correction.


def _kennedy_joint(
    X: pd.DataFrame,
    y_values: np.ndarray,
    confounders: list[str],
    perm_indices: np.ndarray,
    family: ModelFamily,
    fit_intercept: bool = True,
    n_jobs: int = 1,
) -> tuple[float, np.ndarray, str, list[str]]:
    """Family-generic Kennedy joint test.

    Replaces the former ``is_binary``-branched implementation with a
    single code path that delegates model-specific operations to the
    ``ModelFamily`` protocol:

    * ``family.fit()`` / ``family.predict()`` for reduced and full
      model fits.
    * ``family.fit_metric()`` for the goodness-of-fit measure (RSS
      for linear, deviance for logistic).

    The exposure model (X_target regressed on Z) is always linear
    regardless of the outcome family.

    Returns:
        A ``(obs_improvement, perm_improvements, metric_type,
        features_tested)`` tuple.
    """
    features_to_test = [c for c in X.columns if c not in confounders]

    # Metric label for display — derived from family name.
    metric_type = "Deviance Reduction" if family.name == "logistic" else "RSS Reduction"

    n_perm, n = perm_indices.shape
    X_target = X[features_to_test].values

    if confounders:
        Z = X[confounders].values
    else:
        Z = np.zeros((n, 0))

    # --- Reduced model (confounders only) ---
    # Under H₀, the non-confounder predictors add nothing, so the
    # reduced model represents the best fit achievable without them.
    if Z.shape[1] > 0:
        reduced_model = family.fit(Z, y_values, fit_intercept)
        preds_reduced = family.predict(reduced_model, Z)
    else:
        # No confounders: the "reduced model" is the intercept-only
        # prediction (grand mean for linear, base rate for logistic).
        if fit_intercept:
            preds_reduced = np.full(n, np.mean(y_values), dtype=float)
        else:
            preds_reduced = np.zeros(n, dtype=float)

    base_metric = family.fit_metric(y_values, preds_reduced)

    # --- Full model (confounders + predictors of interest) ---
    # The observed improvement is how much better the full model fits.
    full_features = np.hstack([X_target, Z]) if Z.shape[1] > 0 else X_target
    full_model = family.fit(full_features, y_values, fit_intercept)
    preds_full = family.predict(full_model, full_features)
    obs_improvement = base_metric - family.fit_metric(y_values, preds_full)

    # --- Exposure model residuals ---
    # Regress X_target on Z to get X̂ = Z·Γ̂ and residuals E = X_target - X̂.
    # Row-wise permutation of E preserves inter-predictor correlations.
    if Z.shape[1] > 0:
        exp_model = LinearRegression(fit_intercept=fit_intercept).fit(Z, X_target)
        x_hat = exp_model.predict(Z)
    else:
        if fit_intercept:
            x_hat = np.full_like(X_target, X_target.mean(axis=0))
        else:
            x_hat = np.zeros_like(X_target)
    x_resids = X_target - x_hat

    # --- Permutation loop ---
    # For each permutation, apply the SAME index shuffle to all rows of
    # the residual matrix E, reconstruct X* = X̂ + shuffled(E), refit the
    # full model, and measure the improvement under H₀.

    def _joint_one_perm(idx: np.ndarray) -> float:
        x_star = x_hat + x_resids[idx]
        perm_features = np.hstack([x_star, Z]) if Z.shape[1] > 0 else x_star
        perm_model = family.fit(perm_features, y_values, fit_intercept)
        perm_preds = family.predict(perm_model, perm_features)
        return float(base_metric - family.fit_metric(y_values, perm_preds))

    if n_jobs == 1:
        perm_improvements = np.zeros(n_perm)
        for i in range(n_perm):
            perm_improvements[i] = _joint_one_perm(perm_indices[i])
    else:
        from joblib import Parallel, delayed

        perm_improvements = np.array(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_joint_one_perm)(perm_indices[i]) for i in range(n_perm)
            )
        )

    return obs_improvement, perm_improvements, metric_type, features_to_test


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #
#
# This function is the single entry point for all three permutation
# methods.  It handles:
#   1. Automatic model selection (linear vs. logistic) based on whether
#      Y contains only {0, 1} values.
#   2. Fitting the observed model to get β̂.
#   3. Computing model diagnostics via statsmodels.
#   4. Pre-generating unique permutation indices (see permutations.py).
#   5. Dispatching to the appropriate method-specific engine.
#   6. Computing Phipson & Smyth-corrected p-values (see pvalues.py).
#   7. Packaging everything into a results dictionary.


def permutation_test_regression(
    X: DataFrameLike,
    y: DataFrameLike,
    n_permutations: int = 5_000,
    precision: int = 3,
    p_value_threshold_one: float = 0.05,
    p_value_threshold_two: float = 0.01,
    method: str = "ter_braak",
    confounders: list[str] | None = None,
    random_state: int | None = None,
    fit_intercept: bool = True,
    family: str = "auto",
    n_jobs: int = 1,
) -> dict:
    """Run a permutation test for regression coefficients.

    By default (``family="auto"``), detects binary vs. continuous
    outcomes and selects logistic or linear regression accordingly.
    Pass an explicit family string (``"linear"``, ``"logistic"``) to
    override auto-detection.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
            Accepts pandas or Polars DataFrames.
        y: Target values of shape ``(n_samples,)``.  When
            ``family="auto"``, binary targets (values in ``{0, 1}``)
            trigger logistic regression; otherwise linear regression
            is used.  Accepts pandas or Polars DataFrames.
        n_permutations: Number of unique permutations.
        precision: Decimal places for reported p-values.
        p_value_threshold_one: First significance level.
        p_value_threshold_two: Second significance level.
        method: One of ``'ter_braak'``, ``'kennedy'``, or
            ``'kennedy_joint'``.
        confounders: Column names of confounders (required for Kennedy
            methods).
        random_state: Seed for reproducibility.
        fit_intercept: Whether to include an intercept term in the
            model.  When ``True`` (default), an intercept is estimated
            in both the observed and permuted models — matching
            ``sklearn.linear_model.LinearRegression(fit_intercept=True)``
            and ``LogisticRegression(fit_intercept=True)``.  Set to
            ``False`` when integrating with a scikit-learn pipeline
            that has already centred or otherwise pre-processed the
            features.
        family: Model family string.  ``"auto"`` (default) detects
            binary {0, 1} targets → logistic, otherwise linear.
            Explicit values (``"linear"``, ``"logistic"``) bypass
            auto-detection and are validated against the response
            via the family's ``validate_y()`` method.
        n_jobs: Number of parallel jobs for the permutation batch-fit
            loop.  ``1`` (default) means sequential execution.
            ``-1`` uses all available CPU cores.  Values > 1 enable
            ``joblib.Parallel(prefer="threads")``, which is effective
            because the underlying BLAS/LAPACK routines and sklearn's
            L-BFGS solver release the GIL.  Ignored when the JAX
            backend is active (JAX uses its own ``vmap``
            vectorisation).

    Returns:
        Dictionary containing coefficients, p-values, diagnostics, and
        method metadata.  Includes ``"model_type"`` set to the
        resolved family name (e.g. ``"linear"`` or ``"logistic"``).

    Raises:
        ValueError: If *method* is not one of the recognised options,
            if *family* is unknown, or if *y* fails the family's
            ``validate_y()`` check.

    References:
        * ter Braak, C. J. F. (1992). Permutation versus bootstrap
          significance tests in multiple regression and ANOVA.
          *Handbook of Statistics*, Vol. 9.
        * Kennedy, P. E. (1995). Randomization tests in econometrics.
          *J. Business & Economic Statistics*, 13(1), 85–94.
        * Phipson, B. & Smyth, G. K. (2010). Permutation p-values
          should never be zero. *Stat. Appl. Genet. Mol. Biol.*,
          9(1), Article 39.
    """
    X = _ensure_pandas_df(X, name="X")
    y = _ensure_pandas_df(y, name="y")

    if confounders is None:
        confounders = []

    # ---- Input validation ------------------------------------------------
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one observation.")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if y.shape[1] > 1:
        raise ValueError(f"y must be a single column, got {y.shape[1]} columns.")

    y_values = np.ravel(y)

    if X.shape[0] != len(y_values):
        raise ValueError(f"X has {X.shape[0]} rows but y has {len(y_values)} elements.")
    if np.isnan(y_values).any():
        raise ValueError(
            "y contains NaN values. Remove or impute missing data before testing."
        )
    if not np.isfinite(y_values).all():
        raise ValueError(
            "y contains infinite values. Remove or correct these before testing."
        )

    # Check X for NaN / Inf / non-numeric
    non_numeric = [
        str(c) for c in X.columns if not np.issubdtype(X[c].dtype, np.number)
    ]
    if non_numeric:
        raise ValueError(
            f"Features have non-numeric dtype: {non_numeric}. "
            "Encode categorical variables before testing."
        )
    if X.isnull().any().any():
        raise ValueError(
            "X contains NaN values. Remove or impute missing data before testing."
        )
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "X contains infinite values. Remove or correct these before testing."
        )

    constant_cols = [str(c) for c in X.columns if X[c].nunique() <= 1]
    if constant_cols:
        raise ValueError(
            f"Features have zero variance: {constant_cols}. "
            "Remove constant columns before testing."
        )

    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}.")

    # Validate confounder names
    if confounders:
        missing = [c for c in confounders if c not in X.columns]
        if missing:
            raise ValueError(f"Confounders not found in X columns: {missing}")

    # Kennedy without confounders is valid but likely a misunderstanding
    if method in ("kennedy", "kennedy_joint") and not confounders:
        warnings.warn(
            f"{method!r} method called without confounders — all features "
            "will be tested. Consider 'ter_braak' for unconditional tests.",
            UserWarning,
            stacklevel=2,
        )

    # ---- Family resolution ------------------------------------------------
    # Resolve the model family via the registry.  "auto" inspects Y for
    # binary {0, 1} values and selects logistic; otherwise linear.
    # Explicit family strings (e.g. "linear", "logistic") bypass
    # auto-detection and map directly to the registered class.
    # The family object encapsulates all model-specific operations so
    # that the dispatch below is family-agnostic wherever possible.
    resolved = resolve_family(family, y_values)

    # Validate Y against the resolved family's constraints.
    # For "auto" this is a no-op (auto-detection already chose the
    # right family).  For explicit families it catches mismatches
    # early — e.g. passing continuous Y with family="logistic".
    if family != "auto":
        resolved.validate_y(y_values)

    # Warn and override n_jobs when the JAX backend is active.
    # JAX uses vmap vectorisation for batch fits, so joblib-based
    # parallelism has no effect.  Resetting to 1 avoids any
    # unexpected behaviour downstream while keeping the user informed.
    if n_jobs != 1:
        from ._backends import resolve_backend

        _backend = resolve_backend()
        if _backend.name == "jax":
            warnings.warn(
                "n_jobs is ignored when the JAX backend is active because "
                "JAX uses vmap vectorisation for batch fits.  Falling back "
                "to n_jobs=1.",
                UserWarning,
                stacklevel=2,
            )
            n_jobs = 1

    # Fit the observed (unpermuted) model to get the original
    # coefficients β̂.  These are the test statistics that will be
    # compared against the permutation null distribution.
    observed_model = resolved.fit(X.values.astype(float), y_values, fit_intercept)
    model_coefs = resolved.coefs(observed_model)

    # Model diagnostics (R², AIC, BIC, etc.) via the family's
    # statsmodels adapter.  Wrapped in try/except because statsmodels
    # can raise on degenerate data (perfect separation, rank deficiency).
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
            diagnostics = resolved.diagnostics(
                X.values.astype(float), y_values, fit_intercept
            )
    except Exception:
        # Degenerate data — return NaN placeholders.
        # The key structure must match the family's diagnostics() output
        # so that the display module can render the table correctly.
        diagnostics = _compute_diagnostics(X, y_values, resolved.name, fit_intercept)

    # Pre-generate unique permutation indices.  This is done once and
    # shared across all features/methods, ensuring consistency and
    # avoiding duplicate permutations (see permutations.py).
    perm_indices = generate_unique_permutations(
        n_samples=len(y_values),
        n_permutations=n_permutations,
        random_state=random_state,
        exclude_identity=True,
    )

    # ---- Dispatch to method-specific engine ----

    if method == "ter_braak":
        if resolved.name == "logistic" and X.shape[1] == 1:
            raise ValueError(
                "ter Braak method with logistic regression requires at least "
                "2 features because the reduced model (dropping the single "
                "feature) has 0 predictors.  Use method='kennedy' with "
                "confounders, or add additional features."
            )
        permuted_coefs = _ter_braak_generic(
            X,
            y_values,
            resolved,
            perm_indices,
            fit_intercept,
            n_jobs=n_jobs,
        )

    elif method == "kennedy":
        permuted_coefs = _kennedy_individual_generic(
            X,
            y_values,
            resolved,
            confounders,
            perm_indices,
            model_coefs,
            fit_intercept,
            n_jobs=n_jobs,
        )

    elif method == "kennedy_joint":
        obs_improvement, perm_improvements, metric_type, features_tested = (
            _kennedy_joint(
                X,
                y_values,
                confounders,
                perm_indices,
                resolved,
                fit_intercept,
                n_jobs=n_jobs,
            )
        )

        # Phipson & Smyth (2010) corrected p-value for the joint test:
        # Count how many permuted improvements >= the observed one, then
        # add 1 to both numerator and denominator.
        p_value = (np.sum(perm_improvements >= obs_improvement) + 1) / (
            n_permutations + 1
        )
        rounded = np.round(p_value, precision)
        val = f"{rounded:.{precision}f}"
        if p_value < p_value_threshold_two:
            p_value_str = f"{val} (**)"
        elif p_value < p_value_threshold_one:
            p_value_str = f"{val} (*)"
        else:
            p_value_str = f"{val} (ns)"

        return {
            "observed_improvement": obs_improvement,
            "permuted_improvements": perm_improvements.tolist(),
            "p_value": p_value,
            "p_value_str": p_value_str,
            "metric_type": metric_type,
            "model_type": resolved.name,
            "features_tested": features_tested,
            "confounders": confounders,
            "p_value_threshold_one": p_value_threshold_one,
            "p_value_threshold_two": p_value_threshold_two,
            "method": method,
            "diagnostics": diagnostics,
        }

    else:
        raise ValueError(
            f"Invalid method '{method}'. Choose 'ter_braak', 'kennedy', or 'kennedy_joint'."
        )

    # Compute empirical (permutation) and classical (asymptotic) p-values.
    # See pvalues.py for the Phipson & Smyth correction details.
    permuted_p_values, classic_p_values, raw_empirical_p, raw_classic_p = (
        calculate_p_values(
            X,
            y,
            permuted_coefs,
            model_coefs,
            precision,
            p_value_threshold_one,
            p_value_threshold_two,
            fit_intercept=fit_intercept,
            family=resolved,
        )
    )

    # For Kennedy method with confounders, mark confounder p-values as
    # N/A since they are controls (not hypotheses being tested).
    # Their coefficients are held constant across permutations, so
    # computing a p-value for them would be meaningless.
    if method == "kennedy" and confounders:
        for i, col in enumerate(X.columns):
            if col in confounders:
                permuted_p_values[i] = "N/A (confounder)"
                classic_p_values[i] = "N/A (confounder)"
                raw_empirical_p[i] = np.nan
                raw_classic_p[i] = np.nan

    # Extended diagnostics — per-predictor and model-level checks.
    extended_diagnostics = compute_all_diagnostics(
        X=X,
        y_values=y_values,
        model_coefs=model_coefs,
        model_type=resolved.name,
        raw_empirical_p=raw_empirical_p,
        raw_classic_p=raw_classic_p,
        n_permutations=n_permutations,
        p_value_threshold=p_value_threshold_one,
        method=method,
        confounders=confounders,
        fit_intercept=fit_intercept,
    )

    return {
        "model_coefs": model_coefs.tolist(),
        "permuted_coefs": permuted_coefs.tolist(),
        "permuted_p_values": permuted_p_values,
        "classic_p_values": classic_p_values,
        "raw_empirical_p": raw_empirical_p,
        "raw_classic_p": raw_classic_p,
        "p_value_threshold_one": p_value_threshold_one,
        "p_value_threshold_two": p_value_threshold_two,
        "method": method,
        "confounders": confounders,
        "model_type": resolved.name,
        "diagnostics": diagnostics,
        "extended_diagnostics": extended_diagnostics,
    }

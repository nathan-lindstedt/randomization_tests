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

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error

from ._compat import DataFrameLike, _ensure_pandas_df
from ._config import get_backend
from .diagnostics import compute_all_diagnostics
from .permutations import generate_unique_permutations
from .pvalues import calculate_p_values

# ------------------------------------------------------------------ #
# Optional JAX import
# ------------------------------------------------------------------ #
# JAX provides two features that dramatically accelerate logistic
# regression permutation loops:
#
# 1. **Automatic differentiation (autodiff)** — jax.grad computes
#    exact gradients of the log-likelihood without hand-coding
#    derivative formulas.  This makes the Newton–Raphson solver both
#    concise and numerically stable.
#
# 2. **vmap (vectorised map)** — transforms a function that processes
#    one permutation into a function that processes all B permutations
#    in a single batched call.  Combined with JIT compilation, this
#    leverages XLA's kernel fusion and, when available, GPU parallelism.
#
# If JAX is not installed the module falls back to sklearn's
# LogisticRegression in a Python loop — correct but slower.
#
# The actual decision of whether to USE JAX is deferred to runtime via
# get_backend(), which checks (in order): programmatic override →
# RANDOMIZATION_TESTS_BACKEND env var → auto-detection.  The import
# here only determines _CAN_IMPORT_JAX (whether it's installed); the
# policy decision is separate.  See _config.py for details.
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    _CAN_IMPORT_JAX = True
except ImportError:
    _CAN_IMPORT_JAX = False


def _use_jax() -> bool:
    """Return True if JAX should be used for the current call."""
    return _CAN_IMPORT_JAX and get_backend() == "jax"


# ------------------------------------------------------------------ #
# JAX helpers (logistic regression via autodiff)
# ------------------------------------------------------------------ #
# Logistic regression maximises the log-likelihood:
#
#   ℓ(β) = Σᵢ [ yᵢ log(pᵢ) + (1 - yᵢ) log(1 - pᵢ) ]
#
# where pᵢ = σ(Xᵢβ) = 1 / (1 + exp(-Xᵢβ)) is the predicted
# probability from the sigmoid function.
#
# Minimising the *negative* log-likelihood (NLL) is equivalent and
# fits the standard optimisation convention.  The functions below
# implement a Newton–Raphson solver:
#
#   β_{t+1} = β_t − H⁻¹(β_t) · ∇ℓ(β_t)
#
# where ∇ℓ is the gradient vector and H is the Hessian matrix.  JAX
# computes both via automatic differentiation — no manual derivation
# of the gradient or Hessian is needed.  The solver converges
# quadratically near the optimum (typically < 10 iterations).

if _CAN_IMPORT_JAX:
    @jit
    def _logistic_nll(
        beta: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the negative log-likelihood for logistic regression.

        Uses the numerically stable form:
            NLL = Σᵢ log(1 + exp(-sᵢ · Xᵢβ))
        where sᵢ = 2yᵢ - 1 ∈ {-1, +1}.  This avoids computing log(0)
        when predicted probabilities saturate near 0 or 1.
        """
        logits = X @ beta
        # Numerically stable: log(1 + exp(-y_signed * logits))
        # where y_signed = 2*y - 1
        return jnp.sum(jnp.logaddexp(0.0, -logits * (2.0 * y - 1.0)))

    _logistic_grad = jit(grad(_logistic_nll))

    @jit
    def _logistic_hessian_diag(
        beta: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the Hessian of the logistic NLL.

        The Hessian of the NLL is X'WX, where W = diag(pᵢ(1-pᵢ)) is
        the diagonal matrix of variance weights.  This is always
        positive semi-definite, guaranteeing the Newton step descends.
        """
        p = jax.nn.sigmoid(X @ beta)
        W = p * (1.0 - p)
        return (X.T * W[None, :]) @ X

    def _fit_logistic_batch_jax(
        X_base: np.ndarray,
        Y_matrix: np.ndarray,
        max_iter: int = 100,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Fit logistic regression for many Y vectors at once using vmap.

        vmap transforms the single-permutation solver into a batched
        version that processes all B permutations simultaneously.  Each
        column of Y_matrix is an independent binary response vector;
        the design matrix X_base is shared across all solves.

        Args:
            X_base: Design matrix of shape ``(n, p)`` shared across
                permutations.
            Y_matrix: Matrix of shape ``(B, n)`` where each row is a
                permuted response vector.
            max_iter: Maximum Newton iterations per solve.

        Returns:
            Coefficient matrix of shape ``(B, p)``.
        """
        # When fit_intercept is True, prepend an intercept column to
        # match sklearn's default.  The solver returns [β₀, β₁, …, βₚ];
        # we slice off β₀ so the caller sees only feature coefficients.
        if fit_intercept:
            ones = np.ones((X_base.shape[0], 1), dtype=X_base.dtype)
            X_aug = np.hstack([ones, X_base])
        else:
            X_aug = X_base
        X_j = jnp.array(X_aug, dtype=jnp.float32)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float32)

        def _solve_one(y_vec):
            beta = jnp.zeros(X_j.shape[1], dtype=jnp.float32)
            for _ in range(max_iter):
                g = _logistic_grad(beta, X_j, y_vec)
                H = _logistic_hessian_diag(beta, X_j, y_vec)
                step = jnp.linalg.solve(H, g)
                beta = beta - step
            return beta

        # vmap across the batch dimension (rows of Y_j)
        batched_solve = jit(vmap(_solve_one))
        all_coefs = np.asarray(batched_solve(Y_j))
        return all_coefs[:, 1:] if fit_intercept else all_coefs


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
    is_binary: bool,
    fit_intercept: bool = True,
) -> dict:
    """Compute model diagnostics via statsmodels.

    Fits the full model (with all features) using statsmodels and
    extracts summary statistics for display in the results table.

    Args:
        X: Feature matrix.
        y_values: Response vector.
        is_binary: Whether the response is binary (logistic) or
            continuous (OLS).
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

    if is_binary:
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

def _ter_braak_linear(
    X: pd.DataFrame,
    y_values: np.ndarray,
    perm_indices: np.ndarray,
    fit_intercept: bool = True,
) -> np.ndarray:
    """Vectorised ter Braak for OLS.

    For each feature *j*:

    1. Fit reduced OLS model (drop column *j*) → predicted values +
       residuals.
    2. Build all *B* permuted Y vectors in one shot using fancy indexing.
    3. Batch-compute full-model coefficients via pseudoinverse.

    Args:
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

    for j in range(n_features):
        # Step 1: Fit the reduced model Y ~ X_{-j}.
        # Drop column j from the design matrix and compute OLS predictions.
        # When fit_intercept is True, an intercept column is included so
        # that the reduced-model coefficients live in the same parameter
        # space as the sklearn observed coefficients.
        X_red = np.delete(X_np, j, axis=1)
        if fit_intercept:
            X_red_aug = np.column_stack([np.ones(n), X_red])
        else:
            X_red_aug = X_red
        pinv_red = np.linalg.pinv(X_red_aug)
        preds_red = X_red_aug @ (pinv_red @ y_values)
        resids_red = y_values - preds_red

        # Step 2: Build all B permuted Y vectors simultaneously.
        # perm_indices has shape (B, n); fancy-indexing resids_red
        # gives a (B, n) matrix where each row is a permuted residual
        # vector.  Adding the (unpermuted) reduced-model predictions
        # yields Y* = ŷ_{-j} + π(e_{-j}) for all B permutations at once.
        permuted_resids = resids_red[perm_indices]
        Y_perm = preds_red[np.newaxis, :] + permuted_resids  # (B, n)

        # Step 3: Batch OLS — compute β̂*_b = pinv(X) @ Y*_b for all b.
        # _batch_ols_coefs returns shape (B, p); we extract column j.
        all_coefs = _batch_ols_coefs(X_np, Y_perm, fit_intercept)  # (B, p)
        result[:, j] = all_coefs[:, j]

    return result


def _ter_braak_logistic(
    X: pd.DataFrame,
    y_values: np.ndarray,
    perm_indices: np.ndarray,
    fit_intercept: bool = True,
) -> np.ndarray:
    """ter Braak for logistic regression.

    Uses a GLM-faithful adaptation: the reduced model is logistic,
    residuals on the probability scale are permuted, and Bernoulli
    sampling converts permuted Y* back to binary.

    When JAX is available the full-model refits are batched via
    ``vmap``.  Otherwise falls back to an sklearn loop.
    """
    # The key challenge for logistic regression is that Y ∈ {0, 1}, not
    # a continuous variable.  We cannot simply add permuted residuals to
    # ŷ_{-j} and expect a valid binary outcome.  The adaptation works as
    # follows:
    #
    #   1. Fit the reduced logistic model to get predicted probabilities
    #      p̂_{-j} = P(Y=1 | X_{-j}).
    #
    #   2. Compute "probability-scale" residuals: e = Y - p̂_{-j}.
    #      These range from (−1, 0) for Y=0 and (0, +1) for Y=1.
    #
    #   3. Permute the residuals and add them back:
    #        p* = clip(p̂_{-j} + π(e), 0.001, 0.999)
    #      The result is a continuous probability, not a binary outcome.
    #
    #   4. Generate Y* ~ Bernoulli(p*): draw independent coin flips with
    #      probability p* to obtain a valid binary response.  This
    #      preserves the probabilistic structure of the logistic model
    #      while incorporating the permuted noise.
    #
    #   5. Refit the full logistic model on (X, Y*) to get β*.

    X_np = X.values.astype(float)
    n_perm, n = perm_indices.shape
    n_features = X_np.shape[1]
    result = np.zeros((n_perm, n_features))

    # Derive a deterministic seed from the permutation indices so that
    # the Bernoulli sampling is reproducible given the same permutations.
    rng = np.random.default_rng(int(perm_indices[0, 0]))

    for j in range(n_features):
        # Step 1: Reduced logistic model — drop feature j.
        X_red = np.delete(X_np, j, axis=1)
        model_red = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=5_000,
            fit_intercept=fit_intercept,
        )
        model_red.fit(X_red, y_values)
        preds_red = model_red.predict_proba(X_red)[:, 1]  # P(Y=1 | X_{-j})

        # Step 2: Probability-scale residuals.
        resids_red = y_values - preds_red

        # Steps 3-4: Permute residuals, clip to valid probability range,
        # then draw binary outcomes from Bernoulli(p*).
        permuted_resids = resids_red[perm_indices]  # (B, n)
        Y_perm_probs = np.clip(preds_red[np.newaxis, :] + permuted_resids, 0.001, 0.999)
        Y_perm_binary = rng.binomial(1, Y_perm_probs)  # (B, n)

        # Step 5: Refit the full logistic model for all B permutations.
        if _use_jax():
            # JAX path: batch all B logistic fits via vmap'd Newton solver.
            all_coefs = _fit_logistic_batch_jax(
                X_np, Y_perm_binary, fit_intercept=fit_intercept,
            )
            result[:, j] = all_coefs[:, j]
        else:
            # Fallback: sklearn loop (slower but always available).
            model_cls = LogisticRegression(
                penalty=None, solver="lbfgs", max_iter=5_000,
                fit_intercept=fit_intercept,
            )
            for p in range(n_perm):
                model_cls.fit(X_np, Y_perm_binary[p])
                result[p, j] = model_cls.coef_.flatten()[j]

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

def _kennedy_individual_linear(
    X: pd.DataFrame,
    y_values: np.ndarray,
    confounders: list[str],
    perm_indices: np.ndarray,
    model_coefs: np.ndarray,
    fit_intercept: bool = True,
) -> np.ndarray:
    """Vectorised Kennedy individual for OLS."""
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
        #   X̂_j = Z_aug @ (pinv(Z_aug) @ X_j) = Z_aug @ γ̂
        #   eₓⱼ = X_j - X̂_j
        # When fit_intercept is True, an intercept column is included
        # so that the exposure-model residuals are mean-centred by
        # construction, matching sklearn's LinearRegression default.
        if Z.shape[1] > 0:
            if fit_intercept:
                Z_aug = np.column_stack([np.ones(n), Z])  # add intercept
            else:
                Z_aug = Z
            pinv_z = np.linalg.pinv(Z_aug)
            x_hat = Z_aug @ (pinv_z @ x_target)
        else:
            if fit_intercept:
                # No confounders: X̂_j is just the mean of X_j.
                x_hat = np.full_like(x_target, x_target.mean())
            else:
                # No confounders and no intercept: X̂_j = 0.
                x_hat = np.zeros_like(x_target)
        x_resids = (x_target - x_hat).ravel()  # (n,)

        # Step 2: Permute exposure residuals for all B permutations.
        shuffled = x_resids[perm_indices]  # (B, n)

        # Step 3: Reconstruct X*_j = X̂_j + π(eₓⱼ) and refit.
        # We build all B design matrices at once by broadcasting the
        # original X across the batch dimension, then replacing column j.
        X_perm_all = np.broadcast_to(X_np, (n_perm, n, n_features)).copy()
        X_perm_all[:, :, feat_idx] = x_hat.ravel()[np.newaxis, :] + shuffled  # (B, n)

        # Note: Unlike the ter Braak path, we CANNOT use a single
        # pseudoinverse here because the design matrix itself changes in
        # each permutation (column j is different).  Each permutation
        # requires its own lstsq solve.  When fit_intercept is True, an
        # intercept column is prepended so that the permuted coefficients
        # live in the same parameter space as the sklearn observed
        # coefficients.
        if fit_intercept:
            ones_col = np.ones((n, 1))
            for p in range(n_perm):
                X_perm_aug = np.column_stack([ones_col, X_perm_all[p]])
                coefs, _, _, _ = np.linalg.lstsq(X_perm_aug, y_values, rcond=None)
                result[p, feat_idx] = coefs[feat_idx + 1]  # +1 to skip intercept
        else:
            for p in range(n_perm):
                coefs, _, _, _ = np.linalg.lstsq(X_perm_all[p], y_values, rcond=None)
                result[p, feat_idx] = coefs[feat_idx]

    return result


def _kennedy_individual_logistic(
    X: pd.DataFrame,
    y_values: np.ndarray,
    confounders: list[str],
    perm_indices: np.ndarray,
    model_coefs: np.ndarray,
    fit_intercept: bool = True,
) -> np.ndarray:
    """Kennedy individual for logistic regression.

    The exposure model is always linear (X_j on Z is a continuous
    regression regardless of whether Y is binary).  What changes is
    the outcome model: logistic instead of OLS.
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

    for i, col in enumerate(X.columns):
        if col in confounders:
            result[:, i] = model_coefs[i]

    for feature in features_to_test:
        feat_idx = X.columns.get_loc(feature)
        x_target = X[[feature]].values

        # Exposure model — same linear regression as the OLS path.
        # The exposure model is always linear because X_j is continuous
        # even when Y is binary.
        if Z.shape[1] > 0:
            exp_model = LinearRegression(fit_intercept=fit_intercept).fit(Z, x_target)
            x_hat = exp_model.predict(Z)
        else:
            if fit_intercept:
                x_hat = np.full_like(x_target, x_target.mean())
            else:
                x_hat = np.zeros_like(x_target)
        x_resids = (x_target - x_hat).ravel()

        # Permute exposure residuals and reconstruct X*_j.
        shuffled = x_resids[perm_indices]  # (B, n)

        if _use_jax():
            # JAX path: build all B design matrices, then vmap the
            # Newton solver across the batch dimension.  Each element of
            # X_j gets the same y vector but a different X matrix (column
            # j replaced with X*_j).
            X_perm_all = np.broadcast_to(X_np, (n_perm, n, n_features)).copy()
            X_perm_all[:, :, feat_idx] = x_hat.ravel()[np.newaxis, :] + shuffled

            # Prepend intercept column to match sklearn's default
            # when fit_intercept is True.  Slice off β₀ after solving.
            if fit_intercept:
                ones = np.ones((n_perm, n, 1), dtype=X_perm_all.dtype)
                X_perm_aug = np.concatenate([ones, X_perm_all], axis=2)
            else:
                X_perm_aug = X_perm_all
            X_j = jnp.array(X_perm_aug, dtype=jnp.float32)
            y_j = jnp.array(y_values, dtype=jnp.float32)

            def _solve_one(X_single, _y=y_j):
                beta = jnp.zeros(X_single.shape[1], dtype=jnp.float32)
                for _ in range(100):
                    g = _logistic_grad(beta, X_single, _y)
                    H = _logistic_hessian_diag(beta, X_single, _y)
                    beta = beta - jnp.linalg.solve(H, g)
                return beta

            batched = jit(vmap(_solve_one))
            all_coefs_raw = np.asarray(batched(X_j))
            all_coefs = all_coefs_raw[:, 1:] if fit_intercept else all_coefs_raw
            result[:, feat_idx] = all_coefs[:, feat_idx]
        else:
            # Fallback: sklearn loop.
            model_cls = LogisticRegression(
                penalty=None, solver="lbfgs", max_iter=5_000,
                fit_intercept=fit_intercept,
            )
            for p in range(n_perm):
                X_perm = X.copy()
                X_perm.iloc[:, feat_idx] = x_hat.ravel() + shuffled[p]
                model_cls.fit(X_perm.values, y_values)
                result[p, feat_idx] = model_cls.coef_.flatten()[feat_idx]

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
    is_binary: bool,
    fit_intercept: bool = True,
) -> tuple[float, np.ndarray, str, list[str]]:
    """Kennedy joint test.

    Returns:
        A ``(obs_improvement, perm_improvements, metric_type,
        features_tested)`` tuple.
    """
    features_to_test = [c for c in X.columns if c not in confounders]

    # Choose the fit metric based on the outcome type.
    # Linear: RSS = MSE × n (lower is better → reduction = improvement).
    # Logistic: Deviance = 2 × total binary cross-entropy (lower is better).
    if is_binary:

        def model_cls():
            return LogisticRegression(
                penalty=None, solver="lbfgs", max_iter=5_000,
                fit_intercept=fit_intercept,
            )

        def get_metric(y_true, y_pred_proba):
            return 2 * log_loss(y_true, y_pred_proba, normalize=False)

        metric_type = "Deviance Reduction"
    else:

        def model_cls():
            return LinearRegression(fit_intercept=fit_intercept)

        def get_metric(y_true, y_pred):
            return mean_squared_error(y_true, y_pred) * len(y_true)

        metric_type = "RSS Reduction"

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
        reduced = model_cls().fit(Z, y_values)
        preds_reduced = reduced.predict_proba(Z) if is_binary else reduced.predict(Z)
    else:
        # No confounders: the "reduced model" is just the intercept
        # (the grand mean for linear, the base rate for logistic) when
        # fit_intercept is True.  When fit_intercept is False, the
        # reduced model predicts zero for all observations.
        if fit_intercept:
            if is_binary:
                mean_y = np.mean(y_values)
                preds_reduced = np.column_stack([1 - mean_y * np.ones(n), mean_y * np.ones(n)])
            else:
                preds_reduced = np.full(n, np.mean(y_values), dtype=float)
        else:
            if is_binary:
                preds_reduced = np.column_stack([0.5 * np.ones(n), 0.5 * np.ones(n)])
            else:
                preds_reduced = np.zeros(n, dtype=float)

    base_metric = get_metric(y_values, preds_reduced)

    # --- Full model (confounders + predictors of interest) ---
    # The observed improvement is how much better the full model fits.
    full_features = np.hstack([X_target, Z]) if Z.shape[1] > 0 else X_target
    full_model = model_cls().fit(full_features, y_values)
    preds_full = full_model.predict_proba(full_features) if is_binary else full_model.predict(full_features)
    obs_improvement = base_metric - get_metric(y_values, preds_full)

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
    perm_improvements = np.zeros(n_perm)

    for i in range(n_perm):
        # Row-wise shuffle: perm_indices[i] reorders all columns of E
        # identically, preserving inter-predictor correlation structure.
        x_star = x_hat + x_resids[perm_indices[i]]
        perm_features = np.hstack([x_star, Z]) if Z.shape[1] > 0 else x_star
        perm_model = model_cls().fit(perm_features, y_values)
        perm_preds = perm_model.predict_proba(perm_features) if is_binary else perm_model.predict(perm_features)
        perm_improvements[i] = base_metric - get_metric(y_values, perm_preds)

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
    X: "DataFrameLike",
    y: "DataFrameLike",
    n_permutations: int = 5_000,
    precision: int = 3,
    p_value_threshold_one: float = 0.05,
    p_value_threshold_two: float = 0.01,
    method: str = "ter_braak",
    confounders: list[str] | None = None,
    random_state: int | None = None,
    fit_intercept: bool = True,
) -> dict:
    """Run a permutation test for regression coefficients.

    Automatically detects binary vs. continuous outcomes and uses
    logistic or linear regression accordingly.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
            Accepts pandas or Polars DataFrames.
        y: Target values of shape ``(n_samples,)``.  Binary targets
            (values in ``{0, 1}``) trigger logistic regression;
            otherwise linear regression is used.  Accepts pandas or
            Polars DataFrames.
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

    Returns:
        Dictionary containing coefficients, p-values, diagnostics, and
        method metadata.

    Raises:
        ValueError: If *method* is not one of the recognised options.

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

    y_values = np.ravel(y)
    unique_y = np.unique(y_values)
    # Auto-detect binary outcome: Y must contain exactly two unique
    # values, both in {0, 1}.  This triggers logistic regression;
    # otherwise OLS is used.
    is_binary = bool((len(unique_y) == 2) and np.all(np.isin(unique_y, [0, 1])))

    # Fit the observed (unpermuted) model to get the original
    # coefficients β̂.  These are the test statistics that will be
    # compared against the permutation null distribution.
    if is_binary:
        model = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=5_000,
            fit_intercept=fit_intercept,
        )
    else:
        model = LinearRegression(fit_intercept=fit_intercept)

    model.fit(X, y_values)
    model_coefs = model.coef_.flatten() if is_binary else np.ravel(model.coef_)

    # Model diagnostics (R², AIC, BIC, etc.) from statsmodels.
    diagnostics = _compute_diagnostics(X, y_values, is_binary, fit_intercept)

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
        if is_binary:
            permuted_coefs = _ter_braak_logistic(
                X, y_values, perm_indices, fit_intercept,
            )
        else:
            permuted_coefs = _ter_braak_linear(
                X, y_values, perm_indices, fit_intercept,
            )

    elif method == "kennedy":
        if is_binary:
            permuted_coefs = _kennedy_individual_logistic(
                X, y_values, confounders, perm_indices, model_coefs,
                fit_intercept,
            )
        else:
            permuted_coefs = _kennedy_individual_linear(
                X, y_values, confounders, perm_indices, model_coefs,
                fit_intercept,
            )

    elif method == "kennedy_joint":
        obs_improvement, perm_improvements, metric_type, features_tested = _kennedy_joint(
            X, y_values, confounders, perm_indices, is_binary, fit_intercept,
        )

        # Phipson & Smyth (2010) corrected p-value for the joint test:
        # Count how many permuted improvements >= the observed one, then
        # add 1 to both numerator and denominator.
        p_value = (np.sum(perm_improvements >= obs_improvement) + 1) / (n_permutations + 1)
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
            "p_value": p_value,
            "p_value_str": p_value_str,
            "metric_type": metric_type,
            "model_type": "logistic" if is_binary else "linear",
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
    permuted_p_values, classic_p_values, raw_empirical_p, raw_classic_p = calculate_p_values(
        X, y, permuted_coefs, model_coefs,
        precision, p_value_threshold_one, p_value_threshold_two,
        fit_intercept=fit_intercept,
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
        is_binary=is_binary,
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
        "permuted_p_values": permuted_p_values,
        "classic_p_values": classic_p_values,
        "raw_empirical_p": raw_empirical_p,
        "raw_classic_p": raw_classic_p,
        "p_value_threshold_one": p_value_threshold_one,
        "p_value_threshold_two": p_value_threshold_two,
        "method": method,
        "model_type": "logistic" if is_binary else "linear",
        "diagnostics": diagnostics,
        "extended_diagnostics": extended_diagnostics,
    }

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

This module implements five permutation strategies:

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

4. **Freedman–Lane (1983) individual** – Permute full-model residuals,
   add to reduced-model fitted values.  For each non-confounder
   predictor X_j:

   a) Fit the **full** model Y ~ X + Z to get residuals from the
      complete specification.
   b) Fit the **reduced** model Y ~ Z (confounders only) to get
      fitted values that capture only the confounder contribution.
   c) Permute the full-model residuals and construct
      Y* = ŷ_reduced + π(ê_full).
   d) Refit the full model on (X, Y*) to get β*_j.

   Intuition: By using full-model residuals (which have smaller
   variance) rather than reduced-model residuals, the permutation
   null distribution is tighter and the test has better power when
   predictors are correlated (Anderson & Legendre 1999; Winkler
   et al. 2014).  When ``confounders=[]`` the procedure reduces to
   a special case where the reduced model is intercept-only; users
   should prefer ``"ter_braak"`` for unconditional tests.

5. **Freedman–Lane (1983) joint** – Test collective predictive
   improvement using full-model residuals.  Same structure as
   Kennedy joint but permuting Y (via full-model residuals) instead
   of X (via exposure-model residuals):

   a) Fit the full model and reduced model on observed data.
   b) Permute full-model residuals, construct Y*.
   c) Refit both reduced and full models on Y*.
   d) Compute improvement = metric(reduced on Y*) − metric(full on Y*).

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

    Freedman, D. & Lane, D. (1983). A nonstochastic interpretation
    of reported significance levels. *Journal of Business & Economic
    Statistics*, 1(4), 292–298.

    Anderson, M. J. & Legendre, P. (1999). An empirical comparison
    of permutation methods for tests of partial regression
    coefficients in a linear model. *Journal of Statistical
    Computation and Simulation*, 62(3), 271–303.

    Winkler, A. M. et al. (2014). Permutation inference for the
    general linear model. *NeuroImage*, 92, 381–397.

    Phipson, B. & Smyth, G. K. (2010). Permutation p-values should
    never be zero. *Stat. Appl. Genet. Mol. Biol.*, 9(1), Article 39.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning as SmConvergenceWarning,
)
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationWarning,
)

from ._compat import DataFrameLike, _ensure_pandas_df
from ._results import IndividualTestResult, JointTestResult
from .diagnostics import compute_all_diagnostics
from .families import ModelFamily, resolve_family
from .permutations import generate_unique_permutations
from .pvalues import calculate_p_values

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
# Freedman–Lane (1983) individual — full-model residual permutation
# ------------------------------------------------------------------ #
#
# The Freedman–Lane procedure permutes residuals from the FULL model
# (the complete specification Y ~ X + Z) rather than the reduced model
# (ter Braak) or the exposure model (Kennedy).  Because full-model
# residuals have smaller variance (they condition on more predictors),
# the resulting null distribution is tighter and the test has better
# power when predictors are correlated.
#
# Algorithm:
#
#   1. Fit the **full model** Y ~ X + Z (all features including any
#      confounders) to get predicted values ŷ_full and residuals
#      ê_full = Y − ŷ_full.  This step is done once, outside any
#      feature loop.
#
#   2. Fit the **reduced model** Y ~ Z (confounders only).  When
#      confounders are specified, this is an actual model fit.  When
#      confounders are empty, the reduced model is intercept-only
#      (grand mean for linear, base rate for logistic).
#
#   3. For each permutation b = 1, …, B, randomly reorder ê_full to
#      get π_b(ê_full), then construct:
#
#        Y*_b = ŷ_reduced + π_b(ê_full)
#
#      via ``family.reconstruct_y()``.  This preserves the confounder
#      contribution while destroying any real association between Y
#      and the non-confounder predictors.
#
#   4. Refit the full model Y*_b ~ X + Z and extract β*_{b,j} for
#      each non-confounder feature j.
#
# The critical difference from Kennedy: Kennedy permutes X-residuals
# (exposure model) while keeping Y fixed.  Freedman–Lane permutes
# Y-residuals (from the full model) while keeping X fixed.  The
# Freedman–Lane approach gives better power when the variable of
# interest is correlated with confounders (Anderson & Legendre 1999;
# Winkler et al. 2014).
#
# When confounders are empty, the reduced model is intercept-only and
# the full model is Y ~ X, so the procedure tests predictors against
# an intercept-only null.  In this case ``"ter_braak"`` is preferable
# because it conditions on all other predictors individually.


def _freedman_lane_individual_generic(
    X: pd.DataFrame,
    y_values: np.ndarray,
    family: ModelFamily,
    confounders: list[str],
    perm_indices: np.ndarray,
    model_coefs: np.ndarray,
    fit_intercept: bool = True,
    n_jobs: int = 1,
) -> np.ndarray:
    """Family-generic Freedman–Lane individual permutation engine.

    Permutes full-model residuals and adds them to reduced-model fitted
    values to construct Y*.  The full model is then refitted on each
    Y* to obtain the null distribution of coefficients.

    Unlike ter Braak (which uses a per-feature reduced model) and
    Kennedy (which permutes exposure residuals), this procedure uses a
    single reduced model (confounders only) and a single set of
    permuted Y* vectors for all features.  This means the batch refit
    is done once rather than once per feature.

    Algorithm:

    1. **Full model** — ``family.fit(X, y)`` on all features.
       ``family.residuals()`` gives ê_full.
    2. **Reduced model** — ``family.fit(Z, y)`` on confounders only.
       ``family.predict()`` gives ŷ_reduced.
    3. **Permute** — ê_full[perm_indices] gives B permuted residual
       vectors.
    4. **Reconstruct** — ``family.reconstruct_y(ŷ_reduced, π(ê_full), rng)``
       yields Y* of shape ``(B, n)``.
    5. **Batch refit** — ``family.batch_fit(X, Y*, fit_intercept)``
       produces ``(B, n_features)`` permuted coefficients.

    Args:
        X: Feature matrix as a pandas DataFrame.
        y_values: Response vector of shape ``(n,)``.
        family: Resolved ``ModelFamily`` instance.
        confounders: List of confounder column names.
        perm_indices: Pre-generated permutation indices ``(B, n)``.
        model_coefs: Observed model coefficients ``(p,)`` — used to
            fill confounder slots (they are not being tested).
        fit_intercept: Whether to include an intercept.
        n_jobs: Number of parallel jobs for the batch-fit step.

    Returns:
        Array of shape ``(B, n_features)`` with permuted coefficients.
        Confounder columns are filled with the observed coefficient
        value so that their empirical p-values are always 1.0.
    """
    X_np = X.values.astype(float)
    n_perm, n = perm_indices.shape

    # Derive a deterministic RNG from the permutation indices so that
    # any stochastic reconstruction step (e.g. Bernoulli sampling for
    # logistic) is reproducible given the same permutations.
    rng = np.random.default_rng(int(perm_indices[0, 0]))

    # Step 1: Fit the FULL model Y ~ X (all features) and get residuals.
    # The full model is the same model already fit for the observed
    # coefficients, so we refit here for clean access to residuals.
    full_model = family.fit(X_np, y_values, fit_intercept)
    full_resids = family.residuals(full_model, X_np, y_values)  # (n,)

    # Step 2: Fit the REDUCED model Y ~ Z (confounders only).
    # This captures only the confounder contribution to Y.
    if confounders:
        conf_idx = [X.columns.get_loc(c) for c in confounders]
        Z = X_np[:, conf_idx]
        reduced_model = family.fit(Z, y_values, fit_intercept)
        preds_reduced = family.predict(reduced_model, Z)  # (n,)
    else:
        # No confounders: reduced model is intercept-only.
        if fit_intercept:
            preds_reduced = np.full(n, np.mean(y_values), dtype=float)
        else:
            preds_reduced = np.zeros(n, dtype=float)

    # Step 3: Permute full-model residuals.
    permuted_resids = full_resids[perm_indices]  # (B, n)

    # Step 4: Reconstruct Y* = ŷ_reduced + π(ê_full).
    # family.reconstruct_y handles family-specific transformation:
    #   - Linear: Y* = ŷ_reduced + π(ê_full)  (deterministic)
    #   - Logistic: clip to [0.001, 0.999], then Bernoulli sample
    Y_perm = family.reconstruct_y(
        preds_reduced[np.newaxis, :],  # (1, n) for broadcasting
        permuted_resids,  # (B, n)
        rng,
    )  # (B, n)

    # Step 5: Batch-refit the full model on all B permuted Y vectors.
    all_coefs = np.array(
        family.batch_fit(X_np, Y_perm, fit_intercept, n_jobs=n_jobs)
    )  # (B, n_features) — writable copy

    # Fill confounder columns with the observed coefficient so their
    # empirical p-values are trivially 1.0.  The dispatch logic will
    # then mark them as "N/A (confounder)".
    if confounders:
        for i, col in enumerate(X.columns):
            if col in confounders:
                all_coefs[:, i] = model_coefs[i]

    return all_coefs


# ------------------------------------------------------------------ #
# Freedman–Lane (1983) joint — collective predictive improvement test
# ------------------------------------------------------------------ #
#
# Same idea as Kennedy joint, but permuting Y (via full-model
# residuals) instead of X (via exposure-model residuals).
#
# Because Y* changes each permutation, both the reduced and full
# models must be refit on every permuted Y*.  This is the key
# structural difference from Kennedy joint, where the reduced model
# metric is constant because Y is fixed.
#
# Algorithm:
#   1. Fit full model Y ~ X + Z and reduced model Y ~ Z.
#   2. Compute the observed improvement:
#      T_obs = metric(reduced) − metric(full).
#   3. Get full-model residuals ê = Y − ŷ_full.
#   4. For each permutation b:
#      a) Y*_b = ŷ_reduced + π_b(ê)
#      b) Refit reduced model: metric_red = family.fit_metric(Y*_b, reduced_preds_on_Y*_b)
#      c) Refit full model: metric_full = family.fit_metric(Y*_b, full_preds_on_Y*_b)
#      d) T*_b = metric_red − metric_full
#   5. p-value = (#{T*_b ≥ T_obs} + 1) / (B + 1)


def _freedman_lane_joint(
    X: pd.DataFrame,
    y_values: np.ndarray,
    confounders: list[str],
    perm_indices: np.ndarray,
    family: ModelFamily,
    fit_intercept: bool = True,
    n_jobs: int = 1,
) -> tuple[float, np.ndarray, str, list[str]]:
    """Family-generic Freedman–Lane joint test.

    Tests whether non-confounder predictors collectively contribute
    significant predictive information beyond confounders, using
    full-model residuals to construct Y*.

    Unlike Kennedy joint (which permutes exposure residuals and keeps
    Y fixed), Freedman–Lane joint permutes Y-residuals and keeps X
    fixed.  Because Y* changes, both reduced and full models are
    refit for each permutation.

    Returns:
        A ``(obs_improvement, perm_improvements, metric_type,
        features_tested)`` tuple.
    """
    features_to_test = [c for c in X.columns if c not in confounders]

    # Metric label for display — derived from the family protocol.
    metric_type = family.metric_label

    X_np = X.values.astype(float)
    n_perm, n = perm_indices.shape

    # Derive a deterministic RNG for stochastic Y* reconstruction.
    rng = np.random.default_rng(int(perm_indices[0, 0]))

    if confounders:
        conf_idx = [X.columns.get_loc(c) for c in confounders]
        Z = X_np[:, conf_idx]
    else:
        Z = np.zeros((n, 0))

    # --- Observed reduced model (confounders only) ---
    if Z.shape[1] > 0:
        reduced_model = family.fit(Z, y_values, fit_intercept)
        preds_reduced = family.predict(reduced_model, Z)
    else:
        if fit_intercept:
            preds_reduced = np.full(n, np.mean(y_values), dtype=float)
        else:
            preds_reduced = np.zeros(n, dtype=float)

    base_metric = family.fit_metric(y_values, preds_reduced)

    # --- Observed full model (all features) ---
    full_model = family.fit(X_np, y_values, fit_intercept)
    preds_full = family.predict(full_model, X_np)
    obs_improvement = base_metric - family.fit_metric(y_values, preds_full)

    # --- Full-model residuals ---
    full_resids = family.residuals(full_model, X_np, y_values)  # (n,)

    # --- Permutation loop ---
    # For each permutation, construct Y* from permuted full-model
    # residuals, then refit BOTH reduced and full models on Y* and
    # compute the improvement under H₀.

    def _fl_joint_one_perm(idx: np.ndarray) -> float:
        perm_resids = full_resids[idx]  # (n,)
        y_star = family.reconstruct_y(
            preds_reduced[np.newaxis, :],  # (1, n)
            perm_resids[np.newaxis, :],  # (1, n)
            rng,
        ).ravel()  # (n,)

        # Refit reduced model on Y*
        if Z.shape[1] > 0:
            red_model_star = family.fit(Z, y_star, fit_intercept)
            red_preds_star = family.predict(red_model_star, Z)
        else:
            if fit_intercept:
                red_preds_star = np.full(n, np.mean(y_star), dtype=float)
            else:
                red_preds_star = np.zeros(n, dtype=float)
        metric_red = family.fit_metric(y_star, red_preds_star)

        # Refit full model on Y*
        full_model_star = family.fit(X_np, y_star, fit_intercept)
        full_preds_star = family.predict(full_model_star, X_np)
        metric_full = family.fit_metric(y_star, full_preds_star)

        return float(metric_red - metric_full)

    if n_jobs == 1:
        perm_improvements = np.zeros(n_perm)
        for i in range(n_perm):
            perm_improvements[i] = _fl_joint_one_perm(perm_indices[i])
    else:
        from joblib import Parallel, delayed

        perm_improvements = np.array(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fl_joint_one_perm)(perm_indices[i]) for i in range(n_perm)
            )
        )

    return obs_improvement, perm_improvements, metric_type, features_to_test


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

    # Metric label for display — derived from the family protocol.
    metric_type = family.metric_label

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
) -> IndividualTestResult | JointTestResult:
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
        method: One of ``'ter_braak'``, ``'kennedy'``,
            ``'kennedy_joint'``, ``'freedman_lane'``, or
            ``'freedman_lane_joint'``.
        confounders: Column names of confounders (required for Kennedy
            and Freedman–Lane methods).
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

    # Freedman–Lane without confounders is equivalent to testing against
    # an intercept-only null.  ter Braak is more appropriate because it
    # conditions on all other predictors individually.
    if method in ("freedman_lane", "freedman_lane_joint") and not confounders:
        warnings.warn(
            f"{method!r} method called without confounders — the reduced "
            "model is intercept-only, which yields less power than "
            "conditioning on other predictors. Consider 'ter_braak' for "
            "unconditional tests.",
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

    # Resolve the active backend once — used for the n_jobs/JAX check
    # below and included in every result dict for provenance.
    from ._backends import resolve_backend

    _backend = resolve_backend()
    backend_name = _backend.name

    # Warn and override n_jobs when the JAX backend is active.
    # JAX uses vmap vectorisation for batch fits, so joblib-based
    # parallelism has no effect.  Resetting to 1 avoids any
    # unexpected behaviour downstream while keeping the user informed.
    if n_jobs != 1 and backend_name == "jax":
        warnings.warn(
            "n_jobs is ignored when the JAX backend is active because "
            "JAX uses vmap vectorisation for batch fits.  Falling back "
            "to n_jobs=1.",
            UserWarning,
            stacklevel=2,
        )
        n_jobs = 1

    # Warn when n_jobs has no effect on vectorised OLS paths.
    # The ter_braak and freedman_lane individual methods use batch_ols,
    # which is a single pinv @ Y.T BLAS multiply — there is no loop to
    # parallelise.  Kennedy individual, both joint methods, and all
    # logistic paths DO benefit from n_jobs.
    if (
        n_jobs != 1
        and backend_name == "numpy"
        and resolved.name == "linear"
        and method in ("ter_braak", "freedman_lane")
    ):
        warnings.warn(
            "n_jobs has no effect for linear ter_braak/freedman_lane "
            "because OLS batch fitting is already a single vectorised "
            "BLAS operation (pinv @ Y.T).  Falling back to n_jobs=1.  "
            "n_jobs provides genuine parallelism for logistic families, "
            "Kennedy individual, and joint methods.",
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
        diagnostics = {
            "n_observations": len(y_values),
            "n_features": X.shape[1],
        }

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

        p_value = float(
            (np.sum(perm_improvements >= obs_improvement) + 1) / (n_permutations + 1)
        )
        rounded = np.round(p_value, precision)
        val = f"{rounded:.{precision}f}"
        if p_value < p_value_threshold_two:
            p_value_str = f"{val} (**)"
        elif p_value < p_value_threshold_one:
            p_value_str = f"{val} (*)"
        else:
            p_value_str = f"{val} (ns)"

        return JointTestResult(
            observed_improvement=obs_improvement,
            permuted_improvements=perm_improvements.tolist(),
            p_value=p_value,
            p_value_str=p_value_str,
            metric_type=metric_type,
            model_type=resolved.name,
            family=resolved.name,
            backend=backend_name,
            features_tested=features_tested,
            confounders=confounders or [],
            p_value_threshold_one=p_value_threshold_one,
            p_value_threshold_two=p_value_threshold_two,
            method=method,
            diagnostics=diagnostics,
        )

    elif method == "freedman_lane":
        permuted_coefs = _freedman_lane_individual_generic(
            X,
            y_values,
            resolved,
            confounders,
            perm_indices,
            model_coefs,
            fit_intercept,
            n_jobs=n_jobs,
        )

    elif method == "freedman_lane_joint":
        obs_improvement, perm_improvements, metric_type, features_tested = (
            _freedman_lane_joint(
                X,
                y_values,
                confounders,
                perm_indices,
                resolved,
                fit_intercept,
                n_jobs=n_jobs,
            )
        )

        p_value = float(
            (np.sum(perm_improvements >= obs_improvement) + 1) / (n_permutations + 1)
        )
        rounded = np.round(p_value, precision)
        val = f"{rounded:.{precision}f}"
        if p_value < p_value_threshold_two:
            p_value_str = f"{val} (**)"
        elif p_value < p_value_threshold_one:
            p_value_str = f"{val} (*)"
        else:
            p_value_str = f"{val} (ns)"

        return JointTestResult(
            observed_improvement=obs_improvement,
            permuted_improvements=perm_improvements.tolist(),
            p_value=p_value,
            p_value_str=p_value_str,
            metric_type=metric_type,
            model_type=resolved.name,
            family=resolved.name,
            backend=backend_name,
            features_tested=features_tested,
            confounders=confounders or [],
            p_value_threshold_one=p_value_threshold_one,
            p_value_threshold_two=p_value_threshold_two,
            method=method,
            diagnostics=diagnostics,
        )

    else:
        raise ValueError(
            f"Invalid method '{method}'. Choose 'ter_braak', 'kennedy', "
            "'kennedy_joint', 'freedman_lane', or 'freedman_lane_joint'."
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

    # For Kennedy and Freedman–Lane methods with confounders, mark
    # confounder p-values as N/A since they are controls (not hypotheses
    # being tested).  Their coefficients are held constant across
    # permutations, so computing a p-value for them would be meaningless.
    if method in ("kennedy", "freedman_lane") and confounders:
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

    return IndividualTestResult(
        model_coefs=model_coefs.tolist(),
        permuted_coefs=permuted_coefs.tolist(),
        permuted_p_values=permuted_p_values,
        classic_p_values=classic_p_values,
        raw_empirical_p=raw_empirical_p,
        raw_classic_p=raw_classic_p,
        p_value_threshold_one=p_value_threshold_one,
        p_value_threshold_two=p_value_threshold_two,
        method=method,
        confounders=confounders or [],
        model_type=resolved.name,
        family=resolved.name,
        backend=backend_name,
        diagnostics=diagnostics,
        extended_diagnostics=extended_diagnostics,
    )

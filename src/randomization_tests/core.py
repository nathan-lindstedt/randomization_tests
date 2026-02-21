"""Core permutation testing engine.

Implements three methods:

1. **ter Braak (1992)** – permute residuals under the reduced model.
2. **Kennedy (1995) individual** – partial out confounders, permute
   exposure-model residuals.
3. **Kennedy (1995) joint** – test whether a group of predictors
   collectively improves fit beyond confounders.

Vectorisation strategy:
    * **OLS (linear)** paths use batch matrix algebra: the pseudoinverse
      is computed once and multiplied against all permuted Y (or X)
      vectors simultaneously.
    * **Logistic** paths remain iterative (no closed-form solution) but
      benefit from pre-allocated result arrays and, when JAX is
      available, ``jax.vmap`` over a custom Newton solver with
      ``jax.grad`` for the gradient.

JAX backend:
    If ``jax`` is importable the logistic permutation loops are replaced
    with a ``vmap``-ped Newton/L-BFGS solver that processes all
    permutations in a single batched call.  The numpy/sklearn path is
    used as a transparent fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error

from .permutations import generate_unique_permutations
from .pvalues import calculate_p_values

# ------------------------------------------------------------------ #
# Optional JAX import
# ------------------------------------------------------------------ #
try:
    import jax
    import jax.numpy as jnp
    from jax import vmap, grad, jit
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


# ------------------------------------------------------------------ #
# JAX helpers (logistic regression via autodiff)
# ------------------------------------------------------------------ #

if _HAS_JAX:
    @jit
    def _logistic_nll(beta: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute the negative log-likelihood for logistic regression."""
        logits = X @ beta
        # Numerically stable: log(1 + exp(-y_signed * logits))
        # where y_signed = 2*y - 1
        return jnp.sum(jnp.logaddexp(0.0, -logits * (2.0 * y - 1.0)))

    _logistic_grad = jit(grad(_logistic_nll))

    @jit
    def _logistic_hessian_diag(beta, X, y):
        """Compute the full Hessian for Newton-Raphson logistic regression."""
        p = jax.nn.sigmoid(X @ beta)
        W = p * (1.0 - p)
        return (X.T * W[None, :]) @ X

    def _fit_logistic_jax(
        X_np: np.ndarray,
        y_np: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Fit logistic regression via Newton-Raphson with JAX autodiff."""
        X_j = jnp.array(X_np, dtype=jnp.float32)
        y_j = jnp.array(y_np, dtype=jnp.float32)
        beta = jnp.zeros(X_j.shape[1], dtype=jnp.float32)
        for _ in range(max_iter):
            g = _logistic_grad(beta, X_j, y_j)
            H = _logistic_hessian_diag(beta, X_j, y_j)
            step = jnp.linalg.solve(H, g)
            beta = beta - step
            if jnp.max(jnp.abs(step)) < tol:
                break
        return np.asarray(beta)

    def _fit_logistic_batch_jax(
        X_base: np.ndarray,
        Y_matrix: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Fit logistic regression for many Y vectors at once using vmap.

        Args:
            X_base: Design matrix of shape ``(n, p)`` shared across
                permutations.
            Y_matrix: Matrix of shape ``(B, n)`` where each row is a
                permuted response vector.
            max_iter: Maximum Newton iterations per solve.
            tol: Convergence tolerance (unused in vmap path).

        Returns:
            Coefficient matrix of shape ``(B, p)``.
        """
        X_j = jnp.array(X_base, dtype=jnp.float32)
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
        return np.asarray(batched_solve(Y_j))


# ------------------------------------------------------------------ #
# Batch OLS via pseudoinverse (numpy)
# ------------------------------------------------------------------ #

def _batch_ols_coefs(X: np.ndarray, Y_matrix: np.ndarray) -> np.ndarray:
    """Compute OLS coefficients for many Y vectors via one matrix multiply.

    Args:
        X: Design matrix of shape ``(n, p)``.
        Y_matrix: Matrix of shape ``(B, n)`` where each row is a
            permuted response vector.

    Returns:
        Coefficient matrix of shape ``(B, p)`` where
        ``coefs[b] = pinv(X) @ Y_matrix[b]``.
    """
    pinv = np.linalg.pinv(X)  # (p, n)
    return (pinv @ Y_matrix.T).T  # (B, p)


# ------------------------------------------------------------------ #
# Diagnostics (statsmodels)
# ------------------------------------------------------------------ #

def _compute_diagnostics(
    X: pd.DataFrame,
    y_values: np.ndarray,
    is_binary: bool,
) -> dict:
    """Compute model diagnostics via statsmodels."""
    n_obs = len(y_values)
    n_features = X.shape[1]

    if is_binary:
        sm_model = sm.Logit(y_values, sm.add_constant(X)).fit(disp=0)
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
        sm_model = sm.OLS(y_values, sm.add_constant(X)).fit()
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
# ter Braak (1992)
# ------------------------------------------------------------------ #

def _ter_braak_linear(
    X: pd.DataFrame,
    y_values: np.ndarray,
    perm_indices: np.ndarray,
) -> np.ndarray:
    """Vectorised ter Braak for OLS.

    For each feature *j*:

    1. Fit reduced OLS model (drop column *j*) → predicted values +
       residuals.
    2. Build all *B* permuted Y vectors in one shot using fancy indexing.
    3. Batch-compute full-model coefficients via pseudoinverse.

    Returns:
        Array of shape ``(B, n_features)`` with permuted coefficients.
    """
    X_np = X.values.astype(float)
    n_perm, n = perm_indices.shape
    n_features = X_np.shape[1]
    result = np.zeros((n_perm, n_features))

    X_full_pinv = np.linalg.pinv(X_np)  # (p, n) for the full model

    for j in range(n_features):
        # Reduced model: drop column j
        X_red = np.delete(X_np, j, axis=1)
        pinv_red = np.linalg.pinv(X_red)
        preds_red = X_red @ (pinv_red @ y_values)
        resids_red = y_values - preds_red

        # Permute residuals: (B, n) via fancy indexing
        permuted_resids = resids_red[perm_indices]
        Y_perm = preds_red[np.newaxis, :] + permuted_resids  # (B, n)

        # Batch OLS for full model
        all_coefs = _batch_ols_coefs(X_np, Y_perm)  # (B, p)
        result[:, j] = all_coefs[:, j]

    return result


def _ter_braak_logistic(
    X: pd.DataFrame,
    y_values: np.ndarray,
    perm_indices: np.ndarray,
) -> np.ndarray:
    """ter Braak for logistic regression.

    Uses a GLM-faithful adaptation: the reduced model is logistic,
    residuals on the probability scale are permuted, and Bernoulli
    sampling converts permuted Y* back to binary.

    When JAX is available the full-model refits are batched via
    ``vmap``.  Otherwise falls back to an sklearn loop.
    """
    X_np = X.values.astype(float)
    n_perm, n = perm_indices.shape
    n_features = X_np.shape[1]
    result = np.zeros((n_perm, n_features))

    # We need a separate RNG for Bernoulli sampling that is
    # deterministic given the perm_indices (derive seed from first index row).
    rng = np.random.default_rng(int(perm_indices[0, 0]))

    for j in range(n_features):
        # Reduced logistic model
        X_red = np.delete(X_np, j, axis=1)
        model_red = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5_000)
        model_red.fit(X_red, y_values)
        preds_red = model_red.predict_proba(X_red)[:, 1]
        resids_red = y_values - preds_red

        # Permute residuals → continuous Y* → clip → Bernoulli → binary
        permuted_resids = resids_red[perm_indices]  # (B, n)
        Y_perm_probs = np.clip(preds_red[np.newaxis, :] + permuted_resids, 0.001, 0.999)
        Y_perm_binary = rng.binomial(1, Y_perm_probs)  # (B, n)

        if _HAS_JAX:
            all_coefs = _fit_logistic_batch_jax(X_np, Y_perm_binary)
            result[:, j] = all_coefs[:, j]
        else:
            model_cls = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5_000)
            for p in range(n_perm):
                model_cls.fit(X_np, Y_perm_binary[p])
                result[p, j] = model_cls.coef_.flatten()[j]

    return result


# ------------------------------------------------------------------ #
# Kennedy (1995) individual
# ------------------------------------------------------------------ #

def _kennedy_individual_linear(
    X: pd.DataFrame,
    y_values: np.ndarray,
    confounders: list[str],
    perm_indices: np.ndarray,
    model_coefs: np.ndarray,
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

    # Fill confounder columns with observed coef (not tested)
    for i, col in enumerate(X.columns):
        if col in confounders:
            result[:, i] = model_coefs[i]

    for feature in features_to_test:
        feat_idx = X.columns.get_loc(feature)
        x_target = X[[feature]].values  # (n, 1)

        if Z.shape[1] > 0:
            pinv_z = np.linalg.pinv(Z)
            x_hat = Z @ (pinv_z @ x_target)
        else:
            x_hat = np.full_like(x_target, x_target.mean())
        x_resids = (x_target - x_hat).ravel()  # (n,)

        # Permute exposure residuals: (B, n)
        shuffled = x_resids[perm_indices]

        # For each permutation, reconstruct X* and batch-solve
        # We need to replace one column per permutation — batch via loop
        # on the matrix multiply since the design matrix changes.
        # However we can still vectorise the solve:
        X_perm_all = np.broadcast_to(X_np, (n_perm, n, n_features)).copy()
        X_perm_all[:, :, feat_idx] = x_hat.ravel()[np.newaxis, :] + shuffled  # (B, n)

        # Batch OLS: for each b, coefs[b] = pinv(X_perm_all[b]) @ y
        # Use lstsq per-row (pinv changes each time)
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
) -> np.ndarray:
    """Kennedy individual for logistic regression."""
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
        if Z.shape[1] > 0:
            exp_model = LinearRegression().fit(Z, x_target)
            x_hat = exp_model.predict(Z)
        else:
            x_hat = np.full_like(x_target, x_target.mean())
        x_resids = (x_target - x_hat).ravel()

        shuffled = x_resids[perm_indices]  # (B, n)

        if _HAS_JAX:
            # Build all B design matrices, then vmap
            X_perm_all = np.broadcast_to(X_np, (n_perm, n, n_features)).copy()
            X_perm_all[:, :, feat_idx] = x_hat.ravel()[np.newaxis, :] + shuffled

            X_j = jnp.array(X_perm_all, dtype=jnp.float32)
            y_j = jnp.array(y_values, dtype=jnp.float32)

            def _solve_one(X_single):
                beta = jnp.zeros(X_single.shape[1], dtype=jnp.float32)
                for _ in range(100):
                    g = _logistic_grad(beta, X_single, y_j)
                    H = _logistic_hessian_diag(beta, X_single, y_j)
                    beta = beta - jnp.linalg.solve(H, g)
                return beta

            batched = jit(vmap(_solve_one))
            all_coefs = np.asarray(batched(X_j))
            result[:, feat_idx] = all_coefs[:, feat_idx]
        else:
            model_cls = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5_000)
            for p in range(n_perm):
                X_perm = X.copy()
                X_perm.iloc[:, feat_idx] = x_hat.ravel() + shuffled[p]
                model_cls.fit(X_perm.values, y_values)
                result[p, feat_idx] = model_cls.coef_.flatten()[feat_idx]

    return result


# ------------------------------------------------------------------ #
# Kennedy (1995) joint
# ------------------------------------------------------------------ #

def _kennedy_joint(
    X: pd.DataFrame,
    y_values: np.ndarray,
    confounders: list[str],
    perm_indices: np.ndarray,
    is_binary: bool,
) -> tuple:
    """Kennedy joint test.

    Returns:
        A ``(obs_improvement, perm_improvements, metric_type,
        features_tested)`` tuple.
    """
    features_to_test = [c for c in X.columns if c not in confounders]

    if is_binary:
        model_cls = lambda: LogisticRegression(penalty=None, solver="lbfgs", max_iter=5_000)

        def get_metric(y_true, y_pred_proba):
            return 2 * log_loss(y_true, y_pred_proba, normalize=False)

        metric_type = "Deviance Reduction"
    else:
        model_cls = lambda: LinearRegression()

        def get_metric(y_true, y_pred):
            return mean_squared_error(y_true, y_pred) * len(y_true)

        metric_type = "RSS Reduction"

    n_perm, n = perm_indices.shape
    X_target = X[features_to_test].values

    if confounders:
        Z = X[confounders].values
    else:
        Z = np.zeros((n, 0))

    # Reduced model (confounders only)
    if Z.shape[1] > 0:
        reduced = model_cls().fit(Z, y_values)
        preds_reduced = reduced.predict_proba(Z) if is_binary else reduced.predict(Z)
    else:
        if is_binary:
            mean_y = np.mean(y_values)
            preds_reduced = np.column_stack([1 - mean_y * np.ones(n), mean_y * np.ones(n)])
        else:
            preds_reduced = np.full(n, np.mean(y_values), dtype=float)

    base_metric = get_metric(y_values, preds_reduced)

    # Full model
    full_features = np.hstack([X_target, Z]) if Z.shape[1] > 0 else X_target
    full_model = model_cls().fit(full_features, y_values)
    preds_full = full_model.predict_proba(full_features) if is_binary else full_model.predict(full_features)
    obs_improvement = base_metric - get_metric(y_values, preds_full)

    # Exposure model residuals
    if Z.shape[1] > 0:
        exp_model = LinearRegression().fit(Z, X_target)
        x_hat = exp_model.predict(Z)
    else:
        x_hat = np.full_like(X_target, X_target.mean(axis=0))
    x_resids = X_target - x_hat

    # Permutation loop — row-wise shuffle preserves inter-predictor correlation
    perm_improvements = np.zeros(n_perm)

    for i in range(n_perm):
        x_star = x_hat + x_resids[perm_indices[i]]
        perm_features = np.hstack([x_star, Z]) if Z.shape[1] > 0 else x_star
        perm_model = model_cls().fit(perm_features, y_values)
        perm_preds = perm_model.predict_proba(perm_features) if is_binary else perm_model.predict(perm_features)
        perm_improvements[i] = base_metric - get_metric(y_values, perm_preds)

    return obs_improvement, perm_improvements, metric_type, features_to_test


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def permutation_test_regression(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_permutations: int = 5_000,
    precision: int = 3,
    p_value_threshold_one: float = 0.05,
    p_value_threshold_two: float = 0.01,
    method: str = "ter_braak",
    confounders: list[str] | None = None,
    random_state: int | None = None,
) -> dict:
    """Run a permutation test for regression coefficients.

    Automatically detects binary vs. continuous outcomes and uses
    logistic or linear regression accordingly.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Target values of shape ``(n_samples,)``.  Binary targets
            (values in ``{0, 1}``) trigger logistic regression;
            otherwise linear regression is used.
        n_permutations: Number of unique permutations.
        precision: Decimal places for reported p-values.
        p_value_threshold_one: First significance level.
        p_value_threshold_two: Second significance level.
        method: One of ``'ter_braak'``, ``'kennedy'``, or
            ``'kennedy_joint'``.
        confounders: Column names of confounders (required for Kennedy
            methods).
        random_state: Seed for reproducibility.

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
    if confounders is None:
        confounders = []

    y_values = np.ravel(y)
    unique_y = np.unique(y_values)
    is_binary = (len(unique_y) == 2) and np.all(np.isin(unique_y, [0, 1]))

    # Fit observed model
    if is_binary:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5_000)
    else:
        model = LinearRegression()

    model.fit(X, y_values)
    model_coefs = model.coef_.flatten() if is_binary else np.ravel(model.coef_)

    # Diagnostics
    diagnostics = _compute_diagnostics(X, y_values, is_binary)

    # Pre-generate unique permutation indices
    perm_indices = generate_unique_permutations(
        n_samples=len(y_values),
        n_permutations=n_permutations,
        random_state=random_state,
        exclude_identity=True,
    )

    # ---- Dispatch to method-specific engine ----

    if method == "ter_braak":
        if is_binary:
            permuted_coefs = _ter_braak_logistic(X, y_values, perm_indices)
        else:
            permuted_coefs = _ter_braak_linear(X, y_values, perm_indices)

    elif method == "kennedy":
        if is_binary:
            permuted_coefs = _kennedy_individual_logistic(
                X, y_values, confounders, perm_indices, model_coefs,
            )
        else:
            permuted_coefs = _kennedy_individual_linear(
                X, y_values, confounders, perm_indices, model_coefs,
            )

    elif method == "kennedy_joint":
        obs_improvement, perm_improvements, metric_type, features_tested = _kennedy_joint(
            X, y_values, confounders, perm_indices, is_binary,
        )

        # Phipson & Smyth p-value
        p_value = (np.sum(perm_improvements >= obs_improvement) + 1) / (n_permutations + 1)
        if p_value < p_value_threshold_two:
            p_value_str = f"{np.round(p_value, precision)} (**)"
        elif p_value < p_value_threshold_one:
            p_value_str = f"{np.round(p_value, precision)} (*)"
        else:
            p_value_str = f"{np.round(p_value, precision)} (ns)"

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

    # Compute p-values
    permuted_p_values, classic_p_values = calculate_p_values(
        X, y, permuted_coefs, model_coefs,
        precision, p_value_threshold_one, p_value_threshold_two,
    )

    # Mark confounder p-values as N/A
    if method == "kennedy" and confounders:
        for i, col in enumerate(X.columns):
            if col in confounders:
                permuted_p_values[i] = "N/A (confounder)"
                classic_p_values[i] = "N/A (confounder)"

    return {
        "model_coefs": model_coefs.tolist(),
        "permuted_p_values": permuted_p_values,
        "classic_p_values": classic_p_values,
        "p_value_threshold_one": p_value_threshold_one,
        "p_value_threshold_two": p_value_threshold_two,
        "method": method,
        "model_type": "logistic" if is_binary else "linear",
        "diagnostics": diagnostics,
    }

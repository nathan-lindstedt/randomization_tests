"""P-value calculation for permutation tests.

Provides both empirical (permutation) and classical (asymptotic)
p-values, with vectorised computation over coefficients.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def calculate_p_values(
    X: pd.DataFrame,
    y: pd.DataFrame,
    permuted_coefs: np.ndarray,
    model_coefs: np.ndarray,
    precision: int = 3,
    p_value_threshold_one: float = 0.05,
    p_value_threshold_two: float = 0.01,
) -> tuple[list[str], list[str]]:
    """Calculate empirical (permutation) and classical (asymptotic) p-values.

    Automatically detects binary vs. continuous outcomes and uses logistic
    or linear regression for the classical asymptotic p-values.

    The empirical p-values use the Phipson & Smyth (2010) correction:
    ``p = (b + 1) / (B + 1)`` where *b* is the count of permuted
    ``|beta*_j| >= |beta_j|``.  This guarantees p-values are never exactly
    zero and treats the observed statistic as one member of a
    ``(B + 1)``-sized reference set.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Target values of shape ``(n_samples,)``.
        permuted_coefs: Coefficients from each permutation, shape
            ``(n_permutations, n_features)``.
        model_coefs: Observed (unpermuted) coefficients, shape
            ``(n_features,)``.
        precision: Decimal places for rounding.
        p_value_threshold_one: First significance threshold.
        p_value_threshold_two: Second significance threshold.

    Returns:
        A ``(permuted_p_values, classic_p_values)`` tuple where each
        element is a list of formatted p-value strings with significance
        markers (``*``, ``**``, or ``ns``).
    """
    permuted_coefs = np.asarray(permuted_coefs)
    model_coefs = np.asarray(model_coefs)

    y_values = np.ravel(y)
    unique_y = np.unique(y_values)
    is_binary = (len(unique_y) == 2) and np.all(np.isin(unique_y, [0, 1]))

    # --- Classical asymptotic p-values via statsmodels ---
    if is_binary:
        sm_model = sm.Logit(y_values, sm.add_constant(X)).fit(disp=0)
    else:
        sm_model = sm.OLS(y_values, sm.add_constant(X)).fit()

    # --- Vectorised empirical p-values ---
    # Shape: (n_permutations, n_features) compared against (n_features,)
    n_permutations = permuted_coefs.shape[0]
    counts = np.sum(np.abs(permuted_coefs) >= np.abs(model_coefs)[np.newaxis, :], axis=0)
    raw_p = (counts + 1) / (n_permutations + 1)

    # --- Format strings ---
    def _fmt(p: float) -> str:
        rounded = np.round(p, precision)
        if p < p_value_threshold_two:
            return f"{rounded} (**)"
        elif p < p_value_threshold_one:
            return f"{rounded} (*)"
        return f"{rounded} (ns)"

    permuted_p_values = [_fmt(p) for p in raw_p]
    # sm_model.pvalues[0] is the intercept — skip it
    classic_p_values = [_fmt(p) for p in sm_model.pvalues[1:]]

    return permuted_p_values, classic_p_values

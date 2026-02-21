"""P-value calculation for permutation tests.

Provides both empirical (permutation) and classical (asymptotic)
p-values, with vectorised computation over coefficients.

Empirical p-values — Phipson & Smyth (2010) correction
-------------------------------------------------------
A naïve empirical p-value counts the proportion of permuted test
statistics at least as extreme as the observed one:

    p_naïve = #{|β*_j| >= |β_j|} / B

where B is the number of permutations and β*_j is the j-th coefficient
from permutation b.  This estimator has a fundamental flaw: when the
observed statistic is the most extreme in the reference set, the p-value
is exactly zero — an impossible result, since the observed arrangement
is itself one of the (B+1) equally likely orderings.

Phipson & Smyth (2010) correct this by treating the observed statistic
as a member of the reference set:

    p = (b + 1) / (B + 1)

where b = #{|β*_j| >= |β_j|} and B is the total number of permutations.
This ensures:
  • p is never exactly zero (minimum is 1/(B+1));
  • The test has correct size — Pr(p <= α) <= α under H0.

Classical (asymptotic) p-values
-------------------------------
For comparison, classical p-values are computed via statsmodels using
Wald-type tests.  For OLS these are t-distribution-based; for logistic
regression they are z-distribution-based (from the MLE information
matrix).  These p-values depend on distributional assumptions that the
permutation test avoids.

Reference:
    Phipson, B. & Smyth, G. K. (2010). Permutation p-values should
    never be zero: calculating exact p-values when permutations are
    randomly drawn. *Statistical Applications in Genetics and Molecular
    Biology*, 9(1), Article 39.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ._compat import DataFrameLike, _ensure_pandas_df


def calculate_p_values(
    X: "DataFrameLike",
    y: "DataFrameLike",
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
    X = _ensure_pandas_df(X, name="X")
    y = _ensure_pandas_df(y, name="y")

    permuted_coefs = np.asarray(permuted_coefs)
    model_coefs = np.asarray(model_coefs)

    y_values = np.ravel(y)
    unique_y = np.unique(y_values)
    is_binary = (len(unique_y) == 2) and np.all(np.isin(unique_y, [0, 1]))

    # --- Classical asymptotic p-values via statsmodels ---
    # statsmodels expects an explicit intercept column, added via
    # sm.add_constant().  It returns p-values for [intercept, x1, …, xp],
    # so we skip index 0 when extracting feature-level p-values below.
    if is_binary:
        sm_model = sm.Logit(y_values, sm.add_constant(X)).fit(disp=0)
    else:
        sm_model = sm.OLS(y_values, sm.add_constant(X)).fit()

    # --- Vectorised empirical p-values (Phipson & Smyth correction) ---
    # For each feature j, count how many of the B permuted |β*_j| values
    # are at least as large in absolute value as the observed |β_j|.
    # This is a two-sided test: we compare magnitudes, not signed values.
    #
    # The comparison is broadcast across all B permutations at once:
    #   np.abs(permuted_coefs)            shape: (B, p)
    #   np.abs(model_coefs)[np.newaxis,:] shape: (1, p)
    # yielding a boolean matrix of shape (B, p) whose column sums give
    # the count b_j for each feature.
    n_permutations = permuted_coefs.shape[0]
    counts = np.sum(np.abs(permuted_coefs) >= np.abs(model_coefs)[np.newaxis, :], axis=0)

    # Apply the Phipson & Smyth correction:
    #   p_j = (b_j + 1) / (B + 1)
    # The "+1" in both numerator and denominator accounts for the
    # observed statistic itself being one of B+1 equally-likely outcomes.
    raw_p = (counts + 1) / (n_permutations + 1)

    # --- Format p-value strings with significance markers ---
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

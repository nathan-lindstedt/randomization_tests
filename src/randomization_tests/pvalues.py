r"""P-value calculation for permutation, sign-flip, and asymptotic hypothesis tests.

Provides vectorised empirical (permutation) and classical (asymptotic) p-value
computation for both individual per-feature tests and omnibus joint tests.

Primary Components
------------------
* :func:`calculate_p_values`: Vectorised calculation of empirical and asymptotic
  p-values across individual model coefficients.
* :func:`calculate_joint_p_value`: Omnibus collective p-value evaluation for joint
  hypothesis tests based on fit-improvement metrics.

Empirical P-Values & Phipson–Smyth Correction
---------------------------------------------
A naïve empirical p-value counts the proportion of permuted test statistics
at least as extreme as the observed one:

.. math::
    p_{\mathrm{na\ddot{\imath}ve}} = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(|\beta^*_{b, j}| \ge |\hat\beta_j|)

This estimator has a fundamental flaw: when the observed statistic is the most
extreme value in the reference set, the naïve p-value is zero—an impossible result
in finite samples, as the observed data arrangement is itself an exchangeable realization
under the null hypothesis.

Phipson & Smyth (2010) correct this by incorporating the observed test statistic
directly into the reference set:

.. math::
    p = \frac{b + 1}{B + 1}

where $b = \sum_{b=1}^B \mathbb{I}(|\beta^*_{b, j}| \ge |\hat\beta_j|)$ and $B$ is
the total number of permutations. This formulation guarantees:
- $p$ is strictly positive with a lower bound of $1 / (B + 1)$.
- Exact size control: $\mathbb{P}(p \le \alpha) \le \alpha$ under $H_0$ for any nominal $\alpha$.

Classical Asymptotic Inference Paths
------------------------------------
Classical p-values are computed via ``family.classical_p_values()``, which prioritises
fast JAX observed Fisher information inversion ($H^{-1}$) and empirical Huber–White
sandwich estimators ($V = H^{-1} S^\top S H^{-1}$), with automatic fallback to
statsmodels (including cluster-robust covariance matrices when `groups` is provided).

Cross-Module Integration
------------------------
* Inbound: Called by :func:`~.core.randomization_test_regression` during result packaging.
* Outbound: Calls :meth:`~.families.ModelFamily.classical_p_values` for asymptotic baselines.

References
----------
Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero:
    calculating exact p-values when permutations are randomly drawn.
    *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._compat import DataFrameLike, _ensure_pandas_df
from .exchangeability import ExchangeabilityTree

if TYPE_CHECKING:
    from .families import ModelFamily


def calculate_p_values(
    X: DataFrameLike,
    y: DataFrameLike,
    permuted_coefs: np.ndarray,
    model_coefs: np.ndarray,
    precision: int = 3,
    p_value_threshold_one: float = 0.05,
    p_value_threshold_two: float = 0.01,
    p_value_threshold_three: float = 0.001,
    fit_intercept: bool = True,
    *,
    family: ModelFamily,
    groups: np.ndarray | ExchangeabilityTree | None = None,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Calculate empirical (permutation) and classical (asymptotic) p-values.

    Classical p-values are delegated to ``family.classical_p_values()``.

    The empirical p-values use the Phipson & Smyth (2010) correction:
    ``p = (b + 1) / (B + 1)`` where *b* is the count of permuted
    ``|beta*_j| >= |beta_j|``.  This guarantees p-values are never exactly
    zero and treats the observed statistic as one member of a
    ``(B + 1)``-sized reference set.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Target values of shape ``(n_samples,)``.
        permuted_coefs: Coefficients from each permutation, shape
            ``(n_randomizations, n_features)``.
        model_coefs: Observed (unpermuted) coefficients, shape
            ``(n_features,)``.
        precision: Decimal places for rounding.
        p_value_threshold_one: First significance threshold.
        p_value_threshold_two: Second significance threshold.
        p_value_threshold_three: Third significance threshold.
        fit_intercept: Whether an intercept was included in the model.
            When True (default), ``sm.add_constant`` prepends an
            intercept column for the statsmodels fit and the first
            element of the resulting p-value array is skipped.  When
            False, the raw design matrix is used directly and all
            returned p-values correspond to the features.
        family: ``ModelFamily`` instance used to compute classical
            p-values via ``family.classical_p_values()``.

    Returns:
        A four-element tuple
        ``(permuted_p_values, classic_p_values, raw_empirical_p, raw_classic_p, counts)``
        where the first two elements are lists of formatted p-value
        strings with significance markers (``*``, ``**``, or ``ns``),
        the next two are NumPy arrays of raw (unformatted) numeric
        p-values used for downstream diagnostics, and ``counts`` is
        the per-feature count of permuted |β*| ≥ observed |β| used
        for Clopper-Pearson p-value CIs.
    """
    X = _ensure_pandas_df(X, name="X")
    y = _ensure_pandas_df(y, name="y")

    permuted_coefs = np.asarray(permuted_coefs)
    model_coefs = np.asarray(model_coefs)

    y_values = np.ravel(y)

    # --- Classical asymptotic p-values ---
    # Delegate to the family's classical_p_values() method, which
    # encapsulates the correct statsmodels model and warning suppression.
    # When groups are provided, pass them through so the family can
    # compute cluster-robust SEs matching the permutation test's
    # within-group structure. If groups is an ExchangeabilityTree,
    # collapse to leaf-level cluster labels for statsmodels covariance estimation.
    cluster_groups = (
        groups.to_flat_cells() if isinstance(groups, ExchangeabilityTree) else groups
    )
    raw_classic_p = family.classical_p_values(
        np.asarray(X), y_values, fit_intercept, groups=cluster_groups
    )

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
    n_randomizations = permuted_coefs.shape[0]
    counts = np.sum(
        np.abs(permuted_coefs) >= np.abs(model_coefs)[np.newaxis, :], axis=0
    )

    # Apply the Phipson & Smyth correction:
    #   p_j = (b_j + 1) / (B + 1)
    # The "+1" in both numerator and denominator accounts for the
    # observed statistic itself being one of B+1 equally-likely outcomes.
    raw_p = (counts + 1) / (n_randomizations + 1)

    # --- Format p-value strings with significance markers ---
    def _fmt(p: float) -> str:
        rounded = np.round(p, precision)
        val = f"{rounded:.{precision}f}"
        if p < p_value_threshold_three:
            return f"{val} (***)"
        elif p < p_value_threshold_two:
            return f"{val}  (**)"
        elif p < p_value_threshold_one:
            return f"{val}   (*)"
        return f"{val}  (ns)"

    permuted_p_values = [_fmt(p) for p in raw_p]
    classic_p_values = [_fmt(p) for p in raw_classic_p]

    return permuted_p_values, classic_p_values, raw_p, raw_classic_p, counts

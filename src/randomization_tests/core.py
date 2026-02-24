"""Core permutation testing orchestrator.

This module is the single entry-point for running a permutation test.
It delegates to three subsystems:

1. **Engine** (:mod:`randomization_tests.engine`) — resolves the
   family, backend, and shared state (observed model, permutation
   indices).
2. **Strategy** (:mod:`randomization_tests._strategies`) — executes
   the chosen permutation algorithm (ter Braak, Kennedy,
   Freedman–Lane).
3. **Result packaging** — computes p-values, diagnostics, and
   constructs the typed result objects.

The five permutation strategies are documented in their respective
modules under ``_strategies/``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._compat import DataFrameLike, _ensure_pandas_df
from ._results import IndividualTestResult, JointTestResult
from ._strategies import resolve_strategy
from .diagnostics import compute_all_diagnostics
from .engine import PermutationEngine
from .families import ModelFamily
from .pvalues import calculate_p_values

# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #


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
    family: str | ModelFamily = "auto",
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
        family: Model family string **or** a ``ModelFamily`` instance.
            ``"auto"`` (default) detects binary {0, 1} targets →
            logistic, otherwise linear.  Explicit strings
            (``"linear"``, ``"logistic"``, ``"poisson"``, etc.)
            bypass auto-detection and are validated against the
            response via the family's ``validate_y()`` method.
            Pre-configured instances (e.g.
            ``NegativeBinomialFamily(alpha=2.0)``) are accepted
            directly — ``calibrate()`` is still called but is a
            no-op when the instance is already configured.
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

    # ---- Input validation ----------------------------------------
    _validate_inputs(X, y, confounders, n_permutations)

    y_values = np.ravel(y)

    # ---- Contextual warnings -------------------------------------
    if method in ("kennedy", "kennedy_joint") and not confounders:
        warnings.warn(
            f"{method!r} method called without confounders — all features "
            "will be tested. Consider 'ter_braak' for unconditional tests.",
            UserWarning,
            stacklevel=2,
        )

    if method in ("freedman_lane", "freedman_lane_joint") and not confounders:
        warnings.warn(
            f"{method!r} method called without confounders — the reduced "
            "model is intercept-only, which yields less power than "
            "conditioning on other predictors. Consider 'ter_braak' for "
            "unconditional tests.",
            UserWarning,
            stacklevel=2,
        )

    # ---- Engine (family + backend + observed model + perm indices) -
    engine = PermutationEngine(
        X,
        y_values,
        family=family,
        fit_intercept=fit_intercept,
        n_permutations=n_permutations,
        random_state=random_state,
        n_jobs=n_jobs,
        method=method,
    )

    # ---- Strategy resolution -------------------------------------
    # Validate method string and special-case guards.
    if method == "ter_braak" and engine.family.name == "logistic" and X.shape[1] == 1:
        raise ValueError(
            "ter Braak method with logistic regression requires at least "
            "2 features because the reduced model (dropping the single "
            "feature) has 0 predictors.  Use method='kennedy' with "
            "confounders, or add additional features."
        )

    strategy = resolve_strategy(method)

    # ---- Execute strategy ----------------------------------------
    result = strategy.execute(
        X,
        y_values,
        engine.family,
        engine.perm_indices,
        confounders=confounders,
        model_coefs=engine.model_coefs,
        fit_intercept=fit_intercept,
        n_jobs=engine._n_jobs,
    )

    # ---- Package results -----------------------------------------
    if strategy.is_joint:
        return _package_joint_result(
            result,  # type: ignore[arg-type]
            engine=engine,
            method=method,
            confounders=confounders,
            precision=precision,
            p_value_threshold_one=p_value_threshold_one,
            p_value_threshold_two=p_value_threshold_two,
            n_permutations=n_permutations,
        )

    return _package_individual_result(
        result,  # type: ignore[arg-type]
        X=X,
        y=y,
        y_values=y_values,
        engine=engine,
        method=method,
        confounders=confounders,
        precision=precision,
        p_value_threshold_one=p_value_threshold_one,
        p_value_threshold_two=p_value_threshold_two,
        n_permutations=n_permutations,
        fit_intercept=fit_intercept,
    )


# ------------------------------------------------------------------ #
# Input validation (extracted for readability)
# ------------------------------------------------------------------ #


def _validate_inputs(
    X: pd.DataFrame,
    y: pd.DataFrame,
    confounders: list[str],
    n_permutations: int,
) -> None:
    """Raise ``ValueError`` for invalid inputs."""
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

    if confounders:
        missing = [c for c in confounders if c not in X.columns]
        if missing:
            raise ValueError(f"Confounders not found in X columns: {missing}")


# ------------------------------------------------------------------ #
# Result packaging helpers
# ------------------------------------------------------------------ #


def _package_joint_result(
    raw: tuple[float, np.ndarray, str, list[str]],
    *,
    engine: PermutationEngine,
    method: str,
    confounders: list[str],
    precision: int,
    p_value_threshold_one: float,
    p_value_threshold_two: float,
    n_permutations: int,
) -> JointTestResult:
    """Build a :class:`JointTestResult` from a joint strategy's output."""
    obs_improvement, perm_improvements, metric_type, features_tested = raw

    # Phipson & Smyth (2010) empirical p-value: (b + 1) / (B + 1)
    # where b = #{permuted Δ ≥ observed Δ}.  The +1 in numerator
    # and denominator guarantees p ∈ (0, 1] and accounts for the
    # observed statistic as one of the possible permutations.
    p_value = float(
        (np.sum(perm_improvements >= obs_improvement) + 1) / (n_permutations + 1)
    )
    rounded = np.round(p_value, precision)  # round to display precision
    val = f"{rounded:.{precision}f}"  # fixed-width string representation
    if p_value < p_value_threshold_two:
        p_value_str = f"{val} (**)"  # highly significant
    elif p_value < p_value_threshold_one:
        p_value_str = f"{val} (*)"  # significant
    else:
        p_value_str = f"{val} (ns)"  # not significant

    return JointTestResult(
        observed_improvement=obs_improvement,
        permuted_improvements=perm_improvements.tolist(),
        p_value=p_value,
        p_value_str=p_value_str,
        metric_type=metric_type,
        model_type=engine.family.name,
        family=engine.family.name,
        backend=engine.backend_name,
        features_tested=features_tested,
        confounders=confounders or [],
        p_value_threshold_one=p_value_threshold_one,
        p_value_threshold_two=p_value_threshold_two,
        method=method,
        diagnostics=engine.diagnostics,
    )


def _package_individual_result(
    permuted_coefs: np.ndarray,
    *,
    X: pd.DataFrame,
    y: pd.DataFrame,
    y_values: np.ndarray,
    engine: PermutationEngine,
    method: str,
    confounders: list[str],
    precision: int,
    p_value_threshold_one: float,
    p_value_threshold_two: float,
    n_permutations: int,
    fit_intercept: bool,
) -> IndividualTestResult:
    """Build an :class:`IndividualTestResult` from an individual strategy's output."""
    permuted_p_values, classic_p_values, raw_empirical_p, raw_classic_p = (
        calculate_p_values(
            X,
            y,
            permuted_coefs,
            engine.model_coefs,
            precision,
            p_value_threshold_one,
            p_value_threshold_two,
            fit_intercept=fit_intercept,
            family=engine.family,
        )
    )

    # Mask confounder p-values for Kennedy / Freedman–Lane individual.
    if method in ("kennedy", "freedman_lane") and confounders:
        for i, col in enumerate(X.columns):
            if col in confounders:
                permuted_p_values[i] = "N/A (confounder)"
                classic_p_values[i] = "N/A (confounder)"
                raw_empirical_p[i] = np.nan
                raw_classic_p[i] = np.nan

    extended_diagnostics = compute_all_diagnostics(
        X=X,
        y_values=y_values,
        model_coefs=engine.model_coefs,
        family=engine.family,
        raw_empirical_p=raw_empirical_p,
        raw_classic_p=raw_classic_p,
        n_permutations=n_permutations,
        p_value_threshold=p_value_threshold_one,
        method=method,
        confounders=confounders,
        fit_intercept=fit_intercept,
    )

    return IndividualTestResult(
        model_coefs=engine.model_coefs.tolist(),
        permuted_coefs=permuted_coefs.tolist(),
        permuted_p_values=permuted_p_values,
        classic_p_values=classic_p_values,
        raw_empirical_p=raw_empirical_p,
        raw_classic_p=raw_classic_p,
        p_value_threshold_one=p_value_threshold_one,
        p_value_threshold_two=p_value_threshold_two,
        method=method,
        confounders=confounders or [],
        model_type=engine.family.name,
        family=engine.family.name,
        backend=engine.backend_name,
        diagnostics=engine.diagnostics,
        extended_diagnostics=extended_diagnostics,
    )

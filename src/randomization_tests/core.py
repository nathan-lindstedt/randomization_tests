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
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ._compat import DataFrameLike, _ensure_pandas_df
from ._results import IndividualTestResult, JointTestResult
from ._strategies import resolve_strategy
from .diagnostics import compute_all_diagnostics
from .engine import PermutationEngine
from .families import ModelFamily
from .permutations import _between_cell_total
from .pvalues import calculate_p_values

# Valid permutation strategy strings.
_VALID_STRATEGIES = {"within", "between", "two-stage"}

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
    backend: str | None = None,
    groups: np.ndarray | list[int] | pd.Series | pd.DataFrame | None = None,
    permutation_strategy: str | None = None,
    permutation_constraints: Callable[[np.ndarray], np.ndarray] | None = None,
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
        backend: ``"numpy"``, ``"jax"``, or ``None`` (default).
            When ``None``, the global policy from
            :func:`~randomization_tests.set_backend` is used.
            An explicit value overrides the global setting for this
            call only, enabling test injection and per-call backend
            selection.
        groups: Exchangeability group labels.  When provided,
            permutations are constrained to respect group structure
            rather than shuffling globally.  Accepts a 1-D array-like
            of integer labels ``(n_samples,)`` or a ``DataFrame``
            with one or more blocking columns (multi-column DataFrames
            are cross-classified into integer cell labels).
        permutation_strategy: Which cell-level permutation strategy
            to use.  ``"within"`` shuffles only within cells,
            ``"between"`` permutes entire cells as units, and
            ``"two-stage"`` composes both.  When ``groups`` is
            provided without a strategy, defaults to ``"within"``.
            Cannot be set without ``groups``.
        permutation_constraints: Optional post-filter callback.
            Receives a ``(B, n)`` permutation array and must return
            a ``(B', n)`` array with ``B' ≤ B`` rows.  Used to
            apply domain-specific constraints that cannot be
            expressed via cell structure alone.

    Returns:
        Typed result object containing coefficients, p-values,
        diagnostics, and method metadata.  The ``family`` attribute
        holds the resolved :class:`ModelFamily` instance.

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

    # ---- Groups validation ---------------------------------------
    cells, resolved_strategy = _validate_groups(
        X, groups, permutation_strategy, n_permutations
    )

    # ---- Callback validation -------------------------------------
    if permutation_constraints is not None:
        _validate_constraints(permutation_constraints, len(y_values))

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
        backend=backend,
        groups=cells,
        permutation_strategy=resolved_strategy,
        permutation_constraints=permutation_constraints,
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
            X=X,
            y=y,
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
# Groups validation (Step 10a + 11b-d)
# ------------------------------------------------------------------ #


def _validate_groups(
    X: pd.DataFrame,
    groups: np.ndarray | list[int] | pd.Series | pd.DataFrame | None,
    permutation_strategy: str | None,
    n_permutations: int,
) -> tuple[np.ndarray | None, str | None]:
    """Validate and resolve ``groups`` / ``permutation_strategy``.

    Converts heterogeneous group inputs into an integer cell-label
    array and resolves the strategy string (defaulting to ``"within"``
    when groups are provided without an explicit strategy).

    Args:
        X: Feature matrix — used only for its row count.
        groups: Raw user-supplied group labels (array-like, DataFrame,
            or ``None``).
        permutation_strategy: ``"within"``, ``"between"``,
            ``"two-stage"``, or ``None``.
        n_permutations: Requested number of permutations (used for
            minimum-group-count checks).

    Returns:
        Tuple ``(cells, resolved_strategy)`` where *cells* is an
        integer array of shape ``(n,)`` or ``None``, and
        *resolved_strategy* is a string or ``None``.

    Raises:
        ValueError: If ``permutation_strategy`` is set without
            ``groups``, if the strategy string is invalid, if
            ``groups`` has wrong length, or if ``"between"`` is
            requested with fewer than 5 groups.
    """
    n = X.shape[0]

    # ---- Strategy without groups: always an error ----------------
    if permutation_strategy is not None and groups is None:
        raise ValueError(
            f"permutation_strategy={permutation_strategy!r} requires "
            "groups= to be specified."
        )

    # ---- No groups: nothing to validate --------------------------
    if groups is None:
        return None, None

    # ---- Strategy string validation ------------------------------
    if permutation_strategy is not None:
        if permutation_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"permutation_strategy must be one of "
                f"{sorted(_VALID_STRATEGIES)}, got "
                f"{permutation_strategy!r}."
            )
        resolved = permutation_strategy
    else:
        # Default when groups are provided without explicit strategy.
        resolved = "within"

    # ---- Convert groups to integer cell labels -------------------
    if isinstance(groups, pd.DataFrame):
        if groups.shape[1] == 1:
            # Single blocking column → extract as 1-D.
            cells_raw = groups.iloc[:, 0].values
        else:
            # Multi-column blocking → cross-classify into unique
            # integer cell labels (Cartesian product of levels).
            cells_raw = _cross_classify(groups)
    elif isinstance(groups, pd.Series):
        cells_raw = groups.values
    else:
        cells_raw = np.asarray(groups)

    cells = _to_integer_labels(cells_raw)

    # ---- Shape check ---------------------------------------------
    if len(cells) != n:
        raise ValueError(f"groups has {len(cells)} elements but X has {n} rows.")

    # ---- Strategy-specific guards --------------------------------
    unique_labels = np.unique(cells)
    G = len(unique_labels)

    if resolved == "between" and G < 5:
        raise ValueError(
            f"permutation_strategy='between' requires at least 5 groups "
            f"(G={G} gives G!={_factorial_safe(G)}, min p≈"
            f"{1 / _factorial_safe(G):.4f} — too coarse for α=0.05).  "
            f"Use 'within' or 'two-stage' for small G, or add more "
            f"groups."
        )

    # ---- Between-cell infeasibility (unbalanced cells) -----------
    #
    # Between-cell permutations can only swap cells of the same size.
    # If all cells have unique sizes, no valid permutations exist.
    # Guide the user to within or two-stage as alternatives.
    cell_sizes = np.bincount(cells)
    non_empty_sizes = cell_sizes[cell_sizes > 0].tolist()

    if resolved == "between":
        between_total = _between_cell_total(non_empty_sizes)
        between_available = between_total - 1  # excluding identity

        if between_available == 0:
            raise ValueError(
                f"permutation_strategy='between' is infeasible: all "
                f"{G} cells have different sizes "
                f"({sorted(set(non_empty_sizes))}), so no valid "
                f"between-cell permutations exist.  Between-cell "
                f"permutations can only swap cells of the same size.  "
                f"Use permutation_strategy='within' (shuffles within "
                f"each cell) or 'two-stage' (composes between- and "
                f"within-cell shuffles) instead."
            )

        if between_available < 100:
            warnings.warn(
                f"Only {between_available} unique between-cell "
                f"permutations are available (cells have sizes "
                f"{sorted(set(non_empty_sizes))}).  Between-cell "
                f"permutations can only swap same-size cells, limiting "
                f"the reference distribution.  Consider "
                f"permutation_strategy='two-stage' for a richer "
                f"reference set.",
                UserWarning,
                stacklevel=3,
            )

    # ---- Singleton warnings (Step 11b) ---------------------------
    n_singletons = int(np.sum(cell_sizes == 1))

    if n_singletons > 0 and resolved in ("within", "two-stage"):
        warnings.warn(
            f"{n_singletons} cell(s) contain a single observation and "
            f"contribute nothing to the '{resolved}' within-cell "
            f"reference distribution.  Their members are never shuffled.",
            UserWarning,
            stacklevel=3,
        )

    # ---- Two-stage imbalance (Step 11d) --------------------------
    if resolved == "two-stage":
        ratio = float(max(non_empty_sizes)) / float(min(non_empty_sizes))
        if ratio > 3.0:
            warnings.warn(
                f"Two-stage permutation assumes independence of between-"
                f"cell and within-cell exchangeability.  Cell sizes are "
                f"highly unbalanced (max/min = {ratio:.1f} > 3).  "
                f"Consider 'within' for safer inference.",
                UserWarning,
                stacklevel=3,
            )

    return cells, resolved


def _cross_classify(df: pd.DataFrame) -> np.ndarray:
    """Convert a multi-column DataFrame into integer cell labels.

    Each unique combination of factor levels across all columns gets
    a unique integer label.  The mapping is deterministic (labels are
    assigned in the order the combinations first appear).

    Args:
        df: DataFrame with one or more blocking columns.

    Returns:
        Integer array of shape ``(n,)`` with cell labels.
    """
    # Tuple-ify each row, then map to sequential integers.
    keys = list(map(tuple, df.values.tolist()))
    label_map: dict[tuple[Any, ...], int] = {}
    labels = np.empty(len(keys), dtype=np.intp)
    for i, key in enumerate(keys):
        if key not in label_map:
            label_map[key] = len(label_map)
        labels[i] = label_map[key]
    return labels


def _to_integer_labels(arr: np.ndarray) -> np.ndarray:
    """Map arbitrary labels to 0-indexed integers.

    Args:
        arr: Array of labels (ints, strings, floats, etc.).

    Returns:
        Integer array ``(n,)`` with values in ``[0, G-1]``.
    """
    _, inverse = np.unique(arr, return_inverse=True)
    return np.asarray(inverse, dtype=np.intp)


def _factorial_safe(n: int) -> int:
    """Compute n! capped at a large sentinel to avoid slowdowns."""
    import math

    if n > 20:
        return 10**18  # sentinel — never used in comparisons
    return math.factorial(n)


# ------------------------------------------------------------------ #
# Callback validation (Step 11a)
# ------------------------------------------------------------------ #


def _validate_constraints(
    callback: Callable[[np.ndarray], np.ndarray],
    n_samples: int,
) -> None:
    """Validate a user-supplied permutation constraint callback.

    Performs a shape probe: calls the callback on a small test batch
    to verify it returns an ``np.ndarray`` with the correct second
    dimension and no more rows than it received.

    Args:
        callback: The user's constraint function.
        n_samples: Number of observations (second dimension of the
            permutation array).

    Raises:
        TypeError: If *callback* is not callable, or if the probe
            returns an unexpected type or shape.
    """
    if not callable(callback):
        raise TypeError(
            f"permutation_constraints must be callable, got {type(callback).__name__}."
        )

    # Shape probe with a 2-row identity-ish batch.
    probe = np.tile(np.arange(n_samples, dtype=np.intp), (2, 1))
    try:
        result = callback(probe)
    except Exception as exc:
        raise TypeError(
            f"permutation_constraints raised an error on a probe call: {exc}"
        ) from exc

    if not isinstance(result, np.ndarray):
        raise TypeError(
            f"permutation_constraints must return np.ndarray, got "
            f"{type(result).__name__}."
        )

    if result.ndim != 2 or result.shape[1] != n_samples:
        raise TypeError(
            f"permutation_constraints returned shape {result.shape}, "
            f"expected (B', {n_samples}) with B' ≤ 2."
        )

    if result.shape[0] > probe.shape[0]:
        raise TypeError(
            f"permutation_constraints returned {result.shape[0]} rows — "
            f"more than the input ({probe.shape[0]}).  Callbacks must "
            f"filter (remove rows), not add."
        )


# ------------------------------------------------------------------ #
# Result packaging helpers
# ------------------------------------------------------------------ #


def _package_joint_result(
    raw: tuple[float, np.ndarray, str, list[str]],
    *,
    X: pd.DataFrame,
    y: pd.DataFrame,
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
        family=engine.family,
        backend=engine.backend_name,
        features_tested=features_tested,
        confounders=confounders or [],
        feature_names=list(X.columns),
        target_name=str(y.columns[0]),
        n_permutations=n_permutations,
        groups=engine.groups,
        permutation_strategy=engine.permutation_strategy,
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
        family=engine.family,
        backend=engine.backend_name,
        feature_names=list(X.columns),
        target_name=str(y.columns[0]),
        n_permutations=n_permutations,
        groups=engine.groups,
        permutation_strategy=engine.permutation_strategy,
        diagnostics=engine.diagnostics,
        extended_diagnostics=extended_diagnostics,
    )

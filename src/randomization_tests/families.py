"""Model family protocol and resolution logic.

The ``ModelFamily`` protocol defines the interface that every
regression family must implement.  It decouples model-specific
behaviour (fitting, residual computation, Y-reconstruction, batch
fitting, diagnostics, classical p-values) from the permutation engine
in ``core.py``, which dispatches to the active family via generic
method calls instead of branching on ``is_binary``.

Each concrete family is a ``@dataclass`` that carries no mutable state
and communicates exclusively through the protocol methods.  The
``resolve_family`` helper maps a user-facing string (``"auto"``,
``"linear"``, ``"logistic"``, etc.) to the appropriate family instance.

Architecture
~~~~~~~~~~~~
The protocol is deliberately minimal — every method maps 1:1 to an
existing branch in the v0.2.0 ``core.py`` code, so that the Phase 2
core refactor is a mechanical replacement of ``if is_binary:`` checks
with ``family.<method>()`` calls.  No new statistical logic is
introduced here; the protocol merely formalises the interface that was
previously implicit.

Extensibility
~~~~~~~~~~~~~
New families (Poisson, negative binomial, ordinal, multinomial) are
added by implementing the protocol in this module and registering them
in the ``_FAMILIES`` dict.  The core engine, display module, and
confounder module require zero changes per new family — they program
against the protocol, not against concrete classes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------------ #
# ModelFamily protocol
# ------------------------------------------------------------------ #
#
# ``runtime_checkable`` enables isinstance() checks against the
# protocol at runtime, which is useful for defensive validation in
# resolve_family() and in tests.  The performance cost is negligible
# (a handful of hasattr calls).


@runtime_checkable
class ModelFamily(Protocol):
    """Interface that every regression family must implement.

    Attributes:
        name: Short identifier used in result dicts and display
            headers (e.g. ``"linear"``, ``"logistic"``, ``"poisson"``).
        residual_type: Label for the residual flavour the family
            produces (e.g. ``"raw"``, ``"probability"``,
            ``"deviance"``).  Used in display and documentation, not
            in dispatch logic.
        direct_permutation: When ``True``, permutation is applied to
            Y directly instead of to residuals.  Families whose
            residuals are not well-defined (e.g. ordinal) set this to
            ``True``; the permutation engine then skips the
            ``residuals → permute → reconstruct_y`` pipeline and
            permutes Y rows instead.
    """

    @property
    def name(self) -> str: ...

    @property
    def residual_type(self) -> str: ...

    @property
    def direct_permutation(self) -> bool: ...

    # ---- Validation ------------------------------------------------

    def validate_y(self, y: np.ndarray) -> None:
        """Raise ``ValueError`` if *y* is unsuitable for this family.

        Called once before fitting, so that invalid data produces a
        clear message rather than an opaque downstream error.

        Args:
            y: Response vector of shape ``(n,)``.
        """
        ...

    # ---- Single-model operations -----------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> Any:
        """Fit the model and return the fitted model object.

        The returned object is opaque to the caller — it is only
        passed back into ``predict``, ``coefs``, and ``residuals``.

        Args:
            X: Design matrix of shape ``(n, p)``.
            y: Response vector of shape ``(n,)``.
            fit_intercept: Whether to include an intercept term.
        """
        ...

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Return predictions from a fitted model.

        For linear families this is ``ŷ``; for logistic it is
        ``P(Y=1|X)``; for count families it is ``μ̂``.

        Args:
            model: A fitted model object returned by ``fit``.
            X: Design matrix of shape ``(n, p)``.

        Returns:
            Prediction vector of shape ``(n,)``.
        """
        ...

    def coefs(self, model: Any) -> np.ndarray:
        """Extract the slope coefficient vector from a fitted model.

        Intercept terms are excluded — only the ``p`` slope
        coefficients corresponding to the columns of *X* are returned.

        Args:
            model: A fitted model object returned by ``fit``.

        Returns:
            Coefficient vector of shape ``(p,)``.
        """
        ...

    def residuals(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute residuals from a fitted model.

        The residual flavour is family-specific (raw for linear,
        probability-scale for logistic, deviance for count models).
        See ``residual_type`` for the label.

        Args:
            model: A fitted model object returned by ``fit``.
            X: Design matrix of shape ``(n, p)``.
            y: Observed response vector of shape ``(n,)``.

        Returns:
            Residual vector of shape ``(n,)``.
        """
        ...

    def reconstruct_y(
        self,
        predictions: np.ndarray,
        permuted_residuals: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Reconstruct permuted Y* from predictions and permuted residuals.

        For continuous families this is simply ``predictions +
        permuted_residuals``.  For binary families the sum is clipped
        to a valid probability range and passed through a Bernoulli
        sampler.  Count families exponentiate and round.

        Args:
            predictions: Predicted values from the reduced model,
                shape ``(n,)`` or ``(B, n)``.
            permuted_residuals: Permuted residual matrix, shape
                ``(B, n)``.
            rng: NumPy random generator for stochastic reconstruction
                (e.g. Bernoulli sampling).

        Returns:
            Permuted response matrix of shape ``(B, n)``.
        """
        ...

    def fit_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Compute the goodness-of-fit metric used in the joint test.

        The metric should be **lower-is-better** so that the joint
        test statistic (reduced − full) is positive when the features
        improve fit.

        Linear: RSS = MSE × n.
        Logistic: Deviance = 2 × binary cross-entropy.
        Count: Deviance.

        Args:
            y_true: Observed response vector of shape ``(n,)``.
            y_pred: Predicted values of shape ``(n,)``.

        Returns:
            Scalar fit metric.
        """
        ...

    def diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> dict[str, Any]:
        """Compute model diagnostics via statsmodels.

        Called once on the observed (unpermuted) data.  The returned
        dict is stored in the result under ``"diagnostics"`` and
        consumed by the display module.

        Args:
            X: Design matrix (pandas DataFrame or ndarray).
            y: Response vector of shape ``(n,)``.
            fit_intercept: Whether to include an intercept.

        Returns:
            Dictionary of diagnostic statistics.  Keys are
            family-specific (e.g. ``r_squared`` for linear,
            ``pseudo_r_squared`` for logistic).
        """
        ...

    def classical_p_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Compute classical (asymptotic) p-values via statsmodels.

        Used alongside the permutation p-values for comparison.

        Args:
            X: Design matrix (pandas DataFrame or ndarray).
            y: Response vector of shape ``(n,)``.
            fit_intercept: Whether to include an intercept.

        Returns:
            Array of p-values of shape ``(p,)``, one per slope
            coefficient.  Intercept p-value is excluded.
        """
        ...

    def batch_fit(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch-fit the model across many permuted response vectors.

        This is the hot loop of the permutation test.  Implementations
        should delegate to the active backend (NumPy or JAX) for
        maximum throughput.

        Args:
            X: Design matrix of shape ``(n, p)``.
            Y_matrix: Permuted response matrix of shape ``(B, n)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: Backend-specific options (e.g. ``max_iter``,
                ``tol`` for iterative solvers).

        Returns:
            Coefficient matrix of shape ``(B, p)`` where ``result[b]``
            contains the slope coefficients for ``Y_matrix[b]``.
        """
        ...


# ------------------------------------------------------------------ #
# LinearFamily
# ------------------------------------------------------------------ #
#
# OLS linear regression is the simplest and fastest family.  The
# closed-form solution β̂ = (X'X)⁻¹X'Y means every operation —
# fitting, prediction, residuals, batch fitting — reduces to matrix
# algebra with no iteration.
#
# **Residuals** are raw: e = Y - ŷ.  Under the Gauss-Markov
# assumptions (E[ε] = 0, Var(ε) = σ²I), these residuals are
# exchangeable under H₀: β_j = 0, which is the foundation for both
# the ter Braak and Kennedy permutation schemes.
#
# **Reconstruction** is additive: Y* = ŷ_{-j} + π(e_{-j}).  No
# stochastic step (Bernoulli sampling, rounding) is needed because
# Y is continuous — the sum of a continuous prediction and a
# continuous permuted residual is always a valid response value.
#
# **Joint-test metric** is residual sum of squares (RSS).  The test
# statistic RSS_reduced − RSS_full measures how much the tested
# features improve fit, analogous to the F-test numerator in
# classical ANOVA.  RSS is lower-is-better, so a large positive
# difference indicates that the features matter.
#
# **Batch fitting** is the hot loop.  The OLS pseudoinverse
# pinv(X) = (X'X)⁻¹X' depends only on X, which is constant across
# all B permutations.  Computing it once and multiplying against the
# (n × B) matrix of permuted Y vectors replaces B separate lstsq
# calls with a single BLAS-level matrix multiply — typically a 100×
# speedup for B = 5 000.  See ``_backends/_numpy.py`` for the
# implementation and ``_backends/_jax.py`` for the JIT-compiled
# variant.


@dataclass(frozen=True)
class LinearFamily:
    """OLS linear regression family.

    Implements the ``ModelFamily`` protocol for continuous outcomes
    using ordinary least squares.  Residuals are raw (``y - ŷ``),
    Y-reconstruction is additive (``ŷ + permuted residuals``), and
    the joint-test metric is residual sum of squares (RSS).

    The class is stateless — all data flows through method arguments.
    ``batch_fit`` delegates to the active backend (NumPy pseudoinverse
    or JAX ``vmap``'d ``lstsq``).
    """

    @property
    def name(self) -> str:
        return "linear"

    @property
    def residual_type(self) -> str:
        return "raw"

    @property
    def direct_permutation(self) -> bool:
        return False

    # ---- Validation ------------------------------------------------
    #
    # Catch two common data errors early:
    #   1. Non-numeric Y (e.g. string labels passed by mistake).
    #   2. Constant Y (zero variance) — OLS is vacuous when the
    #      response never varies because all residuals are zero and
    #      every permutation yields identical coefficients.

    def validate_y(self, y: np.ndarray) -> None:
        """Check that *y* is numeric and non-constant."""
        if not np.issubdtype(y.dtype, np.number):
            msg = "LinearFamily requires numeric Y values."
            raise ValueError(msg)
        if np.ptp(y) == 0:
            msg = "LinearFamily requires non-constant Y (zero variance)."
            raise ValueError(msg)

    # ---- Single-model operations -----------------------------------
    #
    # These methods wrap sklearn's LinearRegression, which internally
    # uses LAPACK (dgelsd) for the least-squares solve.  Each method
    # is a thin adapter that translates between the protocol's
    # array-in / array-out contract and sklearn's estimator API.

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> Any:
        """Fit an OLS model via sklearn ``LinearRegression``.

        Returns the fitted estimator object, which stores
        ``coef_`` (slopes) and ``intercept_`` internally.
        """
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X, y)
        return model

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Return ``ŷ = Xβ + β₀``.

        The prediction is deterministic — no link function inversion
        or rounding is needed for continuous outcomes.
        """
        return np.asarray(model.predict(X)).ravel()  # shape: (n,)

    def coefs(self, model: Any) -> np.ndarray:
        """Extract slope coefficients (intercept excluded).

        ``model.coef_`` may be 1-D or 2-D depending on sklearn's
        internal representation; ``np.ravel`` normalises both cases.
        """
        return np.ravel(model.coef_)  # shape: (p,)

    def residuals(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Raw residuals: ``e = y - ŷ``.

        For OLS with an intercept, residuals sum to exactly zero by
        construction (the normal equations guarantee Σeᵢ = 0).  This
        is the "raw" flavour — no standardisation or studentisation
        is applied because the permutation scheme needs residuals on
        the original scale of Y.
        """
        return np.asarray(y - self.predict(model, X))  # shape: (n,)

    # ---- Permutation helpers ---------------------------------------
    #
    # ``reconstruct_y`` and ``fit_metric`` are called inside the
    # permutation loop (once per predictor for ter Braak / Kennedy
    # individual, once total for Kennedy joint).

    def reconstruct_y(
        self,
        predictions: np.ndarray,
        permuted_residuals: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Additive reconstruction: ``Y* = ŷ + π(e)``.

        *rng* is accepted for protocol compatibility but unused —
        linear reconstruction is deterministic (no Bernoulli sampling
        or rounding step).  The resulting Y* is a valid continuous
        response because it is the sum of two real-valued arrays.
        """
        return np.asarray(predictions + permuted_residuals)  # shape: (B, n)

    def fit_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Residual sum of squares: ``RSS = Σ(yᵢ - ŷᵢ)²``.

        Computed as ``MSE × n`` via sklearn's ``mean_squared_error``
        to avoid a manual loop.  RSS is the natural goodness-of-fit
        measure for OLS — it is the quantity that the least-squares
        estimator minimises, and the F-statistic is a ratio of RSS
        reductions.
        """
        return float(mean_squared_error(y_true, y_pred) * len(y_true))

    # ---- Diagnostics & classical inference -------------------------
    #
    # These methods delegate to statsmodels, which provides a rich
    # suite of OLS summary statistics.  They are called once on the
    # observed (unpermuted) data — not inside any permutation loop —
    # so the cost of fitting a second model via statsmodels is
    # negligible.
    #
    # Key metrics:
    #   R²      — fraction of Y variance explained by X.
    #   Adj. R² — R² penalised for model complexity (p).
    #   F-stat  — joint test that all slope coefficients are zero:
    #               F = [(RSS_null − RSS_full) / p] / [RSS_full / (n − p − 1)]
    #   AIC     — Akaike information criterion: −2ℓ + 2k.
    #   BIC     — Bayesian information criterion: −2ℓ + k·ln(n).
    #
    # Classical p-values come from the Wald t-test:
    #   t_j = β̂_j / SE(β̂_j),   p = 2·P(|T| > |t_j|)
    # where SE(β̂_j) = σ̂ · √[(X'X)⁻¹]_{jj} and T ~ t(n − p − 1).
    # These assume normally distributed errors — the permutation test
    # provides an alternative that does not.

    def diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> dict[str, Any]:
        """OLS diagnostics via statsmodels (R², F-stat, AIC, BIC).

        When *fit_intercept* is True, ``sm.add_constant`` prepends a
        column of ones so that statsmodels estimates the same model as
        sklearn's ``LinearRegression(fit_intercept=True)``.
        """
        # statsmodels requires an explicit intercept column.
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        sm_model = sm.OLS(y, X_sm).fit()
        return {
            "n_observations": len(y),
            "n_features": X.shape[1],
            "r_squared": np.round(sm_model.rsquared, 4),
            "r_squared_adj": np.round(sm_model.rsquared_adj, 4),
            "f_statistic": np.round(sm_model.fvalue, 4),
            "f_p_value": sm_model.f_pvalue,
            "aic": np.round(sm_model.aic, 4),
            "bic": np.round(sm_model.bic, 4),
        }

    def classical_p_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Asymptotic t-test p-values via statsmodels OLS.

        Returns one p-value per slope coefficient (intercept excluded).
        RuntimeWarnings from near-singular designs are suppressed —
        the permutation p-value is the primary inference tool.
        """
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        with warnings.catch_warnings():
            # Near-singular X'X can trigger floating-point warnings in
            # the Wald SE computation; suppress them because the user
            # cares about the permutation p-value, not the asymptotic one.
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sm_model = sm.OLS(y, X_sm).fit()
        # sm_model.pvalues includes the intercept at index 0 when
        # fit_intercept is True; strip it to match the protocol contract
        # of returning only slope p-values.
        pvals = sm_model.pvalues[1:] if fit_intercept else sm_model.pvalues
        return np.asarray(pvals)  # shape: (p,)

    # ---- Batch fitting (hot loop) ----------------------------------
    #
    # This is by far the most performance-critical method.  For B = 5 000
    # permutations with n = 200 observations and p = 5 features, the
    # NumPy backend computes:
    #
    #   pinv(X_aug)  →  shape (p+1, n), computed once
    #   pinv @ Y'    →  (p+1, n) @ (n, B) = (p+1, B), one BLAS call
    #
    # Total: one SVD (for pinv) + one dgemm.  The JAX backend does the
    # same but JIT-compiles and optionally runs on GPU.  Either way,
    # the Python overhead is a single function call — no per-permutation
    # loop.

    def batch_fit(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch OLS via the active backend.

        Delegates to ``backend.batch_ols()`` resolved from the
        current configuration.  The backend handles intercept
        augmentation, pseudoinverse computation, and coefficient
        extraction internally.
        """
        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        if backend is None:
            backend = resolve_backend()
        return np.asarray(backend.batch_ols(X, Y_matrix, fit_intercept=fit_intercept))


# ------------------------------------------------------------------ #
# Family resolution
# ------------------------------------------------------------------ #
#
# The registry maps user-facing strings to factory callables.  Using
# callables (rather than pre-instantiated singletons) avoids import-
# time side effects from heavyweight families that may pull in optional
# dependencies.
#
# Concrete family classes are registered here as they are implemented
# in later steps.  Until then, the registry contains only placeholders
# that are populated during the Phase 1 build-out.

_FAMILIES: dict[str, type] = {}
"""Registry mapping family name strings to concrete ModelFamily classes."""


def register_family(name: str, cls: type) -> None:
    """Register a concrete ``ModelFamily`` class under *name*.

    Args:
        name: Lookup key (e.g. ``"linear"``, ``"logistic"``).
        cls: A class implementing the ``ModelFamily`` protocol.

    Raises:
        TypeError: If *cls* does not satisfy the ``ModelFamily``
            protocol.
    """
    # runtime_checkable protocols with non-method members do not
    # support issubclass(); use isinstance() on a sentinel instance.
    try:
        instance = cls()
    except Exception:  # noqa: BLE001
        msg = f"{cls!r} could not be instantiated for protocol check."
        raise TypeError(msg) from None
    if not isinstance(instance, ModelFamily):
        msg = f"{cls!r} does not implement the ModelFamily protocol."
        raise TypeError(msg)
    _FAMILIES[name] = cls


def resolve_family(family: str, y: np.ndarray) -> ModelFamily:
    """Resolve a family string to a concrete ``ModelFamily`` instance.

    ``"auto"`` → ``"linear"`` or ``"logistic"`` via binary detection
    (exactly two unique values, both in ``{0, 1}``).

    Explicit strings (``"linear"``, ``"logistic"``, ``"poisson"``,
    etc.) map directly to the corresponding registered class.

    Args:
        family: Family identifier.  ``"auto"`` triggers automatic
            detection.
        y: Response vector of shape ``(n,)``, used only when
            *family* is ``"auto"``.

    Returns:
        A ``ModelFamily`` instance ready for use by the permutation
        engine.

    Raises:
        ValueError: If *family* is not ``"auto"`` and is not found
            in the registry.
    """
    if family == "auto":
        unique_y = np.unique(y)
        is_binary = bool(len(unique_y) == 2 and np.all(np.isin(unique_y, [0, 1])))
        family = "logistic" if is_binary else "linear"

    if family not in _FAMILIES:
        available = ", ".join(sorted(_FAMILIES)) or "(none registered)"
        msg = f"Unknown family {family!r}.  Available families: {available}."
        raise ValueError(msg)

    instance: ModelFamily = _FAMILIES[family]()
    return instance


# ------------------------------------------------------------------ #
# Register built-in families
# ------------------------------------------------------------------ #

register_family("linear", LinearFamily)

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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning as SmConvergenceWarning,
)
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationWarning,
)

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
        metric_label: Human-readable label for the joint-test fit
            metric (e.g. ``"RSS Reduction"``, ``"Deviance Reduction"``).
    """

    @property
    def name(self) -> str: ...

    @property
    def residual_type(self) -> str: ...

    @property
    def direct_permutation(self) -> bool: ...

    @property
    def metric_label(self) -> str:
        """Human-readable label for the joint-test fit metric.

        Displayed in result dicts and table output to identify which
        goodness-of-fit measure underlies the joint test statistic.
        Examples: ``"RSS Reduction"`` (linear), ``"Deviance Reduction"``
        (logistic, Poisson, negative binomial).
        """
        ...

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

    # ---- Exchangeability (v0.4.0 forward-compat) -------------------

    def exchangeability_cells(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray | None:
        """Return group labels defining exchangeability cells, or ``None``.

        Under v0.4.0 the permutation engine will restrict permutations
        to within-cell shuffles when this returns a non-``None`` array.
        Families that assume global exchangeability (linear, logistic)
        return ``None``; families with structured residuals (e.g.
        mixed-effects) may return cluster labels.

        This method exists on the protocol now so that v0.4.0 can
        call it on any family without a protocol-breaking change.

        Args:
            X: Design matrix ``(n, p)``.
            y: Response vector ``(n,)``.

        Returns:
            Integer label array ``(n,)`` or ``None`` for global
            exchangeability.
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

    def batch_fit_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch-fit the model across many permuted design matrices.

        Used by the Kennedy individual path, where the *design matrix*
        changes each permutation (column *j* is replaced with permuted
        exposure residuals) while the response vector stays the same.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared response vector ``(n,)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: Backend-specific options.

        Returns:
            Coefficient matrix ``(B, p)`` (intercept excluded).
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
class _InterceptOnlyOLS:
    """Lightweight stub for the intercept-only OLS model (0 features).

    When the ter Braak engine drops the single column from a
    one-feature dataset, the reduced design matrix has shape ``(n, 0)``.
    sklearn's ``LinearRegression`` rejects that, so ``LinearFamily.fit``
    returns this stub instead.

    The stub exposes the same ``predict`` / ``coef_`` surface that
    ``LinearFamily.predict``, ``LinearFamily.coefs``, and
    ``LinearFamily.residuals`` rely on:

    * ``predict(X)`` → constant vector of ``intercept_`` (or 0).
    * ``coef_`` → empty 1-D array.
    """

    intercept_: float
    coef_: np.ndarray  # always shape (0,)

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Return a constant prediction vector (intercept only)."""
        n = X.shape[0]
        return np.full(n, self.intercept_)


def _make_intercept_only_ols(y: np.ndarray, fit_intercept: bool) -> _InterceptOnlyOLS:
    """Factory for the intercept-only OLS stub."""
    intercept = float(np.mean(y)) if fit_intercept else 0.0
    return _InterceptOnlyOLS(intercept_=intercept, coef_=np.empty(0))


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

    @property
    def metric_label(self) -> str:
        return "RSS Reduction"

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

        When *X* has zero columns (intercept-only model, arising from
        the ter Braak reduced fit on a single-feature dataset), a
        lightweight stub is returned instead, since sklearn rejects
        zero-feature arrays.
        """
        if X.shape[1] == 0:
            return _make_intercept_only_ols(y, fit_intercept)
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

    # ---- Exchangeability (v0.4.0 forward-compat) -------------------

    def exchangeability_cells(
        self,
        X: np.ndarray,  # noqa: ARG002
        y: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray | None:
        """Linear models assume globally exchangeable residuals."""
        return None

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

        The ``batch_ols`` path is already fully vectorised (single
        pseudoinverse multiply), so ``n_jobs`` is accepted for
        interface consistency but has no effect here.
        """
        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        kwargs.pop("n_jobs", None)  # OLS is vectorised; n_jobs unused
        if backend is None:
            backend = resolve_backend()
        return np.asarray(backend.batch_ols(X, Y_matrix, fit_intercept=fit_intercept))

    def batch_fit_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch OLS with per-permutation design matrices.

        Delegates to ``backend.batch_ols_varying_X()`` for the
        Kennedy individual path where column *j* of *X* differs
        across permutations.  Forwards ``n_jobs`` to the NumPy
        backend for parallel ``lstsq`` solves; the JAX backend
        uses ``vmap`` and does not accept ``n_jobs``.
        """
        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()
        if backend.name == "numpy":
            return np.asarray(
                backend.batch_ols_varying_X(
                    X_batch,
                    y,
                    fit_intercept=fit_intercept,
                    n_jobs=n_jobs,
                )
            )
        return np.asarray(
            backend.batch_ols_varying_X(X_batch, y, fit_intercept=fit_intercept)
        )


# ------------------------------------------------------------------ #
# LogisticFamily
# ------------------------------------------------------------------ #
#
# Logistic regression models binary outcomes Y ∈ {0, 1} via the
# logistic (sigmoid) link function:
#
#   P(Y = 1 | X) = σ(Xβ) = 1 / (1 + exp(−Xβ))
#
# There is no closed-form solution — fitting requires iterative
# optimisation (Newton–Raphson, L-BFGS, etc.).  This has two major
# implications for the permutation framework:
#
# 1. **Residuals are on the probability scale**: e = Y − P̂(Y=1|X).
#    These range from (−1, 0) when Y=0 and (0, +1) when Y=1,
#    unlike OLS residuals which span the full real line.  Permuting
#    them and adding back to P̂ can produce values outside [0, 1],
#    so we clip to [0.001, 0.999] before reconstruction.
#
# 2. **Reconstruction requires Bernoulli sampling**: after clipping,
#    Y* ~ Bernoulli(clip(P̂ + π(e))).  This stochastic step
#    converts the continuous permuted probability back to a valid
#    binary outcome.  Without it, the refitted logistic model would
#    receive non-binary Y, which violates the likelihood.
#
# **Joint-test metric** is deviance = 2 × binary cross-entropy
# (unnormalised).  The test statistic Deviance_reduced − Deviance_full
# measures how much the tested features improve fit, analogous to
# the likelihood-ratio test in classical GLM theory.
#
# **Batch fitting** is the performance bottleneck.  Each of the B
# permutations requires a fresh Newton–Raphson solve.  When JAX is
# available, ``jax.vmap`` vectorises all B solves into a single XLA
# kernel launch (GPU-acceleratable).  The NumPy fallback loops over
# B sklearn fits — correct but ~100× slower for large B.
#
# **Diagnostics** use statsmodels ``Logit`` for McFadden's pseudo-R²,
# log-likelihood, LLR test, AIC, and BIC.  Classical p-values come
# from the Wald z-test (asymptotic normality of the MLE), which can
# be unreliable with small samples or quasi-complete separation —
# precisely the situations where the permutation test adds value.


@dataclass(frozen=True)
class LogisticFamily:
    """Logistic regression family for binary outcomes.

    Implements the ``ModelFamily`` protocol for binary Y ∈ {0, 1}
    using maximum-likelihood logistic regression.  Residuals are on
    the probability scale (``Y − P̂``), reconstruction uses Bernoulli
    sampling, and the joint-test metric is deviance.

    The class is stateless — all data flows through method arguments.
    ``batch_fit`` delegates to the active backend (sklearn loop or
    JAX ``vmap``'d Newton–Raphson).
    """

    @property
    def name(self) -> str:
        return "logistic"

    @property
    def residual_type(self) -> str:
        return "probability"

    @property
    def direct_permutation(self) -> bool:
        return False

    @property
    def metric_label(self) -> str:
        return "Deviance Reduction"

    # ---- Validation ------------------------------------------------
    #
    # Logistic regression requires exactly two classes, coded as 0/1.
    # Other binary encodings (−1/+1, "yes"/"no") must be recoded
    # before reaching this point.  A single-class Y is degenerate —
    # the MLE does not exist because the log-likelihood is flat.

    def validate_y(self, y: np.ndarray) -> None:
        """Check that *y* is binary with values in {0, 1}."""
        unique = np.unique(y)
        if not (len(unique) == 2 and np.all(np.isin(unique, [0, 1]))):
            msg = (
                "LogisticFamily requires binary Y with exactly two "
                "unique values in {0, 1}."
            )
            raise ValueError(msg)

    # ---- Single-model operations -----------------------------------
    #
    # These methods wrap sklearn's LogisticRegression with
    # ``penalty=None`` (unregularised MLE) and ``solver='lbfgs'``.
    # The max_iter=5000 default gives the L-BFGS solver ample room
    # to converge even on ill-conditioned designs.  For the hot-loop
    # batch path, the JAX Newton–Raphson solver (with damping and
    # float64) is used instead — these single-model methods are only
    # called once on the observed data.

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> Any:
        """Fit a logistic model via sklearn ``LogisticRegression``.

        Uses unregularised MLE (``penalty=None``) so that
        coefficients are comparable to the statsmodels Wald test.
        """
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=5_000,
            fit_intercept=fit_intercept,
        )
        model.fit(X, y)
        return model

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities ``P̂(Y=1|X)``.

        For logistic models, "prediction" is the probability of the
        positive class, not the hard label.  This is what the
        residual and reconstruction methods expect.
        """
        # predict_proba returns shape (n, 2); column 1 is P(Y=1).
        return np.asarray(model.predict_proba(X)[:, 1])  # shape: (n,)

    def coefs(self, model: Any) -> np.ndarray:
        """Extract slope coefficients (intercept excluded).

        ``model.coef_`` is shape (1, p) for binary classification;
        ``flatten`` normalises to (p,).
        """
        return np.ravel(model.coef_)  # shape: (p,)

    def residuals(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Probability-scale residuals: ``e = Y − P̂(Y=1|X)``.

        These range from (−1, 0) when Y=0 and (0, +1) when Y=1.
        Unlike OLS raw residuals, they do NOT sum to zero in general
        (the score equation for logistic regression is
        Σ(yᵢ − p̂ᵢ)xᵢ = 0, not Σ(yᵢ − p̂ᵢ) = 0 unless there is
        an intercept and X contains a constant column).
        """
        return np.asarray(y - self.predict(model, X))  # shape: (n,)

    # ---- Permutation helpers ---------------------------------------
    #
    # The logistic reconstruction pipeline has three stages:
    #   1. Add permuted residuals to reduced-model predictions.
    #   2. Clip to [0.001, 0.999] to avoid log(0) in the refit.
    #   3. Draw Y* ~ Bernoulli(clipped probability).
    #
    # The clip bounds are deliberately not [0, 1] because exact 0/1
    # probabilities cause infinite log-odds, which makes the
    # Newton–Raphson solver diverge immediately.

    def reconstruct_y(
        self,
        predictions: np.ndarray,
        permuted_residuals: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Clip + Bernoulli reconstruction for binary outcomes.

        Steps:
          1. ``p* = clip(P̂ + π(e), 0.001, 0.999)``
          2. ``Y* ~ Bernoulli(p*)``

        *rng* is required — unlike the linear family, logistic
        reconstruction is stochastic.
        """
        # Stage 1-2: clipped probability.
        probs = np.clip(
            predictions + permuted_residuals, 0.001, 0.999
        )  # shape: (B, n) or (n,)
        # Stage 3: Bernoulli draw — each element of Y* is an
        # independent coin flip with probability p*.
        return np.asarray(rng.binomial(1, probs))  # shape: same as probs

    def fit_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Deviance: ``2 × binary cross-entropy (unnormalised)``.

        Deviance = −2 Σ[yᵢ log p̂ᵢ + (1−yᵢ) log(1−p̂ᵢ)]

        This is the logistic analogue of RSS — the quantity that MLE
        minimises (up to a constant).  The joint test uses
        Deviance_reduced − Deviance_full as its test statistic,
        mirroring the likelihood-ratio test.

        ``y_pred`` is clipped to [0.001, 0.999] to avoid log(0).
        """
        # Clip predictions to avoid log(0) / inf in cross-entropy.
        y_pred_safe = np.clip(y_pred, 0.001, 0.999)
        return float(2.0 * log_loss(y_true, y_pred_safe, normalize=False))

    # ---- Diagnostics & classical inference -------------------------
    #
    # Logistic diagnostics use statsmodels ``Logit``, which provides:
    #
    #   Pseudo R² — McFadden's R² = 1 − ℓ(model)/ℓ(null).
    #     Unlike linear R², this is NOT bounded above by 1 in
    #     practice.  Values of 0.2–0.4 are considered excellent fit
    #     for discrete-choice models (McFadden, 1977).
    #
    #   Log-likelihood — ℓ(model) and ℓ(null) for the intercept-only
    #     model.  The difference drives the LLR test.
    #
    #   LLR p-value — likelihood-ratio test: −2[ℓ(null) − ℓ(model)]
    #     ~ χ²(p) under H₀.  This is the logistic analogue of the
    #     F-test in OLS.
    #
    #   AIC / BIC — information criteria for model comparison.
    #
    # Classical p-values come from the Wald z-test:
    #   z_j = β̂_j / SE(β̂_j),   p = 2·P(|Z| > |z_j|)
    # where SE is derived from the observed Fisher information
    # (inverse Hessian of the log-likelihood).  These assume
    # asymptotic normality of the MLE — unreliable with small n,
    # rare events, or quasi-complete separation.

    def diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> dict[str, Any]:
        """Logistic diagnostics via statsmodels (pseudo-R², LLR, AIC, BIC).

        When *fit_intercept* is True, ``sm.add_constant`` prepends a
        column of ones so that statsmodels estimates the same model as
        sklearn's ``LogisticRegression(fit_intercept=True)``.
        """
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        # disp=0 suppresses the iteration log that statsmodels prints
        # by default for iterative MLE solvers.
        sm_model = sm.Logit(y, X_sm).fit(disp=0)
        return {
            "n_observations": len(y),
            "n_features": X.shape[1],
            "pseudo_r_squared": np.round(sm_model.prsquared, 4),
            "log_likelihood": np.round(sm_model.llf, 4),
            "log_likelihood_null": np.round(sm_model.llnull, 4),
            "llr_p_value": sm_model.llr_pvalue,
            "aic": np.round(sm_model.aic, 4),
            "bic": np.round(sm_model.bic, 4),
        }

    def classical_p_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Asymptotic Wald z-test p-values via statsmodels Logit.

        Returns one p-value per slope coefficient (intercept excluded).
        PerfectSeparationWarning and ConvergenceWarning are suppressed
        — the permutation p-value is the primary inference tool.
        """
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        with warnings.catch_warnings():
            # Quasi-complete separation inflates SEs to infinity,
            # making Wald p-values meaningless.  Suppress the warning
            # because the user relies on permutation p-values.
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sm_model = sm.Logit(y, X_sm).fit(disp=0)
        # sm_model.pvalues includes the intercept at index 0 when
        # fit_intercept is True; strip it to match the protocol contract.
        pvals = sm_model.pvalues[1:] if fit_intercept else sm_model.pvalues
        return np.asarray(pvals)  # shape: (p,)

    # ---- Exchangeability (v0.4.0 forward-compat) -------------------

    def exchangeability_cells(
        self,
        X: np.ndarray,  # noqa: ARG002
        y: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray | None:
        """Logistic models assume globally exchangeable residuals."""
        return None

    # ---- Batch fitting (hot loop) ----------------------------------
    #
    # Unlike OLS, logistic batch fitting requires B independent
    # Newton–Raphson solves — there is no pseudoinverse shortcut.
    #
    # Backend dispatch:
    #   NumPy: sequential sklearn loop, ~1 fit/ms per permutation.
    #   JAX:   vmap'd Newton–Raphson with jit compilation and
    #          optional GPU execution.  All B solves run as a single
    #          XLA kernel.  Float64, damped Hessian, triple
    #          convergence criteria.  See ``_backends/_jax.py``.
    #
    # kwargs forwarded to the backend:
    #   max_iter (int)  — Newton–Raphson iteration cap (default 100).
    #   tol (float)     — convergence tolerance (default 1e-8).

    def batch_fit(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch logistic via the active backend.

        Delegates to ``backend.batch_logistic()`` for the shared-X
        case (ter Braak: same X, many permuted Y vectors).  Forwards
        ``n_jobs`` only to the NumPy backend; the JAX backend uses
        ``vmap`` and does not accept ``n_jobs``.
        """
        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()
        if backend.name == "numpy":
            return np.asarray(
                backend.batch_logistic(
                    X,
                    Y_matrix,
                    fit_intercept=fit_intercept,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            )
        return np.asarray(
            backend.batch_logistic(
                X,
                Y_matrix,
                fit_intercept=fit_intercept,
                **kwargs,
            )
        )

    def batch_fit_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch logistic with per-permutation design matrices.

        Delegates to ``backend.batch_logistic_varying_X()`` for the
        Kennedy individual path where column *j* of *X* differs
        across permutations.  Forwards ``n_jobs`` only to the NumPy
        backend.
        """
        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()
        if backend.name == "numpy":
            return np.asarray(
                backend.batch_logistic_varying_X(
                    X_batch,
                    y,
                    fit_intercept=fit_intercept,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            )
        return np.asarray(
            backend.batch_logistic_varying_X(
                X_batch,
                y,
                fit_intercept=fit_intercept,
                **kwargs,
            )
        )


# ------------------------------------------------------------------ #
# PoissonFamily
# ------------------------------------------------------------------ #
#
# Poisson regression models count outcomes Y ∈ {0, 1, 2, …} via the
# log link function:
#
#   E[Y | X] = μ = exp(Xβ)
#
# The canonical family for equi-dispersed count data.  Fitting uses
# statsmodels GLM with IRLS (iteratively reweighted least squares).
#
# **Residuals** are deviance residuals:
#   d_i = sign(y_i − μ̂_i) × √(2 [y_i log(y_i/μ̂_i) − (y_i − μ̂_i)])
#
# Deviance residuals have approximate unit variance when the Poisson
# assumption holds and μ̂ is moderately large (> 5).  For small counts,
# they are noticeably right-skewed — this does NOT invalidate the
# permutation test (exchangeability under H₀ is what matters, not
# normality), but it explains why Poisson permutation p-values may
# differ more from classical p-values than linear ones do.
#
# **Reconstruction** uses Poisson sampling:
#   1. Transform predictions to log-link scale: η = log(μ̂)
#   2. Add permuted deviance residuals: η* = η + π(d)
#   3. Back-transform: μ* = exp(η*)
#   4. Draw Y* ~ Poisson(max(μ*, 1e-10))
#
# This mirrors the logistic family's Bernoulli sampling — both
# reconstruct valid responses from the family's natural distribution
# rather than rounding or truncating continuous values.
#
# **Joint-test metric** is Poisson deviance:
#   D = 2 Σ[y_i log(y_i/μ̂_i) − (y_i − μ̂_i)]
#
# with the convention that 0·log(0/μ̂) = 0 (L'Hôpital).
#
# **Batch fitting** is a joblib-parallelised statsmodels IRLS loop —
# there is no closed-form solution or JAX Poisson solver, so each of
# the B permutations requires an independent GLM fit.  Convergence
# failures on individual permuted Y vectors produce NaN coefficient
# rows; an aggregated warning is emitted after the loop.


@dataclass(frozen=True)
class PoissonFamily:
    """Poisson regression family for count outcomes.

    Implements the ``ModelFamily`` protocol for non-negative integer
    outcomes using the Poisson log-link GLM.  Residuals are response-
    scale (y − μ̂), reconstruction uses Poisson sampling on the
    response scale, and the joint-test metric is deviance.

    The class is stateless — all data flows through method arguments.
    ``batch_fit`` uses a joblib-parallelised statsmodels loop (no
    backend delegation — no JAX Poisson solver exists yet).
    """

    @property
    def name(self) -> str:
        return "poisson"

    @property
    def residual_type(self) -> str:
        return "response"

    @property
    def direct_permutation(self) -> bool:
        return False

    @property
    def metric_label(self) -> str:
        return "Deviance Reduction"

    # ---- Validation ------------------------------------------------
    #
    # Poisson requires non-negative values.  We accept floats that are
    # non-negative whole numbers (statsmodels does too), but reject
    # negative values, NaN, and non-integer floats.

    def validate_y(self, y: np.ndarray) -> None:
        """Check that *y* contains non-negative integer-valued data."""
        if not np.issubdtype(y.dtype, np.number):
            msg = "PoissonFamily requires numeric Y values."
            raise ValueError(msg)
        if np.any(np.isnan(y)):
            msg = "PoissonFamily does not accept NaN values in Y."
            raise ValueError(msg)
        if np.any(y < 0):
            msg = "PoissonFamily requires non-negative Y values."
            raise ValueError(msg)
        # Allow floats that happen to be whole numbers (e.g. 3.0),
        # but reject genuinely fractional values like 3.5.
        if not np.allclose(y, np.round(y)):
            msg = "PoissonFamily requires integer-valued Y. Got non-integer values."
            raise ValueError(msg)

    # ---- Single-model operations -----------------------------------
    #
    # These methods wrap statsmodels GLM with a Poisson family.
    # ``disp=0`` suppresses the iteration log.

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> Any:
        """Fit a Poisson GLM via statsmodels IRLS.

        Returns the fitted ``GLMResultsWrapper`` object.  Convergence
        and runtime warnings are suppressed — the permutation p-value
        is the primary inference tool.
        """
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model = sm.GLM(y, X_sm, family=sm.families.Poisson()).fit(disp=0)
        return model

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Return predicted mean μ̂ on the response scale.

        ``model.predict()`` applies the inverse log link
        (exponentiation) automatically, so the returned values are
        non-negative counts (not log-counts).
        """
        # Use fittedvalues when X matches the training data (avoids
        # re-applying add_constant).  Fall back to model.predict()
        # with explicit X when shapes differ (e.g. reduced model).
        return np.asarray(model.fittedvalues)

    def coefs(self, model: Any) -> np.ndarray:
        """Extract slope coefficients (intercept excluded).

        ``model.params`` includes the intercept at index 0 when the
        model was fit with a constant column; strip it.
        """
        # GLM models always include the constant in params when
        # add_constant was used.  The model's df_model tells us the
        # number of slope parameters, but it's simpler to check
        # whether the first column is the constant by convention.
        # We follow the same pattern as LogisticFamily: always strip
        # index 0 when the model has more params than the original X.
        params = np.asarray(model.params)
        # If the model has an intercept, the number of params exceeds
        # the model's df_model by 1 (for the constant).
        if model.k_constant:
            return params[1:]
        return params

    def residuals(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Response-scale residuals: ``e = y − μ̂``.

        These are the direct analogue of the linear family's raw
        residuals and the logistic family's probability-scale
        residuals.  The permutation engine permutes these and then
        ``reconstruct_y`` adds them back on the response scale:
        μ* = μ̂ + π(e), Y* ~ Poisson(μ*).

        Using response-scale residuals (rather than deviance residuals)
        is critical because ``reconstruct_y`` operates on the response
        scale — deviance residuals are approximately N(0, 1) and adding
        them to μ̂ (which is on the count scale) produces wildly
        inflated/deflated rates.
        """
        return np.asarray(y - model.fittedvalues)

    # ---- Permutation helpers ---------------------------------------
    #
    # ``reconstruct_y`` uses Poisson sampling (not rounding) to
    # produce valid count responses from permuted response-scale
    # residuals.  This mirrors LogisticFamily's Bernoulli sampling —
    # both reconstruct on the response scale and sample from the
    # family's natural distribution.

    def reconstruct_y(
        self,
        predictions: np.ndarray,
        permuted_residuals: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Poisson-sampled reconstruction from permuted residuals.

        Steps:
          1. ``μ* = clip(μ̂ + π(e), 1e-10, 1e8)``
          2. ``Y* ~ Poisson(μ*)``

        The residuals are on the response scale (y − μ̂), matching
        the logistic family's Bernoulli reconstruction from
        probability-scale residuals (y − p̂).  Working on the
        response scale (instead of the link scale) ensures that
        the reconstructed rates μ* stay in a realistic range.

        *rng* is required — like the logistic family, Poisson
        reconstruction is stochastic.
        """
        # Response-scale reconstruction: μ̂ + π(y − μ̂).
        mu_star = predictions + permuted_residuals
        # Clamp to avoid negative or extreme λ values that would
        # cause Poisson sampling to fail or consume excessive memory.
        mu_star = np.clip(mu_star, 1e-10, 1e8)
        # Poisson draw.
        return np.asarray(rng.poisson(lam=mu_star))

    def fit_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Poisson deviance: 2 Σ[y log(y/μ̂) − (y − μ̂)].

        The convention 0·log(0/μ̂) = 0 is applied via ``np.where``
        to avoid NaN from 0 × −inf in floating point.  The metric is
        lower-is-better, so the joint test statistic
        D_reduced − D_full is positive when the features improve fit.
        """
        mu = np.maximum(y_pred, 1e-10)
        # 0 * log(0/μ̂) = 0 by convention (L'Hôpital).
        # Use mask-and-index to avoid evaluating log(0) (np.where
        # evaluates both branches eagerly, producing RuntimeWarnings).
        pos = y_true > 0
        contrib = np.zeros_like(y_true, dtype=float)
        contrib[pos] = y_true[pos] * np.log(y_true[pos] / mu[pos])
        deviance_i = contrib - (y_true - mu)
        return float(2.0 * np.sum(deviance_i))

    # ---- Diagnostics & classical inference -------------------------
    #
    # Poisson diagnostics use statsmodels GLM, which provides:
    #
    #   Deviance    — 2[ℓ(saturated) − ℓ(model)].
    #   Pearson χ²  — Σ(y − μ̂)²/μ̂.
    #   Dispersion  — Pearson χ² / df_resid.  Values > 1 indicate
    #                 overdispersion (violation of the equi-dispersion
    #                 assumption).
    #   AIC / BIC   — information criteria.
    #
    # Classical p-values come from the Wald z-test:
    #   z_j = β̂_j / SE(β̂_j),   p = 2·P(|Z| > |z_j|)
    # where SE is derived from the Fisher information matrix.

    def diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> dict[str, Any]:
        """Poisson GLM diagnostics (deviance, Pearson χ², dispersion, AIC, BIC)."""
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sm_model = sm.GLM(y, X_sm, family=sm.families.Poisson()).fit(disp=0)
        pearson_chi2 = float(sm_model.pearson_chi2)
        df_resid = float(sm_model.df_resid)
        dispersion = pearson_chi2 / df_resid if df_resid > 0 else float("nan")
        return {
            "n_observations": len(y),
            "n_features": X.shape[1],
            "deviance": np.round(float(sm_model.deviance), 4),
            "pearson_chi2": np.round(pearson_chi2, 4),
            "dispersion": np.round(dispersion, 4),
            "log_likelihood": np.round(float(sm_model.llf), 4),
            "aic": np.round(float(sm_model.aic), 4),
            "bic": np.round(float(sm_model.bic_llf), 4),
        }

    def classical_p_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Asymptotic Wald z-test p-values via statsmodels Poisson GLM.

        Returns one p-value per slope coefficient (intercept excluded).
        Convergence warnings are suppressed — the permutation p-value
        is the primary inference tool.
        """
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sm_model = sm.GLM(y, X_sm, family=sm.families.Poisson()).fit(disp=0)
        pvals = sm_model.pvalues[1:] if fit_intercept else sm_model.pvalues
        return np.asarray(pvals)

    # ---- Exchangeability (v0.4.0 forward-compat) -------------------

    def exchangeability_cells(
        self,
        X: np.ndarray,  # noqa: ARG002
        y: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray | None:
        """Poisson models assume globally exchangeable residuals."""
        return None

    # ---- Batch fitting (hot loop) ----------------------------------
    #
    # Unlike OLS (pseudoinverse) or logistic (JAX vmap), Poisson batch
    # fitting requires B independent IRLS solves via statsmodels.
    # There is no closed-form shortcut and no JAX Poisson solver.
    #
    # The loop is parallelised via joblib when n_jobs != 1.  Individual
    # fit failures (convergence issues on extreme permuted Y vectors)
    # produce NaN coefficient rows; an aggregated warning is emitted
    # after the loop.

    def batch_fit(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson GLM via joblib-parallelised statsmodels loop.

        Each row of *Y_matrix* is a permuted response vector.  Returns
        ``(B, p)`` coefficient matrix with intercept excluded.
        Convergence failures produce NaN rows and a single aggregated
        warning.
        """
        kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)

        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        n_params = X.shape[1]
        B = Y_matrix.shape[0]

        def _fit_one(y_b: np.ndarray) -> np.ndarray:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y_b, X_sm, family=sm.families.Poisson()).fit(
                        disp=0, maxiter=100
                    )
                params = np.asarray(model.params)
                return params[1:] if fit_intercept else params
            except Exception:  # noqa: BLE001
                return np.full(n_params, np.nan)

        if n_jobs == 1:
            results = [_fit_one(Y_matrix[i]) for i in range(B)]
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(Y_matrix[i]) for i in range(B)
            )

        coef_matrix = np.array(results)

        # Aggregated convergence warning.
        n_failed = int(np.sum(np.any(np.isnan(coef_matrix), axis=1)))
        if n_failed > 0:
            pct = 100.0 * n_failed / B
            warnings.warn(
                f"{n_failed} of {B} Poisson fits did not converge "
                f"({pct:.1f}%). Consider checking for overdispersion "
                f"or zero-inflation.",
                UserWarning,
                stacklevel=2,
            )

        return coef_matrix

    def batch_fit_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson with per-permutation design matrices.

        Used by the Kennedy individual path where column *j* of *X*
        differs across permutations while Y stays the same.  Same
        joblib-parallelised statsmodels loop as ``batch_fit``.
        """
        kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)

        B, _n, p = X_batch.shape
        n_params = p

        def _fit_one(X_b: np.ndarray) -> np.ndarray:
            X_sm = sm.add_constant(X_b) if fit_intercept else np.asarray(X_b)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y, X_sm, family=sm.families.Poisson()).fit(
                        disp=0, maxiter=100
                    )
                params = np.asarray(model.params)
                return params[1:] if fit_intercept else params
            except Exception:  # noqa: BLE001
                return np.full(n_params, np.nan)

        if n_jobs == 1:
            results = [_fit_one(X_batch[i]) for i in range(B)]
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(X_batch[i]) for i in range(B)
            )

        coef_matrix = np.array(results)

        n_failed = int(np.sum(np.any(np.isnan(coef_matrix), axis=1)))
        if n_failed > 0:
            pct = 100.0 * n_failed / B
            warnings.warn(
                f"{n_failed} of {B} Poisson fits did not converge "
                f"({pct:.1f}%). Consider checking for overdispersion "
                f"or zero-inflation.",
                UserWarning,
                stacklevel=2,
            )

        return coef_matrix


# ------------------------------------------------------------------ #
# NegativeBinomialFamily
# ------------------------------------------------------------------ #
#
# Negative binomial regression models overdispersed count outcomes
# Y ∈ {0, 1, 2, …} via the log link function:
#
#   E[Y | X] = μ = exp(Xβ)
#   Var(Y | X) = μ + α·μ²       (NB2 parameterisation)
#
# where α > 0 is the dispersion parameter.  When α = 0 the model
# reduces to Poisson.  The NB2 parameterisation is the statsmodels
# default and the most common in applied work.
#
# **Key design decision** (from the v0.3.0 plan):
# Estimate α ONCE on the observed (unpermuted) data using
# ``sm.NegativeBinomial(y, X).fit()``, which jointly estimates β and
# α via maximum likelihood.  Hold α fixed for all permutation refits
# by passing it to ``sm.GLM(family=NegativeBinomial(alpha=α_hat))``.
# This avoids re-estimating α B times (expensive, unnecessary under
# H₀, and can cause convergence failures on extreme permuted Ys).
#
# **Residuals** are response-scale: ``e = y − μ̂``.  This follows
# the same rationale as PoissonFamily — response-scale residuals
# are scale-compatible with ``reconstruct_y``, which operates on the
# response scale.
#
# **Reconstruction** uses negative binomial sampling:
#   1. μ* = clip(μ̂ + π(e), 1e-10, 1e8)
#   2. Convert to NB parameters: n = 1/α, p = 1/(1 + α·μ*)
#   3. Draw Y* ~ NegBin(n, p)
#
# **Joint-test metric** is NB deviance:
#   D = 2 Σ[y·log(y/μ̂) − (y + 1/α)·log((1 + α·y)/(1 + α·μ̂))]
#
# with the convention that 0·log(0/·) = 0.
#
# **Batch fitting** uses the same joblib-parallelised statsmodels
# loop as PoissonFamily, but with ``sm.families.NegativeBinomial``
# and a fixed α.


@dataclass(frozen=True)
class NegativeBinomialFamily:
    """Negative binomial regression family for overdispersed count outcomes.

    Implements the ``ModelFamily`` protocol for non-negative integer
    outcomes where the variance exceeds the mean (overdispersion).
    Uses the NB2 parameterisation: ``Var(Y) = μ + α·μ²``.

    The dispersion parameter α is estimated once on the observed data
    via :meth:`calibrate` (which calls ``sm.NegativeBinomial`` MLE)
    and returns a **new frozen instance** with α baked in.  All
    subsequent calls to ``fit``, ``batch_fit``, ``reconstruct_y``,
    and ``fit_metric`` use the fixed α without re-estimating.

    Users may also supply α directly at construction time, in which
    case :meth:`calibrate` is a no-op (idempotent).

    Parameters
    ----------
    alpha : float or None
        Dispersion parameter.  If ``None`` (default), must be
        resolved via :meth:`calibrate` before fitting.
    """

    alpha: float | None = None

    @property
    def name(self) -> str:
        return "negative_binomial"

    @property
    def residual_type(self) -> str:
        return "response"

    @property
    def direct_permutation(self) -> bool:
        return False

    @property
    def metric_label(self) -> str:
        return "Deviance Reduction"

    # ---- Internal helpers ------------------------------------------

    def _nb_family(self, alpha: float) -> sm.families.NegativeBinomial:
        """Create a statsmodels NB2 family object with fixed α."""
        return sm.families.NegativeBinomial(alpha=alpha)

    def _require_alpha(self, method: str) -> float:
        """Return α or raise if uncalibrated."""
        if self.alpha is None:
            msg = (
                f"NegativeBinomialFamily.{method} requires α to be "
                "estimated first.  Call calibrate() on the observed data "
                "before running permutations."
            )
            raise RuntimeError(msg)
        return self.alpha

    # ---- Calibration (nuisance-parameter estimation) ---------------

    def calibrate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> ModelFamily:
        """Estimate dispersion α from the observed data.

        Uses ``sm.NegativeBinomial(y, X).fit()`` which jointly
        estimates β and ln(α) via MLE, then extracts
        ``α = exp(ln_alpha)``.

        Returns a **new** ``NegativeBinomialFamily(alpha=α_hat)``
        instance.  If ``self.alpha`` is already set (either by
        construction or a prior call), returns ``self`` — making
        this method **idempotent**.

        In v0.4.0+ this hook will be called by
        ``PermutationEngine.__init__`` before the permutation loop,
        keeping nuisance-parameter estimation orthogonal to
        ``exchangeability_cells()``.

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            y: Response vector ``(n,)``.
            fit_intercept: Whether the model includes an intercept.

        Returns:
            A calibrated ``NegativeBinomialFamily`` with α resolved.
        """
        if self.alpha is not None:
            return self
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            nb_model = sm.NegativeBinomial(y, X_sm).fit(disp=0, maxiter=200)
        alpha_hat = float(np.exp(nb_model.lnalpha))
        return NegativeBinomialFamily(alpha=alpha_hat)

    # ---- Validation ------------------------------------------------

    def validate_y(self, y: np.ndarray) -> None:
        """Check that *y* contains non-negative integer-valued data."""
        if not np.issubdtype(y.dtype, np.number):
            msg = "NegativeBinomialFamily requires numeric Y values."
            raise ValueError(msg)
        if np.any(np.isnan(y)):
            msg = "NegativeBinomialFamily does not accept NaN values in Y."
            raise ValueError(msg)
        if np.any(y < 0):
            msg = "NegativeBinomialFamily requires non-negative Y values."
            raise ValueError(msg)
        if not np.allclose(y, np.round(y)):
            msg = (
                "NegativeBinomialFamily requires integer-valued Y. "
                "Got non-integer values."
            )
            raise ValueError(msg)

    # ---- Single-model operations -----------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> Any:
        """Fit a negative binomial GLM with fixed α.

        Requires :meth:`calibrate` to have been called first (or α
        supplied at construction).  Raises ``RuntimeError`` if α is
        not yet resolved.
        """
        alpha = self._require_alpha("fit")
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model = sm.GLM(y, X_sm, family=self._nb_family(alpha)).fit(disp=0)
        return model

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Return predicted mean μ̂ on the response scale."""
        return np.asarray(model.fittedvalues)

    def coefs(self, model: Any) -> np.ndarray:
        """Extract slope coefficients (intercept excluded)."""
        params = np.asarray(model.params)
        if model.k_constant:
            return params[1:]
        return params

    def residuals(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Response-scale residuals: ``e = y − μ̂``.

        Same rationale as PoissonFamily — response-scale residuals are
        compatible with ``reconstruct_y`` which operates on the
        response scale.
        """
        return np.asarray(y - model.fittedvalues)

    # ---- Permutation helpers ---------------------------------------

    def reconstruct_y(
        self,
        predictions: np.ndarray,
        permuted_residuals: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Negative-binomial-sampled reconstruction from permuted residuals.

        Steps:
          1. ``μ* = clip(μ̂ + π(e), 1e-10, 1e8)``
          2. Convert to NB parameters: ``n = 1/α``, ``p = 1/(1 + α·μ*)``
          3. ``Y* ~ NegBin(n, p)``

        The NB2 parameterisation implies:
          ``E[Y*] = μ*`` and ``Var(Y*) = μ* + α·μ*²``

        which preserves the overdispersion structure of the observed
        data in the permuted samples.
        """
        alpha = self._require_alpha("reconstruct_y")
        mu_star = predictions + permuted_residuals
        mu_star = np.clip(mu_star, 1e-10, 1e8)
        # NB2 parameterisation: n = 1/α, p = 1/(1 + α·μ)
        n_param = 1.0 / alpha
        p_param = 1.0 / (1.0 + alpha * mu_star)
        return np.asarray(rng.negative_binomial(n=n_param, p=p_param))

    def fit_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Negative binomial deviance.

        D = 2 Σ[y·log(y/μ̂) − (y + 1/α)·log((1 + α·y)/(1 + α·μ̂))]

        Uses mask-and-index to avoid log(0) warnings, following the
        same pattern as PoissonFamily.
        """
        alpha = self._require_alpha("fit_metric")
        mu = np.maximum(y_pred, 1e-10)
        inv_a = 1.0 / alpha

        # Term 1: y · log(y / μ̂), with 0·log(0/·) = 0 by convention.
        pos = y_true > 0
        term1 = np.zeros_like(y_true, dtype=float)
        term1[pos] = y_true[pos] * np.log(y_true[pos] / mu[pos])

        # Term 2: (y + 1/α) · log((1 + α·y) / (1 + α·μ̂))
        term2 = (y_true + inv_a) * np.log((1.0 + alpha * y_true) / (1.0 + alpha * mu))

        return float(2.0 * np.sum(term1 - term2))

    # ---- Diagnostics & classical inference -------------------------

    def diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> dict[str, Any]:
        """NB GLM diagnostics (deviance, Pearson χ², dispersion, α, AIC, BIC)."""
        alpha = self._require_alpha("diagnostics")
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sm_model = sm.GLM(y, X_sm, family=self._nb_family(alpha)).fit(disp=0)
        pearson_chi2 = float(sm_model.pearson_chi2)
        df_resid = float(sm_model.df_resid)
        dispersion = pearson_chi2 / df_resid if df_resid > 0 else float("nan")
        return {
            "n_observations": len(y),
            "n_features": X.shape[1],
            "deviance": np.round(float(sm_model.deviance), 4),
            "pearson_chi2": np.round(pearson_chi2, 4),
            "dispersion": np.round(dispersion, 4),
            "alpha": np.round(alpha, 4),
            "log_likelihood": np.round(float(sm_model.llf), 4),
            "aic": np.round(float(sm_model.aic), 4),
            "bic": np.round(float(sm_model.bic_llf), 4),
        }

    def classical_p_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Asymptotic Wald z-test p-values via statsmodels NB GLM.

        Returns one p-value per slope coefficient (intercept excluded).
        """
        alpha = self._require_alpha("classical_p_values")
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sm_model = sm.GLM(y, X_sm, family=self._nb_family(alpha)).fit(disp=0)
        pvals = sm_model.pvalues[1:] if fit_intercept else sm_model.pvalues
        return np.asarray(pvals)

    # ---- Exchangeability (v0.4.0 forward-compat) -------------------

    def exchangeability_cells(
        self,
        X: np.ndarray,  # noqa: ARG002
        y: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray | None:
        """NB models assume globally exchangeable residuals."""
        return None

    # ---- Batch fitting (hot loop) ----------------------------------

    def batch_fit(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB GLM via joblib-parallelised statsmodels loop.

        Each row of *Y_matrix* is a permuted response vector.  The
        dispersion α is held fixed at the value estimated from the
        observed data.  Returns ``(B, p)`` coefficient matrix with
        intercept excluded.
        """
        alpha = self._require_alpha("batch_fit")

        kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)

        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        n_params = X.shape[1]
        B = Y_matrix.shape[0]
        nb_family = self._nb_family(alpha)

        def _fit_one(y_b: np.ndarray) -> np.ndarray:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y_b, X_sm, family=nb_family).fit(disp=0, maxiter=100)
                params = np.asarray(model.params)
                return params[1:] if fit_intercept else params
            except Exception:  # noqa: BLE001
                return np.full(n_params, np.nan)

        if n_jobs == 1:
            results = [_fit_one(Y_matrix[i]) for i in range(B)]
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(Y_matrix[i]) for i in range(B)
            )

        coef_matrix = np.array(results)

        n_failed = int(np.sum(np.any(np.isnan(coef_matrix), axis=1)))
        if n_failed > 0:
            pct = 100.0 * n_failed / B
            warnings.warn(
                f"{n_failed} of {B} NB fits did not converge "
                f"({pct:.1f}%). Consider checking for zero-inflation "
                f"or model mis-specification.",
                UserWarning,
                stacklevel=2,
            )

        return coef_matrix

    def batch_fit_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB with per-permutation design matrices.

        Used by the Kennedy individual path where column *j* of *X*
        differs across permutations while Y stays the same.
        """
        alpha = self._require_alpha("batch_fit_varying_X")

        kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)

        B, _n, p = X_batch.shape
        n_params = p
        nb_family = self._nb_family(alpha)

        def _fit_one(X_b: np.ndarray) -> np.ndarray:
            X_sm = sm.add_constant(X_b) if fit_intercept else np.asarray(X_b)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y, X_sm, family=nb_family).fit(disp=0, maxiter=100)
                params = np.asarray(model.params)
                return params[1:] if fit_intercept else params
            except Exception:  # noqa: BLE001
                return np.full(n_params, np.nan)

        if n_jobs == 1:
            results = [_fit_one(X_batch[i]) for i in range(B)]
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(X_batch[i]) for i in range(B)
            )

        coef_matrix = np.array(results)

        n_failed = int(np.sum(np.any(np.isnan(coef_matrix), axis=1)))
        if n_failed > 0:
            pct = 100.0 * n_failed / B
            warnings.warn(
                f"{n_failed} of {B} NB fits did not converge "
                f"({pct:.1f}%). Consider checking for zero-inflation "
                f"or model mis-specification.",
                UserWarning,
                stacklevel=2,
            )

        return coef_matrix


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
register_family("logistic", LogisticFamily)
register_family("poisson", PoissonFamily)
register_family("negative_binomial", NegativeBinomialFamily)

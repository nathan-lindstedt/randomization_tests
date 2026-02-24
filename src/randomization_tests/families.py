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
    HessianInversionWarning,
    PerfectSeparationWarning,
)

# ------------------------------------------------------------------ #
# Display helpers
# ------------------------------------------------------------------ #


def _fmt_p(p: float | None) -> str:
    """Format a p-value for display: scientific notation if tiny, 4 dp otherwise."""
    if p is None:
        return "N/A"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


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

    @property
    def stat_label(self) -> str:
        """Symbol for the per-coefficient test statistic.

        Displayed in result table column headers to identify the
        statistic reported for each predictor.
        Examples: ``"t"`` (linear), ``"z"`` (logistic, Poisson,
        negative binomial, ordinal), ``"χ²"`` (multinomial).
        """
        ...

    # ---- Display ---------------------------------------------------

    def display_header(
        self,
        diagnostics: dict[str, Any],
    ) -> list[tuple[str, str, str, str]]:
        """Return structured row descriptors for the results table header.

        Each 4-tuple is ``(left_label, left_value, right_label,
        right_value)``.  ``display.py`` owns all column-width and
        alignment logic; families own content and value formatting.
        Use empty strings for absent cells.

        Args:
            diagnostics: The ``diagnostics`` dict from the result object.

        Returns:
            A list of 4-tuples, one per header row.
        """
        ...

    def display_diagnostics(
        self,
        diagnostics: dict[str, Any],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Return model-level diagnostic lines and interpretive notes.

        The first element is a list of ``(label, formatted_value)``
        pairs rendered in the "Model-level Diagnostics" section.  The
        second element is a list of plain-text warning strings appended
        to the "Notes" section when a diagnostic flags a concern.

        Bundling notes with lines keeps ``display.py`` fully
        family-agnostic — each family owns both its diagnostic values
        and their interpretive warnings.

        Args:
            diagnostics: The ``extended_diagnostics`` dict from the
                result object.

        Returns:
            ``(lines, notes)`` — diagnostic lines and warning strings.
        """
        ...

    def compute_extended_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
    ) -> dict[str, Any]:
        """Compute family-specific model-level diagnostics.

        Returns a dict whose keys match exactly what
        ``display_diagnostics()`` reads — the contract is locked in
        by Step 4.  Each family produces its own diagnostic key:

        =============================  =========================
        Family                         Dict key produced
        =============================  =========================
        ``LinearFamily``               ``"breusch_pagan"``
        ``LogisticFamily``             ``"deviance_residuals"``
        ``PoissonFamily``              ``"poisson_gof"``
        ``NegativeBinomialFamily``     ``"nb_gof"``
        ``OrdinalFamily``              ``"ordinal_gof"``
        ``MultinomialFamily``          ``"multinomial_gof"``
        =============================  =========================

        Implementations wrap their calculations in ``try/except`` so
        that degenerate data (perfect separation, rank-deficient
        designs) returns NaN-filled sentinel dicts rather than
        crashing the results pipeline.

        Args:
            X: Design matrix of shape ``(n, p)`` — no intercept column.
            y: Response vector of shape ``(n,)``.
            fit_intercept: Whether the model includes an intercept.

        Returns:
            Dictionary with a single family-specific diagnostic key.
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

    # ---- Unified scoring interface ---------------------------------
    #
    # The joint permutation test (Kennedy 1995) needs to compare the
    # fit of a full model (all features) against a reduced model
    # (confounders only).  The test statistic is:
    #
    #   Δ = S(reduced) − S(full)
    #
    # where S(·) is a "score" — a goodness-of-fit metric where
    # **lower values indicate better fit**.  When the full model
    # fits better than the reduced model, Δ > 0.
    #
    # Before this interface existed, scoring was split across two
    # incompatible code paths:
    #
    # 1. **Prediction-based families** (linear, logistic, Poisson, NB)
    #    computed S via ``fit_metric(y, predict(model, X))`` — the
    #    metric only needed predicted values, not the model object.
    #
    # 2. **Model-object families** (ordinal, multinomial) stored the
    #    log-likelihood on the fitted model and used
    #    ``-2 * model.llf`` as the score — but this was accessed
    #    through duck-typed ``model_fit_metric()`` methods that
    #    strategies detected via ``hasattr()``.
    #
    # ``score()`` and ``null_score()`` unify both paths behind a
    # single protocol interface.  Every family implements both.
    # Strategy code never needs to know which path is taken.
    #
    # Convention: "lower is better" for all families:
    #   - Linear:  RSS = Σ(yᵢ − ŷᵢ)² — decreases with better fit.
    #   - GLMs:    Deviance = −2·ℓ(model) — decreases as the
    #              log-likelihood ℓ increases with better fit.
    #   - Ordinal/Multinomial: −2·ℓ(model) — same convention.
    #
    # The null-model score is needed as the baseline for the joint
    # test when there are no confounders (Z has zero columns).  In
    # that case, the reduced model IS the null model, and we need
    # S(null) without fitting anything.

    def score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Goodness-of-fit score from a fitted model object.

        Computes the family-specific metric that the joint test uses
        to measure model fit.  The convention is **lower is better**:

        * **Prediction-based families** (linear, logistic, Poisson,
          negative binomial): delegates to
          ``self.fit_metric(y, self.predict(model, X))``.  For linear
          this is RSS = Σ(yᵢ − ŷᵢ)²; for GLMs this is the deviance
          D = 2·Σ[ℓᵢ(saturated) − ℓᵢ(model)].

        * **Model-object families** (ordinal, multinomial): extracts
          the log-likelihood from the fitted statsmodels object and
          returns ``−2 · ℓ(model)``.  This is necessary because these
          models produce category-probability predictions rather than
          scalar predictions, so ``fit_metric(y, predict(...))`` is
          not meaningful.

        The joint test statistic is ``Δ = score(reduced) − score(full)``.
        When features improve fit, the full model has a lower score,
        so Δ > 0.

        Args:
            model: A fitted model object returned by ``fit``.
            X: Design matrix of shape ``(n, p)``.
            y: Observed response vector of shape ``(n,)``.

        Returns:
            Scalar fit metric (lower is better).
        """
        ...

    def null_score(self, y: np.ndarray, fit_intercept: bool = True) -> float:
        """Null-model (intercept-only) baseline score.

        Computes the score of a model with **no predictor variables**
        — only an intercept (if ``fit_intercept=True``).  This is the
        baseline for the joint test when there are no confounders:

            Δ = null_score(y) − score(full_model, X, y)

        Implementation varies by family type:

        * **Prediction-based families**: the intercept-only MLE
          predicts ``ȳ`` (the sample mean) for every observation.
          The null score is therefore ``fit_metric(y, [ȳ, ȳ, …, ȳ])``.
          For linear: RSS_null = Σ(yᵢ − ȳ)² = (n−1)·Var(y).
          For logistic: Deviance_null = 2·Σ[−yᵢ·log(p̄) − (1−yᵢ)·log(1−p̄)]
          where p̄ = P(Y=1) = ȳ.

        * **Ordinal**: analytical formula from empirical proportions.
          See ``OrdinalFamily.null_score()`` for derivation.

        * **Multinomial**: fits an intercept-only ``MNLogit`` model.
          See ``MultinomialFamily.null_score()`` for rationale.

        When ``fit_intercept=False``, predictions are all zeros —
        this is a degenerate model used only in edge-case testing.

        Args:
            y: Observed response vector of shape ``(n,)``.
            fit_intercept: Whether the model includes an intercept.

        Returns:
            Scalar null-model metric (lower is better).
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

    @property
    def stat_label(self) -> str:
        return "t"

    # ---- Display ---------------------------------------------------

    def display_header(
        self,
        diagnostics: dict[str, Any],
    ) -> list[tuple[str, str, str, str]]:
        """Return header rows for the linear results table."""
        f_p = diagnostics.get("f_p_value")
        f_p_str = f"{f_p:.4e}" if f_p is not None else "N/A"
        return [
            (
                "R-squared:",
                str(diagnostics.get("r_squared", "N/A")),
                "BIC:",
                str(diagnostics.get("bic", "N/A")),
            ),
            (
                "Adj. R-squared:",
                str(diagnostics.get("r_squared_adj", "N/A")),
                "F-statistic:",
                str(diagnostics.get("f_statistic", "N/A")),
            ),
            ("", "", "Prob (F-stat):", f_p_str),
        ]

    def display_diagnostics(
        self,
        diagnostics: dict[str, Any],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Return diagnostic lines and notes for the linear family."""
        lines: list[tuple[str, str]] = []
        notes: list[str] = []
        bp = diagnostics.get("breusch_pagan", {})
        if bp:
            lines.append(
                (
                    "Breusch-Pagan LM:",
                    f"{bp.get('lm_stat', 'N/A'):>10}   "
                    f"p = {_fmt_p(bp.get('lm_p_value'))}",
                )
            )
            lines.append(
                (
                    "Breusch-Pagan F:",
                    f"{bp.get('f_stat', 'N/A'):>10}   "
                    f"p = {_fmt_p(bp.get('f_p_value'))}",
                )
            )
            bp_p = bp.get("lm_p_value")
            if bp_p is not None and bp_p < 0.05:
                notes.append(
                    f"Breusch-Pagan p = {bp_p:.4f}: "
                    f"heteroscedastic residuals detected; "
                    f"exchangeability assumption may be violated."
                )
        return lines, notes

    def compute_extended_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Compute Breusch-Pagan heteroscedasticity test."""
        from .diagnostics import compute_breusch_pagan

        try:
            return {"breusch_pagan": compute_breusch_pagan(X, y)}
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug(
                "Breusch-Pagan diagnostics failed: %s", exc
            )
            return {
                "breusch_pagan": {
                    "lm_stat": float("nan"),
                    "lm_p_value": float("nan"),
                    "f_stat": float("nan"),
                    "f_p_value": float("nan"),
                    "warning": f"Diagnostics unavailable: {exc}",
                }
            }

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

    # ---- Scoring (joint test interface) ----------------------------
    #
    # For OLS the natural goodness-of-fit measure is RSS:
    #
    #   RSS = Σᵢ (yᵢ − ŷᵢ)²
    #
    # The joint test statistic is Δ = RSS_reduced − RSS_full.
    # Adding informative features reduces RSS, so Δ > 0 when the
    # features collectively improve the fit.  The F-statistic is a
    # scaled version of this same quantity:
    #
    #   F = [(RSS_reduced − RSS_full) / q] / [RSS_full / (n − p − 1)]
    #
    # where q is the number of features being tested and p is the
    # total number of predictors including confounders.  The
    # permutation test replaces the F-distribution assumption with
    # an empirical reference distribution of Δ values.
    #
    # The null-model (intercept-only) predicts ȳ for every
    # observation — this is the OLS MLE when the design matrix
    # contains only an intercept column.  Its RSS equals the
    # total sum of squares: RSS_null = Σ(yᵢ − ȳ)² = (n−1)·Var(y).

    def score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """RSS from a fitted linear model.

        Delegates to ``fit_metric(y, predict(model, X))`` which
        computes RSS = Σ(yᵢ − ŷᵢ)² = MSE × n.
        """
        return self.fit_metric(y, self.predict(model, X))

    def null_score(self, y: np.ndarray, fit_intercept: bool = True) -> float:
        """RSS of the intercept-only (mean) model.

        The OLS intercept-only model predicts ŷᵢ = ȳ for all i.
        RSS_null = Σ(yᵢ − ȳ)² is the total sum of squares (TSS),
        which equals (n − 1) · Var(y).  This is the maximum RSS
        achievable by any model with an intercept — adding any
        predictor can only reduce RSS (OLS is a projection).

        When ``fit_intercept=False``, predicts zero for all
        observations — a degenerate baseline for edge-case testing.
        """
        n = len(y)
        if fit_intercept:
            preds = np.full(n, np.mean(y), dtype=float)
        else:
            preds = np.zeros(n, dtype=float)
        return self.fit_metric(y, preds)

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
        with warnings.catch_warnings():
            # Near-singular X'X can trigger floating-point warnings;
            # suppress because the user relies on permutation p-values.
            warnings.filterwarnings("ignore", category=RuntimeWarning)
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

    @property
    def stat_label(self) -> str:
        return "z"

    # ---- Display ---------------------------------------------------

    def display_header(
        self,
        diagnostics: dict[str, Any],
    ) -> list[tuple[str, str, str, str]]:
        """Return header rows for the logistic results table."""
        llr_p = diagnostics.get("llr_p_value")
        llr_p_str = f"{llr_p:.4e}" if llr_p is not None else "N/A"
        return [
            (
                "Pseudo R-sq:",
                str(diagnostics.get("pseudo_r_squared", "N/A")),
                "BIC:",
                str(diagnostics.get("bic", "N/A")),
            ),
            (
                "Log-Likelihood:",
                str(diagnostics.get("log_likelihood", "N/A")),
                "LL-Null:",
                str(diagnostics.get("log_likelihood_null", "N/A")),
            ),
            ("", "", "LLR p-value:", llr_p_str),
        ]

    def display_diagnostics(
        self,
        diagnostics: dict[str, Any],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Return diagnostic lines and notes for the logistic family."""
        lines: list[tuple[str, str]] = []
        notes: list[str] = []
        dr = diagnostics.get("deviance_residuals", {})
        if dr:
            lines.append(("Deviance resid. mean:", f"{dr.get('mean', 'N/A'):>10}"))
            lines.append(("Deviance resid. var:", f"{dr.get('variance', 'N/A'):>10}"))
            lines.append(("|d_i| > 2 count:", f"{dr.get('n_extreme', 'N/A'):>10}"))
            lines.append(
                (
                    "Runs test Z:",
                    f"{dr.get('runs_test_z', 'N/A'):>10}   "
                    f"p = {_fmt_p(dr.get('runs_test_p'))}",
                )
            )
            n_extreme = dr.get("n_extreme", 0)
            if isinstance(n_extreme, (int, float)) and n_extreme > 0:
                notes.append(f"{int(n_extreme)} obs. with |deviance residual| > 2.")
            runs_p = dr.get("runs_test_p")
            if runs_p is not None and runs_p < 0.05:
                notes.append(
                    "Runs test p < 0.05: non-random residual pattern detected."
                )
        return lines, notes

    def compute_extended_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Compute deviance residual diagnostics for logistic models."""
        from .diagnostics import compute_deviance_residual_diagnostics

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
                return {
                    "deviance_residuals": compute_deviance_residual_diagnostics(X, y)
                }
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug(
                "Deviance residual diagnostics failed: %s", exc
            )
            return {
                "deviance_residuals": {
                    "mean": float("nan"),
                    "variance": float("nan"),
                    "n_extreme": 0,
                    "runs_test_z": float("nan"),
                    "runs_test_p": float("nan"),
                    "warning": f"Diagnostics unavailable: {exc}",
                }
            }

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

    # ---- Scoring (joint test interface) ----------------------------
    #
    # For logistic regression, the goodness-of-fit measure is the
    # deviance — twice the negative log-likelihood:
    #
    #   D = −2·ℓ(model) = 2·Σᵢ [−yᵢ·log(p̂ᵢ) − (1−yᵢ)·log(1−p̂ᵢ)]
    #
    # where p̂ᵢ = P(Yᵢ = 1 | Xᵢ) is the predicted probability.
    # Deviance is the logistic analogue of RSS — the MLE minimises
    # it (equivalently, maximises the log-likelihood).  The joint
    # test statistic Δ = D_reduced − D_full is the likelihood-ratio
    # statistic, which under classical theory follows a χ² distribution
    # with q degrees of freedom.  The permutation test replaces this
    # distributional assumption with an empirical reference.
    #
    # The null-model predicts p̂ᵢ = ȳ = P(Y=1) for every observation
    # — this is the MLE of the intercept-only logistic model.
    # (The logistic intercept β₀ satisfies P = 1/(1+exp(−β₀)) = ȳ,
    # i.e. β₀ = log(ȳ/(1−ȳ)), which gives predicted probability ȳ.)
    #
    # Predictions are clipped to [0.001, 0.999] before computing
    # deviance to avoid log(0) singularities.

    def score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Deviance from a fitted logistic model.

        Delegates to ``fit_metric(y, predict(model, X))`` which
        computes D = 2·Σ[−yᵢ·log(p̂ᵢ) − (1−yᵢ)·log(1−p̂ᵢ)].
        """
        return self.fit_metric(y, self.predict(model, X))

    def null_score(self, y: np.ndarray, fit_intercept: bool = True) -> float:
        """Deviance of the intercept-only (base-rate) logistic model.

        The intercept-only MLE predicts p̂ᵢ = ȳ = P(Y = 1) for
        all observations.  The null deviance is:

            D_null = 2·Σ[−yᵢ·log(ȳ) − (1−yᵢ)·log(1−ȳ)]
                   = −2·[n₁·log(ȳ) + n₀·log(1−ȳ)]

        where n₁ = Σyᵢ and n₀ = n − n₁.  This is the maximum
        deviance for any logistic model with an intercept.

        When ``fit_intercept=False``, predicts p̂ = 0.5 (logistic
        sigmoid of zero) — but we pass 0.0 through ``fit_metric``
        which clips to 0.001, giving a conservative baseline.
        """
        n = len(y)
        if fit_intercept:
            preds = np.full(n, np.mean(y), dtype=float)
        else:
            preds = np.zeros(n, dtype=float)
        return self.fit_metric(y, preds)

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
        with warnings.catch_warnings():
            # Quasi-complete separation can trigger convergence and
            # separation warnings; suppress because the user relies
            # on permutation p-values, not asymptotic diagnostics.
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # disp=0 suppresses the iteration log that statsmodels
            # prints by default for iterative MLE solvers.
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
# **Batch fitting** delegates to the active backend via
# ``resolve_backend()``.  The JAX backend uses vmap'd
# Newton–Raphson with log-link warm start; the NumPy backend
# falls back to a joblib-parallelised statsmodels IRLS loop.
# Convergence failures produce NaN coefficient rows; an
# aggregated warning is emitted after the loop.


@dataclass(frozen=True)
class PoissonFamily:
    """Poisson regression family for count outcomes.

    Implements the ``ModelFamily`` protocol for non-negative integer
    outcomes using the Poisson log-link GLM.  Residuals are response-
    scale (y − μ̂), reconstruction uses Poisson sampling on the
    response scale, and the joint-test metric is deviance.

    The class is stateless — all data flows through method arguments.
    ``batch_fit`` delegates to the active backend (JAX or NumPy)
    via ``resolve_backend()``.
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

    @property
    def stat_label(self) -> str:
        return "z"

    # ---- Display ---------------------------------------------------

    def display_header(
        self,
        diagnostics: dict[str, Any],
    ) -> list[tuple[str, str, str, str]]:
        """Return header rows for the Poisson results table."""
        pearson = diagnostics.get("pearson_chi2")
        pearson_str = f"{pearson:.4f}" if pearson is not None else "N/A"
        return [
            (
                "Deviance:",
                str(diagnostics.get("deviance", "N/A")),
                "BIC:",
                str(diagnostics.get("bic", "N/A")),
            ),
            (
                "Log-Likelihood:",
                str(diagnostics.get("log_likelihood", "N/A")),
                "Dispersion:",
                str(diagnostics.get("dispersion", "N/A")),
            ),
            (
                "",
                "",
                "\u03c7\u00b2 (Pearson):",
                pearson_str,
            ),
        ]

    def display_diagnostics(
        self,
        diagnostics: dict[str, Any],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Return diagnostic lines and notes for the Poisson family."""
        lines: list[tuple[str, str]] = []
        notes: list[str] = []
        gof = diagnostics.get("poisson_gof", {})
        if gof:
            lines.append(
                (
                    "Pearson \u03c7\u00b2:",
                    f"{gof.get('pearson_chi2', 'N/A'):>10}",
                )
            )
            lines.append(("Deviance:", f"{gof.get('deviance', 'N/A'):>10}"))
            disp = gof.get("dispersion", None)
            disp_str = f"{disp:.4f}" if disp is not None else "N/A"
            lines.append(("Dispersion:", f"{disp_str:>10}"))
            if gof.get("overdispersed", False):
                notes.append(
                    f"Dispersion = {disp_str}: overdispersion "
                    f"detected (> 1.5). Consider using "
                    f"family='negative_binomial'."
                )
        return lines, notes

    def compute_extended_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
    ) -> dict[str, Any]:
        """Compute Poisson goodness-of-fit diagnostics."""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
                pois_model = sm.GLM(y, X_sm, family=sm.families.Poisson()).fit(disp=0)
                pearson_chi2 = float(pois_model.pearson_chi2)
                df_resid = float(pois_model.df_resid)
                dispersion = pearson_chi2 / df_resid if df_resid > 0 else float("nan")
                return {
                    "poisson_gof": {
                        "deviance": float(pois_model.deviance),
                        "pearson_chi2": pearson_chi2,
                        "dispersion": dispersion,
                        "overdispersed": dispersion > 1.5,
                    }
                }
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug("Poisson GoF diagnostics failed: %s", exc)
            return {
                "poisson_gof": {
                    "deviance": float("nan"),
                    "pearson_chi2": float("nan"),
                    "dispersion": float("nan"),
                    "overdispersed": False,
                    "warning": f"Diagnostics unavailable: {exc}",
                }
            }

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
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
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
        mu = np.maximum(y_pred, 1e-10)  # μ̂, shape (n,) — clamp to avoid log(0)
        # 0 * log(0/μ̂) = 0 by convention (L'Hôpital).
        # Use mask-and-index to avoid evaluating log(0) (np.where
        # evaluates both branches eagerly, producing RuntimeWarnings).
        pos = y_true > 0  # boolean mask for non-zero observations
        contrib = np.zeros_like(
            y_true, dtype=float
        )  # y·log(y/μ̂) accumulator, shape (n,)
        contrib[pos] = y_true[pos] * np.log(
            y_true[pos] / mu[pos]
        )  # non-zero terms only
        deviance_i = contrib - (
            y_true - mu
        )  # per-obs deviance contribution, shape (n,)
        return float(2.0 * np.sum(deviance_i))  # D = 2·Σ deviance_i, scalar

    # ---- Scoring (joint test interface) ----------------------------
    #
    # For Poisson regression, fit is measured by the deviance:
    #
    #   D = 2·Σᵢ [yᵢ·log(yᵢ/μ̂ᵢ) − (yᵢ − μ̂ᵢ)]
    #
    # where μ̂ᵢ = exp(Xᵢ·β̂) is the predicted count (mean of the
    # Poisson distribution for observation i).  The deviance equals
    # 2·[ℓ(saturated) − ℓ(model)], where the saturated model sets
    # μ̂ᵢ = yᵢ exactly.  Lower deviance means better fit.
    #
    # The null-model predicts μ̂ᵢ = ȳ (the sample mean count) for
    # every observation.  This is the Poisson MLE with intercept
    # only: the intercept is β₀ = log(ȳ), giving
    # μ̂ = exp(β₀) = ȳ.
    #
    # The joint test statistic Δ = D_reduced − D_full is analogous
    # to the likelihood-ratio test: large Δ means the tested
    # features collectively reduce the deviance.

    def score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Deviance from a fitted Poisson model.

        Delegates to ``fit_metric(y, predict(model, X))`` which
        computes D = 2·Σ[yᵢ·log(yᵢ/μ̂ᵢ) − (yᵢ − μ̂ᵢ)].
        """
        return self.fit_metric(y, self.predict(model, X))

    def null_score(self, y: np.ndarray, fit_intercept: bool = True) -> float:
        """Deviance of the intercept-only Poisson model.

        The Poisson intercept-only MLE is β₀ = log(ȳ), predicting
        μ̂ᵢ = ȳ for all observations.  The null deviance is:

            D_null = 2·Σ[yᵢ·log(yᵢ/ȳ) − (yᵢ − ȳ)]

        The second term vanishes (Σ(yᵢ − ȳ) = 0), leaving
        D_null = 2·Σ yᵢ·log(yᵢ/ȳ) for observations where yᵢ > 0
        (0·log(0/·) = 0 by convention).
        """
        n = len(y)
        if fit_intercept:
            preds = np.full(n, np.mean(y), dtype=float)
        else:
            preds = np.zeros(n, dtype=float)
        return self.fit_metric(y, preds)

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
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sm_model = sm.GLM(y, X_sm, family=sm.families.Poisson()).fit(disp=0)
        pearson_chi2 = float(sm_model.pearson_chi2)  # Σ (yᵢ − μ̂ᵢ)² / μ̂ᵢ
        df_resid = float(sm_model.df_resid)  # n − p − 1
        dispersion = (
            pearson_chi2 / df_resid if df_resid > 0 else float("nan")
        )  # > 1 signals overdispersion
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
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
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
    # Delegates to the active backend (JAX or NumPy) via
    # ``resolve_backend()``.  The JAX backend uses vmap'd
    # Newton–Raphson; the NumPy backend falls back to a
    # joblib-parallelised statsmodels IRLS loop.

    def batch_fit(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson GLM via the active backend.

        Delegates to ``backend.batch_poisson()`` resolved from the
        current configuration.  Forwards ``n_jobs`` only to the
        NumPy backend.
        """
        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()
        if backend.name == "numpy":
            return np.asarray(
                backend.batch_poisson(
                    X,
                    Y_matrix,
                    fit_intercept=fit_intercept,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            )
        return np.asarray(
            backend.batch_poisson(
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
        """Batch Poisson with per-permutation design matrices.

        Delegates to ``backend.batch_poisson_varying_X()`` for the
        Kennedy individual path.  Forwards ``n_jobs`` only to the
        NumPy backend.
        """
        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()
        if backend.name == "numpy":
            return np.asarray(
                backend.batch_poisson_varying_X(
                    X_batch,
                    y,
                    fit_intercept=fit_intercept,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            )
        return np.asarray(
            backend.batch_poisson_varying_X(
                X_batch,
                y,
                fit_intercept=fit_intercept,
                **kwargs,
            )
        )


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
# **Batch fitting** delegates to the active backend via
# ``resolve_backend()``.  The JAX backend uses vmap'd
# Newton–Raphson with fixed α and log-link warm start; the NumPy
# backend falls back to a joblib-parallelised statsmodels loop
# with ``sm.families.NegativeBinomial`` and fixed α.


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

    @property
    def stat_label(self) -> str:
        return "z"

    # ---- Display ---------------------------------------------------

    def display_header(
        self,
        diagnostics: dict[str, Any],
    ) -> list[tuple[str, str, str, str]]:
        """Return header rows for the negative binomial results table."""
        pearson = diagnostics.get("pearson_chi2")
        pearson_str = f"{pearson:.4f}" if pearson is not None else "N/A"
        # Show Alpha (NB) on the left if available, else empty cell.
        alpha_disp = diagnostics.get("alpha")
        left_label = "Alpha (NB):" if alpha_disp is not None else ""
        left_value = str(alpha_disp) if alpha_disp is not None else ""
        return [
            (
                "Deviance:",
                str(diagnostics.get("deviance", "N/A")),
                "BIC:",
                str(diagnostics.get("bic", "N/A")),
            ),
            (
                "Log-Likelihood:",
                str(diagnostics.get("log_likelihood", "N/A")),
                "Dispersion:",
                str(diagnostics.get("dispersion", "N/A")),
            ),
            (
                left_label,
                left_value,
                "\u03c7\u00b2 (Pearson):",
                pearson_str,
            ),
        ]

    def display_diagnostics(
        self,
        diagnostics: dict[str, Any],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Return diagnostic lines and notes for the negative binomial family."""
        lines: list[tuple[str, str]] = []
        notes: list[str] = []
        gof = diagnostics.get("nb_gof", {})
        if gof:
            lines.append(
                (
                    "Pearson \u03c7\u00b2:",
                    f"{gof.get('pearson_chi2', 'N/A'):>10}",
                )
            )
            lines.append(("Deviance:", f"{gof.get('deviance', 'N/A'):>10}"))
            disp = gof.get("dispersion", None)
            disp_str = f"{disp:.4f}" if disp is not None else "N/A"
            lines.append(("Dispersion:", f"{disp_str:>10}"))
            alpha_val = gof.get("alpha", None)
            alpha_str = f"{alpha_val:.4f}" if alpha_val is not None else "N/A"
            lines.append(("\u03b1 (NB Dispersion):", f"{alpha_str:>10}"))
            if gof.get("overdispersed", False):
                notes.append(
                    f"Dispersion = {disp_str}: residual "
                    f"overdispersion detected after NB fit."
                )
        return lines, notes

    def compute_extended_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
    ) -> dict[str, Any]:
        """Compute negative binomial goodness-of-fit diagnostics."""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
                # Estimate α via MLE, then fit GLM with fixed α.
                nb_mle = sm.NegativeBinomial(y, X_sm).fit(disp=0, maxiter=200)
                alpha_hat = float(np.exp(nb_mle.lnalpha))
                nb_model = sm.GLM(
                    y,
                    X_sm,
                    family=sm.families.NegativeBinomial(alpha=alpha_hat),
                ).fit(disp=0)
                pearson_chi2 = float(nb_model.pearson_chi2)
                df_resid = float(nb_model.df_resid)
                dispersion = pearson_chi2 / df_resid if df_resid > 0 else float("nan")
                return {
                    "nb_gof": {
                        "deviance": float(nb_model.deviance),
                        "pearson_chi2": pearson_chi2,
                        "dispersion": dispersion,
                        "alpha": alpha_hat,
                        "overdispersed": dispersion > 1.5,
                    }
                }
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug("NB GoF diagnostics failed: %s", exc)
            return {
                "nb_gof": {
                    "deviance": float("nan"),
                    "pearson_chi2": float("nan"),
                    "dispersion": float("nan"),
                    "alpha": float("nan"),
                    "warning": f"Diagnostics unavailable: {exc}",
                }
            }

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
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
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
        mu_star = predictions + permuted_residuals  # μ* = μ̂ + π(e), shape (n,)
        mu_star = np.clip(mu_star, 1e-10, 1e8)  # clamp to valid NB mean range
        # NB2 parameterisation: n = 1/α, p = 1/(1 + α·μ)
        n_param = 1.0 / alpha  # NB size param, scalar (broadcast)
        p_param = 1.0 / (1.0 + alpha * mu_star)  # NB success prob, shape (n,)
        return np.asarray(
            rng.negative_binomial(n=n_param, p=p_param)
        )  # Y* ~ NB(n, p), shape (n,)

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
        mu = np.maximum(y_pred, 1e-10)  # μ̂, shape (n,) — clamp to avoid log(0)
        inv_a = 1.0 / alpha  # 1/α, scalar — used in both deviance terms

        # Term 1: y · log(y / μ̂), with 0·log(0/·) = 0 by convention.
        pos = y_true > 0  # boolean mask for non-zero counts
        term1 = np.zeros_like(y_true, dtype=float)  # y·log(y/μ̂) accumulator, shape (n,)
        term1[pos] = y_true[pos] * np.log(y_true[pos] / mu[pos])  # non-zero terms only

        # Term 2: (y + 1/α) · log((1 + α·y) / (1 + α·μ̂))
        term2 = (y_true + inv_a) * np.log(
            (1.0 + alpha * y_true) / (1.0 + alpha * mu)
        )  # shape (n,)

        return float(2.0 * np.sum(term1 - term2))  # D_NB = 2·Σ(term1 − term2), scalar

    # ---- Scoring (joint test interface) ----------------------------
    #
    # The negative binomial deviance generalises the Poisson deviance
    # to allow overdispersion (Var(Y) = μ + α·μ²).  It is:
    #
    #   D_NB = 2·Σᵢ [yᵢ·log(yᵢ/μ̂ᵢ)
    #               − (yᵢ + 1/α)·log((1 + α·yᵢ)/(1 + α·μ̂ᵢ))]
    #
    # where α is the overdispersion parameter estimated during
    # calibration.  When α → 0 this reduces to the Poisson deviance.
    # The score-based joint test statistic Δ = D_reduced − D_full
    # measures collective feature importance as with all other
    # families.
    #
    # The null-model predicts μ̂ᵢ = ȳ for all i, just like Poisson.
    # The NB intercept-only MLE is β₀ = log(ȳ), with the α parameter
    # held fixed at its calibrated value.

    def score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """NB deviance from a fitted negative binomial model.

        Delegates to ``fit_metric(y, predict(model, X))`` which
        computes the NB2 deviance with the calibrated α.
        """
        return self.fit_metric(y, self.predict(model, X))

    def null_score(self, y: np.ndarray, fit_intercept: bool = True) -> float:
        """NB deviance of the intercept-only (mean) model.

        Predicts μ̂ᵢ = ȳ for all observations, then evaluates the
        NB2 deviance formula with the calibrated α.  This is the
        baseline for the joint test when no confounders are present.
        """
        n = len(y)
        if fit_intercept:
            preds = np.full(n, np.mean(y), dtype=float)
        else:
            preds = np.zeros(n, dtype=float)
        return self.fit_metric(y, preds)

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
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sm_model = sm.GLM(y, X_sm, family=self._nb_family(alpha)).fit(disp=0)
        pearson_chi2 = float(sm_model.pearson_chi2)  # Σ (yᵢ − μ̂ᵢ)² / Var(μ̂ᵢ)
        df_resid = float(sm_model.df_resid)  # n − p − 1
        dispersion = (
            pearson_chi2 / df_resid if df_resid > 0 else float("nan")
        )  # > 1 signals residual overdispersion
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
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
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
    #
    # Delegates to the active backend (JAX or NumPy) via
    # ``resolve_backend()``.  The JAX backend uses vmap'd
    # Newton–Raphson with fixed α; the NumPy backend falls
    # back to a joblib-parallelised statsmodels IRLS loop.

    def batch_fit(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB GLM via the active backend.

        Delegates to ``backend.batch_negbin()`` resolved from the
        current configuration.  The dispersion α (estimated once
        from observed data) is forwarded to the backend.
        """
        alpha = self._require_alpha("batch_fit")

        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()
        if backend.name == "numpy":
            return np.asarray(
                backend.batch_negbin(
                    X,
                    Y_matrix,
                    fit_intercept=fit_intercept,
                    alpha=alpha,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            )
        return np.asarray(
            backend.batch_negbin(
                X,
                Y_matrix,
                fit_intercept=fit_intercept,
                alpha=alpha,
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
        """Batch NB with per-permutation design matrices.

        Delegates to ``backend.batch_negbin_varying_X()`` for the
        Kennedy individual path.  Forwards ``n_jobs`` only to the
        NumPy backend.
        """
        alpha = self._require_alpha("batch_fit_varying_X")

        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()
        if backend.name == "numpy":
            return np.asarray(
                backend.batch_negbin_varying_X(
                    X_batch,
                    y,
                    fit_intercept=fit_intercept,
                    alpha=alpha,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            )
        return np.asarray(
            backend.batch_negbin_varying_X(
                X_batch,
                y,
                fit_intercept=fit_intercept,
                alpha=alpha,
                **kwargs,
            )
        )


# ------------------------------------------------------------------ #
# OrdinalFamily
# ------------------------------------------------------------------ #
#
# Ordinal regression (proportional-odds logistic regression) models
# ordered categorical outcomes Y ∈ {0, 1, …, K−1} via the cumulative
# logit link:
#
#   logit(P(Y ≤ k | X)) = α_k − Xβ    for k = 0, …, K−2
#
# where α_k are threshold (cutpoint) parameters and β are shared
# slope coefficients.  The model assumes proportional odds: the
# effect of each predictor is constant across cutpoints.
#
# **Key design decisions** (from the v0.3.0 plan):
#
# 1. **direct_permutation = True** — Ordinal residuals are not well-
#    defined, so the residual→permute→reconstruct pipeline is replaced
#    by direct Y permutation (Manly 1997).  The ter Braak engine path
#    detects this flag and permutes Y rows directly.
#
# 2. **Freedman-Lane rejection** — FL requires meaningful residuals
#    for the reduced-model partial regression approach.  Since ordinal
#    residuals are ill-defined, FL raises ValueError with guidance to
#    use ter_braak or kennedy methods instead.
#
# 3. **score() / null_score()** — The joint test metric is deviance
#    (−2 × log-likelihood), computed from the fitted model object
#    via the ``score()`` protocol method.  ``null_score()`` computes
#    the thresholds-only baseline analytically from empirical
#    category proportions.
#
# 4. **fit_intercept is ignored** — OrderedModel always estimates
#    threshold parameters, which serve as category-specific intercepts.
#    The fit_intercept parameter is accepted but has no effect.
#
# 5. **method='bfgs'** — The default Newton optimizer often fails
#    to converge for ordinal models.  BFGS converges reliably.


@dataclass(frozen=True)
class OrdinalFamily:
    """Ordinal regression (proportional-odds logistic) family.

    Implements the ``ModelFamily`` protocol for ordered categorical
    outcomes with ≥ 3 levels (integer-coded 0, 1, …, K−1).

    Uses ``statsmodels.miscmodels.ordinal_model.OrderedModel`` with
    the logit link (proportional-odds assumption).

    **Permutation method restrictions**: Only ``ter_braak``,
    ``kennedy``, and ``kennedy_joint`` are supported.  Freedman-Lane
    methods raise ``ValueError`` because ordinal residuals are not
    meaningfully defined.

    The ter Braak path uses direct Y permutation (equivalent to
    Manly 1997) rather than the residual-based approach used by
    continuous/binary families.

    **Optimizer strategy**: The single-model ``fit()`` (used for
    reported coefficients and diagnostics) uses BFGS for exact
    optima.  The batch fitting methods delegate to the active
    backend via ``resolve_backend()``.  The JAX backend uses
    vmap'd Newton–Raphson with ``jax.grad``/``jax.hessian``
    autodiff; the NumPy backend falls back to statsmodels
    ``OrderedModel`` with the Powell optimizer.
    """

    @property
    def name(self) -> str:
        return "ordinal"

    @property
    def residual_type(self) -> str:
        return "none"

    @property
    def direct_permutation(self) -> bool:
        return True

    @property
    def metric_label(self) -> str:
        return "Deviance Reduction"

    @property
    def stat_label(self) -> str:
        return "z"

    # ---- Display ---------------------------------------------------

    def display_header(
        self,
        diagnostics: dict[str, Any],
    ) -> list[tuple[str, str, str, str]]:
        """Return header rows for the ordinal results table."""
        n_cats = diagnostics.get("n_categories")
        n_cats_str = str(n_cats) if n_cats is not None else "N/A"
        llr_p = diagnostics.get("llr_p_value")
        llr_p_str = f"{llr_p:.4e}" if llr_p is not None else "N/A"
        return [
            (
                "Pseudo R-sq:",
                str(diagnostics.get("pseudo_r_squared", "N/A")),
                "BIC:",
                str(diagnostics.get("bic", "N/A")),
            ),
            (
                "Log-Likelihood:",
                str(diagnostics.get("log_likelihood", "N/A")),
                "LL-Null:",
                str(diagnostics.get("log_likelihood_null", "N/A")),
            ),
            ("Categories:", n_cats_str, "LLR p-value:", llr_p_str),
        ]

    def display_diagnostics(
        self,
        diagnostics: dict[str, Any],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Return diagnostic lines and notes for the ordinal family."""
        lines: list[tuple[str, str]] = []
        notes: list[str] = []
        gof = diagnostics.get("ordinal_gof", {})
        if gof:
            pr2 = gof.get("pseudo_r_squared", None)
            pr2_str = f"{pr2:.4f}" if pr2 is not None else "N/A"
            lines.append(("Pseudo R-sq:", f"{pr2_str:>10}"))
            ll = gof.get("log_likelihood", None)
            ll_str = f"{ll:.4f}" if ll is not None else "N/A"
            lines.append(("Log-Likelihood:", f"{ll_str:>10}"))
            n_cats = gof.get("n_categories", None)
            n_cats_str = str(n_cats) if n_cats is not None else "N/A"
            lines.append(("Categories:", f"{n_cats_str:>10}"))
            # Proportional odds test
            po_chi2 = gof.get("prop_odds_chi2", None)
            po_p = gof.get("prop_odds_p", None)
            if po_chi2 is not None:
                po_chi2_str = f"{po_chi2:.4f}"
                lines.append(
                    (
                        "Prop. Odds \u03c7\u00b2:",
                        f"{po_chi2_str:>10}   p = {_fmt_p(po_p)}",
                    )
                )
                po_df = gof.get("prop_odds_df", None)
                if po_df is not None:
                    lines.append(("Prop. Odds df:", f"{po_df:>10}"))
                if po_p is not None and po_p < 0.05:
                    notes.append(
                        "Proportional odds "
                        + "\u03c7\u00b2"
                        + f" p = {po_p:.4f}: the proportional odds "
                        "assumption may be violated."
                    )
        return lines, notes

    def compute_extended_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Compute ordinal goodness-of-fit and proportional odds test."""
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        from .diagnostics import _proportional_odds_test

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                warnings.filterwarnings("ignore", category=HessianInversionWarning)
                X_arr = np.asarray(X, dtype=float)
                ord_model = OrderedModel(y, X_arr, distr="logit").fit(
                    disp=0, method="bfgs"
                )
            llf = float(ord_model.llf)
            llnull = float(ord_model.llnull)
            pseudo_r2 = 1.0 - llf / llnull if llnull != 0.0 else float("nan")
            po_result = _proportional_odds_test(X_arr, y, ord_model)
            return {
                "ordinal_gof": {
                    "pseudo_r_squared": pseudo_r2,
                    "log_likelihood": llf,
                    "log_likelihood_null": llnull,
                    "n_categories": len(np.unique(y)),
                    **po_result,
                }
            }
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug("Ordinal GoF diagnostics failed: %s", exc)
            return {
                "ordinal_gof": {
                    "pseudo_r_squared": float("nan"),
                    "log_likelihood": float("nan"),
                    "log_likelihood_null": float("nan"),
                    "n_categories": len(np.unique(y)),
                    "warning": f"Diagnostics unavailable: {exc}",
                }
            }

    # ---- Validation ------------------------------------------------

    def validate_y(self, y: np.ndarray) -> None:
        """Check that *y* contains ordered categorical integer data with ≥ 3 levels."""
        if not np.issubdtype(y.dtype, np.number):
            msg = "OrdinalFamily requires numeric Y values."
            raise ValueError(msg)
        if np.any(np.isnan(y)):
            msg = "OrdinalFamily does not accept NaN values in Y."
            raise ValueError(msg)
        if not np.allclose(y, np.round(y)):
            msg = (
                "OrdinalFamily requires integer-coded Y values "
                "(e.g. 0, 1, 2, …, K−1). Got non-integer values."
            )
            raise ValueError(msg)
        n_levels = len(np.unique(y))
        if n_levels < 3:
            msg = (
                f"OrdinalFamily requires ≥ 3 ordered categories, "
                f"got {n_levels}. For binary outcomes, use "
                f"family='logistic' instead."
            )
            raise ValueError(msg)

    # ---- Single-model operations -----------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,  # noqa: ARG002
    ) -> Any:
        """Fit a proportional-odds logistic regression via BFGS.

        BFGS provides exact optima for the coefficients and
        diagnostics displayed to the user.  Batch permutation fits
        use the faster Powell method instead — see ``batch_fit``.

        ``fit_intercept`` is accepted for protocol compatibility but
        ignored — thresholds (cutpoints) always serve as
        category-specific intercepts in ordinal models.

        Returns the fitted ``OrderedResults`` object.
        """
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
            warnings.filterwarnings("ignore", category=HessianInversionWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model = OrderedModel(y_arr, X_arr, distr="logit").fit(disp=0, method="bfgs")
        return model

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Return expected value E[Y|X] = Σ k·P(Y=k|X).

        The returned vector has shape ``(n,)`` rather than a
        probability matrix, keeping the interface consistent with
        other families (all return a 1-D prediction).
        """
        prob_matrix = np.asarray(model.predict())  # (n, K)
        levels = np.arange(prob_matrix.shape[1])  # [0, 1, …, K-1] category indices
        return np.asarray(prob_matrix @ levels)  # E[Y|X] = Σ k·P(Y=k|X), shape (n,)

    def coefs(self, model: Any) -> np.ndarray:
        """Extract slope coefficients (thresholds excluded).

        ``model.params[:p]`` are the β coefficients;
        ``model.params[p:]`` are the K−1 threshold parameters.
        """
        n_features = model.model.exog.shape[1]  # p — slope count (excl. thresholds)
        return np.asarray(model.params[:n_features])  # β̂ slope coefficients, shape (p,)

    def residuals(
        self,
        model: Any,  # noqa: ARG002
        X: np.ndarray,  # noqa: ARG002
        y: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray:
        """Not implemented — ordinal residuals are ill-defined.

        Raises:
            NotImplementedError: Always.
        """
        msg = (
            "Ordinal residuals are not well-defined. "
            "OrdinalFamily uses direct_permutation=True, which "
            "bypasses the residual pipeline entirely."
        )
        raise NotImplementedError(msg)

    # ---- Permutation helpers ---------------------------------------

    def reconstruct_y(
        self,
        predictions: np.ndarray,  # noqa: ARG002
        permuted_residuals: np.ndarray,  # noqa: ARG002
        rng: np.random.Generator,  # noqa: ARG002
    ) -> np.ndarray:
        """Not implemented — ordinal Y-reconstruction is ill-defined.

        Raises:
            NotImplementedError: Always.
        """
        msg = (
            "Ordinal Y-reconstruction from residuals is not supported. "
            "OrdinalFamily uses direct_permutation=True; the engine "
            "permutes Y directly instead of reconstructing."
        )
        raise NotImplementedError(msg)

    def fit_metric(
        self,
        y_true: np.ndarray,  # noqa: ARG002
        y_pred: np.ndarray,  # noqa: ARG002
    ) -> float:
        """Not implemented — use ``score(model, X, y)`` instead.

        Ordinal deviance requires the fitted model object (for
        log-likelihood), not just predicted values.

        Raises:
            NotImplementedError: Always.
        """
        msg = (
            "OrdinalFamily.fit_metric() is not available because ordinal "
            "deviance requires the fitted model object.  Use "
            "score(model, X, y) instead."
        )
        raise NotImplementedError(msg)

    # ---- Scoring (joint test interface) ----------------------------
    #
    # Ordinal models (proportional-odds / cumulative-link) differ
    # from prediction-based families in a fundamental way: the model
    # output is a vector of cumulative probabilities for K ordered
    # categories, not a single scalar prediction.  There is no
    # natural "prediction" that can be plugged into a residual-style
    # metric.  The natural goodness-of-fit measure is the deviance:
    #
    #   D = −2 · ℓ(model)
    #
    # where ℓ(model) is the log-likelihood of the proportional-odds
    # model.  The log-likelihood for observation i in category k is:
    #
    #   ℓᵢ = log(pᵢₖ)
    #
    # where pᵢₖ = P(Yᵢ = k | Xᵢ) is determined by the cumulative
    # link function Φ and the threshold (cutpoint) parameters αₖ:
    #
    #   P(Yᵢ ≤ k | Xᵢ) = Φ(αₖ − Xᵢ·β)
    #   pᵢₖ = P(Yᵢ ≤ k) − P(Yᵢ ≤ k−1)
    #
    # The score is extracted directly from the fitted statsmodels
    # ``OrderedModel`` object via ``model.llf``.  The ``X`` and ``y``
    # arguments are accepted for protocol compatibility but unused.
    #
    # Null model:
    # The null (no-predictor) ordinal model has only threshold
    # parameters — no β coefficients.  Its MLE sets cumulative
    # probabilities equal to the empirical cumulative proportions,
    # which gives category probabilities pₖ = nₖ/n.  The null
    # log-likelihood is therefore:
    #
    #   ℓ_null = Σₖ nₖ · log(nₖ / n)
    #
    # This is computed analytically rather than by fitting an
    # ``OrderedModel`` with zero predictors, because statsmodels'
    # ``OrderedModel`` does not support zero-column exog arrays
    # (the Hessian becomes singular, causing a LinAlgError).
    # The analytical formula is exact and has been validated against
    # the ``model.llnull`` attribute from a full ordinal fit.

    def score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:  # noqa: ARG002
        """Deviance from a fitted ordinal model: ``−2 · ℓ(model)``.

        The log-likelihood ℓ is stored on the fitted
        ``OrderedModel`` object.  Lower deviance (higher ℓ) means
        better fit.  ``X`` and ``y`` are unused — present only for
        protocol compatibility.
        """
        return -2.0 * float(model.llf)

    def null_score(self, y: np.ndarray, fit_intercept: bool = True) -> float:  # noqa: ARG002
        """Deviance of the thresholds-only (no predictors) ordinal model.

        Computed analytically from empirical category proportions
        rather than by fitting a model (see section comment above).

        The analytical formula is:

            ℓ_null = Σₖ nₖ · log(nₖ / n)
            null_score = −2 · ℓ_null

        where nₖ is the count of observations in category k and
        n = Σₖ nₖ is the total sample size.  This is the multinomial
        log-likelihood evaluated at pₖ = nₖ/n — the maximum-likelihood
        estimates for the proportional-odds model with no predictors.

        ``fit_intercept`` is accepted for protocol compatibility but
        ignored — ordinal models always include threshold parameters
        (the thresholds play the role of the intercept).

        Validated against ``model.llnull`` from statsmodels full fits.
        """
        _, counts = np.unique(y, return_counts=True)  # nₖ — per-category counts
        n = len(y)  # total observations
        # pₖ = nₖ/n — empirical category proportions (MLE).
        proportions = counts / n
        # ℓ_null = Σₖ nₖ · log(pₖ) = Σₖ nₖ · log(nₖ/n).
        ll_null = float(np.sum(counts * np.log(proportions)))
        return -2.0 * ll_null

    # ---- Diagnostics & classical inference -------------------------

    def diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> dict[str, Any]:
        """Ordinal model diagnostics.

        Returns log-likelihood, null log-likelihood, pseudo-R²,
        AIC, BIC, LLR p-value, threshold estimates, and number of
        categories.
        """
        model = self.fit(X, y, fit_intercept)
        llf = float(model.llf)  # ℓ(model) — fitted log-likelihood
        llnull = float(model.llnull)  # ℓ₀ — null (intercept-only) log-likelihood
        pseudo_r2 = (
            1.0 - llf / llnull if llnull != 0.0 else float("nan")
        )  # McFadden's R² = 1 − ℓ/ℓ₀

        # Threshold parameters (cutpoints α₀, …, α_{K-2})
        n_features = X.shape[1]  # p — number of slope predictors
        thresholds = np.asarray(model.params[n_features:])  # cutpoints, shape (K-1,)

        # Log-likelihood ratio test p-value
        try:
            llr_p = float(model.llr_pvalue)
        except (AttributeError, TypeError):
            llr_p = float("nan")

        return {
            "n_observations": len(y),
            "n_features": n_features,
            "n_categories": len(np.unique(y)),
            "pseudo_r_squared": np.round(pseudo_r2, 4),
            "log_likelihood": np.round(llf, 4),
            "log_likelihood_null": np.round(llnull, 4),
            "aic": np.round(float(model.aic), 4),
            "bic": np.round(float(model.bic), 4),
            "llr_p_value": np.round(llr_p, 6),
            "thresholds": np.round(thresholds, 4).tolist(),
        }

    def classical_p_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Asymptotic Wald z-test p-values for slope coefficients.

        Returns one p-value per slope coefficient (thresholds excluded).
        """
        model = self.fit(X, y, fit_intercept)
        n_features = X.shape[1]  # p — slope count (excl. thresholds)
        return np.asarray(model.pvalues[:n_features])  # Wald z p-values, shape (p,)

    # ---- Exchangeability (v0.4.0 forward-compat) -------------------

    def exchangeability_cells(
        self,
        X: np.ndarray,  # noqa: ARG002
        y: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray | None:
        """Ordinal models assume globally exchangeable responses."""
        return None

    # ---- Batch fitting (hot loop) ----------------------------------
    #
    # Delegates to the active backend (JAX or NumPy) via
    # ``resolve_backend()``.  The JAX backend uses vmap'd
    # Newton–Raphson with autodiff; the NumPy backend falls
    # back to a joblib-parallelised statsmodels OrderedModel loop.

    def batch_fit(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch ordinal fitting via the active backend.

        Delegates to ``backend.batch_ordinal()`` resolved from the
        current configuration.  The number of categories K is
        computed from the union of unique values across all
        permuted Y vectors and forwarded to the backend.

        ``fit_intercept`` is accepted for protocol compatibility but
        ignored — ordinal thresholds always serve as intercepts.
        """
        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()

        # Compute K from observed categories across all permutations.
        K = int(len(np.unique(Y_matrix)))

        if backend.name == "numpy":
            return np.asarray(
                backend.batch_ordinal(
                    X,
                    Y_matrix,
                    fit_intercept=fit_intercept,
                    K=K,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            )
        return np.asarray(
            backend.batch_ordinal(
                X,
                Y_matrix,
                fit_intercept=fit_intercept,
                K=K,
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
        """Batch ordinal with per-permutation design matrices.

        Delegates to ``backend.batch_ordinal_varying_X()`` for the
        Kennedy individual path.  Forwards ``n_jobs`` only to the
        NumPy backend.

        ``fit_intercept`` is accepted for protocol compatibility but
        ignored — ordinal thresholds always serve as intercepts.
        """
        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()

        K = int(len(np.unique(y)))

        if backend.name == "numpy":
            return np.asarray(
                backend.batch_ordinal_varying_X(
                    X_batch,
                    y,
                    fit_intercept=fit_intercept,
                    K=K,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            )
        return np.asarray(
            backend.batch_ordinal_varying_X(
                X_batch,
                y,
                fit_intercept=fit_intercept,
                K=K,
                **kwargs,
            )
        )


# ------------------------------------------------------------------ #
# MultinomialFamily
# ------------------------------------------------------------------ #
#
# Multinomial logistic regression (softmax) models unordered
# categorical outcomes Y ∈ {0, 1, …, K−1} via the multiclass
# softmax link:
#
#   P(Y = k | X) = exp(Xβ_k) / Σ_j exp(Xβ_j)
#
# with class 0 as the reference category (β_0 = 0 ⇒ identifiability).
#
# **Key design decisions** (from the v0.3.0 plan):
#
# 1. **Wald χ² test statistic** — Each predictor j has K−1
#    coefficients across the non-reference categories.  The scalar
#    test statistic for the permutation engine is the Wald χ²:
#        χ²_j = β_j^T [Var(β_j)]^{-1} β_j
#    where β_j is (K-1,) and Var(β_j) is the (K-1, K-1)
#    covariance sub-block from the Hessian inverse.
#
# 2. **direct_permutation = True** — Permute class labels directly.
#    Avoids residual definition issues for nominal outcomes.
#    Blocks Freedman-Lane with clear error message.
#
# 3. **coefs() returns (p,) Wald χ²** — Consistent with the
#    (B, p) contract of the permutation engine.  The convenience
#    method category_coefs() provides the full (p, K-1) matrix for
#    users who need per-category detail.
#
# 4. **score() / null_score()** — Protocol methods for deviance-based
#    joint test, replacing the former duck-typed model_fit_metric /
#    null_fit_metric methods.  Same pattern as OrdinalFamily.


class MultinomialFamily:
    """Multinomial logistic regression (softmax) family.

    Implements the ``ModelFamily`` protocol for unordered categorical
    outcomes with ≥ 3 levels (integer-coded 0, 1, …, K−1).

    Uses ``statsmodels.discrete.discrete_model.MNLogit`` for
    single-model fitting (``fit()``, ``diagnostics()``) and
    delegates batch fitting to the active backend via
    ``resolve_backend()``.

    **Test statistic**: Returns per-predictor Wald χ² statistics
    rather than raw coefficients, since each predictor has K−1
    coefficients in the multinomial model and the permutation engine
    requires a scalar per predictor.

    **Permutation method restrictions**: Only ``ter_braak``,
    ``kennedy``, and ``kennedy_joint`` are supported.  Freedman-Lane
    methods raise ``ValueError`` because multinomial residuals do not
    support the reduced-model residual exchange required by the
    Freedman-Lane algorithm.

    **Optimizer strategy**: The single-model ``fit()`` uses
    statsmodels MNLogit with Newton-Raphson (default).  Batch
    fitting delegates to the active backend (JAX: vmap'd
    Newton-Raphson with autodiff; NumPy: statsmodels MNLogit loop).
    """

    @property
    def name(self) -> str:
        return "multinomial"

    @property
    def residual_type(self) -> str:
        return "none"

    @property
    def direct_permutation(self) -> bool:
        return True

    @property
    def metric_label(self) -> str:
        return "Deviance Reduction"

    @property
    def stat_label(self) -> str:
        return "\u03c7\u00b2"

    # ---- Display ---------------------------------------------------

    def display_header(
        self,
        diagnostics: dict[str, Any],
    ) -> list[tuple[str, str, str, str]]:
        """Return header rows for the multinomial results table."""
        n_cats = diagnostics.get("n_categories")
        n_cats_str = str(n_cats) if n_cats is not None else "N/A"
        llr_p = diagnostics.get("llr_p_value")
        llr_p_str = f"{llr_p:.4e}" if llr_p is not None else "N/A"
        return [
            (
                "Pseudo R-sq:",
                str(diagnostics.get("pseudo_r_squared", "N/A")),
                "BIC:",
                str(diagnostics.get("bic", "N/A")),
            ),
            (
                "Log-Likelihood:",
                str(diagnostics.get("log_likelihood", "N/A")),
                "LL-Null:",
                str(diagnostics.get("log_likelihood_null", "N/A")),
            ),
            ("Categories:", n_cats_str, "LLR p-value:", llr_p_str),
        ]

    def display_diagnostics(
        self,
        diagnostics: dict[str, Any],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Return diagnostic lines and notes for the multinomial family."""
        lines: list[tuple[str, str]] = []
        notes: list[str] = []
        gof = diagnostics.get("multinomial_gof", {})
        if gof:
            pr2 = gof.get("pseudo_r_squared", None)
            pr2_str = f"{pr2:.4f}" if pr2 is not None else "N/A"
            lines.append(("Pseudo R-sq:", f"{pr2_str:>10}"))
            ll = gof.get("log_likelihood", None)
            ll_str = f"{ll:.4f}" if ll is not None else "N/A"
            lines.append(("Log-Likelihood:", f"{ll_str:>10}"))
            n_cats = gof.get("n_categories", None)
            n_cats_str = str(n_cats) if n_cats is not None else "N/A"
            lines.append(("Categories:", f"{n_cats_str:>10}"))
            llr_p = gof.get("llr_p_value", None)
            if llr_p is not None:
                lines.append(("LLR p-value:", f"{_fmt_p(llr_p):>10}"))
            cat_counts = gof.get("category_counts", {})
            if cat_counts:
                counts_str = ", ".join(
                    f"{k}: {v}" for k, v in sorted(cat_counts.items())
                )
                lines.append(("Category counts:", counts_str))
        return lines, notes

    def compute_extended_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
    ) -> dict[str, Any]:
        """Compute multinomial goodness-of-fit diagnostics."""
        from statsmodels.discrete.discrete_model import MNLogit

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                warnings.filterwarnings("ignore", category=HessianInversionWarning)
                X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
                mn_model = MNLogit(y, X_sm).fit(disp=0, maxiter=200)
            llf = float(mn_model.llf)
            llnull = float(mn_model.llnull)
            pseudo_r2 = 1.0 - llf / llnull if llnull != 0.0 else float("nan")
            try:
                llr_p = float(mn_model.llr_pvalue)
            except (AttributeError, TypeError):
                llr_p = float("nan")
            unique_y, counts = np.unique(y, return_counts=True)
            return {
                "multinomial_gof": {
                    "pseudo_r_squared": pseudo_r2,
                    "log_likelihood": llf,
                    "log_likelihood_null": llnull,
                    "n_categories": len(unique_y),
                    "category_counts": dict(
                        zip(unique_y.tolist(), counts.tolist(), strict=True)
                    ),
                    "aic": float(mn_model.aic),
                    "bic": float(mn_model.bic),
                    "llr_p_value": llr_p,
                }
            }
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug(
                "Multinomial GoF diagnostics failed: %s", exc
            )
            return {
                "multinomial_gof": {
                    "pseudo_r_squared": float("nan"),
                    "log_likelihood": float("nan"),
                    "log_likelihood_null": float("nan"),
                    "n_categories": len(np.unique(y)),
                    "warning": f"Diagnostics unavailable: {exc}",
                }
            }

    # ---- Validation ------------------------------------------------

    def validate_y(self, y: np.ndarray) -> None:
        """Check that *y* contains unordered categorical integer data with ≥ 3 levels."""
        if not np.issubdtype(y.dtype, np.number):
            msg = "MultinomialFamily requires numeric Y values."
            raise ValueError(msg)
        if np.any(np.isnan(y)):
            msg = "MultinomialFamily does not accept NaN values in Y."
            raise ValueError(msg)
        if not np.allclose(y, np.round(y)):
            msg = (
                "MultinomialFamily requires integer-coded Y values "
                "(e.g. 0, 1, 2, …, K−1). Got non-integer values."
            )
            raise ValueError(msg)
        n_levels = len(np.unique(y))
        if n_levels < 3:
            msg = (
                f"MultinomialFamily requires ≥ 3 categories, "
                f"got {n_levels}. For binary outcomes, use "
                f"family='logistic' instead."
            )
            raise ValueError(msg)

    # ---- Single-model operations -----------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> Any:
        """Fit a multinomial logistic regression via statsmodels MNLogit.

        Returns the fitted ``MNLogitResults`` object.
        """
        from statsmodels.discrete.discrete_model import MNLogit

        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
            warnings.filterwarnings("ignore", category=HessianInversionWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model = MNLogit(y_arr, X_sm).fit(disp=0, maxiter=200)
        return model

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Return predicted class probabilities as expected value E[Y|X].

        The returned vector has shape ``(n,)`` — the expected value
        E[Y|X] = Σ k·P(Y=k|X), keeping the interface consistent
        with other families (all return a 1-D prediction).
        """
        prob_matrix = np.asarray(model.predict())  # (n, K)
        levels = np.arange(prob_matrix.shape[1])  # [0, 1, …, K-1] category indices
        return np.asarray(prob_matrix @ levels)  # E[Y|X] = Σ k·P(Y=k|X), shape (n,)

    def coefs(self, model: Any) -> np.ndarray:
        """Extract per-predictor Wald χ² test statistics.

        The multinomial model has (K-1) coefficients per predictor.
        To produce the ``(p,)`` vector required by the permutation
        engine, this method computes the multivariate Wald χ² for
        each slope predictor:

            χ²_j = β_j^T [Var(β_j)]^{-1} β_j

        where β_j is the (K-1)-vector of coefficients for predictor j
        across the non-reference categories, and Var(β_j) is the
        corresponding covariance sub-block.

        Use :meth:`category_coefs` for the full ``(p, K-1)`` matrix.
        """
        from ._backends._numpy import _wald_chi2_from_mnlogit

        p_aug = model.model.exog.shape[
            1
        ]  # total columns in design matrix (incl. intercept)
        has_intercept = bool(
            model.k_constant
        )  # True if statsmodels added a constant column
        return np.asarray(  # Wald χ²_j per predictor, shape (p,)
            _wald_chi2_from_mnlogit(model, p_aug, has_intercept)
        )

    def category_coefs(self, model: Any) -> np.ndarray:
        """Extract the full ``(p, K-1)`` coefficient matrix.

        Each row corresponds to a slope predictor (intercept
        excluded), each column to a non-reference category.

        This is a convenience method not on the ModelFamily protocol;
        callers should check ``hasattr(family, 'category_coefs')``
        before use.
        """
        params = np.asarray(model.params)  # (p_aug, K-1)
        if model.k_constant:
            return params[1:]  # (p, K-1)
        return params

    def residuals(
        self,
        model: Any,  # noqa: ARG002
        X: np.ndarray,  # noqa: ARG002
        y: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray:
        """Not implemented — multinomial residuals are not supported.

        Raises:
            NotImplementedError: Always.
        """
        msg = (
            "Multinomial residuals are not supported for the "
            "Freedman-Lane pipeline. MultinomialFamily uses "
            "direct_permutation=True, which bypasses the residual "
            "pipeline entirely."
        )
        raise NotImplementedError(msg)

    # ---- Permutation helpers ---------------------------------------

    def reconstruct_y(
        self,
        predictions: np.ndarray,  # noqa: ARG002
        permuted_residuals: np.ndarray,  # noqa: ARG002
        rng: np.random.Generator,  # noqa: ARG002
    ) -> np.ndarray:
        """Not implemented — multinomial Y-reconstruction is not supported.

        Raises:
            NotImplementedError: Always.
        """
        msg = (
            "Multinomial Y-reconstruction from residuals is not supported. "
            "MultinomialFamily uses direct_permutation=True; the engine "
            "permutes Y directly instead of reconstructing."
        )
        raise NotImplementedError(msg)

    def fit_metric(
        self,
        y_true: np.ndarray,  # noqa: ARG002
        y_pred: np.ndarray,  # noqa: ARG002
    ) -> float:
        """Not implemented — use ``score(model, X, y)`` instead.

        Multinomial deviance requires the fitted model object (for
        log-likelihood), not just predicted values.

        Raises:
            NotImplementedError: Always.
        """
        msg = (
            "MultinomialFamily.fit_metric() is not available because "
            "multinomial deviance requires the fitted model object.  Use "
            "score(model, X, y) instead."
        )
        raise NotImplementedError(msg)

    # ---- Scoring (joint test interface) ----------------------------
    #
    # Multinomial logistic regression (MNLogit / softmax) models a
    # categorical outcome Y ∈ {0, 1, …, K−1} with K ≥ 3 unordered
    # categories.  The model estimates K−1 coefficient vectors
    # (one per non-reference category), and the predicted probability
    # for category k is:
    #
    #   P(Y = k | X) = exp(Xβₖ) / Σⱼ exp(Xβⱼ)   (softmax)
    #
    # Like ordinal models, the output is a probability vector rather
    # than a scalar, so prediction-based metrics are not applicable.
    # The natural fit measure is the deviance:
    #
    #   D = −2 · ℓ(model) = −2 · Σᵢ log P(Yᵢ = yᵢ | Xᵢ, β̂)
    #
    # The score is extracted from ``model.llf`` on the fitted
    # ``MNLogit`` object.  The joint test statistic is
    # Δ = D_reduced − D_full, the same convention as all families.
    #
    # Null model:
    # Unlike the ordinal family, the multinomial null score is NOT
    # computed analytically.  The reason: the multinomial
    # log-likelihood with an intercept-only model is:
    #
    #   ℓ_null = Σₖ nₖ · log(nₖ / n)
    #
    # which looks simple, but statsmodels' MNLogit parameterises the
    # intercept as K−1 log-odds ratios (softmax), not as direct
    # probabilities.  The analytical formula IS equivalent, but
    # rather than risking a discrepancy if the parameterisation
    # changes, we fit an intercept-only MNLogit to get ℓ_null
    # directly.  This is numerically cheap (converges in 1–2
    # iterations) and guaranteed to match the full model's scale.
    # The result has been validated against ``model.llnull`` from
    # full multinomial fits.

    def score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:  # noqa: ARG002
        """Deviance from a fitted multinomial model: ``−2 · ℓ(model)``.

        The log-likelihood ℓ is stored on the fitted ``MNLogit``
        object.  Lower deviance (higher ℓ) means better fit.
        ``X`` and ``y`` are unused — present for protocol
        compatibility.
        """
        return -2.0 * float(model.llf)

    def null_score(self, y: np.ndarray, fit_intercept: bool = True) -> float:
        """Deviance of the intercept-only multinomial model.

        Fits a zero-predictor ``MNLogit`` (intercept-only design
        matrix of shape (n, 1)) via statsmodels and returns
        ``−2 · ℓ_null``.  The MLE converges to category
        probabilities pₖ = nₖ/n in 1–2 Newton iterations.

        This is done by model-fitting rather than analytically to
        stay consistent with the softmax parameterisation used by
        ``MNLogit`` — see the section comment above for rationale.

        Validated against ``model.llnull`` from full multinomial fits.
        """
        from statsmodels.discrete.discrete_model import MNLogit

        n = len(y)  # total observations
        # Intercept-only design matrix: a column of ones.
        if fit_intercept:
            X_null = np.ones((n, 1), dtype=float)  # (n, 1) — intercept-only design
        else:
            X_null = np.zeros((n, 0), dtype=float)  # empty design → degenerate model
        y_arr = np.asarray(y)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SmConvergenceWarning)
            warnings.filterwarnings("ignore", category=HessianInversionWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            null_model = MNLogit(y_arr, X_null).fit(disp=0, maxiter=200)
        return -2.0 * float(null_model.llf)

    # ---- Diagnostics & classical inference -------------------------

    def diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> dict[str, Any]:
        """Multinomial model diagnostics.

        Returns log-likelihood, null log-likelihood, pseudo-R²,
        AIC, BIC, LLR p-value, and category counts.
        """
        model = self.fit(X, y, fit_intercept)
        llf = float(model.llf)  # ℓ(model) — fitted log-likelihood
        llnull = float(model.llnull)  # ℓ₀ — null (intercept-only) log-likelihood
        pseudo_r2 = (
            1.0 - llf / llnull if llnull != 0.0 else float("nan")
        )  # McFadden's R² = 1 − ℓ/ℓ₀

        # Log-likelihood ratio test p-value
        try:
            llr_p = float(model.llr_pvalue)
        except (AttributeError, TypeError):
            llr_p = float("nan")

        unique_y, counts = np.unique(
            y, return_counts=True
        )  # category labels and frequencies

        return {
            "n_observations": len(y),
            "n_features": X.shape[1],
            "n_categories": len(unique_y),
            "category_counts": dict(
                zip(unique_y.tolist(), counts.tolist(), strict=True)
            ),
            "pseudo_r_squared": np.round(pseudo_r2, 4),
            "log_likelihood": np.round(llf, 4),
            "log_likelihood_null": np.round(llnull, 4),
            "aic": np.round(float(model.aic), 4),
            "bic": np.round(float(model.bic), 4),
            "llr_p_value": np.round(llr_p, 6),
        }

    def classical_p_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Asymptotic Wald χ² p-values for slope coefficients.

        Returns one p-value per slope predictor.  Each p-value is
        the survival function of the χ²(K-1) distribution evaluated
        at the predictor's Wald χ² statistic.
        """
        from scipy import stats as sp_stats

        model = self.fit(X, y, fit_intercept)
        wald_stats = self.coefs(model)  # (p,)
        K = len(np.unique(y))  # number of response categories
        df = K - 1  # degrees of freedom per predictor
        return np.asarray(
            sp_stats.chi2.sf(wald_stats, df=df)
        )  # 1 − F_{χ²}(χ²_j; K−1), shape (p,)

    # ---- Exchangeability (v0.4.0 forward-compat) -------------------

    def exchangeability_cells(
        self,
        X: np.ndarray,  # noqa: ARG002
        y: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray | None:
        """Multinomial models assume globally exchangeable responses."""
        return None

    # ---- Batch fitting (hot loop) ----------------------------------
    #
    # Delegates to the active backend (JAX or NumPy) via
    # ``resolve_backend()``.  The JAX backend uses vmap'd
    # Newton–Raphson with autodiff and Wald χ² extraction;
    # the NumPy backend falls back to a joblib-parallelised
    # statsmodels MNLogit loop.

    def batch_fit(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch multinomial fitting via the active backend.

        Delegates to ``backend.batch_multinomial()`` resolved from
        the current configuration.  The number of categories K is
        computed from the union of unique values across all
        permuted Y vectors and forwarded to the backend.

        Returns Wald χ² statistics ``(B, p)`` — one scalar per
        slope predictor per permutation.
        """
        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()

        # Compute K from observed categories across all permutations.
        K = int(len(np.unique(Y_matrix)))

        if backend.name == "numpy":
            return np.asarray(
                backend.batch_multinomial(
                    X,
                    Y_matrix,
                    fit_intercept=fit_intercept,
                    K=K,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            )
        return np.asarray(
            backend.batch_multinomial(
                X,
                Y_matrix,
                fit_intercept=fit_intercept,
                K=K,
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
        """Batch multinomial with per-permutation design matrices.

        Delegates to ``backend.batch_multinomial_varying_X()`` for
        the Kennedy individual path.  Returns Wald χ² statistics
        ``(B, p)``.
        """
        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()

        K = int(len(np.unique(y)))

        if backend.name == "numpy":
            return np.asarray(
                backend.batch_multinomial_varying_X(
                    X_batch,
                    y,
                    fit_intercept=fit_intercept,
                    K=K,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            )
        return np.asarray(
            backend.batch_multinomial_varying_X(
                X_batch,
                y,
                fit_intercept=fit_intercept,
                K=K,
                **kwargs,
            )
        )


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


def resolve_family(
    family: str | ModelFamily,
    y: np.ndarray | None = None,
) -> ModelFamily:
    """Resolve a family string or instance to a concrete ``ModelFamily``.

    When *family* is already a ``ModelFamily`` instance, it is returned
    as-is (pass-through).  This enables callers to pass pre-configured
    instances (e.g. ``NegativeBinomialFamily(alpha=2.0)``) without
    triggering fresh resolution or construction.

    ``"auto"`` → ``"linear"`` or ``"logistic"`` via binary detection
    (exactly two unique values, both in ``{0, 1}``).

    Explicit strings (``"linear"``, ``"logistic"``, ``"poisson"``,
    etc.) map directly to the corresponding registered class.

    Args:
        family: Family identifier string **or** a ``ModelFamily``
            instance.  ``"auto"`` triggers automatic detection.
            Instances are returned immediately.
        y: Response vector of shape ``(n,)``, used only when
            *family* is ``"auto"``.  May be omitted when *family*
            is an explicit string or a ``ModelFamily`` instance.

    Returns:
        A ``ModelFamily`` instance ready for use by the permutation
        engine.

    Raises:
        ValueError: If *family* is ``"auto"`` and *y* is not
            provided, or if *family* is a string that is not
            ``"auto"`` and is not found in the registry.
    """
    # Pass-through for pre-configured instances.
    if isinstance(family, ModelFamily):
        return family
    if family == "auto":
        if y is None:
            msg = "resolve_family() requires 'y' when family='auto'."
            raise ValueError(msg)
        unique_y = np.unique(y)  # sorted distinct values of y
        is_binary = bool(
            len(unique_y) == 2 and np.all(np.isin(unique_y, [0, 1]))
        )  # True iff y ∈ {0, 1}
        if is_binary:
            family = "logistic"
        else:
            family = "linear"
            # Warn if Y looks like count data (non-negative integers,
            # > 2 unique values) — the user may want a count model.
            _is_integer = np.all(
                np.equal(np.mod(y, 1), 0)
            )  # True if every y_i is whole
            _is_nonneg = bool(np.all(y >= 0))  # True if no negative values
            if _is_integer and _is_nonneg and len(unique_y) > 2:
                import warnings

                warnings.warn(
                    "Y looks like count data (non-negative integers with "
                    f"{len(unique_y)} unique values). Consider specifying "
                    "family='poisson' or family='negative_binomial' "
                    "explicitly.",
                    UserWarning,
                    stacklevel=2,
                )

    if family not in _FAMILIES:
        available = ", ".join(sorted(_FAMILIES)) or "(none registered)"
        msg = f"Unknown family {family!r}.  Available families: {available}."
        raise ValueError(msg)

    instance: ModelFamily = _FAMILIES[family]()
    return instance


# ------------------------------------------------------------------ #
# Reduced-model fitting (shared across all strategies)
# ------------------------------------------------------------------ #
#
# Every permutation strategy needs to fit a "reduced model" at some
# point — a model that includes confounders Z but excludes the
# features being tested.  The reduced model serves different roles
# depending on the strategy:
#
# * **ter Braak (1992)**: For each feature j, fit Y ~ X_{-j} to get
#   predicted values ŷ₋ⱼ and residuals e₋ⱼ = Y − ŷ₋ⱼ.  The
#   residuals are then permuted.
#
# * **Kennedy joint (1995)**: Fit Y ~ Z to get the base score
#   S(reduced).  The joint test statistic Δ = S(reduced) − S(full)
#   measures how much the tested features improve fit.
#
# * **Freedman–Lane (1983)**: Fit Y ~ Z to get reduced-model
#   predicted values ŷ_Z.  Permuted responses are constructed as
#   Y* = ŷ_Z + π(residuals_full).
#
# The logic is identical across all strategies:
#   - If Z has columns: fit the model and predict.
#   - If Z has zero columns (no confounders): use the intercept-only
#     prediction, which is ŷᵢ = ȳ for all i.
#
# The mean(y) fallback is correct for ALL families because the
# intercept-only MLE always predicts the sample mean on the
# response scale:
#   - Linear:  β₀ = ȳ              → ŷ = ȳ
#   - Logistic: β₀ = logit(ȳ)      → P̂ = ȳ
#   - Poisson:  β₀ = log(ȳ)        → μ̂ = ȳ
#   - NB:       β₀ = log(ȳ)        → μ̂ = ȳ
# For ordinal/multinomial, the zero-column case doesn't arise in
# practice (these families use direct_permutation or model-object
# scoring), but mean(y) is still a reasonable numeric fallback.


def fit_reduced(
    family: ModelFamily,
    Z: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool,
) -> tuple[Any | None, np.ndarray]:
    """Fit a reduced (confounders-only) model, with intercept-only fallback.

    When *Z* has one or more columns the family's ``fit`` + ``predict``
    are used.  When *Z* has zero columns (no confounders), predictions
    fall back to ``mean(y)`` (if ``fit_intercept``) or zeros.

    This centralises the "fit reduced or fall back" logic that was
    previously duplicated in four places across three strategy files
    (ter Braak, Kennedy joint, Freedman-Lane individual and joint).

    Args:
        family: A resolved ``ModelFamily`` instance.
        Z: Confounder matrix of shape ``(n, q)`` where *q* may be 0.
        y: Response vector of shape ``(n,)``.
        fit_intercept: Whether the model includes an intercept.

    Returns:
        ``(model, predictions)`` — *model* is ``None`` when *Z* has
        zero columns; *predictions* always has shape ``(n,)``.
    """
    n = len(y)
    if Z.shape[1] > 0:
        # Confounders present: fit the family's model on Z and predict.
        model = family.fit(Z, y, fit_intercept)
        preds = family.predict(model, Z)
        return model, preds
    # No confounders: intercept-only prediction = sample mean.
    # This is the MLE prediction for all GLM families (see above).
    if fit_intercept:
        return None, np.full(n, np.mean(y), dtype=float)
    # No intercept, no confounders: degenerate zero prediction.
    return None, np.zeros(n, dtype=float)


# ------------------------------------------------------------------ #
# Register built-in families
# ------------------------------------------------------------------ #

register_family("linear", LinearFamily)
register_family("logistic", LogisticFamily)
register_family("poisson", PoissonFamily)
register_family("negative_binomial", NegativeBinomialFamily)
register_family("ordinal", OrdinalFamily)
register_family("multinomial", MultinomialFamily)

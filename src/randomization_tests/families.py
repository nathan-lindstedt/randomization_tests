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

from typing import Any, Protocol, runtime_checkable

import numpy as np

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

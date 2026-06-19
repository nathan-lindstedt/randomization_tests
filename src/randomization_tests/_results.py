"""Typed result objects for permutation tests.

Frozen dataclasses that provide:

* **Attribute access** — ``result.family``, ``result.method``, etc.
* **Dict-like access** — ``result["family"]``, ``result.get("key")``,
  ``"key" in result`` for consumers that prefer bracket syntax.
* **Serialisation** — ``.to_dict()`` returns a plain ``dict[str, Any]``
  with all NumPy types converted to native Python.

Two concrete result types mirror the two test topologies:

* :class:`IndividualTestResult` — per-coefficient tests
  (ter Braak, Kennedy individual, Freedman–Lane individual).
* :class:`JointTestResult` — group-level improvement tests
  (Kennedy joint, Freedman–Lane joint).

Both types are frozen (immutable after construction) to communicate
that results are a snapshot of a completed test — they should not be
mutated after creation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from ._context import FitContext
    from .families import ModelFamily

# ------------------------------------------------------------------ #
# Serialisation helper
# ------------------------------------------------------------------ #


def _numpy_to_python(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to Python-native types.

    Handles nested dicts, lists, np.ndarray, np.integer, and
    np.floating so that :meth:`to_dict` returns a fully
    JSON-serialisable structure.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # np.bool_ is grouped with np.integer → int() because
    # json.dumps rejects np.bool_ (it is not a Python bool
    # subclass in NumPy ≥ 2.0, where np.bool_ was decoupled
    # from builtins.bool).  int() is safe: True → 1, False → 0.
    if isinstance(obj, (np.integer, np.bool_)):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_numpy_to_python(item) for item in obj]
        return type(obj)(converted)
    return obj


# ------------------------------------------------------------------ #
# Dict-compatibility mixin
# ------------------------------------------------------------------ #


class _DictAccessMixin:
    """Dict-like access convenience for result dataclasses.

    Supports three access patterns:

    1. ``result["key"]``     — raises ``KeyError`` on miss
    2. ``result.get(key, d)`` — returns *d* on miss (default ``None``)
    3. ``"key" in result``   — membership test

    Subclasses may override ``_SERIALIZERS`` to register custom
    conversion functions for non-primitive fields (e.g.
    ``ModelFamily`` → ``str``).  Serializers compose with
    :func:`_numpy_to_python` — serialized values containing NumPy
    types are still converted.
    """

    _SERIALIZERS: ClassVar[dict[str, Any]] = {}
    """Per-field serializer callables.  Subclasses override this to register
    custom conversion functions for non-primitive fields.  The base default
    is empty — domain-specific knowledge (e.g. ``ModelFamily`` → ``str``)
    belongs on the concrete subclass, not the generic mixin."""

    # Fields to exclude from to_dict() serialisation.
    _EXCLUDE_FROM_DICT: ClassVar[frozenset[str]] = frozenset({"context"})

    def __getitem__(self, key: str) -> Any:
        """Attribute lookup via bracket syntax."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        """Attribute lookup with a fallback default."""
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        """Membership test: ``"key" in result``.

        Returns ``True`` only for declared dataclass *data fields*,
        not for methods, ``ClassVar`` entries, or other class
        attributes.  This ensures semantically correct container
        behaviour — ``"to_dict" in result`` is ``False``.
        """
        if not isinstance(key, str):
            return False
        return key in {f.name for f in fields(self)}  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary.

        Applies per-field serializers from ``_SERIALIZERS``, then runs
        :func:`_numpy_to_python` on every value so the returned dict
        is fully JSON-serialisable.
        """
        result: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            if f.name in self._EXCLUDE_FROM_DICT:
                continue
            val = getattr(self, f.name)
            if f.name in self._SERIALIZERS:
                val = self._SERIALIZERS[f.name](val)
            result[f.name] = _numpy_to_python(val)
        return result


# ------------------------------------------------------------------ #
# IndividualTestResult
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class IndividualTestResult(_DictAccessMixin):
    """Result from a per-coefficient permutation test.

    Returned by ``randomization_test_regression`` for methods
    ``"ter_braak"``, ``"kennedy"``, and ``"freedman_lane"``.

    All fields are accessible both as attributes (``result.family``)
    and via dict syntax (``result["family"]``).
    """

    _SERIALIZERS: ClassVar[dict[str, Any]] = {
        "family": lambda f: f.name,
    }

    # ---- Coefficients & null distribution --------------------------
    model_coefs: list[float]
    """Observed (unpermuted) slope coefficients, one per feature."""

    permuted_coefs: list[list[float]]
    """Permuted coefficient matrix ``(B, p)`` as nested lists."""

    # ---- P-values --------------------------------------------------
    permuted_p_values: list[str]
    """Formatted empirical p-value strings with significance markers."""

    classic_p_values: list[str]
    """Formatted classical (asymptotic) p-value strings."""

    raw_empirical_p: np.ndarray
    """Raw numeric empirical p-values, shape ``(p,)``."""

    raw_classic_p: np.ndarray
    """Raw numeric classical p-values, shape ``(p,)``."""

    # ---- Thresholds ------------------------------------------------
    p_value_threshold_one: float
    """First significance level (default 0.05)."""

    p_value_threshold_two: float
    """Second significance level (default 0.01)."""

    p_value_threshold_three: float
    """Third significance level (default 0.001)."""

    # ---- Metadata --------------------------------------------------
    method: str
    """Permutation method used (e.g. ``"ter_braak"``)."""

    confounders: list[str]
    """Confounder column names (empty list if none)."""

    family: ModelFamily
    """Model family instance (e.g. ``LinearFamily()``)."""

    backend: str
    """Compute backend used (``"numpy"`` or ``"jax"``)."""

    feature_names: list[str]
    """Feature column names from the design matrix."""

    target_name: str
    """Target column name."""

    n_randomizations: int
    """Actual number of randomizations used."""

    groups: np.ndarray | None
    """Exchangeability group labels (``None`` until v0.4.1)."""

    permutation_strategy: str | None
    """``"within"``, ``"between"``, ``"two-stage"``, or ``None``."""

    # ---- Diagnostics -----------------------------------------------
    diagnostics: dict[str, Any]
    """Model diagnostics (R², AIC, etc.) from statsmodels."""

    extended_diagnostics: dict[str, Any]
    """Per-predictor diagnostics (VIF, standardised coefs, etc.)."""

    confidence_intervals: dict[str, Any] = field(default_factory=dict)
    """Confidence intervals dict (permutation, Wald, Clopper-Pearson,
    standardised).  Empty when ``confidence_level`` is not provided."""

    # ---- Computation context (not serialised) ----------------------
    context: FitContext | None = field(default=None, repr=False, compare=False)
    """Pipeline computation context.  Carries intermediate artifacts
    (predictions, residuals, fit metric, etc.) for downstream display
    and debugging.  Excluded from ``to_dict()`` serialisation."""


# ------------------------------------------------------------------ #
# JointTestResult
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class JointTestResult(_DictAccessMixin):
    """Result from a joint (group-level) permutation test.

    Returned by ``randomization_test_regression`` for methods
    ``"kennedy_joint"`` and ``"freedman_lane_joint"``.

    All fields are accessible both as attributes and via dict syntax.
    """

    _SERIALIZERS: ClassVar[dict[str, Any]] = {
        "family": lambda f: f.name,
    }

    # ---- Test statistic & null distribution -------------------------
    observed_improvement: float
    """Observed fit-improvement statistic (RSS or deviance reduction)."""

    permuted_improvements: list[float]
    """Permuted improvement values under H₀, length ``B``."""

    # ---- P-value ---------------------------------------------------
    p_value: float
    """Phipson & Smyth corrected joint p-value."""

    p_value_str: str
    """Formatted p-value string with significance marker."""

    # ---- Metric metadata -------------------------------------------
    metric_type: str
    """Label for the fit metric (``"RSS Reduction"`` or ``"Deviance Reduction"``)."""

    family: ModelFamily
    """Model family instance (e.g. ``LinearFamily()``)."""

    backend: str
    """Compute backend used."""

    # ---- Features & confounders ------------------------------------
    features_tested: list[str]
    """Non-confounder feature names included in the joint test."""

    confounders: list[str]
    """Confounder column names."""

    feature_names: list[str]
    """All feature column names from the design matrix."""

    target_name: str
    """Target column name."""

    n_randomizations: int
    """Actual number of randomizations used."""

    groups: np.ndarray | None
    """Exchangeability group labels (``None`` until v0.4.1)."""

    permutation_strategy: str | None
    """``"within"``, ``"between"``, ``"two-stage"``, or ``None``."""

    # ---- Thresholds ------------------------------------------------
    p_value_threshold_one: float
    """First significance level."""

    p_value_threshold_two: float
    """Second significance level."""

    p_value_threshold_three: float
    """Third significance level."""

    # ---- Method & diagnostics --------------------------------------
    method: str
    """Permutation method used."""

    diagnostics: dict[str, Any]
    """Model diagnostics from statsmodels."""

    extended_diagnostics: dict[str, Any] = field(default_factory=dict)
    """Per-predictor diagnostics (parity with IndividualTestResult)."""

    # ---- Computation context (not serialised) ----------------------
    context: FitContext | None = field(default=None, repr=False, compare=False)
    """Pipeline computation context.  Carries intermediate artifacts
    (predictions, residuals, fit metric, etc.) for downstream display
    and debugging.  Excluded from ``to_dict()`` serialisation."""


# ------------------------------------------------------------------ #
# Confounder analysis result
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class ConfounderAnalysisResult(_DictAccessMixin):
    """Typed result from the confounder sieve.

    Groups classified candidates by causal role:

    * **confounders** — should be controlled for (X ← Z → Y).
    * **mediators** — lie on the causal path (X → M → Y).
    * **moderators** — change the strength of X → Y (informational;
      variable stays in the confounder pool).
    * **colliders** — must NOT be controlled for (X → Z ← Y).

    Provides dict-like access and ``.to_dict()`` for backward
    compatibility with the legacy ``dict`` return format.
    """

    _EXCLUDE_FROM_DICT: ClassVar[frozenset[str]] = frozenset()

    predictor: str
    """Predictor of interest."""

    identified_confounders: list[str]
    """Variables classified as confounders (should be controlled)."""

    identified_mediators: list[str]
    """Variables classified as mediators (should NOT be controlled)."""

    identified_moderators: list[str]
    """Variables classified as moderators (informational)."""

    identified_colliders: list[str]
    """Variables classified as colliders (must NOT be controlled)."""

    screening_results: dict[str, Any]
    """Output from :func:`screen_potential_confounders`."""

    mediation_results: dict[str, Any] = field(default_factory=dict)
    """Per-candidate mediation analysis results."""

    moderation_results: dict[str, Any] = field(default_factory=dict)
    """Per-candidate moderation analysis results."""

    collider_results: dict[str, Any] = field(default_factory=dict)
    """Per-candidate collider test results."""


# ------------------------------------------------------------------ #
# KernelTestResult  (L3 — Kernel / Distribution Testing)
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class KernelTestResult(_DictAccessMixin):
    """Result from a kernel-based two-sample or independence test.

    Returned by :func:`mmd_test`, :func:`hsic_test`, and
    :func:`kernel_regression_test`.

    All fields are accessible both as attributes and via dict syntax.
    """

    _SERIALIZERS: ClassVar[dict[str, Any]] = {}
    _EXCLUDE_FROM_DICT: ClassVar[frozenset[str]] = frozenset()

    statistic: float
    """Observed test statistic (MMD², HSIC, or kernel regression score)."""

    p_value: float
    """Phipson & Smyth corrected permutation p-value."""

    null_distribution: np.ndarray
    """Permuted statistic values under H₀, shape ``(n_permutations,)``."""

    kernel_name: str
    """Name of the kernel used (e.g. ``"gaussian"``, ``"cosine"``)."""

    n_permutations: int
    """Number of permutations used."""

    method: str
    """Test method: ``"mmd"``, ``"hsic"``, or ``"kernel_regression"``."""


# ------------------------------------------------------------------ #
# ConformalResult  (L2 — Inference / Conformal Prediction)
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class ConformalResult(_DictAccessMixin):
    """Result from a conformal prediction procedure.

    Returned by :func:`conformal_prediction`.

    For regression families, ``prediction_interval`` is populated and
    ``prediction_set`` is ``None``.  For classification families
    (logistic, ordinal, multinomial), ``prediction_set`` is populated
    and ``prediction_interval`` is ``None``.
    """

    _SERIALIZERS: ClassVar[dict[str, Any]] = {
        # Tuple is not JSON-serialisable as a tuple; emit list instead.
        "prediction_interval": lambda t: list(t) if t is not None else None,
    }
    _EXCLUDE_FROM_DICT: ClassVar[frozenset[str]] = frozenset()

    prediction_interval: tuple[float, float] | None
    """``(lower, upper)`` prediction interval for regression; ``None`` for
    classification."""

    prediction_set: np.ndarray | None
    """Array of class labels included in the prediction set for
    classification; ``None`` for regression."""

    nonconformity_scores: np.ndarray
    """Nonconformity scores from the calibration set, shape
    ``(n_calibration,)``."""

    confidence_level: float
    """Nominal coverage level (e.g. ``0.95``)."""

    method: str
    """Conformal method: ``"split"`` or ``"full"``."""


# ------------------------------------------------------------------ #
# InvarianceResult  (L2 — Inference / Invariance Testing)
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class InvarianceResult(_DictAccessMixin):
    """Result from a multi-environment invariance test.

    Returned by :func:`invariance_test`.

    Tests H₀: the coefficient of the predictor of interest is identical
    across all environments (Peters, Bühlmann & Meinshausen, 2016).
    """

    _SERIALIZERS: ClassVar[dict[str, Any]] = {}
    _EXCLUDE_FROM_DICT: ClassVar[frozenset[str]] = frozenset()

    is_invariant: bool
    """``True`` when the test fails to reject H₀ (invariant effect)."""

    p_value: float
    """Permutation p-value for the max-deviation test statistic."""

    environment_coefficients: dict[str, float]
    """Per-environment coefficient estimates, keyed by environment label."""

    pooled_coefficient: float
    """Coefficient estimate from the pooled (full-data) model."""

    max_deviation: float
    """Observed max|β̂_e − β̂_pool| across environments."""

    null_distribution: np.ndarray
    """Permuted max-deviation values under H₀, shape ``(n_permutations,)``."""


# ------------------------------------------------------------------ #
# KnockoffResult  (L2/L3 — Model-X Knockoffs)
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class KnockoffResult(_DictAccessMixin):
    """Result from a Model-X knockoff variable-selection procedure.

    Returned by :func:`knockoff_test`.

    Provides FDR-controlled feature selection via the knockoff+
    threshold (Barber & Candès, 2015).
    """

    _SERIALIZERS: ClassVar[dict[str, Any]] = {}
    _EXCLUDE_FROM_DICT: ClassVar[frozenset[str]] = frozenset()

    selected_features: list[str]
    """Feature names that pass the knockoff+ selection threshold."""

    knockoff_statistics: np.ndarray
    """W-statistics ``W_j = |β_j| − |β̃_j|`` for each feature,
    shape ``(n_features,)``."""

    fdr_level: float
    """Target FDR level (e.g. ``0.1``)."""

    threshold: float
    """Knockoff+ threshold τ such that estimated FDP ≤ ``fdr_level``."""

    feature_names: list[str]
    """All feature names in the order matching ``knockoff_statistics``."""

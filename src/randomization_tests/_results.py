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

    _SERIALIZERS: ClassVar[dict[str, Any]] = {
        "family": lambda f: f.name,
    }

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
        """Membership test: ``"key" in result``."""
        if not isinstance(key, str):
            return False
        # Only report fields that are actual dataclass fields (or
        # inherited attributes), not arbitrary object attributes like
        # __class__.  Using hasattr is intentional — it catches both
        # dataclass fields and any future @property additions.
        return hasattr(self, key)

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

    Returned by ``permutation_test_regression`` for methods
    ``"ter_braak"``, ``"kennedy"``, and ``"freedman_lane"``.

    All fields are accessible both as attributes (``result.family``)
    and via dict syntax (``result["family"]``).
    """

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

    n_permutations: int
    """Actual number of permutations used."""

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

    Returned by ``permutation_test_regression`` for methods
    ``"kennedy_joint"`` and ``"freedman_lane_joint"``.

    All fields are accessible both as attributes and via dict syntax.
    """

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

    n_permutations: int
    """Actual number of permutations used."""

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

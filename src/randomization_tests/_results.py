"""Typed result objects for permutation tests.

Replaces the plain-dictionary return values from
``permutation_test_regression`` with frozen dataclasses that provide:

* **Attribute access** — ``result.family``, ``result.method``, etc.
* **Dict-like access** — ``result["family"]``, ``result.get("key")``,
  ``"key" in result`` for backward compatibility with existing code
  that consumes the old dict interface (display functions, tests,
  downstream scripts).
* **Serialisation** — ``.to_dict()`` returns a plain ``dict[str, Any]``
  identical to the v0.3.17 return format.

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

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

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
#
# All existing consumers of permutation_test_regression — display
# functions, example scripts, user code — access results via dict
# syntax (result["key"], result.get("key", default)).  This mixin
# provides that surface so the dataclass is a drop-in replacement.


class _DictAccessMixin:
    """Backward-compatible dict-like access for result dataclasses.

    Supports three patterns used throughout the codebase:

    1. ``result["key"]``     — raises ``KeyError`` on miss
    2. ``result.get(key, d)`` — returns *d* on miss (default ``None``)
    3. ``"key" in result``   — membership test
    """

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
        """Convert to a plain dictionary (v0.3.17 format).

        NumPy arrays are converted to nested Python lists so that the
        returned dict is JSON-serialisable and matches the format that
        ``permutation_test_regression`` returned prior to v0.3.17.5.
        """
        result: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            val = _numpy_to_python(val)
            result[f.name] = val
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
    and via dict syntax (``result["family"]``) for backward
    compatibility.
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

    # ---- Metadata --------------------------------------------------
    method: str
    """Permutation method used (e.g. ``"ter_braak"``)."""

    confounders: list[str]
    """Confounder column names (empty list if none)."""

    model_type: str
    """Resolved family name (e.g. ``"linear"``, ``"logistic"``)."""

    family: str
    """Alias for ``model_type`` — semantic provenance key."""

    backend: str
    """Compute backend used (``"numpy"`` or ``"jax"``)."""

    # ---- Diagnostics -----------------------------------------------
    diagnostics: dict[str, Any]
    """Model diagnostics (R², AIC, etc.) from statsmodels."""

    extended_diagnostics: dict[str, Any]
    """Per-predictor diagnostics (VIF, standardised coefs, etc.)."""


# ------------------------------------------------------------------ #
# JointTestResult
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class JointTestResult(_DictAccessMixin):
    """Result from a joint (group-level) permutation test.

    Returned by ``permutation_test_regression`` for methods
    ``"kennedy_joint"`` and ``"freedman_lane_joint"``.

    All fields are accessible both as attributes and via dict syntax
    for backward compatibility.
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

    model_type: str
    """Resolved family name."""

    family: str
    """Alias for ``model_type``."""

    backend: str
    """Compute backend used."""

    # ---- Features & confounders ------------------------------------
    features_tested: list[str]
    """Non-confounder feature names included in the joint test."""

    confounders: list[str]
    """Confounder column names."""

    # ---- Thresholds ------------------------------------------------
    p_value_threshold_one: float
    """First significance level."""

    p_value_threshold_two: float
    """Second significance level."""

    # ---- Method & diagnostics --------------------------------------
    method: str
    """Permutation method used."""

    diagnostics: dict[str, Any]
    """Model diagnostics from statsmodels."""

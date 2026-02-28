"""Computation context — mutable accumulator for pipeline artifacts.

A :class:`FitContext` travels through the permutation-test pipeline,
collecting intermediate artifacts at their natural computation points.
Downstream consumers (display, diagnostics, benchmarking) read from
the context instead of re-computing.

The context is **not** part of the public serialisation API: it carries
NumPy arrays and opaque model objects that should not be JSON'd.
:meth:`~_results.IndividualTestResult.to_dict` and
:meth:`~_results.JointTestResult.to_dict` skip it automatically.

Lifecycle::

    ┌─────────────────────────────────────────────┐
    │  permutation_test_regression()              │
    │  ├─ ctx = FitContext()                      │
    │  ├─ PermutationEngine(…, ctx=ctx)           │
    │  │   ├─ ctx.family = resolved_family        │
    │  │   ├─ ctx.observed_model = family.fit(…)  │
    │  │   ├─ ctx.predictions = family.predict(…) │
    │  │   ├─ ctx.coefficients = family.coefs(…)  │
    │  │   ├─ ctx.residuals = family.residuals(…) │
    │  │   ├─ ctx.fit_metric = family.fit_metric  │
    │  │   ├─ ctx.diagnostics = family.diag(…)    │
    │  │   └─ ctx.exch_cells = family.exch_…(…)   │
    │  ├─ strategy.execute(…)                     │
    │  ├─ _package_*_result(…, ctx=ctx)           │
    │  │   ├─ ctx.classical_p_values = …          │
    │  │   └─ result.context = ctx                │
    │  └─ return result                           │
    └─────────────────────────────────────────────┘

    # Later — zero re-computation:
    print_protocol_usage_table(result)  # reads result.context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FitContext:
    """Mutable accumulator for computation artifacts.

    Every field defaults to ``None`` (or an empty container) so the
    context can be created empty at the start of the pipeline and
    populated incrementally.  Consumers should check for ``None``
    before using a field — missing data means that pipeline stage
    has not run yet.

    Sections
    --------
    **Inputs** — raw arrays as seen by the engine.

    **Family** — the resolved ``ModelFamily`` instance and its
    protocol properties.

    **Observed fit** — model, predictions, coefficients, residuals,
    and fit metric from the single observed-data fit.

    **Diagnostics** — model-level and per-predictor diagnostics.

    **Inference** — classical p-values and exchangeability cells.

    **Permutation metadata** — method, backend, strategy, groups,
    permutation count.

    **Batch-fit stats** — convergence information from the
    permutation loop.
    """

    # ---- Inputs --------------------------------------------------
    X: np.ndarray | None = None
    """Design matrix ``(n, p)``."""

    y: np.ndarray | None = None
    """Response vector ``(n,)``."""

    feature_names: list[str] | None = None
    """Feature column names from the original DataFrame."""

    target_name: str | None = None
    """Target column name from the original DataFrame."""

    # ---- Family --------------------------------------------------
    family: Any = None
    """Resolved ``ModelFamily`` instance."""

    family_name: str | None = None
    """Short name (e.g. ``"poisson"``)."""

    residual_type: str | None = None
    """Residual type string (e.g. ``"deviance"``)."""

    direct_permutation: bool | None = None
    """Whether this family uses direct Y permutation."""

    metric_label: str | None = None
    """Fit-metric label (e.g. ``"Deviance"``)."""

    # ---- Observed fit --------------------------------------------
    observed_model: Any = None
    """Opaque fitted model object from ``family.fit()``."""

    predictions: np.ndarray | None = None
    """Predicted values ``(n,)`` from the observed model."""

    coefficients: np.ndarray | None = None
    """Observed slope coefficients ``(p,)``."""

    residuals: np.ndarray | None = None
    """Residuals ``(n,)`` from the observed model.

    ``None`` for families where residuals are not well-defined
    (ordinal, multinomial with ``direct_permutation=True``).
    """

    fit_metric_value: float | None = None
    """Scalar fit metric (RSS, deviance, etc.) from the observed model."""

    # ---- Diagnostics ---------------------------------------------
    diagnostics: dict[str, Any] = field(default_factory=dict)
    """Model-level diagnostics dict from ``family.diagnostics()``."""

    extended_diagnostics: dict[str, Any] = field(default_factory=dict)
    """Per-predictor diagnostics from ``compute_all_diagnostics()``."""

    # ---- Inference -----------------------------------------------
    classical_p_values: np.ndarray | None = None
    """Classical (asymptotic) p-values ``(p,)``."""

    exchangeability_cells: np.ndarray | None = None
    """Cell labels from ``family.exchangeability_cells()``, or ``None``."""

    # ---- Permutation metadata ------------------------------------
    method: str | None = None
    """Permutation method (e.g. ``"ter_braak"``)."""

    backend: str | None = None
    """Compute backend (``"numpy"`` or ``"jax"``)."""

    n_permutations: int | None = None
    """Number of permutations actually used."""

    groups: np.ndarray | None = None
    """User-supplied group labels, or ``None``."""

    permutation_strategy: str | None = None
    """``"within"``, ``"between"``, ``"two-stage"``, or ``None``."""

    confounders: list[str] = field(default_factory=list)
    """Confounder column names."""

    # ---- Batch-fit stats -----------------------------------------
    batch_shape: tuple[int, ...] | None = None
    """Shape of the permuted-coefficients matrix ``(B, p)``."""

    convergence_count: int | None = None
    """Number of permutation fits that converged (non-NaN rows)."""

    # ---- LMM-specific artifacts ----------------------------------
    raw_groups: Any = None
    """Original grouping specification before integer encoding.

    Preserved from the user's input so downstream consumers
    (display, diagnostics) can show named factors rather than
    opaque integer labels.  May be a 1-D array, a dict of arrays,
    or ``None``.
    """

    groups_arr: np.ndarray | None = None
    """Integer group labels ``(n,)`` for the first (outermost) factor.

    Derived once from ``Z`` during calibration and reused by
    ``diagnostics()``, ``classical_p_values()``, and display.
    Eliminates the repeated ``np.argmax(Z[:, ...])`` reconstruction.
    """

    factor_names: list[str] | None = None
    """Named grouping factors when ``groups`` is a dict.

    ``None`` when ``groups`` is a simple 1-D array or not provided.
    """

    exog_re_kw: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for ``statsmodels.MixedLM`` random-effects
    specification.  Built once during calibration; reused by
    ``diagnostics()`` and ``classical_p_values()`` to avoid
    reconstructing the ``exog_re`` matrix each time.
    """

    sm_mixed_model: Any = None
    """Fitted ``statsmodels.MixedLMResults`` object from calibration.

    Cached so that ``diagnostics()`` can read AIC/BIC and
    ``classical_p_values()`` can read Wald p-values without
    re-fitting the model.  ``None`` when the JAX calibration
    path was used (statsmodels is not invoked).
    """

    # ---- Warnings ------------------------------------------------
    warnings_captured: list[str] = field(default_factory=list)
    """Warning messages captured during the pipeline."""


__all__ = ["FitContext"]

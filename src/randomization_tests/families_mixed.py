"""Linear mixed-effects model family for clustered/grouped outcomes.

Implements the ``ModelFamily`` protocol for continuous outcomes with
random-intercept and/or random-slope structure arising from
clustering or repeated measures.

Model:

    y = Xβ + Zu + ε,   u ~ N(0, σ²Γ),   ε ~ N(0, σ²I)

where Γ is block-diagonal:

    Γ = block_diag( kron(I_{G_1}, Σ_1), …, kron(I_{G_K}, Σ_K) )

and Σ_k is the (d_k × d_k) covariance matrix for one group in
factor k (intercept + slopes), estimated via Cholesky decomposition
Σ_k = L_k L_k' with log-Cholesky parameterisation.

The fixed effects β and variance components (σ², {Σ_k}) are
estimated via REML using Henderson's mixed-model equations — no
per-cluster iteration, no padding, no explicit N×N covariance matrix.

The key quantity for the permutation test is the **GLS projection
matrix** A = S⁻¹ X'Ṽ⁻¹, which is σ²-free.  This matrix is computed
once during ``calibrate()`` and reused for all B permutations via
the batch matmul A @ Y_π — structurally identical to what ``batch_ols``
does with ``pinv(X) @ Y.T``.

Architecture
~~~~~~~~~~~~
* **Step A.1:** ``_reml_solve()`` in ``_backends/_jax.py`` estimates
  β̂, σ̂², {Σ̂_k}, and the projection A via JAX autodiff Newton–Raphson.
* **Step A.2:** ``batch_mixed_lm()`` / ``batch_mixed_lm_varying_X()``
  in both backends delegate to the projection matmul (or per-X
  rebuild for Kennedy).
* **Step A.3 (this file):** ``LinearMixedFamily`` orchestrates
  calibration, single-model fit, and batch dispatch.

The grouping factor(s) can be supplied as:

* A **1-D array** of integer labels → single random intercept.
* A **dict** mapping factor names to label arrays (or tuples with
  slope column indices) → multiple grouping factors with optional
  correlated random slopes.

``calibrate()`` builds the random-effect design matrix Z and
``re_struct`` from whichever form is provided.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, final

import numpy as np
import statsmodels.api as sm
from typing_extensions import Self

from .families import _augment_intercept

# ------------------------------------------------------------------ #
# Z-construction helpers
# ------------------------------------------------------------------ #


def _build_random_effects_design(
    groups: np.ndarray | dict[str, np.ndarray | tuple[np.ndarray, list[int]]],
    X: np.ndarray | None = None,
    random_slopes: list[int] | dict[str, list[int]] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build the random-effect design matrix Z and re_struct.

    Constructs Z with intercept and (optionally) slope columns per
    group per factor.  For each factor k with G_k groups and slope
    columns ``[c_1, …, c_s]``, the RE dimension is ``d_k = 1 + s``
    and Z_k has ``G_k * d_k`` columns arranged as::

        [group_0_intercept, group_0_slope_c1, …,
         group_1_intercept, group_1_slope_c1, …, …]

    When no slopes are requested (``random_slopes=None``), d_k = 1
    and Z_k is the standard one-hot indicator matrix — identical
    to the random-intercept-only case.

    Args:
        groups: Grouping factor specification.
            * 1-D array ``(n,)`` of integer labels → single factor.
            * dict ``{name: array}`` → g grouping factors (intercept
              only for each).
            * dict ``{name: (array, slope_cols)}`` → factors with
              correlated random slopes + intercept.
        X: Fixed-effect design matrix ``(n, p)`` **without**
            intercept column.  Required when ``random_slopes`` is
            not ``None``.
        random_slopes: Slope columns (0-based indices into *X*).
            * ``None`` → intercept only for all factors.
            * ``list[int]`` → slope columns for the single factor
              (only valid when *groups* is a 1-D array).
            * ``dict {name: list[int]}`` → slope columns per factor
              (only valid when *groups* is a dict).

    Returns:
        ``(Z, re_struct)`` where:
            * ``Z`` is ``(n, q)`` with ``q = Σ_k G_k · d_k``.
            * ``re_struct`` is ``[(G_1, d_1), …, (G_K, d_K)]``.

    Raises:
        ValueError: If label arrays differ in length, groups is
            empty, or slopes are requested without X.
    """
    # Normalise (groups, random_slopes) into a list of factors
    factors: list[tuple[str, np.ndarray, list[int]]] = []

    if isinstance(groups, dict):
        if len(groups) == 0:
            msg = "groups dict must contain at least one grouping factor."
            raise ValueError(msg)
        # Normalise random_slopes to dict
        if random_slopes is None:
            slopes_dict: dict[str, list[int]] = {}
        elif isinstance(random_slopes, dict):
            slopes_dict = random_slopes
        else:
            msg = (
                "random_slopes must be a dict when groups is a dict, "
                f"got {type(random_slopes).__name__}."
            )
            raise ValueError(msg)

        od = OrderedDict(groups)
        for name, val in od.items():
            if isinstance(val, tuple):
                # dict value is (labels_array, slope_col_list)
                labels, slope_cols = val
                labels = np.asarray(labels)
            else:
                labels = np.asarray(val)
                slope_cols = slopes_dict.get(name, [])
            factors.append((name, labels, slope_cols))
    else:
        # Single 1-D array
        labels = np.asarray(groups)
        if labels.ndim != 1:
            msg = f"groups must be a 1-D array or a dict, got shape {labels.shape}."
            raise ValueError(msg)
        if random_slopes is None:
            slope_cols_single: list[int] = []
        elif isinstance(random_slopes, list):
            slope_cols_single = random_slopes
        else:
            msg = (
                "random_slopes must be a list[int] when groups is a "
                f"1-D array, got {type(random_slopes).__name__}."
            )
            raise ValueError(msg)
        factors.append(("factor_0", labels, slope_cols_single))

    # Validate X is provided when slopes are requested
    has_slopes = any(len(sc) > 0 for _, _, sc in factors)
    if has_slopes and X is None:
        msg = (
            "X must be provided when random_slopes is specified "
            "(needed to build slope columns of Z)."
        )
        raise ValueError(msg)

    # Build Z and re_struct
    n: int | None = None
    Z_list: list[np.ndarray] = []
    re_struct: list[tuple[int, int]] = []

    for name, labels, slope_cols in factors:
        if labels.ndim != 1:
            msg = (
                f"Grouping factor '{name}' must be a 1-D array, "
                f"got shape {labels.shape}."
            )
            raise ValueError(msg)
        if n is None:
            n = len(labels)
        elif len(labels) != n:
            msg = (
                f"Grouping factor '{name}' has {len(labels)} "
                f"observations, expected {n}."
            )
            raise ValueError(msg)

        # Map labels to 0-based contiguous integers
        unique_labels, coded = np.unique(labels, return_inverse=True)
        G_k = len(unique_labels)
        d_k = 1 + len(slope_cols)  # intercept + number of slopes

        # Build Z_k with G_k * d_k columns
        Z_k = np.zeros((n, G_k * d_k), dtype=np.float64)
        for j in range(G_k):
            mask = coded == j
            col_base = j * d_k
            # Intercept column
            Z_k[mask, col_base] = 1.0
            # Slope columns
            for s_idx, col_idx in enumerate(slope_cols):
                assert X is not None  # validated above
                Z_k[mask, col_base + 1 + s_idx] = X[mask, col_idx]

        Z_list.append(Z_k)
        re_struct.append((G_k, d_k))

    Z = np.hstack(Z_list) if len(Z_list) > 1 else Z_list[0]
    return Z, re_struct


# ------------------------------------------------------------------ #
# Variance-component extraction (shared by all 3 mixed families)
# ------------------------------------------------------------------ #


def _extract_variance_components(
    re_struct: Sequence[tuple[Any, int]],
    re_covariances: Sequence[np.ndarray],
) -> tuple[list[dict[str, Any]], float]:
    """Build structured variance-component dicts from RE covariance matrices.

    Iterates over ``re_struct`` / ``re_covariances`` and returns a
    ``factors`` list with intercept variance, optional slope variances,
    and intercept–slope correlations (for factors with *d_k* > 1),
    together with ``total_tau2 = Σ_k τ²_k`` (sum of intercept
    variances across all factors).

    Returns
    -------
    factors : list[dict]
        One dict per random-effects factor.
    total_tau2 : float
        Sum of intercept variances, used by the caller for ICC.
    """
    factors: list[dict[str, Any]] = []
    for k, cov_k in enumerate(re_covariances):
        _G_k, d_k = re_struct[k]
        entry: dict[str, Any] = {
            "index": k,
            "d_k": d_k,
            "intercept_var": float(cov_k[0, 0]),
        }
        if d_k > 1:
            entry["slope_vars"] = [float(cov_k[s, s]) for s in range(1, d_k)]
            # Correlations between intercept and each slope
            correlations: list[dict[str, Any]] = []
            stds = np.sqrt(np.diag(cov_k))
            for s in range(1, d_k):
                if stds[0] > 0 and stds[s] > 0:
                    rho = float(cov_k[0, s] / (stds[0] * stds[s]))
                else:
                    rho = 0.0
                correlations.append(
                    {
                        "label": f"int, slope {s - 1}",
                        "value": rho,
                    }
                )
            entry["correlations"] = correlations
        factors.append(entry)

    total_tau2 = sum(f["intercept_var"] for f in factors)
    return factors, total_tau2


def _format_variance_components(
    vc: dict[str, Any],
) -> list[tuple[str, str, str]]:
    """Format variance-component factors as ``(label, stat, detail)`` display lines.

    Shared by the ``display_diagnostics()`` methods of all three mixed
    families.  The caller appends family-specific rows (σ², ICC,
    dispersion, convergence notes) before and/or after these lines.
    """
    lines: list[tuple[str, str, str]] = []
    for k_info in vc.get("factors", []):
        k = k_info["index"]
        d_k = k_info["d_k"]
        label = f"factor {k}" if len(vc.get("factors", [])) > 1 else ""
        if d_k == 1:
            tau2 = k_info["intercept_var"]
            tag = f" ({label})" if label else ""
            lines.append((f"τ² (intercept{tag}):", f"{tau2:.4f}", ""))
        else:
            tag = f" [{label}]" if label else ""
            lines.append(
                (f"τ² (intercept{tag}):", f"{k_info['intercept_var']:.4f}", "")
            )
            for s_idx, sv in enumerate(k_info.get("slope_vars", [])):
                lines.append((f"τ² (slope {s_idx}{tag}):", f"{sv:.4f}", ""))
            for corr_entry in k_info.get("correlations", []):
                lines.append(
                    (
                        f"ρ ({corr_entry['label']}{tag}):",
                        f"{corr_entry['value']:.4f}",
                        "",
                    )
                )
    return lines


def _require_calibrated_guard(
    obj: Any,
    method: str,
    *,
    field: str,
    family_name: str,
) -> None:
    """Guard: raise ``RuntimeError`` if ``calibrate()`` not yet called.

    Shared implementation for all three mixed-family classes.
    Each class exposes a thin ``_require_calibrated`` method that
    delegates here with its specific field and family name.
    """
    if getattr(obj, field) is None:
        msg = (
            f"{family_name}.{method}() requires calibration. "
            "Call calibrate(X, y, fit_intercept, groups=...) first."
        )
        raise RuntimeError(msg)


# ------------------------------------------------------------------ #
# LMM fit result (returned by fit())
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class _LMMFitResult:
    """Lightweight container from a single LMM fit.

    Stores the quantities needed by ``predict()``, ``coefs()``,
    ``residuals()``, and ``score()`` — nothing else.  Keeps the
    ``fit()`` return type opaque to the caller while giving
    protocol methods a typed object to extract from.
    """

    beta: np.ndarray  # (p_aug,) — includes intercept if fit_intercept
    sigma2: float  # residual variance σ̂²
    predictions: np.ndarray  # (n,) = X_aug @ beta
    fit_intercept: bool  # whether beta[0] is an intercept


# ------------------------------------------------------------------ #
# LinearMixedFamily
# ------------------------------------------------------------------ #


@final
@dataclass(frozen=True)
class LinearMixedFamily:
    """Linear mixed-effects model family for grouped/clustered outcomes.

    Implements the ``ModelFamily`` protocol for continuous outcomes
    with random-effect structure.  Uses Henderson-based REML to
    estimate variance components once (during ``calibrate``), then
    exploits the GLS projection matrix A for all B permutation
    refits — no per-permutation REML re-solve.

    The grouping structure is supplied at calibration time via the
    ``groups`` keyword argument (a 1-D array for a single random
    intercept, or a dict of arrays for g grouping factors).

    Construction fields
    -------------------
    All fields are ``None`` until ``calibrate()`` populates them.
    Users should never need to set these directly — they are
    populated internally by the calibration path.

    Parameters
    ----------
    re_struct : tuple[tuple[int, int], ...] or None
        ``(G_k, d_k)`` per random-effect component — number of
        groups and RE dimension (1 = intercept only, >1 = intercept
        + slopes).
    projection_A : np.ndarray or None
        ``(p_aug, n)`` GLS projection matrix (σ²-free).
    sigma2 : float or None
        Residual variance σ̂².
    re_covariances : tuple[np.ndarray, ...] or None
        Per-factor covariance matrices ``σ̂² · Σ̂_k``, each
        of shape ``(d_k, d_k)``.  For intercept-only factors
        (d_k = 1), these are 1×1 arrays equal to ``[[τ̂²_k]]``.
    log_chol : np.ndarray or None
        Optimised log-Cholesky parameters θ̂.
    Z : np.ndarray or None
        ``(n, q)`` random-effect design matrix.
    C22 : np.ndarray or None
        ``(q, q)`` Henderson C₂₂ = Z'Z + Γ⁻¹ — needed for
        varying-X batch rebuild.
    converged : bool or None
        Whether REML converged.
    n_iter : int or None
        Number of Newton–Raphson iterations.
    """

    re_struct: tuple[tuple[int, int], ...] | None = None
    projection_A: np.ndarray | None = None
    sigma2: float | None = None
    re_covariances: tuple[np.ndarray, ...] | None = None
    log_chol: np.ndarray | None = None
    Z: np.ndarray | None = None
    C22: np.ndarray | None = None
    converged: bool | None = None
    n_iter: int | None = None

    # ---- Cached calibration artifacts (avoid redundant refits) -----
    _groups_arr: np.ndarray | None = None
    """Integer group labels ``(n,)`` for the first factor.

    Derived once from ``Z`` during calibration.  Reused by
    ``diagnostics()`` and ``classical_p_values()`` instead of
    re-deriving via ``np.argmax(Z[:, ...])`` each time.
    """

    _exog_re_kw: dict[str, Any] | None = None
    """Keyword arguments for ``statsmodels.MixedLM`` random-effects.

    Built once during calibration; reused to avoid reconstructing
    the exog_re matrix for each statsmodels call.
    """

    _sm_model: Any = None
    """Fitted ``statsmodels.MixedLMResults`` from calibration.

    Cached so ``diagnostics()`` can read AIC/BIC and
    ``classical_p_values()`` can read Wald p-values without
    re-fitting.  ``None`` when JAX calibration was used.
    """

    _raw_groups: Any = None
    """Original grouping specification before integer encoding.

    Preserved from the user's input so downstream consumers
    (display, diagnostics) can show named factors.
    """

    # ---- Protocol constants ----------------------------------------

    @property
    def name(self) -> str:
        return "linear_mixed"

    @property
    def residual_type(self) -> str:
        return "conditional"

    @property
    def direct_permutation(self) -> bool:
        # LMM now supports the residual-based permutation path
        # (ter Braak 1992, Freedman–Lane 1983).  Variance components
        # are estimated once during calibration; fit() rebuilds the
        # GLS projection on the fly when called with a reduced X
        # (different column count), keeping REML fixed.  No per-
        # permutation re-solve needed.
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
        """Return header rows for the linear mixed results table."""
        sigma2_val = diagnostics.get("sigma2")
        sigma2_str = f"{sigma2_val:.4f}" if sigma2_val is not None else "N/A"
        icc = diagnostics.get("icc")
        icc_str = f"{icc:.4f}" if icc is not None else "N/A"

        bic_val = diagnostics.get("bic")
        if bic_val is None or (isinstance(bic_val, float) and bic_val != bic_val):
            bic_str = "N/A"
        else:
            bic_str = str(bic_val)

        n_groups = diagnostics.get("n_groups", "N/A")
        re_summary = diagnostics.get("re_summary", "N/A")
        conv = diagnostics.get("converged")
        conv_str = "Yes" if conv is True else ("No" if conv is False else "N/A")

        return [
            (
                "RE Structure:",
                str(re_summary),
                "BIC:",
                bic_str,
            ),
            (
                "Marginal R²:",
                str(diagnostics.get("r_squared_marginal", "N/A")),
                "No. Groups:",
                str(n_groups),
            ),
            (
                "Conditional R²:",
                str(diagnostics.get("r_squared_conditional", "N/A")),
                "σ² (residual):",
                sigma2_str,
            ),
            (
                "ICC:",
                icc_str,
                "REML converged:",
                conv_str,
            ),
        ]

    def display_diagnostics(
        self,
        diagnostics: dict[str, Any],
    ) -> tuple[list[tuple[str, str, str]], list[str]]:
        """Return diagnostic lines and notes for the LMM family.

        Variance components are presented as individual labelled rows:

        * ``σ² (residual)`` — level-1 residual variance
        * ``τ² (intercept)`` — random-intercept variance per factor
        * ``τ² (slope j)`` — random-slope variance (when present)
        * ``ρ (int, slope j)`` — intercept–slope correlation (when >1 RE)
        * ``ICC`` — intraclass correlation coefficient
        """
        lines: list[tuple[str, str, str]] = []
        notes: list[str] = []
        lmm = diagnostics.get("lmm_gof", {})
        if not lmm:
            return lines, notes

        # σ² (residual)
        sigma2 = lmm.get("sigma2")
        if sigma2 is not None:
            lines.append(("σ² (residual):", f"{sigma2:.4f}", ""))

        # Variance components — flattened representation
        vc = lmm.get("variance_components", {})
        lines.extend(_format_variance_components(vc))

        # ICC
        icc = lmm.get("icc")
        if icc is not None:
            lines.append(("ICC:", f"{icc:.4f}", ""))

        # Convergence note
        conv = lmm.get("converged")
        if conv is not None and not conv:
            notes.append(
                "REML solver did not converge — variance component "
                "estimates may be unreliable."
            )
        return lines, notes

    def compute_extended_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
    ) -> dict[str, Any]:
        """Compute LMM goodness-of-fit diagnostics."""
        self._require_calibrated("compute_extended_diagnostics")
        assert self.sigma2 is not None  # for mypy
        assert self.re_covariances is not None
        assert self.re_struct is not None

        # Build structured variance components
        factors, total_tau2 = _extract_variance_components(
            self.re_struct, self.re_covariances
        )
        var_comps: dict[str, Any] = {"factors": factors}

        # ICC = Σ τ²_intercept_k / (Σ τ²_intercept_k + σ²)
        total_var = total_tau2 + self.sigma2
        icc = total_tau2 / total_var if total_var > 0 else 0.0

        return {
            "lmm_gof": {
                "sigma2": float(self.sigma2),
                "variance_components": var_comps,
                "icc": icc,
                "converged": self.converged,
                "n_iter": self.n_iter,
            }
        }

    # ---- Validation ------------------------------------------------

    def validate_y(self, y: np.ndarray) -> None:
        """Check that *y* is numeric and non-constant."""
        if not np.issubdtype(y.dtype, np.number):
            msg = "LinearMixedFamily requires numeric Y values."
            raise ValueError(msg)
        if np.ptp(y) == 0:
            msg = "LinearMixedFamily requires non-constant Y (zero variance)."
            raise ValueError(msg)

    # ---- Single-model operations -----------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> Any:
        """Compute β̂ using the calibrated variance components.

        When *X* matches the calibrated design, uses the pre-computed
        GLS projection ``β̂ = A @ y`` (fast path).  When *X* has a
        different number of columns (e.g. the reduced design in
        ter Braak or Freedman–Lane), rebuilds the GLS projection
        on the fly from stored Henderson components ``C₂₂`` and
        ``Z`` — keeping variance components fixed from REML.

        Requires ``calibrate()`` to have been called first.

        Returns:
            An ``_LMMFitResult`` with β̂, σ̂², predictions, and
            the intercept flag.
        """
        self._require_calibrated("fit")
        assert self.projection_A is not None  # for mypy
        assert self.sigma2 is not None

        X_aug = _augment_intercept(X, fit_intercept)

        p_aug = X_aug.shape[1]
        if p_aug == self.projection_A.shape[0]:
            # Fast path: X matches calibrated dimensions.
            beta = self.projection_A @ y
        else:
            # Reduced/different X: rebuild GLS estimate via Woodbury.
            # Variance components (encoded in C₂₂, Z) stay fixed
            # from calibration; only the fixed-effect projection
            # changes.  This mirrors batch_mixed_lm_varying_X in
            # the numpy backend.
            assert self.Z is not None and self.C22 is not None
            Z = self.Z  # (n, q)
            C22 = self.C22  # (q, q)
            C22_inv_Zt = np.linalg.solve(C22, Z.T)  # (q, n)
            XtZ = X_aug.T @ Z  # (p_aug, q)
            S = X_aug.T @ X_aug - XtZ @ (C22_inv_Zt @ X_aug)  # (p_aug, p_aug)
            Xt_Vinv = X_aug.T - XtZ @ C22_inv_Zt  # (p_aug, n)
            A_red = np.linalg.solve(S, Xt_Vinv)  # (p_aug, n)
            beta = A_red @ y  # (p_aug,)

        predictions = X_aug @ beta
        return _LMMFitResult(
            beta=beta,
            sigma2=self.sigma2,
            predictions=predictions,
            fit_intercept=fit_intercept,
        )

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Return marginal predictions ŷ = X_aug @ β̂.

        Excludes random effects — these are marginal (population-
        level) predictions, appropriate for permutation testing
        where we test fixed-effect significance.
        """
        return np.asarray(model.predictions)

    def coefs(self, model: Any) -> np.ndarray:
        """Extract slope coefficients (intercept excluded)."""
        beta = np.asarray(model.beta)
        if model.fit_intercept:
            return beta[1:]
        return beta

    def residuals(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Conditional residuals: ``e = y − ŷ``.

        These are marginal residuals (y − Xβ̂).  For the mixed
        model, the "full" residual is y − Xβ̂ − Zû, but for the
        Freedman–Lane permutation scheme we need the marginal
        residual because the random effects are not features being
        tested — they are a nuisance covariance structure.
        """
        return np.asarray(y - model.predictions)

    # ---- Score projection ------------------------------------------

    def score_project(
        self,
        X: np.ndarray,
        feature_idx: int,
        residuals: np.ndarray,
        perm_indices: np.ndarray,
        *,
        fit_intercept: bool = True,
        y: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Score projection via GLS projection matrix A.

        Computes ``projection_A[j] @ residuals[perm_indices]`` for
        all B permutations simultaneously — a single matmul.

        This is mathematically identical to the Freedman–Lane refit
        coefficient (not an approximation).  The projection matrix A
        already incorporates the variance structure V̂⁻¹ from REML
        calibration.
        """
        self._require_calibrated("score_project")
        assert self.projection_A is not None  # for mypy
        j = feature_idx + 1 if fit_intercept else feature_idx
        projection_row = self.projection_A[j]  # (n,)
        E_pi = residuals[perm_indices]  # (B, n)
        return np.asarray(E_pi @ projection_row)  # (B,)

    # ---- Permutation helpers ---------------------------------------

    def reconstruct_y(
        self,
        predictions: np.ndarray,
        permuted_residuals: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Additive reconstruction: ``Y* = ŷ + π(e)``.

        Same as LinearFamily — no stochastic step needed because Y
        is continuous.  *rng* accepted for protocol compatibility.
        """
        return np.asarray(predictions + permuted_residuals)

    def fit_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Residual sum of squares: ``RSS = Σ(yᵢ − ŷᵢ)²``.

        Uses marginal predictions (Xβ̂, not Xβ̂ + Zû) because the
        permutation test is about fixed effects — the same metric
        as LinearFamily.
        """
        resid = y_true - y_pred
        return float(np.sum(resid**2))

    # ---- Scoring (joint test interface) ----------------------------

    def score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """RSS from a fitted LMM (marginal predictions).

        Delegates to ``fit_metric(y, predict(model, X))``.
        """
        return self.fit_metric(y, self.predict(model, X))

    def null_score(self, y: np.ndarray, fit_intercept: bool = True) -> float:
        """RSS of the intercept-only (mean) model.

        Predicts ȳ for all observations — same as LinearFamily.
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
        """LMM diagnostics: marginal/conditional R², ICC, variance components."""
        self._require_calibrated("diagnostics")
        assert self.sigma2 is not None  # for mypy
        assert self.re_covariances is not None

        # Marginal R²: var(Xβ̂) / var(y)
        X_aug = _augment_intercept(X, fit_intercept)
        assert self.projection_A is not None
        beta = self.projection_A @ y
        y_pred = X_aug @ beta
        var_y = float(np.var(y))
        var_fixed = float(np.var(y_pred))

        r2_marginal = var_fixed / var_y if var_y > 0 else 0.0

        # Total random-effect variance (intercept variances only for ICC)
        total_tau2 = sum(float(c[0, 0]) for c in self.re_covariances)

        # Conditional R²: (var(Xβ̂) + Σ τ²_k) / (var(Xβ̂) + Σ τ²_k + σ²)
        r2_conditional = (
            (var_fixed + total_tau2) / (var_fixed + total_tau2 + self.sigma2)
            if (var_fixed + total_tau2 + self.sigma2) > 0
            else 0.0
        )

        icc = (
            total_tau2 / (total_tau2 + self.sigma2)
            if (total_tau2 + self.sigma2) > 0
            else 0.0
        )

        # Variance components: structured per-factor representation
        factors, _ = _extract_variance_components(
            self.re_struct,  # type: ignore[arg-type]
            self.re_covariances,  # type: ignore[arg-type]
        )
        var_comps: dict[str, Any] = {"factors": factors}

        # AIC/BIC from cached statsmodels model, or refit if needed.
        # Note: statsmodels returns nan for AIC/BIC under REML — the
        # display layer handles nan gracefully by showing "N/A".
        aic: float | str = "N/A"
        bic: float | str = "N/A"
        if self._sm_model is not None:
            # Use cached model — no refit needed
            try:
                aic = np.round(float(self._sm_model.aic), 4)
                bic = np.round(float(self._sm_model.bic), 4)
            except Exception:
                pass
        else:
            # JAX path: fall back to statsmodels refit for AIC/BIC
            try:
                import statsmodels.regression.mixed_linear_model as mlm

                if self.Z is not None and self.re_struct is not None:
                    groups_arr = self._groups_arr
                    if groups_arr is None:
                        G_first, d_first = self.re_struct[0]
                        groups_arr = np.argmax(
                            self.Z[:, : G_first * d_first].reshape(
                                -1, G_first, d_first
                            )[:, :, 0],
                            axis=1,
                        )
                    X_sm = _augment_intercept(X, fit_intercept)
                    exog_re_kw: dict[str, Any] = {}
                    if self._exog_re_kw is not None:
                        exog_re_kw = self._exog_re_kw
                    else:
                        G_first, d_first = self.re_struct[0]
                        if d_first > 1:
                            exog_re_cols = [np.ones(len(y))]
                            for s in range(1, d_first):
                                slope_col = np.zeros(len(y))
                                for g in range(G_first):
                                    mask = groups_arr == g
                                    slope_col[mask] = self.Z[mask, g * d_first + s]
                                exog_re_cols.append(slope_col)
                            exog_re_kw["exog_re"] = np.column_stack(exog_re_cols)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        sm_model = mlm.MixedLM(
                            y, X_sm, groups=groups_arr, **exog_re_kw
                        ).fit(reml=True, disp=0)
                    aic = np.round(float(sm_model.aic), 4)
                    bic = np.round(float(sm_model.bic), 4)
            except Exception:
                pass

        # Number of groups (first factor for display)
        n_groups = self.re_struct[0][0] if self.re_struct else 0  # type: ignore[index]

        # Random-effects structure summary for header display
        d_k_first = self.re_struct[0][1] if self.re_struct else 1  # type: ignore[index]
        if d_k_first == 1:
            re_summary = "Random intercept"
        else:
            re_summary = f"Random int. + {d_k_first - 1} slope(s)"

        return {
            "n_observations": len(y),
            "n_features": X.shape[1],
            "n_groups": n_groups,
            "re_summary": re_summary,
            "r_squared_marginal": np.round(r2_marginal, 4),
            "r_squared_conditional": np.round(r2_conditional, 4),
            "icc": np.round(icc, 4),
            "sigma2": np.round(float(self.sigma2), 4),
            "variance_components": var_comps,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "aic": aic,
            "bic": bic,
        }

    def classical_p_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        *,
        robust_se: bool = False,
    ) -> np.ndarray:
        """Approximate Wald t-test p-values via statsmodels MixedLM.

        Falls back to OLS p-values if statsmodels MixedLM fails.
        Returns one p-value per slope coefficient (intercept excluded).

        ``robust_se`` is accepted for protocol compatibility but
        ignored — mixed-model SEs already account for the
        random-effects covariance structure.
        """
        self._require_calibrated("classical_p_values")
        assert self.Z is not None
        assert self.re_struct is not None

        try:
            # Use cached statsmodels model if available
            if self._sm_model is not None:
                pvals = self._sm_model.pvalues
                X_sm = _augment_intercept(X, fit_intercept)
                n_fe = X_sm.shape[1]
                fe_pvals = pvals[:n_fe]
                return (
                    np.asarray(fe_pvals[1:]) if fit_intercept else np.asarray(fe_pvals)
                )

            # JAX path: fall back to statsmodels refit for p-values
            import statsmodels.regression.mixed_linear_model as mlm

            groups_arr = self._groups_arr
            if groups_arr is None:
                G_first, d_first = self.re_struct[0]
                groups_arr = np.argmax(
                    self.Z[:, : G_first * d_first].reshape(-1, G_first, d_first)[
                        :, :, 0
                    ],
                    axis=1,
                )
            X_sm = _augment_intercept(X, fit_intercept)
            exog_re_kw_local: dict[str, Any] = {}
            if self._exog_re_kw is not None:
                exog_re_kw_local = self._exog_re_kw
            else:
                G_first, d_first = self.re_struct[0]
                if d_first > 1:
                    exog_re_cols = [np.ones(len(y))]
                    for s in range(1, d_first):
                        slope_col = np.zeros(len(y))
                        for g in range(G_first):
                            mask = groups_arr == g
                            slope_col[mask] = self.Z[mask, g * d_first + s]
                        exog_re_cols.append(slope_col)
                    exog_re_kw_local["exog_re"] = np.column_stack(exog_re_cols)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                sm_model = mlm.MixedLM(
                    y, X_sm, groups=groups_arr, **exog_re_kw_local
                ).fit(reml=True, disp=0)
            pvals = sm_model.pvalues
            n_fe = X_sm.shape[1]
            fe_pvals = pvals[:n_fe]
            return np.asarray(fe_pvals[1:]) if fit_intercept else np.asarray(fe_pvals)
        except Exception:
            # Fallback: OLS p-values (approximate)
            X_sm = _augment_intercept(X, fit_intercept)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                sm_model = sm.OLS(y, X_sm).fit()
            pvals = sm_model.pvalues[1:] if fit_intercept else sm_model.pvalues
            return np.asarray(pvals)

    # ---- Exchangeability (v0.4.0) ----------------------------------

    def exchangeability_cells(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray | None:
        """Return group labels defining exchangeability cells.

        For LMMs, observations within the same cluster are
        exchangeable under H₀ — but observations across clusters
        are not (they have different random intercepts).

        Returns the group labels from the first (outermost) grouping
        factor, which defines the exchangeability structure.  For
        nested designs, this is the top-level grouping factor.
        """
        if self._groups_arr is not None:
            return self._groups_arr.copy()
        if self.Z is None or self.re_struct is None:
            return None
        # Fallback: derive from Z (should not happen after calibrate())
        G_first, d_first = self.re_struct[0]
        intercept_cols = self.Z[:, : G_first * d_first].reshape(-1, G_first, d_first)[
            :, :, 0
        ]
        return np.asarray(np.argmax(intercept_cols, axis=1))

    # ---- Calibration -----------------------------------------------
    #
    # REML calibration: estimate β̂, σ², τ², and the GLS projection A.
    # The projection A = S⁻¹ X'Ṽ⁻¹ is σ²-free, so the same A works
    # for all permutations.
    #
    # Groups are supplied via **kwargs (the protocol signature accepts
    # ``**kwargs`` for extensibility):
    #
    #   family.calibrate(X, y, fit_intercept, groups=group_vec)
    #
    # The JAX backend is preferred for REML (autodiff + lax.while_loop);
    # if JAX is unavailable, falls back to statsmodels MixedLM + manual
    # Woodbury projection construction.

    def calibrate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Estimate variance components and GLS projection via REML.

        Requires ``groups=`` in kwargs — a 1-D array for a single
        random intercept, or a dict of arrays for g grouping factors.
        Optionally accepts ``random_slopes=`` to specify which X
        columns get correlated random slopes.

        Returns a **new** ``LinearMixedFamily`` with all calibrated
        fields populated.  If already calibrated (``projection_A is
        not None``), returns ``self`` — making this **idempotent**.

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            y: Response vector ``(n,)``.
            fit_intercept: Whether the model includes an intercept.
            **kwargs: Must include ``groups`` (1-D array or dict).
                Optional ``random_slopes`` (list[int] or dict).

        Returns:
            A calibrated ``LinearMixedFamily``.

        Raises:
            ValueError: If ``groups`` is not provided.
        """
        if self.projection_A is not None:
            return self

        groups = kwargs.get("groups")
        if groups is None:
            msg = (
                "LinearMixedFamily.calibrate() requires groups= for "
                "calibration.  Pass a 1-D array of group labels for a "
                "single random intercept, or a dict of arrays for "
                "multiple grouping factors."
            )
            raise ValueError(msg)

        random_slopes = kwargs.get("random_slopes")
        Z, re_struct = _build_random_effects_design(
            groups, X=X, random_slopes=random_slopes
        )

        # Try JAX REML solver first (fastest, autodiff-based)
        try:
            return self._calibrate_jax(
                X, y, Z, re_struct, fit_intercept, raw_groups=groups
            )
        except (ImportError, ModuleNotFoundError):
            pass

        # Fallback: statsmodels MixedLM + manual projection
        return self._calibrate_statsmodels(
            X, y, Z, re_struct, fit_intercept, raw_groups=groups
        )

    def _calibrate_jax(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray,
        re_struct: list[tuple[int, int]],
        fit_intercept: bool,
        raw_groups: Any = None,
    ) -> LinearMixedFamily:
        """REML calibration via the JAX Henderson solver."""
        from ._backends._jax import _reml_solve

        result = _reml_solve(X, Z, y, re_struct, fit_intercept=fit_intercept)

        # Derive integer group labels from Z (same logic as statsmodels path)
        G_first, d_first = re_struct[0]
        intercept_cols = Z[:, : G_first * d_first].reshape(-1, G_first, d_first)[
            :, :, 0
        ]
        groups_arr = np.argmax(intercept_cols, axis=1)

        return LinearMixedFamily(
            re_struct=tuple(re_struct),
            projection_A=result.projection,
            sigma2=result.sigma2,
            re_covariances=result.re_covariances,
            log_chol=result.log_chol,
            Z=Z,
            C22=result.C22,
            converged=result.converged,
            n_iter=result.n_iter,
            _groups_arr=groups_arr,
            _exog_re_kw=None,
            _sm_model=None,
            _raw_groups=raw_groups,
        )

    def _calibrate_statsmodels(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray,
        re_struct: list[tuple[int, int]],
        fit_intercept: bool,
        raw_groups: Any = None,
    ) -> LinearMixedFamily:
        """REML calibration fallback via statsmodels MixedLM.

        Estimates variance components using statsmodels, then builds
        the GLS projection matrix from the Woodbury identity.  This
        path is slower but requires no JAX dependency.
        """
        import statsmodels.regression.mixed_linear_model as mlm

        X_aug = _augment_intercept(X, fit_intercept)

        n, p = X_aug.shape
        q = Z.shape[1]

        # Reconstruct groups from Z for first factor
        G_first, d_first = re_struct[0]
        intercept_cols = Z[:, : G_first * d_first].reshape(-1, G_first, d_first)[
            :, :, 0
        ]
        groups_arr = np.argmax(intercept_cols, axis=1)

        X_sm = _augment_intercept(X, fit_intercept)
        exog_re_kw: dict[str, Any] = {}
        if d_first > 1:
            exog_re_cols = [np.ones(len(y))]
            for s in range(1, d_first):
                slope_col = np.zeros(len(y))
                for g in range(G_first):
                    mask = groups_arr == g
                    slope_col[mask] = Z[mask, g * d_first + s]
                exog_re_cols.append(slope_col)
            exog_re_kw["exog_re"] = np.column_stack(exog_re_cols)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sm_model = mlm.MixedLM(y, X_sm, groups=groups_arr, **exog_re_kw).fit(
                reml=True, disp=0
            )

        # Extract variance components from statsmodels
        sigma2 = float(sm_model.scale)
        cov_re = np.atleast_2d(np.asarray(sm_model.cov_re))

        # Build per-factor covariance matrices
        re_covariances_list: list[np.ndarray] = []
        re_covariances_list.append(cov_re)

        # For additional components beyond what statsmodels can estimate,
        # default to σ² · I
        for k in range(1, len(re_struct)):
            _, d_k = re_struct[k]
            re_covariances_list.append(sigma2 * np.eye(d_k))

        # Build Γ⁻¹ from covariance matrices
        Gamma_inv = np.zeros((q, q))
        factor_offset = 0
        for k in range(len(re_struct)):
            G_k, d_k = re_struct[k]
            cov_k = re_covariances_list[k]
            # Variance ratio: Σ_k / σ²
            ratio_k = cov_k / max(sigma2, 1e-20)
            ratio_inv = np.linalg.solve(ratio_k + 1e-20 * np.eye(d_k), np.eye(d_k))
            block_k = np.kron(np.eye(G_k), ratio_inv)
            size_k = G_k * d_k
            Gamma_inv[
                factor_offset : factor_offset + size_k,
                factor_offset : factor_offset + size_k,
            ] = block_k
            factor_offset += size_k

        # Build log_chol from covariance estimates (for consistency)
        log_chol_parts: list[float] = []
        for cov_k in re_covariances_list:
            ratio_k = cov_k / max(sigma2, 1e-20)
            try:
                L_k = np.linalg.cholesky(ratio_k)
            except np.linalg.LinAlgError:
                d_k = cov_k.shape[0]
                L_k = np.eye(d_k) * np.sqrt(max(ratio_k[0, 0], 1e-20))
            # Extract vech with log diagonal
            theta_parts: list[float] = []
            for i in range(L_k.shape[0]):
                for j in range(i):
                    theta_parts.append(float(L_k[i, j]))
                theta_parts.append(float(np.log(max(L_k[i, i], 1e-20))))
            log_chol_parts.extend(theta_parts)
        log_chol = np.array(log_chol_parts)

        C22 = Z.T @ Z + Gamma_inv

        # Build projection A = S⁻¹ X'Ṽ⁻¹ via the Woodbury identity.
        #
        # Ṽ = I + Z Γ Z' is the marginal covariance scaled by σ².
        # The Woodbury identity lets us invert Ṽ without forming
        # the dense n×n matrix:
        #   Ṽ⁻¹ = I − Z (Z'Z + Γ⁻¹)⁻¹ Z' = I − Z C₂₂⁻¹ Z'
        # where C₂₂ = Z'Z + Γ⁻¹ is only q×q (much smaller than n×n).
        XtZ = X_aug.T @ Z  # (p, q)
        C22_inv_ZtX = np.linalg.solve(C22, XtZ.T)  # (q, p)
        S = X_aug.T @ X_aug - XtZ @ C22_inv_ZtX  # (p, p)  GLS info matrix
        C22_inv_Zt = np.linalg.solve(C22, Z.T)  # (q, n)
        Xt_Vtilde_inv = X_aug.T - XtZ @ C22_inv_Zt  # (p, n)  X'Ṽ⁻¹
        A = np.linalg.solve(S, Xt_Vtilde_inv)  # (p, n)  projection

        return LinearMixedFamily(
            re_struct=tuple(re_struct),
            projection_A=A,
            sigma2=sigma2,
            re_covariances=tuple(re_covariances_list),
            log_chol=log_chol,
            Z=Z,
            C22=C22,
            converged=True,  # statsmodels converged if we got here
            n_iter=0,
            _groups_arr=groups_arr,
            _exog_re_kw=exog_re_kw,
            _sm_model=sm_model,
            _raw_groups=raw_groups,
        )

    # ---- Batch fitting (hot loop) ----------------------------------

    def batch_fit(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch LMM via the active backend.

        Delegates to ``backend.batch_mixed_lm()`` with the
        pre-computed GLS projection A.  The projection is σ²-free
        and constant across all B permutations, so the batch
        operation is a single BLAS-3 matmul — structurally identical
        to ``LinearFamily.batch_fit()``.
        """
        self._require_calibrated("batch_fit")

        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        kwargs.pop("n_jobs", None)  # LMM projection is vectorised
        if backend is None:
            backend = resolve_backend()
        return np.asarray(
            backend.batch_mixed_lm(
                X,
                Y_matrix,
                fit_intercept=fit_intercept,
                projection_A=self.projection_A,
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
        """Batch LMM with per-permutation design matrices.

        Kennedy individual path — each permutation has its own
        design matrix.  Variance components are fixed from
        calibration; only the GLS projection changes.

        Delegates to ``backend.batch_mixed_lm_varying_X()`` with
        the calibrated Z and C₂₂.
        """
        self._require_calibrated("batch_fit_varying_X")

        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()
        if backend.name == "numpy":
            return np.asarray(
                backend.batch_mixed_lm_varying_X(
                    X_batch,
                    y,
                    fit_intercept=fit_intercept,
                    Z=self.Z,
                    C22=self.C22,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            )
        return np.asarray(
            backend.batch_mixed_lm_varying_X(
                X_batch,
                y,
                fit_intercept=fit_intercept,
                Z=self.Z,
                C22=self.C22,
                **kwargs,
            )
        )

    def batch_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch LMM returning ``(coefs, RSS)``.

        Uses the GLS projection A to compute coefficients and RSS
        when `X` matches the calibration dimensions.  Falls back to
        plain OLS when `X` has different dimensions (e.g. reduced
        confounder-only model in joint tests).
        """
        self._require_calibrated("batch_fit_and_score")

        X_aug = _augment_intercept(X, fit_intercept)

        assert self.projection_A is not None
        if X_aug.shape[1] == self.projection_A.shape[0]:
            # GLS path — X matches calibration dimensions.
            coefs = self.batch_fit(X, Y_matrix, fit_intercept, **kwargs)
            full_coefs = Y_matrix @ self.projection_A.T  # (B, p_aug)
        else:
            # OLS fallback — X has different dimensions (e.g. reduced
            # confounder-only model in joint tests).  The GLS
            # projection is valid only for the full design; the
            # reduced model uses unweighted OLS.
            kwargs.pop("n_jobs", None)
            pinv = np.linalg.pinv(X_aug)  # (p_red, n)
            full_coefs = Y_matrix @ pinv.T  # (B, p_red)
            if fit_intercept:
                coefs = full_coefs[:, 1:]  # drop intercept
            else:
                coefs = full_coefs

        preds = full_coefs @ X_aug.T  # (B, n)
        rss = np.sum((Y_matrix - preds) ** 2, axis=1)  # (B,)
        return coefs, rss

    def batch_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch LMM (varying X) returning ``(coefs, RSS)``.

        Each permutation gets its own design matrix, so both the
        projection rebuild and RSS are computed per permutation.
        """
        self._require_calibrated("batch_fit_and_score_varying_X")
        assert self.Z is not None
        assert self.C22 is not None

        from ._backends import resolve_backend

        backend = kwargs.pop("backend", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        if backend is None:
            backend = resolve_backend()

        # Get slope coefficients via batch_fit_varying_X
        coefs = self.batch_fit_varying_X(
            X_batch, y, fit_intercept, backend=backend, n_jobs=n_jobs, **kwargs
        )

        # Compute RSS per permutation
        B = X_batch.shape[0]
        rss = np.empty(B)
        C22_inv_Zt = np.linalg.solve(self.C22, self.Z.T)

        for b in range(B):
            X_b = X_batch[b]
            X_aug = _augment_intercept(X_b, fit_intercept)
            XtZ = X_aug.T @ self.Z
            C22_inv_ZtX = C22_inv_Zt @ X_aug
            S = X_aug.T @ X_aug - XtZ @ C22_inv_ZtX
            Xt_Vtilde_inv = X_aug.T - XtZ @ C22_inv_Zt
            A = np.linalg.solve(S, Xt_Vtilde_inv)
            beta_b = A @ y
            preds_b = X_aug @ beta_b
            rss[b] = float(np.sum((y - preds_b) ** 2))

        return coefs, rss

    def batch_fit_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch LMM where both X and Y vary per replicate.

        Each replicate requires its own projection rebuild AND
        its own β̂ computation.
        """
        self._require_calibrated("batch_fit_paired")
        assert self.Z is not None
        assert self.C22 is not None

        B, n, p = X_batch.shape
        C22_inv_Zt = np.linalg.solve(self.C22, self.Z.T)
        result = np.empty((B, p))

        for b in range(B):
            X_b = X_batch[b]
            y_b = Y_batch[b]
            X_aug = _augment_intercept(X_b, fit_intercept)
            XtZ = X_aug.T @ self.Z
            C22_inv_ZtX = C22_inv_Zt @ X_aug
            S = X_aug.T @ X_aug - XtZ @ C22_inv_ZtX
            Xt_Vtilde_inv = X_aug.T - XtZ @ C22_inv_Zt
            A = np.linalg.solve(S, Xt_Vtilde_inv)
            beta_full = A @ y_b
            result[b] = beta_full[1:] if fit_intercept else beta_full

        return result

    # ---- Internal helpers ------------------------------------------

    def _require_calibrated(self, method: str) -> None:
        """Guard: raise if calibrate() has not been called."""
        _require_calibrated_guard(
            self, method, field="projection_A", family_name="LinearMixedFamily"
        )


# ------------------------------------------------------------------ #
# Fit result for GLMM families
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class _GLMMFitResult:
    """Lightweight container from a single GLMM fit.

    Stores the quantities consumed by the protocol methods
    ``predict()``, ``coefs()``, ``residuals()``.  Unlike
    ``_LMMFitResult``, there is no ``sigma2`` (GLMMs have no
    residual-variance parameter) — predictions are on the
    response scale (probabilities for logistic, rates for
    Poisson).
    """

    beta: np.ndarray  # (p_aug,) — includes intercept if fit_intercept
    u: np.ndarray  # (q,) random-effect BLUPs
    predictions: np.ndarray  # (n,) marginal fitted values (no RE)
    fit_intercept: bool


# ------------------------------------------------------------------ #
# GLMM shared helpers
# ------------------------------------------------------------------ #


class _GLMMBatchStubMixin:
    """Mixin providing ``batch_*`` stubs that reject non-score methods.

    GLMM families (logistic_mixed, poisson_mixed) require
    ``method='score'`` or ``method='score_exact'`` — conventional
    batch-refit strategies are not supported because re-estimating
    variance components per permutation is prohibitively expensive.
    """

    def batch_fit(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Not supported — use ``method='score'``."""
        raise NotImplementedError(
            "GLMM families require method='score'. "
            "Use permutation_test_regression(..., method='score')."
        )

    def batch_fit_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Not supported — use ``method='score'``."""
        raise NotImplementedError(
            "GLMM families require method='score'. "
            "Use permutation_test_regression(..., method='score')."
        )

    def batch_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Not supported — use ``method='score'``."""
        raise NotImplementedError(
            "GLMM families require method='score'. "
            "Use permutation_test_regression(..., method='score')."
        )

    def batch_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Not supported — use ``method='score'``."""
        raise NotImplementedError(
            "GLMM families require method='score'. "
            "Use permutation_test_regression(..., method='score')."
        )

    def batch_fit_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Not supported — use ``method='score'``."""
        raise NotImplementedError(
            "GLMM families require method='score'. "
            "Use permutation_test_regression(..., method='score')."
        )


def _calibrate_glmm(
    cls: type,
    instance: LogisticMixedFamily | PoissonMixedFamily,
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool,
    family_name: str,
    **kwargs: Any,
) -> LogisticMixedFamily | PoissonMixedFamily:
    """Shared Laplace-approximation calibration for GLMM families.

    Parameters
    ----------
    cls : type
        The concrete family class to construct (``LogisticMixedFamily``
        or ``PoissonMixedFamily``).
    instance : LogisticMixedFamily | PoissonMixedFamily
        The uncalibrated instance (used for the idempotency guard).
    X, y : np.ndarray
        Design matrix and response vector.
    fit_intercept : bool
        Whether the model includes an intercept.
    family_name : str
        ``"logistic"`` or ``"poisson"`` — selects the conditional NLL
        and working-response functions from ``_backends._jax``.
    **kwargs
        Must include ``groups=``; may include ``random_slopes=``.

    Returns
    -------
    LogisticMixedFamily | PoissonMixedFamily
        A **new** calibrated instance with all Laplace fields populated.
    """
    # Idempotent — return immediately if already calibrated.
    if instance.fisher_info is not None:
        return instance

    groups = kwargs.get("groups")
    if groups is None:
        msg = f"{cls.__name__}.calibrate() requires groups= for calibration."
        raise ValueError(msg)

    random_slopes = kwargs.get("random_slopes")
    Z, re_struct = _build_random_effects_design(
        groups, X=X, random_slopes=random_slopes
    )

    # Lazy-import the family-specific functions from the JAX backend.
    import importlib

    jax_mod = importlib.import_module("._backends._jax", package=__package__)
    _laplace_solve = jax_mod._laplace_solve
    working_fn = getattr(jax_mod, f"_{family_name}_working_response_and_weights")
    cond_nll_fn = getattr(jax_mod, f"_{family_name}_conditional_nll")

    result = _laplace_solve(
        X,
        Z,
        y,
        re_struct,
        working_fn=working_fn,
        cond_nll_fn=cond_nll_fn,
        family=family_name,
        fit_intercept=fit_intercept,
    )

    # Derive group labels from Z intercept columns.
    G_first, d_first = re_struct[0]
    intercept_cols = Z[:, : G_first * d_first].reshape(-1, G_first, d_first)[:, :, 0]
    groups_arr = np.argmax(intercept_cols, axis=1)

    return cls(  # type: ignore[no-any-return]
        re_struct=tuple(re_struct),
        beta=result.beta,
        u=result.u,
        W=result.W,
        mu=result.mu,
        V_inv_diag=result.V_inv_diag,
        fisher_info=result.fisher_info,
        re_covariances=result.re_covariances,
        log_chol=result.log_chol,
        Z=Z,
        C22=result.C22,
        converged=result.converged,
        n_iter=result.n_iter_outer,
        nll=result.nll,
        _groups_arr=groups_arr,
        _raw_groups=groups,
    )


# ------------------------------------------------------------------ #
# LogisticMixedFamily
# ------------------------------------------------------------------ #


@final
@dataclass(frozen=True)
class LogisticMixedFamily(_GLMMBatchStubMixin):
    """Logistic mixed-effects model for clustered binary outcomes.

    Implements the ``ModelFamily`` protocol for binary Y ∈ {0, 1}
    with random-intercept and/or random-slope structure.

    Model (Laplace approximation):

        y_i | u ~ Bernoulli(μ_i),   logit(μ_i) = x_i'β + z_i'u,
        u ~ N(0, Γ(θ))

    The variance components θ are estimated once during
    ``calibrate()`` via the Laplace marginal NLL (JAX autodiff +
    Newton on θ, unrolled IRLS inner loop on (β, u)).

    Permutation testing uses the **one-step corrector** (Le Cam
    estimator) — no per-permutation IRLS refit.  The key quantities
    are:

    * ``V_inv_diag`` — diagonal of the approximate marginal
      precision ``V⁻¹ = W − W Z C₂₂⁻¹ Z' W``
    * ``fisher_info`` — Fisher information ``I = X' V⁻¹ X``
      (Schur complement from the last Henderson solve)

    The one-step score for feature j under permutation π is::

        β̂_j^(1) = U_{j,π} / I_{jj}
        where  U_{j,π} = (x_j ⊙ V⁻¹_diag)' e_π

    Construction fields
    -------------------
    All fields are ``None`` until ``calibrate()`` populates them.
    """

    re_struct: tuple[tuple[int, int], ...] | None = None
    beta: np.ndarray | None = None
    u: np.ndarray | None = None
    W: np.ndarray | None = None
    mu: np.ndarray | None = None
    V_inv_diag: np.ndarray | None = None
    fisher_info: np.ndarray | None = None
    re_covariances: tuple[np.ndarray, ...] | None = None
    log_chol: np.ndarray | None = None
    Z: np.ndarray | None = None
    C22: np.ndarray | None = None
    converged: bool | None = None
    n_iter: int | None = None
    nll: float | None = None
    _groups_arr: np.ndarray | None = None
    _raw_groups: Any = None

    # ---- Protocol properties ---------------------------------------

    @property
    def name(self) -> str:
        return "logistic_mixed"

    @property
    def residual_type(self) -> str:
        return "deviance"

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
        """Header rows for logistic mixed results table."""
        icc = diagnostics.get("icc")
        icc_str = f"{icc:.4f}" if icc is not None else "N/A"
        n_groups = diagnostics.get("n_groups", "N/A")
        re_summary = diagnostics.get("re_summary", "N/A")
        conv = diagnostics.get("converged")
        conv_str = "Yes" if conv is True else ("No" if conv is False else "N/A")
        deviance = diagnostics.get("deviance")
        dev_str = f"{deviance:.4f}" if deviance is not None else "N/A"

        return [
            (
                "RE Structure:",
                str(re_summary),
                "Deviance:",
                dev_str,
            ),
            (
                "No. Groups:",
                str(n_groups),
                "ICC:",
                icc_str,
            ),
            (
                "Laplace converged:",
                conv_str,
                "",
                "",
            ),
        ]

    def display_diagnostics(
        self,
        diagnostics: dict[str, Any],
    ) -> tuple[list[tuple[str, str, str]], list[str]]:
        """Variance component lines for logistic GLMM."""
        lines: list[tuple[str, str, str]] = []
        notes: list[str] = []
        glmm = diagnostics.get("glmm_gof", {})
        if not glmm:
            return lines, notes

        vc = glmm.get("variance_components", {})
        lines.extend(_format_variance_components(vc))

        icc = glmm.get("icc")
        if icc is not None:
            lines.append(("ICC:", f"{icc:.4f}", ""))

        conv = glmm.get("converged")
        if conv is not None and not conv:
            notes.append(
                "Laplace solver did not converge — variance "
                "component estimates may be unreliable."
            )
        return lines, notes

    def compute_extended_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
    ) -> dict[str, Any]:
        """Structured variance component diagnostics."""
        if self.re_covariances is None or self.re_struct is None:
            return {}

        factors, total_tau2 = _extract_variance_components(
            self.re_struct, self.re_covariances
        )
        # ICC on the latent scale: τ² / (τ² + π²/3)
        #
        # The level-1 residual variance for the logistic GLMM is
        # π²/3 ≈ 3.29 — the variance of the standard logistic
        # distribution, which serves as the implicit residual
        # distribution on the latent (linear predictor) scale.
        # Ref: Snijders & Bosker (2012), *Multilevel Analysis*,
        # §17.2, Eq. 17.5.
        icc = total_tau2 / (total_tau2 + np.pi**2 / 3.0) if total_tau2 > 0 else 0.0

        return {
            "glmm_gof": {
                "variance_components": {"factors": factors},
                "icc": icc,
                "converged": self.converged,
                "n_iter": self.n_iter,
            }
        }

    # ---- Validation ------------------------------------------------

    def validate_y(self, y: np.ndarray) -> None:
        """Assert binary Y ∈ {0, 1}."""
        unique = np.unique(y)
        if not (
            np.array_equal(unique, [0.0, 1.0])
            or np.array_equal(unique, [0, 1])
            or np.array_equal(unique, [0.0])
            or np.array_equal(unique, [1.0])
        ):
            msg = (
                "LogisticMixedFamily requires binary Y ∈ {0, 1}. "
                f"Got unique values: {unique[:10]}"
            )
            raise ValueError(msg)

    # ---- Single-model operations -----------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> Any:
        """Return fitted model quantities.

        When *X* matches calibration dimensions, uses stored β̂
        from the Laplace solver (fast path).  Otherwise falls back
        to a fixed-effects GLM for the reduced model (needed by
        the Freedman–Lane residual step in the score strategy).
        """
        self._require_calibrated("fit")
        assert self.beta is not None

        X_aug = _augment_intercept(X, fit_intercept)

        p_aug = X_aug.shape[1]
        if p_aug == len(self.beta):
            # Fast path: X matches calibration
            eta = X_aug @ self.beta
            mu = 1.0 / (1.0 + np.exp(-eta))
            return _GLMMFitResult(
                beta=self.beta,
                u=self.u if self.u is not None else np.zeros(0),
                predictions=mu,
                fit_intercept=fit_intercept,
            )

        # Reduced X: fit a fixed-effects GLM (ignoring RE).
        # Variance structure is already encoded in V_inv_diag
        # for the score projection — this is just for residuals.
        from ._backends._jax import (
            _fit_glm_irls,
            _logistic_nll_np,
        )
        from ._backends._jax import (
            _logistic_working_response_and_weights as _logistic_wf,
        )

        X_aug_red = _augment_intercept(X, fit_intercept)
        glm_res = _fit_glm_irls(
            X_aug_red,
            y,
            _logistic_wf,
            _logistic_nll_np,
            family="logistic",
        )
        return _GLMMFitResult(
            beta=glm_res.beta,
            u=np.zeros(0),
            predictions=glm_res.mu,
            fit_intercept=fit_intercept,
        )

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Marginal predictions (probabilities, no RE)."""
        return np.asarray(model.predictions)

    def coefs(self, model: Any) -> np.ndarray:
        """Slope coefficients (intercept excluded)."""
        beta = np.asarray(model.beta)
        return beta[1:] if model.fit_intercept else beta

    def residuals(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Response-scale marginal residuals: ``y − μ̂``."""
        return np.asarray(y - model.predictions)

    # ---- Score projection ------------------------------------------

    def score_project(
        self,
        X: np.ndarray,
        feature_idx: int,
        residuals: np.ndarray,
        perm_indices: np.ndarray,
        *,
        fit_intercept: bool = True,
        y: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """One-step corrector via score projection.

        Computes the Le Cam estimator for feature *j* across all B
        permutations in a single matmul — O(n·B), no IRLS.

        .. math::
            \\hat\\beta_{j,\\pi}^{(1)}
            = \\frac{U_{j,\\pi}}{\\mathcal{I}_{jj}}
            = \\frac{(x_j \\odot V^{-1}_{\\mathrm{diag}})' e_\\pi}
                   {\\mathcal{I}_{jj}}
        """
        self._require_calibrated("score_project")
        assert self.V_inv_diag is not None
        assert self.fisher_info is not None

        j = feature_idx + 1 if fit_intercept else feature_idx
        X_full = _augment_intercept(X, fit_intercept)

        score_weights = X_full[:, j] * self.V_inv_diag  # (n,)
        E_pi = residuals[perm_indices]  # (B, n)
        U_j = E_pi @ score_weights  # (B,)
        # Full Fisher inverse [I⁻¹]_{jj} accounts for cross-correlations.
        try:
            fisher_inv_jj = np.linalg.inv(self.fisher_info)[j, j]
        except np.linalg.LinAlgError:
            fisher_inv_jj = np.linalg.pinv(self.fisher_info)[j, j]
        return np.asarray(U_j * fisher_inv_jj)  # (B,)

    # ---- Permutation helpers ---------------------------------------

    def reconstruct_y(
        self,
        predictions: np.ndarray,
        permuted_residuals: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Bernoulli reconstruction: ``Y* ~ Bern(clip(ŷ + π(e)))``."""
        probabilities = np.clip(predictions + permuted_residuals, 1e-10, 1 - 1e-10)
        result: np.ndarray = (rng.random(probabilities.shape) < probabilities).astype(
            float
        )
        return result

    def fit_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Deviance: ``−2 Σ [y log(p) + (1−y) log(1−p)]``."""
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-2.0 * np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Deviance of the fitted model."""
        return self.fit_metric(y, self.predict(model, X))

    def null_score(self, y: np.ndarray, fit_intercept: bool = True) -> float:
        """Deviance of the intercept-only (null) model."""
        if fit_intercept:
            p_bar = np.clip(float(np.mean(y)), 1e-15, 1 - 1e-15)
            preds = np.full(len(y), p_bar)
        else:
            preds = np.full(len(y), 0.5)
        return self.fit_metric(y, preds)

    # ---- Classical inference & diagnostics -------------------------

    def diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> dict[str, Any]:
        """GLMM diagnostics: variance components, ICC, deviance."""
        self._require_calibrated("diagnostics")
        assert self.beta is not None
        assert self.re_covariances is not None
        assert self.re_struct is not None

        X_aug = _augment_intercept(X, fit_intercept)

        # Marginal predictions
        eta = X_aug @ self.beta
        mu = 1.0 / (1.0 + np.exp(-eta))

        # Deviance
        p_clip = np.clip(mu, 1e-15, 1 - 1e-15)
        deviance = float(
            -2.0 * np.sum(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))
        )

        # Variance components
        factors, total_tau2 = _extract_variance_components(
            self.re_struct, self.re_covariances
        )
        # ICC on the latent scale: τ² / (τ² + π²/3)
        #
        # The level-1 residual variance for the logistic GLMM is
        # π²/3 ≈ 3.29 — the variance of the standard logistic
        # distribution, which serves as the implicit residual
        # distribution on the latent (linear predictor) scale.
        # Ref: Snijders & Bosker (2012), *Multilevel Analysis*,
        # §17.2, Eq. 17.5.
        icc = total_tau2 / (total_tau2 + np.pi**2 / 3.0) if total_tau2 > 0 else 0.0

        n_groups = self.re_struct[0][0]
        d_k_first = self.re_struct[0][1]
        re_summary = (
            "Random intercept"
            if d_k_first == 1
            else f"Random int. + {d_k_first - 1} slope(s)"
        )

        return {
            "n_observations": len(y),
            "n_features": X.shape[1],
            "n_groups": n_groups,
            "re_summary": re_summary,
            # Marginal deviance (fixed effects only, excludes BLUPs Zb̂).
            "deviance": np.round(deviance, 4),
            "deviance_note": "marginal (fixed-effects only)",
            "icc": np.round(icc, 4),
            "variance_components": {"factors": factors},
            "converged": self.converged,
            "n_iter": self.n_iter,
            "glmm_gof": {
                "variance_components": {"factors": factors},
                "icc": icc,
                "converged": self.converged,
                "n_iter": self.n_iter,
            },
        }

    def classical_p_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        *,
        robust_se: bool = False,
    ) -> np.ndarray:
        """Wald z-test p-values from the Fisher information.

        Computes the Wald statistic z_j = β̂_j / se(β̂_j) where
        se is derived from the inverse Fisher information matrix.

        ``robust_se`` is accepted for protocol compatibility but
        ignored — GLMM SEs already account for the random-effects
        covariance structure.
        """
        self._require_calibrated("classical_p_values")
        assert self.beta is not None
        assert self.fisher_info is not None

        from scipy.stats import norm

        # Fisher info inverse gives asymptotic covariance
        try:
            cov_beta = np.linalg.inv(self.fisher_info)
        except np.linalg.LinAlgError:
            cov_beta = np.linalg.pinv(self.fisher_info)

        se = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
        se = np.where(se < 1e-15, 1e-15, se)
        z = self.beta / se
        pvals = 2.0 * norm.sf(np.abs(z))
        return np.asarray(pvals[1:]) if fit_intercept else np.asarray(pvals)

    # ---- Exchangeability -------------------------------------------

    def exchangeability_cells(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray | None:
        """Return group labels for within-cluster exchangeability."""
        if self._groups_arr is not None:
            return self._groups_arr.copy()
        if self.Z is None or self.re_struct is None:
            return None
        G_first, d_first = self.re_struct[0]
        intercept_cols = self.Z[:, : G_first * d_first].reshape(-1, G_first, d_first)[
            :, :, 0
        ]
        return np.asarray(np.argmax(intercept_cols, axis=1))

    # ---- Calibration -----------------------------------------------

    def calibrate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Estimate variance components via Laplace approximation.

        Requires ``groups=`` in kwargs plus JAX.  Returns a **new**
        ``LogisticMixedFamily`` with all calibrated fields populated.
        Idempotent — returns ``self`` if already calibrated.
        """
        return _calibrate_glmm(  # type: ignore[return-value]
            LogisticMixedFamily, self, X, y, fit_intercept, "logistic", **kwargs
        )

    # ---- Internal --------------------------------------------------

    def _require_calibrated(self, method: str) -> None:
        """Guard: raise if calibrate() has not been called."""
        _require_calibrated_guard(
            self, method, field="fisher_info", family_name="LogisticMixedFamily"
        )


# ------------------------------------------------------------------ #
# PoissonMixedFamily
# ------------------------------------------------------------------ #


@final
@dataclass(frozen=True)
class PoissonMixedFamily(_GLMMBatchStubMixin):
    """Poisson mixed-effects model for clustered count outcomes.

    Implements the ``ModelFamily`` protocol for non-negative
    integer outcomes with random-effect structure.

    Model (Laplace approximation):

        y_i | u ~ Poisson(μ_i),   log(μ_i) = x_i'β + z_i'u,
        u ~ N(0, Γ(θ))

    Architecture mirrors ``LogisticMixedFamily`` — Laplace GLMM
    calibration via JAX, one-step corrector for permutation
    testing.  See ``LogisticMixedFamily`` docstring for details.
    """

    re_struct: tuple[tuple[int, int], ...] | None = None
    beta: np.ndarray | None = None
    u: np.ndarray | None = None
    W: np.ndarray | None = None
    mu: np.ndarray | None = None
    V_inv_diag: np.ndarray | None = None
    fisher_info: np.ndarray | None = None
    re_covariances: tuple[np.ndarray, ...] | None = None
    log_chol: np.ndarray | None = None
    Z: np.ndarray | None = None
    C22: np.ndarray | None = None
    converged: bool | None = None
    n_iter: int | None = None
    nll: float | None = None
    _groups_arr: np.ndarray | None = None
    _raw_groups: Any = None

    # ---- Protocol properties ---------------------------------------

    @property
    def name(self) -> str:
        return "poisson_mixed"

    @property
    def residual_type(self) -> str:
        return "deviance"

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
        """Header rows for Poisson mixed results table."""
        icc = diagnostics.get("icc")
        icc_str = f"{icc:.4f}" if icc is not None else "N/A"
        n_groups = diagnostics.get("n_groups", "N/A")
        re_summary = diagnostics.get("re_summary", "N/A")
        conv = diagnostics.get("converged")
        conv_str = "Yes" if conv is True else ("No" if conv is False else "N/A")
        deviance = diagnostics.get("deviance")
        dev_str = f"{deviance:.4f}" if deviance is not None else "N/A"

        return [
            (
                "RE Structure:",
                str(re_summary),
                "Deviance:",
                dev_str,
            ),
            (
                "No. Groups:",
                str(n_groups),
                "ICC:",
                icc_str,
            ),
            (
                "Laplace converged:",
                conv_str,
                "",
                "",
            ),
        ]

    def display_diagnostics(
        self,
        diagnostics: dict[str, Any],
    ) -> tuple[list[tuple[str, str, str]], list[str]]:
        """Variance component lines for Poisson GLMM."""
        lines: list[tuple[str, str, str]] = []
        notes: list[str] = []
        glmm = diagnostics.get("glmm_gof", {})
        if not glmm:
            return lines, notes

        vc = glmm.get("variance_components", {})
        lines.extend(_format_variance_components(vc))

        icc = glmm.get("icc")
        if icc is not None:
            lines.append(("ICC:", f"{icc:.4f}", ""))

        disp = glmm.get("dispersion")
        if disp is not None:
            lines.append(("Dispersion:", f"{disp:.4f}", ""))
            if disp > 1.5:
                notes.append(
                    f"Dispersion = {disp:.4f}: overdispersion "
                    f"detected. Consider family='negative_binomial'."
                )

        conv = glmm.get("converged")
        if conv is not None and not conv:
            notes.append(
                "Laplace solver did not converge — variance "
                "component estimates may be unreliable."
            )
        return lines, notes

    def compute_extended_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
    ) -> dict[str, Any]:
        """Structured variance component diagnostics."""
        if self.re_covariances is None or self.re_struct is None:
            return {}

        factors, total_tau2 = _extract_variance_components(
            self.re_struct, self.re_covariances
        )
        # Poisson ICC on the log scale: τ² / (τ² + 1)
        #
        # Under the log-normal approximation the level-1 variance
        # on the link (log) scale is 1.0, so the ICC denominator
        # is simply τ² + 1.  This is the "latent-variable" ICC
        # for count outcomes.
        # Ref: Goldstein, Browne & Rasbash (2002), "Partitioning
        # variation in multilevel models", *Understanding
        # Statistics*, 1(4), 223–231.
        icc = total_tau2 / (total_tau2 + 1.0) if total_tau2 > 0 else 0.0

        # Pearson dispersion (overdispersion diagnostic)
        assert self.beta is not None
        X_aug = _augment_intercept(X, fit_intercept)
        eta = X_aug @ self.beta
        mu = np.exp(np.clip(eta, -20.0, 20.0))
        pearson_chi2 = float(np.sum((y - mu) ** 2 / np.maximum(mu, 1e-15)))
        df_resid = max(len(y) - X_aug.shape[1], 1)
        dispersion = pearson_chi2 / df_resid

        return {
            "glmm_gof": {
                "variance_components": {"factors": factors},
                "icc": icc,
                "dispersion": dispersion,
                "converged": self.converged,
                "n_iter": self.n_iter,
            }
        }

    # ---- Validation ------------------------------------------------

    def validate_y(self, y: np.ndarray) -> None:
        """Assert non-negative count Y."""
        if not np.issubdtype(y.dtype, np.number):
            msg = "PoissonMixedFamily requires numeric Y."
            raise ValueError(msg)
        if np.any(y < 0):
            msg = "PoissonMixedFamily requires non-negative Y."
            raise ValueError(msg)
        # Check integer-valued (allow float representation)
        if not np.allclose(y, np.round(y)):
            msg = "PoissonMixedFamily requires integer-valued Y."
            raise ValueError(msg)

    # ---- Single-model operations -----------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> Any:
        """Fit the Poisson GLMM (or fallback GLM for reduced X)."""
        self._require_calibrated("fit")
        assert self.beta is not None

        X_aug = _augment_intercept(X, fit_intercept)

        p_aug = X_aug.shape[1]
        if p_aug == len(self.beta):
            eta = X_aug @ self.beta
            mu = np.exp(np.clip(eta, -20.0, 20.0))
            return _GLMMFitResult(
                beta=self.beta,
                u=self.u if self.u is not None else np.zeros(0),
                predictions=mu,
                fit_intercept=fit_intercept,
            )

        # Reduced X: fixed-effects GLM fallback (pure internal IRLS)
        from ._backends._jax import (
            _fit_glm_irls,
            _poisson_nll_np,
        )
        from ._backends._jax import (
            _poisson_working_response_and_weights as _poisson_wf,
        )

        X_aug_red = _augment_intercept(X, fit_intercept)
        glm_res = _fit_glm_irls(
            X_aug_red,
            y,
            _poisson_wf,
            _poisson_nll_np,
            family="poisson",
        )
        return _GLMMFitResult(
            beta=glm_res.beta,
            u=np.zeros(0),
            predictions=glm_res.mu,
            fit_intercept=fit_intercept,
        )

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Marginal predictions (rates, no RE)."""
        return np.asarray(model.predictions)

    def coefs(self, model: Any) -> np.ndarray:
        """Slope coefficients (intercept excluded)."""
        beta = np.asarray(model.beta)
        return beta[1:] if model.fit_intercept else beta

    def residuals(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Response-scale marginal residuals: ``y − μ̂``."""
        return np.asarray(y - model.predictions)

    # ---- Score projection ------------------------------------------

    def score_project(
        self,
        X: np.ndarray,
        feature_idx: int,
        residuals: np.ndarray,
        perm_indices: np.ndarray,
        *,
        fit_intercept: bool = True,
        y: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """One-step corrector via score projection.

        Same formula as ``LogisticMixedFamily`` — the family-
        specific information is already encoded in ``V_inv_diag``
        and ``fisher_info`` from calibration.
        """
        self._require_calibrated("score_project")
        assert self.V_inv_diag is not None
        assert self.fisher_info is not None

        j = feature_idx + 1 if fit_intercept else feature_idx
        X_full = _augment_intercept(X, fit_intercept)

        score_weights = X_full[:, j] * self.V_inv_diag  # (n,)
        E_pi = residuals[perm_indices]  # (B, n)
        U_j = E_pi @ score_weights  # (B,)
        # Full Fisher inverse [I⁻¹]_{jj} accounts for cross-correlations.
        try:
            fisher_inv_jj = np.linalg.inv(self.fisher_info)[j, j]
        except np.linalg.LinAlgError:
            fisher_inv_jj = np.linalg.pinv(self.fisher_info)[j, j]
        return np.asarray(U_j * fisher_inv_jj)  # (B,)

    # ---- Permutation helpers ---------------------------------------

    def reconstruct_y(
        self,
        predictions: np.ndarray,
        permuted_residuals: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Poisson sampling reconstruction.

        Reconstructed rate ``μ* = max(ŷ + π(e), ε)`` is used as
        the Poisson rate for sampling.
        """
        rates = np.maximum(predictions + permuted_residuals, 1e-10)
        result: np.ndarray = rng.poisson(rates).astype(float)
        return result

    def fit_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Poisson deviance: ``2 Σ [y log(y/μ) − (y − μ)]``."""
        mu = np.maximum(y_pred, 1e-15)
        y_safe = np.maximum(y_true, 0.0)
        # y log(y/μ) is 0 when y = 0 (convention 0·log(0) = 0).
        # Guard the log argument to avoid RuntimeWarning from NumPy
        # evaluating both branches of np.where eagerly.
        log_ratio = np.log(np.maximum(y_safe / mu, 1e-300))
        ratio = np.where(y_safe > 0, y_safe * log_ratio, 0.0)
        return float(2.0 * np.sum(ratio - (y_safe - mu)))

    def score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Poisson deviance of the fitted model."""
        return self.fit_metric(y, self.predict(model, X))

    def null_score(self, y: np.ndarray, fit_intercept: bool = True) -> float:
        """Deviance of the intercept-only (null) model."""
        if fit_intercept:
            mu_bar = max(float(np.mean(y)), 1e-15)
            preds = np.full(len(y), mu_bar)
        else:
            preds = np.ones(len(y))
        return self.fit_metric(y, preds)

    # ---- Classical inference & diagnostics -------------------------

    def diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
    ) -> dict[str, Any]:
        """GLMM diagnostics: variance components, ICC, deviance."""
        self._require_calibrated("diagnostics")
        assert self.beta is not None
        assert self.re_covariances is not None
        assert self.re_struct is not None

        X_aug = _augment_intercept(X, fit_intercept)

        # Marginal predictions
        eta = X_aug @ self.beta
        mu = np.exp(np.clip(eta, -20.0, 20.0))

        # Deviance
        y_safe = np.maximum(y, 0.0)
        mu_safe = np.maximum(mu, 1e-15)
        log_ratio = np.log(np.maximum(y_safe / mu_safe, 1e-300))
        ratio = np.where(y_safe > 0, y_safe * log_ratio, 0.0)
        deviance = float(2.0 * np.sum(ratio - (y_safe - mu_safe)))

        # Pearson dispersion
        pearson_chi2 = float(np.sum((y - mu) ** 2 / np.maximum(mu, 1e-15)))
        df_resid = max(len(y) - X_aug.shape[1], 1)
        dispersion = pearson_chi2 / df_resid

        # Variance components
        factors, total_tau2 = _extract_variance_components(
            self.re_struct, self.re_covariances
        )
        # Poisson ICC on the log scale: τ² / (τ² + 1)
        #
        # Level-1 residual variance on the log scale is 1.0 under
        # the log-normal approximation (see Goldstein et al. 2002).
        icc = total_tau2 / (total_tau2 + 1.0) if total_tau2 > 0 else 0.0

        n_groups = self.re_struct[0][0]
        d_k_first = self.re_struct[0][1]
        re_summary = (
            "Random intercept"
            if d_k_first == 1
            else f"Random int. + {d_k_first - 1} slope(s)"
        )

        return {
            "n_observations": len(y),
            "n_features": X.shape[1],
            "n_groups": n_groups,
            "re_summary": re_summary,
            # Marginal deviance (fixed effects only, excludes BLUPs Zb̂).
            "deviance": np.round(deviance, 4),
            "deviance_note": "marginal (fixed-effects only)",
            "dispersion": np.round(dispersion, 4),
            "icc": np.round(icc, 4),
            "variance_components": {"factors": factors},
            "converged": self.converged,
            "n_iter": self.n_iter,
            "glmm_gof": {
                "variance_components": {"factors": factors},
                "icc": icc,
                "dispersion": dispersion,
                "converged": self.converged,
                "n_iter": self.n_iter,
            },
        }

    def classical_p_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        *,
        robust_se: bool = False,
    ) -> np.ndarray:
        """Wald z-test p-values from the Fisher information.

        ``robust_se`` is accepted for protocol compatibility but
        ignored — GLMM SEs already account for the random-effects
        covariance structure.
        """
        self._require_calibrated("classical_p_values")
        assert self.beta is not None
        assert self.fisher_info is not None

        from scipy.stats import norm

        try:
            cov_beta = np.linalg.inv(self.fisher_info)
        except np.linalg.LinAlgError:
            cov_beta = np.linalg.pinv(self.fisher_info)

        se = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
        se = np.where(se < 1e-15, 1e-15, se)
        z = self.beta / se
        pvals = 2.0 * norm.sf(np.abs(z))
        return np.asarray(pvals[1:]) if fit_intercept else np.asarray(pvals)

    # ---- Exchangeability -------------------------------------------

    def exchangeability_cells(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray | None:
        """Return group labels for within-cluster exchangeability."""
        if self._groups_arr is not None:
            return self._groups_arr.copy()
        if self.Z is None or self.re_struct is None:
            return None
        G_first, d_first = self.re_struct[0]
        intercept_cols = self.Z[:, : G_first * d_first].reshape(-1, G_first, d_first)[
            :, :, 0
        ]
        return np.asarray(np.argmax(intercept_cols, axis=1))

    # ---- Calibration -----------------------------------------------

    def calibrate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Estimate variance components via Laplace approximation.

        Requires ``groups=`` in kwargs plus JAX.  Returns a **new**
        ``PoissonMixedFamily`` with all calibrated fields populated.
        Idempotent — returns ``self`` if already calibrated.
        """
        return _calibrate_glmm(  # type: ignore[return-value]
            PoissonMixedFamily, self, X, y, fit_intercept, "poisson", **kwargs
        )

    # ---- Internal --------------------------------------------------

    def _require_calibrated(self, method: str) -> None:
        """Guard: raise if calibrate() has not been called."""
        _require_calibrated_guard(
            self, method, field="fisher_info", family_name="PoissonMixedFamily"
        )


# ------------------------------------------------------------------ #
# Registry
# ------------------------------------------------------------------ #

from .families import register_family  # noqa: E402

register_family("linear_mixed", LinearMixedFamily)
register_family("logistic_mixed", LogisticMixedFamily)
register_family("poisson_mixed", PoissonMixedFamily)

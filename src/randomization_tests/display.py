"""Formatted ASCII table display utilities for permutation test results.

These tables mirror the statsmodels summary style, presenting both the
model diagnostics (top panel) and the per-feature coefficients with
empirical (permutation) and classical (asymptotic) p-values side by
side (bottom panel).

Having both p-values in the same table makes it easy to spot
discrepancies — cases where the permutation and asymptotic p-values
diverge (common with small samples, non-normal residuals, or heavy-
tailed error distributions) are precisely the situations where the
permutation test adds value.
"""

from __future__ import annotations

import math
import textwrap
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats as _sp_stats

from ._context import FitContext
from .families import ModelFamily, _fmt_p, resolve_family
from .families_mixed import _format_variance_components

if TYPE_CHECKING:
    from ._results import IndividualTestResult, JointTestResult


def _truncate(name: str, max_len: int) -> str:
    """Truncate *name* to *max_len*, appending ``'...'`` if needed."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def _fmt_coef(value: float, width: int) -> str:
    """Format a coefficient for display.

    Uses fixed notation (4 dp) for values with absolute magnitude
    below 1 000, and scientific notation (2 dp) for larger values
    so that columns never overflow.
    """
    if abs(value) >= 1000:
        return f"{value:>{width}.2e}"
    return f"{value:>{width}.4f}"


def _fmt_diag_val(val: object) -> str:
    """Format a diagnostic value for display.

    Converts ``nan`` floats and ``None`` to ``'N/A'``.  Leaves
    strings and other values as-is via ``str()``.
    """
    if val is None:
        return "N/A"
    if isinstance(val, float) and (val != val):  # nan check
        return "N/A"
    return str(val)


def _wrap(text: str, width: int = 80, indent: int = 2) -> str:
    """Word-wrap *text* to *width*, indenting continuation lines.

    Unlike ``textwrap.fill``, this keeps the first line unindented
    (the caller typically supplies its own prefix) and indents only
    the continuation lines by *indent* spaces.
    """
    return textwrap.fill(
        text,
        width=width,
        initial_indent="",
        subsequent_indent=" " * indent,
    )


def _significance_marker(
    ci_lo: float,
    ci_hi: float,
    thresholds: list[float],
) -> str:
    """Return ``' [!]'`` when the CI straddles any threshold.

    A CI *straddles* a threshold when the lower bound is strictly
    below it and the upper bound is strictly above it, meaning the
    data are consistent with the p-value falling on either side.

    Args:
        ci_lo: Lower bound of the Clopper-Pearson CI.
        ci_hi: Upper bound of the Clopper-Pearson CI.
        thresholds: Significance thresholds to check (e.g.
            ``[0.05, 0.01, 0.001]``).

    Returns:
        ``'  [!]'`` if the CI straddles any threshold, else ``''``.
    """
    for t in thresholds:
        if ci_lo < t < ci_hi:
            return "  [!]"
    return ""


def _recommend_n_randomizations(
    p_hat: float,
    threshold: float,
    alpha: float = 0.05,
    n_randomizations: int | None = None,
) -> int:
    """Minimum *B* so the Clopper-Pearson CI no longer straddles *threshold*.

    Derivation & Statistical Regimes:
    ---------------------------------
    1. **Regime A (Zero Exceedances / Resolution Floor, k = 0)**:
       When no permuted statistics exceed the observed value, the Phipson &
       Smyth (2010) estimator hits the resolution floor:
       ``p_hat = 1 / (B + 1)``.
       If this point estimate is at or below the significance threshold
       (e.g. ``p_hat = 0.001`` at ``alpha_thresh = 0.001`` when ``B = 999``),
       the observed count is ``k = 0``. By the exact Poisson limit / Binomial
       Rule of Three, the one-sided upper ``(1 - alpha)`` confidence bound is:
       ``U_B ≈ -ln(alpha) / (B + 1)``.
       To certify that the true permutation tail probability is strictly
       below ``threshold`` with ``(1 - alpha)`` confidence (i.e. ``U_B < threshold``):
       ``B* >= ceil(-ln(alpha) / threshold) - 1``.
       For 95% confidence (``alpha = 0.05``, ``-ln(0.05) ≈ 2.9957``):
       - To certify p < 0.05:  B* >= 59.
       - To certify p < 0.01:  B* >= 299.
       - To certify p < 0.001: B* >= 2,995.
       This replaces artificial multi-million caps with an exact, principled
       stopping bound.

    2. **Regime B (Normal Approximation Half-Width)**:
       When ``p_hat`` is interior and separated from ``threshold`` by a non-zero
       gap (``|p_hat - threshold| >= 1e-6``), the normal approximation to the
       binomial variance yields:
       ``B* = ceil( (z_{1 - alpha/2} ** 2) * p_hat * (1 - p_hat) / gap**2 )``.

    3. **Regime C (Exact Point Equality with k > 0)**:
       If an interior estimate happens to land exactly on the threshold, increasing
       B cannot separate a point estimate identical to the threshold under the same
       sample proportion. We recommend an order-of-magnitude increase (capped at
       100,000) to refine the empirical resolution.

    Args:
        p_hat: Observed empirical p-value.
        threshold: The nearest significance threshold that the CI straddles.
        alpha: Monte Carlo confidence level (default 0.05 for 95% CI).
        n_randomizations: Total randomizations B in the current run (optional).

    Returns:
        Recommended minimum number of permutations, clamped to ``[100, 100_000]``.
    """
    # Regime A: Zero exceedances floor (p_hat == 1 / (B + 1) <= threshold)
    is_zero_exceedance = False
    if n_randomizations is not None and n_randomizations > 0:
        floor_p = 1.0 / (n_randomizations + 1)
        if abs(p_hat - floor_p) < 1e-9 and p_hat <= threshold:
            is_zero_exceedance = True
    elif threshold > 0 and abs(p_hat - threshold) < 1e-9 and p_hat <= 0.01:
        # Fallback detection for zero-exceedance resolution floor when B is omitted
        is_zero_exceedance = True

    if is_zero_exceedance:
        # Poisson / Binomial Rule of Three bound
        rule_of_three_b = math.ceil(-math.log(alpha) / threshold) - 1
        return max(100, min(rule_of_three_b, 100_000))

    gap = abs(p_hat - threshold)
    if gap < 1e-6:
        # Near-exact point equality on the threshold with k > 0
        current_b = n_randomizations if n_randomizations is not None else 1_000
        return max(100, min(current_b * 10, 100_000))

    z = _sp_stats.norm.ppf(1 - alpha / 2)
    b_min = math.ceil((z**2) * p_hat * (1 - p_hat) / (gap**2))
    return max(100, min(b_min, 100_000))  # type: ignore[no-any-return]


def _render_header_rows(
    rows: list[tuple[str, str, str, str]],
    col1: int,
    col2: int,
) -> None:
    """Print structured header rows from ``family.display_header()``.

    Each 4-tuple is ``(left_label, left_value, right_label,
    right_value)``.  The left pair is flush-left in *col1* columns;
    the right pair is right-aligned in *col2* columns.
    """
    for ll, lv, rl, rv in rows:
        left = f"{ll:<16}{lv:<{col1 - 16}}" if ll else f"{'':<{col1}}"
        right = f"{rl:>{col2 - 11}} {rv:>10}" if rl else ""
        print(f"{left}{right}")


def _resolve_guarantee_tier(results: Any) -> str:
    """Resolve human-readable mathematical guarantee tier for result objects.

    Follows the 4-tier inferential guarantee taxonomy:
    1. Finite-Sample Exact: label permutation of iid units, within-stratum
       permutation, fixed-X knockoffs, split conformal prediction.
    2. Exact up to Model TV Error: Conditional Permutation Test (CPT).
    3. Asymptotically Exact: Freedman–Lane, DML cross-fitting, whitened
       mixed models, Rao score projection, ter Braak, Kennedy.
    4. Distribution-Free Worst-Case (1-2α): Jackknife+ conformal prediction.
    """
    tier = getattr(results, "guarantee_tier", None)
    if tier:
        return str(tier)
    ctx = getattr(results, "context", None)
    if ctx and getattr(ctx, "guarantee_tier", None):
        return str(ctx.guarantee_tier)

    # ConformalResult
    if hasattr(results, "prediction_interval") and hasattr(results, "prediction_set"):
        method = getattr(results, "method", "split")
        if method == "jackknife_plus":
            return "Distribution-Free Worst-Case (Coverage \u2265 1-2\u03b1)"
        return "Finite-Sample Exact (Marginal Coverage \u2265 1-\u03b1)"

    # InvarianceResult
    if hasattr(results, "is_invariant") and hasattr(
        results, "environment_coefficients"
    ):
        method = getattr(results, "method", "kennedy")
        if method == "stratified":
            return "Finite-Sample Exact (Within-Stratum Permutation)"
        elif method == "conditional":
            return "Exact up to Model TV Error (CPT Resampling)"
        return "Asymptotically Exact (Kennedy Environment Block)"

    # KnockoffResult
    if hasattr(results, "selected_features") and hasattr(
        results, "knockoff_statistics"
    ):
        method = getattr(results, "method", "equicorrelated")
        if method == "fixed_x":
            return "Finite-Sample Exact FDR (Fixed-X)"
        return "Model-X Approximate FDR (Gaussian Design / Estimated \u03a3)"

    # Regression result (IndividualTestResult / JointTestResult)
    randomization = getattr(results, "randomization", None)
    if ctx and randomization is None:
        randomization = getattr(ctx, "randomization", None)

    if randomization == "sign_flip":
        return "Exact under Symmetry (Rademacher Sign-Flip)"

    method = getattr(results, "method", "")
    family = getattr(results, "family", None)
    fam_name = family.name if family is not None else ""

    if method in ("manly", "manly_joint"):
        return "Finite-Sample Exact (Unconditional Y-Permutation)"

    if method == "score_exact":
        return "Asymptotically Exact (PQL-Fixed IRLS)"

    if fam_name.endswith("_mixed"):
        return "Asymptotically Exact (Whitened Mixed Model)"

    if method in ("freedman_lane", "freedman_lane_joint"):
        return "Asymptotically Exact (Canonical Freedman\u2013Lane)"

    if method in ("kennedy", "kennedy_joint"):
        return "Asymptotically Exact (Kennedy Exposure Residualization)"

    if method == "ter_braak":
        return "Asymptotically Exact (Canonical ter Braak)"

    if method in ("score", "score_joint"):
        return "Asymptotically Exact (Rao Score Projection)"

    return "Asymptotically Exact"


_FAMILY_DISPLAY_NAMES: dict[str, str] = {
    "linear": "Linear Regression",
    "linear_mixed": "Linear Mixed Model",
    "logistic": "Logistic Regression",
    "logistic_mixed": "Logistic Mixed Model",
    "poisson": "Poisson Regression",
    "poisson_mixed": "Poisson Mixed Model",
    "negative_binomial": "Negative Binomial Regression",
    "ordinal": "Ordinal Logistic Regression",
    "multinomial": "Multinomial Logistic Regression",
}


def _resolve_results_title(results: Any) -> str:
    """Resolve concise standardized title: [Model Type] — [Procedure]."""
    family = getattr(results, "family", None)
    fam_name = family.name if family is not None else "linear"
    model_name = _FAMILY_DISPLAY_NAMES.get(fam_name, fam_name.replace("_", " ").title())

    randomization = getattr(results, "randomization", None)
    ctx = getattr(results, "context", None)
    if ctx and randomization is None:
        randomization = getattr(ctx, "randomization", None)

    if randomization == "sign_flip":
        return f"{model_name} \u2014 Fisher (1935) Sign-Flip Test"

    method = getattr(results, "method", "")
    method_titles = {
        "ter_braak": "ter Braak (1992) Test",
        "freedman_lane": "Freedman\u2013Lane (1983) Test",
        "kennedy": "Kennedy (1995) Test",
        "manly": "Manly (1997) Test",
        "score": "Rao (1948) Score Test",
        "score_exact": "Score Exact Test",
    }
    proc = method_titles.get(method, f"{method.replace('_', ' ').title()} Test")
    return f"{model_name} \u2014 {proc}"


def _resolve_joint_title(results: Any) -> str:
    """Resolve concise standardized title for joint tests: [Model Type] — [Procedure] Joint Test."""
    family = getattr(results, "family", None)
    fam_name = family.name if family is not None else "linear"
    model_name = _FAMILY_DISPLAY_NAMES.get(fam_name, fam_name.replace("_", " ").title())

    method = getattr(results, "method", "")
    method_titles = {
        "freedman_lane_joint": "Freedman\u2013Lane (1983) Joint Test",
        "kennedy_joint": "Kennedy (1995) Joint Test",
        "manly_joint": "Manly (1997) Joint Test",
        "score_joint": "Rao (1948) Score Joint Test",
    }
    proc = method_titles.get(method, f"{method.replace('_', ' ').title()} Joint Test")
    return f"{model_name} \u2014 {proc}"


def print_results_table(
    results: IndividualTestResult,
    *,
    title: str | None = None,
) -> None:
    """Print regression results in a formatted ASCII table similar to statsmodels.

    All metadata (family, feature names, target name) is extracted
    from the result object — no additional context is needed.

    Args:
        results: Typed result object returned by
            :func:`~randomization_tests.randomization_test_regression`.
        title: Title for the output table. If None, a standardized
            title is generated from the model type and test method.
    """
    family: ModelFamily = results.family
    feature_names: list[str] = results.feature_names
    target_name: str | None = results.target_name

    if title is None:
        title = _resolve_results_title(results)

    print("=" * 80)
    for line in textwrap.wrap(title, width=78):
        print(f"{line:^80}")
    guarantee = _resolve_guarantee_tier(results)
    if guarantee:
        g_banner = f"[ Guarantee: {guarantee} ]"
        print(f"{g_banner:^80}")
    print("=" * 80)

    diag = getattr(results, "diagnostics", {})
    col1 = 40
    col2 = 38

    if target_name:
        trunc_target = _truncate(target_name, 20)
        print(
            f"{'Dep. Variable:':<16}{trunc_target:<{col1 - 16}}"
            f"{'No. Observations:':>{col2 - 11}} {diag.get('n_observations', 'N/A'):>10}"
        )
    print(
        f"{'Model Type:':<16}{family.name:<{col1 - 16}}"
        f"{'No. Features:':>{col2 - 11}} {diag.get('n_features', 'N/A'):>10}"
    )
    aic_str = _fmt_diag_val(diag.get("aic", "N/A"))
    print(
        f"{'Method:':<16}{results.method:<{col1 - 16}}"
        f"{'AIC:':>{col2 - 11}} {aic_str:>10}"
    )

    groups_val = getattr(results, "groups", None)
    if groups_val is not None and not family.name.endswith("_mixed"):
        n_clusters = len(np.unique(groups_val))
        strat = getattr(results, "permutation_strategy", None) or "within"
        print(f"{'Clusters:':<16}{f'{n_clusters} ({strat})':<{col1 - 16}}{'':<{col2}}")

    _render_header_rows(family.display_header(diag), col1, col2)

    print("-" * 80)

    # ── Table geometry (W = 80 chars) ─────────────────────────── #
    #
    #   Feature  (fc=22, left)  |  Coef (9, right)  |  2-space gap
    #   |  Emp p-value (23, right)  |  1 space  |  Asy p-value (23, right)
    #   Total: 22 + 9 + 2 + 23 + 1 + 23 = 80
    #
    # The ± margin sub-row re-uses the same grid:
    #   33 blank prefix  (22 feat + 9 coef + 2 gap)
    #   17 right-aligned  core  (± X.XXX)
    #   5 suffix  ([!] or blank)
    #   = 55 visible + 22 prefix padding = matches right edge.
    fc = 22
    stat_label = family.stat_label
    emp_hdr = f"P>|{stat_label}| (Emp)"
    asy_hdr = f"P>|{stat_label}| (Asy)"
    print(f"{'Feature':<{fc}}{'Coef':>9}  {emp_hdr:>23} {asy_hdr:>23}")
    print("-" * 80)

    coefs = results.model_coefs
    emp_p = results.permuted_p_values
    asy_p = results.classic_p_values

    # Clopper-Pearson CI for the empirical p-value (may be absent)
    ci = getattr(results, "confidence_intervals", None) or {}
    pval_ci: list[list[float]] | None = ci.get("pvalue_ci")
    thresholds = [
        results.p_value_threshold_one,
        results.p_value_threshold_two,
        results.p_value_threshold_three,
    ]
    borderline_features: list[tuple[str, float, float]] = []

    for i, feat in enumerate(feature_names):
        trunc_feat = _truncate(feat, fc)
        coef_str = _fmt_coef(coefs[i], 9)
        print(f"{trunc_feat:<{fc}}{coef_str}  {emp_p[i]:>23} {asy_p[i]:>23}")

        # Sub-row: ± margin from the Clopper-Pearson CI, aligned
        # under the empirical p-value column.  Confounders get a
        # blank sub-row (no meaningful permutation p-value).
        if pval_ci is not None and i < len(pval_ci):
            lo, hi = pval_ci[i]
            if np.isnan(lo) or np.isnan(hi):
                # Confounder — print blank sub-row for vertical spacing.
                print(f"{'':<33}{'':>22}")
            else:
                margin = (hi - lo) / 2
                marker = _significance_marker(lo, hi, thresholds)
                # Scientific e-notation when margin < 0.001 (smaller
                # than 3 decimal places can represent); 3 dp otherwise.
                if margin < 0.001 and margin > 0:
                    num_str = f"{margin:.0e}"
                else:
                    num_str = f"{margin:.3f}"
                core = f"\u00b1 {num_str}"
                # Right-align core in 17 chars so the decimal of
                # "\u00b1 0.XXX" aligns with the p-value decimal above.
                # Fixed 5-char suffix for the [!] marker so it never
                # shifts the number.
                _warn_suffix = "  [!]" if marker else "     "
                margin_display = f"{core:>17}{_warn_suffix}"
                # 22 (feat) + 9 (coef) + 2 (gap) = 33 chars of prefix.
                print(f"{'':<33}{margin_display}")
                if marker:
                    # Identify which threshold is straddled to recommend B
                    raw_p = float(results.raw_empirical_p[i])
                    for t in thresholds:
                        if lo < t < hi:
                            borderline_features.append((feat, raw_p, t))
                            break

        # Blank row between feature groups for vertical spacing
        if i < len(feature_names) - 1:
            print()

    # ── Notes ──────────────────────────────────────────────────── #
    notes: list[str] = []

    # Recommend larger n_randomizations for borderline cases.
    if borderline_features:
        ci_alpha = ci.get("confidence_level", 0.95)
        alpha = 1 - ci_alpha if ci_alpha > 0.5 else ci_alpha
        n_rand = getattr(results, "n_randomizations", None)
        b_recs = [
            (
                feat,
                _recommend_n_randomizations(p_hat, t, alpha, n_randomizations=n_rand),
            )
            for feat, p_hat, t in borderline_features
        ]
        max_b = max(b for _, b in b_recs)
        feat_list = ", ".join(feat for feat, _ in b_recs)
        notes.append(
            f"Consider n_randomizations \u2265 {max_b:,} to resolve "
            f"borderline p-values for: {feat_list}."
        )

    # Append any warnings or context notes captured during the pipeline
    method = getattr(results, "method", "")
    confounders = getattr(results, "confounders", None)
    ctx = getattr(results, "context", None)
    if ctx is not None and getattr(ctx, "warnings_captured", None):
        for w_msg in ctx.warnings_captured:
            notes.append(w_msg)
    elif not confounders and method in ("kennedy", "freedman_lane"):
        method_label = "Freedman\u2013Lane" if method == "freedman_lane" else "Kennedy"
        notes.append(
            f"{method_label} method called without confounders \u2014 each feature "
            "is tested partialling out all remaining predictors (standard "
            "multiple regression)."
        )

    if notes:
        print("-" * 80)
        print("Notes")
        print("-" * 80)
        for note in notes:
            print(_wrap(f"  [!] {note}", width=80, indent=6))

    print("=" * 80)
    print(
        f"(***) p < {results.p_value_threshold_three}   "
        f"(**) p < {results.p_value_threshold_two}   "
        f"(*) p < {results.p_value_threshold_one}   "
        f"(ns) p >= {results.p_value_threshold_one}"
    )
    n_rand = getattr(results, "n_randomizations", None)
    if n_rand is not None and n_rand > 0:
        res_floor = 1.0 / (n_rand + 1)
        print(
            f"Permutations: B = {n_rand:,}  |  "
            f"Resolution floor: 1/(B+1) = {res_floor:.4f}  |  "
            f"\u03b1 = {results.p_value_threshold_one}"
        )
    print()


def print_joint_results_table(
    results: JointTestResult,
    *,
    title: str | None = None,
) -> None:
    """Print joint test results in a formatted ASCII table.

    All metadata (family, target name) is extracted from the result
    object — no additional context is needed.

    Args:
        results: Typed result object returned by
            :func:`~randomization_tests.randomization_test_regression` with
            ``method='kennedy_joint'`` or ``method='freedman_lane_joint'``.
        title: Title for the output table. If None, a standardized
            title is generated from the model type and test method.
    """
    family: ModelFamily = results.family
    target_name: str | None = results.target_name

    if title is None:
        title = _resolve_joint_title(results)

    print("=" * 80)
    for line in textwrap.wrap(title, width=78):
        print(f"{line:^80}")
    guarantee = _resolve_guarantee_tier(results)
    if guarantee:
        g_banner = f"[ Guarantee: {guarantee} ]"
        print(f"{g_banner:^80}")
    print("=" * 80)

    diag = getattr(results, "diagnostics", {})
    col1 = 40
    col2 = 38

    if target_name:
        trunc_target = _truncate(target_name, 20)
        print(
            f"{'Dep. Variable:':<16}{trunc_target:<{col1 - 16}}"
            f"{'No. Observations:':>{col2 - 11}} {diag.get('n_observations', 'N/A'):>10}"
        )
    print(
        f"{'Model Type:':<16}{family.name:<{col1 - 16}}"
        f"{'No. Features:':>{col2 - 11}} {diag.get('n_features', 'N/A'):>10}"
    )
    aic_str = _fmt_diag_val(diag.get("aic", "N/A"))
    print(
        f"{'Method:':<16}{results.method:<{col1 - 16}}"
        f"{'AIC:':>{col2 - 11}} {aic_str:>10}"
    )

    _render_header_rows(family.display_header(diag), col1, col2)

    print(f"{'Metric:':<16}{results.metric_type}")

    feat_list = ", ".join(_truncate(f, 25) for f in results.features_tested)
    print(_wrap(f"Features Tested: {feat_list}", width=80, indent=17))
    if results.confounders:
        conf_list = ", ".join(_truncate(c, 25) for c in results.confounders)
        print(_wrap(f"Confounders: {conf_list}", width=80, indent=13))
    print("-" * 80)

    print(f"{'Observed Improvement:':<30} {results.observed_improvement:>12.4f}")
    print(f"{'Joint p-Value:':<30} {results.p_value_str:>12}")

    # ── Notes ──────────────────────────────────────────────────── #
    notes: list[str] = []
    method = getattr(results, "method", "")
    confounders = getattr(results, "confounders", None)
    ctx = getattr(results, "context", None)
    if ctx is not None and getattr(ctx, "warnings_captured", None):
        for w_msg in ctx.warnings_captured:
            notes.append(w_msg)
    elif not confounders and method in ("kennedy_joint", "freedman_lane_joint"):
        method_label = (
            "Freedman\u2013Lane" if method == "freedman_lane_joint" else "Kennedy"
        )
        notes.append(
            f"{method_label} method called without confounders \u2014 all "
            "features will be tested against the null model."
        )

    if notes:
        print("-" * 80)
        print("Notes")
        print("-" * 80)
        for note in notes:
            print(_wrap(f"  [!] {note}", width=80, indent=6))

    print("=" * 80)
    print(
        f"(***) p < {results.p_value_threshold_three}   "
        f"(**) p < {results.p_value_threshold_two}   "
        f"(*) p < {results.p_value_threshold_one}   "
        f"(ns) p >= {results.p_value_threshold_one}"
    )
    print("Omnibus test: single p-value for all tested features combined.")
    n_rand = getattr(results, "n_randomizations", None)
    if n_rand is not None and n_rand > 0:
        res_floor = 1.0 / (n_rand + 1)
        print(
            f"Permutations: B = {n_rand:,}  |  "
            f"Resolution floor: 1/(B+1) = {res_floor:.4f}  |  "
            f"\u03b1 = {results.p_value_threshold_one}"
        )
    print()


def print_diagnostics_table(
    results: IndividualTestResult,
    *,
    title: str = "Permutation Diagnostics",
) -> None:
    """Print extended model diagnostics in a formatted ASCII table.

    All metadata (family, feature names) is extracted from the result
    object — no additional context is needed.

    This table complements :func:`print_results_table` with additional
    per-predictor and model-level diagnostics.  It is intended to be
    displayed optionally (e.g., behind a ``verbose`` flag) to help
    users assess whether permutation-test assumptions are met and
    identify potential problems.

    The table is structured in four sections:

    1. **Per-predictor Diagnostics** — a columnar table of standardized
       coefficients, VIF, Monte Carlo SE, and divergence flags.
    2. **Legend** — a compact 2-3 line key explaining each column,
       including VIF thresholds and what DIVERGENT means.
    3. **Model-level Diagnostics** — residual-based assumption checks
       delegated to the family's ``display_diagnostics()`` method,
       plus Cook's distance and permutation coverage.
    4. **Notes** (conditional) — plain-language warnings printed only
       when a diagnostic flags a potential concern.

    Args:
        results: Typed result object returned by
            :func:`~randomization_tests.randomization_test_regression`.
            Must contain ``extended_diagnostics``.
        title: Title for the output table.
    """
    family: ModelFamily = results.family
    feature_names: list[str] = results.feature_names

    ext = getattr(results, "extended_diagnostics", None)
    if ext is None:
        return

    W = 80
    notes: list[str] = []

    # ── Title ──────────────────────────────────────────────────── #

    print("=" * W)
    for line in textwrap.wrap(title, width=W - 2):
        print(f"{line:^{W}}")
    print("=" * W)

    # ── Per-predictor Diagnostics ──────────────────────────────── #

    fc = 22  # feature name column width
    has_divergent = False

    # Exposure R² column is shown only for the Kennedy individual
    # method, where it quantifies how much of each predictor's
    # variance is already explained by the confounders.  A value
    # near 1.0 means the exposure-model residuals (the "clean"
    # variation actually used by the permutation test) have near-zero
    # variance, making the permuted coefficients wildly unstable and
    # the resulting p-value inflated toward 1.0.  Surfacing this
    # metric lets users immediately diagnose *why* a predictor shows
    # an unexpectedly large p-value under the Kennedy method.
    exp_r2 = ext.get("exposure_r_squared")
    show_exp_r2 = exp_r2 is not None and len(exp_r2) > 0

    # Clopper-Pearson CI column (in non-Exp-R² layouts only)
    ci = getattr(results, "confidence_intervals", None) or {}
    pval_ci: list[list[float]] | None = ci.get("pvalue_ci")
    show_pval_ci = pval_ci is not None and not show_exp_r2

    print("Per-predictor Diagnostics")
    print("-" * W)
    if show_exp_r2:
        _exp_hdr = "Exp R\u00b2"
        print(
            f"{'Feature':<{fc}}"
            f"{'Std Coef':>10} "
            f"{'VIF':>8} "
            f"{'MC SE':>10} "
            f"{_exp_hdr:>11}  "
            f"{'Emp vs Asy':>14}"
        )
    elif show_pval_ci:
        print(
            f"{'Feature':<{fc}}"
            f"{'Std Coef':>9} "
            f"{'VIF':>8} "
            f"{'MC SE':>9} "
            f"{'P-Val CI':>13}   "
            f"{'Emp vs Asy':>13}"
        )
    else:
        print(
            f"{'Feature':<{fc}}"
            f"{'Std Coef':>10} "
            f"{'VIF':>10} "
            f"{'MC SE':>12} "
            f"{'Emp vs Asy':>18}"
        )
    print("-" * W)

    std_coefs = ext.get("standardized_coefs", [])
    vifs = ext.get("vif", [])
    mc_ses = ext.get("monte_carlo_se", [])
    div_flags = ext.get("divergence_flags", [])

    vif_problems: list[tuple[str, float, str]] = []
    # Track predictors whose exposure R² exceeds 0.99 so a Notes
    # warning can explain that the Kennedy permutation null is
    # degenerate for those features.  The 0.99 threshold is
    # deliberately conservative: at R² = 0.99 only 1% of X_j's
    # variance survives partialling out the confounders, which is
    # typically insufficient for a stable permutation distribution.
    exp_r2_problems: list[tuple[str, float]] = []

    for i, feat in enumerate(feature_names):
        trunc_feat = _truncate(feat, fc)

        # Standardized coefficient
        if show_exp_r2:
            std_c = _fmt_coef(std_coefs[i], 10) if i < len(std_coefs) else f"{'':>10}"
        elif show_pval_ci:
            std_c = _fmt_coef(std_coefs[i], 9) if i < len(std_coefs) else f"{'':>9}"
        else:
            std_c = _fmt_coef(std_coefs[i], 10) if i < len(std_coefs) else f"{'':>10}"

        # VIF — flag problematic values for Notes
        if i < len(vifs):
            v = vifs[i]
            if show_exp_r2 or show_pval_ci:
                vif_str = _fmt_coef(v, 8)
            else:
                vif_str = _fmt_coef(v, 10)
            if v > 10:
                vif_problems.append((feat, v, "severe"))
            elif v > 5:
                vif_problems.append((feat, v, "moderate"))
        else:
            if show_exp_r2 or show_pval_ci:
                vif_str = f"{'':>8}"
            else:
                vif_str = f"{'':>10}"

        # Monte Carlo SE — 4 dp sufficient for precision assessment.
        # Confounders have no permutation distribution, so their MC SE
        # is NaN; display an em dash instead of a bare "nan".
        if i < len(mc_ses):
            mc_val = mc_ses[i]
            if show_exp_r2:
                if isinstance(mc_val, float) and mc_val != mc_val:  # NaN check
                    mc_str = f"{'—':>8}  "
                else:
                    mc_str = f"{mc_val:>10.4f}"
            elif show_pval_ci:
                if isinstance(mc_val, float) and mc_val != mc_val:  # NaN check
                    mc_str = f"{'—':>7}  "
                else:
                    mc_str = f"{mc_val:>9.4f}"
            else:
                if isinstance(mc_val, float) and mc_val != mc_val:  # NaN check
                    mc_str = f"{'—':>10}  "
                else:
                    mc_str = f"{mc_val:>12.4f}"
        else:
            if show_exp_r2:
                mc_str = f"{'':>10}"
            elif show_pval_ci:
                mc_str = f"{'':>9}"
            else:
                mc_str = f"{'':>12}"

        # P-Value CI column (non-Exp-R² layouts only)
        if show_pval_ci:
            if pval_ci is not None and i < len(pval_ci):
                lo, hi = pval_ci[i]
                if np.isnan(lo) or np.isnan(hi):
                    # Confounder — no meaningful p-value CI.
                    ci_str = f"{'—':>14}  "
                else:
                    ci_str = f"[{lo:.3f}, {hi:.3f}]"
                    ci_str = f"{ci_str:>16}"
            else:
                ci_str = f"{'—':>14}  "

        # Exposure R² (Kennedy only)
        # For non-confounder features, this is the R² from regressing
        # X_j on the confounders Z.  A high value (> 0.99) means
        # nearly all of X_j's variance is absorbed by Z, leaving the
        # permutation test with almost no residual signal to work
        # with.  Confounders themselves are marked with an em dash
        # because they are controls, not hypotheses.
        if show_exp_r2:
            er2 = exp_r2[i] if i < len(exp_r2) else None
            if er2 is None:
                # Confounder — not part of the hypothesis; exposure
                # R² is undefined for controls.
                er2_str = f"{'—':>9}  "
            else:
                er2_str = f"{er2:>11.4f}"
                if er2 > 0.99:
                    exp_r2_problems.append((feat, er2))

        # Divergence flag
        flag = div_flags[i] if i < len(div_flags) else ""
        if flag:
            has_divergent = True

        if show_exp_r2:
            div_str = f"{flag:>14}"
            print(f"{trunc_feat:<{fc}}{std_c} {vif_str} {mc_str} {er2_str}  {div_str}")
        elif show_pval_ci:
            div_str = f"{flag:>13}"
            print(f"{trunc_feat:<{fc}}{std_c} {vif_str} {mc_str} {ci_str}{div_str}")
        else:
            div_str = f"{flag:>18}"
            print(f"{trunc_feat:<{fc}}{std_c} {vif_str} {mc_str} {div_str}")

    # ── Legend ──────────────────────────────────────────────────── #

    print("-" * W)
    print("  Std Coef: effect per SD.  VIF: collinearity (> 5 moderate, > 10 severe).")
    print("  MC SE: p-value precision (increase B if large relative to p).")
    if show_pval_ci:
        print("  P-Val CI: Clopper-Pearson exact 95% CI for the empirical p-value.")
    if show_exp_r2:
        print(
            "  Exp R\u00b2: variance of X_j explained by "
            "confounders (> 0.99 = collinear)."
        )
    if has_divergent:
        print("  DIVERGENT = permutation and classical p-values disagree at alpha.")

    # Collect VIF notes
    if vif_problems:
        parts = ", ".join(
            f"{name} = {val:.2f} ({label})" for name, val, label in vif_problems
        )
        notes.append(f"VIF: {parts}.")

    # Collect exposure R² notes
    # When any non-confounder feature has R² > 0.99, the permutation
    # null distribution is degenerate because there is almost no
    # residual variance left to permute.  The permuted |β*| values
    # become wildly inflated relative to the observed |β|, pushing
    # the p-value toward 1.0.  This is not a test error — it is a
    # direct consequence of the predictor being near-collinear with
    # the confounders — but it warrants an explicit warning so users
    # understand *why* the p-value is extreme.
    if exp_r2_problems:
        parts = ", ".join(f"{name} = {val:.4f}" for name, val in exp_r2_problems)
        notes.append(
            f"Exp R\u00b2: {parts}. Near-collinear with "
            f"confounders; permuted coefficients are unstable "
            f"and p-values are inflated."
        )

    # ── Model-level Diagnostics ────────────────────────────────── #

    lw = 28  # label column width (not counting 2-space indent)
    sw = 14  # stat column width

    print("-" * W)
    print("Model-level Diagnostics")
    print("-" * W)

    diag_lines, diag_notes = family.display_diagnostics(ext)
    for label, stat, detail in diag_lines:
        print(f"  {label:<{lw}}{stat:<{sw}}{detail}")
    notes.extend(diag_notes)

    # Symmetry diagnostic (sign-flip tests)
    sym = ext.get("symmetry")
    if sym:
        stat_val = sym.get("test_statistic", float("nan"))
        p_val = sym.get("p_value", float("nan"))
        is_sym = sym.get("is_symmetric", False)
        stat_str = f"{stat_val:.2f}" if np.isfinite(stat_val) else "N/A"
        sym_detail = f"p = {_fmt_p(p_val)} ({'Symmetric' if is_sym else 'Asymmetric'})"
        print(f"  {'Wilcoxon Symmetry:':<{lw}}{stat_str:<{sw}}{sym_detail}")
        if is_sym:
            notes.append(
                f"Wilcoxon signed-rank p = {p_val:.4f}: residuals are symmetric about "
                "zero (Rademacher sign-flip assumption satisfied)."
            )
        else:
            notes.append(
                f"Wilcoxon signed-rank p = {p_val:.4f}: residuals may be asymmetric. "
                "Sign-flip test may have inflated Type I error; consider permutation "
                "testing instead."
            )

    # Cook's distance
    cd = ext.get("cooks_distance", {})
    if cd:
        n_inf = cd.get("n_influential", 0)
        thresh = cd.get("threshold", 0)
        cooks_arr = cd.get("cooks_d")
        max_d = (
            float(np.nanmax(cooks_arr))
            if cooks_arr is not None and len(cooks_arr) > 0
            else 0.0
        )
        n_inf_str = f"{n_inf} obs."
        cd_label = "Cook's D (> 4/n):"
        print(f"  {cd_label:<{lw}}{n_inf_str:<{sw}}threshold = {thresh:.4f}")
        if isinstance(n_inf, (int, float)) and n_inf > 0:
            if max_d > 0.5:
                notes.append(
                    f"{int(n_inf)} obs. with Cook's D > {thresh:.4f} (4/n), "
                    f"max D = {max_d:.3f} (> 0.5); severe leverage detected, "
                    "results may be sensitive to influential points."
                )
            else:
                notes.append(
                    f"{int(n_inf)} obs. with Cook's D > {thresh:.4f} (4/n), "
                    f"max D = {max_d:.3f}. All points remain below severe "
                    "threshold (D < 0.5); influence on parameter estimates "
                    "is minor."
                )

    # Permutation coverage with sufficiency verdict
    pc = ext.get("permutation_coverage", {})
    if pc:
        cov_pct = pc.get("coverage_pct", "")
        n_fact_str = pc.get("n_factorial_str", "")
        n_perm = getattr(results, "n_randomizations", None)
        b_str = f"{n_perm:,}" if n_perm is not None else "?"

        # Compute sufficiency verdict from Clopper-Pearson CIs
        ci = getattr(results, "confidence_intervals", None) or {}
        pval_ci_list: list[list[float]] | None = ci.get("pvalue_ci")
        thresholds = [
            getattr(results, "p_value_threshold_one", 0.05),
            getattr(results, "p_value_threshold_two", 0.01),
            getattr(results, "p_value_threshold_three", 0.001),
        ]
        emp_p_vals = getattr(results, "permuted_p_values", []) or []
        is_borderline = False
        if pval_ci_list is not None:
            for idx, (lo_v, hi_v) in enumerate(pval_ci_list):
                # Skip confounders
                if idx < len(emp_p_vals) and emp_p_vals[idx] == "(confounder)":
                    continue
                if np.isnan(lo_v) or np.isnan(hi_v):
                    continue
                for t in thresholds:
                    if lo_v < t < hi_v:
                        is_borderline = True
                        break
                if is_borderline:
                    break
        verdict = "borderline" if is_borderline else "sufficient"
        detail = f"B = {b_str} / {n_fact_str} \u2014 {verdict}"
        print(f"  {'Coverage:':<{lw}}{cov_pct:<{sw}}{detail}")

    # Append any warnings or context notes captured during the pipeline
    ctx = getattr(results, "context", None)
    if ctx is not None and getattr(ctx, "warnings_captured", None):
        for w_msg in ctx.warnings_captured:
            notes.append(w_msg)

    # ── Notes ──────────────────────────────────────────────────── #

    if notes:
        print("-" * W)
        print("Notes")
        print("-" * W)
        for note in notes:
            print(_wrap(f"  [!] {note}", width=W, indent=6))

    print("=" * W)
    print()


def print_symmetry_table(
    symmetry: dict[str, Any] | IndividualTestResult,
    *,
    title: str = "Symmetry Diagnostic (Fisher 1935 / Wilcoxon Signed-Rank Test)",
) -> None:
    """Print residual symmetry diagnostic in a formatted ASCII table.

    Displays the output of :func:`~randomization_tests.validate_symmetry`
    in a bordered 80-character table matching the visual style of other
    ``print_*`` display functions.

    Accepts either a dictionary returned by ``validate_symmetry()`` or a
    completed ``IndividualTestResult`` from a test run with
    ``randomization='sign_flip'``.

    The sign-flip test assumes residuals are symmetric about zero under
    the null.  This table shows whether the Wilcoxon signed-rank test
    detects significant asymmetry at α = 0.05.

    Args:
        symmetry: Dict returned by ``validate_symmetry()`` or a completed
            ``IndividualTestResult`` containing symmetry diagnostics.
        title: Title for the output table.
    """
    if isinstance(symmetry, dict):
        symmetry_data = symmetry
    else:
        ext = getattr(symmetry, "extended_diagnostics", {}) or {}
        sym_dict = ext.get("symmetry")
        if sym_dict is None:
            ctx = getattr(symmetry, "context", None)
            residuals = getattr(ctx, "residuals", None) if ctx else None
            if residuals is not None:
                from .sign_flips import validate_symmetry

                sym_dict = validate_symmetry(residuals)
            else:
                sym_dict = {}
        symmetry_data = sym_dict

    W = 80
    lw = 28
    sw = 14

    print("=" * W)
    for line in textwrap.wrap(title, width=W - 2):
        print(f"{line:^{W}}")
    g_banner = "[ Diagnostic: Nonparametric Wilcoxon Signed-Rank Symmetry Test ]"
    print(f"{g_banner:^{W}}")
    print("=" * W)

    stat = symmetry_data.get("test_statistic", float("nan"))
    p = symmetry_data.get("p_value", float("nan"))
    is_sym = symmetry_data.get("is_symmetric", False)

    print(f"  {'Test statistic:':<{lw}}{stat:<{sw}.2f}")
    print(f"  {'p-value:':<{lw}}{p:<{sw}.4f}")
    print(f"  {'Symmetric (α = 0.05):':<{lw}}{'Yes' if is_sym else 'No':<{sw}}")

    print("-" * W)
    if is_sym:
        print(
            _wrap(
                "  Residuals are symmetric — sign-flip test is appropriate.",
                width=W,
                indent=2,
            )
        )
    else:
        print(
            _wrap(
                "  Residuals may be asymmetric — interpret sign-flip results "
                "with caution. Consider permutation test as primary.",
                width=W,
                indent=2,
            )
        )
    print("=" * W)
    print()


def print_comparison_table(
    results: list[tuple[str, IndividualTestResult] | IndividualTestResult],
    *,
    title: str | None = None,
    alpha: float = 0.05,
    is_ar: bool = False,
) -> None:
    """Print a side-by-side p-value comparison across two model variants.

    A pairwise comparison table (capped at k=2) that accepts a pair of
    ``(label, result)`` tuples or result objects, cleanly spaced across the
    full 80-character display grid.

    If labels are omitted or empty, they default to ``"A"`` and ``"B"``.

    For AR-specific comparisons, the baseline (No AR) and highest AR order
    are contrasted with AR artifact classification in the Verdict column.

    Args:
        results: Sequence of ``(label, result)`` pairs or result objects.
            When more than 2 items are provided for AR comparisons, the first
            (baseline) and last (highest order) models are compared.
        title: Title for the output table. If None, generated automatically.
        alpha: Significance threshold for the verdict column.
        is_ar: Whether to enable autoregressive artifact classification.
    """
    if not results:
        return

    W = 80
    fc, c1, c2, c3 = 16, 18, 22, 24

    # Auto-detect AR mode if not explicitly specified
    if not is_ar:
        for item in results:
            lbl = item[0] if isinstance(item, tuple) and len(item) == 2 else ""
            res = item[1] if isinstance(item, tuple) and len(item) == 2 else item
            if "AR" in str(lbl):
                is_ar = True
                break
            ctx = getattr(res, "context", None)
            if ctx is not None and getattr(ctx, "ar_order", None) is not None:
                is_ar = True
                break

    # Extract pairwise comparison: for AR, contrast first (baseline) vs last (highest order)
    if is_ar and len(results) > 2:
        pair = [results[0], results[-1]]
    else:
        pair = list(results[:2])

    default_labels = ["A", "B"]
    norm: list[tuple[str, IndividualTestResult]] = []
    for idx, item in enumerate(pair):
        if isinstance(item, tuple) and len(item) == 2:
            lbl, res = item
            lbl_str = (
                str(lbl).strip()
                if lbl is not None and str(lbl).strip()
                else default_labels[idx]
            )
        else:
            lbl_str = default_labels[idx]
            res = item
        norm.append((lbl_str, res))

    if len(norm) < 2:
        norm.append((default_labels[1], norm[0][1]))

    lbl1, lbl2 = norm[0][0], norm[1][0]
    res1, res2 = norm[0][1], norm[1][1]

    feature_names: list[str] = res1.feature_names
    n_features = len(feature_names)

    # ── Title ──────────────────────────────────────────────────── #
    if title is None:
        if is_ar:
            title = f"P-Value Comparison: {lbl1} vs. {lbl2}"
        else:
            title = f"{lbl1} vs. {lbl2} P-Value Comparison"

    print("=" * W)
    for line in textwrap.wrap(title, width=W - 2):
        print(f"{line:^{W}}")
    if is_ar:
        g_banner = "[ Guarantee: Asymptotically Exact (FGLS Autoregressive Whitening) ]"
        print(f"{g_banner:^{W}}")
    print("=" * W)

    # ── Column widths (W = 80: fc=16, c1=18, c2=22, c3=24) ────── #
    hdr = f"{'Feature':<{fc}}{lbl1:>{c1}}{lbl2:>{c2}}{'Verdict':>{c3}}"
    print(hdr)
    print("-" * W)

    # ── Rows ───────────────────────────────────────────────────── #
    p1_vals = [float(v) for v in res1.raw_empirical_p[:n_features]]
    p2_vals = [float(v) for v in res2.raw_empirical_p[:n_features]]

    for i, feat in enumerate(feature_names):
        trunc = _truncate(feat, fc - 1)
        p1 = p1_vals[i]
        p2 = p2_vals[i]

        sig1 = p1 < alpha if not np.isnan(p1) else False
        sig2 = p2 < alpha if not np.isnan(p2) else False

        if np.isnan(p1) or np.isnan(p2):
            p1_str = f"{'—':>{c1}}" if np.isnan(p1) else f"{p1:>{c1}.4f}"
            p2_str = f"{'—':>{c2}}" if np.isnan(p2) else f"{p2:>{c2}.4f}"
            verdict = "Confounder"
        else:
            p1_str = f"{p1:>{c1}.4f}"
            p2_str = f"{p2:>{c2}.4f}"
            if sig1 and sig2:
                verdict = "Both sig."
            elif not sig1 and not sig2:
                verdict = "Both (ns)"
            elif is_ar and sig1 and not sig2:
                verdict = "ARTIFACT"
            elif is_ar and not sig1 and sig2:
                verdict = "EMERGENT"
            else:
                verdict = "DIVERGENT"

        print(f"{trunc:<{fc}}{p1_str}{p2_str}{verdict:>{c3}}")

    print("=" * W)
    print()


def print_confounder_table(
    confounder_results: dict[str, Any] | object,
    title: str = "Confounder Identification",
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    family: ModelFamily | None = None,
    correlation_method: str | None = None,
    correction_method: str | None = None,
) -> None:
    """Print confounder identification results in a formatted ASCII table.

    Presents the output of :func:`~randomization_tests.identify_confounders`
    in a bordered 80-character table matching the visual style of
    :func:`print_results_table` and :func:`print_diagnostics_table`.

    The table is structured in up to three sections:

    1. **Per-predictor results** — for each predictor that has at
       least one identified confounder or mediator, lists the
       confounder and mediator names as comma-separated text
       (not raw Python lists).
    2. **Clean predictors** — a compact list of predictors for which
       no confounders or mediators were identified.
    3. **Notes** (conditional) — printed only when mediators are
       detected, warning that they should *not* be controlled for
       because they transmit the causal effect.

    Accepts either a single :func:`identify_confounders` result dict
    (or :class:`ConfounderAnalysisResult` dataclass), or a
    ``dict[str, dict]`` mapping predictor names to their individual
    result dicts (for multiple predictors).

    Args:
        confounder_results: Either a single result dict / dataclass
            (must contain ``identified_confounders`` and ``predictor``
            keys) or a dict mapping predictor names to result dicts.
        title: Title for the output table.
        correlation_threshold: The ``|r|`` threshold used during
            screening (displayed in the parameter header).
        p_value_threshold: The screening p-value cutoff (displayed in
            the parameter header).
        n_bootstrap: Bootstrap iterations used for mediation analysis
            (displayed in the parameter header).
        confidence_level: Confidence-interval level used for mediation
            (displayed in the parameter header).
        correlation_method: Correlation method used during screening.
        correction_method: Multiple-testing correction method.
    """
    W = 80

    # ── Normalise input ────────────────────────────────────────── #
    # Accept ConfounderAnalysisResult dataclass — convert to dict.
    if hasattr(confounder_results, "to_dict") and not isinstance(
        confounder_results, dict
    ):
        confounder_results = confounder_results.to_dict()
    assert isinstance(confounder_results, dict)

    # A single identify_confounders() result has an
    # "identified_confounders" key at the top level.  A multi-
    # predictor dict has predictor names as keys, each mapping to a
    # result dict.  Detecting this lets the function accept both
    # forms without requiring the caller to wrap a single result.
    if "identified_confounders" in confounder_results:
        results_by_pred: dict[str, Any] = {
            confounder_results["predictor"]: confounder_results,
        }
    else:
        results_by_pred = confounder_results

    # ── Title ──────────────────────────────────────────────────── #

    print("=" * W)
    for line in textwrap.wrap(title, width=W - 2):
        print(f"{line:^{W}}")
    print("=" * W)

    # ── Parameter header ───────────────────────────────────────── #
    # Show the screening and mediation thresholds so the reader
    # knows which settings produced these results without having
    # to inspect the calling code.

    ci_pct = int(confidence_level * 100)
    corr_label = (correlation_method or "pearson").capitalize()
    screen_line = (
        f"Screen: |r|>={correlation_threshold}, "
        f"p<{p_value_threshold} ({corr_label}"
        + (f", {correction_method}" if correction_method else "")
        + ")"
    )
    med_line = f"BCa bootstrap (B={n_bootstrap}, {ci_pct}% CI)"
    # Keep within 80 chars: join on one line if it fits, else two.
    combined = f"{screen_line}   {med_line}"
    if len(combined) <= W:
        print(combined)
    else:
        print(screen_line)
        print(med_line)
    if family is not None:
        print(f"Family:    {family.name}")
    print("-" * W)

    # ── Partition predictors ───────────────────────────────────── #
    # Separate predictors with findings (confounders, mediators,
    # colliders, moderators) from those without.

    has_findings: dict[str, dict] = {}
    no_findings: list[str] = []
    has_mediators = False
    has_colliders = False
    has_moderators = False

    for pred, res in results_by_pred.items():
        confounders = res.get("identified_confounders", [])
        mediators = res.get("identified_mediators", [])
        colliders = res.get("identified_colliders", [])
        moderators = res.get("identified_moderators", [])
        if confounders or mediators or colliders or moderators:
            has_findings[pred] = res
            if mediators:
                has_mediators = True
            if colliders:
                has_colliders = True
            if moderators:
                has_moderators = True
        else:
            no_findings.append(pred)

    # ── Per-predictor results ──────────────────────────────────── #
    # Each predictor with at least one confounder or mediator gets
    # its own block separated by a blank line: the predictor name
    # as a left-aligned label, then indented "Confounders:" and/or
    # "Mediators:" lines with the variable names formatted as
    # comma-separated text.  Long lists wrap at 80 characters with
    # continuation lines aligned to column 18.

    # "  Confounders:    " and "  Mediators:      " both occupy 18
    # chars, aligning the variable-name text in a consistent column.
    indent_label = 18

    if has_findings:
        first = True
        for pred, res in has_findings.items():
            if not first:
                print()
            first = False

            print(f"  Predictor:      {_truncate(pred, W - 18)}")

            confounders = res.get("identified_confounders", [])
            mediators = res.get("identified_mediators", [])
            colliders = res.get("identified_colliders", [])
            moderators = res.get("identified_moderators", [])

            if confounders:
                conf_str = ", ".join(confounders)
                print(
                    _wrap(
                        f"  Confounders:    {conf_str}",
                        width=W,
                        indent=indent_label,
                    )
                )

            if mediators:
                med_str = ", ".join(mediators)
                print(
                    _wrap(
                        f"  Mediators:      {med_str}",
                        width=W,
                        indent=indent_label,
                    )
                )

            if colliders:
                coll_str = ", ".join(colliders)
                print(
                    _wrap(
                        f"  Colliders:      {coll_str}",
                        width=W,
                        indent=indent_label,
                    )
                )

            if moderators:
                mod_str = ", ".join(moderators)
                print(
                    _wrap(
                        f"  Moderators:     {mod_str}",
                        width=W,
                        indent=indent_label,
                    )
                )
    else:
        # No predictor had any findings at all.
        print("  No confounders, mediators, colliders, or moderators identified.")

    # ── Clean predictors ───────────────────────────────────────── #
    # List predictors with no findings in a compact comma-separated
    # line, avoiding the verbose per-predictor block format.

    if no_findings:
        print("-" * W)
        no_str = ", ".join(no_findings)
        print(
            _wrap(
                f"  No issues:      {no_str}",
                width=W,
                indent=indent_label,
            )
        )

    # ── Notes ──────────────────────────────────────────────────── #
    # Mediator warnings are only shown when at least one mediator
    # was detected, keeping the output compact for the common case
    # where every candidate is a confounder.

    notes: list[str] = []
    if has_mediators:
        notes.append(
            "Mediators transmit the causal effect of the predictor "
            "on the outcome. Do not control for them as confounders "
            "in the Kennedy method."
        )
    if has_colliders:
        notes.append(
            "Controlling for colliders introduces bias — do NOT add "
            "to confounders=. Colliders are caused by BOTH the "
            "predictor and the outcome (X → Z ← Y)."
        )
    if has_moderators:
        notes.append(
            "Moderators change the strength of the predictor's effect. "
            "Consider including the interaction term X×Z as a predictor. "
            "Moderator variables remain in the confounder list."
        )

    if hasattr(confounder_results, "advisories") and getattr(
        confounder_results, "advisories", None
    ):
        for adv in confounder_results.advisories:
            if adv not in notes:
                notes.append(adv)

    if notes:
        print("-" * W)
        print("Notes")
        print("-" * W)
        for note in notes:
            print(_wrap(f"  [!] {note}", width=W, indent=6))

    print("=" * W)
    print()


def print_dataset_info_table(
    *,
    name: str,
    X: Any | None = None,
    y: Any | None = None,
    n_observations: int | None = None,
    n_features: int | None = None,
    feature_names: list[str] | None = None,
    target_name: str | None = None,
    target_description: str | None = None,
    y_range: tuple[float, float] | tuple[int, int] | None = None,
    y_mean: float | None = None,
    y_var: float | None = None,
    extra_stats: dict[str, str] | None = None,
    title: str = "Dataset Information",
) -> None:
    """Print dataset metadata in a formatted ASCII table.

    Displays dataset name, dimensions, feature names, target variable
    information, and optional outcome statistics in a bordered table
    matching the visual style of other ``print_*`` functions.

    If *X* and *y* are provided, metadata and outcome statistics are
    computed automatically from the data containers.

    Args:
        name: Dataset name (e.g., ``'Abalone'``).
        X: Optional feature matrix (DataFrame or array).
        y: Optional target vector/DataFrame.
        n_observations: Number of observations (rows). Computed from X/y if omitted.
        n_features: Number of features (columns in X). Computed from X if omitted.
        feature_names: Optional list of feature names. If provided or extracted
            from X, displayed as a comma-separated list, truncated if too long.
        target_name: Optional name of the target variable (e.g., ``'Rings'``).
        target_description: Optional description of the target (e.g.,
            ``'growth-ring count'``).
        y_range: Optional tuple ``(min, max)`` of outcome values.
        y_mean: Optional mean of the outcome.
        y_var: Optional variance of the outcome.
        extra_stats: Optional dict of additional statistics to display
            as ``{label: value}`` pairs (e.g., ``{'Var/Mean': '1.05'}``).
        title: Title for the output table.
    """
    if X is not None:
        if n_observations is None:
            n_observations = len(X)
        if n_features is None:
            n_features = X.shape[1] if hasattr(X, "shape") else len(X.columns)
        if feature_names is None and hasattr(X, "columns"):
            feature_names = [str(c) for c in X.columns]

    if y is not None:
        if target_name is None:
            if hasattr(y, "columns") and len(y.columns) > 0:
                target_name = str(y.columns[0])
            elif hasattr(y, "name") and y.name:
                target_name = str(y.name)
            else:
                target_name = "y"

        try:
            y_arr = np.asarray(y, dtype=float).ravel()
            if len(y_arr) > 0 and not np.all(np.isnan(y_arr)):
                if y_range is None:
                    y_min = float(np.nanmin(y_arr))
                    y_max = float(np.nanmax(y_arr))
                    if np.all(np.equal(np.mod(y_arr, 1), 0)):
                        y_range = (int(y_min), int(y_max))
                    else:
                        y_range = (y_min, y_max)
                if y_mean is None:
                    y_mean = float(np.nanmean(y_arr))
                if y_var is None:
                    y_var = float(np.nanvar(y_arr))
        except (ValueError, TypeError):
            pass

    if n_observations is None:
        n_observations = 0
    if n_features is None:
        n_features = 0

    W = 80
    lw = 20  # label column width

    # ── Title ──────────────────────────────────────────────────── #

    print("=" * W)
    for line in textwrap.wrap(title, width=W - 2):
        print(f"{line:^{W}}")
    print("=" * W)

    # ── Dataset info ───────────────────────────────────────────── #

    print(f"  {'Dataset:':<{lw}}{name}")
    print(f"  {'No. Observations:':<{lw}}{n_observations}")

    if feature_names:
        feat_str = ", ".join(feature_names)
        # 2 (indent) + lw (label) + len(n_features digits) + len(" (") + len(")") = overhead
        prefix = f"{n_features} ("
        max_feat_len = W - 2 - lw - len(prefix) - 1  # -1 for closing ")"
        if len(feat_str) > max_feat_len:
            feat_str = feat_str[: max_feat_len - 3] + "..."
        print(f"  {'No. Features:':<{lw}}{prefix}{feat_str})")
    else:
        print(f"  {'No. Features:':<{lw}}{n_features}")

    if target_name:
        if target_description:
            print(f"  {'Target:':<{lw}}{target_name} ({target_description})")
        else:
            print(f"  {'Target:':<{lw}}{target_name}")

    # ── Outcome statistics ─────────────────────────────────────── #

    has_y_stats = y_range is not None or y_mean is not None or y_var is not None
    if has_y_stats or extra_stats:
        print("-" * W)

    if y_range is not None:
        if isinstance(y_range[0], int) and isinstance(y_range[1], int):
            print(f"  {'Y Range:':<{lw}}[{y_range[0]}, {y_range[1]}]")
        else:
            print(f"  {'Y Range:':<{lw}}[{y_range[0]:.4g}, {y_range[1]:.4g}]")

    if y_mean is not None:
        print(f"  {'Y Mean:':<{lw}}{y_mean:.4f}")

    if y_var is not None:
        if y_mean is not None and y_mean != 0:
            var_mean_ratio = y_var / y_mean
            print(
                f"  {'Y Variance:':<{lw}}{y_var:.4f}  (var/mean = {var_mean_ratio:.2f})"
            )
        else:
            print(f"  {'Y Variance:':<{lw}}{y_var:.4f}")

    if extra_stats:
        for label, value in extra_stats.items():
            print(f"  {label + ':':<{lw}}{value}")

    print("=" * W)
    print()


def print_family_info_table(
    *,
    auto_family: ModelFamily | None = None,
    explicit_family: ModelFamily | None = None,
    y: Any | None = None,
    advisory: list[str] | None = None,
    title: str = "Family Resolution",
) -> None:
    """Print family resolution and properties in a formatted ASCII table.

    Shows the result of auto-detection (if performed), the explicitly
    selected family, and key family properties.  Any advisory messages
    (e.g. warnings captured from ``resolve_family``) are displayed as
    clean notes inside the table instead of raw stderr warnings.

    Args:
        auto_family: Family instance returned by
            ``resolve_family("auto", y)``.  Omit if auto-detection
            was not tested or if *y* is passed.
        explicit_family: Family instance actually used for analysis
            (e.g. ``resolve_family("poisson", y)``).  Omit if only
            auto-detection is shown.
        y: Optional outcome vector/array. When provided, automatically resolves
            auto-detection and captures any count or data-type advisories into the
            table's Notes section.
        advisory: Optional list of advisory strings (e.g. captured
            warning messages) to display in the Notes section.
        title: Title for the output table.
    """
    if y is not None:
        y_arr = np.ravel(np.asarray(y))
        res_ctx = FitContext()
        if auto_family is None:
            auto_family = resolve_family("auto", y_arr, ctx=res_ctx)
        if res_ctx.warnings_captured:
            advisory = list(advisory or [])
            for w in res_ctx.warnings_captured:
                if w not in advisory:
                    advisory.append(w)

    W = 80
    lw = 22  # label column width

    family = explicit_family or auto_family
    if family is None:
        return

    # ── Title ──────────────────────────────────────────────────── #

    print("=" * W)
    for line in textwrap.wrap(title, width=W - 2):
        print(f"{line:^{W}}")
    print("=" * W)

    # ── Resolution results ─────────────────────────────────────── #

    if auto_family is not None:
        print(f"  {'Auto-detect:':<{lw}}{auto_family.name!r}")

    if explicit_family is not None:
        print(f"  {'Explicit:':<{lw}}{explicit_family.name!r}")

    # ── Family properties ──────────────────────────────────────── #

    print("-" * W)
    print(f"  {'Residual Type:':<{lw}}{family.residual_type}")
    print(f"  {'Direct Permutation:':<{lw}}{family.direct_permutation}")
    print(f"  {'Metric Label:':<{lw}}{family.metric_label}")

    # ── Notes ──────────────────────────────────────────────────── #

    if advisory:
        print("-" * W)
        print("Notes")
        print("-" * W)
        for note in advisory:
            print(_wrap(f"  [!] {note}", width=W, indent=6))

    print("=" * W)
    print()


def print_compatibility_table(
    family: str | ModelFamily,
    *,
    methods: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Print method compatibility matrix for a model family in a formatted ASCII table.

    Displays which permutation methods are supported versus incompatible for the
    specified family, explaining the mathematical or structural reason for any
    incompatible combinations (e.g. why residual-based methods are rejected for
    direct-permutation families).

    Args:
        family: ModelFamily instance or family name string (e.g. 'ordinal').
        methods: Optional list of methods to check. Defaults to standard methods.
        title: Title for the output table. If None, generated automatically.
    """
    from ._validation import validate_compatibility

    if isinstance(family, str):
        fam_name = family
        fam = resolve_family(family)
    else:
        fam_name = family.name
        fam = family

    model_name = _FAMILY_DISPLAY_NAMES.get(fam_name, fam_name.replace("_", " ").title())

    W = 80
    mc_w = 23
    sc_w = 16

    if title is None:
        title = f"Method Compatibility \u2014 {model_name}"

    print("=" * W)
    for line in textwrap.wrap(title, width=W - 2):
        print(f"{line:^{W}}")
    g_banner = "[ Permutation Method \u00d7 Model Family Compatibility Matrix ]"
    print(f"{g_banner:^{W}}")
    print("=" * W)

    hdr = f"{'Method':<{mc_w}}{'Status':<{sc_w}}{'Details'}"
    print(hdr)
    print("-" * W)

    if methods is None:
        methods = [
            "manly",
            "manly_joint",
            "kennedy",
            "kennedy_joint",
            "score",
            "score_joint",
            "freedman_lane",
            "freedman_lane_joint",
            "ter_braak",
        ]
        if fam_name.endswith("_mixed"):
            methods.append("score_exact")

    _METHOD_DETAILS = {
        "manly": "Direct Y permutation (exact)",
        "manly_joint": "Direct Y joint deviance test",
        "kennedy": "Exposure-residual permutation",
        "kennedy_joint": "Exposure-residual joint test",
        "score": "Rao score projection (Fisher info)",
        "score_joint": "Score joint deviance test",
        "freedman_lane": "Reduced-model residual permutation",
        "freedman_lane_joint": "Reduced-model residual joint test",
        "ter_braak": "Full-model residual permutation",
        "score_exact": "PQL-fixed IRLS refits",
    }

    is_glmm = fam_name.endswith("_mixed") and fam_name != "linear_mixed"

    for m in methods:
        issues = validate_compatibility(
            m,
            fam_name,
            direct_permutation=fam.direct_permutation,
            is_glmm=is_glmm,
        )
        errors = [i for i in issues if i.level == "error"]
        if errors:
            status = "Incompatible"
            if fam.direct_permutation and m in (
                "freedman_lane",
                "freedman_lane_joint",
                "ter_braak",
            ):
                detail = "Requires residuals (discrete Y has none)"
            elif is_glmm:
                detail = "Iterative GLMM refit unsupported"
            else:
                detail = errors[0].message
        else:
            status = "Supported"
            detail = _METHOD_DETAILS.get(m, "Supported")

        row_prefix = f"{m:<{mc_w}}{status:<{sc_w}}"
        detail_w = W - mc_w - sc_w
        if len(detail) > detail_w:
            detail = detail[: detail_w - 3] + "..."
        print(f"{row_prefix}{detail}")

    print("=" * W)
    print()


def print_protocol_usage_table(
    result: IndividualTestResult | JointTestResult,
    *,
    title: str | None = None,
) -> None:
    """Print observed-model artifacts from a completed test result.

    Reads the :attr:`FitContext` attached to *result* and renders
    family properties, observed-fit artifacts (coefficients,
    predictions, residuals, fit metric), model diagnostics, and
    inference metadata in a formatted ASCII table.

    All data comes from the pipeline's natural computation — nothing
    is re-computed.

    Args:
        result: A completed ``IndividualTestResult`` or
            ``JointTestResult`` with a ``.context`` attribute.
        title: Optional override for the table title.  Defaults to
            ``"<FamilyName> Protocol Summary"``.

    Raises:
        ValueError: If *result* has no attached context.
    """
    ctx: FitContext | None = getattr(result, "context", None)
    if ctx is None:
        raise ValueError(
            "Result has no attached FitContext.  Ensure the result was "
            "produced by randomization_test_regression()."
        )

    W = 80
    VAL_COL = 30
    lw_top = VAL_COL - 2  # 28 chars for 2-space indented top-level items
    lw_sub = VAL_COL - 4  # 26 chars for 4-space indented sub-items
    family_name = ctx.family_name or "Unknown"

    if title is None:
        title = f"{family_name.title()}Family Protocol Summary"

    # ── Title ──────────────────────────────────────────────────── #

    print("=" * W)
    for line in textwrap.wrap(title, width=W - 2):
        print(f"{line:^{W}}")
    g_banner = "[ Execution & Protocol Artifacts ]"
    print(f"{g_banner:^{W}}")
    print("=" * W)

    # ── Family properties ─────────────────────────────────────── #

    print(f"  {'Name:':<{lw_top}}{family_name}")
    print(f"  {'Residual Type:':<{lw_top}}{ctx.residual_type or 'N/A'}")
    print(f"  {'Direct Permutation:':<{lw_top}}{ctx.direct_permutation}")
    print(f"  {'Metric Label:':<{lw_top}}{ctx.metric_label or 'N/A'}")

    # ── Observed fit ───────────────────────────────────────────── #

    print("-" * W)
    print("  Observed Fit")
    print("-" * W)

    # Coefficients with feature names
    if ctx.coefficients is not None:
        coefs = ctx.coefficients
        names = ctx.feature_names or [f"x{i}" for i in range(len(coefs))]
        print("  Coefficients:")
        for _i, (name, c) in enumerate(zip(names, coefs, strict=False)):
            trunc_name = _truncate(name, 20)
            print(f"    {trunc_name + ':':<{lw_sub}}{c:.6f}")

    # Predictions
    if ctx.predictions is not None:
        preds = ctx.predictions
        print(f"  {'Pred Range:':<{lw_top}}[{preds.min():.4f}, {preds.max():.4f}]")
        print(f"  {'Pred Mean:':<{lw_top}}{preds.mean():.4f}")

    # Residuals (may be None for direct-permutation families)
    if ctx.residuals is not None:
        resids = ctx.residuals
        print(f"  {'Mean |Residual|:':<{lw_top}}{np.mean(np.abs(resids)):.4f}")
    elif ctx.direct_permutation:
        print(f"  {'Residuals:':<{lw_top}}N/A (direct permutation)")

    # Fit metric
    if ctx.fit_metric_value is not None:
        label = ctx.metric_label or "Fit Metric"
        print(f"  {label + ':':<{lw_top}}{ctx.fit_metric_value:.4f}")

    # ── Diagnostics ────────────────────────────────────────────── #

    if ctx.diagnostics:
        print("-" * W)
        print("  Diagnostics")
        print("-" * W)
        # Skip redundant raw dictionary bundles and metrics already reported in primary tables
        _SKIP_DIAG_KEYS = {
            "n_observations",
            "n_features",
            "aic",
            "bic",
            "r_squared",
            "r_squared_adj",
            "f_statistic",
            "f_p_value",
            "glmm_gof",
            "lmm_gof",
        }
        for key, val in ctx.diagnostics.items():
            if key in _SKIP_DIAG_KEYS:
                continue
            display_key = key.replace("_", " ").title()
            if key == "variance_components" and isinstance(val, dict):
                # Clean nested rendering of variance components factors
                print("  Variance Components:")
                for factor_line, factor_stat, _ in _format_variance_components(val):
                    print(f"    {factor_line:<{lw_sub}}{factor_stat}")
            elif isinstance(val, float):
                print(f"  {display_key + ':':<{lw_top}}{val:.4f}")
            elif isinstance(val, dict):
                print(f"  {display_key + ':'}")
                for sub_k, sub_v in val.items():
                    sub_label = str(sub_k).replace("_", " ").title() + ":"
                    if isinstance(sub_v, float):
                        print(f"    {sub_label:<{lw_sub}}{sub_v:.4f}")
                    else:
                        print(f"    {sub_label:<{lw_sub}}{sub_v}")
            else:
                print(f"  {display_key + ':':<{lw_top}}{val}")

    # ── Inference ──────────────────────────────────────────────── #

    print("-" * W)
    print("  Inference")
    print("-" * W)

    p_vals = (
        ctx.classical_p_values
        if ctx.classical_p_values is not None
        else getattr(result, "raw_classic_p", None)
    )
    if p_vals is not None:
        names = ctx.feature_names or [f"x{i}" for i in range(len(p_vals))]
        print("  Classical P-values:")
        for name, p_v in zip(names, p_vals, strict=False):
            trunc_name = _truncate(name, 20)
            p_str = f"{p_v:.6f}" if not np.isnan(p_v) else "NaN"
            print(f"    {trunc_name + ':':<{lw_sub}}{p_str}")

    cells = ctx.exchangeability_cells
    if cells is None:
        print(f"  {'Exchangeability:':<{lw_top}}global (None)")
    else:
        n_cells = len(np.unique(cells))
        print(f"  {'Exchangeability:':<{lw_top}}{n_cells} cell(s)")

    # ── Permutation metadata ───────────────────────────────────── #

    print("-" * W)
    print("  Permutation Config")
    print("-" * W)

    print(f"  {'Method:':<{lw_top}}{ctx.method or result.method}")
    print(f"  {'Backend:':<{lw_top}}{ctx.backend or 'N/A'}")
    print(
        f"  {'N Randomizations:':<{lw_top}}{ctx.n_randomizations or result.n_randomizations}"
    )
    if ctx.permutation_strategy:
        print(f"  {'Strategy:':<{lw_top}}{ctx.permutation_strategy}")
    if ctx.confounders:
        print(f"  {'Confounders:':<{lw_top}}{', '.join(ctx.confounders)}")

    # Batch-fit convergence
    if ctx.batch_shape is not None:
        B, p = ctx.batch_shape
        print(f"  {'Batch Shape:':<{lw_top}}({B}, {p})")
    if ctx.convergence_count is not None and ctx.batch_shape is not None:
        total = ctx.batch_shape[0]
        print(
            f"  {'Convergence:':<{lw_top}}{ctx.convergence_count}/{total} fits converged"
        )

    print("=" * W)
    print()

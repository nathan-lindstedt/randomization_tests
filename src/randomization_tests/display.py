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

from .families import ModelFamily

if TYPE_CHECKING:
    from ._context import FitContext
    from ._results import IndividualTestResult, JointTestResult


def _truncate(name: str, max_len: int) -> str:
    """Truncate *name* to *max_len*, appending ``'...'`` if needed."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


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


def _recommend_n_permutations(
    p_hat: float,
    threshold: float,
    alpha: float = 0.05,
) -> int:
    """Minimum *B* so the Clopper-Pearson CI no longer straddles *threshold*.

    Uses the normal approximation to the Clopper-Pearson half-width,
    ``z_{1-α/2} √{p(1-p)/B}``, and solves for *B* such that
    the half-width is at most ``|p_hat - threshold|``.

    The result is rounded up and clamped to ``[100, 10_000_000]``.

    Args:
        p_hat: Observed empirical p-value.
        threshold: The nearest significance threshold that the CI
            straddles.
        alpha: Confidence level for the CI (default 0.05).

    Returns:
        Recommended minimum number of permutations.
    """
    gap = abs(p_hat - threshold)
    if gap < 1e-12:
        return 10_000_000  # effectively tied — need extreme B
    z = _sp_stats.norm.ppf(1 - alpha / 2)
    b_min = math.ceil((z**2) * p_hat * (1 - p_hat) / (gap**2))
    return max(100, min(b_min, 10_000_000))  # type: ignore[no-any-return]


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


def print_results_table(
    results: IndividualTestResult,
    *,
    title: str = "Permutation Test Results",
) -> None:
    """Print regression results in a formatted ASCII table similar to statsmodels.

    All metadata (family, feature names, target name) is extracted
    from the result object — no additional context is needed.

    Args:
        results: Typed result object returned by
            :func:`~randomization_tests.permutation_test_regression`.
        title: Title for the output table.
    """
    family: ModelFamily = results.family
    feature_names: list[str] = results.feature_names
    target_name: str | None = results.target_name

    print("=" * 80)
    for line in textwrap.wrap(title, width=78):
        print(f"{line:^80}")
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
        coef_str = f"{coefs[i]:>9.4f}"
        print(f"{trunc_feat:<{fc}}{coef_str}  {emp_p[i]:>23} {asy_p[i]:>23}")

        # Sub-row: ± margin from the Clopper-Pearson CI, aligned
        # under the empirical p-value column.
        if pval_ci is not None and i < len(pval_ci):
            lo, hi = pval_ci[i]
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

    # Kennedy / Freedman–Lane without confounders is valid but unusual —
    # surface a note so the user knows ter Braak may be more appropriate.
    method = getattr(results, "method", "")
    confounders = getattr(results, "confounders", None)
    if method in ("kennedy", "freedman_lane") and not confounders:
        method_label = "Freedman\u2013Lane" if method == "freedman_lane" else "Kennedy"
        notes.append(
            f"{method_label} method called without confounders \u2014 all "
            "features will be tested unconditionally. Consider 'ter_braak' "
            "for unconditional tests."
        )

    # Recommend larger n_permutations for borderline cases (Step 25)
    if borderline_features:
        ci_alpha = ci.get("confidence_level", 0.95)
        alpha = 1 - ci_alpha if ci_alpha > 0.5 else ci_alpha
        b_recs = [
            (feat, _recommend_n_permutations(p_hat, t, alpha))
            for feat, p_hat, t in borderline_features
        ]
        max_b = max(b for _, b in b_recs)
        feat_list = ", ".join(feat for feat, _ in b_recs)
        notes.append(
            f"Consider n_permutations \u2265 {max_b:,} to resolve "
            f"borderline p-values for: {feat_list}."
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
    print()


def print_joint_results_table(
    results: JointTestResult,
    *,
    title: str = "Joint Permutation Test Results",
) -> None:
    """Print joint test results in a formatted ASCII table.

    All metadata (family, target name) is extracted from the result
    object — no additional context is needed.

    Args:
        results: Typed result object returned by
            :func:`~randomization_tests.permutation_test_regression` with
            ``method='kennedy_joint'`` or ``method='freedman_lane_joint'``.
        title: Title for the output table.
    """
    family: ModelFamily = results.family
    target_name: str | None = results.target_name

    print("=" * 80)
    for line in textwrap.wrap(title, width=78):
        print(f"{line:^80}")
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
    method = getattr(results, "method", "")
    confounders = getattr(results, "confounders", None)
    if method in ("kennedy_joint", "freedman_lane_joint") and not confounders:
        method_label = (
            "Freedman\u2013Lane" if method == "freedman_lane_joint" else "Kennedy"
        )
        print("-" * 80)
        print("Notes")
        print("-" * 80)
        print(
            _wrap(
                f"  [!] {method_label} method called without confounders \u2014 all "
                "features will be tested unconditionally. Consider 'ter_braak' "
                "for unconditional tests.",
                width=80,
                indent=6,
            )
        )

    print("=" * 80)
    print(
        f"(***) p < {results.p_value_threshold_three}   "
        f"(**) p < {results.p_value_threshold_two}   "
        f"(*) p < {results.p_value_threshold_one}   "
        f"(ns) p >= {results.p_value_threshold_one}"
    )
    print("Omnibus test: single p-value for all tested features combined.")
    print()


def print_diagnostics_table(
    results: IndividualTestResult,
    *,
    title: str = "Extended Diagnostics",
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
            :func:`~randomization_tests.permutation_test_regression`.
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
            std_c = f"{std_coefs[i]:>10.4f}" if i < len(std_coefs) else f"{'':>10}"
        elif show_pval_ci:
            std_c = f"{std_coefs[i]:>9.4f}" if i < len(std_coefs) else f"{'':>9}"
        else:
            std_c = f"{std_coefs[i]:>10.4f}" if i < len(std_coefs) else f"{'':>10}"

        # VIF — flag problematic values for Notes
        if i < len(vifs):
            v = vifs[i]
            if show_exp_r2 or show_pval_ci:
                vif_str = f"{v:>8.2f}" if v < 1000 else f"{'> 1000':>8}"
            else:
                vif_str = f"{v:>10.2f}" if v < 1000 else f"{'> 1000':>10}"
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

    print("-" * W)
    print("Model-level Diagnostics")
    print("-" * W)

    diag_lines, diag_notes = family.display_diagnostics(ext)
    for label, value in diag_lines:
        print(f"  {label:<{lw}}{value}")
    notes.extend(diag_notes)

    # Cook's distance
    cd = ext.get("cooks_distance", {})
    if cd:
        n_inf = cd.get("n_influential", 0)
        thresh = cd.get("threshold", 0)
        cd_val = f"{n_inf}   threshold = {thresh:.4f}"
        cd_label = "Cook's D (> 4/n):"
        print(f"  {cd_label:<{lw}}{cd_val}")
        if isinstance(n_inf, (int, float)) and n_inf > 0:
            notes.append(
                f"{int(n_inf)} obs. with Cook's D > {thresh:.4f} "
                f"(4/n); results may be sensitive to "
                f"influential points."
            )

    # Permutation coverage
    pc = ext.get("permutation_coverage", {})
    if pc:
        cov_str = pc.get("coverage_str", "N/A")
        print(f"  {'Permutation coverage:':<{lw}}{cov_str}")

    # ── Notes ──────────────────────────────────────────────────── #

    if notes:
        print("-" * W)
        print("Notes")
        print("-" * W)
        for note in notes:
            print(_wrap(f"  [!] {note}", width=W, indent=6))

    print("=" * W)
    print()


def print_confounder_table(
    confounder_results: dict[str, Any] | object,
    title: str = "Confounder Identification Results",
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
    n_observations: int,
    n_features: int,
    feature_names: list[str] | None = None,
    target_name: str | None = None,
    target_description: str | None = None,
    y_range: tuple[float, float] | None = None,
    y_mean: float | None = None,
    y_var: float | None = None,
    extra_stats: dict[str, str] | None = None,
    title: str = "Dataset Information",
) -> None:
    """Print dataset metadata in a formatted ASCII table.

    Displays dataset name, dimensions, feature names, target variable
    information, and optional outcome statistics in a bordered table
    matching the visual style of other ``print_*`` functions.

    Args:
        name: Dataset name (e.g., ``'Abalone'``).
        n_observations: Number of observations (rows).
        n_features: Number of features (columns in X).
        feature_names: Optional list of feature names. If provided,
            they are displayed as a comma-separated list, truncated
            if too long.
        target_name: Optional name of the target variable (e.g.,
            ``'Rings'``).
        target_description: Optional description of the target (e.g.,
            ``'growth-ring count'``).
        y_range: Optional tuple ``(min, max)`` of outcome values.
        y_mean: Optional mean of the outcome.
        y_var: Optional variance of the outcome.
        extra_stats: Optional dict of additional statistics to display
            as ``{label: value}`` pairs (e.g., ``{'Var/Mean': '1.05'}``).
        title: Title for the output table.
    """
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
        print(f"  {'Y Range:':<{lw}}[{y_range[0]}, {y_range[1]}]")

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
            was not tested.
        explicit_family: Family instance actually used for analysis
            (e.g. ``resolve_family("poisson", y)``).  Omit if only
            auto-detection is shown.
        advisory: Optional list of advisory strings (e.g. captured
            warning messages) to display in the Notes section.
        title: Title for the output table.
    """
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
            "produced by permutation_test_regression()."
        )

    W = 80
    lw = 22  # label column width
    family_name = ctx.family_name or "Unknown"

    if title is None:
        title = f"{family_name.title()}Family Protocol Summary"

    # ── Title ──────────────────────────────────────────────────── #

    print("=" * W)
    for line in textwrap.wrap(title, width=W - 2):
        print(f"{line:^{W}}")
    print("=" * W)

    # ── Family properties ─────────────────────────────────────── #

    print(f"  {'Name:':<{lw}}{family_name}")
    print(f"  {'Residual Type:':<{lw}}{ctx.residual_type or 'N/A'}")
    print(f"  {'Direct Permutation:':<{lw}}{ctx.direct_permutation}")
    print(f"  {'Metric Label:':<{lw}}{ctx.metric_label or 'N/A'}")

    # ── Observed fit ───────────────────────────────────────────── #

    print("-" * W)
    print("  Observed Fit")
    print("-" * W)

    # Coefficients with feature names
    if ctx.coefficients is not None:
        coefs = ctx.coefficients
        names = ctx.feature_names or [f"x{i}" for i in range(len(coefs))]
        print(
            f"    {'Coefs:':<{lw}}{np.array2string(coefs, precision=4, suppress_small=True)}"
        )
        for _i, (name, c) in enumerate(zip(names, coefs, strict=False)):
            trunc_name = _truncate(name, 20)
            print(f"      {trunc_name + ':':<{lw}}{c:.6f}")

    # Predictions
    if ctx.predictions is not None:
        preds = ctx.predictions
        print(f"    {'Pred Range:':<{lw}}[{preds.min():.4f}, {preds.max():.4f}]")
        print(f"    {'Pred Mean:':<{lw}}{preds.mean():.4f}")

    # Residuals (may be None for direct-permutation families)
    if ctx.residuals is not None:
        resids = ctx.residuals
        print(f"    {'Mean |Residual|:':<{lw}}{np.mean(np.abs(resids)):.4f}")
    elif ctx.direct_permutation:
        print(f"    {'Residuals:':<{lw}}N/A (direct permutation)")

    # Fit metric
    if ctx.fit_metric_value is not None:
        label = ctx.metric_label or "Fit Metric"
        print(f"    {label + ':':<{lw}}{ctx.fit_metric_value:.4f}")

    # ── Diagnostics ────────────────────────────────────────────── #

    if ctx.diagnostics:
        print("-" * W)
        print("  Diagnostics")
        print("-" * W)
        for key, val in ctx.diagnostics.items():
            display_key = key.replace("_", " ").title()
            if isinstance(val, float):
                print(f"    {display_key + ':':<{lw}}{val:.4f}")
            else:
                print(f"    {display_key + ':':<{lw}}{val}")

    # ── Inference ──────────────────────────────────────────────── #

    print("-" * W)
    print("  Inference")
    print("-" * W)

    if ctx.classical_p_values is not None:
        p_vals = ctx.classical_p_values
        print(
            f"    {'Classical P-values:':<{lw}}{np.array2string(p_vals, precision=6, suppress_small=True)}"
        )
    else:
        # Fall back to result's own classical p-values
        if hasattr(result, "raw_classic_p"):
            p_vals = result.raw_classic_p
            print(
                f"    {'Classical P-values:':<{lw}}{np.array2string(p_vals, precision=6, suppress_small=True)}"
            )

    cells = ctx.exchangeability_cells
    if cells is None:
        print(f"    {'Exchangeability:':<{lw}}global (None)")
    else:
        n_cells = len(np.unique(cells))
        print(f"    {'Exchangeability:':<{lw}}{n_cells} cell(s)")

    # ── Permutation metadata ───────────────────────────────────── #

    print("-" * W)
    print("  Permutation Config")
    print("-" * W)

    print(f"    {'Method:':<{lw}}{ctx.method or result.method}")
    print(f"    {'Backend:':<{lw}}{ctx.backend or 'N/A'}")
    print(f"    {'N Permutations:':<{lw}}{ctx.n_permutations or result.n_permutations}")
    if ctx.permutation_strategy:
        print(f"    {'Strategy:':<{lw}}{ctx.permutation_strategy}")
    if ctx.confounders:
        print(f"    {'Confounders:':<{lw}}{', '.join(ctx.confounders)}")

    # Batch-fit convergence
    if ctx.batch_shape is not None:
        B, p = ctx.batch_shape
        print(f"    {'Batch Shape:':<{lw}}({B}, {p})")
    if ctx.convergence_count is not None and ctx.batch_shape is not None:
        total = ctx.batch_shape[0]
        print(
            f"    {'Convergence:':<{lw}}{ctx.convergence_count}/{total} fits converged"
        )

    print("=" * W)
    print()

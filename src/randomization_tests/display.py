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

import textwrap


def _truncate(name: str, max_len: int) -> str:
    """Truncate *name* to *max_len*, appending ``'...'`` if needed."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


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


def print_results_table(
    results: dict,
    feature_names: list[str],
    target_name: str | None = None,
    title: str = "Permutation Test Results",
) -> None:
    """Print regression results in a formatted ASCII table similar to statsmodels.

    Args:
        results: Results dictionary returned by
            :func:`~randomization_tests.permutation_test_regression`.
        feature_names: Names of the features/predictors.
        target_name: Name of the target variable.
        title: Title for the output table.
    """
    print("=" * 80)
    for line in textwrap.wrap(title, width=78):
        print(f"{line:^80}")
    print("=" * 80)

    diag = results.get("diagnostics", {})
    model_type = results["model_type"]
    col1 = 40
    col2 = 38

    if target_name:
        trunc_target = _truncate(target_name, 20)
        print(
            f"{'Dep. Variable:':<16}{trunc_target:<{col1 - 16}}"
            f"{'No. Observations:':>{col2 - 11}} {diag.get('n_observations', 'N/A'):>10}"
        )
    print(
        f"{'Model Type:':<16}{model_type:<{col1 - 16}}"
        f"{'No. Features:':>{col2 - 11}} {diag.get('n_features', 'N/A'):>10}"
    )
    print(
        f"{'Method:':<16}{results['method']:<{col1 - 16}}"
        f"{'AIC:':>{col2 - 11}} {diag.get('aic', 'N/A'):>10}"
    )

    if model_type == "linear":
        print(
            f"{'R-squared:':<16}{diag.get('r_squared', 'N/A'):<{col1 - 16}}"
            f"{'BIC:':>{col2 - 11}} {diag.get('bic', 'N/A'):>10}"
        )
        print(
            f"{'Adj. R-squared:':<16}{diag.get('r_squared_adj', 'N/A'):<{col1 - 16}}"
            f"{'F-statistic:':>{col2 - 11}} {diag.get('f_statistic', 'N/A'):>10}"
        )
        f_p = diag.get("f_p_value", None)
        f_p_str = f"{f_p:.4e}" if f_p is not None else "N/A"
        print(f"{'':<{col1}}{'Prob (F-stat):':>{col2 - 11}} {f_p_str:>10}")
    else:
        print(
            f"{'Pseudo R-sq:':<16}{diag.get('pseudo_r_squared', 'N/A'):<{col1 - 16}}"
            f"{'BIC:':>{col2 - 11}} {diag.get('bic', 'N/A'):>10}"
        )
        print(
            f"{'Log-Likelihood:':<16}{diag.get('log_likelihood', 'N/A'):<{col1 - 16}}"
            f"{'LL-Null:':>{col2 - 11}} {diag.get('log_likelihood_null', 'N/A'):>10}"
        )
        llr_p = diag.get("llr_p_value", None)
        llr_p_str = f"{llr_p:.4e}" if llr_p is not None else "N/A"
        print(f"{'':<{col1}}{'LLR p-value:':>{col2 - 11}} {llr_p_str:>10}")

    print("-" * 80)

    fc = 25
    stat_label = "t" if model_type == "linear" else "z"
    emp_hdr = f"P>|{stat_label}| (Emp)"
    asy_hdr = f"P>|{stat_label}| (Asy)"
    print(f"{'Feature':<{fc}} {'Coef':>12} {emp_hdr:>18} {asy_hdr:>18}")
    print("-" * 80)

    coefs = results["model_coefs"]
    emp_p = results["permuted_p_values"]
    asy_p = results["classic_p_values"]

    for i, feat in enumerate(feature_names):
        trunc_feat = _truncate(feat, fc)
        coef_str = f"{coefs[i]:>12.4f}"
        print(f"{trunc_feat:<{fc}} {coef_str} {emp_p[i]:>18} {asy_p[i]:>18}")

    print("=" * 80)
    print(
        f"(*) p < {results['p_value_threshold_one']}   "
        f"(**) p < {results['p_value_threshold_two']}   "
        f"(ns) p >= {results['p_value_threshold_one']}"
    )
    print()


def print_joint_results_table(
    results: dict,
    target_name: str | None = None,
    title: str = "Joint Permutation Test Results",
) -> None:
    """Print joint test results in a formatted ASCII table.

    Args:
        results: Results dictionary returned by
            :func:`~randomization_tests.permutation_test_regression` with
            ``method='kennedy_joint'``.
        target_name: Name of the target variable.
        title: Title for the output table.
    """
    print("=" * 80)
    for line in textwrap.wrap(title, width=78):
        print(f"{line:^80}")
    print("=" * 80)

    diag = results.get("diagnostics", {})
    model_type = results["model_type"]
    col1 = 40
    col2 = 38

    if target_name:
        trunc_target = _truncate(target_name, 20)
        print(
            f"{'Dep. Variable:':<16}{trunc_target:<{col1 - 16}}"
            f"{'No. Observations:':>{col2 - 11}} {diag.get('n_observations', 'N/A'):>10}"
        )
    print(
        f"{'Model Type:':<16}{model_type:<{col1 - 16}}"
        f"{'No. Features:':>{col2 - 11}} {diag.get('n_features', 'N/A'):>10}"
    )
    print(
        f"{'Method:':<16}{results['method']:<{col1 - 16}}"
        f"{'AIC:':>{col2 - 11}} {diag.get('aic', 'N/A'):>10}"
    )

    if model_type == "linear":
        print(
            f"{'R-squared:':<16}{diag.get('r_squared', 'N/A'):<{col1 - 16}}"
            f"{'BIC:':>{col2 - 11}} {diag.get('bic', 'N/A'):>10}"
        )
        print(
            f"{'Adj. R-squared:':<16}{diag.get('r_squared_adj', 'N/A'):<{col1 - 16}}"
            f"{'F-statistic:':>{col2 - 11}} {diag.get('f_statistic', 'N/A'):>10}"
        )
        f_p = diag.get("f_p_value", None)
        f_p_str = f"{f_p:.4e}" if f_p is not None else "N/A"
        print(f"{'':<{col1}}{'Prob (F-stat):':>{col2 - 11}} {f_p_str:>10}")
    else:
        print(
            f"{'Pseudo R-sq:':<16}{diag.get('pseudo_r_squared', 'N/A'):<{col1 - 16}}"
            f"{'BIC:':>{col2 - 11}} {diag.get('bic', 'N/A'):>10}"
        )
        print(
            f"{'Log-Likelihood:':<16}{diag.get('log_likelihood', 'N/A'):<{col1 - 16}}"
            f"{'LL-Null:':>{col2 - 11}} {diag.get('log_likelihood_null', 'N/A'):>10}"
        )
        llr_p = diag.get("llr_p_value", None)
        llr_p_str = f"{llr_p:.4e}" if llr_p is not None else "N/A"
        print(f"{'':<{col1}}{'LLR p-value:':>{col2 - 11}} {llr_p_str:>10}")

    print(f"{'Metric:':<16}{results['metric_type']}")
    print("-" * 80)

    feat_list = ", ".join(_truncate(f, 25) for f in results["features_tested"])
    print(_wrap(f"Features Tested: {feat_list}", width=80, indent=17))
    if results["confounders"]:
        conf_list = ", ".join(_truncate(c, 25) for c in results["confounders"])
        print(_wrap(f"Confounders: {conf_list}", width=80, indent=13))
    print("-" * 80)

    print(f"{'Observed Improvement:':<30} {results['observed_improvement']:>12.4f}")
    print(f"{'Joint p-Value:':<30} {results['p_value_str']:>12}")

    print("=" * 80)
    print(
        f"(*) p < {results['p_value_threshold_one']}   "
        f"(**) p < {results['p_value_threshold_two']}   "
        f"(ns) p >= {results['p_value_threshold_one']}"
    )
    print("Omnibus test: single p-value for all tested features combined.")
    print()


def print_diagnostics_table(
    results: dict,
    feature_names: list[str],
    title: str = "Extended Diagnostics",
) -> None:
    """Print extended model diagnostics in a formatted ASCII table.

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
       (Breusch-Pagan for linear, deviance residuals for logistic),
       Cook's distance influence counts, and permutation coverage.
    4. **Notes** (conditional) — plain-language warnings printed only
       when a diagnostic flags a potential concern.

    Args:
        results: Results dictionary returned by
            :func:`~randomization_tests.permutation_test_regression`.
            Must contain ``extended_diagnostics`` and ``model_type``.
        feature_names: Names of the features/predictors.
        title: Title for the output table.
    """
    ext = results.get("extended_diagnostics")
    if ext is None:
        return

    model_type = results["model_type"]
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

    print("Per-predictor Diagnostics")
    print("-" * W)
    if show_exp_r2:
        _exp_hdr = "Exp R\u00b2"
        print(
            f"{'Feature':<{fc}}"
            f"{'Std Coef':>9} "
            f"{'VIF':>9} "
            f"{'MC SE':>9} "
            f"{_exp_hdr:>7}   "
            f"{'Emp vs Asy':>10}"
        )
    else:
        print(
            f"{'Feature':<{fc}}"
            f"{'Std Coef':>9} "
            f"{'VIF':>9} "
            f"{'MC SE':>9}   "
            f"{'Emp vs Asy':>10}"
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
        if i < len(std_coefs):
            std_c = f"{std_coefs[i]:>9.4f}"
        else:
            std_c = f"{'':>9}"

        # VIF — flag problematic values for Notes
        if i < len(vifs):
            v = vifs[i]
            vif_str = f"{v:>9.2f}" if v < 1000 else f"{'> 1000':>9}"
            if v > 10:
                vif_problems.append((feat, v, "severe"))
            elif v > 5:
                vif_problems.append((feat, v, "moderate"))
        else:
            vif_str = f"{'':>9}"

        # Monte Carlo SE — 4 dp sufficient for precision assessment
        if i < len(mc_ses):
            mc_str = f"{mc_ses[i]:>9.4f}"
        else:
            mc_str = f"{'':>9}"

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
                er2_str = f"{'—':>7}"
            else:
                er2_str = f"{er2:>7.4f}"
                if er2 > 0.99:
                    exp_r2_problems.append((feat, er2))

        # Divergence flag
        flag = div_flags[i] if i < len(div_flags) else ""
        if flag:
            has_divergent = True
        div_str = f"{flag:>10}"

        if show_exp_r2:
            print(
                f"{trunc_feat:<{fc}}{std_c} {vif_str} "
                f"{mc_str} {er2_str}   {div_str}"
            )
        else:
            print(f"{trunc_feat:<{fc}}{std_c} {vif_str} {mc_str}   {div_str}")

    # ── Legend ──────────────────────────────────────────────────── #

    print("-" * W)
    print(
        "  Std Coef: effect per SD.  "
        "VIF: collinearity (> 5 moderate, > 10 severe)."
    )
    print(
        "  MC SE: p-value precision "
        "(increase B if large relative to p)."
    )
    if show_exp_r2:
        print(
            "  Exp R\u00b2: variance of X_j explained by "
            "confounders (> 0.99 = collinear)."
        )
    if has_divergent:
        print(
            "  DIVERGENT = permutation and classical "
            "p-values disagree at alpha."
        )

    # Collect VIF notes
    if vif_problems:
        parts = ", ".join(
            f"{name} = {val:.2f} ({label})"
            for name, val, label in vif_problems
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
        parts = ", ".join(
            f"{name} = {val:.4f}" for name, val in exp_r2_problems
        )
        notes.append(
            f"Exp R\u00b2: {parts}. Near-collinear with "
            f"confounders; permuted coefficients are unstable "
            f"and p-values are inflated."
        )

    # ── Model-level Diagnostics ────────────────────────────────── #

    lw = 28  # label column width (including 2-space indent)

    print("-" * W)
    print("Model-level Diagnostics")
    print("-" * W)

    if model_type == "linear":
        bp = ext.get("breusch_pagan", {})
        if bp:
            print(
                f"{'  Breusch-Pagan LM:':<{lw}}"
                f"{bp.get('lm_stat', 'N/A'):>10}   "
                f"p = {_fmt_p(bp.get('lm_p_value'))}"
            )
            print(
                f"{'  Breusch-Pagan F:':<{lw}}"
                f"{bp.get('f_stat', 'N/A'):>10}   "
                f"p = {_fmt_p(bp.get('f_p_value'))}"
            )
            bp_p = bp.get("lm_p_value")
            if bp_p is not None and bp_p < 0.05:
                notes.append(
                    f"Breusch-Pagan p = {bp_p:.4f}: "
                    f"heteroscedastic residuals detected; "
                    f"exchangeability assumption may be violated."
                )
    else:
        dr = ext.get("deviance_residuals", {})
        if dr:
            print(
                f"{'  Deviance resid. mean:':<{lw}}"
                f"{dr.get('mean', 'N/A'):>10}"
            )
            print(
                f"{'  Deviance resid. var:':<{lw}}"
                f"{dr.get('variance', 'N/A'):>10}"
            )
            print(
                f"{'  |d_i| > 2 count:':<{lw}}"
                f"{dr.get('n_extreme', 'N/A'):>10}"
            )
            print(
                f"{'  Runs test Z:':<{lw}}"
                f"{dr.get('runs_test_z', 'N/A'):>10}   "
                f"p = {_fmt_p(dr.get('runs_test_p'))}"
            )
            n_extreme = dr.get("n_extreme", 0)
            if isinstance(n_extreme, (int, float)) and n_extreme > 0:
                notes.append(
                    f"{int(n_extreme)} obs. with "
                    f"|deviance residual| > 2."
                )
            runs_p = dr.get("runs_test_p")
            if runs_p is not None and runs_p < 0.05:
                notes.append(
                    "Runs test p < 0.05: non-random "
                    "residual pattern detected."
                )

    # Cook's distance
    cd = ext.get("cooks_distance", {})
    if cd:
        n_inf = cd.get("n_influential", 0)
        thresh = cd.get("threshold", 0)
        cd_label = "  Cook's D (> 4/n):"
        print(
            f"{cd_label:<{lw}}"
            f"{n_inf:>10}   "
            f"threshold = {thresh:.4f}"
        )
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
        print(f"{'  Permutation coverage:':<{lw}}{cov_str}")

    # ── Notes ──────────────────────────────────────────────────── #

    if notes:
        print("-" * W)
        print("Notes")
        print("-" * W)
        for note in notes:
            print(_wrap(f"  [!] {note}", width=W, indent=6))

    print("=" * W)
    print()


def _fmt_p(p) -> str:
    """Format a p-value for the diagnostics table."""
    if p is None:
        return "N/A"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def print_confounder_table(
    confounder_results: dict,
    title: str = "Confounder Identification Results",
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
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
    (for one predictor) or a ``dict[str, dict]`` mapping predictor
    names to their individual result dicts (for multiple predictors).

    Args:
        confounder_results: Either a single result dict (must contain
            ``identified_confounders`` and ``predictor`` keys) or a
            dict mapping predictor names to result dicts.
        title: Title for the output table.
        correlation_threshold: The ``|r|`` threshold used during
            screening (displayed in the parameter header).
        p_value_threshold: The screening p-value cutoff (displayed in
            the parameter header).
        n_bootstrap: Bootstrap iterations used for mediation analysis
            (displayed in the parameter header).
        confidence_level: Confidence-interval level used for mediation
            (displayed in the parameter header).
    """
    W = 80

    # ── Normalise input ────────────────────────────────────────── #
    # A single identify_confounders() result has an
    # "identified_confounders" key at the top level.  A multi-
    # predictor dict has predictor names as keys, each mapping to a
    # result dict.  Detecting this lets the function accept both
    # forms without requiring the caller to wrap a single result.
    if "identified_confounders" in confounder_results:
        results_by_pred: dict[str, dict] = {
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
    print(
        f"Screening: |r| >= {correlation_threshold}, "
        f"p < {p_value_threshold}   "
        f"Mediation: BCa bootstrap (B={n_bootstrap}, {ci_pct}% CI)"
    )
    print("-" * W)

    # ── Partition predictors ───────────────────────────────────── #
    # Separate predictors with findings (confounders and/or
    # mediators) from those without, preserving insertion order so
    # the table matches the feature order in the dataset.

    has_findings: dict[str, dict] = {}
    no_findings: list[str] = []
    has_mediators = False

    for pred, res in results_by_pred.items():
        confounders = res.get("identified_confounders", [])
        mediators = res.get("identified_mediators", [])
        if confounders or mediators:
            has_findings[pred] = res
            if mediators:
                has_mediators = True
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
    else:
        # No predictor had any findings at all.
        print("  No confounders or mediators identified for any predictor.")

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

    if notes:
        print("-" * W)
        print("Notes")
        print("-" * W)
        for note in notes:
            print(_wrap(f"  [!] {note}", width=W, indent=6))

    print("=" * W)
    print()

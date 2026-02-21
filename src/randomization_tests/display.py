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


def _truncate(name: str, max_len: int) -> str:
    """Truncate *name* to *max_len*, appending ``'...'`` if needed."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


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
    print(f"{title:^80}")
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
    print(f"{title:^80}")
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
    print(f"Features Tested: {feat_list}")
    if results["confounders"]:
        conf_list = ", ".join(_truncate(c, 25) for c in results["confounders"])
        print(f"Confounders: {conf_list}")
    print("-" * 80)

    print(f"{'Observed Improvement:':<30} {results['observed_improvement']:>12.4f}")
    print(f"{'Joint p-Value:':<30} {results['p_value_str']:>12}")

    print("=" * 80)
    print(
        f"(*) p < {results['p_value_threshold_one']}   "
        f"(**) p < {results['p_value_threshold_two']}   "
        f"(ns) p >= {results['p_value_threshold_one']}"
    )
    print()

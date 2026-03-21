"""v0.4.2 verification gate — mixed-effects contracts + audit/display/polish.

Validates:
  1. Mixed-effects protocol conformance (3 mixed families)
  2. Henderson REML + score projection contracts
  3. GLMM families reject non-score strategies
  4. calibrate() is a formal protocol method on all families
  5. Statistical audit fixes (null_score, clipping, n! overflow, dynamic error)
  6. Confounder display contracts (label, pval_ci masking)
  7. Model diagnostics polish (3-tuple display_diagnostics, coverage keys)
  8. GLMM deviance note
"""

import inspect
import io
import sys

import numpy as np
import pandas as pd

import randomization_tests.confounders as conf_mod
import randomization_tests.families as fam_mod
from randomization_tests import (
    LinearFamily,
    LinearMixedFamily,
    LogisticFamily,
    LogisticMixedFamily,
    MultinomialFamily,
    NegativeBinomialFamily,
    OrdinalFamily,
    PoissonFamily,
    PoissonMixedFamily,
    randomization_test_regression,
)
from randomization_tests._strategies import score as score_mod
from randomization_tests.diagnostics import compute_permutation_coverage
from randomization_tests.display import print_diagnostics_table, print_results_table

print("=" * 64)
print("v0.4.2 VERIFICATION GATE")
print("=" * 64)

passed = 0
failed = 0


def check(label, condition):
    global passed, failed
    if condition:
        print(f"  [PASS] {label}")
        passed += 1
    else:
        print(f"  [FAIL] {label}")
        failed += 1


# ================================================================== #
# 1. Mixed-effects protocol conformance
# ================================================================== #
print("\n--- 1. Mixed-effects protocol conformance ---")

MIXED_FAMILIES = {
    "linear_mixed": LinearMixedFamily,
    "logistic_mixed": LogisticMixedFamily,
    "poisson_mixed": PoissonMixedFamily,
}

PROTOCOL_METHODS = [
    "name",
    "stat_label",
    "display_header",
    "display_diagnostics",
    "compute_extended_diagnostics",
    "fit",
    "predict",
    "fit_metric",
    "score",
    "null_score",
    "calibrate",
    "score_project",
]

for name, cls in MIXED_FAMILIES.items():
    fam = cls()
    check(f"{cls.__name__}.name == '{name}'", fam.name == name)
    for method in PROTOCOL_METHODS:
        check(
            f"{cls.__name__} has {method}",
            hasattr(fam, method),
        )

# ================================================================== #
# 2. calibrate() is a protocol method on all 9+3 families
# ================================================================== #
print("\n--- 2. calibrate() on all families ---")

ALL_FAMILIES = [
    LinearFamily,
    LogisticFamily,
    PoissonFamily,
    NegativeBinomialFamily,
    OrdinalFamily,
    MultinomialFamily,
    LinearMixedFamily,
    LogisticMixedFamily,
    PoissonMixedFamily,
]

for cls in ALL_FAMILIES:
    fam = cls()
    check(f"{cls.__name__} has calibrate()", hasattr(fam, "calibrate"))

# ================================================================== #
# 3. GLMM families reject non-score strategies
# ================================================================== #
print("\n--- 3. GLMM families reject non-score strategies ---")

for cls in [LogisticMixedFamily, PoissonMixedFamily]:
    fam = cls()
    for bad_method in ("ter_braak", "freedman_lane"):
        rejected = False
        try:
            from randomization_tests.engine import PermutationEngine

            X_dummy = pd.DataFrame({"x1": np.random.default_rng(0).normal(size=10)})
            y_dummy = (
                np.array([0, 1] * 5, dtype=float)
                if "logistic" in fam.name
                else np.array([1, 2, 3, 4, 5] * 2, dtype=float)
            )
            PermutationEngine(
                X=X_dummy,
                y_values=y_dummy,
                family=fam,
                method=bad_method,
            )
        except (ValueError, NotImplementedError):
            rejected = True
        check(f"{cls.__name__} rejects method='{bad_method}'", rejected)

# ================================================================== #
# 4. Statistical audit: null_score for intercept-free models
# ================================================================== #
print("\n--- 4. null_score intercept-free predictions ---")

n = 20
y_bin = np.array([0, 1] * (n // 2), dtype=float)
y_cnt = np.arange(1, n + 1, dtype=float)

# Logistic: sigma(0) = 0.5
logistic_null = LogisticFamily().null_score(y_bin, fit_intercept=False)
# fit_metric returns deviance = 2 * binary_cross_entropy(y, 0.5)
# = 2 * n * log(2) since log_loss(y, 0.5) = log(2) per obs.
expected_dev = 2.0 * n * np.log(2.0)
check(
    "LogisticFamily.null_score(fit_intercept=False) uses 0.5 prediction",
    abs(logistic_null - expected_dev) < 1e-10,
)

# Poisson: exp(0) = 1
poisson_null = PoissonFamily().null_score(y_cnt, fit_intercept=False)
# fit_metric returns deviance = 2 * sum(y*log(y/mu) - (y - mu))
# With mu=1: 2 * sum(y*log(y) - (y - 1)) for y > 0
pos = y_cnt > 0
contrib = np.zeros_like(y_cnt)
contrib[pos] = y_cnt[pos] * np.log(y_cnt[pos] / 1.0)
deviance_i = contrib - (y_cnt - 1.0)
expected_pois = float(2.0 * np.sum(deviance_i))
check(
    "PoissonFamily.null_score(fit_intercept=False) uses 1.0 prediction",
    abs(poisson_null - expected_pois) < 1e-8,
)

# NegativeBinomial: exp(0) = 1
nb_fam = NegativeBinomialFamily(alpha=1.0)
nb_null = nb_fam.null_score(y_cnt, fit_intercept=False)
check(
    "NegativeBinomialFamily.null_score(fit_intercept=False) is finite",
    np.isfinite(nb_null),
)

# ================================================================== #
# 5. Logistic clipping harmonisation
# ================================================================== #
print("\n--- 5. Logistic clipping harmonisation ---")

# fit_metric with extreme predictions should not raise
logistic = LogisticFamily()
metric = logistic.fit_metric(
    np.array([0.0, 1.0]),
    np.array([1e-20, 1.0 - 1e-20]),
)
check("LogisticFamily.fit_metric handles extreme predictions", np.isfinite(metric))

# Inspect source for 1e-15 clipping
src = inspect.getsource(fam_mod.LogisticFamily.fit_metric)
check("LogisticFamily.fit_metric clips at 1e-15", "1e-15" in src)
check("LogisticFamily.fit_metric no 0.001 clip", "0.001" not in src)

# ================================================================== #
# 6. Coverage n! overflow notation
# ================================================================== #
print("\n--- 6. Coverage n! overflow notation ---")

result = compute_permutation_coverage(n_samples=200, n_randomizations=5000)
check("coverage_pct key exists", "coverage_pct" in result)
check("n_factorial_str key exists", "n_factorial_str" in result)
check(
    "overflow uses n! notation (not > 10^)",
    "!" in result["n_factorial_str"] and "> 10^" not in result["n_factorial_str"],
)
check(
    "n_factorial_str == '200!'",
    result["n_factorial_str"] == "200!",
)
check("coverage_pct == '< 0.1%'", result["coverage_pct"] == "< 0.1%")

# Non-overflow case: small n
result_small = compute_permutation_coverage(n_samples=8, n_randomizations=5000)
check(
    "non-overflow n_factorial_str is formatted integer",
    result_small["n_factorial_str"] == "40,320",
)
check(
    "non-overflow coverage_pct is a percentage",
    "%" in result_small["coverage_pct"],
)

# ================================================================== #
# 7. Dynamic engine error message
# ================================================================== #
print("\n--- 7. Dynamic engine error message ---")

src_engine = inspect.getsource(fam_mod.resolve_family)
check(
    "resolve_family error is dynamic (no hardcoded list)",
    "Available families:" in src_engine and "sorted(_FAMILIES)" in src_engine,
)

# ================================================================== #
# 8. Confounder display contracts
# ================================================================== #
print("\n--- 8. Confounder display contracts ---")

rng = np.random.default_rng(42)
n = 50
X_df = pd.DataFrame(
    {
        "treatment": rng.normal(size=n),
        "covariate": rng.normal(size=n),
    }
)
y_df = pd.DataFrame({"y": 2.0 * X_df["treatment"] + rng.normal(size=n)})

result = randomization_test_regression(
    X_df,
    y_df,
    family="linear",
    n_randomizations=200,
    confounders=["covariate"],
    random_state=42,
    method="kennedy",
)

# Label check
cov_idx = list(X_df.columns).index("covariate")
check(
    "confounder p-value label is '(confounder)'",
    result.permuted_p_values[cov_idx] == "(confounder)",
)

# pval_ci NaN masking
pv_ci = result.confidence_intervals["pvalue_ci"]
check(
    "confounder pval_ci is NaN",
    np.isnan(pv_ci[cov_idx][0]) and np.isnan(pv_ci[cov_idx][1]),
)

# Non-confounder has valid CI
check(
    "treatment pval_ci is not NaN",
    not np.isnan(pv_ci[0][0]) and not np.isnan(pv_ci[0][1]),
)

# Capture display output for confounder rendering
old_stdout = sys.stdout
sys.stdout = captured = io.StringIO()
print_results_table(result)
sys.stdout = old_stdout
out = captured.getvalue()

check("'(confounder)' appears in results table", "(confounder)" in out)
check("'N/A (confounder)' does NOT appear", "N/A (confounder)" not in out)

# ================================================================== #
# 9. display_diagnostics() returns 3-tuples (all 10 families)
# ================================================================== #
print("\n--- 9. display_diagnostics() 3-tuple contract ---")

# GLM families — need a dummy diagnostics dict
GLM_FAMILIES = [
    LinearFamily,
    LogisticFamily,
    PoissonFamily,
    NegativeBinomialFamily,
    OrdinalFamily,
    MultinomialFamily,
]

# Build a minimal diagnostics dict for each family type
dummy_bp = {
    "lm_stat": 1.0,
    "lm_p_value": 0.5,
    "f_stat": 1.0,
    "f_p_value": 0.5,
}
dummy_diag = {
    "breusch_pagan": dummy_bp,  # linear only
    "deviance_residuals": {
        "mean": 0.0,
        "variance": 1.0,
        "n_large": 0,
        "runs_z": 0.5,
        "runs_p": 0.6,
    },
    "pearson_chi2": 10.0,
    "deviance_stat": 10.0,
    "dispersion": 1.0,
    "alpha": 1.0,
    "pseudo_r2": 0.3,
    "log_likelihood": -50.0,
    "n_categories": 3,
    "prop_odds_chi2": 5.0,
    "prop_odds_p": 0.1,
    "prop_odds_df": 2,
    "llr_p_value": 0.01,
    "category_counts": {"a": 10, "b": 20, "c": 30},
    "variance_components": {},
    "icc": 0.1,
    "sigma2": 1.0,
}

for cls in GLM_FAMILIES:
    fam = cls()
    lines, notes = fam.display_diagnostics(dummy_diag)
    all_3_tuples = all(isinstance(t, tuple) and len(t) == 3 for t in lines)
    check(f"{cls.__name__}.display_diagnostics returns 3-tuples", all_3_tuples)

# Mixed families — need variance_components in diagnostics
mixed_diag = {
    "variance_components": {"subject": {"variance": 0.5, "std_dev": 0.707}},
    "icc": 0.15,
    "sigma2": 1.0,
    "dispersion": 1.2,
}

for cls in [LinearMixedFamily, LogisticMixedFamily, PoissonMixedFamily]:
    fam = cls()
    lines, notes = fam.display_diagnostics(mixed_diag)
    all_3_tuples = all(isinstance(t, tuple) and len(t) == 3 for t in lines)
    check(f"{cls.__name__}.display_diagnostics returns 3-tuples", all_3_tuples)

# ================================================================== #
# 10. GLMM deviance note
# ================================================================== #
print("\n--- 10. GLMM deviance note ---")

for cls in [LogisticMixedFamily, PoissonMixedFamily]:
    fam = cls()
    # deviance_note is in diagnostics(), not compute_extended_diagnostics()
    src = inspect.getsource(cls.diagnostics)
    check(
        f"{cls.__name__} has deviance_note",
        "deviance_note" in src and "marginal" in src,
    )

# ================================================================== #
# 11. Coverage sufficiency verdict in diagnostics table
# ================================================================== #
print("\n--- 11. Coverage sufficiency verdict ---")

# Capture diagnostics table output
old_stdout = sys.stdout
sys.stdout = captured = io.StringIO()
print_diagnostics_table(result)
sys.stdout = old_stdout
diag_out = captured.getvalue()

check(
    "diagnostics table has Coverage: line",
    "Coverage:" in diag_out,
)
has_verdict = "sufficient" in diag_out or "borderline" in diag_out
check("coverage line has sufficiency verdict", has_verdict)

# ================================================================== #
# 12. Distance correlation denominator guard
# ================================================================== #
print("\n--- 12. Distance correlation denominator guard ---")

src_conf = inspect.getsource(conf_mod)
check(
    "dcor denominator uses max(..., 1e-300)",
    "max(dvar_x * dvar_y - dcov2**2, 1e-300)" in src_conf
    or "max(dvar_x * dvar_y - dcov2 ** 2, 1e-300)" in src_conf,
)

# ================================================================== #
# 13. Score strategy regularisation
# ================================================================== #
print("\n--- 13. Score strategy regularisation ---")

src_score = inspect.getsource(score_mod)
check(
    "score strategy has 1e-10 * I regularisation",
    "1e-10" in src_score and "np.eye" in src_score,
)

# ================================================================== #
# Summary
# ================================================================== #
print()
print("=" * 64)
if failed == 0:
    print(f"ALL {passed} CHECKS PASSED")
else:
    print(f"{failed} FAILED, {passed} passed")
print("=" * 64)

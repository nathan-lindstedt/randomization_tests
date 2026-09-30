"""Tests for Double/Debiased Machine Learning (DML; Chernozhukov et al. 2018).

Verifies the mathematical and algorithmic invariants of DML across linear,
logistic, joint, and kernel independence tests:
1. Linear equivalence: when confounding is linear, DML matches classical Freedman–Lane.
2. Nonlinear confounder recovery: under nonlinear confounding, DML recovers true signal
   and controls Type I error, whereas linear Freedman–Lane collapses.
3. API validation: reduced_model is restricted to freedman_lane and requires confounders.
4. Axiom D26 diagnostic: warns toward exact stratification when confounders are discrete.
5. Bit-level determinism under fixed seeds.
6. GroupKFold cluster preservation under grouped/clustered designs.
7. Probability score DML for binary outcomes.
8. Joint collective-improvement DML test.
9. Kernel regression CI test: OLS hat projection leaks nonlinear confounding;
   cross-fitted DML residualization restores calibration.
10. Realistic synthetic stress tests: multi-predictor collinearity, heteroscedasticity,
    and heavy-tailed Student-t disturbances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from randomization_tests import (
    IndividualTestResult,
    JointTestResult,
    randomization_test_regression,
)
from randomization_tests._kernel_tests import kernel_regression_test

# ------------------------------------------------------------------ #
# Linear Equivalence & Validation
# ------------------------------------------------------------------ #


def test_dml_linear_matches_freedman_lane() -> None:
    """Under additive linear confounding, DML with LinearRegression matches FL."""
    rng = np.random.default_rng(42)
    n = 150
    Z = rng.normal(0, 1, size=(n, 2))
    X = 0.8 * Z[:, 0] - 0.5 * Z[:, 1] + rng.normal(0, 0.5, size=n)
    y = 1.5 * X + 1.2 * Z[:, 0] - 0.7 * Z[:, 1] + rng.normal(0, 0.5, size=n)

    df = pd.DataFrame({"X": X, "Z0": Z[:, 0], "Z1": Z[:, 1]})
    y_df = pd.DataFrame({"y": y})

    res_fl = randomization_test_regression(
        df,
        y_df,
        confounders=["Z0", "Z1"],
        method="freedman_lane",
        n_randomizations=300,
        random_state=123,
    )

    res_dml = randomization_test_regression(
        df,
        y_df,
        confounders=["Z0", "Z1"],
        method="freedman_lane",
        reduced_model=LinearRegression(),
        n_randomizations=300,
        random_state=123,
    )
    assert isinstance(res_fl, IndividualTestResult)
    assert isinstance(res_dml, IndividualTestResult)

    # Point estimates must be within cross-fitting estimation tolerance of OLS
    b_fl = float(res_fl.model_coefs[0])
    b_dml = float(res_dml.model_coefs[0])
    assert abs(b_fl - 1.5) < 0.2
    assert abs(b_dml - 1.5) < 0.2
    assert abs(b_fl - b_dml) < 0.15

    # Both reject H0 strongly
    p_fl = float(res_fl.raw_empirical_p[0])
    p_dml = float(res_dml.raw_empirical_p[0])
    assert p_fl < 0.05
    assert p_dml < 0.05
    assert res_dml.guarantee_tier == "Asymptotically Exact (DML)"


def test_dml_requires_freedman_lane() -> None:
    """reduced_model passed to non-Freedman-Lane methods raises ValueError."""
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0], "Z": [0.1, 0.2, 0.3, 0.4]})
    y_df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})

    with pytest.raises(
        ValueError, match="reduced_model is supported only with method='freedman_lane'"
    ):
        randomization_test_regression(
            df,
            y_df,
            confounders=["Z"],
            method="ter_braak",
            reduced_model=LinearRegression(),
        )


def test_dml_requires_confounders() -> None:
    """reduced_model without confounders raises ValueError."""
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0], "Z": [0.1, 0.2, 0.3, 0.4]})
    y_df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})

    with pytest.raises(
        ValueError, match="reduced_model requires confounders to be specified"
    ):
        randomization_test_regression(
            df,
            y_df,
            confounders=None,
            method="freedman_lane",
            reduced_model=LinearRegression(),
        )


def test_dml_discrete_confounder_advisory() -> None:
    """Axiom D26 diagnostic: warns toward exact stratification when confounders are discrete."""
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "Z": [0, 1, 0, 1, 0, 1]})
    y_df = pd.DataFrame({"y": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})

    res = randomization_test_regression(
        df,
        y_df,
        confounders=["Z"],
        method="freedman_lane",
        reduced_model=LinearRegression(),
        n_randomizations=50,
        random_state=42,
    )
    # Check advisory captured on context
    assert res.context is not None
    assert any("Axiom D26" in note for note in res.context.warnings_captured)


def test_dml_cross_fitting_determinism() -> None:
    """Fixed random_state guarantees bit-level identical DML results."""
    rng = np.random.default_rng(99)
    n = 100
    Z = rng.uniform(-2, 2, size=(n, 1))
    X = np.sin(2 * Z[:, 0]) + rng.normal(0, 0.5, size=n)
    y = np.cos(2 * Z[:, 0]) + rng.normal(0, 0.5, size=n)

    df = pd.DataFrame({"X": X, "Z": Z[:, 0]})
    y_df = pd.DataFrame({"y": y})

    gb1 = GradientBoostingRegressor(n_estimators=20, max_depth=3)
    res1 = randomization_test_regression(
        df,
        y_df,
        confounders=["Z"],
        method="freedman_lane",
        reduced_model=gb1,
        n_randomizations=100,
        random_state=777,
    )

    gb2 = GradientBoostingRegressor(n_estimators=20, max_depth=3)
    res2 = randomization_test_regression(
        df,
        y_df,
        confounders=["Z"],
        method="freedman_lane",
        reduced_model=gb2,
        n_randomizations=100,
        random_state=777,
    )
    assert isinstance(res1, IndividualTestResult)
    assert isinstance(res2, IndividualTestResult)

    np.testing.assert_array_equal(res1.model_coefs, res2.model_coefs)
    np.testing.assert_array_equal(res1.raw_empirical_p, res2.raw_empirical_p)
    np.testing.assert_array_equal(res1.permuted_coefs, res2.permuted_coefs)


# ------------------------------------------------------------------ #
# Nonlinear Confounding & Signal Recovery
# ------------------------------------------------------------------ #


def test_dml_nonlinear_confounders_h0() -> None:
    """Under nonlinear confounding and H0, DML controls Type I error while linear FL can leak."""
    rng = np.random.default_rng(2026)
    n = 200
    Z = rng.uniform(-2, 2, size=(n, 1))
    z = Z[:, 0]

    # Strong nonlinear confounding
    m_z = np.sin(2 * z) + 0.5 * (z**2)
    g_z = np.cos(2 * z) + 0.5 * (z**2)

    X = m_z + rng.normal(0, 0.4, size=n)
    # True beta = 0 (H0 true)
    y = 0.0 * X + g_z + rng.normal(0, 0.4, size=n)

    df = pd.DataFrame({"X": X, "Z": z})
    y_df = pd.DataFrame({"y": y})

    gb = GradientBoostingRegressor(n_estimators=40, max_depth=3, learning_rate=0.1)
    res_dml = randomization_test_regression(
        df,
        y_df,
        confounders=["Z"],
        method="freedman_lane",
        reduced_model=gb,
        n_randomizations=400,
        random_state=42,
    )
    assert isinstance(res_dml, IndividualTestResult)

    # Point estimate is close to zero
    assert abs(float(res_dml.model_coefs[0])) < 0.15
    # Permutation null is zero-centered
    perm_null = np.array(res_dml.permuted_coefs)[:, 0]
    assert abs(np.mean(perm_null)) < 0.05
    # Does not falsely reject at alpha = 0.01
    assert float(res_dml.raw_empirical_p[0]) >= 0.01


def test_dml_nonlinear_signal_recovery() -> None:
    """Under true signal beta = -2.0 with nonlinear confounding, DML recovers effect and rejects H0."""
    rng = np.random.default_rng(3030)
    n = 250
    Z = rng.uniform(-2, 2, size=(n, 1))
    z = Z[:, 0]

    m_z = np.sin(2 * z) + (z**2)
    X = m_z + rng.normal(0, 0.5, size=n)
    # E[Y | Z] = -2*m_z + 2*m_z = 0
    y = -2.0 * X + 2.0 * m_z + rng.normal(0, 0.5, size=n)

    df = pd.DataFrame({"X": X, "Z": z})
    y_df = pd.DataFrame({"y": y})

    gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
    res_dml = randomization_test_regression(
        df,
        y_df,
        confounders=["Z"],
        method="freedman_lane",
        reduced_model=gb,
        n_randomizations=400,
        random_state=42,
    )
    assert isinstance(res_dml, IndividualTestResult)

    # Recovers true beta = -2.0 accurately (linear FL collapses to ~0)
    b_dml = float(res_dml.model_coefs[0])
    assert abs(b_dml - (-2.0)) < 0.3
    # Empirical p-value rejects H0 decisively
    assert float(res_dml.raw_empirical_p[0]) < 0.01


# ------------------------------------------------------------------ #
# GroupKFold & Clustering
# ------------------------------------------------------------------ #


def test_dml_group_kfold() -> None:
    """Passing groups ensures cluster boundaries are respected during cross-fitting."""
    n_clusters = 10
    cluster_size = 15
    n = n_clusters * cluster_size
    rng = np.random.default_rng(4040)

    groups = np.repeat(np.arange(n_clusters), cluster_size)
    gamma_g = rng.normal(0, 1.0, size=n_clusters)
    gamma = np.repeat(gamma_g, cluster_size)

    Z = gamma + rng.normal(0, 0.3, size=n)
    X = np.sin(Z) + rng.normal(0, 0.5, size=n)
    y = 1.0 * X + np.cos(Z) + rng.normal(0, 0.5, size=n)

    df = pd.DataFrame({"X": X, "Z": Z})
    y_df = pd.DataFrame({"y": y})

    gb = GradientBoostingRegressor(n_estimators=25, max_depth=3)
    res = randomization_test_regression(
        df,
        y_df,
        confounders=["Z"],
        groups=groups,
        method="freedman_lane",
        reduced_model=gb,
        n_randomizations=100,
        random_state=42,
    )
    assert isinstance(res, IndividualTestResult)

    assert float(res.raw_empirical_p[0]) < 0.05
    assert abs(float(res.model_coefs[0]) - 1.0) < 0.35


# ------------------------------------------------------------------ #
# Classifier Predict Proba & Joint Strategy
# ------------------------------------------------------------------ #


def test_dml_classifier_predict_proba() -> None:
    """Logistic regression DML supports classifiers with predict_proba."""
    rng = np.random.default_rng(42)
    n = 250
    Z = rng.uniform(-1.5, 1.5, size=(n, 1))
    z = Z[:, 0]

    X = np.sin(z) + rng.normal(0, 1.0, size=n)
    # Logit link under true effect beta = 2.0
    logit_p = 2.0 * X + 0.5 * (z**2)
    prob = 1.0 / (1.0 + np.exp(-logit_p))
    y = rng.binomial(1, prob).astype(float)
    if len(np.unique(y)) < 2:
        y[0] = 0.0
        y[1] = 1.0

    df = pd.DataFrame({"X": X, "Z": z})
    y_df = pd.DataFrame({"y": y})

    gb_clf = GradientBoostingClassifier(n_estimators=20, max_depth=2)
    gb_reg = GradientBoostingRegressor(n_estimators=20, max_depth=2)
    res = randomization_test_regression(
        df,
        y_df,
        confounders=["Z"],
        family="logistic",
        method="freedman_lane",
        reduced_model=(gb_clf, gb_reg),
        n_randomizations=200,
        random_state=42,
    )
    assert isinstance(res, IndividualTestResult)

    assert res.family.name == "logistic"
    assert res.guarantee_tier == "Asymptotically Exact (DML)"
    # Confirms score null evaluated successfully and detected signal
    assert float(res.raw_empirical_p[0]) < 0.05


def test_dml_classifier_without_regressor_raises() -> None:
    """Passing a single classifier when continuous predictors exist raises ValueError."""
    df = pd.DataFrame(
        {"X": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "Z": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    )
    y_df = pd.DataFrame({"y": [0, 1, 0, 1, 0, 1]})

    gb_clf = GradientBoostingClassifier(n_estimators=10)
    with pytest.raises(ValueError, match="specify reduced_model as a tuple"):
        randomization_test_regression(
            df,
            y_df,
            confounders=["Z"],
            family="logistic",
            method="freedman_lane",
            reduced_model=gb_clf,
            n_randomizations=50,
            random_state=42,
        )


def test_dml_estimator_agnosticism() -> None:
    """DML works with any scikit-learn regressor (RandomForest, Ridge, etc.)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(12345)
    n = 150
    Z = rng.uniform(-1.5, 1.5, size=(n, 2))
    X = np.sin(Z[:, 0]) + 0.3 * (Z[:, 1] ** 2) + rng.normal(0, 0.4, size=n)
    y = 1.5 * X + np.cos(Z[:, 0]) + 0.3 * (Z[:, 1] ** 2) + rng.normal(0, 0.4, size=n)

    df = pd.DataFrame({"X": X, "Z0": Z[:, 0], "Z1": Z[:, 1]})
    y_df = pd.DataFrame({"y": y})

    # Test with RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42)
    res_rf = randomization_test_regression(
        df,
        y_df,
        confounders=["Z0", "Z1"],
        method="freedman_lane",
        reduced_model=rf,
        n_randomizations=150,
        random_state=42,
    )
    assert isinstance(res_rf, IndividualTestResult)
    assert float(res_rf.raw_empirical_p[0]) < 0.05
    assert abs(float(res_rf.model_coefs[0]) - 1.5) < 0.35

    # Test with Ridge
    ridge = Ridge(alpha=1.0)
    res_ridge = randomization_test_regression(
        df,
        y_df,
        confounders=["Z0", "Z1"],
        method="freedman_lane",
        reduced_model=ridge,
        n_randomizations=150,
        random_state=42,
    )
    assert isinstance(res_ridge, IndividualTestResult)
    assert float(res_ridge.raw_empirical_p[0]) < 0.05


def test_dml_joint_strategy() -> None:
    """freedman_lane_joint with reduced_model evaluates collective improvement."""
    rng = np.random.default_rng(6060)
    n = 150
    Z = rng.uniform(-2, 2, size=(n, 1))
    z = Z[:, 0]

    X1 = np.sin(z) + rng.normal(0, 0.5, size=n)
    X2 = np.cos(z) + rng.normal(0, 0.5, size=n)
    y = 1.0 * X1 - 1.0 * X2 + 0.5 * (z**2) + rng.normal(0, 0.5, size=n)

    df = pd.DataFrame({"X1": X1, "X2": X2, "Z": z})
    y_df = pd.DataFrame({"y": y})

    gb = GradientBoostingRegressor(n_estimators=30, max_depth=3)
    res_joint = randomization_test_regression(
        df,
        y_df,
        confounders=["Z"],
        method="freedman_lane_joint",
        reduced_model=gb,
        n_randomizations=200,
        random_state=42,
    )
    assert isinstance(res_joint, JointTestResult)

    assert res_joint.observed_improvement > 0.0
    assert len(res_joint.permuted_improvements) == 200
    assert res_joint.p_value < 0.05
    assert res_joint.guarantee_tier == "Asymptotically Exact (DML)"


# ------------------------------------------------------------------ #
# Kernel CI Tests with Cross-Fitted DML
# ------------------------------------------------------------------ #


def test_kernel_regression_nonlinear_confounder_leak() -> None:
    """kernel_regression_test: OLS hat leaks nonlinear confounding, DML controls it."""
    rng = np.random.default_rng(42)
    n = 200
    Z = rng.uniform(-2, 2, size=(n, 1))
    z = Z[:, 0]

    # Conditional independence: X ⊥ Y | Z under pure quadratic confounding
    # Because E[Z^3] = 0, OLS projection removes 0% of the Z^2 confounding.
    X = z**2 + rng.normal(0, 0.4, size=n)
    y = z**2 + rng.normal(0, 0.4, size=n)

    # 1. Shipped OLS path: linear projection leaves quadratic terms in residuals
    res_ols = kernel_regression_test(
        X,
        y,
        confounders=Z,
        n_randomizations=200,
        random_state=42,
    )
    # OLS path leaks nonlinear confounding (false positive under H0: p < 0.05)
    assert res_ols.p_value < 0.05

    # 2. Cross-fitted DML path: purges nonlinear terms out-of-fold
    gb = GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=42)
    res_dml = kernel_regression_test(
        X,
        y,
        confounders=Z,
        reduced_model=gb,
        n_randomizations=200,
        random_state=42,
    )

    # Cross-fitted DML p-value is well above significance threshold (H0 preserved)
    assert res_dml.p_value >= 0.05


def test_kernel_regression_reduced_model_deterministic() -> None:
    """kernel_regression_test with reduced_model is bit-identical under fixed seed."""
    rng = np.random.default_rng(8080)
    n = 100
    Z = rng.uniform(-1, 1, size=(n, 1))
    X = Z[:, 0] ** 2 + rng.normal(0, 0.3, size=n)
    y = Z[:, 0] ** 2 + rng.normal(0, 0.3, size=n)

    gb1 = GradientBoostingRegressor(n_estimators=20, max_depth=2)
    res1 = kernel_regression_test(
        X,
        y,
        confounders=Z,
        reduced_model=gb1,
        n_randomizations=100,
        random_state=999,
    )

    gb2 = GradientBoostingRegressor(n_estimators=20, max_depth=2)
    res2 = kernel_regression_test(
        X,
        y,
        confounders=Z,
        reduced_model=gb2,
        n_randomizations=100,
        random_state=999,
    )

    assert res1.statistic == pytest.approx(res2.statistic, abs=1e-12)
    assert res1.p_value == pytest.approx(res2.p_value, abs=1e-12)
    np.testing.assert_array_equal(res1.null_distribution, res2.null_distribution)


# ------------------------------------------------------------------ #
# Realistic Synthetic Stress Tests
# ------------------------------------------------------------------ #


def test_dml_real_world_multi_predictor_multicollinearity() -> None:
    """Stress test: multi-predictor design with correlated nonlinear confounders."""
    rng = np.random.default_rng(9090)
    n = 200

    # Correlated continuous confounders (e.g. age, income)
    cov_z = np.array([[1.0, 0.6], [0.6, 1.0]])
    Z = rng.multivariate_normal([0.0, 0.0], cov_z, size=n)

    # Nonlinear interaction confounding
    g_z = np.sin(Z[:, 0]) + 0.3 * (Z[:, 1] ** 2) + 0.5 * Z[:, 0] * Z[:, 1]
    m_z1 = 0.5 * (Z[:, 0] ** 2) - 0.4 * Z[:, 1]
    m_z2 = np.cos(Z[:, 1]) + 0.3 * Z[:, 0]

    X1 = m_z1 + rng.normal(0, 0.5, size=n)
    X2 = m_z2 + 0.3 * X1 + rng.normal(0, 0.5, size=n)

    # Outcome: X1 has real effect (beta1 = 1.2), X2 has zero effect (beta2 = 0)
    y = 1.2 * X1 + 0.0 * X2 + g_z + rng.normal(0, 0.5, size=n)

    df = pd.DataFrame({"X1": X1, "X2": X2, "Z1": Z[:, 0], "Z2": Z[:, 1]})
    y_df = pd.DataFrame({"y": y})

    gb = GradientBoostingRegressor(n_estimators=45, max_depth=3, learning_rate=0.1)
    res = randomization_test_regression(
        df,
        y_df,
        confounders=["Z1", "Z2"],
        method="freedman_lane",
        reduced_model=gb,
        n_randomizations=300,
        random_state=42,
    )
    assert isinstance(res, IndividualTestResult)

    # X1 is detected as highly significant
    assert float(res.raw_empirical_p[0]) < 0.01
    assert abs(float(res.model_coefs[0]) - 1.2) < 0.35

    # X2 is correctly identified as not significant (no false positive from collinearity)
    assert float(res.raw_empirical_p[1]) >= 0.05
    assert abs(float(res.model_coefs[1])) < 0.25


def test_dml_heteroscedastic_errors() -> None:
    """Stress test: confounder-dependent heteroscedastic disturbances."""
    rng = np.random.default_rng(9191)
    n = 200
    Z = rng.uniform(-2, 2, size=(n, 1))
    z = Z[:, 0]

    # Heteroscedastic noise scale: sigma(Z) varies with |Z|
    sigma_z = 0.3 + 0.4 * np.abs(z)
    eps = rng.normal(0, 1.0, size=n) * sigma_z

    X = np.sin(z) + rng.normal(0, 0.5, size=n)
    # H0 true: beta = 0
    y = 0.0 * X + (z**2) + eps

    df = pd.DataFrame({"X": X, "Z": z})
    y_df = pd.DataFrame({"y": y})

    gb = GradientBoostingRegressor(n_estimators=35, max_depth=3)
    res = randomization_test_regression(
        df,
        y_df,
        confounders=["Z"],
        method="freedman_lane",
        reduced_model=gb,
        n_randomizations=300,
        random_state=42,
    )
    assert isinstance(res, IndividualTestResult)

    # Type I error controlled despite heteroscedasticity
    assert float(res.raw_empirical_p[0]) >= 0.05
    assert abs(float(res.model_coefs[0])) < 0.2


def test_dml_heavy_tailed_errors() -> None:
    """Stress test: heavy-tailed Student-t disturbances (nu = 3)."""
    rng = np.random.default_rng(9292)
    n = 200
    Z = rng.uniform(-2, 2, size=(n, 1))
    z = Z[:, 0]

    # Student-t with 3 degrees of freedom (infinite 4th moment, heavy tails)
    t_dist = rng.standard_t(df=3, size=n) * 0.5

    X = np.sin(2 * z) + rng.normal(0, 0.4, size=n)
    # Real effect beta = 1.0 with heavy-tailed noise
    y = 1.0 * X + np.cos(2 * z) + t_dist

    df = pd.DataFrame({"X": X, "Z": z})
    y_df = pd.DataFrame({"y": y})

    gb = GradientBoostingRegressor(n_estimators=40, max_depth=3)
    res = randomization_test_regression(
        df,
        y_df,
        confounders=["Z"],
        method="freedman_lane",
        reduced_model=gb,
        n_randomizations=300,
        random_state=42,
    )
    assert isinstance(res, IndividualTestResult)

    # Freedman–Lane permutation test remains robust to fat-tailed errors
    assert float(res.raw_empirical_p[0]) < 0.01
    assert abs(float(res.model_coefs[0]) - 1.0) < 0.45

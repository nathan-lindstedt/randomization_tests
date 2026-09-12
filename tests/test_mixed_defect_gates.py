"""Gates encoding *known* mixed-model inference defects, and the invariants beside them.

Every ``xfail(strict=True)`` here encodes a measured defect. When the corresponding
fix lands the test XPASSes, strict mode turns that into a suite failure, and the
marker must be removed — so a landed fix cannot silently leave a stale marker.
This ratchet only works while pytest *collects* the file, which is why these live
here rather than as a standalone script under ``verify_gates/``.

Tests without a marker are invariants that pass today and must keep passing.

Fast deterministic gates come first, one dataset each. Monte Carlo corroboration
is ``slow``-marked at the bottom and deselected by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from randomization_tests import randomization_test_regression
from randomization_tests.families import fit_reduced
from randomization_tests.families_mixed import (
    LinearMixedFamily,
    LogisticMixedFamily,
)

G, M = 20, 10
N = G * M
CLUSTER = np.repeat(np.arange(G), M)
CELLS = [np.arange(g * M, (g + 1) * M) for g in range(G)]
B = 399


def _identity(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.intp).reshape(1, -1)


def _lmm_data(seed: int, *, slopes: bool = False, tau2: float = 9.0):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=N), rng.normal(size=N)
    y = np.empty(N)
    for ci in CELLS:
        b1 = rng.normal(scale=2.0) if slopes else 0.0
        y[ci] = (
            0.7 * z[ci]
            + rng.normal(scale=np.sqrt(tau2))
            + b1 * x[ci]
            + rng.normal(size=len(ci))
        )
    return np.column_stack([x, z]), y


def _glmm_data(seed: int, tau2: float = 4.0):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=N), rng.normal(size=N)
    b = rng.normal(scale=np.sqrt(tau2), size=G)
    y = rng.binomial(1, 1 / (1 + np.exp(-(1.5 * z + b[CLUSTER])))).astype(float)
    return np.column_stack([x, z]), y


def _poisson_glmm_data(seed: int, tau2: float = 1.0):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=N), rng.normal(size=N)
    b = rng.normal(scale=np.sqrt(tau2), size=G)
    eta = 0.5 + 0.3 * z + b[CLUSTER]
    mu = np.exp(np.clip(eta, -10.0, 10.0))
    y = rng.poisson(mu).astype(float)
    return np.column_stack([x, z]), y


def _offset(family, X, y) -> float:
    """|beta_hat_j - U_0| relative to beta_hat scale, for feature 0.

    Zero exactly when the score projection is exact, because row j of S^-1 X' V^-1
    satisfies A_j X_k = delta_jk, hence A_j X_reduced = 0.
    """
    beta = float(family.coefs(family.fit(X, y, True))[0])
    _, preds_red = fit_reduced(family, np.delete(X, 0, axis=1), y, True)
    u0 = float(
        family.score_project(
            X, 0, y - preds_red, _identity(len(y)), fit_intercept=True, y=y
        )[0]
    )
    return abs(beta - u0) / max(abs(beta), 1e-12)


def _spread_ratio(result) -> float:
    """sd(null draws) / SE(beta_hat), SE recovered by inverting the Wald p-value.

    Must be ~1: the null draws should mimic the sampling distribution of beta_hat.
    """
    beta = float(np.ravel(result.model_coefs)[0])
    p_classic = float(np.ravel(result.raw_classic_p)[0])
    se = abs(beta) / norm.ppf(1 - p_classic / 2)
    sd = float(np.std(np.asarray(result.permuted_coefs)[:, 0]))
    return sd / se


# ---------------------------------------------------------------- invariants


def test_lmm_score_offset_is_zero():
    """LMM score projection is exact, so the null is zero-centred by construction."""
    X, y = _lmm_data(101)
    fam = LinearMixedFamily().calibrate(X, y, groups=CLUSTER)
    assert _offset(fam, X, y) < 1e-8


def test_lmm_spread_ratio_random_intercept():
    """Healthy configuration: null spread must match SE(beta_hat)."""
    X, y = _lmm_data(102)
    res = randomization_test_regression(
        pd.DataFrame({"x": X[:, 0], "z": X[:, 1]}),
        pd.DataFrame({"y": y}),
        family="linear_mixed",
        groups=CLUSTER,
        method="score",
        n_randomizations=B,
        random_state=102,
    )
    assert 0.8 < _spread_ratio(res) < 1.25


def test_level2_null_is_repaired_by_whitening():
    """M3b, side-effect finding (2026-09-07): whitening repairs the level-2
    near-invariance too, though nothing in Step 11c targeted it.

    A cluster-constant regressor's near-invariance under raw within-cluster
    permutation was M3b: ``sd(draws)/SE`` collapsed and power equalled Type I at
    0.080. Once `score_project` whitens (fixing M3), the SAME mechanism was
    measured to repair this too -- robustly, not as one lucky seed: ratio
    0.819-0.902 across 8 seeds at the gate's own B=399, and
    Type I 0.025 / power 0.475 (power now ~19x its own Type I, against the
    original 0.080/0.080).

    This does not make ``permutation_strategy="between"`` redundant -- it is
    the correct, assumption-light tool for level-2 effects in general, and
    this measurement is method="score" under "within" specifically. Recorded
    as a positive, verified side effect, not assumed from the mechanism alone.
    """
    rng = np.random.default_rng(103)
    x2 = np.repeat(rng.normal(size=G), M)
    y = np.repeat(rng.normal(scale=3.0, size=G), M) + rng.normal(size=N)
    res = randomization_test_regression(
        pd.DataFrame({"x": x2, "z": rng.normal(size=N)}),
        pd.DataFrame({"y": y}),
        family="linear_mixed",
        groups=CLUSTER,
        method="score",
        n_randomizations=B,
        random_state=103,
    )
    assert 0.8 < _spread_ratio(res) < 1.25


# ---------------------------------------------------------------- gates


@pytest.mark.xfail(
    strict=True, reason="M1: observed reduced fit is GLS, permuted is OLS"
)
def test_lmm_reduced_fit_uses_same_estimator():
    """The reduced fit behind the observed statistic and behind the null draws must
    come from the same estimator; today fit_reduced is GLS and the
    batch_fit_and_score fallback is pinv/OLS."""
    X, y = _lmm_data(104)
    fam = LinearMixedFamily().calibrate(X, y, groups=CLUSTER)
    X_red = np.delete(X, 0, axis=1)

    _, preds_gls = fit_reduced(fam, X_red, y, True)
    coefs_ols, _ = fam.batch_fit_and_score(X_red, y.reshape(1, -1), True)
    preds_ols = (
        np.column_stack([np.ones(N), X_red])
        @ np.r_[np.mean(y - X_red @ np.ravel(coefs_ols)), np.ravel(coefs_ols)]
    )
    np.testing.assert_allclose(preds_gls, preds_ols, rtol=1e-6)


def test_glmm_score_offset_is_zero():
    """M4/M6 CLOSED 2026-09-12: The score offset |beta_hat - U_0| vanishes to
    machine precision (< 1e-14) under the tangent-space linear model
    (closed-form GLS reduced fit on frozen z_tilde + full V_z^-1 whitening).
    """
    X, y = _glmm_data(105)
    fam = LogisticMixedFamily().calibrate(X, y, groups=CLUSTER)
    assert _offset(fam, X, y) < 1e-6


def test_lmm_spread_ratio_random_slopes():
    """M3 CLOSED 2026-09-07: score/Freedman-Lane/ter Braak now whiten together
    via ``residual_permutation_refit`` before permuting, repairing the
    within-cluster exchangeability random slopes broke. Was xfail (0.34-0.42);
    now a plain regression guard.
    """
    X, y = _lmm_data(106, slopes=True)
    res = randomization_test_regression(
        pd.DataFrame({"x": X[:, 0], "z": X[:, 1]}),
        pd.DataFrame({"y": y}),
        family="linear_mixed",
        groups=CLUSTER,
        random_slopes=[0],
        method="score",
        n_randomizations=B,
        random_state=106,
    )
    assert 0.8 < _spread_ratio(res) < 1.25


@pytest.mark.xfail(
    strict=True, reason="M8: panel_id does not reach calibration without ar_order"
)
def test_panel_id_works_without_ar_order():
    """panel_id is documented as setting groups=panel_id, but linear_mixed rejects it
    unless ar_order is also supplied."""
    X, y = _lmm_data(107)
    randomization_test_regression(
        pd.DataFrame({"x": X[:, 0], "z": X[:, 1]}),
        pd.DataFrame({"y": y}),
        family="linear_mixed",
        panel_id=CLUSTER,
        time_id=np.tile(np.arange(M), G),
        method="score",
        n_randomizations=99,
        random_state=107,
    )


def test_glmm_null_is_zero_centred():
    """M4/M6 CLOSED 2026-09-12: Under the exact score projection on the
    tangent-space working scale, the permuted draws center on ZERO.
    """
    X, y = _glmm_data(108)
    res = randomization_test_regression(
        pd.DataFrame({"x": X[:, 0], "z": X[:, 1]}),
        pd.DataFrame({"y": y}),
        family="logistic_mixed",
        groups=CLUSTER,
        method="score",
        n_randomizations=B,
        random_state=108,
    )
    draws = np.asarray(res.permuted_coefs)[:, 0]
    assert abs(float(np.mean(draws))) / float(np.std(draws)) < 0.5


def test_kennedy_unblocked_for_mixed_families():
    """Kennedy individual and joint work across linear and generalized mixed models."""
    X_lmm, y_lmm = _lmm_data(109)
    res_lmm_ind = randomization_test_regression(
        pd.DataFrame({"x": X_lmm[:, 0], "z": X_lmm[:, 1]}),
        pd.DataFrame({"y": y_lmm}),
        family="linear_mixed",
        groups=CLUSTER,
        method="kennedy",
        confounders=["z"],
        n_randomizations=99,
        random_state=109,
    )
    assert 0.0 <= float(res_lmm_ind.raw_empirical_p[0]) <= 1.0

    res_lmm_jnt = randomization_test_regression(
        pd.DataFrame({"x": X_lmm[:, 0], "z": X_lmm[:, 1]}),
        pd.DataFrame({"y": y_lmm}),
        family="linear_mixed",
        groups=CLUSTER,
        method="kennedy_joint",
        confounders=["z"],
        n_randomizations=99,
        random_state=109,
    )
    assert 0.0 <= float(res_lmm_jnt.p_value) <= 1.0

    X_log, y_log = _glmm_data(109)
    res_log_ind = randomization_test_regression(
        pd.DataFrame({"x": X_log[:, 0], "z": X_log[:, 1]}),
        pd.DataFrame({"y": y_log}),
        family="logistic_mixed",
        groups=CLUSTER,
        method="kennedy",
        confounders=["z"],
        n_randomizations=99,
        random_state=109,
    )
    assert 0.0 <= float(res_log_ind.raw_empirical_p[0]) <= 1.0

    res_log_jnt = randomization_test_regression(
        pd.DataFrame({"x": X_log[:, 0], "z": X_log[:, 1]}),
        pd.DataFrame({"y": y_log}),
        family="logistic_mixed",
        groups=CLUSTER,
        method="kennedy_joint",
        confounders=["z"],
        n_randomizations=99,
        random_state=109,
    )
    assert 0.0 <= float(res_log_jnt.p_value) <= 1.0

    X_poi, y_poi = _poisson_glmm_data(109)
    res_poi_ind = randomization_test_regression(
        pd.DataFrame({"x": X_poi[:, 0], "z": X_poi[:, 1]}),
        pd.DataFrame({"y": y_poi}),
        family="poisson_mixed",
        groups=CLUSTER,
        method="kennedy",
        confounders=["z"],
        n_randomizations=99,
        random_state=109,
    )
    assert 0.0 <= float(res_poi_ind.raw_empirical_p[0]) <= 1.0

    res_poi_jnt = randomization_test_regression(
        pd.DataFrame({"x": X_poi[:, 0], "z": X_poi[:, 1]}),
        pd.DataFrame({"y": y_poi}),
        family="poisson_mixed",
        groups=CLUSTER,
        method="kennedy_joint",
        confounders=["z"],
        n_randomizations=99,
        random_state=109,
    )
    assert 0.0 <= float(res_poi_jnt.p_value) <= 1.0


@pytest.mark.xfail(strict=True, reason="M7a: residuals permuted without AR whitening")
def test_ar_lmm_spread_ratio():
    """M7a inflates the null because Omega^-1 is applied to residuals the permutation
    has already de-correlated.

    Asserted on the MEDIAN over several datasets, not one: a single seed passing or
    failing would be selecting on the outcome. The effect is systematic and
    one-directional — every measured ratio is >= 1.0 (1.43, 2.39, 1.02, 1.71, 3.62,
    1.28, 1.15, 1.01) — where the no-AR arm sits at 1.0 with 1.17x spread.

    The sub-1 ratios seen before (0.14, 0.30) were the diverging REML solver, not
    M7a; they disappeared once that was fixed.
    """
    ratios = []
    for seed in (110, 111, 112, 113, 114):
        rng = np.random.default_rng(seed)
        x, z = rng.normal(size=N), rng.normal(size=N)
        y = np.empty(N)
        for ci in CELLS:
            e = np.zeros(M)
            e[0] = rng.normal()
            for t in range(1, M):
                e[t] = 0.6 * e[t - 1] + rng.normal()
            y[ci] = 0.8 * x[ci] + 0.7 * z[ci] + rng.normal(scale=2.0) + e
        res = randomization_test_regression(
            pd.DataFrame({"x": x, "z": z}),
            pd.DataFrame({"y": y}),
            family="linear_mixed",
            panel_id=CLUSTER,
            time_id=np.tile(np.arange(M), G),
            ar_order=1,
            method="score",
            n_randomizations=299,
            random_state=seed,
        )
        ratios.append(_spread_ratio(res))
    assert 0.8 < float(np.median(ratios)) < 1.25


@pytest.mark.parametrize("family", ["linear_mixed", "logistic_mixed", "poisson_mixed"])
def test_kennedy_mixed_models_individual_and_joint(family: str) -> None:
    """Kennedy individual and joint permutation tests work across all mixed models."""
    rng = np.random.default_rng(42)
    x, z = rng.normal(size=N), rng.normal(size=N)
    b = rng.normal(scale=1.5, size=G)
    df = pd.DataFrame({"x": x, "z": z})

    if family == "linear_mixed":
        y = 0.7 * z + b[CLUSTER] + rng.normal(size=N)
    elif family == "logistic_mixed":
        y = rng.binomial(1, 1 / (1 + np.exp(-(0.7 * z + b[CLUSTER])))).astype(float)
    else:
        y = rng.poisson(np.exp(np.clip(0.5 * z + b[CLUSTER] * 0.5, -5, 5))).astype(
            float
        )

    target = pd.DataFrame({"y": y})

    res_ind = randomization_test_regression(
        df,
        target,
        family=family,
        groups=CLUSTER,
        method="kennedy",
        confounders=["z"],
        n_randomizations=49,
        random_state=42,
    )
    assert 0.0 <= float(res_ind.raw_empirical_p[0]) <= 1.0

    res_jnt = randomization_test_regression(
        df,
        target,
        family=family,
        groups=CLUSTER,
        method="kennedy_joint",
        confounders=["z"],
        n_randomizations=49,
        random_state=42,
    )
    assert 0.0 <= float(res_jnt.p_value) <= 1.0


# ------------------------------------------------------------------ #
# Monte Carlo corroboration — slow-marked, deselected by default
# ------------------------------------------------------------------ #
#
# The gates above pin each defect deterministically on one dataset. These confirm
# the *magnitude* matches what was measured, and cover cases whose per-dataset
# behaviour is too unstable for a single draw.
#
# Tolerances are derived, never tuned: at nominal alpha with N replications the
# Monte Carlo SE is sqrt(a(1-a)/N) and the bound is alpha + 2 SE.
#
# Every Type I gate is paired with a power check: a Type I gate alone *passes* a
# test that never rejects.

ALPHA = 0.05
N_REP, B_MC = 100, 299
# 300, not 60: a gate must be sized to FAIL RELIABLY when its defect is present,
# which is a separate requirement from deriving the tolerance correctly. At
# n_rep = 60 the bound is 0.106, so a true rate of 0.130 passes ~30% of the time
# -- and did, XPASSing while the defect was fully present. At 300 the bound is
# 0.075 and 0.130 sits >2 MC SE above it.
N_REP_GLMM, B_GLMM = 300, 199


def _bound(n_rep: int, alpha: float = ALPHA) -> float:
    """alpha + 2 Monte Carlo SE — derived from the binomial SE, not chosen."""
    return alpha + 2.0 * np.sqrt(alpha * (1.0 - alpha) / n_rep)


def _lmm_trial(rng, beta, *, slopes=False, level2=False, tau2=9.0):
    x = np.repeat(rng.normal(size=G), M) if level2 else rng.normal(size=N)
    z = rng.normal(size=N)
    y = np.empty(N)
    for ci in CELLS:
        b1 = rng.normal(scale=2.0) if slopes else 0.0
        y[ci] = (
            beta * x[ci]
            + 0.7 * z[ci]
            + rng.normal(scale=np.sqrt(tau2))
            + b1 * x[ci]
            + rng.normal(size=len(ci))
        )
    return pd.DataFrame({"x": x, "z": z}), pd.DataFrame({"y": y})


def _rate(seed, beta, *, n_rep=N_REP, b=B_MC, joint=False, **kwargs):
    rng = np.random.default_rng(seed)
    slopes = kwargs.pop("slopes", False)
    level2 = kwargs.pop("level2", False)
    tau2 = kwargs.pop("tau2", 9.0)
    hits = 0
    for _ in range(n_rep):
        X, y = _lmm_trial(rng, beta, slopes=slopes, level2=level2, tau2=tau2)
        res = randomization_test_regression(
            X,
            y,
            family="linear_mixed",
            groups=CLUSTER,
            n_randomizations=b,
            random_state=int(rng.integers(1 << 30)),
            **({"random_slopes": [0]} if slopes else {}),
            **kwargs,
        )
        p = float(res.p_value) if joint else float(np.ravel(res.raw_empirical_p)[0])
        hits += p <= ALPHA
    return hits / n_rep


# ICC = 0 is where GLS and OLS coincide, so M1 cannot bite. This must pass BOTH
# before and after the fix: a change that repairs high ICC but breaks ICC = 0 is a
# different bug, not a fix.


@pytest.mark.slow
def test_joint_type_i_at_zero_icc():
    assert _rate(
        201, 0.0, joint=True, method="freedman_lane_joint", confounders=["z"], tau2=1e-8
    ) <= _bound(N_REP)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="M1: measured 0.340/0.370 at ICC 0.9")
def test_joint_type_i_at_high_icc():
    assert _rate(
        202, 0.0, joint=True, method="freedman_lane_joint", confounders=["z"]
    ) <= _bound(N_REP)


@pytest.mark.slow
def test_individual_type_i_random_intercept():
    assert _rate(203, 0.0, method="score") <= _bound(N_REP)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="M3: measured 0.480 under random slopes")
def test_individual_type_i_random_slopes():
    assert _rate(204, 0.0, method="score", slopes=True) <= _bound(N_REP)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="M3: measured 0.433, two-stage inherits within")
def test_two_stage_type_i_random_slopes():
    assert _rate(
        205, 0.0, method="score", slopes=True, permutation_strategy="two-stage"
    ) <= _bound(N_REP)


@pytest.mark.slow
def test_two_stage_type_i_random_intercept():
    assert _rate(206, 0.0, method="score", permutation_strategy="two-stage") <= _bound(
        N_REP
    )


@pytest.mark.slow
def test_between_strategy_type_i():
    """'between' is correct today because equal-size cell exchange preserves V.
    The refactor must not disturb it."""
    assert _rate(207, 0.0, method="score", permutation_strategy="between") <= _bound(
        N_REP
    )


@pytest.mark.slow
def test_individual_power_random_intercept():
    assert _rate(208, 0.6, method="score") > 0.5


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True, reason="M3b: level-2 power == its own Type I under 'within'"
)
def test_level2_power_under_within():
    assert _rate(209, 1.5, method="score", level2=True) > 0.5


def _glmm_rate(seed, beta, tau2=4.0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(N_REP_GLMM):
        x, z = rng.normal(size=N), rng.normal(size=N)
        b = rng.normal(scale=np.sqrt(tau2), size=G)
        y = rng.binomial(1, 1 / (1 + np.exp(-(beta * x + b[CLUSTER])))).astype(float)
        res = randomization_test_regression(
            pd.DataFrame({"x": x, "z": z}),
            pd.DataFrame({"y": y}),
            family="logistic_mixed",
            groups=CLUSTER,
            method="score",
            n_randomizations=B_GLMM,
            random_state=int(rng.integers(1 << 30)),
        )
        hits += float(np.ravel(res.raw_empirical_p)[0]) <= ALPHA
    return hits / N_REP_GLMM


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True, reason="M4/M6: measured 0.130, caused by the score offset"
)
def test_glmm_type_i_level1():
    """Corroboration only. The primary M4/M6 gates are the deterministic
    ``test_glmm_score_offset_is_zero`` and ``test_glmm_null_is_zero_centred``
    above: both state the defect directly on one dataset, where this needs 300
    replications to say the same thing less precisely."""
    assert _glmm_rate(210, 0.0) <= _bound(N_REP_GLMM)


@pytest.mark.slow
def test_glmm_power_level1():
    assert _glmm_rate(211, 0.8) > 0.5


# M7 note: the former ``test_ar_lmm_not_degenerate`` asserted power > 0.5 at
# beta = 0.8 under the reason "measured Type I 0.000, zero power" — which
# overgeneralised a single Type I measurement into a claim about power, and
# XPASSed while the defect was fully present. M7 has since split: M7b (SE spread
# 61.7x across identical DGPs) was the diverging REML solver and is fixed, with
# spread now 1.15x against the no-AR arm's 1.17x. What remains is M7a, whose
# signature is one-directional — every measured sd(draws)/SE ratio >= 1.0 — and
# is pinned deterministically by ``test_ar_lmm_spread_ratio`` above.

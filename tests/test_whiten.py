"""Tests for ``ModelFamily.whiten`` across all nine model families.

Whitening exists so that residual-permuting methods act on exchangeable units:
if ``Cov(e) = Sigma``, the permuted units are exchangeable only when Sigma is
proportional to I, so a family that models a non-identity error covariance must
expose ``W`` with ``W'W = Sigma^-1``.

The contract is deliberately narrow, and the narrowness is measured rather than
assumed. ``whiten`` standardises **modelled correlation structure only** -- LMM
clustering, GLMM working covariance, AR -- and never a mean-variance relation.
"""

from __future__ import annotations

import numpy as np
import pytest

from randomization_tests.families import (
    LinearFamily,
    LogisticFamily,
    MultinomialFamily,
    NegativeBinomialFamily,
    OrdinalFamily,
    PoissonFamily,
)
from randomization_tests.families_mixed import (
    LinearMixedFamily,
    LogisticMixedFamily,
)

G, M = 12, 8
N = G * M
CLUSTER = np.repeat(np.arange(G), M)
CELLS = [np.arange(g * M, (g + 1) * M) for g in range(G)]


# ------------------------------------------------------------------ #
# Non-mixed families
# ------------------------------------------------------------------ #


def test_linear_is_identity_without_ar():
    """``Cov(e) = sigma^2 I``, so nothing needs standardising."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, 2))
    y = X @ [1.0, -0.5] + rng.normal(size=N)
    fam = LinearFamily()
    fam.fit(X, y, True)
    np.testing.assert_allclose(fam.whiten(np.eye(N)), np.eye(N))


@pytest.mark.parametrize(
    "family_cls", [LogisticFamily, PoissonFamily, NegativeBinomialFamily]
)
def test_glm_does_not_standardise_by_variance(family_cls):
    """GLM ``whiten`` is the identity absent AR -- it must NOT divide by sqrt(v(mu)).

    Scaling by ``1/sqrt(v)`` equalises residual variances but leaves them
    non-identically *distributed* (a Bernoulli residual is two-valued whatever it
    is divided by), so it would buy the appearance of exchangeability without the
    substance.

    It is also unnecessary, which is the decisive point. Measured on logistic
    N=600, 400 replications: Type I was 0.0550 / 0.0525 / 0.0525 at working-weight
    spreads of 1 / 167 / **110,773**, against an acceptance bound of 0.0718, with
    the null p-value distribution indistinguishable from uniform (KS p = 0.17-0.53).
    The score ``X_j'(y - mu_hat)`` normalised by Fisher information is already
    studentised, so heteroscedasticity scales the permutation and sampling
    variances identically and calibration is preserved.

    This test pins that decision so the standardisation is not silently restored.
    """
    fam = family_cls()
    rng = np.random.default_rng(1)
    M_in = rng.normal(size=(N, 3))
    np.testing.assert_allclose(fam.whiten(M_in), M_in)


def test_glm_whitening_does_not_depend_on_the_data():
    """Corollary of the above: the operator is fixed at calibration, so two very
    different fitted-mean profiles must produce the same (identity) operator."""
    fam = LogisticFamily()
    rng = np.random.default_rng(2)
    for scale in (0.1, 25.0):
        X = rng.normal(size=(N, 2)) * scale
        y = rng.binomial(1, 1 / (1 + np.exp(-X @ [1.5, -1.0]))).astype(float)
        fam.fit(X, y, True)
        np.testing.assert_allclose(fam.whiten(np.eye(N)), np.eye(N))


@pytest.mark.parametrize("family_cls", [OrdinalFamily, MultinomialFamily])
def test_direct_permutation_families_are_identity(family_cls):
    """Identity because these permute Y and form no residuals -- not a claim that
    their residuals are homoscedastic."""
    fam = family_cls()
    assert fam.direct_permutation is True
    np.testing.assert_allclose(fam.whiten(np.eye(N)), np.eye(N))


# ------------------------------------------------------------------ #
# Mixed families
# ------------------------------------------------------------------ #


def _lmm(seed=11, tau2=6.0):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=N), rng.normal(size=N)
    y = np.empty(N)
    for ci in CELLS:
        y[ci] = 0.7 * z[ci] + rng.normal(scale=np.sqrt(tau2)) + rng.normal(size=len(ci))
    X = np.column_stack([x, z])
    return LinearMixedFamily().calibrate(X, y, groups=CLUSTER), X, y


def test_lmm_reproduces_marginal_precision():
    """``W'W`` must equal ``Vtilde^-1 = I - Z C22^-1 Z'`` (Woodbury), checking the
    stored factor against the projection the family already uses."""
    fam, _, _ = _lmm()
    assert fam.whitening_blocks is not None

    W = fam.whiten(np.eye(N))
    Z, C22 = np.asarray(fam.Z), np.asarray(fam.C22)
    expected = np.eye(N) - Z @ np.linalg.solve(C22, Z.T)
    np.testing.assert_allclose(W.T @ W, expected, atol=1e-9)


def test_lmm_whitening_is_block_local():
    """Cross-cluster independence must be preserved exactly, which is why the
    block-Cholesky factor is used rather than the symmetric root."""
    fam, _, _ = _lmm()
    W = fam.whiten(np.eye(N))
    for a, ci in enumerate(CELLS):
        for b, cj in enumerate(CELLS):
            if a != b:
                assert np.allclose(W[np.ix_(ci, cj)], 0.0)


def test_lmm_whitening_shape_round_trip():
    fam, _, _ = _lmm()
    rng = np.random.default_rng(3)
    assert fam.whiten(rng.normal(size=N)).shape == (N,)
    assert fam.whiten(rng.normal(size=(N, 4))).shape == (N, 4)


def test_lmm_whitening_unavailable_raises_for_ar():
    """No silent fallback: AR calibration has no block-local whitening, so whiten()
    must raise rather than apply a wrong operator."""
    fam, _, _ = _lmm()
    broken = LinearMixedFamily(
        re_struct=fam.re_struct,
        projection_A=fam.projection_A,
        sigma2=fam.sigma2,
        Z=fam.Z,
        C22=fam.C22,
        _groups_arr=fam._groups_arr,
        whitening_blocks=None,
    )
    with pytest.raises(NotImplementedError, match="not block-diagonal"):
        broken.whiten(np.zeros(N))


def test_glmm_whitens_on_the_working_scale():
    """GLMM whitening targets ``V_z = W^-1 + Z Sigma Z'`` -- the *working*-scale
    covariance, where the PQL score is defined. Whitening on the response scale
    would repeat M6, applying working-scale weights to response-scale residuals.
    """
    rng = np.random.default_rng(5)
    X = rng.normal(size=(N, 2))
    b = rng.normal(scale=2.0, size=G)
    y = rng.binomial(1, 1 / (1 + np.exp(-(X @ [0.5, -0.3] + b[CLUSTER])))).astype(float)
    fam = LogisticMixedFamily().calibrate(X, y, True, groups=CLUSTER)

    W_op = fam.whiten(np.eye(N))
    w = np.asarray(fam.W, dtype=float)
    Z = np.asarray(fam.Z, dtype=float)
    sigma = np.atleast_2d(np.asarray(fam.re_covariances[0], dtype=float))
    _, d = fam.re_struct[0]

    v_z = np.diag(1.0 / w)
    for g in range(G):
        rows = np.flatnonzero(np.equal(np.asarray(fam._groups_arr), g))
        cols = np.arange(g * d, (g + 1) * d)
        Zg = Z[np.ix_(rows, cols)]
        v_z[np.ix_(rows, rows)] += Zg @ sigma @ Zg.T

    np.testing.assert_allclose(W_op.T @ W_op, np.linalg.inv(v_z), atol=1e-8)

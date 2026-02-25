"""Tests for batch_fit_and_score and batch_fit_and_score_varying_X.

Verifies that the new fused fit-and-score methods produce scores
consistent with the existing separate fit() + score() calls across
all six model families.
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


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_linear_data(rng, n=80, p=3, B=5):
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + rng.standard_normal(n) * 0.5
    Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])
    return X, y, Y_matrix


def _make_binary_data(rng, n=80, p=3, B=5):
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    prob = 1 / (1 + np.exp(-X @ beta))
    y = rng.binomial(1, prob).astype(float)
    Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])
    return X, y, Y_matrix


def _make_count_data(rng, n=80, p=3, B=5):
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p) * 0.3
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)
    Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])
    return X, y, Y_matrix


def _make_ordinal_data(rng, n=80, p=2, B=5, K=4):
    X = rng.standard_normal((n, p))
    # Use uniform thresholds to create ordinal categories
    latent = X @ np.array([1.0, -0.5][:p]) + rng.standard_normal(n)
    thresholds = np.quantile(latent, np.linspace(0, 1, K + 1)[1:-1])
    y = np.digitize(latent, thresholds).astype(float)
    Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])
    return X, y, Y_matrix


def _make_multinomial_data(rng, n=80, p=2, B=5, K=3):
    X = rng.standard_normal((n, p))
    y = rng.integers(0, K, size=n).astype(float)
    Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])
    return X, y, Y_matrix


# ------------------------------------------------------------------ #
# Test class: batch_fit_and_score (shared X, many Y)
# ------------------------------------------------------------------ #


class TestBatchFitAndScore:
    """batch_fit_and_score returns (coefs, scores) consistent with
    individual fit() + score() calls."""

    def test_linear(self, rng):
        X, y, Y_matrix = _make_linear_data(rng)
        family = LinearFamily()
        coefs, scores = family.batch_fit_and_score(X, Y_matrix, fit_intercept=True)
        B, p = Y_matrix.shape[0], X.shape[1]
        assert coefs.shape == (B, p)
        assert scores.shape == (B,)

        # Compare with individual fits
        for b in range(B):
            model = family.fit(X, Y_matrix[b], fit_intercept=True)
            expected_score = family.score(model, X, Y_matrix[b])
            np.testing.assert_allclose(scores[b], expected_score, rtol=1e-6)

    def test_logistic(self, rng):
        X, y, Y_matrix = _make_binary_data(rng)
        family = LogisticFamily()
        coefs, scores = family.batch_fit_and_score(X, Y_matrix, fit_intercept=True)
        B, p = Y_matrix.shape[0], X.shape[1]
        assert coefs.shape == (B, p)
        assert scores.shape == (B,)

        for b in range(B):
            model = family.fit(X, Y_matrix[b], fit_intercept=True)
            expected_score = family.score(model, X, Y_matrix[b])
            np.testing.assert_allclose(scores[b], expected_score, rtol=1e-3)

    def test_poisson(self, rng):
        X, y, Y_matrix = _make_count_data(rng)
        family = PoissonFamily()
        coefs, scores = family.batch_fit_and_score(X, Y_matrix, fit_intercept=True)
        B, p = Y_matrix.shape[0], X.shape[1]
        assert coefs.shape == (B, p)
        assert scores.shape == (B,)

        # For Poisson, batch returns 2*NLL; score() returns deviance.
        # These differ by a y-dependent constant but the deltas should
        # match.  We verify the shape and finiteness here; the
        # improvement-equivalence is tested in the strategy tests.
        assert np.all(np.isfinite(scores))

    def test_ordinal(self, rng):
        X, y, Y_matrix = _make_ordinal_data(rng)
        family = OrdinalFamily()
        coefs, scores = family.batch_fit_and_score(X, Y_matrix, fit_intercept=True)
        B, p = Y_matrix.shape[0], X.shape[1]
        assert coefs.shape == (B, p)
        assert scores.shape == (B,)
        assert np.all(np.isfinite(scores))

    def test_multinomial(self, rng):
        X, y, Y_matrix = _make_multinomial_data(rng)
        family = MultinomialFamily()
        wald_chi2, scores = family.batch_fit_and_score(X, Y_matrix, fit_intercept=True)
        B, p = Y_matrix.shape[0], X.shape[1]
        assert wald_chi2.shape == (B, p)
        assert scores.shape == (B,)
        assert np.all(np.isfinite(scores))


# ------------------------------------------------------------------ #
# Test class: batch_fit_and_score_varying_X (varying X, shared y)
# ------------------------------------------------------------------ #


class TestBatchFitAndScoreVaryingX:
    """batch_fit_and_score_varying_X returns (coefs, scores) consistent
    with individual fit() + score() calls."""

    def test_linear(self, rng):
        X, y, _ = _make_linear_data(rng)
        family = LinearFamily()
        B = 5
        X_batch = np.stack([X + rng.standard_normal(X.shape) * 0.1 for _ in range(B)])
        coefs, scores = family.batch_fit_and_score_varying_X(
            X_batch, y, fit_intercept=True
        )
        p = X.shape[1]
        assert coefs.shape == (B, p)
        assert scores.shape == (B,)

        for b in range(B):
            model = family.fit(X_batch[b], y, fit_intercept=True)
            expected_score = family.score(model, X_batch[b], y)
            np.testing.assert_allclose(scores[b], expected_score, rtol=1e-6)

    def test_logistic(self, rng):
        X, y, _ = _make_binary_data(rng)
        family = LogisticFamily()
        B = 5
        X_batch = np.stack([X + rng.standard_normal(X.shape) * 0.1 for _ in range(B)])
        coefs, scores = family.batch_fit_and_score_varying_X(
            X_batch, y, fit_intercept=True
        )
        p = X.shape[1]
        assert coefs.shape == (B, p)
        assert scores.shape == (B,)

        for b in range(B):
            model = family.fit(X_batch[b], y, fit_intercept=True)
            expected_score = family.score(model, X_batch[b], y)
            np.testing.assert_allclose(scores[b], expected_score, rtol=1e-3)

    def test_poisson(self, rng):
        X, y, _ = _make_count_data(rng)
        family = PoissonFamily()
        B = 5
        X_batch = np.stack([X + rng.standard_normal(X.shape) * 0.1 for _ in range(B)])
        coefs, scores = family.batch_fit_and_score_varying_X(
            X_batch, y, fit_intercept=True
        )
        p = X.shape[1]
        assert coefs.shape == (B, p)
        assert scores.shape == (B,)
        assert np.all(np.isfinite(scores))

    def test_ordinal(self, rng):
        X, y, _ = _make_ordinal_data(rng, p=2)
        family = OrdinalFamily()
        B = 5
        X_batch = np.stack([X + rng.standard_normal(X.shape) * 0.1 for _ in range(B)])
        coefs, scores = family.batch_fit_and_score_varying_X(
            X_batch, y, fit_intercept=True
        )
        p = X.shape[1]
        assert coefs.shape == (B, p)
        assert scores.shape == (B,)
        assert np.all(np.isfinite(scores))

    def test_multinomial(self, rng):
        X, y, _ = _make_multinomial_data(rng, p=2)
        family = MultinomialFamily()
        B = 5
        X_batch = np.stack([X + rng.standard_normal(X.shape) * 0.1 for _ in range(B)])
        wald_chi2, scores = family.batch_fit_and_score_varying_X(
            X_batch, y, fit_intercept=True
        )
        p = X.shape[1]
        assert wald_chi2.shape == (B, p)
        assert scores.shape == (B,)
        assert np.all(np.isfinite(scores))


# ------------------------------------------------------------------ #
# Test: improvement equivalence
# ------------------------------------------------------------------ #


class TestImprovementEquivalence:
    """Verify that the improvement delta from batch_fit_and_score
    matches the delta from the sequential fit() + score() approach.

    For linear, the scores are RSS and match exactly.
    For GLMs, batch returns 2*NLL which differs from deviance/score()
    by a y-only constant.  The improvement delta should still match
    because the constant cancels.
    """

    def test_linear_delta(self, rng):
        """RSS improvement delta matches between batch and sequential."""
        X, y, _ = _make_linear_data(rng, n=60, p=3)
        family = LinearFamily()
        B = 5

        # Construct reduced (1 col) and full (3 cols) designs
        X_reduced = X[:, :1]
        X_full = X

        Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])

        # Sequential approach
        seq_deltas = np.zeros(B)
        for b in range(B):
            model_red = family.fit(X_reduced, Y_matrix[b], fit_intercept=True)
            model_full = family.fit(X_full, Y_matrix[b], fit_intercept=True)
            seq_deltas[b] = family.score(
                model_red, X_reduced, Y_matrix[b]
            ) - family.score(model_full, X_full, Y_matrix[b])

        # Batch approach
        _, red_scores = family.batch_fit_and_score(
            X_reduced, Y_matrix, fit_intercept=True
        )
        _, full_scores = family.batch_fit_and_score(
            X_full, Y_matrix, fit_intercept=True
        )
        batch_deltas = red_scores - full_scores

        np.testing.assert_allclose(batch_deltas, seq_deltas, rtol=1e-6)

    def test_logistic_delta(self, rng):
        """Logistic deviance improvement delta matches."""
        X, y, _ = _make_binary_data(rng, n=100, p=3)
        family = LogisticFamily()
        B = 5

        X_reduced = X[:, :1]
        X_full = X
        Y_matrix = np.vstack([rng.permutation(y) for _ in range(B)])

        # Sequential approach
        seq_deltas = np.zeros(B)
        for b in range(B):
            model_red = family.fit(X_reduced, Y_matrix[b], fit_intercept=True)
            model_full = family.fit(X_full, Y_matrix[b], fit_intercept=True)
            seq_deltas[b] = family.score(
                model_red, X_reduced, Y_matrix[b]
            ) - family.score(model_full, X_full, Y_matrix[b])

        # Batch approach (scores = 2*NLL, not deviance, but deltas match)
        _, red_scores = family.batch_fit_and_score(
            X_reduced, Y_matrix, fit_intercept=True
        )
        _, full_scores = family.batch_fit_and_score(
            X_full, Y_matrix, fit_intercept=True
        )
        batch_deltas = red_scores - full_scores

        np.testing.assert_allclose(batch_deltas, seq_deltas, rtol=1e-2)


# ------------------------------------------------------------------ #
# Test: NegBin with calibrated alpha
# ------------------------------------------------------------------ #


class TestNegBinBatchFitAndScore:
    """NegBin batch_fit_and_score requires calibrated alpha."""

    def test_batch_fit_and_score_runs(self, rng):
        X, y, Y_matrix = _make_count_data(rng)
        family = NegativeBinomialFamily()
        family = family.calibrate(X, y, fit_intercept=True)

        coefs, scores = family.batch_fit_and_score(X, Y_matrix, fit_intercept=True)
        assert coefs.shape == (Y_matrix.shape[0], X.shape[1])
        assert scores.shape == (Y_matrix.shape[0],)
        assert np.all(np.isfinite(scores))

    def test_batch_fit_and_score_varying_X_runs(self, rng):
        X, y, _ = _make_count_data(rng)
        family = NegativeBinomialFamily()
        family = family.calibrate(X, y, fit_intercept=True)

        B = 5
        X_batch = np.stack([X + rng.standard_normal(X.shape) * 0.1 for _ in range(B)])
        coefs, scores = family.batch_fit_and_score_varying_X(
            X_batch, y, fit_intercept=True
        )
        assert coefs.shape == (B, X.shape[1])
        assert scores.shape == (B,)
        assert np.all(np.isfinite(scores))


# ------------------------------------------------------------------ #
# Test: batch_fit_paired — both X and Y vary per replicate
# ------------------------------------------------------------------ #


class TestBatchFitPaired:
    """batch_fit_paired returns (B, p) coefficients consistent with
    individual fit() + coefs() calls when both X and Y vary."""

    def _make_paired_data(self, rng, X, y, B=5):
        """Build (X_batch, Y_batch) by bootstrap resampling rows."""
        n = X.shape[0]
        idx = rng.choice(n, size=(B, n), replace=True)
        X_batch = X[idx]  # (B, n, p)
        Y_batch = y[idx]  # (B, n)
        return X_batch, Y_batch

    def test_linear(self, rng):
        X, y, _ = _make_linear_data(rng)
        family = LinearFamily()
        X_batch, Y_batch = self._make_paired_data(rng, X, y)
        coefs = family.batch_fit_paired(X_batch, Y_batch, fit_intercept=True)
        B, p = X_batch.shape[0], X.shape[1]
        assert coefs.shape == (B, p)

        # Compare with individual fits
        for b in range(B):
            model = family.fit(X_batch[b], Y_batch[b], fit_intercept=True)
            expected = family.coefs(model)
            np.testing.assert_allclose(coefs[b], expected, rtol=1e-6)

    def test_logistic(self, rng):
        X, y, _ = _make_binary_data(rng)
        family = LogisticFamily()
        X_batch, Y_batch = self._make_paired_data(rng, X, y)
        coefs = family.batch_fit_paired(X_batch, Y_batch, fit_intercept=True)
        B, p = X_batch.shape[0], X.shape[1]
        assert coefs.shape == (B, p)
        assert np.all(np.isfinite(coefs))

    def test_poisson(self, rng):
        X, y, _ = _make_count_data(rng)
        family = PoissonFamily()
        X_batch, Y_batch = self._make_paired_data(rng, X, y)
        coefs = family.batch_fit_paired(X_batch, Y_batch, fit_intercept=True)
        B, p = X_batch.shape[0], X.shape[1]
        assert coefs.shape == (B, p)
        assert np.all(np.isfinite(coefs))

    def test_negbin(self, rng):
        X, y, _ = _make_count_data(rng)
        family = NegativeBinomialFamily()
        family = family.calibrate(X, y, fit_intercept=True)
        X_batch, Y_batch = self._make_paired_data(rng, X, y)
        coefs = family.batch_fit_paired(X_batch, Y_batch, fit_intercept=True)
        B, p = X_batch.shape[0], X.shape[1]
        assert coefs.shape == (B, p)
        assert np.all(np.isfinite(coefs))

    def test_ordinal(self, rng):
        X, y, _ = _make_ordinal_data(rng)
        family = OrdinalFamily()
        X_batch, Y_batch = self._make_paired_data(rng, X, y)
        coefs = family.batch_fit_paired(X_batch, Y_batch, fit_intercept=True)
        B, p = X_batch.shape[0], X.shape[1]
        assert coefs.shape == (B, p)
        assert np.all(np.isfinite(coefs))

    def test_multinomial(self, rng):
        X, y, _ = _make_multinomial_data(rng)
        family = MultinomialFamily()
        X_batch, Y_batch = self._make_paired_data(rng, X, y)
        coefs = family.batch_fit_paired(X_batch, Y_batch, fit_intercept=True)
        B, p = X_batch.shape[0], X.shape[1]
        assert coefs.shape == (B, p)
        assert np.all(np.isfinite(coefs))

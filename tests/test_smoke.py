"""Large-n smoke tests for regression detection.

These tests verify that the permutation test engine completes within
a reasonable time bound on moderately large datasets (n=10,000).
They catch accidental quadratic behaviour, memory blowouts, and
regressions in the vectorised paths.

All tests are marked ``@pytest.mark.slow`` and excluded from the
default ``pytest`` run.  Run them explicitly::

    pytest -m slow
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from randomization_tests.core import permutation_test_regression

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

N = 10_000
P = 5
SEED = 42


def _make_large_linear(
    n: int = N, p: int = P, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i + 1}": rng.standard_normal(n) for i in range(p)})
    y = pd.DataFrame(
        {"y": 2.0 * X["x1"] - 0.5 * X["x2"] + rng.standard_normal(n) * 0.3}
    )
    return X, y


def _make_large_binary(
    n: int = N, p: int = P, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i + 1}": rng.standard_normal(n) for i in range(p)})
    logits = 1.5 * X["x1"] - 0.8 * X["x2"]
    probs = 1 / (1 + np.exp(-logits))
    y = pd.DataFrame({"y": rng.binomial(1, probs)})
    return X, y


# ------------------------------------------------------------------ #
# Smoke tests
# ------------------------------------------------------------------ #


@pytest.mark.slow
class TestLinearSmoke:
    """n=10,000, p=5, ter Braak linear — should complete quickly."""

    def test_completes_within_bound(self) -> None:
        X, y = _make_large_linear()
        t0 = time.monotonic()
        result = permutation_test_regression(
            X, y, n_permutations=100, method="ter_braak", random_state=SEED
        )
        elapsed = time.monotonic() - t0
        assert elapsed < 30, f"Linear smoke test took {elapsed:.1f}s (limit 30s)"
        assert len(result["model_coefs"]) == P
        assert result.family.name == "linear"

    def test_result_structure(self) -> None:
        X, y = _make_large_linear()
        result = permutation_test_regression(
            X, y, n_permutations=100, method="ter_braak", random_state=SEED
        )
        assert len(result["permuted_p_values"]) == P
        assert len(result["classic_p_values"]) == P
        assert result["diagnostics"]["n_observations"] == N
        assert result["diagnostics"]["n_features"] == P


@pytest.mark.slow
class TestLogisticSmoke:
    """n=10,000, p=5, ter Braak logistic — slower due to iterative fits."""

    def test_completes_within_bound(self) -> None:
        X, y = _make_large_binary()
        t0 = time.monotonic()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="ter_braak", random_state=SEED
        )
        elapsed = time.monotonic() - t0
        assert elapsed < 120, f"Logistic smoke test took {elapsed:.1f}s (limit 120s)"
        assert len(result["model_coefs"]) == P
        assert result.family.name == "logistic"


@pytest.mark.slow
class TestKennedyIndividualSmoke:
    """n=10,000, p=5 (2 confounders), Kennedy individual."""

    def test_completes_within_bound(self) -> None:
        X, y = _make_large_linear()
        t0 = time.monotonic()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=100,
            method="kennedy",
            confounders=["x4", "x5"],
            random_state=SEED,
        )
        elapsed = time.monotonic() - t0
        assert elapsed < 60, (
            f"Kennedy individual smoke test took {elapsed:.1f}s (limit 60s)"
        )
        assert result["method"] == "kennedy"
        # Confounders should be marked N/A
        assert result["permuted_p_values"][3] == "N/A (confounder)"
        assert result["permuted_p_values"][4] == "N/A (confounder)"


@pytest.mark.slow
class TestKennedyJointSmoke:
    """n=10,000, p=5 (2 confounders), Kennedy joint."""

    def test_completes_within_bound(self) -> None:
        X, y = _make_large_linear()
        t0 = time.monotonic()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=100,
            method="kennedy_joint",
            confounders=["x4", "x5"],
            random_state=SEED,
        )
        elapsed = time.monotonic() - t0
        assert elapsed < 60, f"Kennedy joint smoke test took {elapsed:.1f}s (limit 60s)"
        assert result["method"] == "kennedy_joint"
        assert "p_value" in result
        assert 0 < result["p_value"] <= 1.0

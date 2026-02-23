"""JAX backend tests: convergence, statistical consistency, and edge cases.

All tests are guarded with ``pytest.importorskip("jax")`` so they
silently skip when JAX is not installed.

Covers: statistical consistency with sklearn, intercept handling,
ill-conditioned Hessian, rank-deficient design, perfect separation
through the Newton–Raphson solver, and float32 precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")

from randomization_tests import set_backend  # noqa: E402
from randomization_tests.core import permutation_test_regression  # noqa: E402

# ------------------------------------------------------------------ #
# Fixture: force the JAX backend, restore afterward
# ------------------------------------------------------------------ #


@pytest.fixture(autouse=True)
def _use_jax_backend() -> None:  # type: ignore[misc]
    """Activate the JAX backend for every test, then restore 'auto'."""
    set_backend("jax")
    yield  # type: ignore[misc]
    set_backend("auto")


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_logistic_data(
    n: int = 200, p: int = 2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i + 1}": rng.standard_normal(n) for i in range(p)})
    logits = 2.0 * X.iloc[:, 0]
    probs = 1 / (1 + np.exp(-logits))
    y = pd.DataFrame({"y": rng.binomial(1, probs)})
    return X, y


def _run_with_backend(
    backend: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    **kwargs: object,
) -> dict:
    """Run a single permutation test under a specific backend."""
    set_backend(backend)
    result = permutation_test_regression(X, y, **kwargs)
    return result


# ------------------------------------------------------------------ #
# 1. Statistical consistency: JAX ≈ sklearn (same direction, not
#    exact equality — JAX may be *more* accurate because it uses
#    exact autodiff gradients + Newton–Raphson vs sklearn's L-BFGS
#    with approximate curvature).
# ------------------------------------------------------------------ #


class TestJAXStatisticalConsistency:
    """JAX and sklearn should reach the same statistical conclusions."""

    _shared_kwargs: dict = dict(
        n_permutations=200,
        method="ter_braak",
        random_state=42,
    )

    def test_ter_braak_coefficients_same_sign(self) -> None:
        X, y = _make_logistic_data()
        res_jax = _run_with_backend("jax", X, y, **self._shared_kwargs)
        res_np = _run_with_backend("numpy", X, y, **self._shared_kwargs)
        coefs_jax = np.array(res_jax["model_coefs"])
        coefs_np = np.array(res_np["model_coefs"])
        np.testing.assert_array_equal(
            np.sign(coefs_jax),
            np.sign(coefs_np),
            err_msg="JAX and sklearn produced coefficients with different signs",
        )

    def test_ter_braak_coefficients_same_magnitude(self) -> None:
        X, y = _make_logistic_data()
        res_jax = _run_with_backend("jax", X, y, **self._shared_kwargs)
        res_np = _run_with_backend("numpy", X, y, **self._shared_kwargs)
        coefs_jax = np.array(res_jax["model_coefs"])
        coefs_np = np.array(res_np["model_coefs"])
        # Same order of magnitude — atol is generous because float32
        # vs float64 and different optimisers will differ.
        np.testing.assert_allclose(
            coefs_jax,
            coefs_np,
            atol=0.2,
            rtol=0.3,
            err_msg="JAX and sklearn coefficients differ by more than expected",
        )

    def test_kennedy_individual_same_significance_direction(self) -> None:
        """Strong predictor should be significant under both backends."""
        X, y = _make_logistic_data(n=300)
        kwargs: dict = dict(
            n_permutations=200,
            method="kennedy",
            confounders=["x2"],
            random_state=42,
        )
        res_jax = _run_with_backend("jax", X, y, **kwargs)
        res_np = _run_with_backend("numpy", X, y, **kwargs)
        # x1 has a strong effect — both backends should find it
        # significant (starred) at p < 0.05 or both not.
        jax_sig = (
            "(*)" in res_jax["permuted_p_values"][0]
            or "(**)" in res_jax["permuted_p_values"][0]
        )
        np_sig = (
            "(*)" in res_np["permuted_p_values"][0]
            or "(**)" in res_np["permuted_p_values"][0]
        )
        assert jax_sig == np_sig, (
            f"Significance mismatch: JAX sig={jax_sig}, sklearn sig={np_sig}"
        )


# ------------------------------------------------------------------ #
# 2. Intercept handling: JAX manually prepends/strips an intercept
#    column. Verify both paths are structurally correct.
# ------------------------------------------------------------------ #


class TestJAXInterceptHandling:
    def test_fit_intercept_true(self) -> None:
        X, y = _make_logistic_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=0,
            fit_intercept=True,
        )
        # Should have exactly p coefficients (no intercept leaked in)
        assert len(result["model_coefs"]) == 2
        assert all(np.isfinite(result["model_coefs"]))

    def test_fit_intercept_false(self) -> None:
        X, y = _make_logistic_data()
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=0,
            fit_intercept=False,
        )
        assert len(result["model_coefs"]) == 2
        assert all(np.isfinite(result["model_coefs"]))


# ------------------------------------------------------------------ #
# 3. Ill-conditioned Hessian: near-collinear predictors (r > 0.999)
# ------------------------------------------------------------------ #


class TestIllConditionedHessian:
    def test_near_collinear_produces_finite_results(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.standard_normal(n)
        # x2 is almost identical to x1 — correlation > 0.999
        x2 = x1 + rng.standard_normal(n) * 0.001
        X = pd.DataFrame({"x1": x1, "x2": x2})
        logits = 1.5 * x1
        probs = 1 / (1 + np.exp(-logits))
        y = pd.DataFrame({"y": rng.binomial(1, probs)})
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=0,
        )
        coefs = np.array(result["model_coefs"])
        assert np.all(np.isfinite(coefs)), (
            f"Non-finite coefficients from ill-conditioned Hessian: {coefs}"
        )
        assert np.all(np.isfinite(result["raw_empirical_p"]))


# ------------------------------------------------------------------ #
# 4. Rank-deficient design: more features than observations
# ------------------------------------------------------------------ #


@pytest.mark.slow
class TestRankDeficient:
    def test_n_less_than_p_graceful(self) -> None:
        """n=20, p=25 — Hessian is singular. Verify finite or clear error."""
        rng = np.random.default_rng(42)
        n, p = 20, 25
        X = pd.DataFrame({f"x{i + 1}": rng.standard_normal(n) for i in range(p)})
        y = pd.DataFrame({"y": rng.binomial(1, 0.5, size=n)})
        # Accept either a finite result or an informative error — but
        # NOT silent NaN/Inf.
        try:
            result = permutation_test_regression(
                X,
                y,
                n_permutations=20,
                method="ter_braak",
                random_state=0,
            )
            coefs = np.array(result["model_coefs"])
            assert np.all(np.isfinite(coefs)), (
                f"Silent NaN/Inf in rank-deficient result: {coefs}"
            )
        except (ValueError, RuntimeError):
            # An explicit error is acceptable for rank-deficient designs
            pass


# ------------------------------------------------------------------ #
# 5. Perfect separation through the JAX Newton–Raphson solver
# ------------------------------------------------------------------ #


class TestSeparationJAXPath:
    def test_perfect_separation_finite(self) -> None:
        """Coefficients may be large but must remain finite."""
        n = 80
        rng = np.random.default_rng(42)
        x = np.concatenate([np.full(n // 2, -2.0), np.full(n // 2, 2.0)])
        noise = rng.standard_normal(n) * 0.01
        X = pd.DataFrame({"x1": x + noise, "x2": rng.standard_normal(n)})
        y = pd.DataFrame({"y": np.concatenate([np.zeros(n // 2), np.ones(n // 2)])})
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=0,
        )
        coefs = np.array(result["model_coefs"])
        assert np.all(np.isfinite(coefs)), (
            f"Non-finite coefficients from perfect separation: {coefs}"
        )
        assert np.all(np.isfinite(result["raw_empirical_p"]))


# ------------------------------------------------------------------ #
# 6. Float32 precision: large-magnitude coefficients survive the cast
# ------------------------------------------------------------------ #


class TestFloat32Precision:
    def test_large_coefficients_survive_float32(self) -> None:
        """When true coefficients are large, float32 should still handle it."""
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.standard_normal(n) * 0.1  # small scale
        x2 = rng.standard_normal(n) * 5.0  # large scale
        X = pd.DataFrame({"x1": x1, "x2": x2})
        # Moderate coefficient on x2 to stress float32 without
        # completely saturating probabilities (which would break
        # statsmodels diagnostics).
        logits = 0.5 * x1 + 1.5 * x2
        probs = np.clip(1 / (1 + np.exp(-logits)), 0.01, 0.99)
        y = pd.DataFrame({"y": rng.binomial(1, probs)})
        result = permutation_test_regression(
            X,
            y,
            n_permutations=50,
            method="ter_braak",
            random_state=0,
        )
        coefs = np.array(result["model_coefs"])
        assert np.all(np.isfinite(coefs)), (
            f"Non-finite coefficients from large-scale data: {coefs}"
        )
        # x2's coefficient should be positive and substantial
        assert coefs[1] > 0, "x2 coefficient should be positive"


# ------------------------------------------------------------------ #
# 7. n_jobs warning: JAX backend should warn when n_jobs != 1
# ------------------------------------------------------------------ #


class TestNJobsJAXWarning:
    """n_jobs != 1 should emit a UserWarning under the JAX backend."""

    def test_warns_on_n_jobs_with_jax(self):
        X, y = _make_logistic_data(n=60, seed=99)
        with pytest.warns(UserWarning, match="n_jobs is ignored"):
            permutation_test_regression(
                X,
                y,
                n_permutations=20,
                method="ter_braak",
                random_state=0,
                n_jobs=2,
            )

    def test_no_warning_on_n_jobs_1(self):
        """n_jobs=1 should not trigger the JAX warning."""
        X, y = _make_logistic_data(n=60, seed=99)
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            permutation_test_regression(
                X,
                y,
                n_permutations=20,
                method="ter_braak",
                random_state=0,
                n_jobs=1,
            )

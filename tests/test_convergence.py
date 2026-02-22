"""Tests for Newton–Raphson convergence under float64 arithmetic.

Verifies that the JAX logistic solver converges at the default
tolerance (:data:`_DEFAULT_TOL` = 1e-8) across a range of realistic
data scenarios (varying *n*, *p*, signal strength, and pathological
conditions).

All arithmetic uses float64 (matching the production solver).

All tests are guarded with ``pytest.importorskip("jax")``.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from randomization_tests._backends._jax import (  # noqa: E402
    _DEFAULT_TOL,
    JaxBackend,
    _logistic_grad,
    _logistic_hessian,
)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

B = 50  # permutations per scenario


def _make_dataset(
    name: str,
    n: int,
    p: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X_aug_with_intercept, y)`` in float64."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))

    if name == "standard":
        logits = 2.0 * X[:, 0]
    elif name == "weak_signal":
        logits = 0.3 * X[:, 0]
    elif name == "many_features":
        logits = 2.0 * X[:, 0] - 1.0 * X[:, 1]
    elif name == "large_n":
        logits = 1.5 * X[:, 0] - 0.8 * X[:, 1]
    elif name == "collinear":
        X[:, 1] = X[:, 0] + rng.standard_normal(n) * 0.001
        logits = 1.5 * X[:, 0]
    elif name == "separated":
        x_sep = np.concatenate([np.full(n // 2, -2.0), np.full(n // 2, 2.0)])
        X[:, 0] = x_sep + rng.standard_normal(n) * 0.01
        logits = 5.0 * X[:, 0]
    elif name == "balanced_null":
        logits = np.zeros(n)
    else:
        raise ValueError(name)

    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs, size=n).astype(np.float64)
    X_aug = np.column_stack([np.ones(n), X]).astype(np.float64)
    return X_aug, y


def _run_newton(
    X_j: jnp.ndarray,
    y_j: jnp.ndarray,
    max_iter: int = 200,
    tol: float | None = None,
) -> tuple[int | None, float]:
    """Run Newton–Raphson manually.

    Returns ``(converged_at_iter_or_None, final_max_grad)``.
    """
    beta = jnp.zeros(X_j.shape[1], dtype=jnp.float64)
    for i in range(max_iter):
        g = _logistic_grad(beta, X_j, y_j)
        max_g = float(jnp.max(jnp.abs(g)))
        if tol is not None and max_g < tol:
            return i, max_g
        H = _logistic_hessian(beta, X_j, y_j)
        beta = beta - jnp.linalg.solve(H, g)
    final_g = float(jnp.max(jnp.abs(_logistic_grad(beta, X_j, y_j))))
    return None, final_g


# ------------------------------------------------------------------ #
# Well-conditioned scenarios: 100 % convergence at _DEFAULT_TOL
# ------------------------------------------------------------------ #

_WELL_CONDITIONED = [
    ("standard", 200, 2),
    ("weak_signal", 200, 2),
    ("balanced_null", 200, 2),
    ("standard", 200, 5),
    ("standard", 200, 10),
    ("standard", 200, 20),
    ("standard", 200, 50),
    ("standard", 500, 20),
    ("standard", 500, 50),
    ("standard", 1000, 10),
    ("standard", 1000, 50),
    ("many_features", 200, 10),
    ("large_n", 1000, 5),
    ("separated", 200, 2),
    ("standard", 100, 20),
]


@pytest.mark.parametrize(
    "scenario,n,p",
    _WELL_CONDITIONED,
    ids=[f"{s}-n{n}-p{p}" for s, n, p in _WELL_CONDITIONED],
)
class TestConvergenceWellConditioned:
    """All *B* permutations must converge at ``_DEFAULT_TOL``."""

    def test_all_converge_at_default_tol(self, scenario: str, n: int, p: int) -> None:
        X_aug, y_base = _make_dataset(scenario, n, p)
        X_j = jnp.array(X_aug)
        rng = np.random.default_rng(0)

        n_failed = 0
        for _ in range(B):
            y_perm = rng.permutation(y_base).astype(np.float64)
            converged_at, _ = _run_newton(X_j, jnp.array(y_perm), tol=_DEFAULT_TOL)
            if converged_at is None:
                n_failed += 1

        assert n_failed == 0, (
            f"{n_failed}/{B} permutations did not converge at "
            f"tol={_DEFAULT_TOL} for {scenario} (n={n}, p={p})"
        )

    def test_converges_within_10_iterations(
        self, scenario: str, n: int, p: int
    ) -> None:
        """Newton-Raphson on well-conditioned data should converge fast."""
        X_aug, y_base = _make_dataset(scenario, n, p)
        X_j = jnp.array(X_aug)
        rng = np.random.default_rng(0)

        max_iters_seen = 0
        for _ in range(B):
            y_perm = rng.permutation(y_base).astype(np.float64)
            converged_at, _ = _run_newton(X_j, jnp.array(y_perm), tol=_DEFAULT_TOL)
            if converged_at is not None:
                max_iters_seen = max(max_iters_seen, converged_at)

        assert max_iters_seen <= 10, (
            f"Worst-case convergence took {max_iters_seen} iterations "
            f"for {scenario} (n={n}, p={p}); expected <= 10"
        )


# ------------------------------------------------------------------ #
# Pathological scenarios: convergence is NOT guaranteed
# ------------------------------------------------------------------ #

_PATHOLOGICAL = [
    ("collinear", 200, 2),
    ("standard", 100, 50),
]


@pytest.mark.parametrize(
    "scenario,n,p",
    _PATHOLOGICAL,
    ids=[f"{s}-n{n}-p{p}" for s, n, p in _PATHOLOGICAL],
)
class TestConvergencePathological:
    """Pathological data may not converge -- verify finite results."""

    def test_results_finite_even_without_convergence(
        self, scenario: str, n: int, p: int
    ) -> None:
        """Solver should produce finite betas regardless."""
        X_aug, y_base = _make_dataset(scenario, n, p)
        backend = JaxBackend()
        rng = np.random.default_rng(0)
        Y_matrix = np.stack([rng.permutation(y_base) for _ in range(B)]).astype(
            np.float64
        )

        result = backend.batch_logistic(
            X_aug[:, 1:],  # strip intercept -- backend adds it
            Y_matrix,
            fit_intercept=True,
            max_iter=100,
        )
        n_finite = np.sum(np.all(np.isfinite(result), axis=1))
        # At least half should produce finite coefficients
        assert n_finite >= B // 2, (
            f"Only {n_finite}/{B} permutations produced finite "
            f"coefficients for {scenario} (n={n}, p={p})"
        )


# ------------------------------------------------------------------ #
# Default tolerance sanity checks
# ------------------------------------------------------------------ #


class TestDefaultTolValue:
    """Verify the constant is what we expect and is appropriately set."""

    def test_default_tol_value(self) -> None:
        assert _DEFAULT_TOL == 1e-8

    def test_default_tol_above_float64_noise_floor(self) -> None:
        """``_DEFAULT_TOL`` should be well above the float64 gradient noise.

        The irreducible gradient oscillation in float64 scales roughly
        as ``sqrt(n * p) * eps_64``.  For *n* = 1000, *p* = 50 this
        is ~5e-14.  ``_DEFAULT_TOL`` = 1e-8 sits six orders of
        magnitude above this, giving ample margin even for
        ill-conditioned Hessians.
        """
        worst_case_noise = np.sqrt(1000 * 50) * np.finfo(np.float64).eps
        assert worst_case_noise * 2 < _DEFAULT_TOL, (
            f"_DEFAULT_TOL={_DEFAULT_TOL} is too close to the float64 "
            f"noise floor ({worst_case_noise:.2e}) for n=1000, p=50"
        )

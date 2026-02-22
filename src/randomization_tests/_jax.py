"""JAX-accelerated logistic regression for permutation tests.

This module isolates all JAX-specific code — automatic differentiation,
JIT compilation, and vectorised mapping (vmap) — so that ``core.py``
can remain backend-agnostic.

Architectural context
~~~~~~~~~~~~~~~~~~~~~
Prior to v0.2.0, the JAX helpers lived inline in ``core.py``.  Extracting
them into a dedicated ``_jax.py`` module serves two purposes:

1. **Separation of concerns** — ``core.py`` now expresses the three
   permutation strategies (ter Braak, Kennedy individual, Kennedy joint)
   purely in terms of algorithm structure.  All JAX-specific details —
   JIT compilation, automatic differentiation, vmap semantics, array
   type conversions — are encapsulated here.

2. **Clean extraction path for v0.3.0** — the planned ``_backends/``
   package (with a ``BackendProtocol``) will promote this file to
   ``_backends/_jax.py``.  Keeping JAX code in a self-contained module
   means the v0.3.0 refactor only needs to wrap these functions behind
   the protocol interface rather than pulling them out of a 1 200-line
   core module.

Two public functions are exported:

``fit_logistic_batch_jax``
    Shared design matrix *X*, multiple response vectors *Y* (used by
    the ter Braak logistic path where we permute Y, not X).

``fit_logistic_varying_X_jax``
    Multiple design matrices *X*, single response vector *y* (used by
    the Kennedy individual logistic path where each permutation
    replaces one column of X with a permuted exposure-residual
    reconstruction).

Both return **NumPy arrays** of fitted coefficients so callers never
need to touch JAX types directly.  This is deliberate: the JAX arrays
produced by ``jit(vmap(...))`` are materialised into NumPy at the
boundary via ``np.asarray()``, keeping the rest of the codebase free
of JAX-specific array semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._config import get_backend

if TYPE_CHECKING:
    import jax

# ------------------------------------------------------------------ #
# Optional JAX import
# ------------------------------------------------------------------ #
# JAX provides two features that dramatically accelerate logistic
# regression permutation loops:
#
# 1. **Automatic differentiation (autodiff)** — jax.grad computes
#    exact gradients of the log-likelihood without hand-coding
#    derivative formulas.  This makes the Newton–Raphson solver both
#    concise and numerically stable.
#
# 2. **vmap (vectorised map)** — transforms a function that processes
#    one permutation into a function that processes all B permutations
#    in a single batched call.  Combined with JIT compilation, this
#    leverages XLA's kernel fusion and, when available, GPU parallelism.
#
# If JAX is not installed the module falls back to sklearn's
# LogisticRegression in a Python loop — correct but slower.
#
# The actual decision of whether to USE JAX is deferred to runtime via
# get_backend(), which checks (in order): programmatic override →
# RANDOMIZATION_TESTS_BACKEND env var → auto-detection.  The import
# here only determines _CAN_IMPORT_JAX (whether it's installed); the
# policy decision is separate.  See _config.py for details.
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    _CAN_IMPORT_JAX = True
except ImportError:
    _CAN_IMPORT_JAX = False


def jax_is_available() -> bool:
    """Return ``True`` if JAX can be imported."""
    return _CAN_IMPORT_JAX


def use_jax() -> bool:
    """Return True if JAX should be used for the current call.

    This combines two separate checks:

    1. ``_CAN_IMPORT_JAX`` — determined once at module load by the
       ``try/except ImportError`` above.  This answers "is JAX
       installed?" and never changes during a process.

    2. ``get_backend() == "jax"`` — the runtime policy decision.
       ``get_backend()`` checks (in order): programmatic override via
       ``set_backend()``, the ``RANDOMIZATION_TESTS_BACKEND`` env var,
       and auto-detection.  This allows users to disable JAX even when
       it is installed (e.g., for reproducibility against the sklearn
       fallback, or for debugging).  See ``_config.py`` for the full
       resolution logic.

    Callers in ``core.py`` use this as a simple boolean gate::

        if _use_jax():
            all_coefs = fit_logistic_batch_jax(...)
        else:
            # sklearn loop fallback
    """
    return _CAN_IMPORT_JAX and get_backend() == "jax"


# ------------------------------------------------------------------ #
# JAX helpers (logistic regression via autodiff)
# ------------------------------------------------------------------ #
# Logistic regression maximises the log-likelihood:
#
#   ℓ(β) = Σᵢ [ yᵢ log(pᵢ) + (1 - yᵢ) log(1 - pᵢ) ]
#
# where pᵢ = σ(Xᵢβ) = 1 / (1 + exp(-Xᵢβ)) is the predicted
# probability from the sigmoid function.
#
# Minimising the *negative* log-likelihood (NLL) is equivalent and
# fits the standard optimisation convention.  The functions below
# implement a Newton–Raphson solver:
#
#   β_{t+1} = β_t − H⁻¹(β_t) · ∇ℓ(β_t)
#
# where ∇ℓ is the gradient vector and H is the Hessian matrix.  JAX
# computes both via automatic differentiation — no manual derivation
# of the gradient or Hessian is needed.  The solver converges
# quadratically near the optimum (typically < 10 iterations).

if _CAN_IMPORT_JAX:

    @jit
    def _logistic_nll(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the negative log-likelihood for logistic regression.

        The standard NLL is:

            NLL = −Σᵢ [ yᵢ log(pᵢ) + (1 − yᵢ) log(1 − pᵢ) ]

        where pᵢ = σ(Xᵢβ).  Direct computation is numerically
        dangerous: when |Xᵢβ| is large, pᵢ saturates to 0 or 1 and
        log(pᵢ) or log(1 − pᵢ) → −∞.

        We use the *logaddexp* form instead:

            NLL = Σᵢ log(1 + exp(−sᵢ · Xᵢβ))

        where sᵢ = 2yᵢ − 1 ∈ {−1, +1}.  ``jnp.logaddexp(0, x)``
        computes log(1 + exp(x)) without overflow, making this form
        numerically stable for all logit magnitudes.
        """
        logits = X @ beta
        # Numerically stable: log(1 + exp(-y_signed * logits))
        # where y_signed = 2*y - 1
        return jnp.sum(jnp.logaddexp(0.0, -logits * (2.0 * y - 1.0)))

    # The gradient ∇NLL(β) is computed via jax.grad — automatic
    # differentiation of _logistic_nll with respect to its first
    # argument β.  Wrapping in jit compiles the gradient computation
    # into XLA for zero Python-level overhead at evaluation time.
    #
    # The analytical gradient of the logistic NLL is:
    #   ∇NLL = X'(p − y)
    # where p = σ(Xβ).  JAX derives this automatically from the NLL
    # definition above, guaranteeing correctness without hand-coding.
    _logistic_grad = jit(grad(_logistic_nll))

    @jit
    def _logistic_hessian_diag(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the Hessian matrix of the logistic NLL.

        The Hessian of the logistic NLL has a known closed form:

            H = X' W X

        where W = diag(pᵢ(1 − pᵢ)) is the diagonal weight matrix of
        variance factors (each pᵢ = σ(Xᵢβ)).  This matrix is always
        positive semi-definite because pᵢ(1 − pᵢ) ≥ 0, guaranteeing
        the Newton step is a descent direction.

        We implement the closed form directly rather than using
        ``jax.hessian(_logistic_nll)`` because:
          1. The matrix-vector product X'WX is O(n·p²), matching the
             cost of a single gradient evaluation, while the generic
             autodiff Hessian would compute p separate gradient-vector
             products at O(n·p) each — same asymptotic cost but higher
             constant factor from the additional tracing overhead.
          2. The closed form is equivalent to the Fisher information
             matrix, making the Newton step identical to Fisher scoring
             — a well-studied algorithm for GLMs.

        Note: The function name ``_logistic_hessian_diag`` is
        historical; the returned matrix is the full (p × p) Hessian,
        not just its diagonal.
        """
        p = jax.nn.sigmoid(X @ beta)
        W = p * (1.0 - p)
        return (X.T * W[None, :]) @ X

    def fit_logistic_batch_jax(
        X_base: np.ndarray,
        Y_matrix: np.ndarray,
        max_iter: int = 100,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Fit logistic regression for many Y vectors at once using vmap.

        This is the **shared-X** variant used by the ter Braak logistic
        path.  In ter Braak's method, the design matrix X is the same
        for every permutation — only Y is permuted (via residual
        reconstruction and Bernoulli resampling; see ``core.py``
        ``_ter_braak_logistic``).  This means vmap only needs to
        vectorise over the response axis.

        ``vmap(_solve_one)`` transforms the scalar solve (one y vector
        → one β vector) into a batched solve (B y vectors → B β
        vectors) in a single XLA kernel launch.  Combined with ``jit``,
        this fuses the entire B-permutation loop into compiled device
        code — no Python-level loop, no per-permutation dispatch
        overhead.

        Args:
            X_base: Design matrix of shape ``(n, p)`` shared across
                all B permutations.  Should NOT include an intercept
                column (one is prepended when *fit_intercept* is True).
            Y_matrix: Matrix of shape ``(B, n)`` where each row is a
                permuted binary response vector.
            max_iter: Maximum Newton–Raphson iterations per solve.
                100 is conservative; typical convergence is < 10
                iterations for well-conditioned problems.
            fit_intercept: Whether to prepend a column of ones to
                match sklearn's ``LogisticRegression(fit_intercept=True)``.
                The intercept coefficient β₀ is stripped from the
                returned array so callers see only feature coefficients.

        Returns:
            Coefficient matrix of shape ``(B, p)`` — one row per
            permutation, one column per feature (intercept excluded).
        """
        # When fit_intercept is True, prepend an intercept column to
        # match sklearn's default.  The solver returns [β₀, β₁, …, βₚ];
        # we slice off β₀ so the caller sees only feature coefficients.
        if fit_intercept:
            ones = np.ones((X_base.shape[0], 1), dtype=X_base.dtype)
            X_aug = np.hstack([ones, X_base])
        else:
            X_aug = X_base

        # Convert to JAX arrays (float32 for performance).  The design
        # matrix X_j is captured by _solve_one's closure — it does NOT
        # vary across the vmap axis.  Only y_vec changes per call.
        X_j = jnp.array(X_aug, dtype=jnp.float32)
        Y_j = jnp.array(Y_matrix, dtype=jnp.float32)

        def _solve_one(y_vec: jax.Array) -> jax.Array:
            """Newton–Raphson solve for a single response vector.

            The update rule is:
                β_{t+1} = β_t − H⁻¹(β_t) · ∇NLL(β_t)

            where ∇NLL is the gradient of the negative log-likelihood
            (from ``_logistic_grad``) and H is its Hessian (from
            ``_logistic_hessian_diag``).

            We use ``jnp.linalg.solve(H, g)`` instead of explicitly
            computing H⁻¹·g because:
              1. ``solve`` is numerically stabler (no matrix inversion).
              2. It leverages LAPACK's LU-factorisation under the hood,
                 which is O(p³) — the same cost as inversion but with
                 better conditioning.

            JAX traces this Python loop at JIT-compile time, unrolling
            all ``max_iter`` iterations into the XLA computation graph.
            This means the loop count must be a compile-time constant,
            but the payoff is that the entire solve (including
            gradient and Hessian evaluation) is fused into one device
            kernel with no Python dispatch overhead per iteration.

            Note: There is no early-stopping convergence check.  JAX's
            JIT compilation requires static control flow — a dynamic
            ``while`` loop is possible via ``jax.lax.while_loop`` but
            adds complexity.  The fixed iteration count is conservative
            (100 iterations; Newton–Raphson typically converges in < 10
            for well-conditioned logistic regression).  Adding ``tol``-
            based convergence control is planned for v0.3.0.
            """
            beta = jnp.zeros(X_j.shape[1], dtype=jnp.float32)
            for _ in range(max_iter):
                g = _logistic_grad(beta, X_j, y_vec)
                H = _logistic_hessian_diag(beta, X_j, y_vec)
                step = jnp.linalg.solve(H, g)
                beta = beta - step
            return beta

        # vmap transforms _solve_one(y_vec) into a function that accepts
        # a batch of y vectors stacked along axis 0 of Y_j.  Combined
        # with jit, the entire B-solve is compiled once and executed as
        # a single XLA dispatch.
        batched_solve = jit(vmap(_solve_one))
        all_coefs = np.asarray(batched_solve(Y_j))
        return all_coefs[:, 1:] if fit_intercept else all_coefs

    def fit_logistic_varying_X_jax(
        X_batch: np.ndarray,
        y: np.ndarray,
        max_iter: int = 100,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Fit logistic regression with varying X matrices using vmap.

        This is the **varying-X** variant used by the Kennedy individual
        logistic path.  In Kennedy's method, it is the *design matrix*
        that changes across permutations (column j is replaced with
        X̂_j + π(eₓⱼ)) while the response vector y stays fixed.  This
        is the inverse of the ter Braak case, where X is fixed and Y
        is permuted.

        Why a separate function?
        ~~~~~~~~~~~~~~~~~~~~~~~~
        ``fit_logistic_batch_jax`` captures X in the ``_solve_one``
        closure and vmaps over Y.  Here, we need to vmap over the
        first axis of a 3-D X tensor while keeping y constant.  The
        closure signature changes: ``_solve_one(X_single)`` receives
        a different (n, p) design matrix for each permutation, while
        ``_y`` is a default argument bound to the shared y vector.

        vmap semantics
        ~~~~~~~~~~~~~~
        ``vmap(_solve_one)`` maps over axis 0 of X_j (shape: (B, n, p)),
        giving each call a single (n, p) slice.  The default argument
        ``_y = y_j`` is not mapped — it is broadcast identically to
        all B calls.  This is JAX's standard pattern for vmapping over
        one argument while broadcasting another.

        Args:
            X_batch: Design matrices of shape ``(B, n, p)``.  Each
                ``X_batch[b]`` is the full design matrix with column j
                replaced by X̂_j + π_b(eₓⱼ).  Should NOT include an
                intercept column (one is prepended when *fit_intercept*
                is True).
            y: Response vector of shape ``(n,)`` — shared across all
                B solves (the outcome is not permuted in Kennedy's
                method).
            max_iter: Maximum Newton–Raphson iterations per solve.
            fit_intercept: Whether to prepend a column of ones to each
                design matrix.

        Returns:
            Coefficient matrix of shape ``(B, p)`` (intercept excluded
            when *fit_intercept* is True).
        """
        # Prepend intercept column to each of the B design matrices.
        # X_batch has shape (B, n, p); after augmentation it becomes
        # (B, n, p+1) with column 0 as the intercept for every slice.
        if fit_intercept:
            B, n, _ = X_batch.shape
            ones = np.ones((B, n, 1), dtype=X_batch.dtype)
            X_aug = np.concatenate([ones, X_batch], axis=2)
        else:
            X_aug = X_batch
        X_j = jnp.array(X_aug, dtype=jnp.float32)
        y_j = jnp.array(y, dtype=jnp.float32)

        def _solve_one(X_single: jax.Array, _y: jax.Array = y_j) -> jax.Array:
            """Newton–Raphson solve with a per-permutation design matrix.

            Same update rule as in ``fit_logistic_batch_jax._solve_one``:
                β_{t+1} = β_t − H⁻¹(β_t) · ∇NLL(β_t)

            The only difference is that X_single varies across the vmap
            axis (each permutation contributes a different design matrix)
            while _y is broadcast (shared response vector).
            """
            beta = jnp.zeros(X_single.shape[1], dtype=jnp.float32)
            for _ in range(max_iter):
                g = _logistic_grad(beta, X_single, _y)
                H = _logistic_hessian_diag(beta, X_single, _y)
                beta = beta - jnp.linalg.solve(H, g)
            return beta

        # vmap maps over axis 0 of X_j (the B-permutation axis),
        # calling _solve_one with each (n, p+1) slice independently.
        batched = jit(vmap(_solve_one))
        all_coefs_raw = np.asarray(batched(X_j))
        return all_coefs_raw[:, 1:] if fit_intercept else all_coefs_raw

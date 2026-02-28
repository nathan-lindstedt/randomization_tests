"""NumPy / scikit-learn backend (always available).

This is the fallback backend that requires no optional dependencies
beyond NumPy and scikit-learn, both of which are hard requirements
of the package.

Architecture
~~~~~~~~~~~~
Every ``batch_*`` method implements one cell of the strategy matrix:

    strategy dimension  ×  model family  →  batch solver

The three strategy dimensions are:

1. **Shared X, many Y** (``batch_*``): ter Braak path — the design
   matrix X is fixed and the response y is permuted B times.  OLS
   exploits a *single* pseudoinverse multiply ``pinv(X) @ Y.T``,
   replacing B separate least-squares solves with one ``(p, n) @
   (n, B)`` BLAS-3 matrix product.  GLM families (logistic, Poisson,
   NB, ordinal, multinomial) cannot vectorise this way because the
   iterative solver (Newton–Raphson / IRLS) restarts from scratch for
   each permuted y, so they fall back to a Python loop over B.

2. **Many X, shared y** (``batch_*_varying_X``): Kennedy individual
   path — each permutation replaces one column of X with permuted
   exposure residuals, yielding B distinct design matrices.  Even OLS
   requires B separate ``lstsq`` calls here because the pseudoinverse
   changes with every X.

3. **Both X and Y vary** (``batch_*_paired``): Bootstrap / jackknife
   path — each replicate resamples (or leaves-one-out) *rows*, so
   both X and y change simultaneously per replicate.

Parallelism
~~~~~~~~~~~
When ``n_jobs != 1``, loops are parallelised with
``joblib.Parallel(prefer="threads")``.  Thread-based parallelism
(rather than process-based) avoids data serialisation overhead
because the underlying solvers — sklearn's Cython/BLAS logistic
regression, NumPy's LAPACK ``dgelsd``, and statsmodels' IRLS — all
release the GIL during the heavy computation.  Multiple permutations
can therefore overlap on multi-core hardware with near-zero Python
overhead.

Warning suppression
~~~~~~~~~~~~~~~~~~~
Statsmodels emits ``ConvergenceWarning`` and ``RuntimeWarning`` when
IRLS fails to converge within ``maxiter`` iterations, which happens
routinely for degenerate permuted-y vectors (e.g. a Poisson response
where every permuted count lands on 0).  These warnings are suppressed
inside each ``_fit_one`` closure because:

* Non-converged permutations are **retained** in the null distribution
  — discarding them would bias the p-value anti-conservatively.
* The warnings would fire thousands of times (once per non-converged
  permutation), flooding stderr with no actionable information.
* The JAX backend handles the same situation silently (the
  ``lax.while_loop`` simply exits at ``max_iter``).

Failed fits return ``np.nan`` coefficients, which the p-value
calculation in ``pvalues.py`` handles correctly (NaN ≥ anything
is False, so NaN permutations never contribute to the empirical
p-value count).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning as SmConvergenceWarning,
)
from statsmodels.tools.sm_exceptions import (
    HessianInversionWarning,
)


@dataclass(frozen=True)
class NumpyBackend:
    """NumPy / scikit-learn compute backend.

    All methods accept plain NumPy arrays and return plain NumPy
    arrays.  No optional dependencies are required.

    The class is a frozen dataclass with no instance state — it
    exists solely to namespace the batch-fitting methods behind the
    :class:`BackendProtocol` interface.  Frozen = immutable = safe
    to cache in the module-level ``_BACKEND_CACHE`` singleton.
    """

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def is_available(self) -> bool:  # noqa: PLR6301
        return True

    # ================================================================ #
    # OLS — shared X, many Y
    # ================================================================ #
    #
    # OLS admits a closed-form batch solution via the pseudoinverse:
    #
    #   β̂ = (X'X)⁻¹ X'Y  =  pinv(X) @ Y
    #
    # For B permutations, Y is an (n, B) matrix and the entire batch
    # is computed in a single BLAS-3 ``dgemm`` call:
    #
    #   β̂_all = pinv(X) @ Y_matrix.T   →  shape (p+1, B)
    #
    # This is O(p²n + p·n·B) — the pseudoinverse is computed once
    # (O(p²n) via SVD) and the matmul is O(p·n·B).  Compare with B
    # separate ``lstsq`` calls at O(B·p²n).  For typical permutation
    # test sizes (B=5000, n=200, p=5), the batch approach is ~100×
    # faster.
    #
    # When ``fit_intercept=True``, a column of ones is prepended to X
    # before computing the pseudoinverse.  The intercept coefficient
    # (column 0 of the result) is then stripped before returning, so
    # the caller always receives shape ``(B, p)`` — slope coefficients
    # only.

    def batch_ols(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Batch OLS via pseudoinverse multiply.

        Computes ``pinv(X_aug) @ Y_matrix.T`` in a single BLAS call,
        replacing B separate ``lstsq`` solves with one ``(p, n) @ (n, B)``
        matrix product.

        Args:
            X: Design matrix ``(n, p)`` — no intercept column.
            Y_matrix: Permuted responses ``(B, n)``.
            fit_intercept: Prepend an intercept column; the intercept
                coefficient is stripped before returning.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        if fit_intercept:
            X_aug = np.column_stack([np.ones(X.shape[0]), X])  # (n, p+1)
            # Moore–Penrose pseudoinverse via SVD.  For full-rank X this
            # is equivalent to (X'X)⁻¹X', but gracefully handles rank-
            # deficient designs (collinear features) by zeroing singular
            # values below the default rcond threshold.
            pinv = np.linalg.pinv(X_aug)  # (p+1, n)
            # Single BLAS-3 matmul: all B permutations solved at once.
            result: np.ndarray = (pinv @ Y_matrix.T).T  # (B, p+1)
            return result[:, 1:]  # drop intercept column
        else:
            pinv = np.linalg.pinv(X)  # (p, n)
            result = (pinv @ Y_matrix.T).T  # (B, p)
            return result

    # ================================================================ #
    # Logistic regression — shared X, many Y
    # ================================================================ #
    #
    # Logistic regression has no closed-form solution — the MLE is
    # found via iterative optimisation of the log-likelihood:
    #
    #   ℓ(β) = Σᵢ [yᵢ log p(xᵢ) + (1−yᵢ) log(1−p(xᵢ))]
    #
    # where p(x) = σ(x'β) = 1/(1+exp(−x'β)).
    #
    # Unlike OLS, the Hessian H = X' diag(p(1−p)) X depends on the
    # current β, so the pseudoinverse trick does not apply.  Each
    # permuted y requires a fresh optimisation from scratch.
    #
    # We use sklearn's ``LogisticRegression`` with ``penalty=None``
    # (unpenalised MLE) and ``solver='lbfgs'`` (quasi-Newton).  L-BFGS
    # is a good default: it converges in O(10) iterations for
    # well-conditioned logistic problems and handles moderate
    # ill-conditioning (κ < 10,000) without explicit damping.
    #
    # ``max_iter=5000`` is deliberately generous — sklearn's default
    # of 100 frequently triggers convergence warnings on permuted data
    # where the class balance can be extreme.

    def batch_logistic(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch logistic regression via sklearn loop.

        Fits ``LogisticRegression`` once per permutation.  When
        ``n_jobs != 1`` the loop is parallelised with joblib so that
        sklearn's internal BLAS code (which releases the GIL) can
        overlap across cores.

        Args:
            X: Design matrix ``(n, p)``.
            Y_matrix: Permuted binary responses ``(B, n)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: ``n_jobs`` (default 1), plus solver options
                forwarded to ``LogisticRegression`` (e.g. ``max_iter``).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        n_jobs: int = kwargs.pop("n_jobs", 1)
        max_iter = kwargs.get("max_iter", 5_000)
        B, _ = Y_matrix.shape

        def _fit_one(y_b: np.ndarray) -> np.ndarray:
            # Unpenalised (penalty=None) MLE via L-BFGS.  Each permuted
            # binary response vector gets its own independent solve.
            m = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=max_iter,
                fit_intercept=fit_intercept,
            )
            m.fit(X, y_b)
            # sklearn returns coef_ as (1, p) for binary classification;
            # flatten to (p,) for stacking into the (B, p) result.
            return np.asarray(m.coef_.flatten())

        # Sequential path — used when n_jobs=1 (default) to avoid
        # joblib overhead for small B.
        if n_jobs == 1:
            p = X.shape[1]
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _fit_one(Y_matrix[b])
            return result

        # Parallel path — joblib with thread preference.  sklearn's
        # L-BFGS implementation calls BLAS routines that release the
        # GIL, so threads achieve near-linear speedup without the
        # serialisation cost of process-based parallelism.
        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(Y_matrix[b]) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ================================================================ #
    # Logistic regression — many X, shared y
    # ================================================================ #
    #
    # Kennedy individual path: for each permutation b, column j of X
    # is replaced with the permuted exposure residuals.  This gives
    # B distinct design matrices, each requiring its own logistic
    # solve.  The shared y vector is fixed throughout.

    def batch_logistic_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch logistic with per-permutation design matrices.

        Used by the Kennedy individual logistic path where column *j*
        of *X* is replaced with permuted exposure residuals in each
        permutation.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            y: Shared binary response ``(n,)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: ``n_jobs`` (default 1), plus solver options
                forwarded to ``LogisticRegression``.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        n_jobs: int = kwargs.pop("n_jobs", 1)
        max_iter = kwargs.get("max_iter", 5_000)
        B = X_batch.shape[0]

        def _fit_one(X_b: np.ndarray) -> np.ndarray:
            # Each permutation gets its own design matrix with the
            # j-th column replaced — fit an independent logistic MLE.
            m = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=max_iter,
                fit_intercept=fit_intercept,
            )
            m.fit(X_b, y)
            return np.asarray(m.coef_.flatten())

        if n_jobs == 1:
            p = X_batch.shape[2]
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _fit_one(X_batch[b])
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(X_batch[b]) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ================================================================ #
    # OLS — many X, shared y
    # ================================================================ #
    #
    # Kennedy individual linear path.  Each permutation has its own
    # design matrix (column j replaced with permuted exposure
    # residuals), so a separate ``lstsq`` solve is needed per
    # permutation — the single-pseudoinverse trick from batch_ols
    # does not apply because pinv(X) changes with every X.
    #
    # NumPy's ``lstsq`` delegates to LAPACK's ``dgelsd`` (divide-and-
    # conquer SVD), which releases the GIL.  This makes thread-based
    # joblib parallelism effective: each thread's LAPACK call runs
    # concurrently without Python lock contention.

    def batch_ols_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch OLS with per-permutation design matrices.

        Kennedy individual linear path — each permutation has its
        own design matrix (column *j* replaced with permuted exposure
        residuals), so a separate ``lstsq`` solve is needed per
        permutation.

        When ``n_jobs != 1``, the per-permutation ``lstsq`` calls are
        parallelised via joblib.  NumPy's ``lstsq`` calls LAPACK
        (``dgelsd``), which releases the GIL, so ``prefer="threads"``
        avoids serialisation overhead.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared continuous response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``n_jobs`` (default 1).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        n_jobs: int = kwargs.pop("n_jobs", 1)
        B, n, p = X_batch.shape

        if fit_intercept:
            # Pre-allocate the ones column once — reused across all B
            # solves to avoid B redundant allocations.
            ones_col = np.ones((n, 1))

            def _solve(X_b: np.ndarray) -> np.ndarray:
                X_aug = np.column_stack([ones_col, X_b])
                # lstsq via LAPACK dgelsd — releases the GIL.
                coefs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                return np.asarray(coefs[1:])  # drop intercept
        else:

            def _solve(X_b: np.ndarray) -> np.ndarray:
                coefs, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
                return np.asarray(coefs)

        if n_jobs == 1:
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _solve(X_batch[b])
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_solve)(X_batch[b]) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ================================================================ #
    # Poisson GLM — shared X, many Y
    # ================================================================ #
    #
    # Poisson regression models count data via the log link:
    #
    #   log(μ) = X β    ⟹    μ = exp(X β)
    #   y ~ Poisson(μ)
    #
    # The MLE is found via IRLS (Iteratively Reweighted Least Squares),
    # delegated to statsmodels' ``GLM(..., family=Poisson()).fit()``.
    # IRLS iterates the weighted normal equation:
    #
    #   β_{t+1} = (X' W_t X)⁻¹ X' W_t z_t
    #
    # where W_t = diag(μ_t) is the weight matrix and z_t is the
    # working response.  Unlike logistic regression, the Poisson
    # Hessian is always positive-definite (no separation issues), so
    # convergence is typically fast (5–10 iterations).
    #
    # Like the logistic path, each permuted y requires a fresh IRLS
    # solve — no pseudoinverse shortcut exists for GLMs.

    def batch_poisson(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson GLM via statsmodels IRLS loop.

        Args:
            X: Design matrix ``(n, p)``.
            Y_matrix: Permuted count responses ``(B, n)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: ``n_jobs`` (default 1).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        n_jobs: int = kwargs.pop("n_jobs", 1)
        # statsmodels expects the intercept column baked into X.
        # ``sm.add_constant`` prepends a column of ones.
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        n_params = X.shape[1]
        B = Y_matrix.shape[0]

        def _fit_one(y_b: np.ndarray) -> np.ndarray:
            try:
                with warnings.catch_warnings():
                    # Suppress IRLS convergence and runtime warnings
                    # for degenerate permuted-y vectors.  See module
                    # docstring "Warning suppression" for rationale.
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y_b, X_sm, family=sm.families.Poisson()).fit(
                        disp=0, maxiter=100
                    )
                params = np.asarray(model.params)
                # Strip the intercept coefficient — the caller expects
                # slope-only output of shape (p,).
                return params[1:] if fit_intercept else params
            except Exception:  # noqa: BLE001
                # Degenerate data (e.g. all-zero counts after permutation)
                # can cause statsmodels to raise.  Return NaN so the
                # permutation is retained but does not inflate the
                # empirical p-value count.
                return np.full(n_params, np.nan)

        if n_jobs == 1:
            result = np.empty((B, n_params))
            for b in range(B):
                result[b] = _fit_one(Y_matrix[b])
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(Y_matrix[b]) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ================================================================ #
    # Poisson GLM — many X, shared y
    # ================================================================ #
    #
    # Kennedy individual Poisson path — each permutation has its own
    # design matrix.  Same IRLS delegation to statsmodels, but X
    # varies per permutation while y is fixed.

    def batch_poisson_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson with per-permutation design matrices.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            y: Shared count response ``(n,)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: ``n_jobs`` (default 1).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        n_jobs: int = kwargs.pop("n_jobs", 1)
        B, _n, p = X_batch.shape

        def _fit_one(X_b: np.ndarray) -> np.ndarray:
            # Each permutation requires its own intercept-augmented
            # design matrix because X varies per permutation.
            X_sm = sm.add_constant(X_b) if fit_intercept else np.asarray(X_b)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y, X_sm, family=sm.families.Poisson()).fit(
                        disp=0, maxiter=100
                    )
                params = np.asarray(model.params)
                return params[1:] if fit_intercept else params
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan)

        if n_jobs == 1:
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _fit_one(X_batch[b])
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(X_batch[b]) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ================================================================ #
    # Negative Binomial (NB2) GLM — shared X, many Y
    # ================================================================ #
    #
    # The NB2 model generalises Poisson by adding an overdispersion
    # parameter α that allows Var(y) > E(y):
    #
    #   E(y) = μ = exp(Xβ)
    #   Var(y) = μ + α·μ²        (NB2 variance function)
    #
    # The key architectural decision is that α is estimated **once**
    # from the observed (unpermuted) data during ``NegativeBinomialFamily
    # .calibrate()`` and then **held fixed** across all B permutations.
    # This is correct because:
    #
    # * Under H₀ (β_j = 0), the marginal distribution of y does not
    #   change — only the conditional relationship y|X is broken by
    #   permutation.  The dispersion α is a marginal property, so it
    #   should not be re-estimated per permutation.
    # * Re-estimating α per permutation would be O(B × inner-loop)
    #   expensive and would introduce estimation noise into the null
    #   distribution.
    #
    # α is passed via ``kwargs["alpha"]`` by the family's
    # ``batch_fit()`` method.

    def batch_negbin(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB2 GLM via statsmodels IRLS loop with fixed α.

        Args:
            X: Design matrix ``(n, p)``.
            Y_matrix: Permuted count responses ``(B, n)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: ``alpha`` (required), ``n_jobs`` (default 1).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        alpha: float = kwargs.pop("alpha")
        n_jobs: int = kwargs.pop("n_jobs", 1)

        # Build the design matrix once — X is shared across all B
        # permutations so we can pre-augment with a constant column.
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        n_params = X.shape[1]
        B = Y_matrix.shape[0]

        # Construct the NB2 family object **once** with the calibrated α.
        # Reusing the same family object across all B iterations avoids
        # redundant object construction and ensures the held-fixed α is
        # applied consistently.
        nb_family = sm.families.NegativeBinomial(alpha=alpha)

        def _fit_one(y_b: np.ndarray) -> np.ndarray:
            try:
                with warnings.catch_warnings():
                    # Suppress IRLS convergence warnings — permuted
                    # responses can produce degenerate likelihoods.
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y_b, X_sm, family=nb_family).fit(disp=0, maxiter=100)
                params = np.asarray(model.params)
                # Strip the intercept term to return slope-only coefficients.
                return params[1:] if fit_intercept else params
            except Exception:  # noqa: BLE001
                # Graceful degradation — NaN signals a failed permutation
                # and is handled downstream by the p-value computation.
                return np.full(n_params, np.nan)

        # Sequential path — used when n_jobs == 1 to avoid joblib
        # overhead for small B or when thread-safety is uncertain.
        if n_jobs == 1:
            result = np.empty((B, n_params))
            for b in range(B):
                result[b] = _fit_one(Y_matrix[b])
            return result

        # Parallel path — statsmodels GLM releases the GIL during
        # LAPACK calls inside IRLS, so "threads" gives real speedup.
        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(Y_matrix[b]) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ---- Negative binomial (many X, shared y) -------------------------

    def batch_negbin_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB2 with per-permutation design matrices.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            y: Shared count response ``(n,)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: ``alpha`` (required), ``n_jobs`` (default 1).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        alpha: float = kwargs.pop("alpha")
        n_jobs: int = kwargs.pop("n_jobs", 1)
        B, _n, p = X_batch.shape
        nb_family = sm.families.NegativeBinomial(alpha=alpha)

        def _fit_one(X_b: np.ndarray) -> np.ndarray:
            X_sm = sm.add_constant(X_b) if fit_intercept else np.asarray(X_b)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y, X_sm, family=nb_family).fit(disp=0, maxiter=100)
                params = np.asarray(model.params)
                return params[1:] if fit_intercept else params
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan)

        if n_jobs == 1:
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _fit_one(X_batch[b])
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(X_batch[b]) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ================================================================ #
    # Ordinal (Proportional-Odds) Logistic Regression
    # ================================================================ #
    #
    # The ordered logistic model (cumulative logit / proportional-odds)
    # parameterises the cumulative probabilities of an ordinal outcome
    # y ∈ {0, 1, …, K-1} as:
    #
    #   logit P(y ≤ k | X) = α_k − Xβ    for k = 0, …, K-2
    #
    # where α₀ < α₁ < … < α_{K-2} are the **threshold** (cut-point)
    # parameters and β are the **slope** parameters shared across
    # all cumulative splits.
    #
    # Implementation notes:
    # * ``fit_intercept`` is accepted for protocol compatibility but
    #   is **ignored** — the thresholds themselves serve as intercepts.
    # * statsmodels' ``OrderedModel`` is used with ``distr='logit'``
    #   and the **Powell** optimizer (derivative-free) because the
    #   threshold-ordering constraint can produce non-smooth likelihood
    #   landscapes that trip gradient-based methods.
    # * ``model.params[:p]`` returns the slope coefficients (statsmodels
    #   places slopes first, thresholds last in the parameter vector).
    # ------------------------------------------------------------------ #

    # ---- Ordinal (shared X, many Y) ----------------------------------

    def batch_ordinal(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch ordinal via statsmodels OrderedModel loop.

        ``fit_intercept`` is accepted for protocol compatibility but
        ignored — thresholds always serve as intercepts.

        Args:
            X: Design matrix ``(n, p)``.
            Y_matrix: Permuted ordinal responses ``(B, n)``.
            fit_intercept: Accepted but ignored.
            **kwargs: ``n_jobs`` (default 1).

        Returns:
            Slope coefficients ``(B, p)`` (thresholds excluded).
        """
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        n_jobs: int = kwargs.pop("n_jobs", 1)
        kwargs.pop("K", None)  # Not needed for statsmodels path
        X_arr = np.asarray(X, dtype=float)
        n_params = X_arr.shape[1]
        B = Y_matrix.shape[0]

        def _fit_one(y_b: np.ndarray) -> np.ndarray:
            try:
                with warnings.catch_warnings():
                    # Hessian inversion warnings are common for ordinal
                    # models — the Powell optimizer does not use the
                    # Hessian, but statsmodels still attempts to compute
                    # it post-fit for standard errors.
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=HessianInversionWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = OrderedModel(y_b, X_arr, distr="logit").fit(
                        disp=0, method="powell", maxiter=200
                    )
                # Slopes first, thresholds last — extract only the slopes.
                return np.asarray(model.params[:n_params])
            except Exception:  # noqa: BLE001
                return np.full(n_params, np.nan)

        if n_jobs == 1:
            result = np.empty((B, n_params))
            for b in range(B):
                result[b] = _fit_one(Y_matrix[b])
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(Y_matrix[b]) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ---- Ordinal (many X, shared y) ----------------------------------

    def batch_ordinal_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch ordinal with per-permutation design matrices.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            y: Shared ordinal response ``(n,)``.
            fit_intercept: Accepted but ignored.
            **kwargs: ``n_jobs`` (default 1).

        Returns:
            Slope coefficients ``(B, p)`` (thresholds excluded).
        """
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        n_jobs: int = kwargs.pop("n_jobs", 1)
        kwargs.pop("K", None)  # Not needed for statsmodels path
        B, _n, p = X_batch.shape
        y_arr = np.asarray(y)

        def _fit_one(X_b: np.ndarray) -> np.ndarray:
            X_arr = np.asarray(X_b, dtype=float)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=HessianInversionWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = OrderedModel(y_arr, X_arr, distr="logit").fit(
                        disp=0, method="powell", maxiter=200
                    )
                return np.asarray(model.params[:p])
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan)

        if n_jobs == 1:
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _fit_one(X_batch[b])
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(X_batch[b]) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ================================================================ #
    # Multinomial Logistic Regression (MNLogit)
    # ================================================================ #
    #
    # For a nominal (unordered categorical) outcome y ∈ {0, 1, …, K-1},
    # multinomial logistic regression models the log-odds of each
    # non-reference category relative to category 0:
    #
    #   log P(y = k | X) / P(y = 0 | X) = X β_k    for k = 1, …, K-1
    #
    # Each predictor j now has K-1 coefficients (one per non-reference
    # category), so there is no single scalar "effect" per predictor.
    # The permutation test engine, however, requires a scalar test
    # statistic per predictor.
    #
    # **Solution — per-predictor Wald χ²:**
    #
    # For predictor j, collect the K-1 coefficients β_{j1}, …, β_{j,K-1}
    # and their (K-1 × K-1) covariance sub-block Σ_j, then compute:
    #
    #   χ²_j = β_j^T  Σ_j^{-1}  β_j
    #
    # This Wald statistic tests H₀: β_{j1} = … = β_{j,K-1} = 0
    # (i.e., predictor j has no effect on any category) and provides
    # the scalar test statistic the permutation engine needs.
    #
    # The helper ``_wald_chi2_from_mnlogit()`` (defined at module level)
    # handles the index arithmetic to extract the right sub-blocks from
    # statsmodels' covariance matrix layout.
    # ------------------------------------------------------------------ #

    # ---- Multinomial (shared X, many Y) --------------------------------

    def batch_multinomial(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch multinomial via statsmodels MNLogit loop.

        Returns per-predictor Wald χ² statistics (not raw
        coefficients), since the permutation engine requires a
        scalar test statistic per predictor.

        Args:
            X: Design matrix ``(n, p)``.
            Y_matrix: Permuted nominal responses ``(B, n)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``n_jobs`` (default 1).

        Returns:
            Wald χ² statistics ``(B, p)`` — one per slope
            predictor per permutation (intercept excluded).
        """
        from statsmodels.discrete.discrete_model import MNLogit

        n_jobs: int = kwargs.pop("n_jobs", 1)
        kwargs.pop("K", None)  # Not needed — MNLogit infers K

        if fit_intercept:
            X_aug = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_aug = np.asarray(X, dtype=float)

        p_aug = X_aug.shape[1]
        p_slopes = X.shape[1]  # slopes only (intercept excluded)
        B = Y_matrix.shape[0]

        def _fit_one(y_b: np.ndarray) -> np.ndarray:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=HessianInversionWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = MNLogit(y_b, X_aug).fit(disp=0, maxiter=200)
                return _wald_chi2_from_mnlogit(model, p_aug, fit_intercept)
            except Exception:  # noqa: BLE001
                return np.full(p_slopes, np.nan)

        if n_jobs == 1:
            result = np.empty((B, p_slopes))
            for b in range(B):
                result[b] = _fit_one(Y_matrix[b])
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(Y_matrix[b]) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ---- Multinomial (many X, shared y) --------------------------------

    def batch_multinomial_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch multinomial with per-permutation design matrices.

        Args:
            X_batch: Design matrices ``(B, n, p)``.
            y: Shared nominal response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``n_jobs`` (default 1).

        Returns:
            Wald χ² statistics ``(B, p)`` — one per slope
            predictor per permutation (intercept excluded).
        """
        from statsmodels.discrete.discrete_model import MNLogit

        n_jobs: int = kwargs.pop("n_jobs", 1)
        kwargs.pop("K", None)  # Not needed
        B, _n, p = X_batch.shape
        y_arr = np.asarray(y)

        def _fit_one(X_b: np.ndarray) -> np.ndarray:
            if fit_intercept:
                X_aug = np.column_stack([np.ones(X_b.shape[0]), X_b])
            else:
                X_aug = np.asarray(X_b, dtype=float)
            p_aug = X_aug.shape[1]
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=HessianInversionWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = MNLogit(y_arr, X_aug).fit(disp=0, maxiter=200)
                return _wald_chi2_from_mnlogit(model, p_aug, fit_intercept)
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan)

        if n_jobs == 1:
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _fit_one(X_batch[b])
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(X_batch[b]) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ================================================================ #
    # batch_fit_and_score — shared X, varying Y
    # ================================================================ #
    #
    # The ``fit_and_score`` methods return **both** the fitted slope
    # coefficients *and* a scalar goodness-of-fit score for each
    # permutation.  The score is used by the engine's
    # ``_improvement()`` logic to test whether the *full* model
    # (with the predictor of interest) fits significantly better than
    # the *reduced* model (without it).
    #
    # Score conventions (lower ≡ better fit):
    # * **OLS → RSS** (residual sum of squares):  The natural measure
    #   of unexplained variance.  Improvement = RSS_reduced - RSS_full.
    # * **Logistic / Poisson / NB2 → deviance** = 2 · NLL:  Deviance
    #   is twice the negative log-likelihood.  Additive constants that
    #   depend only on the data (not the parameters) cancel in the
    #   improvement calculation, so we do not need the saturated-model
    #   term.
    # * **Ordinal / Multinomial → −2·llf** (negative twice the
    #   maximised log-likelihood):  Equivalent to deviance up to a
    #   data-dependent constant that cancels.
    #
    # For all families, ``improvement = score_reduced − score_full``,
    # with a positive improvement indicating the full model fits
    # better.
    # ------------------------------------------------------------------ #

    def batch_ols_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch OLS returning ``(coefs, RSS)``."""
        if fit_intercept:
            X_aug = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_aug = np.asarray(X, dtype=float)

        # Pseudoinverse computed once — same matrix multiplication
        # strategy as batch_ols (see that method's comments).
        pinv = np.linalg.pinv(X_aug)
        all_coefs = (pinv @ Y_matrix.T).T  # (B, p_aug)

        # Residuals are computed in a single BLAS-3 matmul so that
        # RSS for all B permutations is obtained without a Python loop.
        predictions = X_aug @ all_coefs.T  # (n, B)
        residuals = Y_matrix.T - predictions  # (n, B)
        rss = np.sum(residuals**2, axis=0)  # (B,)

        coefs = all_coefs[:, 1:] if fit_intercept else all_coefs
        return coefs, rss

    def batch_logistic_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch logistic returning ``(coefs, deviance)``."""
        n_jobs: int = kwargs.pop("n_jobs", 1)
        max_iter = kwargs.get("max_iter", 5_000)
        B, _ = Y_matrix.shape
        p = X.shape[1]

        def _fit_one(y_b: np.ndarray) -> tuple[np.ndarray, float]:
            m = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=max_iter,
                fit_intercept=fit_intercept,
            )
            m.fit(X, y_b)
            coef = np.asarray(m.coef_.flatten())
            # Compute deviance = 2·NLL manually from predicted probas.
            # Clipping avoids log(0) when the model is overconfident.
            proba = np.clip(m.predict_proba(X)[:, 1], 1e-15, 1 - 1e-15)
            nll = -float(np.sum(y_b * np.log(proba) + (1 - y_b) * np.log(1 - proba)))
            return coef, 2.0 * nll  # deviance = 2·NLL

        coefs = np.empty((B, p))
        scores = np.empty(B)
        if n_jobs == 1:
            for b in range(B):
                coefs[b], scores[b] = _fit_one(Y_matrix[b])
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(Y_matrix[b]) for b in range(B)
            )
            for b, (c, s) in enumerate(results):
                coefs[b], scores[b] = c, s
        return coefs, scores

    def batch_poisson_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch Poisson returning ``(coefs, deviance)``."""
        n_jobs: int = kwargs.pop("n_jobs", 1)
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        n_params = X.shape[1]
        B = Y_matrix.shape[0]

        def _fit_one(y_b: np.ndarray) -> tuple[np.ndarray, float]:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y_b, X_sm, family=sm.families.Poisson()).fit(
                        disp=0, maxiter=100
                    )
                params = np.asarray(model.params)
                coef = params[1:] if fit_intercept else params
                return coef, float(model.deviance)
            except Exception:  # noqa: BLE001
                return np.full(n_params, np.nan), np.nan

        coefs = np.empty((B, n_params))
        scores = np.empty(B)
        if n_jobs == 1:
            for b in range(B):
                coefs[b], scores[b] = _fit_one(Y_matrix[b])
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(Y_matrix[b]) for b in range(B)
            )
            for b, (c, s) in enumerate(results):
                coefs[b], scores[b] = c, s
        return coefs, scores

    def batch_negbin_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch NB2 returning ``(coefs, deviance)``."""
        alpha: float = kwargs.pop("alpha")
        n_jobs: int = kwargs.pop("n_jobs", 1)
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        n_params = X.shape[1]
        B = Y_matrix.shape[0]
        nb_family = sm.families.NegativeBinomial(alpha=alpha)

        def _fit_one(y_b: np.ndarray) -> tuple[np.ndarray, float]:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y_b, X_sm, family=nb_family).fit(disp=0, maxiter=100)
                params = np.asarray(model.params)
                coef = params[1:] if fit_intercept else params
                return coef, float(model.deviance)
            except Exception:  # noqa: BLE001
                return np.full(n_params, np.nan), np.nan

        coefs = np.empty((B, n_params))
        scores = np.empty(B)
        if n_jobs == 1:
            for b in range(B):
                coefs[b], scores[b] = _fit_one(Y_matrix[b])
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(Y_matrix[b]) for b in range(B)
            )
            for b, (c, s) in enumerate(results):
                coefs[b], scores[b] = c, s
        return coefs, scores

    def batch_ordinal_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch ordinal returning ``(coefs, -2·llf)``."""
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        n_jobs: int = kwargs.pop("n_jobs", 1)
        kwargs.pop("K", None)
        X_arr = np.asarray(X, dtype=float)
        n_params = X_arr.shape[1]
        B = Y_matrix.shape[0]

        def _fit_one(y_b: np.ndarray) -> tuple[np.ndarray, float]:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=HessianInversionWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = OrderedModel(y_b, X_arr, distr="logit").fit(
                        disp=0, method="powell", maxiter=200
                    )
                coef = np.asarray(model.params[:n_params])
                return coef, -2.0 * float(model.llf)
            except Exception:  # noqa: BLE001
                return np.full(n_params, np.nan), np.nan

        coefs = np.empty((B, n_params))
        scores = np.empty(B)
        if n_jobs == 1:
            for b in range(B):
                coefs[b], scores[b] = _fit_one(Y_matrix[b])
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(Y_matrix[b]) for b in range(B)
            )
            for b, (c, s) in enumerate(results):
                coefs[b], scores[b] = c, s
        return coefs, scores

    def batch_multinomial_fit_and_score(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch multinomial returning ``(wald_chi2, -2·llf)``."""
        from statsmodels.discrete.discrete_model import MNLogit

        n_jobs: int = kwargs.pop("n_jobs", 1)
        kwargs.pop("K", None)

        if fit_intercept:
            X_aug = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_aug = np.asarray(X, dtype=float)

        p_aug = X_aug.shape[1]
        p_slopes = X.shape[1]
        B = Y_matrix.shape[0]

        def _fit_one(y_b: np.ndarray) -> tuple[np.ndarray, float]:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=HessianInversionWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = MNLogit(y_b, X_aug).fit(disp=0, maxiter=200)
                wald = _wald_chi2_from_mnlogit(model, p_aug, fit_intercept)
                return wald, -2.0 * float(model.llf)
            except Exception:  # noqa: BLE001
                return np.full(p_slopes, np.nan), np.nan

        coefs = np.empty((B, p_slopes))
        scores = np.empty(B)
        if n_jobs == 1:
            for b in range(B):
                coefs[b], scores[b] = _fit_one(Y_matrix[b])
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(Y_matrix[b]) for b in range(B)
            )
            for b, (c, s) in enumerate(results):
                coefs[b], scores[b] = c, s
        return coefs, scores

    # ================================================================ #
    # batch_fit_and_score_varying_X — varying X, shared Y
    # ================================================================ #
    #
    # Mirror of the ``fit_and_score`` group above, but for the Kennedy
    # strategy where it is the *design matrix* that changes per
    # permutation while the response y remains fixed.
    #
    # Score conventions are identical — RSS for OLS, deviance for GLMs,
    # −2·llf for ordinal and multinomial.
    # ------------------------------------------------------------------ #

    def batch_ols_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch OLS (varying X) returning ``(coefs, RSS)``."""
        n_jobs: int = kwargs.pop("n_jobs", 1)
        B, n, p = X_batch.shape

        if fit_intercept:
            ones_col = np.ones((n, 1))

            def _solve(X_b: np.ndarray) -> tuple[np.ndarray, float]:
                X_aug = np.column_stack([ones_col, X_b])
                coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                resid = y - X_aug @ coef
                return np.asarray(coef[1:]), float(np.sum(resid**2))
        else:

            def _solve(X_b: np.ndarray) -> tuple[np.ndarray, float]:
                coef, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
                resid = y - X_b @ coef
                return np.asarray(coef), float(np.sum(resid**2))

        coefs = np.empty((B, p))
        scores = np.empty(B)
        if n_jobs == 1:
            for b in range(B):
                coefs[b], scores[b] = _solve(X_batch[b])
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_solve)(X_batch[b]) for b in range(B)
            )
            for b, (c, s) in enumerate(results):
                coefs[b], scores[b] = c, s
        return coefs, scores

    def batch_logistic_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch logistic (varying X) returning ``(coefs, deviance)``."""
        n_jobs: int = kwargs.pop("n_jobs", 1)
        max_iter = kwargs.get("max_iter", 5_000)
        B, _n, p = X_batch.shape

        def _fit_one(X_b: np.ndarray) -> tuple[np.ndarray, float]:
            m = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=max_iter,
                fit_intercept=fit_intercept,
            )
            m.fit(X_b, y)
            coef = np.asarray(m.coef_.flatten())
            proba = np.clip(m.predict_proba(X_b)[:, 1], 1e-15, 1 - 1e-15)
            nll = -float(np.sum(y * np.log(proba) + (1 - y) * np.log(1 - proba)))
            return coef, 2.0 * nll

        coefs = np.empty((B, p))
        scores = np.empty(B)
        if n_jobs == 1:
            for b in range(B):
                coefs[b], scores[b] = _fit_one(X_batch[b])
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(X_batch[b]) for b in range(B)
            )
            for b, (c, s) in enumerate(results):
                coefs[b], scores[b] = c, s
        return coefs, scores

    def batch_poisson_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch Poisson (varying X) returning ``(coefs, deviance)``."""
        n_jobs: int = kwargs.pop("n_jobs", 1)
        B, _n, p = X_batch.shape

        def _fit_one(X_b: np.ndarray) -> tuple[np.ndarray, float]:
            X_sm = sm.add_constant(X_b) if fit_intercept else np.asarray(X_b)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y, X_sm, family=sm.families.Poisson()).fit(
                        disp=0, maxiter=100
                    )
                params = np.asarray(model.params)
                coef = params[1:] if fit_intercept else params
                return coef, float(model.deviance)
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan), np.nan

        coefs = np.empty((B, p))
        scores = np.empty(B)
        if n_jobs == 1:
            for b in range(B):
                coefs[b], scores[b] = _fit_one(X_batch[b])
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(X_batch[b]) for b in range(B)
            )
            for b, (c, s) in enumerate(results):
                coefs[b], scores[b] = c, s
        return coefs, scores

    def batch_negbin_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch NB2 (varying X) returning ``(coefs, deviance)``."""
        alpha: float = kwargs.pop("alpha")
        n_jobs: int = kwargs.pop("n_jobs", 1)
        B, _n, p = X_batch.shape
        nb_family = sm.families.NegativeBinomial(alpha=alpha)

        def _fit_one(X_b: np.ndarray) -> tuple[np.ndarray, float]:
            X_sm = sm.add_constant(X_b) if fit_intercept else np.asarray(X_b)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y, X_sm, family=nb_family).fit(disp=0, maxiter=100)
                params = np.asarray(model.params)
                coef = params[1:] if fit_intercept else params
                return coef, float(model.deviance)
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan), np.nan

        coefs = np.empty((B, p))
        scores = np.empty(B)
        if n_jobs == 1:
            for b in range(B):
                coefs[b], scores[b] = _fit_one(X_batch[b])
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(X_batch[b]) for b in range(B)
            )
            for b, (c, s) in enumerate(results):
                coefs[b], scores[b] = c, s
        return coefs, scores

    def batch_ordinal_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch ordinal (varying X) returning ``(coefs, -2·llf)``."""
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        n_jobs: int = kwargs.pop("n_jobs", 1)
        kwargs.pop("K", None)
        B, _n, p = X_batch.shape
        y_arr = np.asarray(y)

        def _fit_one(X_b: np.ndarray) -> tuple[np.ndarray, float]:
            X_arr = np.asarray(X_b, dtype=float)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=HessianInversionWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = OrderedModel(y_arr, X_arr, distr="logit").fit(
                        disp=0, method="powell", maxiter=200
                    )
                coef = np.asarray(model.params[:p])
                return coef, -2.0 * float(model.llf)
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan), np.nan

        coefs = np.empty((B, p))
        scores = np.empty(B)
        if n_jobs == 1:
            for b in range(B):
                coefs[b], scores[b] = _fit_one(X_batch[b])
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(X_batch[b]) for b in range(B)
            )
            for b, (c, s) in enumerate(results):
                coefs[b], scores[b] = c, s
        return coefs, scores

    def batch_multinomial_fit_and_score_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch multinomial (varying X) returning ``(wald_chi2, -2·llf)``."""
        from statsmodels.discrete.discrete_model import MNLogit

        n_jobs: int = kwargs.pop("n_jobs", 1)
        kwargs.pop("K", None)
        B, _n, p = X_batch.shape
        y_arr = np.asarray(y)

        def _fit_one(X_b: np.ndarray) -> tuple[np.ndarray, float]:
            if fit_intercept:
                X_aug = np.column_stack([np.ones(X_b.shape[0]), X_b])
            else:
                X_aug = np.asarray(X_b, dtype=float)
            p_aug = X_aug.shape[1]
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=HessianInversionWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = MNLogit(y_arr, X_aug).fit(disp=0, maxiter=200)
                wald = _wald_chi2_from_mnlogit(model, p_aug, fit_intercept)
                return wald, -2.0 * float(model.llf)
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan), np.nan

        coefs = np.empty((B, p))
        scores = np.empty(B)
        if n_jobs == 1:
            for b in range(B):
                coefs[b], scores[b] = _fit_one(X_batch[b])
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one)(X_batch[b]) for b in range(B)
            )
            for b, (c, s) in enumerate(results):
                coefs[b], scores[b] = c, s
        return coefs, scores

    # ================================================================ #
    # batch_*_paired — both X and Y vary per replicate
    # ================================================================ #
    #
    # These methods support **bootstrap** and **jackknife** loops where
    # each replicate resamples (or leaves-one-out) *rows*, so both the
    # design matrix X and the response y change simultaneously.
    #
    # Shape convention:
    #   X_batch : (B, n, p) — B design matrices, no intercept column
    #   Y_batch : (B, n)    — B response vectors
    #
    # Unlike the permutation methods, paired methods return only slope
    # coefficients (no scores), because bootstrap / jackknife
    # resampling is used for **confidence-interval** construction
    # rather than hypothesis testing.  The engine collects the B
    # coefficient vectors and derives percentile or BCa intervals
    # from the empirical distribution.
    #
    # Parallelism follows the same ``n_jobs`` / ``prefer='threads'``
    # pattern as the permutation methods above.
    # ------------------------------------------------------------------ #

    def batch_ols_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool = True,
    ) -> np.ndarray:
        """Batch OLS where both X and Y vary per replicate.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            Y_batch: Response vectors ``(B, n)``.
            fit_intercept: Prepend intercept column.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        B, n, p = X_batch.shape
        result = np.empty((B, p))
        for b in range(B):
            X_b = X_batch[b]
            if fit_intercept:
                X_aug = np.column_stack([np.ones(n), X_b])
                coefs, _, _, _ = np.linalg.lstsq(X_aug, Y_batch[b], rcond=None)
                result[b] = coefs[1:]
            else:
                coefs, _, _, _ = np.linalg.lstsq(X_b, Y_batch[b], rcond=None)
                result[b] = coefs
        return result

    def batch_logistic_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch logistic where both X and Y vary per replicate.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            Y_batch: Binary response vectors ``(B, n)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: ``n_jobs`` (default 1), ``max_iter``.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        n_jobs: int = kwargs.pop("n_jobs", 1)
        max_iter = kwargs.get("max_iter", 5_000)
        B, _n, p = X_batch.shape

        def _fit_one(b: int) -> np.ndarray:
            try:
                m = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=max_iter,
                    fit_intercept=fit_intercept,
                )
                m.fit(X_batch[b], Y_batch[b])
                return np.asarray(m.coef_.flatten())
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan)

        if n_jobs == 1:
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _fit_one(b)
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(b) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    def batch_poisson_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch Poisson where both X and Y vary per replicate.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            Y_batch: Count response vectors ``(B, n)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: ``n_jobs`` (default 1).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        n_jobs: int = kwargs.pop("n_jobs", 1)
        B, _n, p = X_batch.shape

        def _fit_one(b: int) -> np.ndarray:
            X_sm = (
                sm.add_constant(X_batch[b]) if fit_intercept else np.asarray(X_batch[b])
            )
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(Y_batch[b], X_sm, family=sm.families.Poisson()).fit(
                        disp=0, maxiter=100
                    )
                params = np.asarray(model.params)
                return params[1:] if fit_intercept else params
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan)

        if n_jobs == 1:
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _fit_one(b)
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(b) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    def batch_negbin_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch NB2 where both X and Y vary per replicate.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            Y_batch: Count response vectors ``(B, n)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: ``alpha`` (required), ``n_jobs`` (default 1).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        alpha: float = kwargs.pop("alpha")
        n_jobs: int = kwargs.pop("n_jobs", 1)
        B, _n, p = X_batch.shape
        nb_family = sm.families.NegativeBinomial(alpha=alpha)

        def _fit_one(b: int) -> np.ndarray:
            X_sm = (
                sm.add_constant(X_batch[b]) if fit_intercept else np.asarray(X_batch[b])
            )
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(Y_batch[b], X_sm, family=nb_family).fit(
                        disp=0, maxiter=100
                    )
                params = np.asarray(model.params)
                return params[1:] if fit_intercept else params
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan)

        if n_jobs == 1:
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _fit_one(b)
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(b) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    def batch_ordinal_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch ordinal where both X and Y vary per replicate.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            Y_batch: Ordinal response vectors ``(B, n)``.
            fit_intercept: Accepted but ignored.
            **kwargs: ``n_jobs`` (default 1).

        Returns:
            Slope coefficients ``(B, p)`` (thresholds excluded).
        """
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        n_jobs: int = kwargs.pop("n_jobs", 1)
        kwargs.pop("K", None)
        B, _n, p = X_batch.shape

        def _fit_one(b: int) -> np.ndarray:
            X_arr = np.asarray(X_batch[b], dtype=float)
            y_arr = np.asarray(Y_batch[b])
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=HessianInversionWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = OrderedModel(y_arr, X_arr, distr="logit").fit(
                        disp=0, method="powell", maxiter=200
                    )
                return np.asarray(model.params[:p])
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan)

        if n_jobs == 1:
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _fit_one(b)
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(b) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    def batch_multinomial_paired(
        self,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch multinomial where both X and Y vary per replicate.

        Returns per-predictor Wald χ² (matching
        ``MultinomialFamily.coefs()`` semantics).

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            Y_batch: Nominal response vectors ``(B, n)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: ``n_jobs`` (default 1).

        Returns:
            Wald χ² statistics ``(B, p)`` — one per slope predictor.
        """
        from statsmodels.discrete.discrete_model import MNLogit

        n_jobs: int = kwargs.pop("n_jobs", 1)
        kwargs.pop("K", None)
        B, _n, p = X_batch.shape

        def _fit_one(b: int) -> np.ndarray:
            X_b = X_batch[b]
            if fit_intercept:
                X_aug = np.column_stack([np.ones(X_b.shape[0]), X_b])
            else:
                X_aug = np.asarray(X_b, dtype=float)
            p_aug = X_aug.shape[1]
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=HessianInversionWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = MNLogit(Y_batch[b], X_aug).fit(disp=0, maxiter=200)
                return _wald_chi2_from_mnlogit(model, p_aug, fit_intercept)
            except Exception:  # noqa: BLE001
                return np.full(p, np.nan)

        if n_jobs == 1:
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _fit_one(b)
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(b) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ---- Linear mixed model (shared X, many Y) ----------------------

    def batch_mixed_lm(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch LMM via pre-computed GLS projection.

        Uses the same ``A @ Y.T`` matmul as the JAX path — the
        projection matrix ``A`` is computed during REML calibration
        (by the JAX backend or statsmodels fallback) and passed via
        ``kwargs``.

        Args:
            X: Design matrix ``(n, p)`` — accepted for protocol
                compatibility but not used directly.
            Y_matrix: Permuted responses ``(B, n)``.
            fit_intercept: Whether the projection includes an
                intercept row.
            **kwargs: ``projection_A`` (required) — ``(p, n)``
                GLS projection from REML calibration.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        projection_A: np.ndarray = kwargs["projection_A"]
        all_coefs: np.ndarray = np.asarray(Y_matrix @ projection_A.T)  # (B, p)
        coefs = all_coefs[:, 1:] if fit_intercept else all_coefs
        return np.asarray(coefs)

    # ---- Linear mixed model (many X, shared y) ----------------------

    def batch_mixed_lm_varying_X(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Batch LMM with per-permutation design matrices.

        Kennedy individual linear mixed path.  Variance components
        (encoded in ``C22``) are fixed from calibration; only the
        GLS projection changes per permutation.

        Pre-computes ``C₂₂⁻¹ Z'`` once, then loops over B design
        matrices rebuilding ``A_b`` and extracting ``β̂_b = A_b y``.

        Args:
            X_batch: Design matrices ``(B, n, p)`` — no intercept.
            y: Shared continuous response ``(n,)``.
            fit_intercept: Prepend intercept column.
            **kwargs: ``Z`` (required), ``C22`` (required),
                ``n_jobs`` (default 1).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        Z: np.ndarray = kwargs["Z"]
        C22: np.ndarray = kwargs["C22"]
        n_jobs: int = kwargs.get("n_jobs", 1)

        B, n, p = X_batch.shape

        # Pre-compute invariant: C₂₂⁻¹ Z'  (q, n)
        C22_inv_Zt = np.linalg.solve(C22, Z.T)

        def _solve_one(b: int) -> np.ndarray:
            X_b = X_batch[b]
            if fit_intercept:
                X_aug = np.column_stack([np.ones(n), X_b])
            else:
                X_aug = np.asarray(X_b, dtype=float)
            XtZ = X_aug.T @ Z  # (p_aug, q)
            C22_inv_ZtX = C22_inv_Zt @ X_aug  # (q, p_aug)
            S = X_aug.T @ X_aug - XtZ @ C22_inv_ZtX  # (p_aug, p_aug)
            Xt_Vtilde_inv = X_aug.T - XtZ @ C22_inv_Zt  # (p_aug, n)
            A = np.linalg.solve(S, Xt_Vtilde_inv)  # (p_aug, n)
            coefs: np.ndarray = np.asarray(A @ y)  # (p_aug,)
            return coefs[1:] if fit_intercept else coefs

        if n_jobs == 1:
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _solve_one(b)
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_solve_one)(b) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))


# ------------------------------------------------------------------ #
# Multinomial Wald χ² helper
# ------------------------------------------------------------------ #
#
# This module-level helper is used by all three multinomial methods
# (batch_multinomial, batch_multinomial_fit_and_score,
# batch_multinomial_paired) to extract a per-predictor scalar test
# statistic from a fitted statsmodels MNLogit model.
#
# The core challenge is that statsmodels stores multinomial parameters
# in a (p_aug × K-1) matrix (predictors × non-reference categories)
# and the full covariance matrix in a ((K-1)·p_aug × (K-1)·p_aug)
# block-diagonal-ish layout ordered by **equation first** (category),
# then parameter within equation.  The index arithmetic below maps
# from predictor index j to the correct rows/columns in the
# covariance matrix.
# ------------------------------------------------------------------ #


def _wald_chi2_from_mnlogit(
    model: Any,
    p_aug: int,
    has_intercept: bool,
) -> np.ndarray:
    """Extract per-predictor Wald χ² from a fitted MNLogit model.

    Statsmodels MNLogit stores ``model.params`` as ``(p_aug, K-1)``
    and ``model.cov_params()`` as ``((K-1)*p_aug, (K-1)*p_aug)``
    (ordered by equation first, then parameter within equation).

    For each slope predictor *j*, the K-1 coefficients and their
    covariance sub-block are extracted and the Wald statistic is
    computed as:

        χ²_j = β_j^T [Var(β_j)]^{-1} β_j

    Args:
        model: Fitted ``MNLogitResults``.
        p_aug: Number of augmented predictors (with intercept).
        has_intercept: Whether column 0 is an intercept.

    Returns:
        Wald χ² statistics ``(p_slopes,)`` — one per slope.
    """
    params = np.asarray(model.params)  # (p_aug, K-1)
    cov = np.asarray(model.cov_params())  # ((K-1)*p_aug, (K-1)*p_aug)
    Km1 = params.shape[1]  # K − 1 — non-reference categories
    start = 1 if has_intercept else 0  # skip intercept column
    p_slopes = p_aug - start  # number of slope predictors

    wald = np.empty(p_slopes)  # output buffer for χ²_j values
    for j_slope in range(p_slopes):
        j = j_slope + start  # column index in the augmented design matrix

        # Extract the K-1 coefficients for predictor j — one from
        # each non-reference category equation.
        beta_j = params[j, :]  # (K-1,)

        # ---- Covariance sub-block extraction ----
        # statsmodels orders the full covariance matrix by *equation
        # first*, then parameter within equation.  So equation k
        # occupies rows/cols [k·p_aug : (k+1)·p_aug], and the entry
        # for parameter j within equation k is at linear index
        # k·p_aug + j.  We collect these K-1 indices and fancy-index
        # the covariance matrix to get the (K-1 × K-1) sub-block.
        idx = np.array([k * p_aug + j for k in range(Km1)])
        cov_j = cov[np.ix_(idx, idx)]  # (K-1, K-1)

        # Wald χ²_j = β_j^T  Σ_j^{-1}  β_j
        # Use np.linalg.solve instead of explicit inversion for
        # numerical stability.  Falls back to NaN if the sub-block
        # is singular (can happen with quasi-separated categories).
        try:
            wald[j_slope] = float(beta_j @ np.linalg.solve(cov_j, beta_j))
        except np.linalg.LinAlgError:
            wald[j_slope] = np.nan
    return wald

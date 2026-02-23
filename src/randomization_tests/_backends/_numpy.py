"""NumPy / scikit-learn backend (always available).

This is the fallback backend that requires no optional dependencies
beyond NumPy and scikit-learn, both of which are hard requirements
of the package.

OLS uses a single pseudoinverse–multiply (``np.linalg.pinv``), which
is the same vectorised approach that ``core.py`` used in v0.2.0.

Logistic paths use a scikit-learn ``LogisticRegression`` loop — one
``fit()`` call per permutation.  When ``n_jobs != 1``, the loop is
parallelised with ``joblib.Parallel(prefer="threads")`` so that
sklearn's internal Cython/BLAS code releases the GIL and multiple
permutations can overlap on multi-core hardware.

Poisson, negative binomial, and ordinal paths use statsmodels loops.
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
    """

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def is_available(self) -> bool:  # noqa: PLR6301
        return True

    # ---- OLS -----------------------------------------------------------

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
            pinv = np.linalg.pinv(X_aug)  # (p+1, n)
            result: np.ndarray = (pinv @ Y_matrix.T).T  # (B, p+1)
            return result[:, 1:]  # drop intercept
        else:
            pinv = np.linalg.pinv(X)  # (p, n)
            result = (pinv @ Y_matrix.T).T  # (B, p)
            return result

    # ---- Logistic (shared X, many Y) ----------------------------------

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
            m = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=max_iter,
                fit_intercept=fit_intercept,
            )
            m.fit(X, y_b)
            return np.asarray(m.coef_.flatten())

        if n_jobs == 1:
            p = X.shape[1]
            result = np.empty((B, p))
            for b in range(B):
                result[b] = _fit_one(Y_matrix[b])
            return result

        coefs_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(Y_matrix[b]) for b in range(B)
        )
        return np.asarray(np.vstack(coefs_list))

    # ---- Logistic (many X, shared y) -----------------------------------

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

    # ---- OLS (many X, shared y) ----------------------------------------

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
            ones_col = np.ones((n, 1))

            def _solve(X_b: np.ndarray) -> np.ndarray:
                X_aug = np.column_stack([ones_col, X_b])
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

    # ---- Poisson (shared X, many Y) ----------------------------------

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
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        n_params = X.shape[1]
        B = Y_matrix.shape[0]

        def _fit_one(y_b: np.ndarray) -> np.ndarray:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y_b, X_sm, family=sm.families.Poisson()).fit(
                        disp=0, maxiter=100
                    )
                params = np.asarray(model.params)
                return params[1:] if fit_intercept else params
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

    # ---- Poisson (many X, shared y) ----------------------------------

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

    # ---- Negative binomial (shared X, many Y) -------------------------

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
        X_sm = sm.add_constant(X) if fit_intercept else np.asarray(X)
        n_params = X.shape[1]
        B = Y_matrix.shape[0]
        nb_family = sm.families.NegativeBinomial(alpha=alpha)

        def _fit_one(y_b: np.ndarray) -> np.ndarray:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = sm.GLM(y_b, X_sm, family=nb_family).fit(disp=0, maxiter=100)
                params = np.asarray(model.params)
                return params[1:] if fit_intercept else params
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
                    warnings.filterwarnings("ignore", category=SmConvergenceWarning)
                    warnings.filterwarnings("ignore", category=HessianInversionWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = OrderedModel(y_b, X_arr, distr="logit").fit(
                        disp=0, method="powell", maxiter=200
                    )
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

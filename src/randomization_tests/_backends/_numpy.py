"""NumPy / scikit-learn backend (always available).

This is the fallback backend that requires no optional dependencies
beyond NumPy and scikit-learn, both of which are hard requirements
of the package.

OLS uses a single pseudoinverse–multiply (``np.linalg.pinv``), which
is the same vectorised approach that ``core.py`` used in v0.2.0.

Logistic paths use a scikit-learn ``LogisticRegression`` loop — one
``fit()`` call per permutation.  This is slower than the JAX vmap'd
Newton–Raphson solver but is correct and always available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


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

        Fits ``LogisticRegression`` once per permutation.  Slower than
        the JAX vmap path but always available and numerically robust.

        Args:
            X: Design matrix ``(n, p)``.
            Y_matrix: Permuted binary responses ``(B, n)``.
            fit_intercept: Whether to include an intercept.
            **kwargs: Forwarded to ``LogisticRegression`` (e.g.
                ``max_iter``).

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        max_iter = kwargs.get("max_iter", 5_000)
        B, _ = Y_matrix.shape
        p = X.shape[1]
        result = np.empty((B, p))

        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=max_iter,
            fit_intercept=fit_intercept,
        )
        for b in range(B):
            model.fit(X, Y_matrix[b])
            result[b] = model.coef_.flatten()

        return result

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
            **kwargs: Forwarded to ``LogisticRegression``.

        Returns:
            Slope coefficients ``(B, p)`` (intercept excluded).
        """
        max_iter = kwargs.get("max_iter", 5_000)
        B = X_batch.shape[0]
        p = X_batch.shape[2]
        result = np.empty((B, p))

        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=max_iter,
            fit_intercept=fit_intercept,
        )
        for b in range(B):
            model.fit(X_batch[b], y)
            result[b] = model.coef_.flatten()

        return result

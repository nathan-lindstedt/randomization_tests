"""ter Braak (1992) — residual permutation under the reduced model.

For each feature X_j, fit the model *without* X_j to get predicted
values ŷ₋ⱼ and residuals e₋ⱼ = Y − ŷ₋ⱼ.  Permute the residuals and
add them back to ŷ₋ⱼ to form Y* = ŷ₋ⱼ + π(e₋ⱼ).  Refit the full
model on (X, Y*) to get β*_j.

Two code paths:

* **Residual path** — standard residual-permutation for families with
  well-defined residuals (linear, logistic, Poisson, negative binomial).
* **Direct Y permutation (Manly 1997)** — for families where residuals
  are ill-defined (e.g. ordinal, multinomial).  Controlled by
  ``family.direct_permutation``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..families import fit_reduced

if TYPE_CHECKING:
    from ..families import ModelFamily


class TerBraakStrategy:
    """Residual-permutation strategy (ter Braak 1992 / Manly 1997).

    Individual test: returns ``np.ndarray`` of shape ``(B, n_features)``.
    """

    is_joint: bool = False

    def execute(
        self,
        X: pd.DataFrame,
        y_values: np.ndarray,
        family: ModelFamily,
        perm_indices: np.ndarray,
        *,
        confounders: list[str] | None = None,
        model_coefs: np.ndarray | None = None,
        fit_intercept: bool = True,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Run the ter Braak permutation algorithm.

        Args:
            X: Feature matrix as a pandas DataFrame.
            y_values: Response vector of shape ``(n,)``.
            family: Resolved ``ModelFamily`` instance.
            perm_indices: Pre-generated permutation indices ``(B, n)``.
            confounders: Unused (ter Braak does not partition
                confounders).
            model_coefs: Unused.
            fit_intercept: Whether to include an intercept.
            n_jobs: Parallelism level for the batch-fit step.

        Returns:
            Array of shape ``(B, n_features)`` with permuted
            coefficients.
        """
        X_np = X.values.astype(float)
        n_perm, n = perm_indices.shape
        n_features = X_np.shape[1]

        # --- Direct Y permutation path (Manly 1997) ---
        # For families where residuals are not well-defined (e.g.
        # ordinal), permute Y directly rather than going through the
        # residual pipeline.
        if family.direct_permutation:
            Y_perm = y_values[perm_indices]  # (B, n)
            return family.batch_fit(X_np, Y_perm, fit_intercept, n_jobs=n_jobs)

        # --- Standard residual-permutation path ---
        result = np.zeros((n_perm, n_features))

        # Derive a deterministic RNG from the permutation indices so
        # that any stochastic reconstruction step (e.g. Bernoulli
        # sampling for logistic) is reproducible given the same
        # permutations.
        rng = np.random.default_rng(int(perm_indices[0, 0]))

        for j in range(n_features):
            # Step 1: Fit the reduced model Y ~ X_{-j}.
            X_red = np.delete(X_np, j, axis=1)
            reduced_model, preds_red = fit_reduced(
                family, X_red, y_values, fit_intercept
            )

            # Residuals from the reduced model.
            if reduced_model is not None:
                resids_red = family.residuals(reduced_model, X_red, y_values)
            else:
                # Zero-column edge case: raw residuals from
                # intercept-only predictions.
                resids_red = y_values - preds_red

            # Step 2: Build all B permuted residual vectors.
            permuted_resids = resids_red[perm_indices]  # (B, n)

            # Step 3: Reconstruct permuted response vectors.
            Y_perm = family.reconstruct_y(
                preds_red[np.newaxis, :],
                permuted_resids,
                rng,
            )  # (B, n)

            # Step 4: Batch-refit the full model on all B permuted Y
            # vectors.
            all_coefs = family.batch_fit(
                X_np, Y_perm, fit_intercept, n_jobs=n_jobs
            )  # (B, p)
            result[:, j] = all_coefs[:, j]

        return result

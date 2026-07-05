"""ter Braak (1992) — full-model residual permutation, recentred null.

Canonical algorithm (ter Braak 1992; Anderson & Legendre 1999, method
“t-B”; Winkler et al. 2014, Table 2), for each H₀(j): β_j = 0:

1.  Fit the **full model** Y ~ X once.  Keep the fitted values ŷ and
    residuals e = Y − ŷ.

2.  **Permute** the residuals: π(e).

3.  **Reconstruct** a synthetic response about the full-model fit:
    Y* = ŷ + π(e).  The full-model signal (including X_j's true
    contribution) is retained.

4.  **Refit** the full model on (X, Y*) and **recentre**: the null
    draws are β*_j − β̂_j, because under the permutation distribution
    β*_j varies around the observed β̂_j, not around 0.  Recentring
    makes the draws a zero-centred null for β̂_j − β_j — which is what
    the package's ``|β*| ≥ |β̂|`` p-value count and shift-inversion
    CIs require.

One full-model fit and one batch refit serve ALL features
simultaneously — no per-feature reduced fits (contrast with
Freedman–Lane, which fits a reduced model per tested coefficient).

Guarantee: residuals are only approximately exchangeable under H₀
(they are correlated through the hat matrix), so the test is
**asymptotically exact**, not finite-sample exact (Anderson &
Robinson 2001).

Two code paths:

* **Residual path** — standard residual-permutation for families with
  well-defined residuals (linear, logistic, Poisson, negative binomial).
* **Direct Y permutation families** are rejected with a ``ValueError``
  directing the user to ``method="manly"`` or ``method="kennedy"``.

References:
    ter Braak, C. J. F. (1992). Permutation versus bootstrap
    significance tests in multiple regression and ANOVA.
    In K.-H. Jöckel et al. (Eds.), *Bootstrapping and Related
    Techniques* (pp. 79–86). Springer.

    Anderson, M. J. & Legendre, P. (1999). An empirical comparison of
    permutation methods for tests of partial regression coefficients
    in a linear model. *J. Statistical Computation and Simulation*,
    62(3), 271–303.

    Anderson, M. J. & Robinson, J. (2001). Permutation tests for
    linear models. *Australian & New Zealand J. Statistics*, 43(1),
    75–88.

    Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., &
    Nichols, T. E. (2014). Permutation inference for the general
    linear model. *NeuroImage*, 92, 381–397.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import _apply_randomization

if TYPE_CHECKING:
    from ..families import ModelFamily


class TerBraakStrategy:
    """Full-model residual-permutation strategy (ter Braak 1992).

    Individual test: returns ``np.ndarray`` of shape ``(B, n_features)``
    holding the recentred null draws β*_j − β̂_j.
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
        randomization: str = "permute",
    ) -> np.ndarray:
        """Run the ter Braak permutation algorithm.

        Args:
            X: Feature matrix as a pandas DataFrame.
            y_values: Response vector of shape ``(n,)``.
            family: Resolved ``ModelFamily`` instance.
            perm_indices: Pre-generated permutation indices ``(B, n)``.
            confounders: Unused (ter Braak does not partition
                confounders — every column is tested against its own
                observed coefficient).
            model_coefs: Observed coefficients ``(p,)`` — used to
                recentre the permuted coefficients (β* − β̂).  When
                ``None``, recomputed from the in-strategy full fit.
            fit_intercept: Whether to include an intercept.
            n_jobs: Parallelism level for the batch-fit step.
            randomization: ``"permute"`` (default) or ``"sign_flip"``.

        Returns:
            Array of shape ``(B, n_features)`` with recentred permuted
            coefficients β*_j − β̂_j (zero-centred null draws).
        """
        X_np = X.values.astype(float)  # (n, p) full design matrix

        # --- Guard: reject direct-permutation families -------------
        # Families like ordinal and multinomial have no well-defined
        # residuals, so the ter Braak residual-permutation pipeline
        # cannot run.  Direct the user to Manly (direct Y permutation)
        # or Kennedy (exposure-model residuals).
        if family.direct_permutation:
            raise ValueError(
                f"ter Braak method requires well-defined residuals but "
                f"family='{family.name}' uses direct Y permutation.  "
                f"Use method='manly' (direct Y permutation, Manly 1997) "
                f"or method='kennedy' (exposure-model residuals, "
                f"Kennedy 1995) instead."
            )

        # --- Canonical ter Braak (1992) path ------------------------
        # Derive a deterministic RNG from the first row of perm_indices
        # so that any stochastic reconstruction step (e.g. Bernoulli
        # sampling for logistic residuals → binary Y) is reproducible
        # given the same permutation/sign-flip matrix.  Shifting by
        # perm_indices.shape[1] converts sign-flip values (±1) to
        # positive integers while keeping permutation indices unique.
        rng = np.random.default_rng(
            (perm_indices[0].astype(np.int64) + perm_indices.shape[1]).astype(np.uint64)
        )

        # Step 1: Fit the FULL model Y ~ X once.  ONE fit serves all
        # features — there are no per-feature reduced models.
        full_model = family.fit(X_np, y_values, fit_intercept)
        preds_full = family.predict(full_model, X_np)  # ŷ, shape (n,)
        # Full-model residuals e = Y − ŷ (family-appropriate type).
        resids_full = family.residuals(full_model, X_np, y_values)  # (n,)

        # Step 2: Resample the residual vector.
        # For permutation: fancy-indexing broadcasts the 1-D residual
        # array into B shuffled copies.  For sign-flip: element-wise
        # ±1 multiplication.
        permuted_resids = _apply_randomization(
            resids_full, perm_indices, randomization
        )  # (B, n)

        # Step 3: Reconstruct Y* = ŷ + π(e) about the FULL-model fit.
        # The observed signal (every predictor's contribution) is
        # retained; only the error arrangement is randomized.
        # For logistic, reconstruct_y clamps to [0,1] and draws
        # Bernoulli(p*); for Poisson/NB it applies the link inverse.
        Y_perm = family.reconstruct_y(
            preds_full[np.newaxis, :],  # (1, n) → broadcast to (B, n)
            permuted_resids,  # (B, n)
            rng,
        )  # (B, n) synthetic response vectors

        # Step 4: Batch-refit the full model on all B synthetic
        # responses — a single vectorised call for ALL features.
        all_coefs = np.array(
            family.batch_fit(X_np, Y_perm, fit_intercept, n_jobs=n_jobs)
        )  # (B, p)

        # Step 5: Recentre.  Under the ter Braak construction β*_j
        # varies about the observed β̂_j, so the null draws for
        # β̂_j − β_j are β*_j − β̂_j.  Recentring yields the
        # zero-centred null required by the |β*| ≥ |β̂| p-value count
        # and the shift-inversion confidence intervals.
        if model_coefs is not None:
            beta_hat = np.asarray(model_coefs, dtype=float)
        else:
            beta_hat = np.asarray(family.coefs(full_model), dtype=float)

        return np.asarray(all_coefs - beta_hat[np.newaxis, :])

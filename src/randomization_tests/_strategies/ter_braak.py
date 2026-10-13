r"""ter Braak (1992) full-model residual permutation with recentred null.

Canonical algorithm (ter Braak 1992; Anderson & Legendre 1999; Winkler et al. 2014)
for partial regression testing (Tier 4 of the 7-tier inference abstraction tower).
A single full-model fit and one batch resampling refit serve all predictors
simultaneously, avoiding per-feature reduced model fits.

Primary Components
------------------
* :class:`TerBraakStrategy`: Individual per-feature regression test (``is_joint=False``).

Mathematical Foundations & Recentring Invariant
-----------------------------------------------
For each tested feature $j$:
1. Fit the **full model** $Y \sim X$ once to obtain fitted values $\hat y$ and residuals $e = Y - \hat y$.
2. **Permute** or sign-flip residuals: $\pi(e)$.
3. **Reconstruct** synthetic outcomes retaining full signal: $Y^* = \hat y + \pi(e)$.
4. **Refit** the full model on $(X, Y^*)$ and **recenter**:

.. math::
    \tilde\beta^*_{b, j} = \beta^*_{b, j} - \hat\beta_j

Because $Y^*$ retains the true effect of $X_j$, raw permutation draws $\beta^*$
vary around $\hat\beta_j$ rather than zero. Subtracting $\hat\beta_j$ recentres
the null at zero, satisfying the mathematical invariance required for two-sided
$|\beta^*| \ge |\hat\beta|$ p-value counts and shift-inversion confidence intervals.

[ Guarantee: Asymptotically Exact ]
Residuals are correlated through the full-model hat matrix $H = X(X^\top X)^{-1}X^\top$,
providing asymptotic size control under exchangeable errors (Anderson & Robinson 2001).

Cross-Module Integration
------------------------
* Inbound: Instantiated by :class:`~.engine.PermutationEngine` when `method="ter_braak"`.
* Outbound: Dispatches batch solves to :meth:`~._backends.BackendProtocol.batch_ols` or
  GLM batch methods via :meth:`~.families.ModelFamily.batch_fit`.

References
----------
ter Braak, C. J. F. (1992). Permutation versus bootstrap significance tests in
    multiple regression and ANOVA. In *Bootstrapping and Related Techniques* (pp. 79–86).

Anderson, M. J., & Legendre, P. (1999). An empirical comparison of permutation
    methods for tests of partial regression coefficients in a linear model.
    *Journal of Statistical Computation and Simulation*, 62(3), 271–303.

Anderson, M. J., & Robinson, J. (2001). Permutation tests for linear models.
    *Australian & New Zealand Journal of Statistics*, 43(1), 75–88.

Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014).
    Permutation inference for the general linear model. *NeuroImage*, 92, 381–397.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

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
        **kwargs: Any,
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
        # Step 1: Fit the FULL model Y ~ X once.  ONE fit serves all
        # features — there are no per-feature reduced models.
        full_model = family.fit(X_np, y_values, fit_intercept)
        preds_full = family.predict(full_model, X_np)  # ŷ, shape (n,)
        # Full-model residuals e = Y − ŷ (family-appropriate type).
        resids_full = family.residuals(full_model, X_np, y_values)  # (n,)

        # Steps 2-4: whiten (if applicable), permute, reconstruct, refit —
        # owned by the family so it cannot diverge from method="score",
        # which computes the same estimator through the same seam.
        all_coefs, _ = family.residual_permutation_refit(
            X_np,
            preds_full,
            resids_full,
            perm_indices,
            fit_intercept=fit_intercept,
            randomization=randomization,
            n_jobs=n_jobs,
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

        return np.asarray(np.asarray(all_coefs) - beta_hat[np.newaxis, :])

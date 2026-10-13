r"""Manly direct response permutation testing for complete null marginal independence.

The Manly (1997) method permutes the observed outcome vector $Y$ directly across all
units and refits the regression model on $(X, \pi(Y))$. This tests the complete null
hypothesis of **marginal** independence between $X$ and $Y$ ($Y \perp X$),
evaluating whether predictors have any association with the response without partialling
out shared variance among covariates.

Primary Components
------------------
* :class:`ManlyStrategy`: Per-feature marginal association permutation test.
* :class:`ManlyJointStrategy`: Group-level omnibus marginal fit improvement test.

Mathematical Foundations & Inferential Guarantees
-------------------------------------------------
Direct outcome permutation tests the complete independence null:

.. math::
    H_0: Y \perp X

The synthetic response is formed without residualization:

.. math::
    Y^* = \pi(Y)

[ Guarantee: Finite-Sample Exact ]
Direct $Y$ permutation is finite-sample exact for the complete null hypothesis
when observations are exchangeable under $H_0$ (no residual estimation, no hat matrix
projection, no asymptotic approximation).

Sign-flipping is **not supported** because raw response values cannot be sign-flipped
without altering the marginal mean structure.

Role in the Architecture & Advisory
-----------------------------------
* **Baseline for Non-Residual Families**: For discrete categorical families
  (:class:`~.families.OrdinalFamily`, :class:`~.families.MultinomialFamily`),
  unambiguous continuous residuals do not exist. Manly serves as the exact
  nonparametric baseline where residual-based methods (ter Braak, Freedman–Lane)
  cannot be applied.
* **Substantive Advisory on Continuous Models**: When applied to models with
  correlated predictors, permuting $Y$ destroys covariate structure, leaking
  correlation from non-tested predictors into the test statistic. The package
  permits Manly for all families but emits a clear advisory (⚠¹) clarifying
  that the resulting p-values evaluate marginal rather than partial conditional effects.

Cross-Module Integration
------------------------
* Inbound: Instantiated by :func:`~._strategies.resolve_strategy` when
  ``method="manly"`` or ``"manly_joint"``.
* Outbound: Dispatches batch refits to :meth:`~._backends.BackendProtocol.batch_ols`
  or GLM batch solvers.

References
----------
Manly, B. F. J. (1997). *Randomization, Bootstrap and Monte Carlo Methods in Biology*
    (2nd ed.). Chapman & Hall.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..families import fit_reduced

if TYPE_CHECKING:
    from ..families import ModelFamily


# ------------------------------------------------------------------ #
# Manly individual
# ------------------------------------------------------------------ #


class ManlyStrategy:
    """Manly (1997) direct Y permutation — per-coefficient test."""

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
        """Run the Manly direct Y permutation algorithm.

        Permutes the response vector Y directly and refits the full
        model on (X, Y*).  No residuals are computed.

        Args:
            X: Feature matrix as a pandas DataFrame.
            y_values: Response vector of shape ``(n,)``.
            family: Resolved ``ModelFamily`` instance.
            perm_indices: Pre-generated permutation indices ``(B, n)``.
            confounders: Unused (Manly does not partition confounders).
            model_coefs: Unused.
            fit_intercept: Whether to include an intercept.
            n_jobs: Parallelism level for the batch-fit step.
            randomization: Must be ``"permute"``.  Sign-flipping is not
                supported (no residuals to flip).

        Returns:
            Array of shape ``(B, n_features)`` with permuted
            coefficients.

        Raises:
            ValueError: If ``randomization="sign_flip"`` — Manly has
                no residuals to flip.
        """
        if randomization != "permute":
            raise ValueError(
                f"Manly method does not support randomization='{randomization}'.  "
                f"Direct Y permutation has no residuals to sign-flip.  "
                f"Use a residual-based method (ter_braak, freedman_lane, "
                f"kennedy, score) for sign-flip tests."
            )

        X_np = X.values.astype(float)  # (n, p)

        # Permute Y directly: fancy-index with (B, n) permutation
        # matrix to produce B shuffled response vectors.
        Y_perm = y_values[perm_indices]  # (B, n)

        # Batch-refit the full model on all B permuted responses.
        return family.batch_fit(X_np, Y_perm, fit_intercept, n_jobs=n_jobs)


# ------------------------------------------------------------------ #
# Manly joint
# ------------------------------------------------------------------ #


class ManlyJointStrategy:
    """Manly (1997) direct Y permutation — collective improvement test.

    Tests whether all non-confounder features collectively improve
    model fit beyond confounders alone.  The test statistic is:

        Δ = M(Y*, ŷ_Z*) − M(Y*, ŷ_full*)

    where M is the family's fit metric (RSS, deviance, etc.) and
    both models are refit on the permuted Y*.
    """

    is_joint: bool = True

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
    ) -> tuple[float, np.ndarray, str, list[str]]:
        """Run the Manly joint permutation algorithm.

        Args:
            X: Feature matrix as a pandas DataFrame.
            y_values: Response vector of shape ``(n,)``.
            family: Resolved ``ModelFamily`` instance.
            perm_indices: Pre-generated permutation indices ``(B, n)``.
            confounders: Confounder column names.
            model_coefs: Unused.
            fit_intercept: Whether to include an intercept.
            n_jobs: Parallelism level for the batch-fit step.
            randomization: Must be ``"permute"``.

        Returns:
            ``(obs_improvement, perm_improvements, metric_type,
            features_tested)`` tuple.

        Raises:
            ValueError: If ``randomization="sign_flip"``.
        """
        if randomization != "permute":
            raise ValueError(
                f"Manly method does not support randomization='{randomization}'.  "
                f"Direct Y permutation has no residuals to sign-flip.  "
                f"Use a residual-based method for sign-flip tests."
            )

        if confounders is None:
            confounders = []

        features_to_test = [c for c in X.columns if c not in confounders]
        metric_type = family.metric_label

        X_np = X.values.astype(float)  # (n, p)

        # Z = confounder design matrix.
        if confounders:
            conf_idx = [X.columns.get_loc(c) for c in confounders]
            Z = X_np[:, conf_idx]  # (n, q_z)
        else:
            Z = np.zeros((X_np.shape[0], 0))  # (n, 0)

        # --- Observed improvement ---
        reduced_model, _ = fit_reduced(family, Z, y_values, fit_intercept)
        if reduced_model is not None:
            base_metric = family.score(reduced_model, Z, y_values)
        else:
            base_metric = family.null_score(y_values, fit_intercept)

        full_model = family.fit(X_np, y_values, fit_intercept)
        obs_improvement = base_metric - family.score(full_model, X_np, y_values)

        # --- Permutation loop (vectorised via batch backend) ---
        # Permute Y directly: (B, n).
        Y_perm = y_values[perm_indices]

        # Batch-fit reduced model on permuted Y.
        _, reduced_scores = family.batch_fit_and_score(
            Z, Y_perm, fit_intercept, n_jobs=n_jobs
        )

        # Batch-fit full model on permuted Y.
        _, full_scores = family.batch_fit_and_score(
            X_np, Y_perm, fit_intercept, n_jobs=n_jobs
        )

        perm_improvements = reduced_scores - full_scores

        return (
            obs_improvement,
            perm_improvements,
            metric_type,
            features_to_test,
        )

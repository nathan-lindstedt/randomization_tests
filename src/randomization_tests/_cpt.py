r"""Conditional Permutation Test (CPT) label sampler (Berrett et al. 2020).

Implements the pairwise-swap MCMC sampler of Berrett, Wang, Barber &
Samworth (2020, JRSS-B §5) for sampling categorical label vectors
conditionally on continuous or multi-dimensional covariates X:

    P(pi | X) \propto \prod_{i=1}^n \hat{P}(E_{pi(i)} | X_i)

Shared infrastructure for:
- Invariance testing (Phase 7, Tier 2 CPT)
- Local Markov condition testing (Phase 11, Markov rigor ladder Tier 2)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def cpt_sample(
    labels: np.ndarray,
    X: np.ndarray,
    n_draws: int = 5000,
    random_state: int | None = None,
    model: str = "multinomial",
    n_steps_factor: int = 5,
) -> np.ndarray:
    """Sample permuted label vectors conditionally on covariates X via MCMC.

    Fits a conditional model P_hat(E | X) and uses a pairwise-swap Metropolis–
    Hastings MCMC chain to draw label permutations approximately distributed
    according to the conditional law.

    Every resampled draw is guaranteed to preserve the exact marginal class
    frequencies of the original labels.

    Args:
        labels: 1D array of categorical labels of length ``n``.
        X: Covariate matrix of shape ``(n, p)``.
        n_draws: Number of independent CPT draws to generate.
        random_state: Optional seed for reproducible MCMC sampling.
        model: Type of conditional model (``"multinomial"``).
        n_steps_factor: Multiplier for MCMC burn-in and transition steps
            (total swap proposals per draw = ``n_steps_factor * n``).

    Returns:
        Array of shape ``(n_draws, n)`` with resampled categorical labels.

    References:
        - Berrett, T. B., Wang, Y., Barber, R. F., & Samworth, R. J. (2020).
          The conditional permutation test for independence while controlling
          for confounders. Journal of the Royal Statistical Society: Series B,
          82(1), 175–197.
    """
    labels_arr = np.asarray(labels)
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    n = len(labels_arr)
    if len(X_arr) != n:
        msg = f"Length mismatch: labels has {n} items but X has {len(X_arr)} rows."
        raise ValueError(msg)

    if n < 4:
        msg = f"CPT requires at least 4 observations, got {n}."
        raise ValueError(msg)

    rng = np.random.default_rng(random_state)

    # Encode labels to discrete integer indices [0, ..., K-1]
    unique_classes, inverse_indices = np.unique(labels_arr, return_inverse=True)
    n_classes = len(unique_classes)

    if n_classes < 2:
        # Trivial case: all labels identical, permutation is identity
        return np.tile(labels_arr, (n_draws, 1))

    # Step 1: Fit conditional model P_hat(E | X)
    # Use l2 regularization with mild penalty C=1.0 to avoid divergence under separation
    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    clf.fit(X_arr, inverse_indices)
    # P_mat[i, k] is P_hat(E_i = k | X_i), shape (n, n_classes)
    P_mat = np.clip(clf.predict_proba(X_arr), 1e-12, 1.0)

    # Step 2: MCMC Pairwise-Swap Sampler (BWBS 2020 §5)
    # State is current permutation pi, initialized to identity
    pi = np.arange(n, dtype=np.intp)
    draws = np.empty((n_draws, n), dtype=labels_arr.dtype)

    steps_between_draws = max(10, int(n_steps_factor * n))

    for d in range(n_draws):
        # Propose steps_between_draws pairwise swaps
        for _ in range(steps_between_draws):
            # Pick two distinct observations (i, j) uniformly at random
            i, j = rng.choice(n, size=2, replace=False)

            cls_i = inverse_indices[pi[i]]
            cls_j = inverse_indices[pi[j]]

            if cls_i == cls_j:
                continue

            # Metropolis-Hastings acceptance ratio:
            # alpha = min(1, [P(E_j | X_i) * P(E_i | X_j)] / [P(E_i | X_i) * P(E_j | X_j)])
            prob_cur = P_mat[i, cls_i] * P_mat[j, cls_j]
            prob_prop = P_mat[i, cls_j] * P_mat[j, cls_i]

            acc = prob_prop / prob_cur
            if acc >= 1.0 or rng.uniform() < acc:
                pi[i], pi[j] = pi[j], pi[i]

        # Record resampled labels for draw d
        draws[d] = labels_arr[pi]

    return draws

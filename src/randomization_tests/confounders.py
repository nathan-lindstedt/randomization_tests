"""Confounder identification via correlation screening and mediation analysis.

Workflow:
    1. :func:`screen_potential_confounders` – find variables correlated
       with both the predictor X and the outcome Y (Pearson *r*).
    2. :func:`mediation_analysis` – Baron & Kenny (1986) decomposition
       with bootstrap CIs to test whether each candidate is a mediator
       (X → M → Y) rather than a confounder.
    3. :func:`identify_confounders` – orchestrates both steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


# ------------------------------------------------------------------ #
# Step 1 – Correlation screening
# ------------------------------------------------------------------ #

def screen_potential_confounders(
    X: pd.DataFrame,
    y: pd.DataFrame,
    predictor: str,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
) -> dict:
    """Screen for variables correlated with both predictor and outcome.

    A potential confounder *Z* satisfies ``|r(Z, X)| >= threshold``
    **and** ``|r(Z, Y)| >= threshold``, both with ``p < p_value_threshold``.

    Args:
        X: Feature matrix.
        y: Target variable.
        predictor: Name of the predictor of interest.
        correlation_threshold: Minimum absolute Pearson *r* to flag a
            variable.
        p_value_threshold: Maximum p-value for significance.

    Returns:
        Dictionary with keys ``predictor``, ``potential_confounders``,
        ``correlations_with_predictor``, ``correlations_with_outcome``,
        and ``excluded_variables``.
    """
    y_values = np.ravel(y)
    other_features = [c for c in X.columns if c != predictor]
    predictor_values = X[predictor].values

    potential_confounders: list[str] = []
    correlations_with_predictor: dict = {}
    correlations_with_outcome: dict = {}
    excluded_variables: list[str] = []

    for feature in other_features:
        fv = X[feature].values
        corr_pred, p_pred = stats.pearsonr(fv, predictor_values)
        corr_out, p_out = stats.pearsonr(fv, y_values)

        sig_pred = (abs(corr_pred) >= correlation_threshold) and (p_pred < p_value_threshold)
        sig_out = (abs(corr_out) >= correlation_threshold) and (p_out < p_value_threshold)

        if sig_pred and sig_out:
            potential_confounders.append(feature)
            correlations_with_predictor[feature] = {"r": corr_pred, "p": p_pred}
            correlations_with_outcome[feature] = {"r": corr_out, "p": p_out}
        else:
            excluded_variables.append(feature)

    return {
        "predictor": predictor,
        "potential_confounders": potential_confounders,
        "correlations_with_predictor": correlations_with_predictor,
        "correlations_with_outcome": correlations_with_outcome,
        "excluded_variables": excluded_variables,
    }


# ------------------------------------------------------------------ #
# Step 2 – Mediation analysis (Baron & Kenny + bootstrap)
# ------------------------------------------------------------------ #

def mediation_analysis(
    X: pd.DataFrame,
    y: pd.DataFrame,
    predictor: str,
    mediator: str,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    precision: int = 4,
    random_state: int | None = None,
) -> dict:
    """Baron & Kenny (1986) mediation analysis with bootstrap CIs.

    Decomposes the total effect of *predictor* on *y* into a direct
    effect (c\u2032) and an indirect effect through *mediator* (a \u00d7 b).
    Uses percentile bootstrap CIs (Preacher & Hayes, 2004) for the
    indirect effect, which avoids the normality assumption of the
    Sobel test.

    The bootstrap is **vectorised**: all *n_bootstrap* index arrays are
    pre-generated and the a-path / b-path regressions are batched via
    :func:`numpy.linalg.lstsq`.

    Args:
        X: Feature matrix.
        y: Target variable.
        predictor: Predictor (X in X \u2192 M \u2192 Y).
        mediator: Potential mediator (M).
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence-interval level.
        precision: Decimal places for rounding.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary containing the mediation decomposition, bootstrap CI,
        and a textual interpretation.
    """
    y_values = np.ravel(y)
    x_vals = X[predictor].values.reshape(-1, 1)
    m_vals = X[mediator].values.reshape(-1, 1)
    n = len(y_values)

    # --- Observed paths ---
    # Total effect: Y = c * X + e1
    model_total = LinearRegression().fit(x_vals, y_values)
    c_total = model_total.coef_[0]

    # a path: M = a * X + e2
    model_a = LinearRegression().fit(x_vals, m_vals.ravel())
    a_path = model_a.coef_[0]

    # b path + direct: Y = c' * X + b * M + e3
    xm = np.hstack([x_vals, m_vals])
    model_full = LinearRegression().fit(xm, y_values)
    c_prime = model_full.coef_[0]
    b_path = model_full.coef_[1]

    indirect_effect = a_path * b_path

    # --- Vectorised bootstrap for a*b ---
    rng = np.random.default_rng(random_state)
    # Pre-generate all bootstrap index arrays: (n_bootstrap, n)
    boot_idx = rng.choice(n, size=(n_bootstrap, n), replace=True)

    # Build design matrices with intercept column for lstsq
    ones = np.ones((n, 1))
    X_a_design = np.hstack([ones, x_vals])          # (n, 2)
    X_full_design = np.hstack([ones, x_vals, m_vals])  # (n, 3)

    bootstrap_indirect = np.empty(n_bootstrap)

    for b_i in range(n_bootstrap):
        idx = boot_idx[b_i]
        x_b = X_a_design[idx]       # (n, 2)
        m_b = m_vals[idx].ravel()    # (n,)
        y_b = y_values[idx]          # (n,)

        # a path via normal equation
        a_coef, _, _, _ = np.linalg.lstsq(x_b, m_b, rcond=None)
        a_boot = a_coef[1]  # slope (index 0 is intercept)

        xm_b = np.hstack([x_b, m_vals[idx]])  # (n, 3) — reuse x_b which has intercept
        b_coef, _, _, _ = np.linalg.lstsq(xm_b, y_b, rcond=None)
        b_boot = b_coef[2]  # mediator slope

        bootstrap_indirect[b_i] = a_boot * b_boot

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_indirect, (alpha / 2) * 100))
    ci_upper = float(np.percentile(bootstrap_indirect, (1 - alpha / 2) * 100))

    is_mediator = (ci_lower > 0) or (ci_upper < 0)

    if abs(c_total) > 1e-10:
        proportion_mediated = indirect_effect / c_total
    else:
        proportion_mediated = np.nan

    if is_mediator:
        if abs(c_prime) < abs(c_total) * 0.1:
            interpretation = (
                f"'{mediator}' is a full mediator. It explains most of the "
                f"effect of '{predictor}' on the outcome. Do not control for "
                f"it as a confounder."
            )
        else:
            interpretation = (
                f"'{mediator}' is a partial mediator. It explains some of the "
                f"effect of '{predictor}' on the outcome. Consider whether to "
                f"control for it based on research question."
            )
    else:
        interpretation = (
            f"'{mediator}' is not a significant mediator. It may be a "
            f"confounder if correlated with both '{predictor}' and outcome. "
            f"Consider controlling for it."
        )

    return {
        "predictor": predictor,
        "mediator": mediator,
        "total_effect": np.round(c_total, precision),
        "direct_effect": np.round(c_prime, precision),
        "indirect_effect": np.round(indirect_effect, precision),
        "a_path": np.round(a_path, precision),
        "b_path": np.round(b_path, precision),
        "indirect_effect_ci": (np.round(ci_lower, precision), np.round(ci_upper, precision)),
        "proportion_mediated": (
            np.round(proportion_mediated, precision)
            if not np.isnan(proportion_mediated)
            else np.nan
        ),
        "is_mediator": is_mediator,
        "interpretation": interpretation,
    }


# ------------------------------------------------------------------ #
# Step 3 – Orchestrator
# ------------------------------------------------------------------ #

def identify_confounders(
    X: pd.DataFrame,
    y: pd.DataFrame,
    predictor: str,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> dict:
    """Two-step confounder identification.

    1. Screen for variables correlated with both predictor and outcome.
    2. Use mediation analysis to filter out mediators.

    Variables that pass screening but are **not** identified as
    mediators are likely confounders.

    Args:
        X: Feature matrix.
        y: Target variable.
        predictor: Predictor of interest.
        correlation_threshold: Minimum absolute Pearson *r*.
        p_value_threshold: Significance cutoff.
        n_bootstrap: Bootstrap iterations for mediation analysis.
        confidence_level: Confidence-interval level.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with keys ``identified_confounders``,
        ``identified_mediators``, ``screening_results``,
        ``mediation_results``, and ``recommendation``.
    """
    screening = screen_potential_confounders(
        X, y, predictor,
        correlation_threshold=correlation_threshold,
        p_value_threshold=p_value_threshold,
    )
    candidates = screening["potential_confounders"]

    identified_confounders: list[str] = []
    identified_mediators: list[str] = []
    mediation_results: dict = {}

    for candidate in candidates:
        med = mediation_analysis(
            X, y, predictor, candidate,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        )
        mediation_results[candidate] = med
        if med["is_mediator"]:
            identified_mediators.append(candidate)
        else:
            identified_confounders.append(candidate)

    if identified_confounders:
        confounder_str = ", ".join(f"'{c}'" for c in identified_confounders)
        recommendation = (
            f"For Kennedy method with predictor '{predictor}', "
            f"control for: {confounder_str}"
        )
    else:
        recommendation = (
            f"No confounders identified for predictor '{predictor}'. "
            f"Consider using ter Braak method instead."
        )

    return {
        "predictor": predictor,
        "identified_confounders": identified_confounders,
        "identified_mediators": identified_mediators,
        "screening_results": screening,
        "mediation_results": mediation_results,
        "recommendation": recommendation,
    }

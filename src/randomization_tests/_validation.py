"""Structured validation utilities for the graph compiler and core pipeline.

All validators return ``list[ValidationIssue]`` rather than raising, so that
the graph compiler (Phase 9) can collect and report all issues at once rather
than failing on the first problem.

``core.py`` and ``engine.py`` continue to raise ``ValueError`` /
``UserWarning`` as before.  The delegation of those hot-path checks to these
validators is deferred to Phase 9 (Step 26), at the same time
``validate_compatibility()`` is first consumed by the graph compiler.

Public API (re-exported via ``__init__.py``):
    ValidationIssue
    validate_compatibility
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------ #
# Valid value sets
# ------------------------------------------------------------------ #

_VALID_METHODS: frozenset[str] = frozenset(
    {
        "ter_braak",
        "kennedy",
        "kennedy_joint",
        "freedman_lane",
        "freedman_lane_joint",
        "manly",
        "manly_joint",
        "score",
        "score_joint",
        "score_exact",
    }
)

_VALID_RANDOMIZATIONS: frozenset[str] = frozenset({"permute", "sign_flip"})

_RESIDUAL_ONLY_METHODS: frozenset[str] = frozenset(
    {"ter_braak", "freedman_lane", "freedman_lane_joint"}
)

_GLMM_BLOCKED_METHODS: frozenset[str] = frozenset(
    {
        "ter_braak",
        "kennedy",
        "kennedy_joint",
        "freedman_lane",
        "freedman_lane_joint",
        "manly",
        "manly_joint",
        "score_joint",
    }
)

_SCORE_METHODS: frozenset[str] = frozenset({"score", "score_joint", "score_exact"})

_DIRECT_PERMUTATION_FAMILIES: frozenset[str] = frozenset({"ordinal", "multinomial"})

_GLMM_FAMILIES: frozenset[str] = frozenset({"logistic_mixed", "poisson_mixed"})

_CATEGORICAL_FAMILIES: frozenset[str] = frozenset({"ordinal", "multinomial"})

_COUNT_FAMILIES: frozenset[str] = frozenset(
    {"poisson", "poisson_mixed", "negative_binomial"}
)

_BINARY_FAMILIES: frozenset[str] = frozenset({"logistic", "logistic_mixed"})

_ALL_KNOWN_FAMILIES: frozenset[str] = frozenset(
    {
        "linear",
        "logistic",
        "poisson",
        "negative_binomial",
        "ordinal",
        "multinomial",
        "linear_mixed",
        "logistic_mixed",
        "poisson_mixed",
    }
)


# ------------------------------------------------------------------ #
# ValidationIssue
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class ValidationIssue:
    """A single structured validation issue.

    Returned from validators as items in a ``list[ValidationIssue]``.
    Callers inspect ``level`` to distinguish errors (which should block
    execution) from warnings (which should be surfaced but not block).

    Attributes:
        level: ``"error"`` for blocking issues; ``"warning"`` for
            non-fatal advisory messages.
        code: A short machine-readable identifier (e.g.
            ``"RESIDUAL_METHOD_DIRECT_FAMILY"``).  Stable across
            releases; suitable for programmatic dispatch.
        message: Human-readable explanation of the issue.
        suggestion: Recommended remediation action.
    """

    level: str  # "error" | "warning"
    code: str
    message: str
    suggestion: str


# ------------------------------------------------------------------ #
# Individual validators
# ------------------------------------------------------------------ #


def validate_family_compatibility(
    family_name: str,
    y: np.ndarray | None = None,
) -> list[ValidationIssue]:
    """Validate that *family_name* is known and compatible with *y*.

    When *y* is ``None``, only the structural check (known family name)
    is performed.  When *y* is provided, data-level checks are added
    (e.g. binary values for logistic, non-negative integers for Poisson).

    Args:
        family_name: The resolved family name string (e.g. ``"logistic"``).
        y: Optional outcome array for data-level validation.

    Returns:
        List of :class:`ValidationIssue` objects.  Empty list means valid.
    """
    issues: list[ValidationIssue] = []

    if family_name not in _ALL_KNOWN_FAMILIES and family_name != "auto":
        issues.append(
            ValidationIssue(
                level="error",
                code="UNKNOWN_FAMILY",
                message=f"Unknown family {family_name!r}.",
                suggestion=(f"Use one of: {sorted(_ALL_KNOWN_FAMILIES)} or 'auto'."),
            )
        )
        return issues

    if y is None:
        return issues

    y_arr = np.asarray(y)

    if family_name in _BINARY_FAMILIES:
        unique_vals = np.unique(y_arr)
        if not np.all(np.isin(unique_vals, [0, 1])):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="BINARY_FAMILY_NON_BINARY_Y",
                    message=(
                        f"family={family_name!r} requires binary outcomes "
                        f"(values in {{0, 1}}), but y contains: "
                        f"{unique_vals.tolist()}."
                    ),
                    suggestion=(
                        "Encode the outcome as 0/1, or choose a different "
                        "family (e.g. 'ordinal' for ordered categories, "
                        "'linear' for continuous outcomes)."
                    ),
                )
            )

    if family_name in _COUNT_FAMILIES:
        if np.any(y_arr < 0):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="COUNT_FAMILY_NEGATIVE_Y",
                    message=(
                        f"family={family_name!r} requires non-negative "
                        f"outcomes, but y contains negative values."
                    ),
                    suggestion=(
                        "Remove or recode negative values, or choose "
                        "'linear' for a continuous outcome."
                    ),
                )
            )
        if not np.allclose(y_arr, np.round(y_arr)):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="COUNT_FAMILY_NON_INTEGER_Y",
                    message=(
                        f"family={family_name!r} requires integer-valued "
                        f"outcomes, but y contains non-integer values."
                    ),
                    suggestion=(
                        "Round or recode y to integers, or choose 'linear' "
                        "for a continuous outcome."
                    ),
                )
            )

    if family_name in _CATEGORICAL_FAMILIES:
        unique_cats = np.unique(y_arr)
        if not np.allclose(y_arr, np.round(y_arr)):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="CATEGORICAL_FAMILY_NON_INTEGER_Y",
                    message=(
                        f"family={family_name!r} requires integer-coded outcomes."
                    ),
                    suggestion=(
                        "Encode categories as consecutive integers starting from 0."
                    ),
                )
            )
        if len(unique_cats) < 3:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="CATEGORICAL_FAMILY_TOO_FEW_LEVELS",
                    message=(
                        f"family={family_name!r} requires at least 3 outcome "
                        f"categories, but y has {len(unique_cats)}."
                    ),
                    suggestion=(
                        "Use 'logistic' for binary outcomes, or ensure at "
                        "least 3 ordered categories."
                    ),
                )
            )

    return issues


def validate_confounders(
    confounders: list[str],
    feature_names: list[str],
) -> list[ValidationIssue]:
    """Validate that all confounder names appear in *feature_names*.

    Args:
        confounders: Requested confounder column names.
        feature_names: Available feature column names from the design matrix.

    Returns:
        List of :class:`ValidationIssue` objects.  Empty list means valid.
    """
    issues: list[ValidationIssue] = []
    missing = [c for c in confounders if c not in feature_names]
    if missing:
        issues.append(
            ValidationIssue(
                level="error",
                code="CONFOUNDERS_NOT_IN_FEATURES",
                message=f"Confounders not found in feature matrix: {missing}.",
                suggestion=(
                    "Ensure confounder names match column names in X exactly.  "
                    f"Available columns: {feature_names}."
                ),
            )
        )
    return issues


def validate_panel_structure(
    panel_id: np.ndarray | None,
    time_id: np.ndarray | None,
    n_obs: int,
) -> list[ValidationIssue]:
    """Validate panel and time identifier arrays.

    Args:
        panel_id: Panel identifier array (or ``None``).
        time_id: Time period identifier array (or ``None``).
        n_obs: Expected number of observations.

    Returns:
        List of :class:`ValidationIssue` objects.  Empty list means valid.
    """
    issues: list[ValidationIssue] = []

    if panel_id is None:
        if time_id is not None:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="TIME_ID_WITHOUT_PANEL_ID",
                    message="time_id requires panel_id to be specified.",
                    suggestion="Provide panel_id alongside time_id.",
                )
            )
        return issues

    panel_arr = np.asarray(panel_id)
    if len(panel_arr) != n_obs:
        issues.append(
            ValidationIssue(
                level="error",
                code="PANEL_ID_LENGTH_MISMATCH",
                message=(
                    f"panel_id has {len(panel_arr)} elements but X has {n_obs} rows."
                ),
                suggestion=(
                    "Ensure panel_id has the same length as the number of observations."
                ),
            )
        )

    _, counts = np.unique(panel_arr, return_counts=True)
    if len(set(counts)) > 1:
        issues.append(
            ValidationIssue(
                level="warning",
                code="UNBALANCED_PANEL",
                message=(
                    f"Unbalanced panel: panels have between {int(counts.min())} "
                    f"and {int(counts.max())} observations.  Within-panel "
                    "permutation still works but statistical power varies "
                    "across panels."
                ),
                suggestion=(
                    "Consider balancing the panel or using a method that "
                    "accounts for unbalanced designs."
                ),
            )
        )

    if time_id is not None:
        time_arr = np.asarray(time_id)
        if len(time_arr) != n_obs:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="TIME_ID_LENGTH_MISMATCH",
                    message=(
                        f"time_id has {len(time_arr)} elements but X has {n_obs} rows."
                    ),
                    suggestion=(
                        "Ensure time_id has the same length as the number "
                        "of observations."
                    ),
                )
            )

    return issues


def validate_groups(
    groups: np.ndarray | None,
    n_obs: int,
) -> list[ValidationIssue]:
    """Validate exchangeability group labels.

    Args:
        groups: Group label array (or ``None`` for global exchangeability).
        n_obs: Expected number of observations.

    Returns:
        List of :class:`ValidationIssue` objects.  Empty list means valid.
    """
    issues: list[ValidationIssue] = []

    if groups is None:
        return issues

    groups_arr = np.asarray(groups)
    if len(groups_arr) != n_obs:
        issues.append(
            ValidationIssue(
                level="error",
                code="GROUPS_LENGTH_MISMATCH",
                message=(
                    f"groups has {len(groups_arr)} elements but X has {n_obs} rows."
                ),
                suggestion=(
                    "Ensure groups has the same length as the number of observations."
                ),
            )
        )
        return issues

    unique_groups = np.unique(groups_arr)
    if len(unique_groups) < 2:
        issues.append(
            ValidationIssue(
                level="warning",
                code="GROUPS_SINGLE_GROUP",
                message=(
                    f"groups contains only 1 unique value "
                    f"({unique_groups[0]!r}); exchangeability constraints "
                    "have no effect."
                ),
                suggestion=(
                    "Either remove groups= (equivalent, but clearer intent) "
                    "or verify that group labels were set correctly."
                ),
            )
        )

    return issues


# ------------------------------------------------------------------ #
# Structured compatibility matrix
# ------------------------------------------------------------------ #


def validate_compatibility(
    method: str,
    family_name: str,
    *,
    confounders: list[str] | None = None,
    ar_order: int | None = None,
    randomization: str = "permute",
    panel_id: np.ndarray | None = None,
    time_id: np.ndarray | None = None,
    direct_permutation: bool | None = None,
    is_glmm: bool | None = None,
    n_features: int | None = None,
) -> list[ValidationIssue]:
    """Structured compatibility matrix for method × family × parameter combinations.

    This is the primary entry point for the graph compiler (Phase 9) to
    perform per-equation validation before dispatching to
    ``randomization_test_regression``.

    Args:
        method: Permutation method string.
        family_name: Resolved family name string.
        confounders: Confounder column names (may be empty or ``None``).
        ar_order: AR order for panel data (``None`` = no AR correction).
        randomization: ``"permute"`` or ``"sign_flip"``.
        panel_id: Panel identifier array for AR requirement checks.
        time_id: Time identifier array for AR requirement checks.
        direct_permutation: Whether the family uses direct Y permutation.
            When ``None``, inferred from *family_name*.
        is_glmm: Whether the family is a GLMM.  When ``None``,
            inferred from *family_name*.
        n_features: Number of features in the design matrix.  Used for
            the ter Braak / logistic / single-feature check.

    Returns:
        List of :class:`ValidationIssue` objects.  Empty list means valid.
        Issues with ``level="error"`` indicate configurations that would
        raise ``ValueError`` in the main pipeline.  Issues with
        ``level="warning"`` mirror ``UserWarning`` calls.
    """
    issues: list[ValidationIssue] = []
    confounders = confounders or []

    if direct_permutation is None:
        direct_permutation = family_name in _DIRECT_PERMUTATION_FAMILIES
    if is_glmm is None:
        is_glmm = family_name in _GLMM_FAMILIES

    # ---- Method validity -----------------------------------------
    if method not in _VALID_METHODS:
        issues.append(
            ValidationIssue(
                level="error",
                code="INVALID_METHOD",
                message=f"Unknown method {method!r}.",
                suggestion=f"Use one of: {sorted(_VALID_METHODS)}.",
            )
        )
        return issues  # no further checks make sense with an unknown method

    # ---- Randomization validity ----------------------------------
    if randomization not in _VALID_RANDOMIZATIONS:
        issues.append(
            ValidationIssue(
                level="error",
                code="INVALID_RANDOMIZATION",
                message=f"Unknown randomization {randomization!r}.",
                suggestion="Use 'permute' or 'sign_flip'.",
            )
        )

    # ---- Residual-method + direct-permutation family -------------
    if direct_permutation and method in _RESIDUAL_ONLY_METHODS:
        issues.append(
            ValidationIssue(
                level="error",
                code="RESIDUAL_METHOD_DIRECT_FAMILY",
                message=(
                    f"method={method!r} requires well-defined residuals but "
                    f"family={family_name!r} uses direct Y permutation."
                ),
                suggestion=(
                    "Use 'manly', 'manly_joint', 'kennedy', 'kennedy_joint', "
                    "'score', or 'score_joint' for this family."
                ),
            )
        )

    # ---- GLMM + blocked methods ----------------------------------
    if is_glmm and method in _GLMM_BLOCKED_METHODS:
        issues.append(
            ValidationIssue(
                level="error",
                code="GLMM_BLOCKED_METHOD",
                message=(
                    f"method={method!r} is not supported for GLMM family "
                    f"{family_name!r}.  Re-estimating variance components "
                    "per permutation is computationally prohibitive and "
                    "statistically incorrect."
                ),
                suggestion="Use method='score' or method='score_exact'.",
            )
        )

    # ---- Kennedy / Freedman-Lane without confounders: warning ----
    if method in ("kennedy", "kennedy_joint") and not confounders:
        issues.append(
            ValidationIssue(
                level="warning",
                code="KENNEDY_NO_CONFOUNDERS",
                message=(
                    f"{method!r} called without confounders — all features "
                    "will be tested."
                ),
                suggestion=(
                    "Consider 'ter_braak' for unconditional tests, or "
                    "supply confounder names."
                ),
            )
        )

    if method in ("freedman_lane", "freedman_lane_joint") and not confounders:
        issues.append(
            ValidationIssue(
                level="warning",
                code="FREEDMAN_LANE_NO_CONFOUNDERS",
                message=(
                    f"{method!r} called without confounders — the reduced "
                    "model is intercept-only, which yields less power than "
                    "conditioning on other predictors."
                ),
                suggestion=(
                    "Consider 'ter_braak' for unconditional tests, or "
                    "supply confounder names."
                ),
            )
        )

    # ---- Manly on residual family: warning -----------------------
    if method in ("manly", "manly_joint") and not direct_permutation:
        issues.append(
            ValidationIssue(
                level="warning",
                code="MANLY_RESIDUAL_FAMILY",
                message=(
                    f"method={method!r} uses direct Y permutation (Manly "
                    f"1997), which tests marginal rather than partial "
                    f"association.  family={family_name!r} supports residuals."
                ),
                suggestion=(
                    "Use 'ter_braak', 'freedman_lane', or 'score' for more "
                    "powerful partial tests."
                ),
            )
        )

    # ---- score_exact on non-mixed family: warning ----------------
    if method == "score_exact" and not family_name.endswith("_mixed"):
        issues.append(
            ValidationIssue(
                level="warning",
                code="SCORE_EXACT_NON_MIXED",
                message=(
                    f"method='score_exact' is designed for GLMM families but "
                    f"family={family_name!r} is not a mixed-effects model.  "
                    "score_exact refits the full model per permutation "
                    "(expensive)."
                ),
                suggestion="Consider method='score' for non-mixed families.",
            )
        )

    # ---- ter Braak + logistic + single feature -------------------
    if (
        method == "ter_braak"
        and family_name == "logistic"
        and n_features is not None
        and n_features == 1
    ):
        issues.append(
            ValidationIssue(
                level="error",
                code="TER_BRAAK_LOGISTIC_SINGLE_FEATURE",
                message=(
                    "ter Braak with logistic regression requires at least "
                    "2 features because the reduced model (dropping the "
                    "single feature) has 0 predictors."
                ),
                suggestion=(
                    "Use method='kennedy' with confounders, or add additional features."
                ),
            )
        )

    # ---- AR order checks -----------------------------------------
    if ar_order is not None:
        if not isinstance(ar_order, int) or ar_order <= 0:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="AR_ORDER_INVALID",
                    message=(f"ar_order must be a positive integer, got {ar_order!r}."),
                    suggestion=("Set ar_order to a positive integer (e.g. 1, 2, 3)."),
                )
            )

        if panel_id is None:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="AR_ORDER_WITHOUT_PANEL_ID",
                    message="ar_order= requires panel_id= to identify panels.",
                    suggestion="Provide panel_id alongside ar_order.",
                )
            )

        if time_id is None:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="AR_ORDER_WITHOUT_TIME_ID",
                    message="ar_order= requires time_id= for temporal ordering.",
                    suggestion="Provide time_id alongside ar_order.",
                )
            )

        if family_name in _CATEGORICAL_FAMILIES:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="AR_ORDER_CATEGORICAL_FAMILY",
                    message=(
                        f"ar_order= is not supported for family={family_name!r}."
                        "  Categorical responses cannot have meaningful AR "
                        "residual structure."
                    ),
                    suggestion=(
                        "Remove ar_order, or use a continuous/count family "
                        "with panel data."
                    ),
                )
            )

        if method not in _SCORE_METHODS:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="AR_ORDER_NON_SCORE_METHOD",
                    message=(
                        f"ar_order= requires method='score' (or "
                        f"'score_joint', 'score_exact').  Got "
                        f"method={method!r}."
                    ),
                    suggestion="Set method='score' when using ar_order.",
                )
            )

    # ---- Sign-flip incompatibilities -----------------------------
    if randomization == "sign_flip":
        if method in ("manly", "manly_joint"):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="SIGN_FLIP_MANLY",
                    message=(
                        f"Sign-flip is incompatible with method={method!r}.  "
                        "Manly uses direct Y randomization — there are no "
                        "residuals to sign-flip."
                    ),
                    suggestion=("Use randomization='permute' with Manly methods."),
                )
            )

        if method == "score_exact":
            issues.append(
                ValidationIssue(
                    level="error",
                    code="SIGN_FLIP_SCORE_EXACT",
                    message=(
                        "Sign-flip is incompatible with method='score_exact'."
                        "  PQL-fixed permutation permutes Y directly."
                    ),
                    suggestion=("Use randomization='permute' with score_exact."),
                )
            )

        if direct_permutation:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="SIGN_FLIP_DIRECT_FAMILY",
                    message=(
                        f"Sign-flip requires well-defined residuals but "
                        f"family={family_name!r} uses direct Y permutation.  "
                        "Sign-flipping is not meaningful for ordinal or "
                        "multinomial responses."
                    ),
                    suggestion=("Use randomization='permute' for this family."),
                )
            )

    return issues

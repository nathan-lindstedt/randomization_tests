"""Tests for _validation.py — structured validation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from randomization_tests._validation import (
    ValidationIssue,
    validate_compatibility,
    validate_confounders,
    validate_family_compatibility,
    validate_groups,
    validate_panel_structure,
)

# ------------------------------------------------------------------ #
# ValidationIssue
# ------------------------------------------------------------------ #


class TestValidationIssue:
    def test_is_frozen(self):
        issue = ValidationIssue(
            level="error",
            code="TEST",
            message="test",
            suggestion="fix it",
        )
        with pytest.raises((AttributeError, TypeError)):
            issue.level = "warning"  # type: ignore[misc]

    def test_fields(self):
        issue = ValidationIssue(
            level="warning",
            code="SOME_CODE",
            message="a message",
            suggestion="a suggestion",
        )
        assert issue.level == "warning"
        assert issue.code == "SOME_CODE"
        assert issue.message == "a message"
        assert issue.suggestion == "a suggestion"

    def test_equality(self):
        a = ValidationIssue(level="error", code="X", message="m", suggestion="s")
        b = ValidationIssue(level="error", code="X", message="m", suggestion="s")
        assert a == b

    def test_inequality(self):
        a = ValidationIssue(level="error", code="X", message="m", suggestion="s")
        b = ValidationIssue(level="warning", code="X", message="m", suggestion="s")
        assert a != b


# ------------------------------------------------------------------ #
# validate_family_compatibility
# ------------------------------------------------------------------ #


class TestValidateFamilyCompatibility:
    def test_unknown_family_returns_error(self):
        issues = validate_family_compatibility("banana")
        assert len(issues) == 1
        assert issues[0].level == "error"
        assert issues[0].code == "UNKNOWN_FAMILY"

    def test_unknown_family_no_further_checks(self):
        # With unknown family + bad y, should still return only 1 issue.
        issues = validate_family_compatibility("banana", y=np.array([1.5, 2.5]))
        assert len(issues) == 1
        assert issues[0].code == "UNKNOWN_FAMILY"

    def test_auto_is_valid(self):
        issues = validate_family_compatibility("auto")
        assert issues == []

    def test_linear_no_y_is_valid(self):
        assert validate_family_compatibility("linear") == []

    def test_linear_with_y_is_valid(self):
        y = np.array([1.0, 2.0, 3.0])
        assert validate_family_compatibility("linear", y=y) == []

    def test_logistic_binary_y_is_valid(self):
        y = np.array([0, 1, 0, 1])
        assert validate_family_compatibility("logistic", y=y) == []

    def test_logistic_non_binary_y_returns_error(self):
        y = np.array([0, 1, 2, 3])
        issues = validate_family_compatibility("logistic", y=y)
        assert any(i.code == "BINARY_FAMILY_NON_BINARY_Y" for i in issues)

    def test_logistic_mixed_non_binary_y_returns_error(self):
        y = np.array([0.0, 0.5, 1.0])
        issues = validate_family_compatibility("logistic_mixed", y=y)
        assert any(i.code == "BINARY_FAMILY_NON_BINARY_Y" for i in issues)

    def test_poisson_valid_y(self):
        y = np.array([0, 1, 2, 5])
        assert validate_family_compatibility("poisson", y=y) == []

    def test_poisson_negative_y_returns_error(self):
        y = np.array([-1, 0, 1])
        issues = validate_family_compatibility("poisson", y=y)
        assert any(i.code == "COUNT_FAMILY_NEGATIVE_Y" for i in issues)

    def test_poisson_non_integer_y_returns_error(self):
        y = np.array([0.5, 1.5, 2.5])
        issues = validate_family_compatibility("poisson", y=y)
        assert any(i.code == "COUNT_FAMILY_NON_INTEGER_Y" for i in issues)

    def test_negative_binomial_valid_y(self):
        y = np.array([0, 1, 3, 10])
        assert validate_family_compatibility("negative_binomial", y=y) == []

    def test_ordinal_valid_y(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        assert validate_family_compatibility("ordinal", y=y) == []

    def test_ordinal_too_few_levels(self):
        y = np.array([0, 1, 0, 1])
        issues = validate_family_compatibility("ordinal", y=y)
        assert any(i.code == "CATEGORICAL_FAMILY_TOO_FEW_LEVELS" for i in issues)

    def test_ordinal_non_integer_y(self):
        y = np.array([0.0, 1.5, 2.0])
        issues = validate_family_compatibility("ordinal", y=y)
        assert any(i.code == "CATEGORICAL_FAMILY_NON_INTEGER_Y" for i in issues)

    def test_multinomial_valid_y(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        assert validate_family_compatibility("multinomial", y=y) == []

    def test_multinomial_single_class(self):
        y = np.array([1, 1, 1, 1])
        issues = validate_family_compatibility("multinomial", y=y)
        assert any(i.code == "CATEGORICAL_FAMILY_TOO_FEW_LEVELS" for i in issues)

    def test_all_known_families_no_y_valid(self):
        families = [
            "linear",
            "logistic",
            "poisson",
            "negative_binomial",
            "ordinal",
            "multinomial",
            "linear_mixed",
            "logistic_mixed",
            "poisson_mixed",
        ]
        for fam in families:
            issues = validate_family_compatibility(fam)
            assert issues == [], f"Unexpected issues for family {fam!r}: {issues}"


# ------------------------------------------------------------------ #
# validate_confounders
# ------------------------------------------------------------------ #


class TestValidateConfounders:
    def test_empty_confounders_is_valid(self):
        assert validate_confounders([], ["a", "b"]) == []

    def test_present_confounders_is_valid(self):
        assert validate_confounders(["a", "b"], ["a", "b", "c"]) == []

    def test_missing_confounder_returns_error(self):
        issues = validate_confounders(["z"], ["a", "b"])
        assert len(issues) == 1
        assert issues[0].code == "CONFOUNDERS_NOT_IN_FEATURES"
        assert "z" in issues[0].message

    def test_partially_missing_reports_only_missing(self):
        issues = validate_confounders(["a", "z"], ["a", "b"])
        assert len(issues) == 1
        assert "z" in issues[0].message
        # 'a' is present in features so should not appear in the missing list;
        # check the reported missing list is exactly ['z'].
        assert "['z']" in issues[0].message

    def test_all_missing_returns_error(self):
        issues = validate_confounders(["x", "y"], ["a", "b"])
        assert len(issues) == 1
        assert issues[0].level == "error"


# ------------------------------------------------------------------ #
# validate_panel_structure
# ------------------------------------------------------------------ #


class TestValidatePanelStructure:
    def test_no_panel_no_time_is_valid(self):
        assert validate_panel_structure(None, None, 10) == []

    def test_time_without_panel_returns_error(self):
        time_id = np.array([1, 2, 3])
        issues = validate_panel_structure(None, time_id, 3)
        assert any(i.code == "TIME_ID_WITHOUT_PANEL_ID" for i in issues)

    def test_panel_correct_length_is_valid(self):
        panel_id = np.array([0, 0, 1, 1])
        assert validate_panel_structure(panel_id, None, 4) == []

    def test_panel_length_mismatch_returns_error(self):
        panel_id = np.array([0, 1])
        issues = validate_panel_structure(panel_id, None, 5)
        assert any(i.code == "PANEL_ID_LENGTH_MISMATCH" for i in issues)

    def test_balanced_panel_no_warning(self):
        panel_id = np.array([0, 0, 1, 1])
        issues = validate_panel_structure(panel_id, None, 4)
        assert not any(i.code == "UNBALANCED_PANEL" for i in issues)

    def test_unbalanced_panel_warns(self):
        panel_id = np.array([0, 0, 0, 1, 1])
        issues = validate_panel_structure(panel_id, None, 5)
        assert any(i.code == "UNBALANCED_PANEL" for i in issues)
        assert all(i.level != "error" or i.code != "UNBALANCED_PANEL" for i in issues)
        warn = next(i for i in issues if i.code == "UNBALANCED_PANEL")
        assert warn.level == "warning"

    def test_time_id_correct_length_is_valid(self):
        panel_id = np.array([0, 0, 1, 1])
        time_id = np.array([1, 2, 1, 2])
        assert validate_panel_structure(panel_id, time_id, 4) == []

    def test_time_id_length_mismatch_returns_error(self):
        panel_id = np.array([0, 0, 1, 1])
        time_id = np.array([1, 2])
        issues = validate_panel_structure(panel_id, time_id, 4)
        assert any(i.code == "TIME_ID_LENGTH_MISMATCH" for i in issues)


# ------------------------------------------------------------------ #
# validate_groups
# ------------------------------------------------------------------ #


class TestValidateGroups:
    def test_none_groups_is_valid(self):
        assert validate_groups(None, 10) == []

    def test_correct_length_is_valid(self):
        groups = np.array([0, 0, 1, 1])
        assert validate_groups(groups, 4) == []

    def test_length_mismatch_returns_error(self):
        groups = np.array([0, 1])
        issues = validate_groups(groups, 5)
        assert any(i.code == "GROUPS_LENGTH_MISMATCH" for i in issues)

    def test_length_mismatch_returns_early(self):
        # With length mismatch, should not also warn about single group.
        groups = np.array([0])
        issues = validate_groups(groups, 5)
        assert not any(i.code == "GROUPS_SINGLE_GROUP" for i in issues)

    def test_multiple_groups_is_valid(self):
        groups = np.array([0, 0, 1, 1, 2, 2])
        assert validate_groups(groups, 6) == []

    def test_single_group_warns(self):
        groups = np.array([0, 0, 0, 0])
        issues = validate_groups(groups, 4)
        assert any(i.code == "GROUPS_SINGLE_GROUP" for i in issues)
        warn = next(i for i in issues if i.code == "GROUPS_SINGLE_GROUP")
        assert warn.level == "warning"


# ------------------------------------------------------------------ #
# validate_compatibility
# ------------------------------------------------------------------ #


class TestValidateCompatibility:
    def test_valid_basic(self):
        issues = validate_compatibility("ter_braak", "linear")
        assert issues == []

    def test_unknown_method_returns_early(self):
        issues = validate_compatibility("bad_method", "linear")
        assert len(issues) == 1
        assert issues[0].code == "INVALID_METHOD"

    def test_invalid_randomization(self):
        issues = validate_compatibility(
            "ter_braak", "linear", randomization="bootstrap"
        )
        assert any(i.code == "INVALID_RANDOMIZATION" for i in issues)

    # ---- Residual-method + direct-permutation family ----

    def test_ter_braak_ordinal_is_error(self):
        issues = validate_compatibility("ter_braak", "ordinal")
        assert any(i.code == "RESIDUAL_METHOD_DIRECT_FAMILY" for i in issues)

    def test_freedman_lane_multinomial_is_error(self):
        issues = validate_compatibility("freedman_lane", "multinomial")
        assert any(i.code == "RESIDUAL_METHOD_DIRECT_FAMILY" for i in issues)

    def test_kennedy_ordinal_is_valid(self):
        # kennedy is not residual-only
        issues = validate_compatibility("kennedy", "ordinal")
        codes = [i.code for i in issues]
        assert "RESIDUAL_METHOD_DIRECT_FAMILY" not in codes

    # ---- GLMM + blocked methods ----

    def test_ter_braak_logistic_mixed_is_error(self):
        issues = validate_compatibility("ter_braak", "logistic_mixed")
        assert any(i.code == "GLMM_BLOCKED_METHOD" for i in issues)

    def test_score_logistic_mixed_is_valid(self):
        issues = validate_compatibility("score", "logistic_mixed")
        codes = [i.code for i in issues]
        assert "GLMM_BLOCKED_METHOD" not in codes

    def test_score_exact_poisson_mixed_is_valid(self):
        issues = validate_compatibility("score_exact", "poisson_mixed")
        codes = [i.code for i in issues]
        assert "GLMM_BLOCKED_METHOD" not in codes

    # ---- Kennedy / Freedman-Lane without confounders ----

    def test_kennedy_no_confounders_warns(self):
        issues = validate_compatibility("kennedy", "linear")
        assert any(i.code == "KENNEDY_NO_CONFOUNDERS" for i in issues)
        warn = next(i for i in issues if i.code == "KENNEDY_NO_CONFOUNDERS")
        assert warn.level == "warning"

    def test_kennedy_with_confounders_no_warning(self):
        issues = validate_compatibility("kennedy", "linear", confounders=["z"])
        codes = [i.code for i in issues]
        assert "KENNEDY_NO_CONFOUNDERS" not in codes

    def test_freedman_lane_no_confounders_warns(self):
        issues = validate_compatibility("freedman_lane", "linear")
        assert any(i.code == "FREEDMAN_LANE_NO_CONFOUNDERS" for i in issues)

    def test_freedman_lane_with_confounders_no_warning(self):
        issues = validate_compatibility("freedman_lane", "linear", confounders=["z"])
        codes = [i.code for i in issues]
        assert "FREEDMAN_LANE_NO_CONFOUNDERS" not in codes

    # ---- Manly on residual family ----

    def test_manly_linear_warns(self):
        issues = validate_compatibility("manly", "linear")
        assert any(i.code == "MANLY_RESIDUAL_FAMILY" for i in issues)

    def test_manly_ordinal_no_warning(self):
        # ordinal is direct_permutation, so no warning for Manly
        issues = validate_compatibility("manly", "ordinal")
        codes = [i.code for i in issues]
        assert "MANLY_RESIDUAL_FAMILY" not in codes

    def test_manly_direct_permutation_override(self):
        # Explicit direct_permutation=True suppresses MANLY_RESIDUAL_FAMILY
        issues = validate_compatibility("manly", "linear", direct_permutation=True)
        codes = [i.code for i in issues]
        assert "MANLY_RESIDUAL_FAMILY" not in codes

    # ---- score_exact on non-mixed family ----

    def test_score_exact_linear_warns(self):
        issues = validate_compatibility("score_exact", "linear")
        assert any(i.code == "SCORE_EXACT_NON_MIXED" for i in issues)

    def test_score_exact_linear_mixed_no_warning(self):
        issues = validate_compatibility("score_exact", "linear_mixed")
        codes = [i.code for i in issues]
        assert "SCORE_EXACT_NON_MIXED" not in codes

    # ---- ter Braak + logistic + single feature ----

    def test_ter_braak_logistic_one_feature_error(self):
        issues = validate_compatibility("ter_braak", "logistic", n_features=1)
        assert any(i.code == "TER_BRAAK_LOGISTIC_SINGLE_FEATURE" for i in issues)

    def test_ter_braak_logistic_two_features_valid(self):
        issues = validate_compatibility("ter_braak", "logistic", n_features=2)
        codes = [i.code for i in issues]
        assert "TER_BRAAK_LOGISTIC_SINGLE_FEATURE" not in codes

    def test_ter_braak_logistic_no_n_features_valid(self):
        # Without n_features, the check is skipped
        issues = validate_compatibility("ter_braak", "logistic")
        codes = [i.code for i in issues]
        assert "TER_BRAAK_LOGISTIC_SINGLE_FEATURE" not in codes

    # ---- AR order checks ----

    def test_ar_order_valid(self):
        panel = np.array([0, 0, 1, 1])
        time = np.array([1, 2, 1, 2])
        issues = validate_compatibility(
            "score",
            "linear",
            ar_order=1,
            panel_id=panel,
            time_id=time,
        )
        assert issues == []

    def test_ar_order_negative_is_error(self):
        issues = validate_compatibility("score", "linear", ar_order=-1)
        assert any(i.code == "AR_ORDER_INVALID" for i in issues)

    def test_ar_order_without_panel_id_is_error(self):
        issues = validate_compatibility("score", "linear", ar_order=1)
        assert any(i.code == "AR_ORDER_WITHOUT_PANEL_ID" for i in issues)

    def test_ar_order_without_time_id_is_error(self):
        panel = np.array([0, 0, 1, 1])
        issues = validate_compatibility("score", "linear", ar_order=1, panel_id=panel)
        assert any(i.code == "AR_ORDER_WITHOUT_TIME_ID" for i in issues)

    def test_ar_order_ordinal_family_is_error(self):
        panel = np.array([0, 0, 1, 1])
        time = np.array([1, 2, 1, 2])
        issues = validate_compatibility(
            "score",
            "ordinal",
            ar_order=1,
            panel_id=panel,
            time_id=time,
        )
        assert any(i.code == "AR_ORDER_CATEGORICAL_FAMILY" for i in issues)

    def test_ar_order_non_score_method_is_error(self):
        panel = np.array([0, 0, 1, 1])
        time = np.array([1, 2, 1, 2])
        issues = validate_compatibility(
            "ter_braak",
            "linear",
            ar_order=1,
            panel_id=panel,
            time_id=time,
        )
        assert any(i.code == "AR_ORDER_NON_SCORE_METHOD" for i in issues)

    # ---- Sign-flip incompatibilities ----

    def test_sign_flip_manly_is_error(self):
        issues = validate_compatibility("manly", "ordinal", randomization="sign_flip")
        assert any(i.code == "SIGN_FLIP_MANLY" for i in issues)

    def test_sign_flip_score_exact_is_error(self):
        issues = validate_compatibility(
            "score_exact", "linear_mixed", randomization="sign_flip"
        )
        assert any(i.code == "SIGN_FLIP_SCORE_EXACT" for i in issues)

    def test_sign_flip_direct_family_is_error(self):
        issues = validate_compatibility("kennedy", "ordinal", randomization="sign_flip")
        assert any(i.code == "SIGN_FLIP_DIRECT_FAMILY" for i in issues)

    def test_sign_flip_linear_is_valid(self):
        issues = validate_compatibility(
            "ter_braak", "linear", randomization="sign_flip"
        )
        sign_flip_codes = {
            "SIGN_FLIP_MANLY",
            "SIGN_FLIP_SCORE_EXACT",
            "SIGN_FLIP_DIRECT_FAMILY",
        }
        assert not any(i.code in sign_flip_codes for i in issues)

    # ---- direct_permutation / is_glmm overrides ----

    def test_direct_permutation_override_true(self):
        # Force a linear family to be treated as direct_permutation.
        issues = validate_compatibility("ter_braak", "linear", direct_permutation=True)
        assert any(i.code == "RESIDUAL_METHOD_DIRECT_FAMILY" for i in issues)

    def test_is_glmm_override_true(self):
        # Force a linear family to be treated as GLMM.
        issues = validate_compatibility("ter_braak", "linear", is_glmm=True)
        assert any(i.code == "GLMM_BLOCKED_METHOD" for i in issues)

    def test_multiple_issues_collected(self):
        # AR order without panel/time and non-score method → 3 errors.
        issues = validate_compatibility("ter_braak", "linear", ar_order=1)
        codes = [i.code for i in issues]
        assert "AR_ORDER_WITHOUT_PANEL_ID" in codes
        assert "AR_ORDER_WITHOUT_TIME_ID" in codes
        assert "AR_ORDER_NON_SCORE_METHOD" in codes

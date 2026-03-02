"""Tests for the display module."""

from types import SimpleNamespace

import numpy as np

from randomization_tests.display import (
    _recommend_n_permutations,
    _significance_marker,
    _truncate,
    print_confounder_table,
    print_diagnostics_table,
    print_joint_results_table,
    print_results_table,
)
from randomization_tests.families import LinearFamily


class TestTruncate:
    def test_short_name_unchanged(self):
        assert _truncate("abc", 10) == "abc"

    def test_exact_length_unchanged(self):
        assert _truncate("abcdefghij", 10) == "abcdefghij"

    def test_long_name_truncated(self):
        result = _truncate("abcdefghijk", 10)
        assert len(result) == 10
        assert result.endswith("...")


class TestPrintResultsTable:
    def test_prints_without_error(self, capsys):
        results = SimpleNamespace(
            model_coefs=[1.0, 2.0],
            permuted_p_values=["0.01  (**)", "0.5  (ns)"],
            classic_p_values=["0.01  (**)", "0.45  (ns)"],
            p_value_threshold_one=0.05,
            p_value_threshold_two=0.01,
            p_value_threshold_three=0.001,
            method="ter_braak",
            family=LinearFamily(),
            feature_names=["x1", "x2"],
            target_name="y",
            diagnostics={
                "n_observations": 100,
                "n_features": 2,
                "r_squared": 0.85,
                "r_squared_adj": 0.84,
                "f_statistic": 50.0,
                "f_p_value": 1e-10,
                "aic": 150.0,
                "bic": 155.0,
            },
        )
        print_results_table(results)
        captured = capsys.readouterr()
        assert "ter_braak" in captured.out
        assert "x1" in captured.out

    def test_kennedy_no_confounders_note(self, capsys):
        """Notes section appears for Kennedy without confounders."""
        results = SimpleNamespace(
            model_coefs=[1.0],
            permuted_p_values=["0.01  (**)"],
            classic_p_values=["0.01  (**)"],
            p_value_threshold_one=0.05,
            p_value_threshold_two=0.01,
            p_value_threshold_three=0.001,
            method="kennedy",
            confounders=[],
            family=LinearFamily(),
            feature_names=["x1"],
            target_name="y",
            diagnostics={
                "n_observations": 50,
                "n_features": 1,
                "aic": 100,
                "bic": 105,
            },
        )
        print_results_table(results)
        out = capsys.readouterr().out
        assert "Notes" in out
        assert "without confounders" in out

    def test_kennedy_with_confounders_no_note(self, capsys):
        """Notes section absent for Kennedy with confounders."""
        results = SimpleNamespace(
            model_coefs=[1.0],
            permuted_p_values=["0.01  (**)"],
            classic_p_values=["0.01  (**)"],
            p_value_threshold_one=0.05,
            p_value_threshold_two=0.01,
            p_value_threshold_three=0.001,
            method="kennedy",
            confounders=["x2"],
            family=LinearFamily(),
            feature_names=["x1"],
            target_name="y",
            diagnostics={
                "n_observations": 50,
                "n_features": 1,
                "aic": 100,
                "bic": 105,
            },
        )
        print_results_table(results)
        out = capsys.readouterr().out
        assert "Notes" not in out

    def test_ter_braak_no_note(self, capsys):
        """Notes section absent for ter Braak method."""
        results = SimpleNamespace(
            model_coefs=[1.0],
            permuted_p_values=["0.01  (**)"],
            classic_p_values=["0.01  (**)"],
            p_value_threshold_one=0.05,
            p_value_threshold_two=0.01,
            p_value_threshold_three=0.001,
            method="ter_braak",
            family=LinearFamily(),
            feature_names=["x1"],
            target_name="y",
            diagnostics={
                "n_observations": 50,
                "n_features": 1,
                "aic": 100,
                "bic": 105,
            },
        )
        print_results_table(results)
        out = capsys.readouterr().out
        assert "Notes" not in out


class TestSignificanceMarker:
    """Tests for _significance_marker helper (Step 23)."""

    def test_straddles_threshold(self):
        """CI that straddles 0.05 returns [!] marker."""
        assert _significance_marker(0.04, 0.06, [0.05]) == "  [!]"

    def test_no_straddle(self):
        """CI entirely below threshold returns empty string."""
        assert _significance_marker(0.01, 0.03, [0.05]) == ""

    def test_entirely_above(self):
        """CI entirely above threshold returns empty string."""
        assert _significance_marker(0.06, 0.08, [0.05]) == ""

    def test_multiple_thresholds(self):
        """CI straddling the 0.01 threshold (but not 0.05) still flags."""
        marker = _significance_marker(0.008, 0.012, [0.05, 0.01, 0.001])
        assert marker == "  [!]"

    def test_ci_equals_threshold_boundary(self):
        """CI boundary exactly equal to threshold is not straddled."""
        # lo < t < hi must be strict; lo == t does not straddle
        assert _significance_marker(0.05, 0.06, [0.05]) == ""
        assert _significance_marker(0.04, 0.05, [0.05]) == ""


class TestRecommendNPermutations:
    """Tests for _recommend_n_permutations helper (Step 25)."""

    def test_returns_positive_int(self):
        b = _recommend_n_permutations(0.048, 0.05, alpha=0.05)
        assert isinstance(b, int)
        assert b >= 100

    def test_small_gap_gives_large_b(self):
        """When p_hat is close to threshold, need many permutations."""
        b = _recommend_n_permutations(0.0499, 0.05, alpha=0.05)
        assert b > 10_000

    def test_large_gap_gives_small_b(self):
        """When p_hat is far from threshold, fewer permutations needed."""
        b = _recommend_n_permutations(0.001, 0.05, alpha=0.05)
        assert b < 1000

    def test_zero_gap_returns_max(self):
        """When p_hat equals threshold, B is clamped to max."""
        b = _recommend_n_permutations(0.05, 0.05, alpha=0.05)
        assert b == 10_000_000

    def test_clamp_min(self):
        """Result never goes below 100."""
        b = _recommend_n_permutations(0.5, 0.05, alpha=0.05)
        assert b >= 100


class TestPValueCIInResultsTable:
    """Tests for p-value CI sub-row in print_results_table (Step 22)."""

    def _make_results(self, *, pval_ci=None, raw_p=None):
        ci_dict = {}
        if pval_ci is not None:
            ci_dict["pvalue_ci"] = pval_ci
            ci_dict["confidence_level"] = 0.95
        return SimpleNamespace(
            model_coefs=[1.0, 2.0],
            permuted_p_values=["0.032  (**)", "0.5  (ns)"],
            classic_p_values=["0.03  (**)", "0.45  (ns)"],
            raw_empirical_p=np.array(raw_p or [0.032, 0.5]),
            p_value_threshold_one=0.05,
            p_value_threshold_two=0.01,
            p_value_threshold_three=0.001,
            method="ter_braak",
            family=LinearFamily(),
            feature_names=["x1", "x2"],
            target_name="y",
            confidence_intervals=ci_dict,
            diagnostics={
                "n_observations": 100,
                "n_features": 2,
                "aic": 150.0,
            },
        )

    def test_ci_subrow_shown(self, capsys):
        """CI sub-row prints ± margin when pvalue_ci is present."""
        results = self._make_results(
            pval_ci=[[0.024, 0.042], [0.40, 0.60]],
        )
        print_results_table(results)
        out = capsys.readouterr().out
        assert "\u00b1" in out  # ± sign present

    def test_no_ci_no_subrow(self, capsys):
        """No sub-row when confidence_intervals is empty."""
        results = self._make_results()
        print_results_table(results)
        out = capsys.readouterr().out
        assert "\u00b1" not in out

    def test_borderline_warning_and_recommendation(self, capsys):
        """Borderline CI triggers [!] marker and n_permutations note."""
        # CI [0.04, 0.06] straddles 0.05
        results = self._make_results(
            pval_ci=[[0.04, 0.06], [0.40, 0.60]],
            raw_p=[0.048, 0.5],
        )
        print_results_table(results)
        out = capsys.readouterr().out
        assert "[!]" in out
        assert "n_permutations" in out
        assert "x1" in out  # borderline feature named

    def test_no_borderline_no_warning(self, capsys):
        """CI that doesn't straddle any threshold has no [!] or note."""
        # Both CIs well away from thresholds
        results = self._make_results(
            pval_ci=[[0.024, 0.042], [0.40, 0.60]],
        )
        print_results_table(results)
        out = capsys.readouterr().out
        assert "[!]" not in out
        assert "n_permutations" not in out

    def test_margin_decimal_aligned_with_pvalue(self, capsys):
        """The decimal in ± X.XXX aligns with the p-value decimal above."""
        results = self._make_results(
            pval_ci=[[0.024, 0.042], [0.40, 0.60]],
        )
        print_results_table(results)
        lines = capsys.readouterr().out.splitlines()
        # Find the p-value row for x1 and the margin sub-row below it
        for i, line in enumerate(lines):
            if line.startswith("x1"):
                pval_line = line
                margin_line = lines[i + 1]
                break
        # Column of '.' in p-value (first occurrence in the Emp column)
        # The Emp column starts at index 33 (22 feat + 9 coef + 2 gap)
        emp_section = pval_line[33:56]  # 23-char Emp column
        pval_dot = 33 + emp_section.index(".")
        # Column of '.' in the margin sub-row
        margin_section = margin_line[33:]
        margin_dot = 33 + margin_section.index(".")
        assert pval_dot == margin_dot, (
            f"Decimal misaligned: p-value at col {pval_dot}, margin at col {margin_dot}"
        )

    def test_small_margin_uses_scientific_notation(self, capsys):
        """Margin < 0.001 uses short scientific e-notation."""
        results = self._make_results(
            pval_ci=[[0.0001, 0.0015], [0.40, 0.60]],
        )
        print_results_table(results)
        out = capsys.readouterr().out
        assert "e-" in out  # scientific notation present

    def test_margin_3dp(self, capsys):
        """Margin uses 3 decimal places, not 4."""
        results = self._make_results(
            pval_ci=[[0.024, 0.042], [0.40, 0.60]],
        )
        print_results_table(results)
        out = capsys.readouterr().out
        # margin = (0.042 - 0.024) / 2 = 0.009 → "0.009"
        assert "\u00b1 0.009" in out


class TestPValueCIInDiagnosticsTable:
    """Tests for P-Val CI column in print_diagnostics_table (Step 24)."""

    def _make_results(self, *, pval_ci=None, exp_r2=None):
        ci_dict = {}
        if pval_ci is not None:
            ci_dict["pvalue_ci"] = pval_ci
            ci_dict["confidence_level"] = 0.95
        ext = {
            "standardized_coefs": [0.5, -0.3],
            "vif": [1.1, 1.2],
            "monte_carlo_se": [0.002, 0.003],
            "divergence_flags": ["", ""],
        }
        if exp_r2 is not None:
            ext["exposure_r_squared"] = exp_r2
        return SimpleNamespace(
            family=LinearFamily(),
            feature_names=["x1", "x2"],
            extended_diagnostics=ext,
            confidence_intervals=ci_dict,
            diagnostics={},
        )

    def test_ci_column_shown(self, capsys):
        """P-Val CI column header and values appear."""
        results = self._make_results(
            pval_ci=[[0.024, 0.042], [0.40, 0.60]],
        )
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        assert "P-Val CI" in out
        assert "[0.024, 0.042]" in out
        assert "[0.400, 0.600]" in out

    def test_no_ci_no_column(self, capsys):
        """P-Val CI column absent when no CI data."""
        results = self._make_results()
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        assert "P-Val CI" not in out

    def test_ci_column_hidden_with_exp_r2(self, capsys):
        """P-Val CI column hidden when Exp R² is shown (Kennedy layout)."""
        results = self._make_results(
            pval_ci=[[0.024, 0.042], [0.40, 0.60]],
            exp_r2=[0.5, 0.3],
        )
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        assert "P-Val CI" not in out
        assert "Exp R\u00b2" in out

    def test_legend_includes_pval_ci(self, capsys):
        """Legend explains P-Val CI when column is shown."""
        results = self._make_results(
            pval_ci=[[0.024, 0.042], [0.40, 0.60]],
        )
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        assert "Clopper-Pearson" in out


class TestConfounderDisplay:
    """Verify confounder rendering in results and diagnostics tables."""

    def _make_results_table_result(self):
        """3 features, x2 is a confounder."""
        return SimpleNamespace(
            model_coefs=[1.0, 0.5, 2.0],
            permuted_p_values=["0.032  (**)", "(confounder)", "0.841  (ns)"],
            classic_p_values=["0.03  (**)", "(confounder)", "0.82  (ns)"],
            raw_empirical_p=np.array([0.032, np.nan, 0.841]),
            p_value_threshold_one=0.05,
            p_value_threshold_two=0.01,
            p_value_threshold_three=0.001,
            method="kennedy",
            confounders=["x2"],
            family=LinearFamily(),
            feature_names=["x1", "x2", "x3"],
            target_name="y",
            confidence_intervals={
                "pvalue_ci": [[0.024, 0.042], [np.nan, np.nan], [0.78, 0.90]],
                "confidence_level": 0.95,
            },
            diagnostics={
                "n_observations": 100,
                "n_features": 3,
                "aic": 150.0,
            },
        )

    def _make_diagnostics_result(self):
        """3 features, x2 is a confounder — diagnostics layout."""
        return SimpleNamespace(
            family=LinearFamily(),
            feature_names=["x1", "x2", "x3"],
            extended_diagnostics={
                "standardized_coefs": [0.5, 0.2, -0.3],
                "vif": [1.1, 1.5, 1.2],
                "monte_carlo_se": [0.002, np.nan, 0.003],
                "divergence_flags": ["", "", ""],
            },
            confidence_intervals={
                "pvalue_ci": [[0.024, 0.042], [np.nan, np.nan], [0.78, 0.90]],
                "confidence_level": 0.95,
            },
            diagnostics={},
        )

    def test_results_table_confounder_label(self, capsys):
        """Confounder row shows '(confounder)' without 'N/A' prefix."""
        results = self._make_results_table_result()
        print_results_table(results)
        out = capsys.readouterr().out
        assert "(confounder)" in out
        assert "N/A (confounder)" not in out

    def test_results_table_confounder_no_margin(self, capsys):
        """Confounder row has no ± margin sub-row content."""
        results = self._make_results_table_result()
        print_results_table(results)
        lines = capsys.readouterr().out.splitlines()
        # Find the confounder row (x2) and check the sub-row below
        for i, line in enumerate(lines):
            if "x2" in line and "(confounder)" in line:
                # Next line should be blank (sub-row) or empty, not ±
                if i + 1 < len(lines):
                    assert "\u00b1" not in lines[i + 1]
                break
        else:
            raise AssertionError("Could not find confounder row for x2")

    def test_results_table_confounder_no_borderline(self, capsys):
        """Confounder never triggers [!] marker or n_permutations note."""
        results = self._make_results_table_result()
        print_results_table(results)
        out = capsys.readouterr().out
        # x2 should not appear in any borderline recommendation
        assert (
            "x2" not in out.split("n_permutations")[0]
            if "n_permutations" in out
            else True
        )

    def test_diagnostics_table_confounder_em_dash_pval_ci(self, capsys):
        """Confounder P-Val CI renders em dash, not [nan, nan]."""
        results = self._make_diagnostics_result()
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        assert "P-Val CI" in out
        assert "[nan" not in out.lower()
        # Non-confounder CIs should render normally
        assert "[0.024, 0.042]" in out
        assert "[0.780, 0.900]" in out

    def test_diagnostics_table_confounder_mc_se_em_dash(self, capsys):
        """Confounder MC SE renders em dash (pre-existing behaviour)."""
        results = self._make_diagnostics_result()
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        # Should have em dash for confounder MC SE (NaN)
        assert "\u2014" in out  # em dash


class TestPrintJointResultsTable:
    def test_prints_without_error(self, capsys):
        results = SimpleNamespace(
            observed_improvement=12.34,
            p_value=0.002,
            p_value_str="0.002  (**)",
            metric_type="RSS Reduction",
            family=LinearFamily(),
            feature_names=["x1", "x2"],
            target_name="y",
            features_tested=["x1", "x2"],
            confounders=[],
            p_value_threshold_one=0.05,
            p_value_threshold_two=0.01,
            p_value_threshold_three=0.001,
            method="kennedy_joint",
            diagnostics={
                "n_observations": 100,
                "n_features": 2,
                "r_squared": 0.85,
                "r_squared_adj": 0.84,
                "f_statistic": 50.0,
                "f_p_value": 1e-10,
                "aic": 150.0,
                "bic": 155.0,
            },
        )
        print_joint_results_table(results)
        captured = capsys.readouterr()
        assert "kennedy_joint" in captured.out
        assert "12.34" in captured.out

    def test_no_confounders_note(self, capsys):
        """Notes section appears for joint Kennedy without confounders."""
        results = SimpleNamespace(
            observed_improvement=5.0,
            p_value=0.04,
            p_value_str="0.04   (*)",
            metric_type="RSS Reduction",
            family=LinearFamily(),
            feature_names=["x1"],
            target_name="y",
            features_tested=["x1"],
            confounders=[],
            p_value_threshold_one=0.05,
            p_value_threshold_two=0.01,
            p_value_threshold_three=0.001,
            method="kennedy_joint",
            diagnostics={
                "n_observations": 50,
                "n_features": 1,
                "aic": 100,
                "bic": 105,
            },
        )
        print_joint_results_table(results)
        out = capsys.readouterr().out
        assert "Notes" in out
        assert "without confounders" in out

    def test_with_confounders_no_note(self, capsys):
        """Notes section absent for joint Kennedy with confounders."""
        results = SimpleNamespace(
            observed_improvement=5.0,
            p_value=0.04,
            p_value_str="0.04   (*)",
            metric_type="RSS Reduction",
            family=LinearFamily(),
            feature_names=["x1"],
            target_name="y",
            features_tested=["x1"],
            confounders=["x2"],
            p_value_threshold_one=0.05,
            p_value_threshold_two=0.01,
            p_value_threshold_three=0.001,
            method="kennedy_joint",
            diagnostics={
                "n_observations": 50,
                "n_features": 1,
                "aic": 100,
                "bic": 105,
            },
        )
        print_joint_results_table(results)
        out = capsys.readouterr().out
        assert "Notes" not in out


class TestPrintConfounderTable:
    """Tests for the print_confounder_table display function."""

    @staticmethod
    def _make_result(predictor, confounders, mediators):
        """Build a minimal identify_confounders-style result dict."""
        return {
            "predictor": predictor,
            "identified_confounders": confounders,
            "identified_mediators": mediators,
            "screening_results": {},
            "mediation_results": {},
        }

    def test_single_result_dict(self, capsys):
        """Accept a single identify_confounders result (not wrapped in a dict)."""
        result = self._make_result("X1", ["X2", "X3"], [])
        print_confounder_table(result)
        out = capsys.readouterr().out
        assert "X1" in out
        assert "X2, X3" in out

    def test_multi_predictor_dict(self, capsys):
        """Accept a dict keyed by predictor name."""
        results = {
            "X1": self._make_result("X1", ["X2"], []),
            "X2": self._make_result("X2", [], []),
        }
        print_confounder_table(results)
        out = capsys.readouterr().out
        # X1 has a confounder; X2 appears in the "No issues" section
        assert "Confounders:" in out
        assert "No issues:" in out

    def test_no_confounders_anywhere(self, capsys):
        """All predictors clean — should show 'No confounders or mediators'."""
        results = {
            "X1": self._make_result("X1", [], []),
            "X2": self._make_result("X2", [], []),
        }
        print_confounder_table(results)
        out = capsys.readouterr().out
        assert "No confounders, mediators, colliders, or moderators identified" in out

    def test_mediator_note_shown(self, capsys):
        """Notes section appears when a mediator is detected."""
        results = {
            "X1": self._make_result("X1", [], ["M1"]),
        }
        print_confounder_table(results)
        out = capsys.readouterr().out
        assert "Mediators:" in out
        assert "Notes" in out
        assert "Kennedy method" in out

    def test_mediator_note_absent_when_none(self, capsys):
        """Notes section is omitted when there are no mediators."""
        results = {
            "X1": self._make_result("X1", ["X2"], []),
        }
        print_confounder_table(results)
        out = capsys.readouterr().out
        assert "Notes" not in out

    def test_parameter_header(self, capsys):
        """Screening and mediation parameters appear in header."""
        result = self._make_result("X1", ["X2"], [])
        print_confounder_table(
            result,
            correlation_threshold=0.2,
            p_value_threshold=0.01,
            n_bootstrap=5000,
            confidence_level=0.99,
        )
        out = capsys.readouterr().out
        assert "|r|>=0.2" in out
        assert "p<0.01" in out
        assert "B=5000" in out
        assert "99% CI" in out

    def test_title_appears(self, capsys):
        """Custom title is centred in the output."""
        result = self._make_result("X1", [], [])
        print_confounder_table(result, title="My Custom Title")
        out = capsys.readouterr().out
        assert "My Custom Title" in out

    def test_80_char_lines(self, capsys):
        """All lines fit within 80 characters."""
        results = {
            "X1": self._make_result(
                "X1",
                ["Very Long Confounder Name Alpha", "Very Long Confounder Name Beta"],
                ["Very Long Mediator Name Gamma"],
            ),
        }
        print_confounder_table(results)
        out = capsys.readouterr().out
        for line in out.splitlines():
            assert len(line) <= 80, f"Line exceeds 80 chars: {line!r}"

    def test_vertical_spacing_between_predictors(self, capsys):
        """Blank line separates predictor blocks."""
        results = {
            "X1": self._make_result("X1", ["X3"], []),
            "X2": self._make_result("X2", ["X3"], []),
        }
        print_confounder_table(results)
        out = capsys.readouterr().out
        # Two predictor blocks should be separated by a blank line
        assert "\n\n  Predictor:" in out


class TestModelLevelDiagnosticsRendering:
    """Tests for the model-level diagnostics section (3-column grid)."""

    @staticmethod
    def _make_results(
        *,
        bp=None,
        cooks_d=None,
        perm_cov=None,
        pval_ci=None,
        emp_p=None,
        n_permutations=5000,
    ):
        ci_dict = {}
        if pval_ci is not None:
            ci_dict["pvalue_ci"] = pval_ci
            ci_dict["confidence_level"] = 0.95
        ext = {
            "standardized_coefs": [0.5, -0.3],
            "vif": [1.1, 1.2],
            "monte_carlo_se": [0.002, 0.003],
            "divergence_flags": ["", ""],
        }
        if bp is not None:
            ext["breusch_pagan"] = bp
        if cooks_d is not None:
            ext["cooks_distance"] = cooks_d
        if perm_cov is not None:
            ext["permutation_coverage"] = perm_cov
        return SimpleNamespace(
            family=LinearFamily(),
            feature_names=["x1", "x2"],
            extended_diagnostics=ext,
            confidence_intervals=ci_dict,
            diagnostics={},
            permuted_p_values=emp_p or ["0.032  (**)", "0.500  (ns)"],
            p_value_threshold_one=0.05,
            p_value_threshold_two=0.01,
            p_value_threshold_three=0.001,
            n_permutations=n_permutations,
        )

    def test_breusch_pagan_3_column(self, capsys):
        """Breusch-Pagan renders label | stat | p-value in 3 columns."""
        results = self._make_results(
            bp={
                "lm_stat": 1.2345,
                "lm_p_value": 0.2661,
                "f_stat": 0.6789,
                "f_p_value": 0.4123,
            }
        )
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        assert "Breusch-Pagan LM:" in out
        assert "1.2345" in out
        assert "p = 0.2661" in out
        assert "Breusch-Pagan F:" in out
        assert "0.6789" in out
        assert "p = 0.4123" in out

    def test_cooks_d_3_column(self, capsys):
        """Cook's D renders n obs. | threshold in 3 columns."""
        results = self._make_results(cooks_d={"n_influential": 3, "threshold": 0.04})
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        assert "Cook's D (> 4/n):" in out
        assert "3 obs." in out
        assert "threshold = 0.0400" in out

    def test_coverage_sufficient(self, capsys):
        """Coverage shows 'sufficient' when no CI straddles threshold."""
        results = self._make_results(
            perm_cov={
                "coverage_pct": "12.4%",
                "n_factorial_str": "40,320",
                "coverage_str": "12.4% of 40,320 possible",
            },
            pval_ci=[[0.024, 0.042], [0.40, 0.60]],
            n_permutations=5000,
        )
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        assert "Coverage:" in out
        assert "12.4%" in out
        assert "sufficient" in out
        assert "borderline" not in out

    def test_coverage_borderline(self, capsys):
        """Coverage shows 'borderline' when a CI straddles a threshold."""
        results = self._make_results(
            perm_cov={
                "coverage_pct": "12.4%",
                "n_factorial_str": "40,320",
                "coverage_str": "12.4% of 40,320 possible",
            },
            pval_ci=[[0.04, 0.06], [0.40, 0.60]],
            n_permutations=5000,
        )
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        assert "Coverage:" in out
        assert "borderline" in out
        assert "sufficient" not in out

    def test_line_width_within_80(self, capsys):
        """All model-level diagnostic lines fit within 80 characters."""
        results = self._make_results(
            bp={
                "lm_stat": 1.2345,
                "lm_p_value": 0.2661,
                "f_stat": 0.6789,
                "f_p_value": 0.4123,
            },
            cooks_d={"n_influential": 3, "threshold": 0.04},
            perm_cov={
                "coverage_pct": "12.4%",
                "n_factorial_str": "40,320",
                "coverage_str": "12.4% of 40,320 possible",
            },
            pval_ci=[[0.024, 0.042], [0.40, 0.60]],
        )
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        in_section = False
        for line in out.splitlines():
            if "Model-level Diagnostics" in line:
                in_section = True
                continue
            if in_section and line.startswith("---"):
                # Skip separator lines
                continue
            if in_section and line.startswith("  "):
                assert len(line) <= 80, f"Line exceeds 80 chars: {line!r}"
            if in_section and (line.startswith("Notes") or line.startswith("===")):
                break

    def test_coverage_b_and_denominator(self, capsys):
        """Coverage detail includes B = n_permutations / denominator."""
        results = self._make_results(
            perm_cov={
                "coverage_pct": "12.4%",
                "n_factorial_str": "40,320",
                "coverage_str": "12.4% of 40,320 possible",
            },
            pval_ci=[[0.024, 0.042], [0.40, 0.60]],
            n_permutations=5000,
        )
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        assert "B = 5,000" in out
        assert "40,320" in out

    def test_coverage_factorial_overflow(self, capsys):
        """Overflow denominator renders as n! notation."""
        results = self._make_results(
            perm_cov={
                "coverage_pct": "< 0.1%",
                "n_factorial_str": "414!",
                "coverage_str": "5,000 of 414! possible",
            },
            pval_ci=[[0.024, 0.042], [0.40, 0.60]],
            n_permutations=5000,
        )
        print_diagnostics_table(results)
        out = capsys.readouterr().out
        assert "414!" in out
        assert "B = 5,000" in out
        assert "> 10^" not in out

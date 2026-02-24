"""Tests for the display module."""

from randomization_tests.display import (
    _truncate,
    print_confounder_table,
    print_joint_results_table,
    print_results_table,
)
from randomization_tests.families import resolve_family


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
        results = {
            "model_coefs": [1.0, 2.0],
            "permuted_p_values": ["0.01 (**)", "0.5 (ns)"],
            "classic_p_values": ["0.01 (**)", "0.45 (ns)"],
            "p_value_threshold_one": 0.05,
            "p_value_threshold_two": 0.01,
            "method": "ter_braak",
            "model_type": "linear",
            "diagnostics": {
                "n_observations": 100,
                "n_features": 2,
                "r_squared": 0.85,
                "r_squared_adj": 0.84,
                "f_statistic": 50.0,
                "f_p_value": 1e-10,
                "aic": 150.0,
                "bic": 155.0,
            },
        }
        print_results_table(
            results,
            ["x1", "x2"],
            family=resolve_family("linear"),
            target_name="y",
        )
        captured = capsys.readouterr()
        assert "ter_braak" in captured.out
        assert "x1" in captured.out

    def test_kennedy_no_confounders_note(self, capsys):
        """Notes section appears for Kennedy without confounders."""
        results = {
            "model_coefs": [1.0],
            "permuted_p_values": ["0.01 (**)"],
            "classic_p_values": ["0.01 (**)"],
            "p_value_threshold_one": 0.05,
            "p_value_threshold_two": 0.01,
            "method": "kennedy",
            "confounders": [],
            "model_type": "linear",
            "diagnostics": {
                "n_observations": 50,
                "n_features": 1,
                "aic": 100,
                "bic": 105,
            },
        }
        print_results_table(results, ["x1"], family=resolve_family("linear"))
        out = capsys.readouterr().out
        assert "Notes" in out
        assert "without confounders" in out

    def test_kennedy_with_confounders_no_note(self, capsys):
        """Notes section absent for Kennedy with confounders."""
        results = {
            "model_coefs": [1.0],
            "permuted_p_values": ["0.01 (**)"],
            "classic_p_values": ["0.01 (**)"],
            "p_value_threshold_one": 0.05,
            "p_value_threshold_two": 0.01,
            "method": "kennedy",
            "confounders": ["x2"],
            "model_type": "linear",
            "diagnostics": {
                "n_observations": 50,
                "n_features": 1,
                "aic": 100,
                "bic": 105,
            },
        }
        print_results_table(results, ["x1"], family=resolve_family("linear"))
        out = capsys.readouterr().out
        assert "Notes" not in out

    def test_ter_braak_no_note(self, capsys):
        """Notes section absent for ter Braak method."""
        results = {
            "model_coefs": [1.0],
            "permuted_p_values": ["0.01 (**)"],
            "classic_p_values": ["0.01 (**)"],
            "p_value_threshold_one": 0.05,
            "p_value_threshold_two": 0.01,
            "method": "ter_braak",
            "model_type": "linear",
            "diagnostics": {
                "n_observations": 50,
                "n_features": 1,
                "aic": 100,
                "bic": 105,
            },
        }
        print_results_table(results, ["x1"], family=resolve_family("linear"))
        out = capsys.readouterr().out
        assert "Notes" not in out


class TestPrintJointResultsTable:
    def test_prints_without_error(self, capsys):
        results = {
            "observed_improvement": 12.34,
            "p_value": 0.002,
            "p_value_str": "0.002 (**)",
            "metric_type": "RSS Reduction",
            "model_type": "linear",
            "features_tested": ["x1", "x2"],
            "confounders": [],
            "p_value_threshold_one": 0.05,
            "p_value_threshold_two": 0.01,
            "method": "kennedy_joint",
            "diagnostics": {
                "n_observations": 100,
                "n_features": 2,
                "r_squared": 0.85,
                "r_squared_adj": 0.84,
                "f_statistic": 50.0,
                "f_p_value": 1e-10,
                "aic": 150.0,
                "bic": 155.0,
            },
        }
        print_joint_results_table(
            results,
            family=resolve_family("linear"),
            target_name="y",
        )
        captured = capsys.readouterr()
        assert "kennedy_joint" in captured.out
        assert "12.34" in captured.out

    def test_no_confounders_note(self, capsys):
        """Notes section appears for joint Kennedy without confounders."""
        results = {
            "observed_improvement": 5.0,
            "p_value": 0.04,
            "p_value_str": "0.04 (*)",
            "metric_type": "RSS Reduction",
            "model_type": "linear",
            "features_tested": ["x1"],
            "confounders": [],
            "p_value_threshold_one": 0.05,
            "p_value_threshold_two": 0.01,
            "method": "kennedy_joint",
            "diagnostics": {
                "n_observations": 50,
                "n_features": 1,
                "aic": 100,
                "bic": 105,
            },
        }
        print_joint_results_table(results, family=resolve_family("linear"))
        out = capsys.readouterr().out
        assert "Notes" in out
        assert "without confounders" in out

    def test_with_confounders_no_note(self, capsys):
        """Notes section absent for joint Kennedy with confounders."""
        results = {
            "observed_improvement": 5.0,
            "p_value": 0.04,
            "p_value_str": "0.04 (*)",
            "metric_type": "RSS Reduction",
            "model_type": "linear",
            "features_tested": ["x1"],
            "confounders": ["x2"],
            "p_value_threshold_one": 0.05,
            "p_value_threshold_two": 0.01,
            "method": "kennedy_joint",
            "diagnostics": {
                "n_observations": 50,
                "n_features": 1,
                "aic": 100,
                "bic": 105,
            },
        }
        print_joint_results_table(results, family=resolve_family("linear"))
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
        assert "No confounders or mediators identified" in out

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
        assert "|r| >= 0.2" in out
        assert "p < 0.01" in out
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

"""Tests for the display module."""

from randomization_tests.display import (
    _truncate,
    print_joint_results_table,
    print_results_table,
)


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
        print_results_table(results, ["x1", "x2"], target_name="y")
        captured = capsys.readouterr()
        assert "ter_braak" in captured.out
        assert "x1" in captured.out


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
        print_joint_results_table(results, target_name="y")
        captured = capsys.readouterr()
        assert "kennedy_joint" in captured.out
        assert "12.34" in captured.out

"""Tests for Polars DataFrame input compatibility."""

import numpy as np
import pandas as pd
import pytest

from randomization_tests._compat import _ensure_pandas_df

# Import polars; skip all tests in this module if not installed.
pl = pytest.importorskip("polars")


class TestEnsurePandasDf:
    """Tests for the _ensure_pandas_df converter."""

    def test_pandas_passthrough(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _ensure_pandas_df(df)
        assert result is df  # exact same object, no copy

    def test_polars_converted(self):
        pl_df = pl.DataFrame({"a": [1, 2, 3]})
        result = _ensure_pandas_df(pl_df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a"]
        assert result["a"].tolist() == [1, 2, 3]

    def test_polars_lazyframe_collected_and_converted(self):
        lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
        result = _ensure_pandas_df(lf)
        assert isinstance(result, pd.DataFrame)
        assert result["a"].tolist() == [1, 2, 3]

    def test_rejects_invalid_type(self):
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            _ensure_pandas_df([1, 2, 3])

    def test_error_includes_name(self):
        with pytest.raises(TypeError, match="'X'"):
            _ensure_pandas_df({"a": 1}, name="X")


class TestPolarsEndToEnd:
    """Verify that public API functions accept Polars DataFrames."""

    @staticmethod
    def _make_polars_data(n=100, seed=42):
        rng = np.random.default_rng(seed)
        X_pl = pl.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "x3": rng.standard_normal(n),
            }
        )
        y_vals = (
            2.0 * X_pl["x1"].to_numpy()
            - 1.0 * X_pl["x2"].to_numpy()
            + rng.standard_normal(n) * 0.5
        )
        y_pl = pl.DataFrame({"y": y_vals})
        return X_pl, y_pl

    def test_permutation_test_regression(self):
        from randomization_tests.core import permutation_test_regression

        X_pl, y_pl = self._make_polars_data()
        result = permutation_test_regression(
            X_pl,
            y_pl,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
        )
        assert "model_coefs" in result
        assert len(result["model_coefs"]) == 3

    def test_screen_potential_confounders(self):
        from randomization_tests.confounders import screen_potential_confounders

        X_pl, y_pl = self._make_polars_data()
        result = screen_potential_confounders(X_pl, y_pl, predictor="x1")
        assert "potential_confounders" in result

    def test_identify_confounders(self):
        from randomization_tests.confounders import identify_confounders

        X_pl, y_pl = self._make_polars_data()
        result = identify_confounders(X_pl, y_pl, predictor="x1", random_state=42)
        assert "identified_confounders" in result

    def test_calculate_p_values(self):
        from randomization_tests.families import LinearFamily
        from randomization_tests.pvalues import calculate_p_values

        X_pl, y_pl = self._make_polars_data()
        model_coefs = np.array([2.0, -1.0, 0.0])
        permuted_coefs = np.random.default_rng(0).standard_normal((50, 3))
        emp, asy, raw_emp, raw_asy = calculate_p_values(
            X_pl, y_pl, permuted_coefs, model_coefs, family=LinearFamily()
        )
        assert len(emp) == 3
        assert len(asy) == 3
        assert len(raw_emp) == 3
        assert len(raw_asy) == 3

    def test_results_match_pandas(self):
        """Polars and pandas inputs should produce identical results."""
        from randomization_tests.core import permutation_test_regression

        X_pl, y_pl = self._make_polars_data()
        X_pd = X_pl.to_pandas()
        y_pd = y_pl.to_pandas()

        result_pl = permutation_test_regression(
            X_pl,
            y_pl,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
        )
        result_pd = permutation_test_regression(
            X_pd,
            y_pd,
            n_permutations=50,
            method="ter_braak",
            random_state=42,
        )
        np.testing.assert_allclose(result_pl["model_coefs"], result_pd["model_coefs"])
        assert result_pl["permuted_p_values"] == result_pd["permuted_p_values"]

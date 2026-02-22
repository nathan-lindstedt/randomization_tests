"""Edge-case tests for input validation and boundary conditions.

Covers: empty data, single-feature models, constant columns, perfect
separation, permutation counts near n!, NaN/Inf/non-numeric inputs,
multi-column y, confounder validation, and n_permutations bounds.
"""

import numpy as np
import pandas as pd
import pytest

from randomization_tests.core import permutation_test_regression
from randomization_tests.permutations import generate_unique_permutations

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _linear_data(n: int = 50, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small linear dataset for fast edge-case runs."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n)})
    y = pd.DataFrame({"y": 2.0 * X["x1"] + rng.standard_normal(n) * 0.5})
    return X, y


def _binary_data(n: int = 100, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small binary dataset for fast edge-case runs."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n)})
    logits = 2.0 * X["x1"]
    probs = 1 / (1 + np.exp(-logits))
    y = pd.DataFrame({"y": rng.binomial(1, probs)})
    return X, y


# ------------------------------------------------------------------ #
# 1. Empty / degenerate DataFrames
# ------------------------------------------------------------------ #


class TestEmptyData:
    def test_zero_rows_raises(self) -> None:
        X = pd.DataFrame({"x1": pd.Series([], dtype=float)})
        y = pd.DataFrame({"y": pd.Series([], dtype=float)})
        with pytest.raises(ValueError, match="at least one observation"):
            permutation_test_regression(X, y, n_permutations=10)

    def test_zero_columns_raises(self) -> None:
        X = pd.DataFrame(index=range(10))
        y = pd.DataFrame({"y": np.random.default_rng(0).standard_normal(10)})
        with pytest.raises(ValueError, match="at least one feature"):
            permutation_test_regression(X, y, n_permutations=10)


# ------------------------------------------------------------------ #
# 2. Single-feature model
# ------------------------------------------------------------------ #


class TestSingleFeature:
    """One predictor, no confounders — verify structure is correct."""

    def test_ter_braak_linear(self) -> None:
        X, y = _linear_data()
        result = permutation_test_regression(
            X, y, n_permutations=50, method="ter_braak", random_state=0
        )
        assert len(result["model_coefs"]) == 1
        assert len(result["permuted_p_values"]) == 1
        assert result["model_type"] == "linear"

    def test_kennedy_linear(self) -> None:
        X, y = _linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X, y, n_permutations=50, method="kennedy", random_state=0
            )
        assert len(result["model_coefs"]) == 1

    def test_kennedy_joint_linear(self) -> None:
        X, y = _linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X, y, n_permutations=50, method="kennedy_joint", random_state=0
            )
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_ter_braak_logistic_single_feature_rejected(self) -> None:
        """ter Braak + logistic + 1 feature is degenerate (0-predictor reduced model)."""
        X, y = _binary_data()
        with pytest.raises(
            ValueError, match="ter Braak.*logistic.*at least 2 features"
        ):
            permutation_test_regression(
                X, y, n_permutations=50, method="ter_braak", random_state=0
            )

    def test_kennedy_logistic(self) -> None:
        X, y = _binary_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X, y, n_permutations=50, method="kennedy", random_state=0
            )
        assert len(result["model_coefs"]) == 1

    def test_kennedy_joint_logistic(self) -> None:
        X, y = _binary_data()
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X, y, n_permutations=50, method="kennedy_joint", random_state=0
            )
        assert "p_value" in result


# ------------------------------------------------------------------ #
# 3. Constant column (zero variance)
# ------------------------------------------------------------------ #


class TestConstantColumn:
    def test_single_constant_among_valid(self) -> None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x1": rng.standard_normal(50), "x_const": np.full(50, 5.0)})
        y = pd.DataFrame({"y": rng.standard_normal(50)})
        with pytest.raises(ValueError, match="zero variance.*x_const"):
            permutation_test_regression(X, y, n_permutations=10)

    def test_all_constant(self) -> None:
        X = pd.DataFrame({"x1": np.ones(50), "x2": np.full(50, 3.0)})
        y = pd.DataFrame({"y": np.random.default_rng(0).standard_normal(50)})
        with pytest.raises(ValueError, match="zero variance"):
            permutation_test_regression(X, y, n_permutations=10)


# ------------------------------------------------------------------ #
# 4. Perfect separation in logistic regression
# ------------------------------------------------------------------ #


@pytest.mark.filterwarnings(
    "ignore::statsmodels.tools.sm_exceptions.PerfectSeparationWarning"
)
@pytest.mark.filterwarnings(
    "ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning"
)
class TestPerfectSeparation:
    """Binary outcome perfectly separated by one predictor.

    sklearn's L-BFGS will converge (possibly with large coefficients)
    but the function should complete without error and return finite
    p-values for all three methods.
    """

    @pytest.fixture()
    def separated_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        n = 80
        rng = np.random.default_rng(42)
        x = np.concatenate([np.full(n // 2, -2.0), np.full(n // 2, 2.0)])
        noise = rng.standard_normal(n) * 0.01
        # x1 perfectly separates classes; x2 is pure noise (included so
        # ter Braak's reduced model always has ≥1 feature).
        X = pd.DataFrame({"x1": x + noise, "x2": rng.standard_normal(n)})
        y = pd.DataFrame({"y": np.concatenate([np.zeros(n // 2), np.ones(n // 2)])})
        return X, y

    def test_ter_braak(self, separated_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, y = separated_data
        result = permutation_test_regression(
            X, y, n_permutations=50, method="ter_braak", random_state=0
        )
        assert all(np.isfinite(result["raw_empirical_p"]))
        assert result["model_type"] == "logistic"

    def test_kennedy(self, separated_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, y = separated_data
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X, y, n_permutations=50, method="kennedy", random_state=0
            )
        assert all(np.isfinite(result["raw_empirical_p"]))

    def test_kennedy_joint(
        self, separated_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        X, y = separated_data
        with pytest.warns(UserWarning, match="without confounders"):
            result = permutation_test_regression(
                X, y, n_permutations=50, method="kennedy_joint", random_state=0
            )
        assert np.isfinite(result["p_value"])


# ------------------------------------------------------------------ #
# 5. Permutation count near n!
# ------------------------------------------------------------------ #


class TestPermutationCountNearFactorial:
    """n=6 → n!=720.  Test boundary conditions."""

    def test_near_max_succeeds(self) -> None:
        """Request 700 of 719 available (exclude_identity=True)."""
        perms = generate_unique_permutations(
            n_samples=6,
            n_permutations=700,
            random_state=0,
            exclude_identity=True,
        )
        assert perms.shape == (700, 6)
        # All rows unique
        unique_rows = {tuple(row) for row in perms}
        assert len(unique_rows) == 700

    def test_exact_max_succeeds(self) -> None:
        """Request exactly 719 (all non-identity permutations)."""
        perms = generate_unique_permutations(
            n_samples=6,
            n_permutations=719,
            random_state=0,
            exclude_identity=True,
        )
        assert perms.shape == (719, 6)

    def test_over_max_with_exclude_raises(self) -> None:
        """Request 720 with exclude_identity=True → only 719 available."""
        with pytest.raises(ValueError, match="720.*719"):
            generate_unique_permutations(
                n_samples=6,
                n_permutations=720,
                random_state=0,
                exclude_identity=True,
            )

    def test_full_count_without_exclude_succeeds(self) -> None:
        """Request all 720 with exclude_identity=False."""
        perms = generate_unique_permutations(
            n_samples=6,
            n_permutations=720,
            random_state=0,
            exclude_identity=False,
        )
        assert perms.shape == (720, 6)
        unique_rows = {tuple(row) for row in perms}
        assert len(unique_rows) == 720

    def test_end_to_end_small_n(self) -> None:
        """Run a full test with small n where n_permutations ≈ n!."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x1": rng.standard_normal(6)})
        y = pd.DataFrame({"y": rng.standard_normal(6)})
        result = permutation_test_regression(
            X, y, n_permutations=100, method="ter_braak", random_state=0
        )
        assert len(result["model_coefs"]) == 1


# ------------------------------------------------------------------ #
# 6. NaN / Inf / non-numeric inputs
# ------------------------------------------------------------------ #


class TestBadData:
    def test_nan_in_y(self) -> None:
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y = pd.DataFrame({"y": [1.0, 2.0, np.nan, 4.0, 5.0]})
        with pytest.raises(ValueError, match="y contains NaN"):
            permutation_test_regression(X, y, n_permutations=10)

    def test_inf_in_y(self) -> None:
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y = pd.DataFrame({"y": [1.0, 2.0, np.inf, 4.0, 5.0]})
        with pytest.raises(ValueError, match="y contains infinite"):
            permutation_test_regression(X, y, n_permutations=10)

    def test_nan_in_X(self) -> None:
        X = pd.DataFrame({"x1": [1.0, np.nan, 3.0, 4.0, 5.0]})
        y = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})
        with pytest.raises(ValueError, match="X contains NaN"):
            permutation_test_regression(X, y, n_permutations=10)

    def test_inf_in_X(self) -> None:
        X = pd.DataFrame({"x1": [1.0, 2.0, np.inf, 4.0, 5.0]})
        y = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})
        with pytest.raises(ValueError, match="X contains infinite"):
            permutation_test_regression(X, y, n_permutations=10)

    def test_non_numeric_column(self) -> None:
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "region": ["a", "b", "c"]})
        y = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="non-numeric dtype.*region"):
            permutation_test_regression(X, y, n_permutations=10)


# ------------------------------------------------------------------ #
# 7. Multi-column y
# ------------------------------------------------------------------ #


class TestMultiColumnY:
    def test_two_column_y_raises(self) -> None:
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y = pd.DataFrame(
            {"y1": [1.0, 2.0, 3.0, 4.0, 5.0], "y2": [5.0, 4.0, 3.0, 2.0, 1.0]}
        )
        with pytest.raises(ValueError, match="single column.*2 columns"):
            permutation_test_regression(X, y, n_permutations=10)


# ------------------------------------------------------------------ #
# 8. Row count mismatch
# ------------------------------------------------------------------ #


class TestShapeMismatch:
    def test_X_y_row_mismatch(self) -> None:
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        y = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})
        with pytest.raises(ValueError, match="3 rows.*5 elements"):
            permutation_test_regression(X, y, n_permutations=10)


# ------------------------------------------------------------------ #
# 9. n_permutations bounds
# ------------------------------------------------------------------ #


class TestNPermutationsBounds:
    def test_zero_permutations_raises(self) -> None:
        X, y = _linear_data()
        with pytest.raises(ValueError, match="n_permutations must be >= 1"):
            permutation_test_regression(X, y, n_permutations=0)

    def test_negative_permutations_raises(self) -> None:
        X, y = _linear_data()
        with pytest.raises(ValueError, match="n_permutations must be >= 1"):
            permutation_test_regression(X, y, n_permutations=-5)


# ------------------------------------------------------------------ #
# 10. Confounder validation
# ------------------------------------------------------------------ #


class TestConfounderValidation:
    def test_missing_confounder_raises(self) -> None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x1": rng.standard_normal(50), "x2": rng.standard_normal(50)})
        y = pd.DataFrame({"y": rng.standard_normal(50)})
        with pytest.raises(ValueError, match="Confounders not found.*age"):
            permutation_test_regression(
                X, y, n_permutations=10, method="kennedy", confounders=["age"]
            )

    def test_kennedy_without_confounders_warns(self) -> None:
        X, y = _linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            permutation_test_regression(
                X, y, n_permutations=50, method="kennedy", random_state=0
            )

    def test_kennedy_joint_without_confounders_warns(self) -> None:
        X, y = _linear_data()
        with pytest.warns(UserWarning, match="without confounders"):
            permutation_test_regression(
                X, y, n_permutations=50, method="kennedy_joint", random_state=0
            )

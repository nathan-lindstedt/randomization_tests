"""Tests for conformal prediction module (Phase 6, Steps 15-17).

Covers:
- Split conformal exact finite-sample coverage on linear models
- Exact quantile constant order statistic formula on small-n
- Jackknife+ distribution-free 1 - 2*alpha theoretical coverage guarantee
- Classification prediction set validity on Logistic and Multinomial models
- Locally-weighted adaptive interval narrowing in high-density / low-noise regions
- Hierarchical clustered conformal prediction validity under strong ICC
- Determinism under fixed random_state
- Input handling (DataFrames, 1D vs 2D queries) and error guards
- Display formatting via print_conformal_table
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from randomization_tests import (
    ExchangeabilityTree,
    conformal_prediction,
    evaluate_conformal,
    print_conformal_evaluation_table,
    print_conformal_table,
)
from randomization_tests._kernels import GaussianKernel


class TestSplitConformalRegression:
    """Tests for split conformal prediction on continuous outcomes."""

    def test_conformal_exact_quantile_constant(self) -> None:
        """Calibration uses the ceil((n_cal + 1)(1 - alpha))-th order statistic.

        Small-n hand example:
        n_cal = 9, alpha = 0.10 -> (9 + 1) * (0.90) = 10.0 -> k = 10 ranks.
        The 10th of 10 ranks is the maximum score.
        """
        rng = np.random.default_rng(42)
        n = 18
        X = rng.standard_normal((n, 2))
        y = 1.5 * X[:, 0] + rng.standard_normal(n)
        X_new = np.array([[1.0, 0.5]])

        # 18 observations with cal_ratio=0.5 -> 9 train, 9 cal
        res = conformal_prediction(
            X,
            y,
            X_new,
            confidence_level=0.90,
            cal_ratio=0.5,
            method="split",
            random_state=42,
        )
        assert res.method == "split"
        assert res.coverage_guarantee == "finite_sample_exact"
        assert res.diagnostics["n_cal"] == 9
        assert res.diagnostics["quantile_rank_k"] == 9
        # Threshold must be the maximum calibration residual
        assert res.diagnostics["threshold_q"] == float(np.max(res.nonconformity_scores))

    def test_conformal_linear_coverage(self) -> None:
        """Monte Carlo verification: empirical coverage >= 1 - alpha."""
        n = 100
        p = 2
        alpha = 0.10
        beta = np.array([2.0, -1.0])

        n_reps = 100
        covered = 0

        for rep in range(n_reps):
            seed = 1000 + rep
            rep_rng = np.random.default_rng(seed)
            X = rep_rng.standard_normal((n, p))
            y = X @ beta + rep_rng.normal(0, 1.0, size=n)

            X_test = rep_rng.standard_normal((1, p))
            y_test = float((X_test @ beta + rep_rng.normal(0, 1.0))[0])

            res = conformal_prediction(
                X,
                y,
                X_test,
                confidence_level=1.0 - alpha,
                cal_ratio=0.5,
                method="split",
                random_state=seed,
            )
            assert res.prediction_interval is not None
            lo, hi = res.prediction_interval
            if lo <= y_test <= hi:
                covered += 1

        empirical_coverage = covered / n_reps
        # Monte Carlo tolerance: binomial SE = sqrt(0.9 * 0.1 / 100) ≈ 0.03
        assert empirical_coverage >= 1.0 - alpha - 0.05
        assert empirical_coverage >= 0.85

    def test_conformal_kernel_narrower(self) -> None:
        """Locally-weighted intervals are narrower in dense/low-noise regions."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.uniform(-2, 2, size=(n, 1))
        # Variance is tiny near 0 (0.2) and large near +/- 2 (1.8)
        sigma_true = 0.2 + 0.8 * np.abs(X.ravel())
        y = 2.0 * X.ravel() + rng.normal(0, sigma_true)

        kernel = GaussianKernel(sigma=0.5)

        # Low noise query point at x=0
        res_dense = conformal_prediction(
            X,
            y,
            np.array([[0.0]]),
            kernel=kernel,
            confidence_level=0.90,
            random_state=42,
        )
        # High noise query point at x=2
        res_sparse = conformal_prediction(
            X,
            y,
            np.array([[2.0]]),
            kernel=kernel,
            confidence_level=0.90,
            random_state=42,
        )

        assert res_dense.prediction_interval is not None
        assert res_sparse.prediction_interval is not None

        width_dense = (
            res_dense.prediction_interval[1] - res_dense.prediction_interval[0]
        )
        width_sparse = (
            res_sparse.prediction_interval[1] - res_sparse.prediction_interval[0]
        )

        assert width_dense < width_sparse
        assert width_dense < 0.6 * width_sparse

    def test_conformal_clustered(self) -> None:
        """Grouped conformal prediction valid under strong intra-cluster correlation."""
        n_clusters = 30
        cluster_size = 8
        n = n_clusters * cluster_size
        clusters = np.repeat(np.arange(n_clusters), cluster_size)

        alpha = 0.10
        n_sims = 50
        covered_cluster = 0

        for sim in range(n_sims):
            sim_rng = np.random.default_rng(2000 + sim)
            u = sim_rng.normal(0, 2.0, size=n_clusters)
            X = sim_rng.standard_normal((n, 2))
            eps = sim_rng.normal(0, 0.5, size=n)
            y = 1.0 * X[:, 0] + u[clusters] + eps

            u_new = sim_rng.normal(0, 2.0)
            X_new = sim_rng.standard_normal((cluster_size, 2))
            y_new = (
                1.0 * X_new[:, 0] + u_new + sim_rng.normal(0, 0.5, size=cluster_size)
            )

            res = conformal_prediction(
                X,
                y,
                X_new,
                groups=clusters,
                confidence_level=1.0 - alpha,
                cal_ratio=0.5,
                random_state=sim,
            )
            assert res.prediction_interval is not None
            lo, hi = res.prediction_interval
            if np.all((lo <= y_new) & (y_new <= hi)):
                covered_cluster += 1

        cluster_cov = covered_cluster / n_sims
        assert cluster_cov >= 1.0 - alpha - 0.08

    def test_conformal_with_exchangeability_tree(self) -> None:
        """Clustered conformal accepts ExchangeabilityTree instances."""
        rng = np.random.default_rng(42)
        districts = np.repeat([0, 1], 20)
        schools = np.repeat(np.arange(4), 10)
        tree = ExchangeabilityTree.from_labels(
            [districts, schools], strategies=["between", "within"]
        )

        X = rng.standard_normal((40, 2))
        y = X[:, 0] + rng.standard_normal(40)
        X_new = np.array([[0.5, -0.5]])

        res = conformal_prediction(
            X,
            y,
            X_new,
            groups=tree,
            confidence_level=0.90,
            random_state=42,
        )
        assert res.prediction_interval is not None
        assert isinstance(res.groups, ExchangeabilityTree)


class TestJackknifePlusRegression:
    """Tests for Jackknife+ conformal prediction (Barber et al. 2021)."""

    def test_conformal_jackknife_plus_coverage(self) -> None:
        """Jackknife+ empirical coverage satisfies 1 - 2*alpha theoretical guarantee."""
        n = 40
        p = 2
        alpha = 0.10
        beta = np.array([1.5, -0.8])

        n_reps = 60
        covered = 0

        for rep in range(n_reps):
            rep_rng = np.random.default_rng(3000 + rep)
            X = rep_rng.standard_normal((n, p))
            y = X @ beta + rep_rng.normal(0, 1.0, size=n)

            X_test = rep_rng.standard_normal((1, p))
            y_test = float((X_test @ beta + rep_rng.normal(0, 1.0))[0])

            res = conformal_prediction(
                X,
                y,
                X_test,
                confidence_level=1.0 - alpha,
                method="jackknife_plus",
            )
            assert res.method == "jackknife_plus"
            assert res.coverage_guarantee == "worst_case_1_minus_2alpha"
            assert res.prediction_interval is not None
            assert res.point_predictions is not None
            lo, hi = res.prediction_interval
            if lo <= y_test <= hi:
                covered += 1

        empirical_coverage = covered / n_reps
        # Theoretical guarantee: >= 1 - 2*alpha = 0.80
        assert empirical_coverage >= 1.0 - 2 * alpha
        assert empirical_coverage >= 0.80


class TestConformalClassification:
    """Tests for conformal prediction sets on discrete outcomes."""

    def test_conformal_logistic_coverage(self) -> None:
        """Binary logistic prediction sets cover true label with probability >= 1 - alpha."""
        n = 150
        p = 2
        alpha = 0.10

        n_reps = 80
        covered = 0

        for rep in range(n_reps):
            rep_rng = np.random.default_rng(4000 + rep)
            X = rep_rng.standard_normal((n, p))
            probs = 1.0 / (1.0 + np.exp(-1.5 * X[:, 0] + 0.8 * X[:, 1]))
            y = rep_rng.binomial(1, probs).astype(float)

            X_test = rep_rng.standard_normal((1, p))
            p_test = 1.0 / (1.0 + np.exp(-1.5 * X_test[0, 0] + 0.8 * X_test[0, 1]))
            y_test = int(rep_rng.binomial(1, p_test))

            res = conformal_prediction(
                X,
                y,
                X_test,
                family="logistic",
                confidence_level=1.0 - alpha,
                cal_ratio=0.5,
                random_state=rep,
            )
            assert res.prediction_set is not None
            assert res.prediction_interval is None
            if y_test in res.prediction_set:
                covered += 1

        empirical_coverage = covered / n_reps
        assert empirical_coverage >= 1.0 - alpha - 0.05
        assert empirical_coverage >= 0.85

    def test_conformal_multinomial_prediction_sets(self) -> None:
        """3-class categorical outcomes yield valid prediction sets."""
        rng = np.random.default_rng(42)
        n = 120
        X = rng.standard_normal((n, 2))
        y = np.digitize(X[:, 0] + rng.standard_normal(n), [-0.5, 0.5]).astype(float)
        X_new = np.array([[0.5, -0.5]])

        res = conformal_prediction(
            X,
            y,
            X_new,
            family="ordinal",
            confidence_level=0.90,
            cal_ratio=0.5,
            random_state=42,
        )
        assert res.prediction_set is not None
        assert len(res.prediction_set) >= 1
        assert res.prediction_interval is None


class TestConformalInputsAndEdgeCases:
    """Tests for edge cases, input formats, determinism, and error handling."""

    def test_conformal_determinism(self) -> None:
        """Fixed random_state produces identical prediction intervals."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 2))
        y = 2.0 * X[:, 0] + rng.standard_normal(60)
        X_new = np.array([[1.0, 0.0]])

        res1 = conformal_prediction(X, y, X_new, confidence_level=0.95, random_state=42)
        res2 = conformal_prediction(X, y, X_new, confidence_level=0.95, random_state=42)
        assert res1.prediction_interval == res2.prediction_interval
        np.testing.assert_array_equal(
            res1.nonconformity_scores, res2.nonconformity_scores
        )

    def test_conformal_accepts_pandas_dataframes(self) -> None:
        """Accepts pandas DataFrame inputs for X and y."""
        rng = np.random.default_rng(42)
        df_X = pd.DataFrame(
            {"x1": rng.standard_normal(50), "x2": rng.standard_normal(50)}
        )
        df_y = pd.Series(2.0 * df_X["x1"] + rng.standard_normal(50), name="target")
        df_new = pd.DataFrame({"x1": [1.0], "x2": [-0.5]})

        res = conformal_prediction(
            df_X, df_y, df_new, confidence_level=0.90, random_state=42
        )
        assert res.prediction_interval is not None
        assert res.prediction_interval[0] < res.prediction_interval[1]

    def test_conformal_query_1d_array(self) -> None:
        """Query X_new can be passed as a 1D vector (p,)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2))
        y = X[:, 0] + rng.standard_normal(50)
        x_1d = np.array([0.5, -0.5])

        res = conformal_prediction(X, y, x_1d, confidence_level=0.90, random_state=42)
        assert res.prediction_interval is not None

    def test_conformal_invalid_parameters_raise(self) -> None:
        """Invalid confidence levels, methods, or sample sizes raise ValueError."""
        X = np.ones((10, 2))
        y = np.ones(10)
        x_new = np.ones((1, 2))

        with pytest.raises(ValueError, match="confidence_level"):
            conformal_prediction(X, y, x_new, confidence_level=1.5)

        with pytest.raises(ValueError, match="confidence_level"):
            conformal_prediction(X, y, x_new, confidence_level=-0.1)

        with pytest.raises(ValueError, match="cal_ratio"):
            conformal_prediction(X, y, x_new, cal_ratio=1.0)

        with pytest.raises(ValueError, match="Unknown method"):
            conformal_prediction(X, y, x_new, method="full_conformal")

        with pytest.raises(ValueError, match="at least 4 observations"):
            conformal_prediction(X[:3], y[:3], x_new)

    def test_conformal_display_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_conformal_table renders ASCII table cleanly for regression and classification."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 2))
        y = 2.0 * X[:, 0] + rng.standard_normal(60)
        x_new = np.array([[1.0, 0.0]])

        res_reg = conformal_prediction(
            X, y, x_new, confidence_level=0.95, random_state=42
        )
        assert res_reg.point_predictions is not None
        assert len(res_reg.point_predictions) == 1
        print_conformal_table(res_reg)
        out_reg = capsys.readouterr().out
        assert "Conformal Prediction — Prediction Interval" in out_reg
        assert "Guarantee: Finite-Sample Exact" in out_reg
        assert "Point Prediction:" in out_reg
        assert "Prediction Interval:" in out_reg

        # Classification table
        y_bin = (y > 0).astype(float)
        res_clf = conformal_prediction(
            X, y_bin, x_new, family="logistic", confidence_level=0.90, random_state=42
        )
        assert res_clf.probabilities is not None
        assert res_clf.probabilities.shape == (1, 2)
        print_conformal_table(res_clf)
        out_clf = capsys.readouterr().out
        assert "Conformal Prediction — Prediction Set" in out_clf
        assert "Class Probabilities:" in out_clf
        assert "Min Inclusion Prob:" in out_clf
        assert "Prediction Set:" in out_clf


class TestConformalBatchAndEvaluation:
    """Tests for multi-sample batch conformal prediction and out-of-sample evaluation."""

    def test_batch_regression_intervals_and_evaluate(self) -> None:
        """Batch regression yields shape (N, 2) intervals and accurate evaluation metrics."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((100, 3))
        y_train = 2.0 * X_train[:, 0] - 1.5 * X_train[:, 1] + rng.standard_normal(100)

        X_test = rng.standard_normal((30, 3))
        y_test = 2.0 * X_test[:, 0] - 1.5 * X_test[:, 1] + rng.standard_normal(30)

        res = conformal_prediction(
            X_train, y_train, X_test, confidence_level=0.90, random_state=42
        )

        assert res.intervals is not None
        assert res.intervals.shape == (30, 2)
        assert res.point_predictions is not None
        assert res.point_predictions.shape == (30,)
        assert np.all(res.intervals[:, 0] <= res.intervals[:, 1])

        # Evaluate via free function and method delegate
        eval_res1 = evaluate_conformal(res, y_test)
        eval_res2 = res.evaluate(y_test)

        assert eval_res1.n_eval == 30
        assert eval_res1.nominal_coverage == 0.90
        assert 0.0 <= eval_res1.empirical_coverage <= 1.0
        assert len(eval_res1.covered) == 30
        assert eval_res1.mean_interval_width is not None
        assert eval_res1.mean_interval_width > 0
        assert eval_res1.median_interval_width is not None

        # Delegate produces identical result
        assert eval_res1.empirical_coverage == eval_res2.empirical_coverage
        assert np.array_equal(eval_res1.covered, eval_res2.covered)

    def test_batch_classification_sets_and_evaluate(self) -> None:
        """Batch classification yields list of prediction sets and diagnostic rates."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((120, 2))
        prob_train = 1.0 / (1.0 + np.exp(-(1.5 * X_train[:, 0])))
        y_train = (rng.uniform(size=120) < prob_train).astype(float)

        X_test = rng.standard_normal((25, 2))
        prob_test = 1.0 / (1.0 + np.exp(-(1.5 * X_test[:, 0])))
        y_test = (rng.uniform(size=25) < prob_test).astype(float)

        res = conformal_prediction(
            X_train,
            y_train,
            X_test,
            family="logistic",
            confidence_level=0.95,
            random_state=42,
        )

        assert res.prediction_sets is not None
        assert len(res.prediction_sets) == 25
        assert res.probabilities is not None
        assert res.probabilities.shape == (25, 2)
        for s in res.prediction_sets:
            assert isinstance(s, np.ndarray)
            assert len(s) >= 1

        eval_res = res.evaluate(y_test)
        assert eval_res.n_eval == 25
        assert eval_res.nominal_coverage == 0.95
        assert 0.0 <= eval_res.empirical_coverage <= 1.0
        assert eval_res.singleton_rate is not None
        assert eval_res.ambiguous_rate is not None
        assert eval_res.empty_rate == 0.0
        assert eval_res.set_size_distribution is not None
        assert isinstance(eval_res.set_size_distribution, dict)

    def test_evaluate_mismatched_length_raises(self) -> None:
        """Passing evaluation response with mismatched length raises ValueError."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((50, 2))
        y_train = rng.standard_normal(50)
        X_test = rng.standard_normal((10, 2))
        y_mismatched = rng.standard_normal(15)

        res = conformal_prediction(X_train, y_train, X_test, random_state=42)
        with pytest.raises(ValueError, match="observations"):
            res.evaluate(y_mismatched)

    def test_display_batch_and_evaluation_tables(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_conformal_table for batch queries and print_conformal_evaluation_table render cleanly."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((80, 2))
        y_train = X_train[:, 0] + rng.standard_normal(80)
        X_test = rng.standard_normal((20, 2))
        y_test = X_test[:, 0] + rng.standard_normal(20)

        # Batch regression table
        res_reg = conformal_prediction(
            X_train, y_train, X_test, confidence_level=0.95, random_state=42
        )
        print_conformal_table(res_reg)
        out_batch = capsys.readouterr().out
        assert "Batch Queries (N):       20" in out_batch
        assert "Mean Interval Width:" in out_batch

        # Evaluation table
        eval_res = res_reg.evaluate(y_test)
        print_conformal_evaluation_table(eval_res)
        out_eval = capsys.readouterr().out
        assert "Guarantee: Empirical Out-of-Sample Verification" in out_eval
        assert "Evaluation Sample N:" in out_eval
        assert "Empirical Coverage:" in out_eval
        assert "Verification Status:" in out_eval
        assert "Mean Interval Width:" in out_eval

    def test_feature_stratified_coverage_regression(self) -> None:
        """evaluate_conformal computes stratified coverage, group counts, widths, and FSC."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((120, 2))
        y_train = 2.0 * X_train[:, 0] + rng.standard_normal(120)

        X_test = rng.standard_normal((40, 2))
        y_test = 2.0 * X_test[:, 0] + rng.standard_normal(40)
        groups_test = np.array(["Urban"] * 20 + ["Rural"] * 20)

        res = conformal_prediction(
            X_train, y_train, X_test, confidence_level=0.90, random_state=42
        )
        eval_res = res.evaluate(y_test, groups=groups_test)

        assert eval_res.group_coverage is not None
        assert "Urban" in eval_res.group_coverage
        assert "Rural" in eval_res.group_coverage
        assert eval_res.group_counts == {"Urban": 20, "Rural": 20}
        assert eval_res.group_mean_width is not None
        assert eval_res.fsc_metric is not None
        assert eval_res.fsc_metric == min(eval_res.group_coverage.values())

        # Verify manual subset calculation matches
        mask_urban = groups_test == "Urban"
        manual_urban_cov = float(np.mean(eval_res.covered[mask_urban]))
        assert eval_res.group_coverage["Urban"] == pytest.approx(manual_urban_cov)

    def test_feature_stratified_coverage_display(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_conformal_evaluation_table formats FSC table when groups are present."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((80, 2))
        y_train = X_train[:, 0] + rng.standard_normal(80)
        X_test = rng.standard_normal((30, 2))
        y_test = X_test[:, 0] + rng.standard_normal(30)
        groups = np.array([0] * 15 + [1] * 15)

        res = conformal_prediction(X_train, y_train, X_test, random_state=42)
        eval_res = res.evaluate(y_test, groups=groups)
        print_conformal_evaluation_table(eval_res)
        out = capsys.readouterr().out

        assert "Feature-Stratified Coverage (FSC Metric = min_g Coverage_g):" in out
        assert "Stratum / Group" in out
        assert "Mean Width/Size" in out

    def test_evaluate_mismatched_groups_length_raises(self) -> None:
        """Passing mismatched length groups to evaluate_conformal raises ValueError."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((50, 2))
        y_train = rng.standard_normal(50)
        X_test = rng.standard_normal((10, 2))
        y_test = rng.standard_normal(10)
        groups_bad = np.array([1, 2, 3])

        res = conformal_prediction(X_train, y_train, X_test, random_state=42)
        with pytest.raises(ValueError, match="groups has"):
            res.evaluate(y_test, groups=groups_bad)

    def test_conformal_table_with_index(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_conformal_table with index= renders specific single query from batch without recomputing."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((80, 2))
        y_train = 2.0 * X_train[:, 0] + rng.standard_normal(80)
        X_test = rng.standard_normal((10, 2))

        # Regression with index
        res_reg = conformal_prediction(X_train, y_train, X_test, random_state=42)
        assert res_reg.intervals is not None
        print_conformal_table(res_reg, index=2)
        out_reg = capsys.readouterr().out
        assert "Point Prediction:" in out_reg
        assert "Prediction Interval:" in out_reg
        assert (
            f"[{res_reg.intervals[2, 0]:.4f}, {res_reg.intervals[2, 1]:.4f}]" in out_reg
        )

        # Classification with index
        y_bin = (y_train > 0).astype(float)
        res_clf = conformal_prediction(
            X_train, y_bin, X_test, family="logistic", random_state=42
        )
        print_conformal_table(res_clf, index=3)
        out_clf = capsys.readouterr().out
        assert "Class Probabilities:" in out_clf
        assert "Prediction Set:" in out_clf

        # Out-of-bounds index raises IndexError
        with pytest.raises(IndexError, match="out of bounds"):
            print_conformal_table(res_reg, index=15)
        with pytest.raises(IndexError, match="out of bounds"):
            print_conformal_table(res_reg, index=-1)

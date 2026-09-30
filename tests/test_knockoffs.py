"""Tests for Model-X and Fixed-X Knockoff filters (Phase 8, Steps 21-23).

Validates the mathematical, geometric, and statistical guarantees of:
1. Barber & Candès (2015): Fixed-X knockoffs (finite-sample exact FDR).
2. Candès, Fan, Janson & Lv (2018): Model-X knockoffs (model-X approximate FDR).
3. Ledoit-Wolf covariance shrinkage and positive-definiteness invariants.
4. Barber & Candès Gram matrix preservation identities to 10^-10.
5. Knockoff+ threshold FDR control under sparse planted signals.
6. High-dimensional penalized estimation regime (n < 2p) with LassoCV.
7. MMD swap-exchangeability diagnostic under Gaussian vs heavy-tailed designs.
8. Result contracts, immutability, serialization, and formatted table rendering.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from scipy import linalg
from sklearn.covariance import LedoitWolf

from randomization_tests._knockoffs import (
    _compute_equicorrelated_s,
    _construct_fixed_x_knockoffs,
    _construct_model_x_knockoffs,
    knockoff_test,
)
from randomization_tests.display import print_knockoff_table


def _make_ar1_design(
    n: int, p: int, rho: float = 0.5, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian features with AR(1) covariance structure."""
    if rng is None:
        rng = np.random.default_rng(42)
    idx = np.arange(p)
    Sigma = rho ** np.abs(np.subtract.outer(idx, idx))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    return X, Sigma


class TestKnockoffMathematicalInvariants:
    """Rigorous mathematical and geometric invariant tests (Dual-Harness)."""

    def test_knockoff_covariance_validity(self) -> None:
        """Harness 2: 2*Sigma - diag(s) >= 0 and conditional covariance V > 0."""
        rng = np.random.default_rng(101)
        n, p = 120, 30
        X, _ = _make_ar1_design(n, p, rho=0.6, rng=rng)

        lw = LedoitWolf().fit(X)
        Sigma_hat = lw.covariance_
        s = _compute_equicorrelated_s(Sigma_hat)

        # Invariant 1: 2*Sigma - diag(s) must be positive semidefinite
        M = 2.0 * Sigma_hat - np.diag(s)
        eigvals_M = linalg.eigvalsh(M)
        assert np.min(eigvals_M) >= -1e-12, (
            f"2*Sigma - diag(s) has negative eigenvalue: {np.min(eigvals_M)}"
        )

        # Invariant 2: Conditional variance V = 2*diag(s) - diag(s) inv(Sigma) diag(s) > 0
        Sigma_inv = linalg.inv(Sigma_hat)
        V = 2.0 * np.diag(s) - np.diag(s) @ Sigma_inv @ np.diag(s)
        eigvals_V = linalg.eigvalsh(V)
        assert np.min(eigvals_V) > 0.0, (
            f"Conditional covariance V is not strictly positive definite: {np.min(eigvals_V)}"
        )

    def test_knockoff_fixed_x_gram_preservation(self) -> None:
        """Harness 2: Fixed-X Barber & Candes (2015) Gram preservation identities to 10^-10."""
        rng = np.random.default_rng(202)
        n, p = 160, 25
        X = rng.standard_normal((n, p))
        # Column standardization to unit norm
        col_norms = np.linalg.norm(X, axis=0)
        X_norm = X / col_norms

        G = X_norm.T @ X_norm
        s = _compute_equicorrelated_s(G)
        X_tilde = _construct_fixed_x_knockoffs(X_norm, s)

        # Invariant A: X_tilde^T X_tilde == X^T X
        gram_diff = float(np.max(np.abs(X_tilde.T @ X_tilde - G)))
        assert gram_diff < 1e-10, f"Fixed-X Gram equality violated: {gram_diff:.2e}"

        # Invariant B: X^T X_tilde == X^T X - diag(s)
        cross_diff = float(np.max(np.abs(X_norm.T @ X_tilde - (G - np.diag(s)))))
        assert cross_diff < 1e-10, (
            f"Fixed-X cross-Gram equality violated: {cross_diff:.2e}"
        )

    def test_knockoff_coin_flip_sign_inversion(self) -> None:
        """Harness 2: Swapping column j with knockoff flips the sign of W_j exactly."""
        rng = np.random.default_rng(303)
        n, p = 150, 20
        X, Sigma = _make_ar1_design(n, p, rho=0.4, rng=rng)
        s = _compute_equicorrelated_s(Sigma)
        X_tilde = _construct_model_x_knockoffs(X, Sigma, s, rng)

        # Plant signal on feature 0, features 1..p-1 are null
        beta = np.zeros(p)
        beta[0] = 3.0
        y = X @ beta + rng.standard_normal(n)

        # Joint augmented design
        X_aug = np.hstack([X, X_tilde])
        coefs = linalg.lstsq(X_aug, y)[0]
        W = np.abs(coefs[:p]) - np.abs(coefs[p:])

        # Swap null feature j=4 with knockoff p+4
        j_swap = 4
        X_swapped = X_aug.copy()
        X_swapped[:, [j_swap, p + j_swap]] = X_swapped[:, [p + j_swap, j_swap]]
        coefs_swapped = linalg.lstsq(X_swapped, y)[0]
        W_swapped = np.abs(coefs_swapped[:p]) - np.abs(coefs_swapped[p:])

        # W_j must flip sign exactly; all other W_k must be unchanged
        assert np.isclose(W_swapped[j_swap], -W[j_swap], atol=1e-10)
        for k in range(p):
            if k != j_swap:
                assert np.isclose(W_swapped[k], W[k], atol=1e-10)


class TestKnockoffValidationAndDispatch:
    """Input validation, family restrictions, and dispatch preference tests."""

    def test_knockoff_fixed_x_requires_n_ge_2p(self) -> None:
        """Fixed-X knockoffs strictly require n >= 2p; raises ValueError if n < 2p."""
        rng = np.random.default_rng(404)
        n, p = 30, 20  # 2p = 40 > n = 30
        df = pd.DataFrame(
            rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)]
        )
        y = rng.standard_normal(n)

        with pytest.raises(ValueError, match=r"Fixed-X knockoffs require n >= 2p"):
            knockoff_test(df, y, method="fixed_x", family="linear")

    def test_knockoff_fixed_x_requires_linear(self) -> None:
        """Fixed-X knockoffs strictly require a linear family; raises ValueError for discrete GLMs."""
        rng = np.random.default_rng(505)
        n, p = 50, 10
        df = pd.DataFrame(
            rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)]
        )
        y = rng.binomial(1, 0.5, size=n)

        with pytest.raises(
            ValueError, match=r"Fixed-X knockoffs require a linear model family"
        ):
            knockoff_test(df, y, method="fixed_x", family="logistic")

    def test_knockoff_dispatch_preference_warning(self) -> None:
        """Axiom D26: Warns user to consider fixed_x when family=linear and n >= 2p."""
        rng = np.random.default_rng(606)
        n, p = 60, 10
        df = pd.DataFrame(
            rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)]
        )
        y = rng.standard_normal(n)

        with pytest.warns(
            UserWarning,
            match=r"method='fixed_x' provides a finite-sample exact FDR guarantee",
        ):
            knockoff_test(
                df, y, method="equicorrelated", family="linear", random_state=42
            )

    @pytest.mark.filterwarnings("ignore:.*method='fixed_x' provides.*:UserWarning")
    def test_knockoff_sdp_fallback_warning(self) -> None:
        """When cvxpy is absent, method='sdp' warns and cleanly falls back to equicorrelated."""
        rng = np.random.default_rng(707)
        n, p = 50, 8
        df = pd.DataFrame(
            rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)]
        )
        y = rng.standard_normal(n)

        with pytest.warns(
            UserWarning, match=r"falling back to method='equicorrelated'"
        ):
            res = knockoff_test(df, y, method="sdp", random_state=42)
        assert res.method == "sdp"
        assert res.guarantee_tier == "model_x_approximate"


class TestKnockoffStatisticalPerformance:
    """Empirical FDR control and power verification (Dual-Harness)."""

    @pytest.mark.filterwarnings("ignore:.*method='fixed_x' provides.*:UserWarning")
    def test_knockoff_detects_signal(self) -> None:
        """Knockoff filter recovers strong sparse planted signals."""
        rng = np.random.default_rng(808)
        n, p = 200, 20
        X, _ = _make_ar1_design(n, p, rho=0.3, rng=rng)
        feature_names = [f"feat_{i}" for i in range(p)]
        df = pd.DataFrame(X, columns=feature_names)

        # Barber & Candes (2015): knockoff+ requires >= 1/q discoveries.
        # At q = 0.20, need at least 5 discoveries; plant 8 strong signals.
        beta = np.zeros(p)
        beta[:8] = 3.5
        y = X @ beta + rng.standard_normal(n)

        res = knockoff_test(
            df,
            y,
            method="equicorrelated",
            fdr_level=0.20,
            random_state=42,
        )

        assert "feat_0" in res.selected_features
        assert "feat_1" in res.selected_features
        assert "feat_2" in res.selected_features
        assert res.threshold < float("inf")
        assert len(res.selected_features) >= 5

    @pytest.mark.filterwarnings("ignore:.*method='fixed_x' provides.*:UserWarning")
    def test_knockoff_fdr_control(self) -> None:
        """Harness 1: Monte Carlo replications verify empirical FDP <= q + epsilon_MC."""
        n, p = 140, 25
        q_target = 0.20
        n_reps = 30
        fdp_list: list[float] = []

        for rep in range(n_reps):
            rng = np.random.default_rng(1000 + rep)
            X, _ = _make_ar1_design(n, p, rho=0.4, rng=rng)
            df = pd.DataFrame(X, columns=[f"x_{i}" for i in range(p)])

            # 8 true signals; knockoff+ requires >= 1/q = 5 discoveries
            beta = np.zeros(p)
            beta[:8] = 3.5
            y = X @ beta + rng.standard_normal(n)

            res = knockoff_test(
                df,
                y,
                method="equicorrelated",
                fdr_level=q_target,
                random_state=rep,
            )

            sel = res.selected_features
            if not sel:
                fdp = 0.0
            else:
                false_disc = sum(1 for name in sel if int(name.split("_")[1]) >= 8)
                fdp = false_disc / len(sel)
            fdp_list.append(fdp)

        empirical_fdr = float(np.mean(fdp_list))
        # With 30 reps, MC standard error is sqrt(0.2*0.8/30) ~= 0.073; 0.20 + 0.09 is safe
        assert empirical_fdr <= q_target + 0.09, (
            f"Empirical FDR {empirical_fdr:.3f} exceeded nominal {q_target}"
        )

    def test_knockoff_fixed_x_fdr_exact(self) -> None:
        """Harness 2: Fixed-X knockoffs achieve exact FDR control on non-Gaussian X."""
        rng = np.random.default_rng(909)
        n, p = 200, 25
        # Non-Gaussian heavy-tailed lognormal design matrix
        X_non_gauss = rng.lognormal(mean=0.0, sigma=0.8, size=(n, p))
        df = pd.DataFrame(X_non_gauss, columns=[f"log_{i}" for i in range(p)])

        # Barber & Candes (2015): knockoff+ requires >= 1/q discoveries.
        # At q = 0.20, need >= 5 discoveries; plant 8 strong signals.
        beta = np.zeros(p)
        beta[:8] = 4.0
        # Gaussian linear model for Y | X (the only requirement of Fixed-X)
        y = X_non_gauss @ beta + rng.standard_normal(n)

        res = knockoff_test(
            df,
            y,
            method="fixed_x",
            fdr_level=0.20,
            family="linear",
            random_state=42,
        )

        assert res.guarantee_tier == "finite_sample_exact"
        assert res.method == "fixed_x"
        assert len(res.selected_features) >= 5
        # Signal features should dominate selections
        true_selected = sum(
            1 for f in res.selected_features if int(f.split("_")[1]) < 8
        )
        assert true_selected >= 5

    def test_knockoff_high_dimensional_lasso(self) -> None:
        """High-dimensional regime n < 2p fits successfully via LassoCV."""
        rng = np.random.default_rng(1010)
        n, p = 50, 40  # 2p = 80 > n = 50
        X = rng.standard_normal((n, p))
        df = pd.DataFrame(X, columns=[f"v_{i}" for i in range(p)])

        beta = np.zeros(p)
        beta[:3] = [3.0, 3.0, 3.0]
        y = X @ beta + rng.standard_normal(n)

        res = knockoff_test(
            df,
            y,
            method="equicorrelated",
            estimator="lasso_cv",
            fdr_level=0.20,
            random_state=42,
        )

        assert res.diagnostics.get("estimator") == "lasso_cv"
        assert "lasso_alpha" in res.diagnostics
        assert isinstance(res.threshold, float)


class TestKnockoffDiagnosticsAndDisplay:
    """MMD swap diagnostics and table formatting."""

    @pytest.mark.filterwarnings("ignore:.*method='fixed_x' provides.*:UserWarning")
    def test_knockoff_swap_diagnostic_gaussian(self) -> None:
        """Gaussian design matrix passes MMD swap diagnostic (swap_violation=False)."""
        rng = np.random.default_rng(1111)
        n, p = 150, 10
        X, _ = _make_ar1_design(n, p, rho=0.3, rng=rng)
        df = pd.DataFrame(X, columns=[f"g_{i}" for i in range(p)])
        y = X[:, 0] * 2.0 + rng.standard_normal(n)

        res = knockoff_test(
            df,
            y,
            method="equicorrelated",
            random_state=42,
        )

        assert "mmd_swap_p_values" in res.diagnostics
        assert res.diagnostics["swap_violation"] is False

    @pytest.mark.filterwarnings("ignore:.*method='fixed_x' provides.*:UserWarning")
    def test_knockoff_swap_diagnostic_heavy_tailed(self) -> None:
        """Heavy-tailed Pareto design violates swap exchangeability (swap_violation=True)."""
        rng = np.random.default_rng(1212)
        n, p = 160, 10
        # Pareto distributed features with heavy tails (alpha = 1.5)
        X_pareto = rng.pareto(a=1.5, size=(n, p)) - 2.0
        df = pd.DataFrame(X_pareto, columns=[f"p_{i}" for i in range(p)])
        y = X_pareto[:, 0] * 2.0 + rng.standard_normal(n)

        res = knockoff_test(
            df,
            y,
            method="equicorrelated",
            random_state=42,
        )

        assert "mmd_swap_p_values" in res.diagnostics
        # Should flag swap violation due to non-Gaussianity
        assert res.diagnostics["swap_violation"] is True

    @pytest.mark.filterwarnings("ignore:.*method='fixed_x' provides.*:UserWarning")
    def test_knockoff_result_contract_and_display(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Result dataclass contract, dictionary access, and 80-column ASCII table rendering."""
        rng = np.random.default_rng(1313)
        n, p = 120, 15
        X, _ = _make_ar1_design(n, p, rho=0.2, rng=rng)
        df = pd.DataFrame(X, columns=[f"cov_{i}" for i in range(p)])
        # Plant 6 true signals for knockoff+ at q=0.20 (requires >= 1/q = 5 discoveries)
        beta = np.zeros(p)
        beta[:6] = 3.5
        y = X @ beta + rng.standard_normal(n)

        res = knockoff_test(
            df,
            y,
            method="equicorrelated",
            fdr_level=0.20,
            random_state=42,
        )

        # Frozen immutability check
        with pytest.raises(FrozenInstanceError):
            res.threshold = 0.99  # type: ignore[misc]

        # Dictionary access & serialization
        d = res.to_dict()
        assert isinstance(d, dict)
        assert d["fdr_level"] == 0.20
        assert isinstance(d["knockoff_statistics"], list)
        assert d["method"] == "equicorrelated"

        # Table rendering
        print_knockoff_table(res)
        captured = capsys.readouterr().out
        assert "Knockoff Filter Feature Selection (FDR Control)" in captured
        assert "[ Guarantee: Model-X Approximate FDR" in captured
        assert "Target FDR (q):" in captured
        assert "Threshold (τ₊):" in captured
        assert "W-Statistic" in captured
        assert "SELECTED (*)" in captured

        # Fixed-X table rendering
        res_fx = knockoff_test(
            df,
            y,
            method="fixed_x",
            fdr_level=0.20,
            random_state=42,
        )
        print_knockoff_table(res_fx)
        captured_fx = capsys.readouterr().out
        assert (
            "[ Guarantee: Finite-Sample Exact FDR (Barber & Candès 2015) ]"
            in captured_fx
        )

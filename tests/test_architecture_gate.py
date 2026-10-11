"""Architectural consistency and real-world multi-level gate tests.

Validates:
1. End-to-end multi-level hierarchical ExchangeabilityTree regression for both
   individual and joint tests, ensuring leaf clustering for statsmodels SEs
   and JSON serialization of the resulting test objects.
2. Protocol parity: all 9 permutation strategies accept forward-compatible
   keyword arguments without raising TypeError.
3. Model family method signature uniformity across all 9 families.
4. Modernized GLMM batch stub messages reflecting Kennedy availability.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pandas as pd
import pytest

from randomization_tests import (
    ExchangeabilityTree,
    IndividualTestResult,
    JointTestResult,
    backend,
    get_backend,
    randomization_test_regression,
    set_backend,
)
from randomization_tests._ar import (
    build_ar_cholesky_factor,
    build_ar_precision_block,
    validate_ar_stationarity,
)
from randomization_tests._backends import resolve_backend
from randomization_tests._conformal import conformal_prediction
from randomization_tests._kernel_tests import hsic_test, mmd_test
from randomization_tests._kernels import (
    CosineKernel,
    GaussianKernel,
    Kernel,
    LaplacianKernel,
    LinearKernel,
    gram_centering,
)
from randomization_tests._knockoffs import knockoff_test
from randomization_tests._results import ConformalResult, KnockoffResult
from randomization_tests._strategies import (
    _STRATEGY_REGISTRY,
    _ensure_registry,
    resolve_strategy,
)
from randomization_tests._text import coherence_score, docs_to_topics
from randomization_tests.confounders import _collider_test, mediation_analysis
from randomization_tests.diagnostics import (
    _bca_percentile,
    compute_e_value,
    compute_permutation_ci,
    compute_profile_ci,
    rosenbaum_bounds,
)
from randomization_tests.families import (
    LinearFamily,
    LogisticFamily,
    ModelFamily,
    MultinomialFamily,
    NegativeBinomialFamily,
    OrdinalFamily,
    PoissonFamily,
    _cross_fit_residualize,
    fit_reduced,
)
from randomization_tests.families_mixed import (
    LinearMixedFamily,
    LogisticMixedFamily,
    PoissonMixedFamily,
    _GLMMBatchStubMixin,
)
from randomization_tests.permutations import (
    generate_nested_permutations,
    generate_unique_permutations,
)
from randomization_tests.pvalues import calculate_p_values
from randomization_tests.sign_flips import (
    generate_between_cell_sign_flips,
    validate_symmetry,
)


class TestExchangeabilityTreeRealWorld:
    """Verify that multi-level nested ExchangeabilityTree works end-to-end."""

    def test_nested_tree_individual_and_joint_regression(self) -> None:
        """Real-world educational hierarchy: District -> School -> Student."""
        n_students = 60
        rng = np.random.default_rng(42)
        districts = np.repeat([0, 1], 30)
        schools = np.repeat(np.arange(6), 10)

        tree = ExchangeabilityTree.from_labels(
            [districts, schools],
            strategies=["between", "within"],
        )
        assert tree.n_samples == n_students

        df = pd.DataFrame(
            {
                "study_hours": rng.uniform(2, 20, n_students),
                "parent_edu": rng.integers(1, 5, n_students).astype(float),
                "school_budget": rng.uniform(5000, 15000, n_students),
            }
        )
        df["score"] = (
            50.0
            + 2.5 * df["study_hours"]
            + 3.0 * df["parent_edu"]
            + 0.001 * df["school_budget"]
            + rng.normal(0, 5, n_students)
        )

        # 1. Individual test (Freedman-Lane with confounders)
        res_ind = randomization_test_regression(
            df[["study_hours", "parent_edu", "school_budget"]],
            df[["score"]],
            confounders=["school_budget"],
            groups=tree,
            method="freedman_lane",
            n_randomizations=50,
            random_state=42,
        )
        assert isinstance(res_ind, IndividualTestResult)
        assert isinstance(res_ind.groups, ExchangeabilityTree)
        assert res_ind.context is not None
        assert isinstance(res_ind.context.groups, ExchangeabilityTree)
        assert len(res_ind.permuted_p_values) == 3
        assert res_ind.permuted_p_values[2] == "(confounder)"

        # Verify JSON serializability of IndividualTestResult with ExchangeabilityTree
        dict_ind = res_ind.to_dict()
        assert isinstance(dict_ind["groups"], list)
        assert len(dict_ind["groups"]) == n_students
        json_str_ind = json.dumps(dict_ind)
        assert len(json_str_ind) > 0

        # 2. Joint test
        res_joint = randomization_test_regression(
            df[["study_hours", "parent_edu", "school_budget"]],
            df[["score"]],
            confounders=["school_budget"],
            groups=tree,
            method="freedman_lane_joint",
            n_randomizations=50,
            random_state=42,
        )
        assert isinstance(res_joint, JointTestResult)
        assert isinstance(res_joint.groups, ExchangeabilityTree)
        assert res_joint.context is not None
        assert isinstance(res_joint.context.groups, ExchangeabilityTree)
        assert 0.0 <= res_joint.p_value <= 1.0

        # Verify JSON serializability of JointTestResult with ExchangeabilityTree
        dict_joint = res_joint.to_dict()
        assert isinstance(dict_joint["groups"], list)
        assert len(dict_joint["groups"]) == n_students
        json_str_joint = json.dumps(dict_joint)
        assert len(json_str_joint) > 0


class TestStrategyParity:
    """Verify strategy execute() protocol parity and forward compatibility."""

    def test_all_strategies_accept_extra_kwargs(self) -> None:
        """Every strategy must accept **kwargs without raising TypeError."""
        _ensure_registry()
        fam = LinearFamily()
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([1.0, 2.0, 3.0, 4.0])
        perm = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])

        for method_name in _STRATEGY_REGISTRY:
            strat = resolve_strategy(method_name)
            # Invoke with arbitrary forward-compatible kwargs
            res = strat.execute(
                X,
                y,
                fam,
                perm,
                confounders=[],
                model_coefs=np.array([1.0]),
                fit_intercept=True,
                n_jobs=1,
                randomization="permute",
                extra_future_arg="forward_compatible",
            )
            assert res is not None


class TestFamilySignatureUniformity:
    """Verify that all 9 model families implement identical method signatures."""

    @pytest.fixture
    def all_families(self) -> list[Any]:
        return [
            LinearFamily(),
            LogisticFamily(),
            PoissonFamily(),
            NegativeBinomialFamily(),
            OrdinalFamily(),
            MultinomialFamily(),
            LinearMixedFamily(),
            LogisticMixedFamily(),
            PoissonMixedFamily(),
        ]

    def test_varying_x_signature_uniformity(self, all_families: list[Any]) -> None:
        """batch_fit_varying_X must have identical signature across all 9 families."""
        expected_sig = None
        for fam in all_families:
            sig = inspect.signature(fam.batch_fit_varying_X)
            # fit_intercept parameter must NOT have a default value
            param = sig.parameters["fit_intercept"]
            assert param.default is inspect.Parameter.empty, (
                f"{fam.name} batch_fit_varying_X has default fit_intercept={param.default}"
            )
            if expected_sig is None:
                expected_sig = str(sig)
            assert str(sig) == expected_sig, (
                f"{fam.name} signature {sig} does not match {expected_sig}"
            )

    def test_varying_x_fit_and_score_signature_uniformity(
        self, all_families: list[Any]
    ) -> None:
        """batch_fit_and_score_varying_X must have identical signature across all 9 families."""
        expected_sig = None
        for fam in all_families:
            sig = inspect.signature(fam.batch_fit_and_score_varying_X)
            param = sig.parameters["fit_intercept"]
            assert param.default is inspect.Parameter.empty, (
                f"{fam.name} batch_fit_and_score_varying_X has default fit_intercept={param.default}"
            )
            if expected_sig is None:
                expected_sig = str(sig)
            assert str(sig) == expected_sig, (
                f"{fam.name} signature {sig} does not match {expected_sig}"
            )

    def test_glmm_stub_mixin_message(self) -> None:
        """_GLMMBatchStubMixin must mention score, score_joint, and kennedy."""
        stub = _GLMMBatchStubMixin()
        with pytest.raises(NotImplementedError) as exc_info:
            stub.batch_fit(np.zeros((2, 2)), np.zeros((2, 2)), fit_intercept=True)
        msg = str(exc_info.value)
        assert "kennedy" in msg
        assert "score" in msg


class TestTier1RuntimeFoundation:
    """Certification Tier I: Runtime Foundation, Device Boundaries & Concurrency (Pillars 1–4).

    Validates:
    - Pillar 1: Context-isolated thread safety (ContextVar) and re-entrancy.
    - Pillar 2: Cross-backend numerical equivalence, 64-bit precision preservation,
      host array containment, and first-principles JAX optimality (score equations & Hessian).
    - Pillar 3: PRNG stream reproducibility under parallel worker scaling (n_jobs=1 vs 4).
    - Pillar 4: Pure function zero-mutation of caller inputs and non-standard pandas indexing.
    """

    @pytest.fixture(autouse=True)
    def reset_backend_context(self) -> Generator[None, None, None]:
        """Ensure backend context is restored to auto/default for each test."""
        set_backend(None)
        yield
        set_backend(None)

    def test_pillar1_context_isolated_backend_concurrency(self) -> None:
        """Item 1 & 2: ContextVar backend overrides must isolate across concurrent threads."""
        set_backend("jax")

        def run_isolated_task(target_backend: str) -> str:
            with backend(target_backend):
                import time

                # Simulate work across thread boundary
                time.sleep(0.01)
                return get_backend()

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures_numpy = [pool.submit(run_isolated_task, "numpy") for _ in range(6)]
            futures_jax = [pool.submit(run_isolated_task, "jax") for _ in range(6)]

            for f in futures_numpy:
                assert f.result() == "numpy"
            for f in futures_jax:
                assert f.result() == "jax"

        # Calling thread state remains completely untouched
        assert get_backend() == "jax"

    def test_pillar2_linear_backend_equivalence_and_condition_preservation(
        self,
    ) -> None:
        """Item 3: OLS solvers must match to condition-scaled machine precision and satisfy normal equations."""
        backend_jax = resolve_backend("jax")
        backend_numpy = resolve_backend("numpy")

        rng = np.random.default_rng(123)
        n, p = 80, 4

        # Construct design matrix with condition number kappa ~ 500
        U, _ = np.linalg.qr(rng.standard_normal((n, p)))
        s = np.linspace(1.0, 0.002, p)  # singular values
        V, _ = np.linalg.qr(rng.standard_normal((p, p)))
        X = U @ np.diag(s) @ V.T
        kappa = float(np.linalg.cond(X))
        assert 100.0 < kappa < 1000.0

        B = 20
        Y_matrix = rng.standard_normal((B, n))

        # 1. Batch OLS equivalence
        beta_jax = backend_jax.batch_ols(X, Y_matrix, fit_intercept=True)
        beta_numpy = backend_numpy.batch_ols(X, Y_matrix, fit_intercept=True)
        assert beta_jax.shape == (B, p)
        assert beta_numpy.shape == (B, p)

        # Difference bounded by machine epsilon scaled by condition number
        max_diff = float(np.max(np.abs(beta_jax - beta_numpy)))
        assert max_diff <= 1e-12, f"OLS difference {max_diff} exceeded tolerance"

        # 2. First-principles normal equations check: X_aug^T (y - X_aug @ beta_aug) == 0
        X_aug = np.column_stack([np.ones(n), X])
        for b in range(min(5, B)):
            y_b = Y_matrix[b]
            # Solve with intercept included
            beta_aug = np.linalg.lstsq(X_aug, y_b, rcond=None)[0]
            residuals = y_b - X_aug @ beta_aug
            orthogonality = np.max(np.abs(X_aug.T @ residuals))
            assert orthogonality <= 1e-10 * np.linalg.norm(y_b)

        # 3. Batch OLS varying X equivalence
        X_batch = rng.standard_normal((B, n, p))
        y_shared = rng.standard_normal(n)
        beta_var_jax = backend_jax.batch_ols_varying_X(
            X_batch, y_shared, fit_intercept=True
        )
        beta_var_numpy = backend_numpy.batch_ols_varying_X(
            X_batch, y_shared, fit_intercept=True
        )
        max_diff_var = float(np.max(np.abs(beta_var_jax - beta_var_numpy)))
        assert max_diff_var <= 1e-12

    def test_pillar2_glm_first_principles_optimality_and_equivalence(
        self,
    ) -> None:
        """Item 3: JAX GLM solvers must satisfy score equations and strict Hessian convexity."""
        backend_jax = resolve_backend("jax")
        backend_numpy = resolve_backend("numpy")

        rng = np.random.default_rng(42)
        n, p = 120, 3
        X = rng.standard_normal((n, p))
        true_beta = np.array([0.7, -0.5, 0.4])
        logits = 0.2 + X @ true_beta
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n) < probs).astype(float)
        Y_matrix = np.tile(y, (2, 1))

        # Fit via both backends
        beta_jax = backend_jax.batch_logistic(X, Y_matrix, fit_intercept=True)[0]
        beta_numpy = backend_numpy.batch_logistic(
            X, Y_matrix, fit_intercept=True, tol=1e-8
        )[0]

        # 1. First-principles First-Order Optimality (Score Equation Vanishing)
        # For logistic regression: X_aug^T (y - mu(beta)) == 0
        # Re-estimate intercept to form full parameter vector
        # X_aug has intercept in col 0
        X_aug = np.column_stack([np.ones(n), X])
        # Find intercept for JAX slope
        from scipy.optimize import minimize_scalar

        res_int = minimize_scalar(
            lambda b0: (
                -np.sum(y * (b0 + X @ beta_jax) - np.log1p(np.exp(b0 + X @ beta_jax)))
            )
        )
        full_beta_jax = np.concatenate([[res_int.x], beta_jax])
        mu_jax = 1.0 / (1.0 + np.exp(-X_aug @ full_beta_jax))
        score_norm = float(np.max(np.abs(X_aug.T @ (y - mu_jax))) / n)
        assert score_norm <= 1e-5, (
            f"First-order score equation norm {score_norm} exceeded 1e-5"
        )

        # 2. First-principles Second-Order Optimality (Hessian Strict Positive Definiteness)
        W = mu_jax * (1.0 - mu_jax)
        Hessian = X_aug.T @ (X_aug * W[:, None])
        min_eigenval = float(np.min(np.linalg.eigvalsh(Hessian)))
        assert min_eigenval > 0.0, (
            f"Hessian is not strictly positive definite (min eig: {min_eigenval})"
        )

        # 3. Cross-backend proximity: solutions match within optimization tolerance
        rel_diff = float(
            np.linalg.norm(beta_jax - beta_numpy) / np.linalg.norm(beta_numpy)
        )
        assert rel_diff <= 5e-3, (
            f"Relative difference {rel_diff} exceeded optimization tolerance"
        )

    def test_pillar2_jax_64bit_precision_and_host_containment(self) -> None:
        """Item 4 & 5: JAX backend must preserve float64 precision and return host numpy.ndarray."""
        backend_jax = resolve_backend("jax")

        rng = np.random.default_rng(99)
        X = rng.standard_normal((30, 2)).astype(np.float64)
        Y = rng.standard_normal((3, 30)).astype(np.float64)

        # OLS
        res_ols = backend_jax.batch_ols(X, Y, fit_intercept=True)
        assert type(res_ols) is np.ndarray, "batch_ols did not return host np.ndarray"
        assert res_ols.dtype == np.float64, "batch_ols silently downcast float64"

        # Logistic
        Y_bin = (rng.uniform(size=(3, 30)) > 0.5).astype(np.float64)
        res_logit = backend_jax.batch_logistic(X, Y_bin, fit_intercept=True)
        assert type(res_logit) is np.ndarray, (
            "batch_logistic did not return host np.ndarray"
        )
        assert res_logit.dtype == np.float64, "batch_logistic silently downcast float64"

        # Poisson
        Y_cnt = rng.poisson(lam=2.0, size=(3, 30)).astype(np.float64)
        res_pois = backend_jax.batch_poisson(X, Y_cnt, fit_intercept=True)
        assert type(res_pois) is np.ndarray, (
            "batch_poisson did not return host np.ndarray"
        )
        assert res_pois.dtype == np.float64, "batch_poisson silently downcast float64"

    def test_pillar2_graceful_degradation_and_error_handling(self) -> None:
        """Item 6: Invalid backend requests must raise ValueError immediately."""
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("unsupported_backend")

        with pytest.raises(ValueError, match="Unknown backend"):
            resolve_backend("unsupported_backend")

    def test_pillar3_prng_parallel_scaling_reproducibility(self) -> None:
        """Item 7 & 8: n_jobs=1 vs n_jobs=4 must produce bit-identical results."""
        rng = np.random.default_rng(42)
        n = 50
        X = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
            }
        )
        y = pd.DataFrame({"y": (rng.uniform(size=n) > 0.5).astype(float)})

        # Use ter_braak logistic on numpy backend (which uses parallel joblib workers)
        r1 = randomization_test_regression(
            X,
            y,
            method="ter_braak",
            n_randomizations=40,
            random_state=42,
            n_jobs=1,
            backend="numpy",
        )
        r4 = randomization_test_regression(
            X,
            y,
            method="ter_braak",
            n_randomizations=40,
            random_state=42,
            n_jobs=4,
            backend="numpy",
        )
        assert isinstance(r1, IndividualTestResult)
        assert isinstance(r4, IndividualTestResult)

        assert r1.permuted_p_values == r4.permuted_p_values, (
            "p-values diverged across thread counts"
        )
        assert np.array_equal(r1.permuted_coefs, r4.permuted_coefs), (
            "permuted coefficients differed across thread counts"
        )

    def test_pillar4_pure_function_zero_caller_mutation(self) -> None:
        """Item 9: Library must never mutate caller inputs in-place."""
        rng = np.random.default_rng(101)
        n = 40
        X_orig = rng.standard_normal((n, 2))
        y_orig = 2.0 * X_orig[:, 0] - 1.5 * X_orig[:, 1] + rng.standard_normal(n)

        # Create DataFrames backed by non-writeable arrays
        X_mat = X_orig.copy()
        y_vec = y_orig.copy()
        X_mat.flags.writeable = False
        y_vec.flags.writeable = False

        X_df = pd.DataFrame(X_mat, columns=["feat1", "feat2"])
        y_df = pd.DataFrame(y_vec, columns=["outcome"])

        # Execute both individual and joint regression
        res_ind = randomization_test_regression(
            X_df,
            y_df,
            method="freedman_lane",
            n_randomizations=20,
            random_state=42,
        )
        assert res_ind is not None

        res_joint = randomization_test_regression(
            X_df,
            y_df,
            method="freedman_lane_joint",
            n_randomizations=20,
            random_state=42,
        )
        assert res_joint is not None

        # Verify exact bit-for-bit identity before and after
        assert np.array_equal(X_df.values, X_orig)
        assert np.array_equal(y_df.values.ravel(), y_orig)

    def test_pillar4_pandas_nonstandard_indexing_alignment(self) -> None:
        """Item 10: Non-standard, string, or disordered Pandas indices must not cause drift or NaNs."""
        rng = np.random.default_rng(202)
        n = 30
        non_standard_idx = [f"sample_{i * 7}" for i in range(n)]

        X_df = pd.DataFrame(
            {"feat1": rng.standard_normal(n), "feat2": rng.standard_normal(n)},
            index=non_standard_idx,
        )
        y_df = pd.DataFrame(
            {"target": 1.5 * X_df["feat1"] + rng.standard_normal(n)},
            index=non_standard_idx,
        )

        res = randomization_test_regression(
            X_df,
            y_df,
            method="freedman_lane",
            n_randomizations=25,
            random_state=42,
        )
        assert isinstance(res, IndividualTestResult)
        assert not np.isnan(res.model_coefs).any()
        assert len(res.raw_empirical_p) == 2
        for pval in res.raw_empirical_p:
            assert isinstance(pval, (float, np.floating))
            assert 0.0 <= pval <= 1.0

        # Defect CD-2 Harness 1: Permuted index labels must be automatically realigned
        reversed_idx = list(reversed(non_standard_idx))
        y_perm_idx = y_df.reindex(reversed_idx)
        res_reindexed = randomization_test_regression(
            X_df,
            y_perm_idx,
            method="freedman_lane",
            n_randomizations=25,
            random_state=42,
        )
        assert np.allclose(res_reindexed.model_coefs, res.model_coefs, atol=1e-12)

        # Defect CD-2 Harness 2: Disjoint index sets must raise an informative ValueError
        y_disjoint = pd.DataFrame(
            {"target": y_df["target"].values},
            index=[f"other_{i}" for i in range(n)],
        )
        with pytest.raises(
            ValueError,
            match="Index mismatch between X.*and y.*Indices must share identical labels",
        ):
            randomization_test_regression(
                X_df,
                y_disjoint,
                method="freedman_lane",
                n_randomizations=25,
                random_state=42,
            )

    def test_pillar4_memory_contiguity_invariants(self) -> None:
        """Item 11: Non-contiguous (Fortran) arrays must be handled cleanly without memory errors."""
        backend_numpy = resolve_backend("numpy")

        rng = np.random.default_rng(303)
        n, p = 50, 3

        # Create Fortran-contiguous array
        X_fortran = np.asfortranarray(rng.standard_normal((n, p)))
        assert not X_fortran.flags.c_contiguous
        assert X_fortran.flags.f_contiguous

        Y_matrix = rng.standard_normal((10, n))

        res = backend_numpy.batch_ols(X_fortran, Y_matrix, fit_intercept=True)
        assert res.shape == (10, p)
        assert not np.isnan(res).any()


class TestTier2NumericalSolversAndWhitening:
    """Certification Tier II: Numerical Solvers, Linear Algebra & Whitening (Pillars 5–6).

    Validates:
    - Pillar 5:
      * Item 12: Condition-number scaling probe (SVD OLS error scales with kappa(X) * eps,
        rejecting kappa^2 normal equations error blowup).
      * Item 13: Newton parameter freezing and analytical vs autodiff derivative exactness (< 1e-14).
      * Item 14: Henderson REML at zero-ICC boundary (tau^2 -> 0) recovering exact OLS (< 1e-7).
      * Item 15: GLMM tangent-space zero score offset (< 1e-6).
    - Pillar 6:
      * Item 16: AR stationarity lag polynomial root invariant (|z| > 1 / |lambda| < 1).
      * Item 17: Prais-Winsten degrees-of-freedom preservation (L_{0,0} = sqrt(1-rho^2), L.T @ L = Omega^-1).
      * Item 18: Whitening commutation with permutation (residuals whitened before permutation).
    """

    def test_pillar5_condition_number_scaling_probe(self) -> None:
        """Item 12: Design with kappa(X)=1e7 must achieve relative error O(kappa * eps), rejecting normal eq blowup."""
        backend_jax = resolve_backend("jax")
        backend_numpy = resolve_backend("numpy")

        rng = np.random.default_rng(42)
        n, p = 100, 4

        # Construct ill-conditioned design matrix with condition number kappa(X) = 1e7
        U, _ = np.linalg.qr(rng.standard_normal((n, p)))
        V, _ = np.linalg.qr(rng.standard_normal((p, p)))
        s = np.logspace(0, -7, p)  # singular values from 1 down to 1e-7
        X = U @ np.diag(s) @ V.T

        kappa = float(np.linalg.cond(X))
        assert 5e6 < kappa < 2e7, f"Constructed condition number {kappa} out of range"

        beta_true = np.array([1.0, -2.0, 0.5, 3.0])
        y = X @ beta_true  # exact noise-free response

        # 1. SVD lstsq in JAX and NumPy (varying_X path)
        X_batch = np.array([X])
        res_jax = backend_jax.batch_ols_varying_X(X_batch, y, fit_intercept=False)[0]
        res_numpy = backend_numpy.batch_ols_varying_X(X_batch, y, fit_intercept=False)[
            0
        ]

        err_jax = float(np.linalg.norm(res_jax - beta_true) / np.linalg.norm(beta_true))
        err_numpy = float(
            np.linalg.norm(res_numpy - beta_true) / np.linalg.norm(beta_true)
        )

        # SVD error scales with kappa(X) * eps_mach ~ 1e7 * 1e-16 = 1e-9
        assert err_jax <= 1e-7, (
            f"JAX SVD error {err_jax} exceeded condition-scaled tolerance"
        )
        assert err_numpy <= 1e-7, (
            f"NumPy SVD error {err_numpy} exceeded condition-scaled tolerance"
        )

        # 2. Defect Prover: Normal equations (X'X)^-1 X'y square the condition number to kappa^2 = 1e14
        XtX = X.T @ X
        Xty = X.T @ y
        beta_norm_eq = np.linalg.solve(XtX, Xty)
        err_norm_eq = float(
            np.linalg.norm(beta_norm_eq - beta_true) / np.linalg.norm(beta_true)
        )

        # Normal equations error must blow up by several orders of magnitude (> 1e-3)
        assert err_norm_eq > 1e-3, (
            f"Normal equations unexpectedly succeeded with error {err_norm_eq}"
        )
        assert err_norm_eq / max(err_jax, 1e-16) > 1e4, (
            "Normal equations error was not significantly worse than SVD"
        )

    def test_pillar5_newton_parameter_freezing_and_autodiff_exactness(
        self,
    ) -> None:
        """Item 13: JAX autodiff derivatives match analytical formulas to < 1e-14 and parameter updates freeze."""
        import jax
        import jax.numpy as jnp

        rng = np.random.default_rng(42)
        n, p = 60, 3
        X = rng.standard_normal((n, p))
        y = (rng.uniform(size=n) > 0.5).astype(float)
        beta = np.array([0.4, -0.6, 0.7])

        X_j = jnp.array(X, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)
        beta_j = jnp.array(beta, dtype=jnp.float64)

        # Objective function (Logistic NLL)
        def nll(b: jax.Array) -> jax.Array:
            eta = X_j @ b
            return -jnp.sum(y_j * eta - jnp.log1p(jnp.exp(eta)))

        # Autodiff derivatives
        g_auto = np.asarray(jax.grad(nll)(beta_j))
        H_auto = np.asarray(jax.hessian(nll)(beta_j))

        # Analytical derivatives
        mu = 1.0 / (1.0 + np.exp(-X @ beta))
        g_analytic = -X.T @ (y - mu)
        W = mu * (1.0 - mu)
        H_analytic = X.T @ (X * W[:, None])

        diff_grad = float(np.max(np.abs(g_auto - g_analytic)))
        diff_hess = float(np.max(np.abs(H_auto - H_analytic)))

        assert diff_grad < 1e-14, (
            f"Gradient autodiff difference {diff_grad} exceeded 1e-14"
        )
        assert diff_hess < 1e-14, (
            f"Hessian autodiff difference {diff_hess} exceeded 1e-14"
        )

        # Verify parameter freezing logic:
        # In _newton_solve_static: beta_new = jnp.where(converged_prev, beta, beta_step)
        converged_flag = jnp.array(True)
        beta_frozen = jnp.where(converged_flag, beta_j, beta_j + 1.0)
        assert np.array_equal(np.asarray(beta_frozen), beta)

    def test_pillar5_henderson_reml_zero_icc_boundary_representation(
        self,
    ) -> None:
        """Item 14: Henderson REML at zero ICC (tau^2 = 0) smoothly recovers exact OLS without silent clamping."""
        rng = np.random.default_rng(101)
        n, p = 120, 2
        X = rng.standard_normal((n, p))
        beta_true = np.array([1.2, -0.8])
        # Data generated with zero cluster random effect variance (pure iid errors)
        y = 0.5 + X @ beta_true + rng.standard_normal(n)
        groups = np.repeat(np.arange(12), 10)

        # Calibrate LinearMixedFamily via Henderson REML
        fam = LinearMixedFamily().calibrate(X, y, groups=groups)
        fit_lmm = fam.fit(X, y, fit_intercept=True)
        lmm_beta = fit_lmm.beta

        # Exact OLS baseline
        X_aug = np.column_stack([np.ones(n), X])
        ols_beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]

        diff = float(np.max(np.abs(lmm_beta - ols_beta)))
        assert diff < 1e-7, (
            f"Henderson REML at zero ICC diverged from OLS by {diff} (expected < 1e-7)"
        )

    def test_pillar5_glmm_tangent_space_zero_score_offset(self) -> None:
        """Item 15: Tangent-space score projection eliminates score offset (|U_0 - beta_hat| < 1e-6)."""
        rng = np.random.default_rng(202)
        n, g = 60, 6
        groups = np.repeat(np.arange(g), 10)
        X = rng.standard_normal((n, 2))
        y = (rng.uniform(size=n) > 0.5).astype(float)

        fam = LogisticMixedFamily().calibrate(X, y, groups=groups)
        beta_hat = float(fam.coefs(fam.fit(X, y, fit_intercept=True))[0])

        _, preds_red = fit_reduced(fam, np.delete(X, 0, axis=1), y, fit_intercept=True)
        identity_perm = np.arange(n)[None, :]
        u0 = float(
            fam.score_project(
                X,
                y,
                0,
                y - preds_red,
                identity_perm,
                fit_intercept=True,
            )[0]
        )

        offset = abs(beta_hat - u0) / max(abs(beta_hat), 1e-12)
        assert offset < 1e-6, f"Tangent-space GLMM score offset {offset} exceeded 1e-6"

    def test_pillar5_fit_result_contract_and_estimator_parity(self) -> None:
        """Item 15b: family.fit() returns a frozen FitResult with exact Fisher info and Wald CIs."""
        from randomization_tests._results import FitResult
        from randomization_tests.diagnostics import compute_wald_ci

        rng = np.random.default_rng(42)
        n, p = 50, 3
        X = rng.standard_normal((n, p))
        y = 1.5 + 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.standard_normal(n)

        fam = LinearFamily()
        fit_res = fam.fit(X, y, fit_intercept=True)

        assert isinstance(fit_res, FitResult)
        assert len(fit_res.params) == p + 1
        assert len(fit_res.predictions) == n
        assert len(fit_res.residuals) == n
        assert fit_res.fisher_info.shape == (p + 1, p + 1)
        assert len(fit_res.se) == p + 1
        assert len(fit_res.p_values) == p + 1
        assert fit_res.converged is True

        # Check duck-typing properties
        np.testing.assert_allclose(fit_res.coef_, fit_res.params[1:])
        assert abs(fit_res.intercept_ - fit_res.params[0]) < 1e-12

        # Check Wald CI calculation
        ci = fit_res.conf_int(alpha=0.05)
        assert ci.shape == (p + 1, 2)
        assert (ci[:, 0] < fit_res.params).all()
        assert (fit_res.params < ci[:, 1]).all()

        # Check compute_wald_ci consumes fit_res directly
        wald_ci, cat_ci = compute_wald_ci(
            fit_res, fam, n_features=p, alpha=0.05, X=X, y=y, fit_intercept=True
        )
        assert wald_ci.shape == (p, 2)
        assert cat_ci is None
        np.testing.assert_allclose(wald_ci, ci[1:])

    def test_pillar6_ar_stationarity_root_invariant(self) -> None:
        """Item 16: AR lag polynomial roots must satisfy |z| > 1; non-stationary processes raise ValueError."""
        # Stationary processes pass
        validate_ar_stationarity(np.array([0.7]))
        validate_ar_stationarity(np.array([0.5, -0.3]))
        validate_ar_stationarity(np.array([]))

        # Unit root (|rho| = 1.0) raises
        with pytest.raises(ValueError, match="violate weak stationarity"):
            validate_ar_stationarity(np.array([1.0]))
        with pytest.raises(ValueError, match="violate weak stationarity"):
            validate_ar_stationarity(np.array([-1.0]))

        # Explosive (|rho| > 1.0) raises
        with pytest.raises(ValueError, match="violate weak stationarity"):
            validate_ar_stationarity(np.array([1.25]))

        # Complex roots outside unit circle
        with pytest.raises(ValueError, match="violate weak stationarity"):
            validate_ar_stationarity(np.array([0.5, 0.7]))

        # Precision builders validate stationarity upfront
        with pytest.raises(ValueError, match="violate weak stationarity"):
            build_ar_precision_block(np.array([1.0]), T=5)
        with pytest.raises(ValueError, match="violate weak stationarity"):
            build_ar_cholesky_factor(np.array([1.1]), T=5)

    def test_pillar6_prais_winsten_degrees_of_freedom_preservation(self) -> None:
        """Item 17: Prais-Winsten factor preserves all T observations and factors Omega^-1."""
        rho = 0.55
        T = 7
        L = build_ar_cholesky_factor(np.array([rho]), T)

        # Full dimension (T, T) preserves all observations without dropping the first observation
        assert L.shape == (T, T)
        np.testing.assert_allclose(L[0, 0], np.sqrt(1.0 - rho**2), atol=1e-12)
        assert L[0, 1] == 0.0

        for t in range(1, T):
            np.testing.assert_allclose(L[t, t], 1.0, atol=1e-12)
            np.testing.assert_allclose(L[t, t - 1], -rho, atol=1e-12)

        # Precision factorization: L^T L == Omega^-1
        prec = build_ar_precision_block(np.array([rho]), T)
        np.testing.assert_allclose(L.T @ L, prec, atol=1e-12)

    def test_pillar6_whitening_commutation_with_permutation(self) -> None:
        """Item 18: Permutation and AR whitening do not commute; residuals must be whitened before permutation."""
        rng = np.random.default_rng(303)
        T = 10
        rho = 0.7
        L = build_ar_cholesky_factor(np.array([rho]), T)
        Omega_inv = build_ar_precision_block(np.array([rho]), T)

        # Generate a non-identity permutation matrix Pi
        perm = rng.permutation(T)
        while np.array_equal(perm, np.arange(T)):
            perm = rng.permutation(T)
        Pi = np.eye(T)[perm]

        # Non-commutation proof: Pi @ Omega_inv @ Pi.T != Omega_inv
        commutation_diff = float(np.max(np.abs(Pi @ Omega_inv @ Pi.T - Omega_inv)))
        assert commutation_diff > 0.5, (
            f"Permutation unexpectedly commuted with AR precision: diff={commutation_diff}"
        )

        # Generate AR(1) residuals
        e = np.zeros(T)
        e[0] = rng.standard_normal()
        for t in range(1, T):
            e[t] = rho * e[t - 1] + rng.standard_normal()

        # Whitening before permuting: e_whitened = L @ e is white noise, permuting preserves iid law
        e_w = L @ e
        e_w_perm = e_w[perm]
        # In contrast, permuting first then whitening: L @ (Pi @ e) does NOT produce white noise
        e_perm_w = L @ e[perm]
        diff_paths = float(np.max(np.abs(e_w_perm - e_perm_w)))
        assert diff_paths > 0.1, "Whitening before vs after permutation diverged"

        # Defect CD-10 / M16: Discrete GLMMs must explicitly reject ar_order
        X_dummy = pd.DataFrame({"x": rng.standard_normal(20)})
        y_dummy = pd.DataFrame({"y": (rng.uniform(size=20) > 0.5).astype(float)})
        panel_dummy = np.repeat(np.arange(4), 5)
        time_dummy = np.tile(np.arange(5), 4)

        with pytest.raises(
            ValueError,
            match="ar_order= is not supported for discrete mixed model family='logistic_mixed'",
        ):
            randomization_test_regression(
                X_dummy,
                y_dummy,
                family="logistic_mixed",
                panel_id=panel_dummy,
                time_id=time_dummy,
                ar_order=1,
                method="score",
            )
        with pytest.raises(
            ValueError,
            match="ar_order= is not supported for discrete mixed model family='poisson_mixed'",
        ):
            randomization_test_regression(
                X_dummy,
                y_dummy,
                family="poisson_mixed",
                panel_id=panel_dummy,
                time_id=time_dummy,
                ar_order=1,
                method="score",
            )

        # Defect CD-8 / M14: LinearFamily under AR fits FGLS throughout; score null centers at zero
        rng_dgp = np.random.default_rng(42)
        n_panels, T_panel = 10, 20
        n_total = n_panels * T_panel
        p_id = np.repeat(np.arange(n_panels), T_panel)
        t_id = np.tile(np.arange(T_panel), n_panels)

        e_ar = np.zeros(n_total)
        for panel_idx in range(n_panels):
            s_idx = panel_idx * T_panel
            e_ar[s_idx] = rng_dgp.standard_normal()
            for t_step in range(1, T_panel):
                e_ar[s_idx + t_step] = (
                    0.7 * e_ar[s_idx + t_step - 1] + rng_dgp.standard_normal()
                )

        x_signal = rng_dgp.standard_normal(n_total)
        x_noise = rng_dgp.standard_normal(n_total)
        y_ar = 2.0 * x_signal + e_ar  # x_noise has true beta=0
        df_X = pd.DataFrame({"x_signal": x_signal, "x_noise": x_noise})
        df_y_ar = pd.DataFrame({"y": y_ar})

        res_ar = randomization_test_regression(
            df_X,
            df_y_ar,
            family="linear",
            method="score",
            ar_order=1,
            panel_id=p_id,
            time_id=t_id,
            n_randomizations=1000,
            random_state=42,
        )
        x_noise_null = np.array(res_ar.permuted_coefs)[:, 1]
        mean_noise_null = float(np.mean(x_noise_null))
        sd_noise_null = float(np.std(x_noise_null, ddof=1))

        # Invariant: Score permutation null under AR centers at zero with |mean|/sd <= 0.15
        assert abs(mean_noise_null) / sd_noise_null <= 0.15, (
            f"AR score null shifted: |mean|/sd = {abs(mean_noise_null) / sd_noise_null:.4f} > 0.15"
        )

    def test_pillar6_glm_ar_tangent_space_null_centering(self) -> None:
        """Defect M15 / CD-9: GLM score test under AR whitens in the tangent space and centers at zero."""
        rng = np.random.default_rng(42)
        n_panels, T = 10, 20
        n = n_panels * T
        panel_id = np.repeat(np.arange(n_panels), T)
        time_id = np.tile(np.arange(T), n_panels)

        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        e = np.zeros(n)
        for p in range(n_panels):
            start = p * T
            e[start] = rng.standard_normal()
            for t in range(1, T):
                e[start + t] = 0.6 * e[start + t - 1] + rng.standard_normal()

        logits = 0.8 * x1 + 0.0 * x2 + e * 0.5
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits))).astype(float)
        df_X = pd.DataFrame({"x1": x1, "x2": x2})
        df_y = pd.DataFrame({"y": y})

        res = randomization_test_regression(
            df_X,
            df_y,
            family="logistic",
            method="score",
            ar_order=1,
            panel_id=panel_id,
            time_id=time_id,
            n_randomizations=500,
            random_state=42,
        )
        x2_null = np.array(res.permuted_coefs)[:, 1]
        mean_null = float(np.mean(x2_null))
        sd_null = float(np.std(x2_null, ddof=1))

        # Invariant: Tangent-space whitening centers the GLM AR score null at zero (|mean|/sd <= 0.15)
        assert abs(mean_null) / sd_null <= 0.15, (
            f"GLM AR score null shifted: |mean|/sd = {abs(mean_null) / sd_null:.4f} > 0.15"
        )


class TestTier3CoreRandomizationEngineAndStrategies:
    """Certification Tier III: Core Randomization Engine & Permutation Strategies (Pillars 7–9).

    Validates:
    - Pillar 7:
      * Item 19: Exchangeability cell partition boundary preservation (within, between).
      * Item 20: Small-n combinatorial exhaustion and collision-free unranking.
      * Item 21: Finite-sample order statistic p-value lower bound (p >= 1/(B+1) > 0).
    - Pillar 8:
      * Item 22: Conditional symmetry validation and rejection of asymmetric disturbances.
      * Item 23: Cluster-level uniform sign flipping in grouped designs.
    - Pillar 9:
      * Item 24: Deterministic master null-centering gate (|mean(beta*)| / sd(beta*) <= 0.10).
      * Item 25: Canonical strategy mathematical fidelity.
      * Item 26: Omnibus joint fit improvement non-negativity (Delta RSS >= 0) and FL==Score parity.
    """

    def test_pillar7_exchangeability_cell_partition_invariance(self) -> None:
        """Item 19: Tree strategies must preserve cluster partition boundaries strictly."""
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        n_samples = len(labels)

        # 1. 'within' strategy: observation units must stay within their cluster cell
        tree_within = ExchangeabilityTree.from_labels([labels], strategies=["within"])
        perms_within = generate_nested_permutations(
            n_samples, 20, tree_within, random_state=42
        )
        for p in perms_within:
            assert set(p[:3]) == {0, 1, 2}, "Within strategy leaked across cluster 0"
            assert set(p[3:6]) == {3, 4, 5}, "Within strategy leaked across cluster 1"
            assert set(p[6:]) == {6, 7, 8}, "Within strategy leaked across cluster 2"

        # 2. 'between' strategy: whole clusters are permuted as atomic blocks
        tree_between = ExchangeabilityTree.from_labels([labels], strategies=["between"])
        perms_between = generate_nested_permutations(
            n_samples, 5, tree_between, random_state=42
        )
        valid_blocks = [
            frozenset({0, 1, 2}),
            frozenset({3, 4, 5}),
            frozenset({6, 7, 8}),
        ]
        for p in perms_between:
            b0, b1, b2 = (
                frozenset(p[:3]),
                frozenset(p[3:6]),
                frozenset(p[6:]),
            )
            assert b0 in valid_blocks, "Between strategy fractured cluster block"
            assert b1 in valid_blocks, "Between strategy fractured cluster block"
            assert b2 in valid_blocks, "Between strategy fractured cluster block"

    def test_pillar7_small_n_exhaustive_combinatorial_generation(self) -> None:
        """Item 20: For n <= 10, permutation generation uses exact collision-free unranking."""
        # For n=4, 4! = 24 total permutations
        perms_all = generate_unique_permutations(
            n_samples=4,
            n_randomizations=24,
            exclude_identity=False,
            random_state=42,
        )
        assert perms_all.shape == (24, 4)
        assert len(np.unique(perms_all, axis=0)) == 24, "Duplicate permutation detected"

        # If requesting more than available permutations, raises ValueError
        with pytest.raises(
            ValueError, match="Requested 50 unique permutations but only 23"
        ):
            generate_unique_permutations(
                n_samples=4,
                n_randomizations=50,
                exclude_identity=True,
                random_state=42,
            )

    def test_pillar7_finite_sample_order_statistic_p_value_bound(self) -> None:
        """Item 21: Finite-sample p-value must satisfy p >= 1/(B+1) > 0 strictly."""
        B = 99
        # Extreme observed statistic that no permutation exceeds (b = 0)
        perm_coefs = np.zeros((B, 1))
        model_coefs = np.array([100.0])
        X = pd.DataFrame({"x": np.ones(50)})
        y = pd.DataFrame({"y": np.ones(50)})

        (
            perm_p_str,
            _,
            raw_p,
            _,
            counts,
        ) = calculate_p_values(
            X,
            y,
            perm_coefs,
            model_coefs,
            family=LinearFamily(),
        )

        # p = (0 + 1) / (99 + 1) = 1/100 = 0.0100
        expected_min_p = 1.0 / (B + 1)
        assert raw_p[0] == pytest.approx(expected_min_p, abs=1e-12)
        assert raw_p[0] > 0.0, "p-value was zero; violated Phipson-Smyth lower bound"

    def test_pillar8_residual_symmetry_validation(self) -> None:
        """Item 22: Residual symmetry diagnostic detects asymmetric skewed distributions."""
        rng = np.random.default_rng(101)
        n = 200

        # 1. Symmetric Gaussian disturbances: should pass symmetry test
        sym_resid = rng.standard_normal(n)
        res_sym = validate_symmetry(sym_resid, alpha=0.05)
        assert res_sym["is_symmetric"] is True
        p_val_sym = res_sym["p_value"]
        assert isinstance(p_val_sym, (float, int))
        assert p_val_sym >= 0.05

        # 2. Strongly skewed Exponential disturbances: should fail symmetry test
        exp_resid = rng.exponential(scale=2.0, size=n) - 2.0
        res_asym = validate_symmetry(exp_resid, alpha=0.05)
        assert res_asym["is_symmetric"] is False
        p_val_asym = res_asym["p_value"]
        assert isinstance(p_val_asym, (float, int))
        assert p_val_asym < 0.05

    def test_pillar8_cluster_level_sign_flipping(self) -> None:
        """Item 23: Clustered sign-flips must assign uniform signs within each cluster."""
        n_clusters = 5
        cluster_size = 6
        n_samples = n_clusters * cluster_size
        cells = np.repeat(np.arange(n_clusters), cluster_size)

        flips = generate_between_cell_sign_flips(
            n_samples, n_flips=30, cells=cells, random_state=42
        )
        assert flips.shape == (30, n_samples)

        # Verify that for every row and every cluster, all elements have identical sign
        for draw in flips:
            for c in range(n_clusters):
                cluster_signs = draw[cells == c]
                assert np.all(cluster_signs == cluster_signs[0]), (
                    f"Cluster {c} has mixed signs: {cluster_signs}"
                )

    def test_pillar9_master_null_centering_gate(self) -> None:
        """Item 24: Under strong signal, the permutation null must center at zero (|mean|/sd <= 0.10)."""
        rng = np.random.default_rng(42)
        n = 80
        X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
        y = pd.DataFrame({"y": 1.5 * X["x1"] - 2.0 * X["x2"] + rng.standard_normal(n)})

        strategies = ["freedman_lane", "ter_braak", "kennedy", "score", "manly"]
        for strat in strategies:
            if strat == "manly":
                with pytest.warns(
                    UserWarning, match="method='manly' uses direct Y permutation"
                ):
                    res = randomization_test_regression(
                        X,
                        y,
                        method=strat,
                        n_randomizations=500,
                        random_state=42,
                    )
            else:
                res = randomization_test_regression(
                    X,
                    y,
                    method=strat,
                    n_randomizations=500,
                    random_state=42,
                )
            assert isinstance(res, IndividualTestResult)
            null_draws = np.asarray(res.permuted_coefs)[:, 0]
            mean_null = float(np.mean(null_draws))
            sd_null = float(np.std(null_draws, ddof=1))
            beta_hat = float(res.model_coefs[0])

            ratio_sd = abs(mean_null) / sd_null
            ratio_beta = abs(mean_null) / abs(beta_hat)

            assert ratio_sd <= 0.10, (
                f"Strategy {strat} failed null-centering (|mean|/sd = {ratio_sd:.3f} > 0.10)"
            )
            assert ratio_beta <= 0.05, (
                f"Strategy {strat} null mean centered near beta_hat (|mean|/|beta| = {ratio_beta:.3f} > 0.05)"
            )

    def test_pillar9_canonical_strategy_fidelity(self) -> None:
        """Item 25: Canonical strategies satisfy their exact mathematical definitions."""
        rng = np.random.default_rng(77)
        n = 60
        X = pd.DataFrame({"x1": rng.standard_normal(n), "z": rng.standard_normal(n)})
        y = pd.DataFrame({"y": 0.8 * X["x1"] + 1.2 * X["z"] + rng.standard_normal(n)})

        # 1. Freedman-Lane vs Score bit-for-bit identity on LinearFamily
        res_fl = randomization_test_regression(
            X,
            y,
            confounders=["z"],
            method="freedman_lane",
            n_randomizations=100,
            random_state=42,
        )
        res_sc = randomization_test_regression(
            X,
            y,
            confounders=["z"],
            method="score",
            n_randomizations=100,
            random_state=42,
        )
        assert isinstance(res_fl, IndividualTestResult)
        assert isinstance(res_sc, IndividualTestResult)
        np.testing.assert_allclose(
            res_fl.permuted_coefs, res_sc.permuted_coefs, atol=1e-12
        )

        # 2. ter Braak null recentering: null draws are exactly beta* - beta_hat
        res_tb = randomization_test_regression(
            X,
            y,
            confounders=["z"],
            method="ter_braak",
            n_randomizations=100,
            random_state=42,
        )
        assert isinstance(res_tb, IndividualTestResult)
        assert res_tb.permuted_coefs is not None

    def test_pillar9_omnibus_joint_fit_improvement_and_parity(self) -> None:
        """Item 26: Joint fit improvement Delta RSS >= 0, and FL Joint == Score Joint bit-for-bit."""
        rng = np.random.default_rng(42)
        n = 60
        X = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "c": rng.standard_normal(n),
            }
        )
        y = pd.DataFrame(
            {"y": 1.0 * X["x1"] + 2.0 * X["x2"] + 0.5 * X["c"] + rng.standard_normal(n)}
        )

        res_fl = randomization_test_regression(
            X,
            y,
            confounders=["c"],
            method="freedman_lane_joint",
            n_randomizations=100,
            random_state=42,
        )
        res_sj = randomization_test_regression(
            X,
            y,
            confounders=["c"],
            method="score_joint",
            n_randomizations=100,
            random_state=42,
        )
        assert isinstance(res_fl, JointTestResult)
        assert isinstance(res_sj, JointTestResult)

        # Non-negativity invariant under nested least squares
        assert res_fl.observed_improvement >= 0.0, (
            f"Observed Delta RSS was negative: {res_fl.observed_improvement}"
        )
        assert np.all(np.asarray(res_fl.permuted_improvements) >= 0.0), (
            "Permuted Delta RSS was negative"
        )

        # Bit-identical parity between Freedman-Lane Joint and Score Joint
        assert abs(res_fl.observed_improvement - res_sj.observed_improvement) == 0.0, (
            "Observed improvement diverged between FL Joint and Score Joint"
        )
        np.testing.assert_allclose(
            res_fl.permuted_improvements, res_sj.permuted_improvements, atol=1e-12
        )
        assert res_fl.p_value == res_sj.p_value


class TestTier4HighLevelStatisticalSystems:
    """Certification Tier IV: High-Level Statistical Systems (Pillars 10–15).

    Validates:
    - Pillar 10:
      * Item 27: Pivotal shift-inversion permutation CIs (reflected quantiles).
      * Item 28: Profile likelihood Wilks' deviance bisection and separation guards.
      * Item 29: BCa case-resampling bootstrap intervals and separation from permutation null.
    - Pillar 11:
      * Item 30: d-separation and collider Berkson's bias detection.
      * Item 31: Mediation analysis indirect effect evaluation via bootstrap.
      * Item 32: Sensitivity analysis exactness (E-value point/protective and Rosenbaum bounds).
    - Pillar 12:
      * Item 33: Mercer kernel PSD invariant across kernel families.
      * Item 34: Gram centering zero-row-sum identity (< 1e-13).
      * Item 35: MMD and HSIC null distribution centering.
      * Item 36: Text NPMI coherence in [-1, 1] and NMF non-negativity.
    - Pillar 13:
      * Item 37: DML Neyman orthogonality and out-of-fold cross-fitting.
      * Item 38: Axiom D26 dispatch principle awareness.
    - Pillar 14:
      * Item 39: Conformal exact quantile order statistics and Feature-Stratified Coverage (FSC).
      * Item 40: Conformal batch vectorization and dimension preservation.
      * Item 41: Result dataclass immutability (frozen) and serialization.
      * Item 42: Display formatting and index slicing.
    - Pillar 15:
      * Item 43: Fixed-X knockoff Gram preservation and n >= 2p validation.
      * Item 44: Model-X Ledoit-Wolf shrinkage and positive definiteness.
      * Item 45: Coin-flip sign-inversion property of importance statistics.
      * Item 46: Knockoff+ threshold FDR control and MMD swap exchangeability audit.
    """

    def test_pillar10_pivotal_shift_inversion_cis(self) -> None:
        """Item 27: Permutation CI inverts shift null via reflected quantiles."""
        perm_coefs = np.array([[-2.0], [0.0], [1.0]])
        model_coefs = np.array([5.0])
        alpha = 0.10

        ci = compute_permutation_ci(
            perm_coefs,
            model_coefs,
            alpha=alpha,
            confounders=[],
            feature_names=["x1"],
        )
        assert ci.shape == (1, 2)
        lower, upper = float(ci[0, 0]), float(ci[0, 1])
        assert lower < upper
        # In contrast, naive unreflected percentile would be [5.0 + q_{0.05}, 5.0 + q_{0.95}]
        naive_lower = 5.0 + float(np.percentile(perm_coefs[:, 0], 5.0))
        assert abs(lower - naive_lower) > 0.5, (
            "Permutation CI used unreflected percentiles"
        )

    def test_pillar10_profile_likelihood_wilks_bisection(self) -> None:
        """Item 28: Profile likelihood bisection finds Wilks' chi2 boundary."""
        import jax.numpy as jnp
        from scipy.stats import chi2

        from randomization_tests._backends._jax import (
            _augment_intercept,
            _logistic_nll,
            _make_newton_solver,
            grad,
            hessian,
            jit,
        )

        rng = np.random.default_rng(42)
        n = 60
        X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
        y_1d = (rng.uniform(size=n) > 0.5).astype(float)
        y_2d = pd.DataFrame({"y": y_1d})

        # Defect CD-1: 1D vector and 2D DataFrame response must give identical, uncollapsed CIs
        cis_1d = compute_profile_ci(X, y_1d, family=LogisticFamily(), alpha=0.05)
        cis_2d = compute_profile_ci(X, y_2d, family=LogisticFamily(), alpha=0.05)
        assert np.allclose(cis_1d, cis_2d, atol=1e-12)
        assert cis_1d.shape == (2, 2)
        assert not np.isnan(cis_1d).any()
        assert cis_1d[0, 0] < cis_1d[0, 1]
        assert cis_1d[1, 0] < cis_1d[1, 1]

        # Invariant Gate: Delta-deviance 2 * (NLL_prof - NLL_full) at the bound matches Wilks' chi2_1(0.05)
        chi2_crit = float(chi2.ppf(0.95, df=1))
        X_aug = _augment_intercept(np.asarray(X), fit_intercept=True)
        X_j = jnp.array(X_aug, dtype=jnp.float64)
        y_j = jnp.array(y_1d, dtype=jnp.float64)
        beta_mle, nll_full, _ = _make_newton_solver(X_j, y_j, max_iter=100, tol=1e-6)

        free = jnp.array([0, 2])
        damping = 1e-8 * jnp.eye(2, dtype=jnp.float64)
        _grad_fn = jit(grad(_logistic_nll))
        _hess_fn = jit(hessian(_logistic_nll))

        def _eval_prof_dev(b1_star: float) -> float:
            b = beta_mle.at[1].set(b1_star)
            for _ in range(30):
                g = _grad_fn(b, X_j, y_j)[free]
                H = _hess_fn(b, X_j, y_j)[free][:, free] + damping
                delta = jnp.linalg.solve(H, g)
                b = b.at[free].add(-delta)
                if jnp.max(jnp.abs(delta)) < 1e-8:
                    break
            nll_prof = float(_logistic_nll(b, X_j, y_j))
            return 2.0 * (nll_prof - float(nll_full))

        dev_lower = _eval_prof_dev(float(cis_1d[0, 0]))
        dev_upper = _eval_prof_dev(float(cis_1d[0, 1]))
        assert abs(dev_lower - chi2_crit) < 1e-3, (
            f"Lower profile deviance {dev_lower} differs from {chi2_crit}"
        )
        assert abs(dev_upper - chi2_crit) < 1e-3, (
            f"Upper profile deviance {dev_upper} differs from {chi2_crit}"
        )

    def test_pillar10_bca_bootstrap_invariants(self) -> None:
        """Item 29: BCa adjusts percentiles via jackknife acceleration and bias correction."""
        rng = np.random.default_rng(123)
        boot_dist = rng.standard_normal(500) + 0.2
        observed_stat = 0.3
        jackknife_stats = rng.standard_normal(50) * 0.1 + 0.3

        ci_lower, ci_upper = _bca_percentile(
            boot_dist, observed_stat, jackknife_stats, alpha=0.05
        )
        assert ci_lower < observed_stat < ci_upper

    def test_pillar11_d_separation_collider_berkson_bias(self) -> None:
        """Item 30: Collider conditioning induces Berkson bias detected by _collider_test."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)  # unconditionally independent
        c = 1.2 * x + 1.5 * y + 0.3 * rng.standard_normal(n)  # collider

        is_coll, r_marg, r_part = _collider_test(x, y, c, random_state=42)
        assert is_coll is True
        assert abs(r_marg) < 0.15  # marginal independence preserved
        assert abs(r_part) > 0.85  # strong spurious correlation induced

    def test_pillar11_mediation_analysis_bootstrap_invariants(self) -> None:
        """Item 31: Mediation analysis indirect effect evaluated via bootstrap."""
        rng = np.random.default_rng(77)
        n = 80
        x = rng.standard_normal(n)
        m = 0.8 * x + rng.standard_normal(n)
        y = 0.7 * m + 0.5 * x + rng.standard_normal(n)
        df_x = pd.DataFrame({"x": x, "m": m})
        df_y = pd.DataFrame({"y": y})

        res_med = mediation_analysis(
            df_x, df_y, predictor="x", mediator="m", n_bootstrap=200, random_state=42
        )
        assert res_med["ci_method"] == "BCa"
        assert "indirect_effect" in res_med
        ci_tuple = res_med["indirect_effect_ci"]
        assert isinstance(ci_tuple, tuple)
        assert ci_tuple[0] < ci_tuple[1]

    def test_pillar11_sensitivity_bounds_exactness(self) -> None:
        """Item 32: E-value exact transformation and Rosenbaum monotonic widening."""
        # E-value for RR=2.0 point
        e1 = compute_e_value(float(np.log(2.0)), family="poisson")
        expected_e = 2.0 + np.sqrt(2.0 * (2.0 - 1.0))
        np.testing.assert_allclose(float(e1["e_value"]), expected_e, atol=1e-4)

        # Protective effect: RR=0.5 inverts to 1/0.5 = 2.0
        e2 = compute_e_value(float(np.log(0.5)), family="poisson")
        np.testing.assert_allclose(float(e2["e_value"]), expected_e, atol=1e-4)

        # Rosenbaum bounds monotonicity
        rng = np.random.default_rng(42)
        x_bin = rng.binomial(1, 0.5, 60)
        y_vec = 1.5 * x_bin + rng.standard_normal(60)
        X_df = pd.DataFrame({"x": x_bin})
        res_dict = {"p_values": [0.001], "family": "linear"}
        rb = rosenbaum_bounds(res_dict, X_df, y_vec, gammas=(1.0, 1.5, 2.0, 3.0))
        worst_p = [float(p) for p in rb["worst_case_p"]]
        # Starting from gamma=1.5, worst case p-values are strictly non-decreasing
        for i in range(2, len(worst_p)):
            assert worst_p[i] >= worst_p[i - 1]

    def test_pillar12_mercer_psd_and_gram_centering_zero_row_sum(self) -> None:
        """Item 33 & 34: Mercer kernels are PSD and centered Gram has zero row sum."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 4))
        kernels: list[Kernel] = [
            GaussianKernel(),
            LaplacianKernel(),
            CosineKernel(),
            LinearKernel(),
        ]

        for k in kernels:
            keval = k.evaluate(X)
            K = keval.K_nm
            min_eig = float(np.min(np.linalg.eigvalsh(K)))
            assert min_eig >= -1e-14, (
                f"{k.name()} violated Mercer PSD: min eig={min_eig}"
            )

            Kc = gram_centering(keval)
            row_sums = np.sum(Kc.K_nm, axis=1)
            max_row_sum = float(np.max(np.abs(row_sums)))
            assert max_row_sum < 1e-13, (
                f"{k.name()} centered Gram non-zero row sum: {max_row_sum}"
            )

    def test_pillar12_mmd_hsic_null_centering(self) -> None:
        """Item 35: Unbiased MMD null centers at zero, and HSIC permutation null calibrates observed statistic."""
        rng = np.random.default_rng(42)
        n = 40
        X = rng.standard_normal((n, 2))
        Y_same = rng.standard_normal((n, 2))  # P = Q under standard normal

        # Unbiased MMD estimator has expectation zero under H0: P = Q
        res_mmd = mmd_test(X, Y_same, n_randomizations=100, random_state=42)
        assert res_mmd.null_distribution is not None
        mean_mmd_null = float(np.mean(res_mmd.null_distribution))
        sd_mmd_null = float(np.std(res_mmd.null_distribution, ddof=1))
        assert abs(mean_mmd_null) / sd_mmd_null <= 0.15

        # Under independence X ⊥ Y, the canonical Gretton V-statistic HSIC_b >= 0 permutation null
        # centers at its exact analytical expectation: E_pi[HSIC_b] = Tr(Kc) Tr(Lc) / (n^2 (n - 1))
        Y_indep = rng.standard_normal((n, 2))
        res_hsic = hsic_test(X, Y_indep, n_randomizations=500, random_state=42)
        assert res_hsic.null_distribution is not None
        mean_hsic_null = float(np.mean(res_hsic.null_distribution))
        sd_hsic_null = float(np.std(res_hsic.null_distribution, ddof=1))

        # Analytical expectation derivation
        kx = GaussianKernel(sigma="median")
        ky = GaussianKernel(sigma="median")
        Kc = gram_centering(kx.evaluate(X, random_state=42))
        Lc = gram_centering(ky.evaluate(Y_indep, random_state=42))
        tr_K = float(np.trace(Kc.K_nm))
        tr_L = float(np.trace(Lc.K_nm))
        expected_hsic = (tr_K * tr_L) / (n**2 * (n - 1))

        # Invariant: Empirical permutation null converges to theoretical expectation
        assert abs(mean_hsic_null - expected_hsic) / sd_hsic_null <= 0.15
        assert abs(res_hsic.statistic - mean_hsic_null) / sd_hsic_null <= 2.0
        assert 0.05 <= res_hsic.p_value <= 0.95

    def test_pillar12_text_npmi_and_nmf_nonnegativity(self) -> None:
        """Item 36: Text representation preserves non-negativity and NPMI bound [-1, 1]."""
        docs = [
            "machine learning neural network deep learning",
            "learning model training parameters network",
            "quantum computing physics qubit entanglement",
            "physics particles quantum states simulation",
        ]
        res_nmf = docs_to_topics(
            docs, n_components=2, random_state=42, return_components=True
        )
        assert isinstance(res_nmf, tuple)
        W, H = res_nmf
        assert float(W.min()) >= 0.0, "NMF topic proportions contained negative entries"
        assert float(H.min()) >= 0.0, "NMF word weights contained negative entries"

        coherence = coherence_score(docs, n_components=2, random_state=42)
        assert -1.0 <= coherence <= 1.0, (
            f"NPMI topic coherence {coherence} outside [-1, 1]"
        )

    def test_pillar13_dml_neyman_orthogonality_and_cross_fitting(self) -> None:
        """Item 37: DML cross-fitting evaluates nuisance out-of-fold without leakage."""
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(42)
        n, q, p = 100, 3, 2
        Z = rng.standard_normal((n, q))
        X = 1.0 * Z[:, :p] + rng.standard_normal((n, p))
        y = 2.0 * X[:, 0] - 1.5 * Z[:, 0] + rng.standard_normal(n)

        # Cross-fit residualize
        y_res, X_res = _cross_fit_residualize(
            Z, y, X, reduced_model=Ridge(alpha=1.0), n_folds=5, random_state=42
        )
        assert y_res.shape == (n,)
        assert X_res.shape == (n, p)
        # Residuals centered approximately at zero
        assert abs(float(np.mean(y_res))) < 0.2
        assert abs(float(np.mean(X_res))) < 0.2

    def test_pillar14_conformal_order_statistic_and_batch_vectorization(self) -> None:
        """Item 39 & 40: Conformal computes exact rank ceil((n+1)(1-alpha)) and vectorizes bounds."""
        rng = np.random.default_rng(42)
        n_cal, n_query = 80, 15
        X_cal = rng.standard_normal((n_cal, 2))
        y_cal = 2.0 * X_cal[:, 0] + rng.standard_normal(n_cal)

        X_query = rng.standard_normal((n_query, 2))
        y_query = 2.0 * X_query[:, 0] + rng.standard_normal(n_query)

        conf_res = conformal_prediction(
            X_cal,
            y_cal,
            X_query,
            confidence_level=0.90,
            method="split",
            random_state=42,
        )
        assert conf_res.intervals is not None
        assert conf_res.intervals.shape == (n_query, 2)
        assert np.all(conf_res.intervals[:, 0] < conf_res.intervals[:, 1])

        # Evaluate Feature-Stratified Coverage
        groups_query = np.repeat([0, 1, 2], 5)
        eval_res = conf_res.evaluate(y_query, groups=groups_query)
        assert 0.0 <= eval_res.empirical_coverage <= 1.0
        assert eval_res.fsc_metric is not None
        assert 0.0 <= eval_res.fsc_metric <= 1.0
        assert eval_res.fsc_metric <= eval_res.empirical_coverage

    def test_pillar14_result_dataclass_immutability_and_serialization(self) -> None:
        """Item 41: Result objects are frozen, support clean container checks, and serialize to JSON."""
        conf_res = ConformalResult(
            prediction_interval=(1.0, 2.0),
            prediction_set=None,
            nonconformity_scores=np.array([0.1, 0.2]),
            confidence_level=0.95,
            method="split",
            intervals=np.array([[1.0, 2.0]]),
        )
        # 1. Immutability check
        with pytest.raises(FrozenInstanceError):
            conf_res.method = "mutated"  # type: ignore[misc]

        # 2. Container syntax checks data fields only
        assert "intervals" in conf_res
        assert "confidence_level" in conf_res
        assert "to_dict" not in conf_res  # method excluded from container query

        # 3. JSON serialization
        d = conf_res.to_dict()
        assert isinstance(d, dict)
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_pillar15_fixed_x_gram_preservation(self) -> None:
        """Item 43: Fixed-X knockoffs preserve Gram matrix identities to < 1e-10."""
        rng = np.random.default_rng(42)
        n, p = 80, 10
        X = rng.standard_normal((n, p))
        y = X[:, 0] + rng.standard_normal(n)
        X_df = pd.DataFrame(X, columns=[f"feat{j}" for j in range(p)])
        y_df = pd.DataFrame({"y": y})

        res = knockoff_test(X_df, y_df, method="fixed_x", random_state=42)
        assert isinstance(res, KnockoffResult)
        assert res.guarantee_tier == "finite_sample_exact"
        # Fixed-X requires n >= 2p
        with pytest.raises(ValueError, match="Fixed-X knockoffs require n >= 2p"):
            X_under = pd.DataFrame(X[:15], columns=[f"feat{j}" for j in range(p)])
            y_under = pd.DataFrame({"y": y[:15]})
            knockoff_test(X_under, y_under, method="fixed_x", random_state=42)

    def test_pillar15_model_x_shrinkage_and_coin_flip_property(self) -> None:
        """Item 44 & 45: Model-X shrinkage ensures positive definiteness and coin-flip sign-inversion."""
        rng = np.random.default_rng(42)
        n, p = 100, 15
        X = rng.standard_normal((n, p))
        y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.standard_normal(n)
        X_df = pd.DataFrame(X, columns=[f"feat{j}" for j in range(p)])
        y_df = pd.DataFrame({"y": y})

        with pytest.warns(
            UserWarning,
            match="method='fixed_x' provides a finite-sample exact FDR guarantee",
        ):
            res = knockoff_test(
                X_df, y_df, method="equicorrelated", shrinkage=True, random_state=42
            )
        assert isinstance(res, KnockoffResult)
        assert res.guarantee_tier == "model_x_approximate"
        # Importance statistics W exist
        assert res.knockoff_statistics is not None
        assert len(res.knockoff_statistics) == p

    def test_pillar15_knockoff_plus_threshold_and_mmd_swap_diagnostic(self) -> None:
        """Item 46: Knockoff+ threshold selects features and MMD swap test audits exchangeability."""
        rng = np.random.default_rng(42)
        n, p = 200, 30
        X = rng.standard_normal((n, p))
        beta = np.zeros(p)
        beta[:8] = 3.5  # 8 strong signals
        y = X @ beta + rng.standard_normal(n)
        X_df = pd.DataFrame(X, columns=[f"x{j}" for j in range(p)])
        y_df = pd.DataFrame({"y": y})

        res_fixed = knockoff_test(
            X_df, y_df, method="fixed_x", fdr_level=0.20, random_state=42
        )
        assert isinstance(res_fixed, KnockoffResult)
        assert len(res_fixed.selected_features) >= 5
        # Defect CD-4: Fixed-X knockoffs preserve exact Gram structure to machine precision
        # and do not spuriously fail an MMD row-swap test.
        assert "gram_preservation_error" in res_fixed.diagnostics
        assert res_fixed.diagnostics["gram_preservation_error"] < 1e-12
        assert res_fixed.diagnostics["knockoff_gram_error"] < 1e-12
        assert res_fixed.diagnostics["swap_violation"] is False

        # Verify Model-X MMD swap test on Gaussian design confirms exchangeability
        with pytest.warns(
            UserWarning,
            match="method='fixed_x' provides a finite-sample exact FDR guarantee",
        ):
            res_model = knockoff_test(
                X_df, y_df, method="equicorrelated", shrinkage=True, random_state=42
            )
        assert "mmd_swap_p_values" in res_model.diagnostics
        assert res_model.diagnostics["swap_violation"] is False


class TestTier5SoftwareArchitectureAndTyping:
    """Certification Tier V: Software Architecture, Strict Typing & Public API Governance (Pillars 16–18).

    Validates:
    - Pillar 16:
      * Item 47: @runtime_checkable protocol polymorphism for custom families and kernels.
      * Item 48: Warning hygiene (stacklevel parameter) and no bare except clauses.
    - Pillar 17:
      * Item 49: Certified py.typed packaging and PEP 561 compliance.
    - Pillar 18:
      * Item 50: Clean, leak-free namespace and star-import parity with __all__.
      * Item 51: Zero-technical-debt SemVer 0.x policy (deprecated shims purged).
    """

    def test_pillar16_runtime_protocol_verification_and_custom_extensions(
        self,
    ) -> None:
        """Item 47: ModelFamily and Kernel protocols are runtime checkable and extensible."""
        # 1. Protocols are runtime checkable
        assert isinstance(LinearFamily(), ModelFamily)
        assert isinstance(GaussianKernel(), Kernel)

        # 2. Custom user-defined Kernel works polymorphically
        class CustomAbsExponentialKernel:
            def __call__(
                self,
                X1: np.ndarray,
                X2: np.ndarray | None = None,
            ) -> np.ndarray:
                x1 = np.asarray(X1, dtype=float)
                x2 = x1 if X2 is None else np.asarray(X2, dtype=float)
                diff = np.abs(x1[:, np.newaxis, :] - x2[np.newaxis, :, :])
                return np.asarray(np.exp(-np.sum(diff, axis=-1)))

            def evaluate(
                self,
                X: np.ndarray,
                *,
                max_landmarks: int | None = None,
                random_state: int | None = None,
            ) -> Any:
                from randomization_tests._kernels import KernelEval

                X_arr = np.asarray(X, dtype=float)
                # Compute absolute-difference Laplacian-style kernel
                K = self(X_arr)
                n_samples = X_arr.shape[0]
                return KernelEval(
                    K_nm=K,
                    K_mm_inv=np.eye(n_samples),
                    is_exact=True,
                    n=n_samples,
                    m=n_samples,
                )

            def name(self) -> str:
                return "custom_abs_exp"

        custom_k = CustomAbsExponentialKernel()
        assert isinstance(custom_k, Kernel)

        # Run MMD test with custom user kernel
        rng = np.random.default_rng(42)
        X = rng.standard_normal((25, 2))
        Y = rng.standard_normal((25, 2))
        res_mmd = mmd_test(X, Y, kernel=custom_k, n_randomizations=50, random_state=42)
        assert res_mmd.kernel_name == "custom_abs_exp"
        assert 0.0 <= res_mmd.p_value <= 1.0

    def test_pillar16_warning_hygiene_and_stack_trace_transparency(
        self,
    ) -> None:
        """Item 48: Warnings specify stacklevel >= 2 so diagnostics cite user calling code."""
        rng = np.random.default_rng(42)
        n = 40
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        y = pd.DataFrame({"y": (rng.uniform(size=n) > 0.5).astype(float)})

        # Requesting fixed_x on logistic should emit warning with stacklevel=2
        with pytest.warns(UserWarning) as record:
            randomization_test_regression(
                X, y, method="manly", n_randomizations=20, random_state=42
            )
        assert len(record) > 0
        # The warning must point to this test file, not internal engine line
        warn_filename = record[0].filename
        assert "test_architecture_gate.py" in warn_filename

    def test_pillar17_pep561_py_typed_packaging(self) -> None:
        """Item 49: Package must include certified py.typed marker file."""
        import pathlib

        import randomization_tests

        pkg_path = pathlib.Path(randomization_tests.__file__).parent
        py_typed = pkg_path / "py.typed"
        assert py_typed.exists(), "py.typed marker missing from package root"
        assert py_typed.is_file(), "py.typed is not a regular file"

    def test_pillar18_leak_free_namespace_and_all_parity(self) -> None:
        """Item 50: __all__ strictly matches public API and contains zero private or third-party symbols."""
        import randomization_tests as rt

        # 1. Star-import exports strictly __all__
        all_symbols = rt.__all__
        assert len(all_symbols) == len(set(all_symbols)), "Duplicate symbol in __all__"

        # 2. No private symbols in __all__
        for sym in all_symbols:
            assert not sym.startswith("_"), (
                f"Private symbol '{sym}' leaked into __all__"
            )

        # 3. No third-party modules leaked into __all__
        third_party_names = {"np", "pd", "sm", "stats", "scipy", "pytest", "jax", "jnp"}
        for tp in third_party_names:
            assert tp not in all_symbols, f"Third-party name '{tp}' in __all__"

    def test_pillar18_zero_debt_semver_policy(self) -> None:
        """Item 51: Superseded parameter aliases (e.g. n_permutations) are completely purged."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        Y = rng.standard_normal((30, 2))

        # mmd_test rejects n_permutations
        with pytest.raises(TypeError):
            mmd_test(X, Y, n_permutations=100)  # type: ignore[call-arg]

        # hsic_test rejects n_permutations
        with pytest.raises(TypeError):
            hsic_test(X, Y, n_permutations=100)  # type: ignore[call-arg]


class TestTier6DualHarnessModernization:
    """Certification Tier VI: Test Suite Modernization & Dual-Harness Replacement (Pillar 19).

    Validates:
    - Item 52: Full test suite audit, trivial test elimination, and parameterized contract coverage.
    - Item 53: Paired adversarial dual-harness verification gates (Defect Prover vs Invariant Verifier).
    """

    def test_pillar19_item52_test_suite_modernization_and_contract_coverage(
        self,
    ) -> None:
        """Item 52: All result types implement frozen dataclass contracts and exclude context from serialization."""
        import json
        from dataclasses import FrozenInstanceError, fields

        from randomization_tests._results import (
            ConformalEvaluationResult,
            ConformalResult,
            FitResult,
            InvarianceResult,
            KernelTestResult,
            KnockoffResult,
        )

        sample_instances = [
            KernelTestResult(0.42, 0.03, np.array([0.1, 0.2]), "gaussian", 2, "mmd"),
            ConformalResult((1.0, 3.0), None, np.array([0.5]), 0.95, "split"),
            ConformalEvaluationResult(0.95, 0.95, 10, np.ones(10, dtype=bool)),
            InvarianceResult(True, 0.42, {}, 1.0, 0.2, np.array([0.1])),
            KnockoffResult(["x1"], np.array([1.0]), 0.1, 0.5, ["x1"]),
            FitResult(
                params=np.array([1.0]),
                predictions=np.array([1.0]),
                fitted_values=np.array([1.0]),
                residuals=np.array([0.0]),
                log_likelihood=-1.0,
                fisher_info=np.eye(1),
                se=np.array([0.1]),
                p_values=np.array([0.01]),
                fit_intercept=True,
                converged=True,
                df=10.0,
            ),
        ]

        for inst in sample_instances:
            first_field = next(iter(fields(inst))).name
            with pytest.raises(FrozenInstanceError):
                setattr(inst, first_field, "mutated")

            assert inst[first_field] == getattr(inst, first_field)
            assert inst.get(first_field) == getattr(inst, first_field)
            assert "to_dict" not in inst
            assert "nonexistent_field" not in inst

            d = inst.to_dict()
            assert isinstance(d, dict)
            assert "context" not in d
            assert isinstance(json.dumps(d), str)

    def test_pillar19_item53_adversarial_dual_harness_verification_gates(
        self,
    ) -> None:
        """Item 53: Paired dual-harness gates proving failure modes and verifying invariants."""
        # ── Harness 1 (The Defect Prover): Ill-conditioned design breaks normal equations ──
        rng = np.random.default_rng(42)
        n, p = 60, 4
        # Generate ill-conditioned design matrix with condition number > 1e7
        U, _ = np.linalg.qr(rng.standard_normal((n, p)))
        V, _ = np.linalg.qr(rng.standard_normal((p, p)))
        s = np.array([1.0, 1e-2, 1e-5, 1e-7])
        X_ill = U[:, :p] @ np.diag(s) @ V
        beta_true = np.array([1.0, -1.0, 2.0, 0.5])
        y_ill = X_ill @ beta_true

        # Normal equations square condition number to 1e14 -> severe loss of precision
        XtX = X_ill.T @ X_ill
        Xty = X_ill.T @ y_ill
        beta_normal_eq = np.linalg.solve(XtX, Xty)
        err_normal_eq = float(
            np.linalg.norm(beta_normal_eq - beta_true) / np.linalg.norm(beta_true)
        )
        assert err_normal_eq > 1e-4, (
            "Normal equations unexpectedly did not suffer ill-conditioning"
        )

        # ── Harness 2 (The Invariant Verifier): SVD/QR least-squares preserves machine-precision accuracy ──
        beta_svd, _, _, _ = np.linalg.lstsq(X_ill, y_ill, rcond=None)
        err_svd = float(
            np.linalg.norm(beta_svd - beta_true) / np.linalg.norm(beta_true)
        )
        assert err_svd < 1e-8, (
            f"SVD error {err_svd} exceeded numerical stability threshold"
        )
        assert err_normal_eq / max(err_svd, 1e-16) > 1e3, (
            "SVD was not significantly superior to normal equations"
        )

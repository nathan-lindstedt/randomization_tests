"""Tests for the backend configuration system."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from randomization_tests._config import backend, get_backend, set_backend


class TestGetBackend:
    """Tests for get_backend() resolution order."""

    def setup_method(self) -> None:
        """Reset state before each test."""
        set_backend(None)
        # Clear the env var if set
        os.environ.pop("RANDOMIZATION_TESTS_BACKEND", None)

    def teardown_method(self) -> None:
        """Reset state after each test."""
        set_backend(None)
        os.environ.pop("RANDOMIZATION_TESTS_BACKEND", None)

    def test_auto_detects_jax_when_installed(self) -> None:
        # JAX is installed in our test env, so auto-detect should find it
        result = get_backend()
        assert result == "jax"

    def test_env_var_overrides_auto(self) -> None:
        os.environ["RANDOMIZATION_TESTS_BACKEND"] = "numpy"
        assert get_backend() == "numpy"

    def test_env_var_jax(self) -> None:
        os.environ["RANDOMIZATION_TESTS_BACKEND"] = "jax"
        assert get_backend() == "jax"

    def test_env_var_case_insensitive(self) -> None:
        os.environ["RANDOMIZATION_TESTS_BACKEND"] = "NumPy"
        assert get_backend() == "numpy"

    def test_programmatic_override_wins_over_env(self) -> None:
        os.environ["RANDOMIZATION_TESTS_BACKEND"] = "numpy"
        set_backend("jax")
        assert get_backend() == "jax"

    def test_auto_restores_default(self) -> None:
        set_backend("numpy")
        assert get_backend() == "numpy"
        set_backend("auto")
        # With JAX installed, auto should return "jax"
        assert get_backend() == "jax"


class TestSetBackend:
    """Tests for set_backend() validation."""

    def setup_method(self) -> None:
        set_backend(None)

    def teardown_method(self) -> None:
        set_backend(None)

    def test_accepts_valid_names(self) -> None:
        for name in ("jax", "numpy", "auto", None):
            set_backend(name)  # should not raise

    def test_case_insensitive(self) -> None:
        set_backend("JAX")
        assert get_backend() == "jax"

    def test_rejects_invalid_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("tensorflow")


class TestBackendContextManager:
    """Tests for the scoped backend() context manager and thread safety."""

    def setup_method(self) -> None:
        set_backend(None)
        os.environ.pop("RANDOMIZATION_TESTS_BACKEND", None)

    def teardown_method(self) -> None:
        set_backend(None)
        os.environ.pop("RANDOMIZATION_TESTS_BACKEND", None)

    def test_context_manager_temporary_override(self) -> None:
        """backend() should override within block and restore previous setting."""
        set_backend("jax")
        assert get_backend() == "jax"
        with backend("numpy") as active:
            assert active == "numpy"
            assert get_backend() == "numpy"
        assert get_backend() == "jax"

    def test_context_manager_nested_scopes(self) -> None:
        """Nested backend() blocks restore enclosing scopes correctly."""
        set_backend("jax")
        with backend("numpy"):
            assert get_backend() == "numpy"
            with backend("jax"):
                assert get_backend() == "jax"
            assert get_backend() == "numpy"
        assert get_backend() == "jax"

    def test_concurrency_thread_isolation(self) -> None:
        """ContextVar ensures concurrent threads maintain isolated backend settings."""
        set_backend("jax")

        def worker_numpy() -> str:
            with backend("numpy"):
                import time

                time.sleep(0.01)
                return get_backend()

        def worker_jax() -> str:
            with backend("jax"):
                import time

                time.sleep(0.01)
                return get_backend()

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures_numpy = [pool.submit(worker_numpy) for _ in range(5)]
            futures_jax = [pool.submit(worker_jax) for _ in range(5)]

            for f in futures_numpy:
                assert f.result() == "numpy"
            for f in futures_jax:
                assert f.result() == "jax"

        # Calling thread should remain unchanged
        assert get_backend() == "jax"


class TestBackendIntegration:
    """Verify that set_backend('numpy') actually disables JAX in core."""

    def setup_method(self) -> None:
        set_backend(None)

    def teardown_method(self) -> None:
        set_backend(None)

    def test_numpy_backend_runs_logistic(self) -> None:
        """Logistic test should work with the numpy backend."""
        import numpy as np
        import pandas as pd

        from randomization_tests.core import randomization_test_regression

        set_backend("numpy")

        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            {"x1": rng.standard_normal(100), "x2": rng.standard_normal(100)}
        )
        logits = 2.0 * X["x1"]
        probs = 1 / (1 + np.exp(-logits))
        y = pd.DataFrame({"y": rng.binomial(1, probs)})

        result = randomization_test_regression(
            X,
            y,
            n_randomizations=20,
            method="ter_braak",
            random_state=42,
        )
        assert result.family.name == "logistic"
        assert "model_coefs" in result

    def test_public_api_exports(self) -> None:
        """get_backend, set_backend, and backend should be importable from the package."""
        import randomization_tests

        assert hasattr(randomization_tests, "backend")
        assert hasattr(randomization_tests, "get_backend")
        assert hasattr(randomization_tests, "set_backend")

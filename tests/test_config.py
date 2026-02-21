"""Tests for the backend configuration system."""

import os

import pytest

from randomization_tests._config import get_backend, set_backend


class TestGetBackend:
    """Tests for get_backend() resolution order."""

    def setup_method(self):
        """Reset state before each test."""
        import randomization_tests._config as _cfg
        _cfg._backend_override = None
        # Clear the env var if set
        os.environ.pop("RANDOMIZATION_TESTS_BACKEND", None)

    def teardown_method(self):
        """Reset state after each test."""
        import randomization_tests._config as _cfg
        _cfg._backend_override = None
        os.environ.pop("RANDOMIZATION_TESTS_BACKEND", None)

    def test_auto_detects_jax_when_installed(self):
        # JAX is installed in our test env, so auto-detect should find it
        result = get_backend()
        assert result == "jax"

    def test_env_var_overrides_auto(self):
        os.environ["RANDOMIZATION_TESTS_BACKEND"] = "numpy"
        assert get_backend() == "numpy"

    def test_env_var_jax(self):
        os.environ["RANDOMIZATION_TESTS_BACKEND"] = "jax"
        assert get_backend() == "jax"

    def test_env_var_case_insensitive(self):
        os.environ["RANDOMIZATION_TESTS_BACKEND"] = "NumPy"
        assert get_backend() == "numpy"

    def test_programmatic_override_wins_over_env(self):
        os.environ["RANDOMIZATION_TESTS_BACKEND"] = "numpy"
        set_backend("jax")
        assert get_backend() == "jax"

    def test_auto_restores_default(self):
        set_backend("numpy")
        assert get_backend() == "numpy"
        set_backend("auto")
        # With JAX installed, auto should return "jax"
        assert get_backend() == "jax"


class TestSetBackend:
    """Tests for set_backend() validation."""

    def setup_method(self):
        import randomization_tests._config as _cfg
        _cfg._backend_override = None

    def teardown_method(self):
        import randomization_tests._config as _cfg
        _cfg._backend_override = None

    def test_accepts_valid_names(self):
        for name in ("jax", "numpy", "auto"):
            set_backend(name)  # should not raise

    def test_case_insensitive(self):
        set_backend("JAX")
        assert get_backend() == "jax"

    def test_rejects_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("tensorflow")


class TestBackendIntegration:
    """Verify that set_backend('numpy') actually disables JAX in core."""

    def setup_method(self):
        import randomization_tests._config as _cfg
        _cfg._backend_override = None

    def teardown_method(self):
        import randomization_tests._config as _cfg
        _cfg._backend_override = None

    def test_numpy_backend_runs_logistic(self):
        """Logistic test should work with the numpy backend."""
        import numpy as np
        import pandas as pd

        from randomization_tests.core import permutation_test_regression

        set_backend("numpy")

        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x1": rng.standard_normal(100), "x2": rng.standard_normal(100)})
        logits = 2.0 * X["x1"]
        probs = 1 / (1 + np.exp(-logits))
        y = pd.DataFrame({"y": rng.binomial(1, probs)})

        result = permutation_test_regression(
            X, y, n_permutations=20, method="ter_braak", random_state=42,
        )
        assert result["model_type"] == "logistic"
        assert "model_coefs" in result

    def test_public_api_exports(self):
        """get_backend and set_backend should be importable from the package."""
        import randomization_tests
        assert hasattr(randomization_tests, "get_backend")
        assert hasattr(randomization_tests, "set_backend")

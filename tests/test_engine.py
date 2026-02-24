"""Unit tests for PermutationEngine.

Step 0b of the v0.4.0 plan: characterise existing engine behaviour
before refactoring (Steps 1, 2, 8, 9).  These are *regression* tests
that pin the current contract, not integration tests — every assertion
should fail if the corresponding engine attribute or guard is broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from randomization_tests.engine import PermutationEngine
from randomization_tests.families import (
    LinearFamily,
    LogisticFamily,
    ModelFamily,
    NegativeBinomialFamily,
)

# ------------------------------------------------------------------ #
# Shared fixtures
# ------------------------------------------------------------------ #

_SEED = 42
_N_PERMS = 50  # small for speed; enough to test shapes


@pytest.fixture()
def rng():
    return np.random.default_rng(_SEED)


@pytest.fixture()
def linear_data(rng):
    """Continuous y with 3 features and known coefficients."""
    n, p = 100, 3
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=["x1", "x2", "x3"])
    y = (
        2.0 * X["x1"].to_numpy()
        - 1.0 * X["x2"].to_numpy()
        + 0.5 * X["x3"].to_numpy()
        + rng.standard_normal(n) * 0.5
    )
    return X, y


@pytest.fixture()
def binary_data(rng):
    """Binary y (0/1) with 2 features."""
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 2)), columns=["x1", "x2"])
    logits = 2.0 * X["x1"].to_numpy()
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(float)
    return X, y


@pytest.fixture()
def count_data(rng):
    """Poisson-distributed count y with 2 features."""
    n = 120
    X = pd.DataFrame(rng.standard_normal((n, 2)), columns=["x1", "x2"])
    eta = 1.0 + 0.8 * X["x1"].to_numpy() - 0.3 * X["x2"].to_numpy()
    mu = np.exp(eta)
    y = rng.poisson(lam=mu).astype(float)
    return X, y


@pytest.fixture()
def ordinal_data(rng):
    """Ordinal y with 4 ordered categories (0–3) and 2 features."""
    n = 120
    X = pd.DataFrame(rng.standard_normal((n, 2)), columns=["x1", "x2"])
    z = (
        0.6 * X["x1"].to_numpy()
        - 0.3 * X["x2"].to_numpy()
        + rng.standard_normal(n) * 0.8
    )
    y = np.digitize(z, bins=np.quantile(z, [0.25, 0.5, 0.75])).astype(float)
    return X, y


@pytest.fixture()
def multinomial_data(rng):
    """Multinomial y with 3 unordered categories (0–2) and 2 features."""
    n = 150
    X = pd.DataFrame(rng.standard_normal((n, 2)), columns=["x1", "x2"])
    logits = np.column_stack(
        [
            np.zeros(n),
            0.8 * X["x1"].to_numpy() - 0.4 * X["x2"].to_numpy(),
            -0.3 * X["x1"].to_numpy() + 0.6 * X["x2"].to_numpy(),
        ]
    )
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=probs[i]) for i in range(n)], dtype=float)
    return X, y


# ------------------------------------------------------------------ #
# TestEngineConstruction
# ------------------------------------------------------------------ #


class TestEngineConstruction:
    """Family resolution for all 6 family strings + 'auto'."""

    @pytest.mark.parametrize(
        ("family_str", "expected_family_name", "data_fixture"),
        [
            ("linear", "linear", "linear_data"),
            ("logistic", "logistic", "binary_data"),
            ("poisson", "poisson", "count_data"),
            ("negative_binomial", "negative_binomial", "count_data"),
            ("ordinal", "ordinal", "ordinal_data"),
            ("multinomial", "multinomial", "multinomial_data"),
        ],
    )
    def test_explicit_family_resolution(
        self, family_str, expected_family_name, data_fixture, request
    ):
        X, y = request.getfixturevalue(data_fixture)
        engine = PermutationEngine(
            X, y, family=family_str, n_permutations=_N_PERMS, random_state=_SEED
        )
        assert engine.family.name == expected_family_name
        assert isinstance(engine.family, ModelFamily)

    def test_auto_resolves_linear_for_continuous(self, linear_data):
        X, y = linear_data
        engine = PermutationEngine(
            X, y, family="auto", n_permutations=_N_PERMS, random_state=_SEED
        )
        assert engine.family.name == "linear"

    def test_auto_resolves_logistic_for_binary(self, binary_data):
        X, y = binary_data
        engine = PermutationEngine(
            X, y, family="auto", n_permutations=_N_PERMS, random_state=_SEED
        )
        assert engine.family.name == "logistic"

    def test_backend_name_is_string(self, linear_data):
        X, y = linear_data
        engine = PermutationEngine(X, y, n_permutations=_N_PERMS, random_state=_SEED)
        assert isinstance(engine.backend_name, str)
        assert engine.backend_name in ("numpy", "jax")

    def test_model_coefs_shape(self, linear_data):
        X, y = linear_data
        engine = PermutationEngine(X, y, n_permutations=_N_PERMS, random_state=_SEED)
        # Linear family: one coefficient per column in X
        assert engine.model_coefs.shape == (X.shape[1],)

    def test_perm_indices_shape(self, linear_data):
        X, y = linear_data
        engine = PermutationEngine(X, y, n_permutations=_N_PERMS, random_state=_SEED)
        assert engine.perm_indices.shape == (_N_PERMS, len(y))

    def test_permute_indices_is_instance_method(self, linear_data):
        """permute_indices is an instance method, not a staticmethod."""
        X, y = linear_data
        engine = PermutationEngine(X, y, n_permutations=_N_PERMS, random_state=_SEED)
        result = engine.permute_indices(
            n_samples=len(y), n_permutations=5, random_state=0
        )
        assert result.shape == (5, len(y))

    def test_permute_hook_override(self, linear_data):
        """Subclassing _permute_hook is picked up by permute_indices."""
        X, y = linear_data

        class IdentityEngine(PermutationEngine):
            def _permute_hook(self, n_samples, n_permutations, random_state=None):
                return np.tile(np.arange(n_samples), (n_permutations, 1))

        engine = IdentityEngine(X, y, n_permutations=_N_PERMS, random_state=_SEED)
        # The constructor already called permute_indices → _permute_hook,
        # so perm_indices should be all-identity rows.
        expected_row = np.arange(len(y))
        for row in engine.perm_indices:
            np.testing.assert_array_equal(row, expected_row)

    def test_diagnostics_is_dict(self, linear_data):
        X, y = linear_data
        engine = PermutationEngine(X, y, n_permutations=_N_PERMS, random_state=_SEED)
        assert isinstance(engine.diagnostics, dict)

    def test_unknown_family_raises(self, linear_data):
        X, y = linear_data
        with pytest.raises(ValueError, match="Unknown family"):
            PermutationEngine(X, y, family="nonexistent", n_permutations=_N_PERMS)


# ------------------------------------------------------------------ #
# TestEngineCalibration
# ------------------------------------------------------------------ #


class TestEngineCalibration:
    """Verify calibrate() is called during construction."""

    def test_nb_alpha_is_set(self, count_data):
        X, y = count_data
        engine = PermutationEngine(
            X,
            y,
            family="negative_binomial",
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert hasattr(engine.family, "alpha")
        assert engine.family.alpha is not None
        assert engine.family.alpha > 0

    def test_linear_has_no_alpha(self, linear_data):
        X, y = linear_data
        engine = PermutationEngine(
            X,
            y,
            family="linear",
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert not hasattr(engine.family, "alpha")


# ------------------------------------------------------------------ #
# TestEngineFreedmanLaneGuard
# ------------------------------------------------------------------ #


class TestEngineFreedmanLaneGuard:
    """ValueError for Freedman-Lane with direct-permutation families."""

    def test_ordinal_rejects_freedman_lane(self, ordinal_data):
        X, y = ordinal_data
        with pytest.raises(ValueError, match="Freedman-Lane.*not supported"):
            PermutationEngine(
                X,
                y,
                family="ordinal",
                method="freedman_lane",
                n_permutations=_N_PERMS,
                random_state=_SEED,
            )

    def test_multinomial_rejects_freedman_lane(self, multinomial_data):
        X, y = multinomial_data
        with pytest.raises(ValueError, match="Freedman-Lane.*not supported"):
            PermutationEngine(
                X,
                y,
                family="multinomial",
                method="freedman_lane",
                n_permutations=_N_PERMS,
                random_state=_SEED,
            )

    def test_ordinal_rejects_freedman_lane_joint(self, ordinal_data):
        X, y = ordinal_data
        with pytest.raises(ValueError, match="Freedman-Lane.*not supported"):
            PermutationEngine(
                X,
                y,
                family="ordinal",
                method="freedman_lane_joint",
                n_permutations=_N_PERMS,
                random_state=_SEED,
            )

    def test_linear_accepts_freedman_lane(self, linear_data):
        """Freedman-Lane should work fine for linear family."""
        X, y = linear_data
        engine = PermutationEngine(
            X,
            y,
            family="linear",
            method="freedman_lane",
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert engine.family.name == "linear"


# ------------------------------------------------------------------ #
# TestEngineBackendWarnings
# ------------------------------------------------------------------ #


class TestEngineBackendWarnings:
    """UserWarning for n_jobs > 1 when parallelism has no effect."""

    def test_jax_njobs_warning(self, linear_data):
        """n_jobs ignored under JAX backend."""
        from randomization_tests._config import get_backend, set_backend

        original = get_backend()
        try:
            set_backend("jax")
            X, y = linear_data
            with pytest.warns(UserWarning, match="n_jobs is ignored.*JAX"):
                PermutationEngine(
                    X,
                    y,
                    n_permutations=_N_PERMS,
                    random_state=_SEED,
                    n_jobs=2,
                )
        except Exception:
            pytest.skip("JAX backend not available")
        finally:
            set_backend(original)

    def test_numpy_linear_tbraak_njobs_warning(self, linear_data):
        """n_jobs has no effect for vectorised OLS ter_braak."""
        from randomization_tests._config import get_backend, set_backend

        original = get_backend()
        try:
            set_backend("numpy")
            X, y = linear_data
            with pytest.warns(UserWarning, match="n_jobs has no effect"):
                PermutationEngine(
                    X,
                    y,
                    family="linear",
                    method="ter_braak",
                    n_permutations=_N_PERMS,
                    random_state=_SEED,
                    n_jobs=2,
                )
        finally:
            set_backend(original)


# ------------------------------------------------------------------ #
# TestPermuteIndices
# ------------------------------------------------------------------ #


class TestPermuteIndices:
    """Permutation index array properties."""

    def _make_engine(self, linear_data):
        X, y = linear_data
        return PermutationEngine(X, y, n_permutations=_N_PERMS, random_state=_SEED)

    def test_shape(self, linear_data):
        X, y = linear_data
        n = len(y)
        engine = self._make_engine(linear_data)
        indices = engine.permute_indices(n, _N_PERMS, random_state=_SEED)
        assert indices.shape == (_N_PERMS, n)

    def test_no_duplicate_rows(self, linear_data):
        X, y = linear_data
        n = len(y)
        engine = self._make_engine(linear_data)
        indices = engine.permute_indices(n, _N_PERMS, random_state=_SEED)
        # Each row should be unique
        unique_rows = np.unique(indices, axis=0)
        assert unique_rows.shape[0] == indices.shape[0]

    def test_no_identity_permutation(self, linear_data):
        X, y = linear_data
        n = len(y)
        engine = self._make_engine(linear_data)
        indices = engine.permute_indices(n, _N_PERMS, random_state=_SEED)
        identity = np.arange(n)
        for row in indices:
            assert not np.array_equal(row, identity)

    def test_deterministic_under_fixed_seed(self, linear_data):
        X, y = linear_data
        n = len(y)
        engine = self._make_engine(linear_data)
        indices1 = engine.permute_indices(n, _N_PERMS, random_state=_SEED)
        indices2 = engine.permute_indices(n, _N_PERMS, random_state=_SEED)
        np.testing.assert_array_equal(indices1, indices2)

    def test_rows_are_valid_permutations(self, linear_data):
        """Each row is a permutation of 0..n-1."""
        X, y = linear_data
        n = len(y)
        engine = self._make_engine(linear_data)
        indices = engine.permute_indices(n, _N_PERMS, random_state=_SEED)
        for row in indices:
            assert sorted(row) == list(range(n))


# ------------------------------------------------------------------ #
# TestEngineDiagnosticsFallback
# ------------------------------------------------------------------ #


class TestEngineDiagnosticsFallback:
    """The try/except guard around diagnostics gives a minimal dict."""

    def test_degenerate_data_fallback(self):
        """Rank-deficient X → diagnostics() may fail; engine should still construct.

        We use auto detection (not explicit family) so that
        validate_y is not called, and we make y non-constant so it
        passes the zero-variance check.  The diagnostics failure is
        triggered by a perfectly collinear feature matrix.
        """
        n = 50
        rng = np.random.default_rng(_SEED)
        col = rng.standard_normal(n)
        X = pd.DataFrame(
            {"x1": col, "x2": col},  # perfectly collinear
        )
        y = col + rng.standard_normal(n) * 0.1  # non-constant
        engine = PermutationEngine(
            X,
            y,
            family="auto",
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        # Engine should construct regardless of diagnostics outcome
        assert isinstance(engine.diagnostics, dict)

    def test_normal_data_has_rich_diagnostics(self, linear_data):
        """Well-formed data produces a diagnostics dict with many keys."""
        X, y = linear_data
        engine = PermutationEngine(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert isinstance(engine.diagnostics, dict)
        # Should have more than just the fallback keys
        assert len(engine.diagnostics) > 2


# ------------------------------------------------------------------ #
# TestEngineObservedCoefs
# ------------------------------------------------------------------ #


class TestEngineObservedCoefs:
    """Observed model coefficients are sensible after construction."""

    def test_linear_coefs_finite(self, linear_data):
        X, y = linear_data
        engine = PermutationEngine(
            X,
            y,
            family="linear",
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert np.all(np.isfinite(engine.model_coefs))

    def test_linear_coefs_match_column_count(self, linear_data):
        X, y = linear_data
        engine = PermutationEngine(
            X,
            y,
            family="linear",
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert engine.model_coefs.shape == (X.shape[1],)

    def test_logistic_coefs_finite(self, binary_data):
        X, y = binary_data
        engine = PermutationEngine(
            X,
            y,
            family="logistic",
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert np.all(np.isfinite(engine.model_coefs))

    def test_linear_coefs_reasonable_magnitude(self, linear_data):
        """True betas are [2, -1, 0.5]; fitted should be in the ballpark."""
        X, y = linear_data
        engine = PermutationEngine(
            X,
            y,
            family="linear",
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        # Not asserting exact values — just that they're not degenerate
        assert np.max(np.abs(engine.model_coefs)) > 0.1
        assert np.max(np.abs(engine.model_coefs)) < 100.0


# ------------------------------------------------------------------ #
# TestEngineImmutability
# ------------------------------------------------------------------ #


class TestEngineImmutability:
    """All key attributes are set after construction with expected types."""

    def test_attributes_exist(self, linear_data):
        X, y = linear_data
        engine = PermutationEngine(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        # These are the documented public attributes
        assert hasattr(engine, "family")
        assert hasattr(engine, "backend_name")
        assert hasattr(engine, "model_coefs")
        assert hasattr(engine, "diagnostics")
        assert hasattr(engine, "perm_indices")

    def test_attribute_types(self, linear_data):
        X, y = linear_data
        engine = PermutationEngine(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert isinstance(engine.family, ModelFamily)
        assert isinstance(engine.backend_name, str)
        assert isinstance(engine.model_coefs, np.ndarray)
        assert isinstance(engine.diagnostics, dict)
        assert isinstance(engine.perm_indices, np.ndarray)

    def test_perm_indices_dtype(self, linear_data):
        X, y = linear_data
        engine = PermutationEngine(
            X,
            y,
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert np.issubdtype(engine.perm_indices.dtype, np.integer)


# ------------------------------------------------------------------ #
# TestFamilyInstancePassthrough
# ------------------------------------------------------------------ #


class TestFamilyInstancePassthrough:
    """Step 1: passing ModelFamily instances directly to the engine."""

    def test_linear_instance_passthrough(self, linear_data):
        """LinearFamily() passed directly resolves correctly."""
        X, y = linear_data
        engine = PermutationEngine(
            X,
            y,
            family=LinearFamily(),
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert engine.family.name == "linear"
        assert isinstance(engine.family, LinearFamily)

    def test_logistic_instance_passthrough(self, binary_data):
        """LogisticFamily() passed directly resolves correctly."""
        X, y = binary_data
        engine = PermutationEngine(
            X,
            y,
            family=LogisticFamily(),
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert engine.family.name == "logistic"
        assert isinstance(engine.family, LogisticFamily)

    def test_nb_preconfigured_alpha_preserved(self, count_data):
        """NegativeBinomialFamily(alpha=2.0) — idempotent calibrate keeps alpha."""
        X, y = count_data
        engine = PermutationEngine(
            X,
            y,
            family=NegativeBinomialFamily(alpha=2.0),
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert engine.family.alpha == 2.0

    def test_nb_uncalibrated_instance_gets_calibrated(self, count_data):
        """NegativeBinomialFamily() without alpha — engine calibrates it."""
        X, y = count_data
        engine = PermutationEngine(
            X,
            y,
            family=NegativeBinomialFamily(),
            n_permutations=_N_PERMS,
            random_state=_SEED,
        )
        assert engine.family.alpha is not None
        assert engine.family.alpha > 0

    def test_instance_validates_y(self, linear_data):
        """ModelFamily instances are validated — explicit choices can be wrong."""
        X, y = linear_data  # continuous y
        with pytest.raises(ValueError, match="binary"):
            PermutationEngine(
                X,
                y,
                family=LogisticFamily(),
                n_permutations=_N_PERMS,
                random_state=_SEED,
            )

    def test_string_validates_y(self, linear_data):
        """Explicit string families are also validated."""
        X, y = linear_data  # continuous y
        with pytest.raises(ValueError, match="binary"):
            PermutationEngine(
                X,
                y,
                family="logistic",
                n_permutations=_N_PERMS,
                random_state=_SEED,
            )

    def test_auto_skips_validate_y(self, linear_data):
        """Auto-detected families skip validate_y (guaranteed compatible)."""
        from unittest.mock import patch

        X, y = linear_data
        with patch.object(LinearFamily, "validate_y") as mock_vy:
            _ = PermutationEngine(
                X,
                y,
                family="auto",
                n_permutations=_N_PERMS,
                random_state=_SEED,
            )
        mock_vy.assert_not_called()

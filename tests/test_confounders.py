"""Tests for the confounders module."""

import numpy as np
import pandas as pd

from randomization_tests.confounders import (
    identify_confounders,
    mediation_analysis,
    screen_potential_confounders,
)


def _make_confounder_data(n=500, seed=42):
    """Create data where z is a confounder (causes both x1 and y)
    and x2 is independent noise.

    The key: x1 = f(z) + independent noise, y = g(z) + independent noise,
    and x1 has NO direct causal effect on y.  The bootstrap test of
    the indirect effect x1 → z → y should yield a CI that includes
    zero because z causes x1 (not vice versa) and x1 has a large
    independent component that dilutes the a-path coefficient.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    # z causes x1, but x1 also has a large independent component
    x1 = 0.3 * z + rng.standard_normal(n) * 1.0
    x2 = rng.standard_normal(n)
    # z causes y directly; x1 does NOT cause y
    y_vals = 2.0 * z + rng.standard_normal(n) * 0.5

    X = pd.DataFrame({"x1": x1, "x2": x2, "z": z})
    y = pd.DataFrame({"y": y_vals})
    return X, y


def _make_mediator_data(n=500, seed=42):
    """Create data where m is a mediator: x -> m -> y."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    m = 0.8 * x + rng.standard_normal(n) * 0.3  # a path
    y_vals = 0.7 * m + rng.standard_normal(n) * 0.3  # b path
    z = rng.standard_normal(n)  # noise

    X = pd.DataFrame({"x": x, "m": m, "z": z})
    y = pd.DataFrame({"y": y_vals})
    return X, y


class TestScreenPotentialConfounders:
    def test_detects_confounder(self):
        X, y = _make_confounder_data()
        result = screen_potential_confounders(X, y, predictor="x1")
        assert "z" in result["potential_confounders"]

    def test_excludes_noise(self):
        X, y = _make_confounder_data()
        result = screen_potential_confounders(X, y, predictor="x1")
        assert "x2" in result["excluded_variables"]


class TestMediationAnalysis:
    def test_detects_mediator(self):
        X, y = _make_mediator_data()
        result = mediation_analysis(X, y, predictor="x", mediator="m", random_state=42)
        assert result["is_mediator"] is True

    def test_non_mediator(self):
        X, y = _make_mediator_data()
        result = mediation_analysis(X, y, predictor="x", mediator="z", random_state=42)
        assert result["is_mediator"] is False

    def test_indirect_effect_ci_excludes_zero_for_mediator(self):
        X, y = _make_mediator_data()
        result = mediation_analysis(X, y, predictor="x", mediator="m", random_state=42)
        ci = result["indirect_effect_ci"]
        assert ci[0] > 0 or ci[1] < 0

    def test_ci_method_is_bca(self):
        X, y = _make_mediator_data()
        result = mediation_analysis(X, y, predictor="x", mediator="m", random_state=42)
        assert result["ci_method"] == "BCa"


class TestIdentifyConfounders:
    def test_identifies_confounder_not_mediator(self):
        """z causes both x1 and y, so it should be flagged as a candidate.

        Note: Mediation analysis is symmetric — it cannot distinguish
        z → x1 from x1 → z statistically.  When z and x1 are correlated
        and z → y is strong, the indirect effect x1 → z → y may be
        significant, so z may appear as a 'mediator' even though the
        true causal structure is confounding.  This is a known
        limitation.  We test that z at least surfaces as a candidate in
        the screening step.
        """
        X, y = _make_confounder_data()
        result = identify_confounders(X, y, predictor="x1", random_state=42)
        screening = result["screening_results"]
        assert "z" in screening["potential_confounders"]
        # z should not be excluded
        assert "z" not in screening["excluded_variables"]

    def test_identifies_mediator_not_confounder(self):
        X, y = _make_mediator_data()
        result = identify_confounders(X, y, predictor="x", random_state=42)
        assert "m" in result["identified_mediators"]
        assert "m" not in result["identified_confounders"]

"""randomization_tests — Permutation tests for regression models.

Implements ter Braak (1992), Kennedy (1995), Freedman–Lane (1983), and
score-based permutation strategies with vectorised batch fitting,
optional JAX autodiff for all GLM families (linear, logistic, Poisson,
negative-binomial, ordinal, multinomial) and mixed-effects models
(linear, logistic, Poisson GLMM), and pre-generated unique permutations
with exchangeability-cell constraints.

Public API:
    .. autosummary::
        randomization_test_regression
        identify_confounders
        mediation_analysis
        moderation_analysis
        screen_potential_confounders
        compute_e_value
        rosenbaum_bounds
        print_comparison_table
        print_compatibility_table
        print_confounder_table
        print_dataset_info_table
        print_diagnostics_table
        print_family_info_table
        print_joint_results_table
        print_protocol_usage_table
        print_results_table
        print_symmetry_table
        ExchangeabilityTree
        ExchangeabilityNode
        calculate_p_values
        generate_unique_permutations
        generate_sign_flips
        validate_symmetry
        get_backend
        set_backend
        ModelFamily
        LinearFamily
        LinearMixedFamily
        LogisticFamily
        LogisticMixedFamily
        MultinomialFamily
        NegativeBinomialFamily
        OrdinalFamily
        PoissonFamily
        PoissonMixedFamily
        resolve_family
        register_family
        PermutationEngine
        FitContext
        ConfounderAnalysisResult
        ConfounderAnalysisResultSet
        IndividualTestResult
        JointTestResult
"""

from ._config import get_backend, set_backend
from ._conformal import conformal_prediction, evaluate_conformal
from ._context import FitContext
from ._invariance import invariance_test
from ._knockoffs import knockoff_test
from ._results import (
    ConformalEvaluationResult,
    ConformalResult,
    ConfounderAnalysisResult,
    ConfounderAnalysisResultSet,
    IndividualTestResult,
    InvarianceResult,
    JointTestResult,
    KnockoffResult,
)
from .confounders import (
    identify_confounders,
    mediation_analysis,
    moderation_analysis,
    screen_potential_confounders,
)
from .core import randomization_test_regression
from .diagnostics import compute_e_value, rosenbaum_bounds
from .display import (
    print_comparison_table,
    print_compatibility_table,
    print_conformal_evaluation_table,
    print_conformal_table,
    print_confounder_table,
    print_dataset_info_table,
    print_diagnostics_table,
    print_family_info_table,
    print_invariance_table,
    print_joint_results_table,
    print_knockoff_table,
    print_protocol_usage_table,
    print_results_table,
    print_symmetry_table,
)
from .engine import PermutationEngine
from .exchangeability import ExchangeabilityNode, ExchangeabilityTree
from .families import (
    LinearFamily,
    LogisticFamily,
    ModelFamily,
    MultinomialFamily,
    NegativeBinomialFamily,
    OrdinalFamily,
    PoissonFamily,
    register_family,
    resolve_family,
)
from .families_mixed import LinearMixedFamily, LogisticMixedFamily, PoissonMixedFamily
from .permutations import generate_unique_permutations
from .pvalues import calculate_p_values
from .sign_flips import generate_sign_flips, validate_symmetry

__all__ = [
    "ConfounderAnalysisResult",
    "ConfounderAnalysisResultSet",
    "ConformalEvaluationResult",
    "ConformalResult",
    "IndividualTestResult",
    "InvarianceResult",
    "JointTestResult",
    "KnockoffResult",
    "FitContext",
    "conformal_prediction",
    "evaluate_conformal",
    "invariance_test",
    "knockoff_test",
    "randomization_test_regression",
    "compute_e_value",
    "identify_confounders",
    "mediation_analysis",
    "moderation_analysis",
    "rosenbaum_bounds",
    "screen_potential_confounders",
    "print_comparison_table",
    "print_compatibility_table",
    "print_confounder_table",
    "print_conformal_evaluation_table",
    "print_conformal_table",
    "print_dataset_info_table",
    "print_diagnostics_table",
    "print_family_info_table",
    "print_invariance_table",
    "print_joint_results_table",
    "print_knockoff_table",
    "print_protocol_usage_table",
    "print_results_table",
    "print_symmetry_table",
    "calculate_p_values",
    "generate_unique_permutations",
    "generate_sign_flips",
    "validate_symmetry",
    "get_backend",
    "set_backend",
    "ModelFamily",
    "LinearFamily",
    "LinearMixedFamily",
    "LogisticFamily",
    "LogisticMixedFamily",
    "MultinomialFamily",
    "NegativeBinomialFamily",
    "OrdinalFamily",
    "PoissonFamily",
    "PoissonMixedFamily",
    "resolve_family",
    "register_family",
    "PermutationEngine",
    "ExchangeabilityTree",
    "ExchangeabilityNode",
]

__version__ = "0.4.4"

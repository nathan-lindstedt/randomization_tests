"""randomization_tests — Permutation tests for regression models.

Implements ter Braak (1992) and Kennedy (1995) methods with vectorised
OLS, optional JAX autodiff for logistic regression, and pre-generated
unique permutations.

Public API:
    .. autosummary::
        permutation_test_regression
        identify_confounders
        mediation_analysis
        screen_potential_confounders
        print_confounder_table
        print_dataset_info_table
        print_diagnostics_table
        print_family_info_table
        print_joint_results_table
        print_protocol_usage_table
        print_results_table
        calculate_p_values
        generate_unique_permutations
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
"""

from ._config import get_backend, set_backend
from ._context import FitContext
from ._results import IndividualTestResult, JointTestResult
from .confounders import (
    identify_confounders,
    mediation_analysis,
    screen_potential_confounders,
)
from .core import permutation_test_regression
from .display import (
    print_confounder_table,
    print_dataset_info_table,
    print_diagnostics_table,
    print_family_info_table,
    print_joint_results_table,
    print_protocol_usage_table,
    print_results_table,
)
from .engine import PermutationEngine
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

__all__ = [
    "IndividualTestResult",
    "JointTestResult",
    "FitContext",
    "permutation_test_regression",
    "identify_confounders",
    "mediation_analysis",
    "screen_potential_confounders",
    "print_confounder_table",
    "print_dataset_info_table",
    "print_diagnostics_table",
    "print_family_info_table",
    "print_joint_results_table",
    "print_protocol_usage_table",
    "print_results_table",
    "calculate_p_values",
    "generate_unique_permutations",
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
]

__version__ = "0.4.1"

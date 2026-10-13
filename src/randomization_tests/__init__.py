r"""Permutation tests and distribution-free inference for statistical regression models.

The ``randomization_tests`` package provides a comprehensive, mathematically rigorous
framework for non-parametric permutation testing, distribution-free conformal prediction,
causal invariance analysis, and controlled feature selection across generalized linear
and mixed-effects regression models.

Core Subsystems
---------------
* Regression Permutation Tests: 9 statistical testing methods (ter Braak, Kennedy,
  Freedman–Lane, Manly, and Rao score) spanning 6 base GLMs and 3 mixed-effects families.
* Double/Debiased Machine Learning (DML): Out-of-fold cross-fitting for non-linear
  confounding absorption with Neyman orthogonality (Chernozhukov et al. 2018).
* Distribution-Free Conformal Prediction: Split conformal and jackknife+ prediction
  intervals with exact finite-sample coverage guarantees (Vovk et al. 2005).
* Controlled Feature Selection: Fixed-X and Model-X knockoff filters with finite-sample
  False Discovery Rate (FDR) control (Barber & Candès 2015, Candès et al. 2018).
* Invariant Causal Prediction: Multi-environment conditional distribution invariance
  testing across discrete and continuous environments (Peters et al. 2016, Berrett et al. 2020).
* Causal Confounder Sieve: Four-stage screening (colliders, mediators, moderators,
  confounders) with bootstrap mediation and sensitivity bounds (E-values, Rosenbaum bounds).

Public API
----------
.. autosummary::
    randomization_test_regression
    identify_confounders
    mediation_analysis
    moderation_analysis
    screen_potential_confounders
    compute_e_value
    rosenbaum_bounds
    conformal_prediction
    evaluate_conformal
    invariance_test
    knockoff_test
    calculate_p_values
    generate_unique_permutations
    generate_sign_flips
    validate_symmetry
    backend
    get_backend
    set_backend
    resolve_family
    register_family
    print_comparison_table
    print_compatibility_table
    print_confounder_table
    print_conformal_table
    print_conformal_evaluation_table
    print_dataset_info_table
    print_diagnostics_table
    print_family_info_table
    print_invariance_table
    print_joint_results_table
    print_knockoff_table
    print_protocol_usage_table
    print_results_table
    print_symmetry_table
    ExchangeabilityTree
    ExchangeabilityNode
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
    PermutationEngine
    FitContext
    FitResult
    IndividualTestResult
    JointTestResult
    ConfounderAnalysisResult
    ConfounderAnalysisResultSet
    ConformalResult
    ConformalEvaluationResult
    InvarianceResult
    KnockoffResult
"""

from ._config import backend, get_backend, set_backend
from ._conformal import conformal_prediction, evaluate_conformal
from ._context import FitContext
from ._invariance import invariance_test
from ._knockoffs import knockoff_test
from ._results import (
    ConformalEvaluationResult,
    ConformalResult,
    ConfounderAnalysisResult,
    ConfounderAnalysisResultSet,
    FitResult,
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
    "FitResult",
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
    "backend",
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

__version__ = "0.5.0"

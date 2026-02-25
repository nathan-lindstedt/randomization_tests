"""v0.4.0 verification gate — structural contract checks."""

import inspect

from randomization_tests import (
    LinearFamily,
    LogisticFamily,
    MultinomialFamily,
    NegativeBinomialFamily,
    OrdinalFamily,
    PoissonFamily,
    pvalues,
)
from randomization_tests._results import (
    IndividualTestResult,
    JointTestResult,
    _DictAccessMixin,
)
from randomization_tests.diagnostics import (
    compute_cooks_distance,
    compute_standardized_coefs,
)
from randomization_tests.display import (
    print_diagnostics_table,
    print_joint_results_table,
    print_results_table,
)

print("=" * 60)
print("v0.4.0 STRUCTURAL CONTRACT VERIFICATION")
print("=" * 60)

# 1. family: ModelFamily on both result classes
for cls in (IndividualTestResult, JointTestResult):
    fam_field = cls.__dataclass_fields__["family"]
    print(f"  {cls.__name__}.family type = {fam_field.type}")
assert "ModelFamily" in str(IndividualTestResult.__dataclass_fields__["family"].type)
assert "ModelFamily" in str(JointTestResult.__dataclass_fields__["family"].type)
print("  [PASS] family: ModelFamily on both result classes")

# 2. _SERIALIZERS on _DictAccessMixin
assert hasattr(_DictAccessMixin, "_SERIALIZERS")
print(f"  _SERIALIZERS = {_DictAccessMixin._SERIALIZERS}")
print("  [PASS] _SERIALIZERS registry on _DictAccessMixin")

# 3. Display functions are self-contained
for fn in (print_results_table, print_joint_results_table, print_diagnostics_table):
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    assert "feature_names" not in params, f"{fn.__name__} has feature_names!"
    assert "target_name" not in params, f"{fn.__name__} has target_name!"
    assert "family" not in params, f"{fn.__name__} has family as param!"
    print(f"  {fn.__name__} params: {params}")
print("  [PASS] Display functions are self-contained")

# 4. compute_standardized_coefs and compute_cooks_distance take family
for fn in (compute_standardized_coefs, compute_cooks_distance):
    sig = inspect.signature(fn)
    assert "family" in sig.parameters, f"{fn.__name__} missing family!"
    print(f"  {fn.__name__}: has family param")
print("  [PASS] Diagnostics functions take family: ModelFamily")

# 5. _classical_p_values_fallback deleted
assert not hasattr(pvalues, "_classical_p_values_fallback")
print("  [PASS] _classical_p_values_fallback deleted")

# 6. All 6 families implement the 4 required methods
for fam_cls in (
    LinearFamily,
    LogisticFamily,
    PoissonFamily,
    NegativeBinomialFamily,
    OrdinalFamily,
    MultinomialFamily,
):
    fam = fam_cls()
    for attr in (
        "stat_label",
        "display_header",
        "display_diagnostics",
        "compute_extended_diagnostics",
    ):
        assert hasattr(fam, attr), f"{fam_cls.__name__} missing {attr}"
    print(f"  {fam_cls.__name__}: stat_label={fam.stat_label!r}, all 4 present")
print("  [PASS] All 6 families implement stat_label + display + diagnostics")

print()
print("ALL STRUCTURAL CONTRACTS VERIFIED")

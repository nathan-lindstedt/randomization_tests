"""v0.4.1 verification gate — restricted same-size cell swap correctness.

Validates that the between-cell and two-stage permutation generators
only swap cells of equal size, producing exclusively valid permutations
of [0..n-1].  This gate was introduced after a bug where unbalanced-
cell cycling (tgt[np.arange(src_len) % tgt_len]) created duplicate
indices in the output.
"""

import warnings

import numpy as np

from randomization_tests.permutations import (
    generate_between_cell_permutations,
    generate_two_stage_permutations,
)


def check_validity(name, result, n):
    invalid = 0
    for row in result:
        if sorted(row) != list(range(n)):
            invalid += 1
            print(f"  INVALID: {row}")
    print(f"  {name}: {result.shape[0]} rows, {invalid} invalid")
    return invalid


total_invalid = 0

# Test 1: unbalanced [3,2] — two-stage
cells1 = np.array([0, 0, 0, 1, 1])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r1 = generate_two_stage_permutations(5, 20, cells1, random_state=42)
    warned = len(w) > 0
print(f"Two-stage unbalanced [3,2] (warned: {warned}):")
total_invalid += check_validity("unbalanced", r1, 5)

# Test 2: balanced [2,2,2] — two-stage
cells2 = np.array([0, 0, 1, 1, 2, 2])
r2 = generate_two_stage_permutations(6, 50, cells2, random_state=42)
print("Two-stage balanced [2,2,2]:")
total_invalid += check_validity("balanced", r2, 6)

# Test 3: partially balanced [2,2,2,3] — between-cell
cells3 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])
r3 = generate_between_cell_permutations(9, 5, cells3, random_state=42)
print("Between partially balanced [2,2,2,3]:")
total_invalid += check_validity("partial", r3, 9)

# Test 4: fully unbalanced [1,2,3] — between-cell
cells4 = np.array([0, 1, 1, 2, 2, 2])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r4 = generate_between_cell_permutations(6, 5, cells4, random_state=42)
    warned4 = len(w) > 0
print(f"Between fully unbalanced [1,2,3] (warned: {warned4}):")
total_invalid += check_validity("full-unbal", r4, 6)

print()
if total_invalid == 0:
    print("ALL OUTPUTS VALID!")
else:
    print(f"BUGS REMAIN: {total_invalid} invalid rows")

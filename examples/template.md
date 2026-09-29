# Example Script Style Guide & Pedagogical Template

This document defines the architectural, pedagogical, and visual standards for all interactive tutorial scripts in the `examples/` directory.

Every script in `examples/` is designed to serve a dual purpose:
1. **Interactive Execution**: Structured with `# %%` cell markers for interactive execution cell-by-cell in the VS Code interactive Python window.
2. **Pedagogical Walkthrough**: Grounded in real-world empirical datasets from the UCI Machine Learning Repository, demonstrating how randomization tests resolve specific parametric failure modes (e.g. heteroscedasticity, clustering, overdispersion, separation, non-linear confounding).

---

## 1. Top-Level Module Docstring Architecture

Every script must begin with an academic docstring block marked as `# %% [markdown]`. It must follow this exact 6-section structure with 4-space indentation:

```python
# %% [markdown]
"""
Example: <Model Family / Procedure Name> (<Outcome Type & Design Description>)
Dataset: <Dataset Name> (UCI Machine Learning Repository ID=<ID>)

Dataset Context & Theoretical Background:
    Provide 2-3 substantive paragraphs establishing:
    1. The domain context, sample size, observation units, and clinical/economic
       significance of the outcome variable.
    2. The empirical distribution of the outcome (continuous, binary, count,
       ordered categorical, multinomial, panel/clustered).
    3. The structural or mathematical failure mode of naive parametric models
       on this data (e.g. OLS under non-Gaussian residuals or spatial clustering;
       logistic MLE under Hauck-Donner near-separation; Poisson GLM under
       overdispersion; linear adjustment under non-linear confounding).

Features Selected for Modeling:
    - <feature 1>: Domain interpretation, measurement scale, and scientific role.
    - <feature 2>: Domain interpretation, measurement scale, and scientific role.
    - Confounders (Z):
      * <confounder 1>: Specific causal or spatial confounding rationale.

Methodological Rationale for Resampling Tests:
    1. <Core Methodological Principle 1>:
       Why permutation, sign-flipping, or double-residualization is required
       (e.g., conditional symmetry vs. exchangeability; block exchangeability;
       Neyman orthogonality).
    2. <Core Methodological Principle 2>:
       How the specific procedure (ter Braak, Freedman-Lane, Kennedy, Score,
       or DML) conditions on nuisance covariates or handles the likelihood scale.
    3. <Outcome Reconstruction & Null Preservation>:
       How permuted responses or scores are evaluated (e.g., stochastic
       Bernoulli/Poisson draws, whitened GLS projections, or matrix FWL).

Demonstrates:
    - family="<family_name>" -- <brief family description>
    - <procedure 1> (<citation>) -- <brief description>
    - <procedure 2> (<citation>) -- <brief description>
    - <diagnostic or comparison feature>
    - Execution and protocol artifacts inspection via print_protocol_usage_table

References:
    - <Author(s)>. (<Year>). <Title>. <Journal/Book>, <Volume>(<Issue>), <Pages>.
    - <Method Citation 1>
    - <Method Citation 2>
"""
```

---

## 2. Cell Banners and Script Cadence

Interactive cells must be delineated by `# %%` and preceded by an 80-column header banner:

```python
# %%
# ============================================================================
# Section Title — Optional Brief Clarifier
# ============================================================================
```

### Standard Cell Cadence

All example scripts should progress through this unified pedagogical sequence:

1. **Imports & Setup**:
   Import needed estimators from `sklearn`, data fetchers from `ucimlrepo`, and display/engine functions from `randomization_tests`.
2. **`Load data`**:
   Fetch the UCI repository dataset by integer ID, extract feature `DataFrame` `X` and target `DataFrame` `y`.
3. **`Dataset Information`**:
   Call `print_dataset_info_table(name=..., X=X, y=y)`.
4. **`Verify resolve_family ...`** (when applicable):
   Demonstrate family resolution via `resolve_family()` and render properties via `print_family_info_table()`.
5. **`Confounder screening`**:
   Execute `identify_confounders(X, y)` and display the causal partition via `print_confounder_table()`.
6. **`Baseline / Unconstrained Test`**:
   Run the primary baseline permutation test (e.g. standard Freedman–Lane, ter Braak, or unwhitened model) and display results via `print_results_table()` and `print_diagnostics_table()`.
7. **`Specialized / Controlled Test`**:
   Run the advanced method addressing the data's specific complexity (e.g. DML with flexible ML nuisance models, random slopes LMM, AR(p) whitening, or sign-flip symmetry test).
8. **`Side-by-side comparison`**:
   Contrast baseline vs. specialized model using `print_comparison_table([("Model A", res_a), ("Model B", res_b)])`.
9. **`Joint omnibus test`**:
   Evaluate collective feature improvement using `print_joint_results_table()`.
10. **`Protocol and execution artifacts`**:
    Inspect computational metadata, convergence counts, and inferential guarantees using `print_protocol_usage_table(result)`.

---

## 3. Explanatory Commentary Style

1. **Concise Decision Commentary**:
   Include a short, focused 2–4 line explanatory comment directly beneath the section banner for critical modeling decisions. Explain *why* a particular model choice was made and *what* scientific insight it yields:
   ```python
   # %%
   # ============================================================================
   # Double/Debiased Machine Learning (DML) via GradientBoostingRegressor
   # ============================================================================
   # In standard linear FL, convenience store density (X4) appears significant
   # (p = 0.002) due to spatial sorting. Cross-fitting both Y and X on spatial
   # confounders (Z) via GBDT purges non-linear geographic bias, revealing that
   # store density has no independent causal effect on house prices.

   reduced_model = GradientBoostingRegressor(
       n_estimators=50,
       max_depth=3,
       learning_rate=0.1,
       random_state=42,
   )
   ```

2. **No Line-by-Line Micro-Commentary**:
   Avoid redundant comments that simply re-state the Python code (e.g. do NOT write `# extract features` above `X = ds.data.features`).

3. **No Defensive Scripting**:
   User-facing tutorial scripts must never contain defensive assertions (`assert`), fallback ternaries, or manual error checking. The package APIs are designed to handle data containers natively and communicate status via formatted tables and structured notes.

4. **Table-First Output**:
   Never use raw `print()` statements for model outputs. Always format tables using the package's dedicated display utilities (`print_results_table`, `print_diagnostics_table`, `print_comparison_table`, etc.).

---

## 4. Complete Annotated Example Template

```python
# %% [markdown]
"""
Example: Linear Regression (Continuous Outcome)
Dataset: Real Estate Valuation (UCI Machine Learning Repository ID=477)

Dataset Context & Theoretical Background:
    ...

Features Selected for Modeling:
    - ...

Methodological Rationale for Resampling Tests:
    1. ...
    2. ...

Demonstrates:
    - ...

References:
    - ...
"""

# %%
from ucimlrepo import fetch_ucirepo

from randomization_tests import (
    identify_confounders,
    print_comparison_table,
    print_confounder_table,
    print_dataset_info_table,
    print_diagnostics_table,
    print_joint_results_table,
    print_protocol_usage_table,
    print_results_table,
    randomization_test_regression,
)

# %%
# ============================================================================
# Load data
# ============================================================================

ds = fetch_ucirepo(id=477)
X = ds.data.features
y = ds.data.targets

print_dataset_info_table(
    name=ds.metadata.name,
    X=X,
    y=y,
)

# %%
# ============================================================================
# Confounder screening
# ============================================================================

confounder_results = identify_confounders(X, y)
print_confounder_table(confounder_results)

confounders = [
    "X3 distance to the nearest MRT station",
    "X5 latitude",
    "X6 longitude",
]

# %%
# ============================================================================
# Freedman–Lane (1983) individual — baseline
# ============================================================================

res_baseline = randomization_test_regression(
    X,
    y,
    confounders=confounders,
    method="freedman_lane",
    n_randomizations=500,
    random_state=42,
)
print_results_table(res_baseline)

# %%
# ============================================================================
# Primary Model with Covariate Control
# ============================================================================
# Concise 2-4 line explanatory commentary explaining the rationale for
# this modeling specification and how it addresses the domain challenge.

res_model = randomization_test_regression(
    X,
    y,
    confounders=confounders,
    method="ter_braak",
    n_randomizations=500,
    random_state=42,
)
print_results_table(res_model)
print_diagnostics_table(res_model)

# %%
# ============================================================================
# Side-by-side comparison
# ============================================================================

print_comparison_table([("Baseline", res_baseline), ("Alternative", res_model)])

# %%
# ============================================================================
# Freedman–Lane (1983) joint omnibus test
# ============================================================================

res_joint = randomization_test_regression(
    X,
    y,
    confounders=confounders,
    method="freedman_lane_joint",
    n_randomizations=500,
    random_state=42,
)
print_joint_results_table(res_joint)

# %%
# ============================================================================
# Protocol and execution artifacts
# ============================================================================

print_protocol_usage_table(res_model)
```

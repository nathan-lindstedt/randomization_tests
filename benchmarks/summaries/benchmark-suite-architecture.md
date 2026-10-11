# Benchmark Suite Architecture & Methodology

## 1. Overview & Purpose

The `benchmarks/` suite provides empirical verification of algorithmic complexity,
numerical precision, throughput scaling, and statistical validity across all
components of the `randomization_tests` framework.

The suite is structured into six complementary profiling modules:

| Script                                               | Domain                                 | Key Research Questions & Theoretical Targets                                                                                                                                                                             | Primary Outputs                                               |
| ---------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| [`profile_solvers.py`](../profile_solvers.py)       | Numerical Solvers & Linear Algebra     | Autodiff exactness ($\|g_{\text{AD}} - g_{\text{FD}}\|_\infty < 10^{-5}$), condition number scaling $\kappa(X)$, $O(G)$ Woodbury whitening vs $O(N^3)$, single-fit parity across all 7 families.                 | `results/solvers_profile.csvimage/solvers-profile/`         |
| [`profile_endtoend.py`](../profile_endtoend.py)     | End-to-End Regression Inference        | Full pipeline throughput across all 7 families, all 10 statistical methods, 4 permutation strategies (`unrestricted`, `within`, `between`, `two-stage`), and predictor scaling grids ($n \times p$).           | `results/endtoend_profile.csvimage/endtoend-profile/`       |
| [`profile_subsystems.py`](../profile_subsystems.py) | Advanced Statistical Subsystems        | Kernel tests (HSIC, MMD, KernelReg), Conformal Prediction (Split, Jackknife+, Kernel-Adaptive, Clustered), Knockoff FDR Filter (`fixed_x`, `equicorrelated`), Invariance tests ($H_0: Y \perp\!\!\perp E \mid X$). | `results/subsystems_profile.csvimage/subsystems-profile/`   |
| [`profile_dml.py`](../profile_dml.py)               | Double/Debiased Machine Learning       | DML double-residualization vs single-residualized Freedman–Lane under linear confounding, nonlinear confounder leakage ($H_0$), signal recovery, cluster leakage, and kernel regimes.                                 | `results/dml_profile.csvimage/dml-profile/`                 |
| [`profile_dedup.py`](../profile_dedup.py)           | Permutation Generation & Deduplication | Multi-tier permutation generation: Lehmer-code unranking ($O(B \cdot n)$), hash-set deduplication, birthday paradox collision bounds ($P \le 10^{-9}$), within-cell and between-cell scaling.                        | `results/dedup_profile.csvimage/permutation-dedup-profile/` |
| [`profile_realworld.py`](../profile_realworld.py)   | Stress Testing & Irregular Designs     | Clinical trial designs (unbalanced RCTs), educational cluster designs (classrooms in schools), longitudinal panels with severe imbalance, validity proofs (no duplicates, no identity).                                  | `results/realworld_profile.csvimage/realworld-profile/`     |

---

## 2. Benchmark Module Details

### 2.1 `profile_solvers.py` (Numerical Solvers & Whitening)

- **Autodiff Exactness vs Finite Differences**: Compares analytical JAX autodiff gradients and Hessians against numerical central finite differences on non-trivial likelihood functions. Demonstrates exact machine-precision match with zero truncation error.
- **Single-Model Fit Latency & Parity**: Profiles calibration and fitting times across `LinearFamily`, `LogisticFamily`, `PoissonFamily`, `NegativeBinomialFamily`, `OrdinalFamily`, `MultinomialFamily`, and `LinearMixedFamily`.
- **Condition Number Scaling**: Evaluates SVD `lstsq` vs Cholesky/QR normal equations as condition number $\kappa(X) \to 10^{12}$, demonstrating that forward error scales as $\kappa(X) \cdot \epsilon_{\text{mach}}$ without error squaring $\kappa^2(X) \cdot \epsilon_{\text{mach}}$.
- **Woodbury Block-Diagonal Whitening**: Benchmarks random-effects whitening runtime as cluster count $G$ scales from 10 to 200 ($N$ from 500 to 10,000). Proves empirical execution time scales strictly linearly as $O(G)$, avoiding the catastrophic $O(N^3)$ cost of naive dense matrix inversion.
- **Prais–Winsten AR Overhead**: Measures residual autocorrelation estimation ($\hat{\rho}$) and Cochrane–Orcutt / Prais–Winsten transformation overhead.

### 2.2 `profile_endtoend.py` (Full Regression Engine Pipeline)

- **Method Shootout**: Direct empirical comparison of wall time and throughput (permutations per second) at matched dimensions ($n=500, p=5, B=999$):
  - *Score tests* (`score`, `score_joint`): Rao score updates evaluated in the null tangent space without refitting achieve $> 25{,}000$ permutations/second.
  - *Residual resampling* (`freedman_lane`, `freedman_lane_joint`, `ter_braak`, `kennedy`, `kennedy_joint`).
  - *Direct outcome resampling* (`manly`, `manly_joint`).
  - *Sign flipping* (`randomization="sign_flip"`).
- **Family Architecture Scaling**: Matched runtime comparison across all 7 families under both refit methods and score test updates.
- **Permutation Strategy Overhead**: Measures relative overhead of `within`, `between`, and `two-stage` strategies relative to flat `unrestricted` permutation.
- **Dimension Grids ($n \times p$)**: Verifies that predictor count $p$ and sample size $n$ scale gracefully up to $n=1000, p=30$.

### 2.3 `profile_subsystems.py` (Subsystems & Post-Inference Guarantees)

- **Kernel Hypothesis Tests**: Evaluates `hsic_test`, `mmd_test`, and `kernel_regression_test` across $n \in [100, 1000]$, verifying quadratic $O(n^2)$ scaling and the Gram matrix 2 GiB memory check.
- **Distribution-Free Conformal Prediction**: Verifies exact empirical coverage $\ge 1 - \alpha$ across Split conformal, Jackknife+, Locally-weighted adaptive (Nadaraya–Watson kernel variance), and Clustered conformal prediction.
- **Knockoff Filter Variable Selection**: Compares `fixed_x` (finite-sample exact FDR control) against `equicorrelated` Model-X knockoffs, demonstrating $< 3\text{ms}$ knockoff construction and Lasso path thresholding up to $p=100$.
- **Conditional Invariance Testing**: Measures wall-clock execution for $H_0: Y \perp\!\!\perp E \mid X$ across varying environment counts $E \in [2, 8]$.

### 2.4 `profile_dml.py` (Double/Debiased Machine Learning)

- Empirical verification of Robinson (1988) / Chernozhukov et al. (2018) DML:
  - **Scenario 1 (Linear Confounding Baseline)**: Exact equivalence to canonical Freedman–Lane under standard linear confounding ($\text{bias} \approx 0.01$, nominal size $0.075$).
  - **Scenario 2 (Nonlinear Confounder Leakage under $H_0$)**:
    * *DGP Design & Harmonic Parity*: In earlier versions, choosing $m(Z) = \sin(2Z) + 0.5 Z^2$ and $g(Z) = \cos(2Z) + 0.5 Z^2$ over the symmetric interval $Z \in [-2, 2]$ caused harmonic parity cancellation ($\text{Cov}(\tilde{m}, \tilde{g}) \approx -0.0132$), masking the omitted variable bias.
    * *Remediated Aligned Confounding*: Using $m(Z) = \sin(2Z) + Z^2$ and $g(Z) = 1.5 Z^2$ establishes strong non-zero linear residual covariance $\text{Cov}(\tilde{m}, \tilde{g}) = 1.5 \text{Var}(Z^2 - \mathbb{E}Z^2) \approx 2.133 \gg 0$.
    * *Empirical Outcome*: Linear Freedman–Lane suffers catastrophic $100\%$ Type I error inflation ($\hat{\beta} \approx 1.026$ under true $\beta = 0.0$). Double-Residualized DML (GBDT) absorbs the nonlinear confounding out-of-fold, recovering nominal calibration ($\text{rejection rate} \approx 0.17 \to 0.05$).
  - **Scenario 3 (Signal Recovery & Power Collapse)**: Under marginal uncorrelation $\mathbb{E}[Y|Z] = 0$, linear FL suffers total power collapse and severe bias ($\hat{\beta} = -0.239$ vs true $\beta = -2.0$). Cross-fitted DML accurately recovers $\hat{\beta} = -1.896$ with $100\%$ empirical power.
  - **Scenario 4 (Clustered Confounding & Fold Leakage)**: Demonstrates that standard `KFold` allows cross-fold cluster leakage, while `GroupKFold` enforces strict cluster boundary isolation.
  - **Scenario 5 (Logistic / Binary Classification GPLM)**: Evaluates probability-residual score DML for binary outcomes. Under nonlinear confounding, classical logistic FL suffers $100\%$ false positive inflation, while Probability Score DML restores nominal calibration ($0.075$).
  - **Scenario 6 (Kernel HSIC Partial Independence Leak)**: Demonstrates that OLS hat matrix projection fails to remove nonlinear dependence from kernel residuals (HSIC rejects $H_0$ $100\%$ of the time), while cross-fitted DML orthogonalizes the residuals and restores nominal non-rejection ($0.025$).

### 2.5 `profile_dedup.py` & `profile_realworld.py` (Combinatorial Permutations)

- **Deduplication Regimes**:
  - Small $n \le 10$: Exact Lehmer unranking ($O(B \cdot n)$), eliminating duplicates by design.
  - Intermediate $10 < n < 20$: Vectorized generation with Python hash-set deduplication.
  - Large $n \ge 20$: Pure vectorized generation (birthday paradox collision probability $P_{\text{collision}} < 10^{-9}$).
- **Real-World Stress Tests**:
  - Tests unbalanced multi-site clinical trials, educational classroom cluster models, and unbalanced longitudinal panels.
  - Asserts 100% validity across all rows (no duplicate permutations, no identity permutations, valid permutation of $[0..n-1]$).

---

## 3. Running the Benchmark Suite

### Fast Smoke-Test Execution (< 30 seconds total)

```bash
conda activate randomization-tests
python benchmarks/profile_solvers.py --quick
python benchmarks/profile_endtoend.py --quick
python benchmarks/profile_subsystems.py --quick
python benchmarks/profile_dml.py --quick
python benchmarks/profile_dedup.py --quick
python benchmarks/profile_realworld.py --quick
```

### Full Production Profiling Runs

```bash
conda activate randomization-tests
python benchmarks/profile_solvers.py
python benchmarks/profile_endtoend.py
python benchmarks/profile_subsystems.py
python benchmarks/profile_dml.py
python benchmarks/profile_dedup.py
python benchmarks/profile_realworld.py
```

All outputs are saved as structured CSV files in `benchmarks/results/` and publication-quality vector/PNG visualizations in `benchmarks/image/`.

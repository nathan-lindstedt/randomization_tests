#%%
# ============================================================================
# Imports
# ============================================================================

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from ucimlrepo import fetch_ucirepo 

#%%
# ============================================================================
# Display Utilities
# ============================================================================

def _truncate(name: str, max_len: int) -> str:
    """Truncate a name to max_len, appending '...' if needed."""
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + '...'

def print_results_table(
    results: Dict,
    feature_names: List[str],
    target_name: str = None,
    title: str = "Permutation Test Results"
) -> None:
    """
    Print regression results in a formatted ASCII table similar to statsmodels.
    
    Parameters
    ----------
    results : dict
        Results dictionary from permutation_test_regression.
    feature_names : list of str
        Names of the features/predictors.
    target_name : str, optional
        Name of the target variable.
    title : str, optional
        Title for the output table.
    """
    # Header
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    # Model info
    diag = results.get('diagnostics', {})
    model_type = results['model_type']
    col1 = 40  # left column width
    col2 = 38  # right column width (total = 80 with 2 spaces)
    if target_name:
        trunc_target = _truncate(target_name, 20)
        print(f"{'Dep. Variable:':<16}{trunc_target:<{col1 - 16}}{'No. Observations:':>{col2 - 11}} {diag.get('n_observations', 'N/A'):>10}")
    print(f"{'Model Type:':<16}{model_type:<{col1 - 16}}{'No. Features:':>{col2 - 11}} {diag.get('n_features', 'N/A'):>10}")
    print(f"{'Method:':<16}{results['method']:<{col1 - 16}}{'AIC:':>{col2 - 11}} {diag.get('aic', 'N/A'):>10}")
    
    if model_type == 'linear':
        print(f"{'R-squared:':<16}{diag.get('r_squared', 'N/A'):<{col1 - 16}}{'BIC:':>{col2 - 11}} {diag.get('bic', 'N/A'):>10}")
        print(f"{'Adj. R-squared:':<16}{diag.get('r_squared_adj', 'N/A'):<{col1 - 16}}{'F-statistic:':>{col2 - 11}} {diag.get('f_statistic', 'N/A'):>10}")
        f_p = diag.get('f_p_value', None)
        f_p_str = f"{f_p:.4e}" if f_p is not None else 'N/A'
        print(f"{'':<{col1}}{'Prob (F-stat):':>{col2 - 11}} {f_p_str:>10}")
    else:
        print(f"{'Pseudo R-sq:':<16}{diag.get('pseudo_r_squared', 'N/A'):<{col1 - 16}}{'BIC:':>{col2 - 11}} {diag.get('bic', 'N/A'):>10}")
        print(f"{'Log-Likelihood:':<16}{diag.get('log_likelihood', 'N/A'):<{col1 - 16}}{'LL-Null:':>{col2 - 11}} {diag.get('log_likelihood_null', 'N/A'):>10}")
        llr_p = diag.get('llr_p_value', None)
        llr_p_str = f"{llr_p:.4e}" if llr_p is not None else 'N/A'
        print(f"{'':<{col1}}{'LLR p-value:':>{col2 - 11}} {llr_p_str:>10}")
    
    print("-" * 80)
    
    # Column headers
    fc = 25  # feature column width
    stat_label = 't' if model_type == 'linear' else 'z'
    emp_hdr = f'P>|{stat_label}| (Emp)'
    asy_hdr = f'P>|{stat_label}| (Asy)'
    print(f"{'Feature':<{fc}} {'Coef':>12} {emp_hdr:>18} {asy_hdr:>18}")
    print("-" * 80)
    
    # Data rows
    coefs = results['model_coefs']
    emp_p = results['permuted_p_values']
    asy_p = results['classic_p_values']
    
    for i, feat in enumerate(feature_names):
        trunc_feat = _truncate(feat, fc)
        coef_str = f"{coefs[i]:>12.4f}"
        print(f"{trunc_feat:<{fc}} {coef_str} {emp_p[i]:>18} {asy_p[i]:>18}")
    
    # Footer
    print("=" * 80)
    print(f"(*) p < {results['p_value_threshold_one']}   "
          f"(**) p < {results['p_value_threshold_two']}   "
          f"(ns) p >= {results['p_value_threshold_one']}")
    print()

def print_joint_results_table(
    results: Dict,
    target_name: str = None,
    title: str = "Joint Permutation Test Results"
) -> None:
    """
    Print joint test results in a formatted ASCII table.
    
    Parameters
    ----------
    results : dict
        Results dictionary from permutation_test_regression with method='kennedy_joint'.
    target_name : str, optional
        Name of the target variable.
    title : str, optional
        Title for the output table.
    """
    # Header
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    # Model info
    diag = results.get('diagnostics', {})
    model_type = results['model_type']
    col1 = 40  # left column width
    col2 = 38  # right column width (total = 80 with 2 spaces)
    if target_name:
        trunc_target = _truncate(target_name, 20)
        print(f"{'Dep. Variable:':<16}{trunc_target:<{col1 - 16}}{'No. Observations:':>{col2 - 11}} {diag.get('n_observations', 'N/A'):>10}")
    print(f"{'Model Type:':<16}{model_type:<{col1 - 16}}{'No. Features:':>{col2 - 11}} {diag.get('n_features', 'N/A'):>10}")
    print(f"{'Method:':<16}{results['method']:<{col1 - 16}}{'AIC:':>{col2 - 11}} {diag.get('aic', 'N/A'):>10}")
    
    if model_type == 'linear':
        print(f"{'R-squared:':<16}{diag.get('r_squared', 'N/A'):<{col1 - 16}}{'BIC:':>{col2 - 11}} {diag.get('bic', 'N/A'):>10}")
        print(f"{'Adj. R-squared:':<16}{diag.get('r_squared_adj', 'N/A'):<{col1 - 16}}{'F-statistic:':>{col2 - 11}} {diag.get('f_statistic', 'N/A'):>10}")
        f_p = diag.get('f_p_value', None)
        f_p_str = f"{f_p:.4e}" if f_p is not None else 'N/A'
        print(f"{'':<{col1}}{'Prob (F-stat):':>{col2 - 11}} {f_p_str:>10}")
    else:
        print(f"{'Pseudo R-sq:':<16}{diag.get('pseudo_r_squared', 'N/A'):<{col1 - 16}}{'BIC:':>{col2 - 11}} {diag.get('bic', 'N/A'):>10}")
        print(f"{'Log-Likelihood:':<16}{diag.get('log_likelihood', 'N/A'):<{col1 - 16}}{'LL-Null:':>{col2 - 11}} {diag.get('log_likelihood_null', 'N/A'):>10}")
        llr_p = diag.get('llr_p_value', None)
        llr_p_str = f"{llr_p:.4e}" if llr_p is not None else 'N/A'
        print(f"{'':<{col1}}{'LLR p-value:':>{col2 - 11}} {llr_p_str:>10}")
    
    print(f"{'Metric:':<16}{results['metric_type']}")
    print("-" * 80)
    
    # Features tested
    feat_list = ', '.join(_truncate(f, 25) for f in results['features_tested'])
    print(f"Features Tested: {feat_list}")
    if results['confounders']:
        conf_list = ', '.join(_truncate(c, 25) for c in results['confounders'])
        print(f"Confounders: {conf_list}")
    print("-" * 80)
    
    # Results
    print(f"{'Observed Improvement:':<30} {results['observed_improvement']:>12.4f}")
    print(f"{'Joint p-Value:':<30} {results['p_value_str']:>12}")
    
    # Footer
    print("=" * 80)
    print(f"(*) p < {results['p_value_threshold_one']}   "
          f"(**) p < {results['p_value_threshold_two']}   "
          f"(ns) p >= {results['p_value_threshold_one']}")
    print()

#%%
# ============================================================================
# P-Value Calculation
# ============================================================================

def calculate_p_values(
    X: pd.DataFrame, 
    y: pd.DataFrame,
    permuted_coefs: List,
    model_coefs: List,
    precision: int = 3,
    p_value_threshold_one: float = 0.05, 
    p_value_threshold_two: float = 0.01
) -> Tuple[List[str], List[str]]:
    """
    Calculate p-values for a given model using both permutation and classical methods.
    
    Automatically detects binary vs continuous outcomes and uses logistic or
    linear regression accordingly for classical p-values.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to fit the model.
    y : array-like of shape (n_samples,)
        The target values (binary for logistic, continuous for linear regression).
    permuted_coefs : list of float
        The coefficients obtained from the permutation test.
    model_coefs : list of float
        The original coefficients of the fitted model.
    precision : int, optional (default=3)
        The number of decimal places to round the p-values.
    p_value_threshold_one : float, optional (default=0.05)
        The threshold for the first level of significance.
    p_value_threshold_two : float, optional (default=0.01)
        The threshold for the second level of significance.

    Returns
    ----------
    permuted_p_values : list of str
        The empirical p-values for each coefficient, formatted with significance stars if below the thresholds.
    classic_p_values : list of str
        The asymptotic p-values for each coefficient, formatted with significance stars if below the thresholds.
    """
    permuted_p_values: List = []
    classic_p_values: List = []
    
    # Detect model type based on Y
    y_values = np.ravel(y)
    unique_y = np.unique(y_values)
    is_binary = (len(unique_y) == 2) and np.all(np.isin(unique_y, [0, 1]))

    if is_binary:
        # Logistic regression via maximum likelihood. The log-likelihood is
        # l(beta) = sum[y_i * log(p_i) + (1 - y_i) * log(1 - p_i)],
        # where p_i = 1 / (1 + exp(-X_i * beta)). Statsmodels computes
        # Wald z-statistics and two-sided p-values for each coefficient.
        model = sm.Logit(y_values, sm.add_constant(X)).fit(disp=0)
    else:
        # OLS regression minimizes sum(Y - X*beta)^2. The solution is
        # beta_hat = (X'X)^(-1) X'Y. Statsmodels computes t-statistics
        # using SE(beta_hat) = sqrt(s^2 * diag((X'X)^(-1))), where
        # s^2 = RSS / (n - p - 1) is the residual variance estimate.
        model = sm.OLS(y, sm.add_constant(X)).fit()

    # Empirical (permutation) p-values. For each coefficient, count how
    # many permuted |beta*_j| are at least as extreme as the observed
    # |beta_j|. The Phipson & Smyth (2010) correction adds 1 to both
    # numerator and denominator: p = (b + 1) / (B + 1), where
    # b = #{|beta*_j| >= |beta_j|}. This ensures the p-value is never
    # exactly zero and properly treats the observed statistic as one
    # member of the reference set of B + 1 values.
    n_permutations = len(permuted_coefs)
    for i in range(len(model_coefs)):
        # Phipson & Smyth (2010) correction: (b + 1) / (B + 1)
        # Ensures p-values are never exactly 0 in finite permutation tests
        p_value = (np.sum(np.abs(np.array(permuted_coefs)[:, i]) >= np.abs(np.array(model_coefs)[i])) + 1) / (n_permutations + 1)
        if p_value_threshold_two <= p_value < p_value_threshold_one:
            p_value_str = str(np.round(p_value, precision)) + ' (*)'
        elif p_value < p_value_threshold_two:
            p_value_str = str(np.round(p_value, precision)) + ' (**)'
        else:
            p_value_str = str(np.round(p_value, precision)) + ' (ns)'
        
        permuted_p_values.append(p_value_str)
    
    # Classical asymptotic p-values from statsmodels. model.pvalues[0] is
    # the intercept; we skip it since the permutation results index
    # coefficients without the intercept term.
    for p_value in model.pvalues[1:]:
        if p_value_threshold_two <= p_value < p_value_threshold_one:
            p_value_str = str(np.round(p_value, precision)) + ' (*)'
        elif p_value < p_value_threshold_two:
            p_value_str = str(np.round(p_value, precision)) + ' (**)'
        else:
            p_value_str = str(np.round(p_value, precision)) + ' (ns)'

        classic_p_values.append(p_value_str)

    return permuted_p_values, classic_p_values

#%%
# ============================================================================
# Confounder Analysis
#
# In observational studies, confounders can bias the estimated effect of a
# predictor X on an outcome Y. A confounder Z satisfies three conditions:
#   1. Z is associated with X
#   2. Z is associated with Y
#   3. Z is NOT on the causal path from X to Y (i.e., Z is not a mediator)
#
# Controlling for a confounder removes bias; controlling for a mediator
# removes part of the true causal effect and should generally be avoided.
#
# The workflow in this section:
#   1. screen_potential_confounders: identifies variables correlated with
#      both X and Y using Pearson correlation (conditions 1 & 2).
#   2. mediation_analysis: tests whether each candidate is a mediator
#      using Baron & Kenny (1986) with bootstrap CIs for the indirect
#      effect (condition 3).
#   3. identify_confounders: orchestrates both steps to distinguish
#      confounders from mediators.
# ============================================================================

def screen_potential_confounders(
    X: pd.DataFrame,
    y: pd.DataFrame,
    predictor: str,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05
) -> Dict:
    """
    Screen for potential confounders using correlation analysis.
    
    A potential confounder is a variable that is correlated with both the predictor
    of interest (X) and the outcome (Y). This function identifies candidates that
    should be further investigated through mediation analysis or domain knowledge.

    Parameters
    ----------
    X : pd.DataFrame
        The input data containing all features.
    y : pd.DataFrame
        The target variable.
    predictor : str
        The name of the predictor variable of interest.
    correlation_threshold : float, optional (default=0.1)
        Minimum absolute correlation to consider a variable as potentially related.
    p_value_threshold : float, optional (default=0.05)
        Maximum p-value to consider a correlation statistically significant.

    Returns
    ----------
    dict : A dictionary containing:
        - 'predictor': str - The predictor being analyzed
        - 'potential_confounders': list of str - Variables correlated with both X and Y
        - 'correlations_with_predictor': dict - Correlations of each candidate with predictor
        - 'correlations_with_outcome': dict - Correlations of each candidate with outcome
        - 'excluded_variables': list of str - Variables not meeting criteria
    """
    from scipy import stats
    
    y_values = np.ravel(y)
    other_features = [col for col in X.columns if col != predictor]
    
    potential_confounders = []
    correlations_with_predictor = {}
    correlations_with_outcome = {}
    excluded_variables = []
    
    predictor_values = X[predictor].values
    
    for feature in other_features:
        feature_values = X[feature].values
        
        # Pearson correlation measures the linear association between two
        # variables. For a candidate confounder Z, we need:
        #   r(Z, X) != 0   and   r(Z, Y) != 0
        # Both must exceed correlation_threshold with p < p_value_threshold
        # to be considered a candidate. The Pearson r is defined as:
        #   r = sum((z_i - z_bar)(x_i - x_bar)) /
        #       sqrt(sum((z_i - z_bar)^2) * sum((x_i - x_bar)^2))
        
        # Correlation with predictor
        corr_with_pred, p_pred = stats.pearsonr(feature_values, predictor_values)
        
        # Correlation with outcome
        corr_with_out, p_out = stats.pearsonr(feature_values, y_values)
        
        # Check if correlated with both predictor and outcome
        corr_pred_significant = (abs(corr_with_pred) >= correlation_threshold) and (p_pred < p_value_threshold)
        corr_out_significant = (abs(corr_with_out) >= correlation_threshold) and (p_out < p_value_threshold)
        
        if corr_pred_significant and corr_out_significant:
            potential_confounders.append(feature)
            correlations_with_predictor[feature] = {'r': corr_with_pred, 'p': p_pred}
            correlations_with_outcome[feature] = {'r': corr_with_out, 'p': p_out}
        else:
            excluded_variables.append(feature)
    
    return {
        'predictor': predictor,
        'potential_confounders': potential_confounders,
        'correlations_with_predictor': correlations_with_predictor,
        'correlations_with_outcome': correlations_with_outcome,
        'excluded_variables': excluded_variables
    }

def mediation_analysis(
    X: pd.DataFrame,
    y: pd.DataFrame,
    predictor: str,
    mediator: str,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    precision: int = 4,
    random_state: int = None
) -> Dict:
    """
    Perform mediation analysis to determine if a variable is a mediator.
    
    Uses the Baron & Kenny (1986) approach with bootstrap confidence intervals
    for the indirect effect (Preacher & Hayes method).
    
    A mediator lies on the causal pathway: X → M → Y
    A confounder is a common cause: M → X and M → Y
    
    If the indirect effect is significant, the variable is likely a mediator
    and should not be controlled for as a confounder.

    Parameters
    ----------
    X : pd.DataFrame
        The input data containing all features.
    y : pd.DataFrame
        The target variable.
    predictor : str
        The name of the predictor variable (X in X → M → Y).
    mediator : str
        The name of the potential mediator variable (M in X → M → Y).
    n_bootstrap : int, optional (default=1000)
        Number of bootstrap samples for confidence interval estimation.
    confidence_level : float, optional (default=0.95)
        Confidence level for the bootstrap confidence interval.
    precision : int, optional (default=4)
        Number of decimal places for reported values.
    random_state : int, optional (default=None)
        Seed for the random number generator to ensure reproducibility.

    Returns
    ----------
    dict : A dictionary containing:
        - 'predictor': str - The predictor variable
        - 'mediator': str - The potential mediator variable
        - 'total_effect': float - Total effect of X on Y (c path)
        - 'direct_effect': float - Direct effect of X on Y controlling for M (c' path)
        - 'indirect_effect': float - Indirect effect through M (a*b)
        - 'a_path': float - Effect of X on M
        - 'b_path': float - Effect of M on Y controlling for X
        - 'indirect_effect_ci': tuple - Bootstrap CI for indirect effect
        - 'proportion_mediated': float - Proportion of total effect mediated
        - 'is_mediator': bool - Whether variable is a significant mediator
        - 'interpretation': str - Plain language interpretation
    """
    y_values = np.ravel(y)
    x_values = X[predictor].values.reshape(-1, 1)
    m_values = X[mediator].values.reshape(-1, 1)
    
    # Baron & Kenny (1986) decomposes the total effect of X on Y into a
    # direct effect and an indirect effect that passes through M:
    #
    #   Total effect (c):     Y = c*X + e1
    #   a path:               M = a*X + e2
    #   b path + direct (c'): Y = c'*X + b*M + e3
    #
    # The indirect (mediated) effect is the product a*b. If a*b is
    # significantly different from zero, M mediates the X -> Y relationship.
    # The total effect decomposes as: c = c' + a*b
    
    # Step 1: Total effect (c path) -- regress Y on X alone
    model_total = LinearRegression().fit(x_values, y_values)
    c_total = model_total.coef_[0]
    
    # Step 2: a path -- regress M on X to quantify how X influences M
    model_a = LinearRegression().fit(x_values, m_values.ravel())
    a_path = model_a.coef_[0]
    
    # Step 3: b path and c' path -- regress Y on both X and M simultaneously
    # c' is the direct effect of X on Y after controlling for M
    # b is the effect of M on Y after controlling for X
    xm_combined = np.hstack([x_values, m_values])
    model_full = LinearRegression().fit(xm_combined, y_values)
    c_prime = model_full.coef_[0]  # Direct effect
    b_path = model_full.coef_[1]   # Effect of M on Y controlling for X
    
    # Indirect effect: a*b
    # This is the portion of X's effect on Y that is transmitted through M.
    # If this is large relative to c, then M explains much of the X -> Y path.
    indirect_effect = a_path * b_path
    
    # Bootstrap confidence interval for the indirect effect (a*b).
    # The sampling distribution of a*b is often non-normal, so percentile
    # bootstrap CIs (Preacher & Hayes, 2004) are preferred over the Sobel
    # test, which assumes normality of the product term.
    rng = np.random.default_rng(random_state)
    n_samples = len(y_values)
    bootstrap_indirect = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        x_boot = x_values[indices]
        m_boot = m_values[indices]
        y_boot = y_values[indices]
        
        # Recalculate a and b paths on the bootstrap sample
        model_a_boot = LinearRegression().fit(x_boot, m_boot.ravel())
        a_boot = model_a_boot.coef_[0]
        
        xm_boot = np.hstack([x_boot, m_boot])
        model_full_boot = LinearRegression().fit(xm_boot, y_boot)
        b_boot = model_full_boot.coef_[1]
        
        bootstrap_indirect[i] = a_boot * b_boot
    
    # Percentile confidence interval: if the interval excludes zero, the
    # indirect effect is statistically significant at the chosen level.
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_indirect, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_indirect, (1 - alpha / 2) * 100)
    
    # Determine if significant mediator (CI doesn't include 0)
    is_mediator = (ci_lower > 0) or (ci_upper < 0)
    
    # Proportion mediated: indirect / total = a*b / c
    # Only meaningful if the total effect is non-zero.
    if abs(c_total) > 1e-10:
        proportion_mediated = indirect_effect / c_total
    else:
        proportion_mediated = np.nan
    
    # Interpretation
    if is_mediator:
        if abs(c_prime) < abs(c_total) * 0.1:
            interpretation = f"'{mediator}' is a full mediator. It explains most of the effect of '{predictor}' on the outcome. Do not control for it as a confounder."
        else:
            interpretation = f"'{mediator}' is a partial mediator. It explains some of the effect of '{predictor}' on the outcome. Consider whether to control for it based on research question."
    else:
        interpretation = f"'{mediator}' is not a significant mediator. It may be a confounder if correlated with both '{predictor}' and outcome. Consider controlling for it."
    
    return {
        'predictor': predictor,
        'mediator': mediator,
        'total_effect': np.round(c_total, precision),
        'direct_effect': np.round(c_prime, precision),
        'indirect_effect': np.round(indirect_effect, precision),
        'a_path': np.round(a_path, precision),
        'b_path': np.round(b_path, precision),
        'indirect_effect_ci': (np.round(ci_lower, precision), np.round(ci_upper, precision)),
        'proportion_mediated': np.round(proportion_mediated, precision) if not np.isnan(proportion_mediated) else np.nan,
        'is_mediator': is_mediator,
        'interpretation': interpretation
    }

def identify_confounders(
    X: pd.DataFrame,
    y: pd.DataFrame,
    predictor: str,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = None
) -> Dict:
    """
    Identify confounders through a two-step process:
    1. Screen for variables correlated with both predictor and outcome
    2. Use mediation analysis to filter out mediators
    
    Variables that pass screening but are NOT mediators are likely confounders.

    Parameters
    ----------
    X : pd.DataFrame
        The input data containing all features.
    y : pd.DataFrame
        The target variable.
    predictor : str
        The name of the predictor variable of interest.
    correlation_threshold : float, optional (default=0.1)
        Minimum absolute correlation to consider a variable as potentially related.
    p_value_threshold : float, optional (default=0.05)
        Maximum p-value to consider a correlation statistically significant.
    n_bootstrap : int, optional (default=1000)
        Number of bootstrap samples for mediation analysis.
    confidence_level : float, optional (default=0.95)
        Confidence level for the bootstrap confidence interval.
    random_state : int, optional (default=None)
        Seed for the random number generator to ensure reproducibility.

    Returns
    ----------
    dict : A dictionary containing:
        - 'predictor': str - The predictor being analyzed
        - 'identified_confounders': list of str - Variables identified as confounders
        - 'identified_mediators': list of str - Variables identified as mediators
        - 'screening_results': dict - Results from correlation screening
        - 'mediation_results': dict - Results from mediation analysis for each candidate
        - 'recommendation': str - Plain language recommendation for Kennedy method
    """
    # Step 1: Screen for variables correlated with both X and Y.
    # Any variable Z with significant r(Z, X) and r(Z, Y) is a candidate
    # for either confounder or mediator status.
    screening = screen_potential_confounders(
        X, y, predictor, 
        correlation_threshold=correlation_threshold,
        p_value_threshold=p_value_threshold
    )
    
    candidates = screening['potential_confounders']
    
    # Step 2: For each candidate, run mediation analysis to test whether
    # it lies on the causal path X -> Z -> Y (mediator) or is a common
    # cause Z -> X, Z -> Y (confounder). Mediators have a significant
    # indirect effect (a*b != 0); confounders do not.
    identified_confounders = []
    identified_mediators = []
    mediation_results = {}
    
    for candidate in candidates:
        med_result = mediation_analysis(
            X, y, predictor, candidate,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state
        )
        mediation_results[candidate] = med_result
        
        if med_result['is_mediator']:
            identified_mediators.append(candidate)
        else:
            identified_confounders.append(candidate)
    
    # Generate recommendation
    if identified_confounders:
        confounder_str = ", ".join([f"'{c}'" for c in identified_confounders])
        recommendation = f"For Kennedy method with predictor '{predictor}', control for: {confounder_str}"
    else:
        recommendation = f"No confounders identified for predictor '{predictor}'. Consider using ter Braak method instead."
    
    return {
        'predictor': predictor,
        'identified_confounders': identified_confounders,
        'identified_mediators': identified_mediators,
        'screening_results': screening,
        'mediation_results': mediation_results,
        'recommendation': recommendation
    }

#%%
# ============================================================================
# Core Permutation Testing
#
# Permutation tests assess statistical significance without relying on
# distributional assumptions (e.g., normality of residuals). Instead of
# deriving p-values from a theoretical distribution, we build an empirical
# null distribution by repeatedly permuting the data under H0 and comparing
# the observed test statistic to this distribution.
#
# Three methods are implemented:
#   1. ter Braak (1992): For each feature j, fit a reduced model without
#      feature j, compute residuals, permute those residuals, and refit the
#      full model. Tests H0: beta_j = 0 given all other predictors.
#   2. Kennedy (1995) individual: Partial out confounders Z from each
#      predictor X via an exposure model (X ~ Z), permute the residuals of
#      X, and refit the outcome model. Tests H0: beta_j = 0 for each
#      non-confounder predictor.
#   3. Kennedy (1995) joint: Tests whether a group of predictors collectively
#      adds significant information beyond confounders, using deviance
#      reduction (logistic) or RSS reduction (linear) as the test statistic.
# ============================================================================

def permutation_test_regression(
    X: pd.DataFrame, 
    y: pd.DataFrame,  
    n_permutations: int = 5_000,
    precision: int = 3,
    p_value_threshold_one: float = 0.05, 
    p_value_threshold_two: float = 0.01,
    method: str = 'ter_braak',
    confounders: List[str] = None,
    random_state: int = None
) -> Dict:
    """
    Perform a permutation test for a regression model to assess the significance of 
    coefficients using the ter Braak (1992) or Kennedy (1995) methods.
    
    This function automatically detects binary vs continuous outcomes and uses
    logistic regression or linear regression accordingly.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to fit the model.
    y : array-like of shape (n_samples,)
        The target values (binary for logistic, continuous for linear regression).
    n_permutations : int, optional (default=5_000)
        The number of permutations to perform.
    precision : int, optional (default=3)
        The number of decimal places to round the p-values.
    p_value_threshold_one : float, optional (default=0.05)
        The threshold for the first level of significance.
    p_value_threshold_two : float, optional (default=0.01)
        The threshold for the second level of significance.
    method : str, optional (default='ter_braak')
        The permutation method to use.
    confounders : list of str, optional (default=None)
        List of confounder column names. Required for 'kennedy' and 'kennedy_joint' methods.
        For Kennedy methods, confounders are controlled for via an exposure model (X ~ Z).
    random_state : int, optional (default=None)
        Seed for the random number generator to ensure reproducibility.

    Methods
    ----------
    'ter_braak' : Permutation test using the method described by ter Braak (1992).
        For each feature, fits a reduced model (without that feature), then permutes
        the residuals from the reduced model. Tests H0: beta_j = 0 given all other predictors.
    'kennedy' : Individual coefficient permutation test using Kennedy (1995).
        Partials out confounders from predictors via an exposure model (X ~ Z),
        then permutes the residuals of X. Tests H0: beta_j = 0 for each non-confounder.
    'kennedy_joint' : Joint permutation test using Kennedy (1995).
        Tests whether a group of non-confounder predictors collectively adds significant
        information beyond the confounders. Uses deviance reduction (logistic) or RSS 
        reduction (linear) as the test statistic. Returns a single p-value for the group.

    Returns
    ----------
    dict : A dictionary containing:
        For 'ter_braak' and 'kennedy' methods:
            - 'model_coefs': list of float - Original coefficients
            - 'permuted_p_values': list of str - Empirical p-values per coefficient
            - 'classic_p_values': list of str - Asymptotic p-values per coefficient
            - 'p_value_threshold_one': float - First significance threshold
            - 'p_value_threshold_two': float - Second significance threshold
            - 'method': str - Method used
            - 'model_type': str - 'linear' or 'logistic'
        
        For 'kennedy_joint' method:
            - 'observed_improvement': float - Observed deviance/RSS reduction
            - 'p_value': float - Joint permutation p-value
            - 'p_value_str': str - Formatted p-value with significance stars
            - 'metric_type': str - 'Deviance Reduction' or 'RSS Reduction'
            - 'model_type': str - 'linear' or 'logistic'
            - 'features_tested': list of str - Non-confounder features tested
            - 'confounders': list of str - Confounders controlled for
            - 'p_value_threshold_one': float - First significance threshold
            - 'p_value_threshold_two': float - Second significance threshold
            - 'method': str - Method used

    References
    ----------
    ter Braak, Cajo J. F. "Permutation versus bootstrap significance tests in multiple regression and ANOVA."
    In Handbook of Statistics, Vol. 9. Elsevier, Amsterdam. (1992).
    
    Kennedy, Peter E. "Randomization Tests in Econometrics." 
    Journal of Business & Economic Statistics, 13:1, 85-94 (1995).
    
    Hardin, Johanna, Lauren Quesada, Julie Ye, and Nicholas J. Horton. "The Exchangeability Assumption for 
    Purmutation Tests of Multiple Regression Models: Implications for Statistics and Data Science Educators." 
    (2024) [Online]. Available: https://arxiv.org/pdf/2406.07756.
    
    Phipson, Belinda, and Gordon K. Smyth. "Permutation P-values Should Never Be Zero: Calculating Exact 
    P-values When Permutations Are Randomly Drawn." Statistical Applications in Genetics and Molecular 
    Biology, 9:1, Article 39 (2010).
    """
    rng = np.random.default_rng(random_state)
    permuted_coefs: List = []
    
    # Step 1: Detect model type from the outcome variable.
    # If Y has exactly two values {0, 1}, we use logistic regression;
    # otherwise we use ordinary least squares (linear regression).
    y_values = np.ravel(y)
    unique_y = np.unique(y_values)
    is_binary = (len(unique_y) == 2) and np.all(np.isin(unique_y, [0, 1]))
    
    if is_binary:
        model_class = lambda: LogisticRegression(penalty=None, solver='lbfgs', max_iter=5_000)
    else:
        model_class = lambda: LinearRegression()

    # Step 2: Fit the observed (unpermuted) model and extract coefficients.
    # These are the "real" estimates we will compare against the permutation
    # null distribution to assess statistical significance.
    model = model_class().fit(X, y_values)
    
    if is_binary:
        model_coefs = model.coef_.flatten().tolist()
    else:
        model_coefs = np.ravel(model.coef_).tolist()

    # Step 3: Compute diagnostic statistics via statsmodels.
    # For linear models: R^2, adjusted R^2, F-statistic, AIC, BIC.
    #   AIC = 2k - 2*ln(L),  BIC = k*ln(n) - 2*ln(L)
    # For logistic models: pseudo R^2 (McFadden), log-likelihood, AIC, BIC.
    #   Pseudo R^2 = 1 - ln(L_full) / ln(L_null)
    n_obs = len(y_values)
    n_features = X.shape[1]
    if is_binary:
        sm_model = sm.Logit(y_values, sm.add_constant(X)).fit(disp=0)
        diagnostics = {
            'n_observations': n_obs,
            'n_features': n_features,
            'pseudo_r_squared': np.round(sm_model.prsquared, 4),
            'log_likelihood': np.round(sm_model.llf, 4),
            'log_likelihood_null': np.round(sm_model.llnull, 4),
            'llr_p_value': sm_model.llr_pvalue,
            'aic': np.round(sm_model.aic, 4),
            'bic': np.round(sm_model.bic, 4),
        }
    else:
        sm_model = sm.OLS(y_values, sm.add_constant(X)).fit()
        diagnostics = {
            'n_observations': n_obs,
            'n_features': n_features,
            'r_squared': np.round(sm_model.rsquared, 4),
            'r_squared_adj': np.round(sm_model.rsquared_adj, 4),
            'f_statistic': np.round(sm_model.fvalue, 4),
            'f_p_value': sm_model.f_pvalue,
            'aic': np.round(sm_model.aic, 4),
            'bic': np.round(sm_model.bic, 4),
        }

    if method == 'ter_braak':
        # ter Braak Method: Permute residuals under the Reduced Model
        # ter Braak (1992) was designed for OLS. For binary outcomes, we use
        # a GLM-faithful adaptation: the reduced model is fit with logistic
        # regression to properly capture the heteroscedastic variance
        # structure Var(Y|X) = mu(1-mu), and Bernoulli sampling converts
        # the continuous permuted Y* back to binary for logistic refitting.
        
        # Create a matrix to hold results: rows=permutations, cols=features
        permuted_matrix = np.zeros((n_permutations, len(X.columns)))

        # Loop over each feature to perform its specific hypothesis test
        for i, feature_name in enumerate(X.columns):
            # 1. Define the Reduced Model (Drop the current feature)
            X_reduced = X.drop(columns=[feature_name])
            
            if is_binary:
                # 2a. Fit reduced logistic model and get residuals on
                # the response (probability) scale. Using logistic regression
                # here ensures the residuals reflect the true binary data
                # variance structure under H0, unlike a linear reduced model
                # which assumes constant variance.
                model_reduced = LogisticRegression(
                    penalty=None, solver='lbfgs', max_iter=5_000
                ).fit(X_reduced, y_values)
                preds_reduced = model_reduced.predict_proba(X_reduced)[:, 1]
                resids_reduced = y_values - preds_reduced
            else:
                # 2b. Fit reduced linear model (standard ter Braak)
                model_reduced = LinearRegression().fit(X_reduced, y_values)
                preds_reduced = model_reduced.predict(X_reduced).ravel()
                resids_reduced = y_values - preds_reduced
            
            # 3. Permutation Loop: generate B synthetic outcomes under H0
            # to build the null distribution of beta_j for this feature.
            for p in range(n_permutations):
                # Create a synthetic outcome by adding randomly reordered
                # residuals back to the reduced model's fitted values. This
                # simulates what Y would look like if this feature had no
                # effect on the outcome.
                y_perm = preds_reduced + rng.permutation(resids_reduced)
                
                if is_binary:
                    # The permuted Y* is continuous (probabilities + permuted
                    # residuals). To maintain scale consistency with the
                    # observed logistic coefficients (log-odds scale), we
                    # convert Y* back to binary via Bernoulli sampling:
                    #   Y_binary* ~ Bernoulli(clip(Y*, eps, 1-eps))
                    # This preserves magnitude information (unlike a hard
                    # threshold at 0.5) and naturally respects class imbalance.
                    y_perm_probs = np.clip(y_perm, 0.001, 0.999)
                    y_perm_binary = rng.binomial(1, y_perm_probs)
                    model_full = model_class().fit(X, y_perm_binary)
                    permuted_matrix[p, i] = model_full.coef_.flatten()[i]
                else:
                    model_full = model_class().fit(X, y_perm)
                    permuted_matrix[p, i] = np.ravel(model_full.coef_)[i]
        
        # Convert matrix back to list of lists format
        permuted_coefs = permuted_matrix.tolist()
        
    elif method == 'kennedy':
        # Kennedy (1995) Individual Coefficient Method
        #
        # The key idea: instead of permuting Y (as in ter Braak), we permute X.
        # Specifically, for each predictor X_j we want to test, we first remove
        # the linear influence of confounders Z by fitting an "exposure model":
        #   X_j = Z * gamma + e_j
        # The residuals e_j represent the part of X_j not explained by Z.
        # Under H0 (beta_j = 0), these residuals are exchangeable, so we
        # permute them and reconstruct X_j* = X_hat_j + permute(e_j).
        # We then refit the full outcome model Y ~ X* to get beta_j*.
        # Repeating B times builds a null distribution for beta_j.
        
        if confounders is None:
            confounders = []
        
        # Identify which features are being tested vs. held fixed as confounders
        features_to_test = [col for col in X.columns if col not in confounders]
        
        # Build confounder matrix Z from the columns the user identified as
        # confounders. Before permuting each predictor X_j, we regress X_j
        # on Z to isolate the part of X_j that is independent of the
        # confounders. Only that independent part gets shuffled, so the
        # confounders' influence on X_j is held constant during the test.
        if len(confounders) > 0:
            Z = X[confounders].values
        else:
            Z = np.zeros((len(X), 0))
        
        permuted_matrix = np.zeros((n_permutations, len(X.columns)))
        
        # Confounder coefficients are not tested (they are assumed fixed
        # controls, not hypotheses). Fill every permutation row with the
        # observed coefficient so they do not affect the p-value calculation.
        for i, col in enumerate(X.columns):
            if col in confounders:
                permuted_matrix[:, i] = model_coefs[i]
        
        for feature in features_to_test:
            feat_idx = X.columns.get_loc(feature)
            X_target = X[[feature]].values
            
            # Exposure model: regress X_j on Z to partial out confounder effects.
            # x_hat = predicted X_j from confounders alone.
            # x_resids = X_j - x_hat = variation in X_j NOT due to confounders.
            # Under H0, these residuals carry no information about Y.
            if Z.shape[1] > 0:
                exposure_model = LinearRegression().fit(Z, X_target)
                x_hat = exposure_model.predict(Z)
                x_resids = X_target - x_hat
            else:
                # No confounders: x_hat is just the column mean of X_j,
                # so the residuals are mean-centered values. Permuting them
                # is equivalent to shuffling X_j itself.
                x_hat = np.full_like(X_target, X_target.mean())
                x_resids = X_target - x_hat
            
            # For each permutation, randomly reorder the exposure residuals
            # and add them back to the confounder-predicted values to create
            # a synthetic version of X_j. This breaks any real association
            # between X_j and Y while preserving X_j's relationship with Z.
            # Refitting the full model on this synthetic X_j gives a
            # coefficient beta_j* drawn from the null distribution.
            for p in range(n_permutations):
                shuffled_resids = rng.permutation(x_resids.ravel())
                
                # X_j* = X_hat_j + shuffled residuals
                X_perm = X.copy()
                X_perm[feature] = x_hat.ravel() + shuffled_resids
                
                # Refit Y ~ X* (all features, with X_j replaced by X_j*)
                perm_model = model_class().fit(X_perm.values, y_values)
                permuted_matrix[p, feat_idx] = perm_model.coef_.flatten()[feat_idx]
        
        permuted_coefs = permuted_matrix.tolist()
        
    elif method == 'kennedy_joint':
        # Kennedy (1995) Joint Test
        #
        # Instead of testing each predictor individually, this tests whether
        # a GROUP of non-confounder predictors collectively adds significant
        # predictive information beyond the confounders alone.
        #
        # The test statistic measures the improvement in model fit when adding
        # the predictors of interest to a confounders-only (reduced) model:
        #   Linear:   RSS_reduced - RSS_full  (residual sum of squares reduction)
        #   Logistic: Deviance_reduced - Deviance_full  (deviance reduction)
        #
        # The null distribution is built by permuting the exposure-model
        # residuals of ALL non-confounder predictors simultaneously (row-wise
        # permutation preserves inter-predictor correlations), reconstructing
        # X*, refitting, and measuring the improvement under H0.
        
        if confounders is None:
            confounders = []
        
        features_to_test = [col for col in X.columns if col not in confounders]
        
        # Define the fit metric used to measure how much better the full
        # model fits compared to the reduced (confounders-only) model.
        # For linear regression, this is the residual sum of squares (RSS):
        #   RSS = sum((y_i - y_hat_i)^2), computed as MSE * n.
        # For logistic regression, this is the deviance (scaled log-likelihood):
        #   Deviance = 2 * sum(-y_i*log(p_i) - (1-y_i)*log(1-p_i)).
        # A larger drop from reduced to full metric means the predictors
        # explain more variance (linear) or reduce more prediction error
        # (logistic) than the confounders alone.
        if is_binary:
            def get_metric(y_true, y_pred_proba):
                return 2 * log_loss(y_true, y_pred_proba, normalize=False)
            metric_type = 'Deviance Reduction'
        else:
            def get_metric(y_true, y_pred):
                return mean_squared_error(y_true, y_pred) * len(y_true)
            metric_type = 'RSS Reduction'
        
        X_target = X[features_to_test].values
        
        if len(confounders) > 0:
            Z = X[confounders].values
        else:
            Z = np.zeros((len(X), 0))
        
        # Fit the reduced model (confounders only) to get the baseline metric.
        # This is the model under H0: the non-confounder predictors add nothing.
        if Z.shape[1] > 0:
            reduced_model = model_class().fit(Z, y_values)
            if is_binary:
                preds_reduced = reduced_model.predict_proba(Z)
            else:
                preds_reduced = reduced_model.predict(Z)
        else:
            # No confounders: the reduced model is just the intercept (mean of Y)
            if is_binary:
                mean_y = np.mean(y_values)
                preds_reduced = np.zeros((len(y_values), 2))
                preds_reduced[:, 0] = 1 - mean_y
                preds_reduced[:, 1] = mean_y
            else:
                preds_reduced = np.full_like(y_values, np.mean(y_values), dtype=float)
                
        base_metric = get_metric(y_values, preds_reduced)
        
        # Fit the full model (confounders + predictors of interest).
        # The observed improvement = base_metric - full_metric.
        if Z.shape[1] > 0:
            full_features = np.hstack([X_target, Z])
        else:
            full_features = X_target
            
        full_model = model_class().fit(full_features, y_values)
        
        if is_binary:
            preds_full = full_model.predict_proba(full_features)
        else:
            preds_full = full_model.predict(full_features)
            
        full_metric = get_metric(y_values, preds_full)
        obs_improvement = base_metric - full_metric
        
        # Exposure model: partial out Z from ALL non-confounder predictors jointly.
        # X_target = Z * Gamma + E, where E is the matrix of residuals.
        # Row-wise permutation of E preserves the correlation structure among
        # the non-confounder predictors, which is important for the joint test.
        if Z.shape[1] > 0:
            exposure_model = LinearRegression().fit(Z, X_target)
            x_hat = exposure_model.predict(Z)
            x_resids = X_target - x_hat
        else:
            x_hat = np.full_like(X_target, X_target.mean(axis=0))
            x_resids = X_target - x_hat
            
        # Permutation loop: for each of B iterations, randomly reorder the
        # rows (not columns) of the residual matrix E so that each
        # observation's residual vector is kept intact — this preserves the
        # observed correlation structure among the non-confounder predictors.
        # Reconstructing X* = X_hat + permuted(E) and refitting measures how
        # much fit improvement arises by chance when the real X-Y associations
        # are broken but inter-predictor relationships are maintained.
        perm_improvements = np.zeros(n_permutations)
        
        for i in range(n_permutations):
            shuffled_idx = rng.permutation(len(y_values))
            shuffled_resids = x_resids[shuffled_idx]
            
            # Reconstruct X*: add the shuffled residuals back to the
            # confounder-predicted values, creating a synthetic predictor
            # matrix that has the same Z-relationship but a randomized X-Y link.
            x_star = x_hat + shuffled_resids
            
            if Z.shape[1] > 0:
                perm_features = np.hstack([x_star, Z])
            else:
                perm_features = x_star
                
            perm_model = model_class().fit(perm_features, y_values)
            
            if is_binary:
                perm_preds = perm_model.predict_proba(perm_features)
            else:
                perm_preds = perm_model.predict(perm_features)
                
            perm_metric_val = get_metric(y_values, perm_preds)
            perm_improvements[i] = base_metric - perm_metric_val
        
        # Compute the empirical p-value using the Phipson & Smyth (2010)
        # correction. Count how many permuted fit improvements equal or
        # exceed the observed improvement, then add 1 to both the count
        # and the total number of permutations. This ensures the p-value
        # is never exactly zero and treats the observed statistic as one
        # member of the (B + 1)-sized reference set.
        p_value = (np.sum(perm_improvements >= obs_improvement) + 1) / (n_permutations + 1)
        
        # Format p-value string
        if p_value < p_value_threshold_two:
            p_value_str = f"{np.round(p_value, precision)} (**)"
        elif p_value < p_value_threshold_one:
            p_value_str = f"{np.round(p_value, precision)} (*)"
        else:
            p_value_str = f"{np.round(p_value, precision)} (ns)"
        
        return {
            'observed_improvement': obs_improvement,
            'p_value': p_value,
            'p_value_str': p_value_str,
            'metric_type': metric_type,
            'model_type': 'logistic' if is_binary else 'linear',
            'features_tested': features_to_test,
            'confounders': confounders,
            'p_value_threshold_one': p_value_threshold_one,
            'p_value_threshold_two': p_value_threshold_two,
            'method': method,
            'diagnostics': diagnostics
        }
        
    else:
        raise ValueError("Invalid method. Please select 'ter_braak', 'kennedy', or 'kennedy_joint'.")

    permuted_p_values, classic_p_values = calculate_p_values(X, y, permuted_coefs, model_coefs, precision, p_value_threshold_one, p_value_threshold_two)
    
    # For Kennedy method with confounders, mark confounder p-values as N/A
    # since they are controls, not hypotheses being tested.
    if method == 'kennedy' and confounders:
        for i, col in enumerate(X.columns):
            if col in confounders:
                permuted_p_values[i] = 'N/A (confounder)'
                classic_p_values[i] = 'N/A (confounder)'
    
    return {
        'model_coefs': model_coefs,
        'permuted_p_values': permuted_p_values,
        'classic_p_values': classic_p_values,
        'p_value_threshold_one': p_value_threshold_one,
        'p_value_threshold_two': p_value_threshold_two,
        'method': method,
        'model_type': 'logistic' if is_binary else 'linear',
        'diagnostics': diagnostics
    }

#%%
# ============================================================================
# Test Case 1: Linear Regression (Continuous Outcome)
# Real Estate Valuation dataset (UCI ML Repository ID=477)
# ============================================================================

# Load the dataset
real_estate_valuation = fetch_ucirepo(id=477)

#%%
# Features and target variable
X = real_estate_valuation.data.features
y = real_estate_valuation.data.targets

#%%
# Perform permutation test by the ter Braak (1992) method
results_ter_braak = permutation_test_regression(X, y, method='ter_braak')

#%%
# Print the results obtained by the ter Braak (1992) method
print_results_table(
    results_ter_braak,
    feature_names=X.columns.tolist(),
    target_name=y.columns[0],
    title="ter Braak (1992) Permutation Test (Linear)"
)

#%%
# Perform permutation test by the Kennedy (1995) individual coefficient method
# Note: When confounders=[], all features are tested with no confounders controlled for.
# Unlike ter Braak, this does NOT condition on other predictors. ter Braak permutes
# residuals of Y|X_{-j}, while Kennedy permutes X_j centered at its mean. Results
# may differ when features are correlated.
results_kennedy = permutation_test_regression(X, y, method='kennedy', confounders=[])

#%%
# Print the results obtained by the Kennedy (1995) individual method
print_results_table(
    results_kennedy,
    feature_names=X.columns.tolist(),
    target_name=y.columns[0],
    title="Kennedy (1995) Individual Coefficient Permutation Test (Linear)"
)

#%%
# Perform permutation test by the Kennedy (1995) joint method
# Tests whether all features collectively add significant information
results_kennedy_joint = permutation_test_regression(X, y, method='kennedy_joint', confounders=[])

#%%
# Print the results obtained by the Kennedy (1995) joint method
print_joint_results_table(
    results_kennedy_joint,
    target_name=y.columns[0],
    title="Kennedy (1995) Joint Permutation Test (Linear)"
)

#%%
# Full confounder identification workflow for all predictors
# Loops through each predictor and identifies potential confounders
print("Confounder Identification for All Predictors\n")

all_confounder_results = {}
for predictor in X.columns:
    confounder_results = identify_confounders(X, y, predictor=predictor)
    all_confounder_results[predictor] = confounder_results

#%%
# Print confounder identification results for all predictors
for predictor, results in all_confounder_results.items():
    print(f"Predictor: '{predictor}'")
    print(f"  Identified Confounders: {results['identified_confounders']}")
    print(f"  Identified Mediators: {results['identified_mediators']}")
    if results['identified_confounders'] or results['identified_mediators']:
        print(f"  Recommendation: {results['recommendation']}")
    print()

#%%
# Summary: Show which predictors have confounders that should be controlled
predictors_with_confounders = {
    pred: res['identified_confounders'] 
    for pred, res in all_confounder_results.items() 
    if res['identified_confounders']
}

print("Summary: Predictors with Identified Confounders\n")

if predictors_with_confounders:
    for pred, confounders in predictors_with_confounders.items():
        print(f"  {pred}: control for {confounders}")
else:
    print("  No confounders identified for any predictor.")
    print("  This suggests the predictors are relatively independent.")
print()

#%%
# Example: Use Kennedy method with identified confounders for a specific predictor
# Pick the first predictor that has identified confounders, if any.
if predictors_with_confounders:
    example_predictor = list(predictors_with_confounders.keys())[0]
    example_confounders = predictors_with_confounders[example_predictor]
    
    results_kennedy_with_confounders = permutation_test_regression(
        X, y, 
        method='kennedy', 
        confounders=example_confounders
    )
    
    print_results_table(
        results_kennedy_with_confounders,
        feature_names=X.columns.tolist(),
        target_name=y.columns[0],
        title=f"Kennedy (1995) Method for '{example_predictor}' (controlling for {example_confounders}) (Linear)"
    )

#%%
# ============================================================================
# Test Case 2: Logistic Regression (Binary Outcome)
# Breast Cancer Wisconsin (Diagnostic) dataset (UCI ML Repository ID=17)
# ============================================================================

# Load the dataset
breast_cancer = fetch_ucirepo(id=17)

#%%
# Features and target variable
X_bc = breast_cancer.data.features
y_bc = breast_cancer.data.targets

# Convert target to binary: malignant (M) -> 1, benign (B) -> 0
y_bc = (y_bc == 'M').astype(int)

# Use a subset of features to keep computation tractable
selected_features = ['radius1', 'texture1', 'perimeter1',
                     'smoothness1', 'compactness1']
X_bc = X_bc[selected_features]

#%%
# Perform permutation test by the ter Braak (1992) method (logistic)
results_ter_braak_bc = permutation_test_regression(X_bc, y_bc, method='ter_braak')

#%%
# Print the results obtained by the ter Braak (1992) method (logistic)
print_results_table(
    results_ter_braak_bc,
    feature_names=X_bc.columns.tolist(),
    target_name=y_bc.columns[0],
    title="ter Braak (1992) Permutation Test (Logistic)"
)

#%%
# Perform permutation test by the Kennedy (1995) individual method (logistic)
# Note: When confounders=[], all features are tested with no confounders controlled for.
# Unlike ter Braak, this does NOT condition on other predictors. ter Braak permutes
# residuals of Y|X_{-j}, while Kennedy permutes X_j centered at its mean. Results
# may differ when features are correlated.
results_kennedy_bc = permutation_test_regression(X_bc, y_bc, method='kennedy', confounders=[])

#%%
# Print the results obtained by the Kennedy (1995) individual method (logistic)
print_results_table(
    results_kennedy_bc,
    feature_names=X_bc.columns.tolist(),
    target_name=y_bc.columns[0],
    title="Kennedy (1995) Individual Coefficient Permutation Test (Logistic)"
)

#%%
# Perform permutation test by the Kennedy (1995) joint method (logistic)
# Tests whether all features collectively add significant information
results_kennedy_joint_bc = permutation_test_regression(X_bc, y_bc, method='kennedy_joint', confounders=[])

#%%
# Print the results obtained by the Kennedy (1995) joint method (logistic)
print_joint_results_table(
    results_kennedy_joint_bc,
    target_name=y_bc.columns[0],
    title="Kennedy (1995) Joint Permutation Test (Logistic)"
)

#%%
# Full confounder identification workflow for all predictors (logistic)
# Loops through each predictor and identifies potential confounders
print("Confounder Identification for All Predictors (Logistic)\n")

all_confounder_results_bc = {}
for predictor in X_bc.columns:
    confounder_results_bc = identify_confounders(X_bc, y_bc, predictor=predictor)
    all_confounder_results_bc[predictor] = confounder_results_bc

#%%
# Print confounder identification results for all predictors (logistic)
for predictor, results in all_confounder_results_bc.items():
    print(f"Predictor: '{predictor}'")
    print(f"  Identified Confounders: {results['identified_confounders']}")
    print(f"  Identified Mediators: {results['identified_mediators']}")
    if results['identified_confounders'] or results['identified_mediators']:
        print(f"  Recommendation: {results['recommendation']}")
    print()

#%%
# Summary: Show which predictors have confounders that should be controlled (logistic)
predictors_with_confounders_bc = {
    pred: res['identified_confounders']
    for pred, res in all_confounder_results_bc.items()
    if res['identified_confounders']
}

print("Summary: Predictors with Identified Confounders (Logistic)\n")

if predictors_with_confounders_bc:
    for pred, confounders in predictors_with_confounders_bc.items():
        print(f"  {pred}: control for {confounders}")
else:
    print("  No confounders identified for any predictor.")
    print("  This suggests the predictors are relatively independent.")
print()

#%%
# Example: Use Kennedy method with identified confounders (logistic)
# Pick the first predictor that has identified confounders, if any
if predictors_with_confounders_bc:
    example_predictor_bc = list(predictors_with_confounders_bc.keys())[0]
    example_confounders_bc = predictors_with_confounders_bc[example_predictor_bc]

    results_kennedy_with_confounders_bc = permutation_test_regression(
        X_bc, y_bc,
        method='kennedy',
        confounders=example_confounders_bc
    )

    print_results_table(
        results_kennedy_with_confounders_bc,
        feature_names=X_bc.columns.tolist(),
        target_name=y_bc.columns[0],
        title=f"Kennedy (1995) Method for '{example_predictor_bc}' (controlling for {example_confounders_bc}) (Logistic)"
    )

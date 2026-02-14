#%%
# Import libraries
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from ucimlrepo import fetch_ucirepo 

#%%
# Load the dataset
real_estate_valuation = fetch_ucirepo(id=477) 

#%%
# Features and target variable
X = real_estate_valuation.data.features 
y = real_estate_valuation.data.targets 

#%%
# Define the permutation regression functions
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
        # Use statsmodels Logit for classical p-values
        model = sm.Logit(y_values, sm.add_constant(X)).fit(disp=0)
    else:
        # Use statsmodels OLS for classical p-values
        model = sm.OLS(y, sm.add_constant(X)).fit()

    for i in range(len(model_coefs)):
        p_value = (np.abs(np.array(permuted_coefs)[:, i]) >= np.abs(np.array(model_coefs)[i])).mean()        
        if p_value_threshold_two <= p_value < p_value_threshold_one:
            p_value_str = str(np.round(p_value, precision)) + ' (*)'
        elif p_value < p_value_threshold_two:
            p_value_str = str(np.round(p_value, precision)) + ' (**)'
        else:
            p_value_str = str(np.round(p_value, precision)) + ' (ns)'
        
        permuted_p_values.append(p_value_str)
    
    for p_value in model.pvalues[1:]:
        if p_value_threshold_two <= p_value < p_value_threshold_one:
            p_value_str = str(np.round(p_value, precision)) + ' (*)'
        elif p_value < p_value_threshold_two:
            p_value_str = str(np.round(p_value, precision)) + ' (**)'
        else:
            p_value_str = str(np.round(p_value, precision)) + ' (ns)'

        classic_p_values.append(p_value_str)

    return permuted_p_values, classic_p_values

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
    if target_name:
        print(f"Dep. Variable: {target_name:<20} Model Type: {results['model_type']}")
    else:
        print(f"Model Type: {results['model_type']}")
    print(f"Method: {results['method']}")
    print("-" * 80)
    
    # Column headers
    print(f"{'Feature':<25} {'Coef':>12} {'P>|z| (Emp)':>18} {'P>|z| (Asy)':>18}")
    print("-" * 80)
    
    # Data rows
    coefs = results['model_coefs']
    emp_p = results['permuted_p_values']
    asy_p = results['classic_p_values']
    
    for i, feat in enumerate(feature_names):
        coef_str = f"{coefs[i]:>12.4f}"
        print(f"{feat:<25} {coef_str} {emp_p[i]:>18} {asy_p[i]:>18}")
    
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
    if target_name:
        print(f"Dep. Variable: {target_name:<20} Model Type: {results['model_type']}")
    else:
        print(f"Model Type: {results['model_type']}")
    print(f"Method: {results['method']}")
    print(f"Metric: {results['metric_type']}")
    print("-" * 80)
    
    # Features tested
    print(f"Features Tested: {', '.join(results['features_tested'])}")
    if results['confounders']:
        print(f"Confounders: {', '.join(results['confounders'])}")
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
    precision: int = 4
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
    
    # Step 1: Total effect (c path) - X → Y
    model_total = LinearRegression().fit(x_values, y_values)
    c_total = model_total.coef_[0]
    
    # Step 2: a path - X → M
    model_a = LinearRegression().fit(x_values, m_values.ravel())
    a_path = model_a.coef_[0]
    
    # Step 3: b path and c' path - X + M → Y
    xm_combined = np.hstack([x_values, m_values])
    model_full = LinearRegression().fit(xm_combined, y_values)
    c_prime = model_full.coef_[0]  # Direct effect
    b_path = model_full.coef_[1]   # Effect of M on Y controlling for X
    
    # Indirect effect
    indirect_effect = a_path * b_path
    
    # Bootstrap confidence interval for indirect effect
    n_samples = len(y_values)
    bootstrap_indirect = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        x_boot = x_values[indices]
        m_boot = m_values[indices]
        y_boot = y_values[indices]
        
        # Recalculate a and b paths
        model_a_boot = LinearRegression().fit(x_boot, m_boot.ravel())
        a_boot = model_a_boot.coef_[0]
        
        xm_boot = np.hstack([x_boot, m_boot])
        model_full_boot = LinearRegression().fit(xm_boot, y_boot)
        b_boot = model_full_boot.coef_[1]
        
        bootstrap_indirect[i] = a_boot * b_boot
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_indirect, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_indirect, (1 - alpha / 2) * 100)
    
    # Determine if significant mediator (CI doesn't include 0)
    is_mediator = (ci_lower > 0) or (ci_upper < 0)
    
    # Proportion mediated (only meaningful if total effect is non-zero)
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
    confidence_level: float = 0.95
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
    # Step 1: Screen for potential confounders
    screening = screen_potential_confounders(
        X, y, predictor, 
        correlation_threshold=correlation_threshold,
        p_value_threshold=p_value_threshold
    )
    
    candidates = screening['potential_confounders']
    
    # Step 2: Run mediation analysis on each candidate
    identified_confounders = []
    identified_mediators = []
    mediation_results = {}
    
    for candidate in candidates:
        med_result = mediation_analysis(
            X, y, predictor, candidate,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
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

def permutation_test_regression(
    X: pd.DataFrame, 
    y: pd.DataFrame,  
    n_permutations: int = 5_000,
    precision: int = 3,
    p_value_threshold_one: float = 0.05, 
    p_value_threshold_two: float = 0.01,
    method: str = 'ter_braak',
    confounders: List[str] = None
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
    """
    permuted_coefs: List = []
    
    # Detect model type based on Y
    y_values = np.ravel(y)
    unique_y = np.unique(y_values)
    is_binary = (len(unique_y) == 2) and np.all(np.isin(unique_y, [0, 1]))
    
    if is_binary:
        model_class = lambda: LogisticRegression(penalty=None, solver='lbfgs', max_iter=5_000)
    else:
        model_class = lambda: LinearRegression()

    model = model_class().fit(X, y_values)
    
    if is_binary:
        model_coefs = model.coef_.flatten().tolist()
    else:
        model_coefs = np.ravel(model.coef_).tolist()

    if method == 'ter_braak':
        # ter Braak Method: Permute residuals under the Reduced Model
        # For logistic regression, we work in linear space using predicted probabilities
        # transformed to continuous residuals
        
        # Create a matrix to hold results: rows=permutations, cols=features
        permuted_matrix = np.zeros((n_permutations, len(X.columns)))

        # Loop over each feature to perform its specific hypothesis test
        for i, feature_name in enumerate(X.columns):
            # 1. Define the Reduced Model (Drop the current feature)
            X_reduced = X.drop(columns=[feature_name])
            
            # 2. Fit Reduced Model and get residuals under H0
            # For both binary and continuous Y, we use LinearRegression to get 
            # continuous residuals in linear space (this is the standard approach
            # for permutation tests - work in linear space for residuals)
            model_reduced = LinearRegression().fit(X_reduced, y_values)
            preds_reduced = model_reduced.predict(X_reduced).ravel()
            resids_reduced = y_values - preds_reduced
            
            # 3. Permutation Loop for this specific feature
            for p in range(n_permutations):
                # Construct new Y under H0: Reduced Preds + Shuffled Reduced Resids
                y_perm = preds_reduced + np.random.permutation(resids_reduced)
                
                # For binary outcomes, clip predictions to valid probability range
                # and convert back to binary for fitting
                if is_binary:
                    y_perm_probs = np.clip(y_perm, 0.001, 0.999)
                    y_perm_binary = (y_perm_probs > 0.5).astype(int)
                    model_full = model_class().fit(X, y_perm_binary)
                    permuted_matrix[p, i] = model_full.coef_.flatten()[i]
                else:
                    model_full = model_class().fit(X, y_perm)
                    permuted_matrix[p, i] = np.ravel(model_full.coef_)[i]
        
        # Convert matrix back to list of lists format
        permuted_coefs = permuted_matrix.tolist()
        
    elif method == 'kennedy':
        # Kennedy Method: Partial out confounders from X via exposure model (X ~ Z)
        # then permute the residuals of X
        
        if confounders is None:
            confounders = []
        
        # Get features to test (all non-confounders)
        features_to_test = [col for col in X.columns if col not in confounders]
        
        # Prepare confounder matrix
        if len(confounders) > 0:
            Z = X[confounders].values
        else:
            Z = np.zeros((len(X), 0))
        
        # Create a matrix to hold results: rows=permutations, cols=features
        permuted_matrix = np.zeros((n_permutations, len(X.columns)))
        
        # For confounder columns, we don't permute - just store observed coefs
        for i, col in enumerate(X.columns):
            if col in confounders:
                permuted_matrix[:, i] = model_coefs[i]
        
        # Loop through each non-confounder feature to test
        for feature in features_to_test:
            feat_idx = X.columns.get_loc(feature)
            X_target = X[[feature]].values
            
            # Partial out confounders from X (exposure model)
            if Z.shape[1] > 0:
                exposure_model = LinearRegression().fit(Z, X_target)
                x_hat = exposure_model.predict(Z)
                x_resids = X_target - x_hat
            else:
                x_hat = np.full_like(X_target, X_target.mean())
                x_resids = X_target - x_hat
            
            # Permutation loop for this feature
            for p in range(n_permutations):
                # Shuffle residuals of X
                shuffled_resids = np.random.permutation(x_resids.ravel())
                
                # Reconstruct X* = X_hat + shuffled_resids
                X_perm = X.copy()
                X_perm[feature] = x_hat.ravel() + shuffled_resids
                
                # Fit full model on original Y with permuted X
                perm_model = model_class().fit(X_perm.values, y_values)
                
                # Extract coefficient for the feature
                permuted_matrix[p, feat_idx] = perm_model.coef_.flatten()[feat_idx]
        
        # Convert matrix back to list of lists format
        permuted_coefs = permuted_matrix.tolist()
        
    elif method == 'kennedy_joint':
        # Kennedy Joint Method: Tests whether a GROUP of non-confounder predictors 
        # collectively adds significant information beyond confounders.
        # Uses deviance reduction (logistic) or RSS reduction (linear) as the metric.
        
        if confounders is None:
            confounders = []
        
        # Get features to test (all non-confounders)
        features_to_test = [col for col in X.columns if col not in confounders]
        
        # Define metric functions based on model type
        if is_binary:
            def get_metric(y_true, y_pred_proba):
                return 2 * log_loss(y_true, y_pred_proba, normalize=False)
            metric_type = 'Deviance Reduction'
        else:
            def get_metric(y_true, y_pred):
                return mean_squared_error(y_true, y_pred) * len(y_true)
            metric_type = 'RSS Reduction'
        
        # Prepare matrices
        X_target = X[features_to_test].values
        
        if len(confounders) > 0:
            Z = X[confounders].values
        else:
            Z = np.zeros((len(X), 0))
        
        # Fit reduced model (only confounders)
        if Z.shape[1] > 0:
            reduced_model = model_class().fit(Z, y_values)
            if is_binary:
                preds_reduced = reduced_model.predict_proba(Z)
            else:
                preds_reduced = reduced_model.predict(Z)
        else:
            # Null model (intercept only)
            if is_binary:
                mean_y = np.mean(y_values)
                preds_reduced = np.zeros((len(y_values), 2))
                preds_reduced[:, 0] = 1 - mean_y
                preds_reduced[:, 1] = mean_y
            else:
                preds_reduced = np.full_like(y_values, np.mean(y_values), dtype=float)
                
        base_metric = get_metric(y_values, preds_reduced)
        
        # Fit observed full model (X_target + Z)
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
        
        # Observed improvement (positive = improvement)
        obs_improvement = base_metric - full_metric
        
        # Kennedy preprocessing: partial out Z from X_target
        if Z.shape[1] > 0:
            exposure_model = LinearRegression().fit(Z, X_target)
            x_hat = exposure_model.predict(Z)
            x_resids = X_target - x_hat
        else:
            x_hat = np.full_like(X_target, X_target.mean(axis=0))
            x_resids = X_target - x_hat
            
        # Permutation loop
        perm_improvements = np.zeros(n_permutations)
        
        for i in range(n_permutations):
            # Shuffle residuals (keeps internal correlation of X_target intact)
            shuffled_idx = np.random.permutation(len(y_values))
            shuffled_resids = x_resids[shuffled_idx]
            
            # Reconstruct X*
            x_star = x_hat + shuffled_resids
            
            # Refit full model
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
        
        # P-Value: Probability that random noise improved the model as much as real X
        p_value = np.mean(perm_improvements >= obs_improvement)
        
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
            'method': method
        }
        
    else:
        raise ValueError("Invalid method. Please select 'ter_braak', 'kennedy', or 'kennedy_joint'.")

    permuted_p_values, classic_p_values = calculate_p_values(X, y, permuted_coefs, model_coefs, precision, p_value_threshold_one, p_value_threshold_two)
    
    # For Kennedy method with confounders, mark confounder p-values as N/A
    # since they are controls, not hypotheses being tested
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
        'model_type': 'logistic' if is_binary else 'linear'
    }

#%%
# Perform permutation test by the ter Braak (1992) method
results_ter_braak = permutation_test_regression(X, y, method='ter_braak')

#%%
# Print the results obtained by the ter Braak (1992) method
print_results_table(
    results_ter_braak,
    feature_names=X.columns.tolist(),
    target_name=y.columns[0],
    title="ter Braak (1992) Permutation Test"
)

#%%
# Perform permutation test by the Kennedy (1995) individual coefficient method
# Note: When confounders=[], all features are tested with no confounders controlled for.
# This is equivalent to ter Braak but uses the Kennedy residual permutation approach.
results_kennedy = permutation_test_regression(X, y, method='kennedy', confounders=[])

#%%
# Print the results obtained by the Kennedy (1995) individual method
print_results_table(
    results_kennedy,
    feature_names=X.columns.tolist(),
    target_name=y.columns[0],
    title="Kennedy (1995) Individual Coefficient Permutation Test"
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
    title="Kennedy (1995) Joint Permutation Test"
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
# Pick the first predictor that has identified confounders, if any
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
        title=f"Kennedy (1995) Method for '{example_predictor}' (controlling for {example_confounders})"
    )

# %%

#%%
# Import libraries
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
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
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to fit the model.
    y : array-like of shape (n_samples,)
        The target values.
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

def permutation_test_regression(
    X: pd.DataFrame, 
    y: pd.DataFrame,  
    n_permutations: int = 100_000,
    precision: int = 3,
    p_value_threshold_one: float = 0.05, 
    p_value_threshold_two: float = 0.01,
    method: str = 'manly'
) -> Tuple[List[float], List[str], List[str], float, float]:
    """
    Perform a permutation test for a multiple linear regression model to assess the significance of model 
    coefficients using the ter Braak (1992) or Manly (1997) methods.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to fit the model.
    y : array-like of shape (n_samples,)
        The target values.
    n_permutations : int, optional (default=100_000)
        The number of permutations to perform.
    precision : int, optional (default=3)
        The number of decimal places to round the p-values.
    p_value_threshold_one : float, optional (default=0.05)
        The threshold for the first level of significance.
    p_value_threshold_two : float, optional (default=0.01)
        The threshold for the second level of significance.
    method : str, optional (default='manly')

    Methods
    ----------
    'terbraak' : Permutation test using the method described by ter Braak (1992).
    'manly' : Permutation test using the method described by Manly (1997).

    Returns
    ----------
    model_coefs : list of float
        The original coefficients of the fitted model.
    permuted_p_values : list of str
        The empirical p-values for each coefficient, formatted with significance stars if below the thresholds.
    classic_p_values : list of str
        The asymptotic p-values for each coefficient, formatted with significance stars if below the thresholds.

    References
    ----------
    ter Braak, Cajo J. F. "Permutation versus bootstrap significance tests in multiple regression and ANOVA."
    In Handbook of Statistics, Vol. 9. Elsevier, Amsterdam. (1992).

    Manly, Bryan F. J. Randomization, Bootstrap, and Monte Carlo Methods in Biology, 2nd ed. Texts in Statistical 
    Science Series. Chapman & Hall, London. (1997).
    
    Hardin, Johanna, Lauren Quesada, Julie Ye, and Nicholas J. Horton. "The Exchangeability Assumption for 
    Purmutation Tests of Multiple Regression Models: Implications for Statistics and Data Science Educators." 
    (2024) [Online]. Available: https://arxiv.org/pdf/2406.07756.
    """
    permuted_coefs: List = []

    model = LinearRegression().fit(X, y)
    model_coefs = np.ravel(model.coef_).tolist()
    model_preds = model.predict(X)
    model_resids = y - model_preds

    if method == 'manly':
        for _ in range(n_permutations):
            model.fit(X, np.random.permutation(y))
            permuted_coefs.append(np.ravel(model.coef_).tolist())
    elif method == 'terbraak':
        for _ in range(n_permutations):
            model.fit(X, np.random.permutation(model_preds) + np.random.permutation(model_resids))
            permuted_coefs.append(np.ravel(model.coef_).tolist())
    else:
        raise ValueError("Invalid method. Please select 'manly' or 'terbraak'.")

    permuted_p_values, classic_p_values = calculate_p_values(X, y, permuted_coefs, model_coefs, precision, p_value_threshold_one, p_value_threshold_two)
    
    return model_coefs, permuted_p_values, classic_p_values, p_value_threshold_one, p_value_threshold_two

#%%
# Perform permutation test by the Manly (1997) method
(
    coefs, 
    permuted_p_values, 
    classic_p_values, 
    p_value_threshold_one, 
    p_value_threshold_two
) = permutation_test_regression(X, y)

#%%
# Print the results obtained by the Manly (1997) method
print("Regression Model Coefficients and p-Values obtained by the Manly (1997) method\n")
print(f"Target: {y.columns.tolist()}")
print(f"Features: {X.columns.tolist()}\n")
print(f"Coefficients: {coefs}\n")
print(f"Empirical p-Values: {permuted_p_values}")
print(f"Asymptotic p-Values: {classic_p_values}")
print(f"\n(*) p-value < {p_value_threshold_one}")
print(f"(**) p-value < {p_value_threshold_two}")
print(f"(ns) p-value >= {p_value_threshold_one}\n")

#%%
# Perform permutation test by the ter Braak (1992) method
(
    coefs, 
    permuted_p_values, 
    classic_p_values, 
    p_value_threshold_one, 
    p_value_threshold_two
) = permutation_test_regression(X, y, method='terbraak')

#%%
# Print the results obtained by the ter Braak (1992) method
print("Regression Model Coefficients and p-Values obtained by the ter Braak (1992) method\n")
print(f"Target: {y.columns.tolist()}")
print(f"Features: {X.columns.tolist()}\n")
print(f"Coefficients: {coefs}\n")
print(f"Empirical p-Values: {permuted_p_values}")
print(f"Asymptotic p-Values: {classic_p_values}")
print(f"\n(*) p-value < {p_value_threshold_one}")
print(f"(**) p-value < {p_value_threshold_two}")
print(f"(ns) p-value >= {p_value_threshold_one}\n")

#%%
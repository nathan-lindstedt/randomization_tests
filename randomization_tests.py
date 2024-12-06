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
def permutation_test(
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    model: LinearRegression, 
    n_permutations: int = 100_000,
    precision: int = 3,
    p_value_threshold_one: float = 0.05, 
    p_value_threshold_two: float = 0.01
) -> Tuple[List[float], List[str], List[str]]:
    """
    Perform a permutation test for a regression model to assess the significance of model coefficients.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to fit the model.
    y : array-like of shape (n_samples,)
        The target values.
    model : object
        The regression model that implements the fit method and has a coef_ attribute.
    n_permutations : int, optional (default=100_000)
        The number of permutations to perform.
    precision : int, optional (default=3)
        The number of decimal places to round the p-values.
    p_value_threshold_one : float, optional (default=0.05)
        The threshold for the first level of significance.
    p_value_threshold_two : float, optional (default=0.01)
        The threshold for the second level of significance.

    Returns
    ----------
    model_coefs : list of float
        The original coefficients of the fitted model.
    permuted_p_values : list of str
        The empirical p-values for each coefficient, formatted with significance stars if below the thresholds.
    classic_p_values : list of str
        The asymptotic p-values for each coefficient, formatted with significance stars if below the thresholds.
    """
    permuted_coefs: List = []
    permuted_p_values: List = []
    classic_p_values: List = []

    model.fit(X, y)
    model_coefs = model.coef_.flatten().tolist()

    for _ in range(n_permutations):
        model.fit(X, np.random.permutation(y))
        permuted_coefs.append(model.coef_.flatten().tolist())

    for i in range(len(model_coefs)):
        p_value = (np.abs(np.array(permuted_coefs)[:, i]) >= np.abs(np.array(model_coefs)[i])).mean()
        if p_value_threshold_two <= p_value < p_value_threshold_one:
            p_value_str = str(np.round(p_value, precision)) + ' (*)'
        elif p_value < p_value_threshold_two:
            p_value_str = str(np.round(p_value, precision)) + ' (**)'
        else:
            p_value_str = str(np.round(p_value, precision)) + ' (ns)'
        permuted_p_values.append(p_value_str)

    sm_X = sm.add_constant(X)
    sm_model = sm.OLS(y, sm_X).fit()

    for p_value in sm_model.pvalues[1:]:
        if p_value_threshold_two <= p_value < p_value_threshold_one:
            p_value_str = str(np.round(p_value, precision)) + ' (*)'
        elif p_value < p_value_threshold_two:
            p_value_str = str(np.round(p_value, precision)) + ' (**)'
        else:
            p_value_str = str(np.round(p_value, precision)) + ' (ns)'
        classic_p_values.append(p_value_str)

    return model_coefs, permuted_p_values, classic_p_values

#%%
# Perform permutation test
model = LinearRegression()
coefs, permuted_p_values, classic_p_values = permutation_test(X, y, model)

#%%
# Print the results
print("Regression Model Coefficients and p-Values\n")
print("Coefficients:", coefs)
print("Empirical p-Values:", permuted_p_values)
print("Asymptotic p-Values:", classic_p_values)
print("\n(*) p-value < 0.05\n(**) p-value < 0.01\n(ns) p-value >= 0.05\n")

#%%
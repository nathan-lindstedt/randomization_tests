#%%
# Import libraries
import numpy as np

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
def permutation_test_regression(X, y, model, n_permutations=100_000, p_value_threshold_one=0.05, p_value_threshold_two=0.01):
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
    p_value_threshold_one : float, optional (default=0.05)
        The threshold for the first level of significance.
    p_value_threshold_two : float, optional (default=0.01)
        The threshold for the second level of significance.

    Returns
    ----------
    model_coefs : list of float
        The original coefficients of the fitted model.
    p_values : list of str
        The p-values for each coefficient, formatted with significance stars if below the thresholds.
    """
    model.fit(X, y)
    model_coefs = model.coef_.flatten().tolist()

    permuted_coefs = []
    for _ in range(n_permutations):
        model.fit(X, np.random.permutation(y))
        permuted_coefs.append(model.coef_.flatten().tolist())

    p_values = []
    for i in range(len(model_coefs)):
        p_value = (np.abs(np.array(permuted_coefs)[:, i]) >= np.abs(np.array(model_coefs)[i])).mean()
        if p_value_threshold_two <= p_value < p_value_threshold_one:
            p_value_str = str(np.round(p_value, 2)) + ' (*)'
        elif p_value < p_value_threshold_two:
            p_value_str = str(np.round(p_value, 2)) + ' (**)'
        else:
            p_value_str = str(np.round(p_value, 2)) + ' (ns)'
        p_values.append(p_value_str)

    return model_coefs, p_values

#%%
# Perform permutation test
model = LinearRegression()
coefs, p_values = permutation_test_regression(X, y, model)

#%%
# Print the results
print("Regression Model Coefficients and Permutation Test p-Values\n")
print("Coefficients:", coefs)
print("p-Values:", p_values)
print("\n(*) p-value < 0.05\n(**) p-value < 0.01\n(ns) p-value >= 0.05\n")

#%%
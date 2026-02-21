"""Shared type aliases for the randomization_tests package."""

import numpy as np
import pandas as pd

# Array-like inputs accepted by the public API.
ArrayLike = np.ndarray | pd.DataFrame | pd.Series

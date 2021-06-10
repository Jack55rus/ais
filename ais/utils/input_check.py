from typing import Any

import numpy as np
import pandas as pd


def nan_check(inp_data: Any):
    arr = np.array(inp_data, dtype=float)
    if np.isnan(arr).sum() > 0:
        raise ValueError("Cannot process sparse input")
    return arr


def array_like_check(inp_arr: Any):
    if not isinstance(inp_arr, (pd.DataFrame, np.ndarray, list)):
        raise TypeError("Only array-like objects can be passed")

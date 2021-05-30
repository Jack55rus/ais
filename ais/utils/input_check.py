from typing import Any

import numpy as np
import pandas as pd


def nan_check(df: pd.DataFrame):
    if df.isna().sum().sum() > 0:
        raise ValueError("Cannot process sparse input")


def array_like_check(inp_arr: Any):
    if not isinstance(inp_arr, (pd.DataFrame, np.ndarray, list)):
        raise TypeError("Only array-like objects can be passed")


def type_check(df: pd.DataFrame):
    try:
        df.astype(float)
    except ValueError:
        print("Could not convert some types to float")

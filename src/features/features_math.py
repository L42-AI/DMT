import pandas as pd
import numpy as np

def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """ Ensure expected columns exist, filling missing ones with 0.0. """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
    return df

def safe_divide(a: pd.Series, b: pd.Series, eps: float = 1e-8) -> pd.Series:
    """ Divide safely by adding a tiny constant to the denominator. """
    return a / (b + eps)

def rowwise_entropy(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """ Compute Shannon entropy row-wise (vectorized for wide dataframes). """
    values = df[cols].fillna(0).to_numpy(dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    
    # Convert to proportions safely
    probs = np.divide(values, np.maximum(row_sums, 1e-10))
    
    # H = -sum(p * log(p))
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)
    
    return pd.Series(entropy, index=df.index)
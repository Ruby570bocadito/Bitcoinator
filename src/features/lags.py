import pandas as pd
import numpy as np


def add_lag_features(df: pd.DataFrame, target_col: str = 'Close',
                     lags: list = [1, 2, 7, 14, 30]) -> pd.DataFrame:
    """Add lag features for target column."""
    df = df.copy()
    
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    return df


def add_return_features(df: pd.DataFrame, target_col: str = 'Close',
                        periods: list = [1, 7, 14, 30]) -> pd.DataFrame:
    """Add return (percentage change) features."""
    df = df.copy()
    
    for period in periods:
        df[f'return_{period}'] = df[target_col].pct_change(period)
    
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = 'Close',
                          windows: list = [7, 14, 30, 60]) -> pd.DataFrame:
    """Add rolling statistics (mean, std, min, max)."""
    df = df.copy()
    
    for window in windows:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df[target_col].rolling(window=window).max()
    
    return df
import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data: handle missing values, outliers, duplicates."""
    df = df.copy()
    
    initial_rows = len(df)
    
    df = df.drop_duplicates()
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Close', 'Open', 'High', 'Low'])
    
    df = df[df['Close'] > 0]
    df = df[df['Volume'] >= 0]
    
    df = df[(df['High'] >= df['Low']) & 
            (df['High'] >= df['Close']) & 
            (df['High'] >= df['Open']) &
            (df['Low'] <= df['Close']) & 
            (df['Low'] <= df['Open'])]
    
    print(f"Cleaned: {initial_rows} -> {len(df)} rows ({initial_rows - len(df)} removed)")
    
    return df.reset_index(drop=True)


def handle_missing(df: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
    """Handle missing values with forward/backward fill or interpolation."""
    df = df.copy()
    
    if method == 'forward':
        df = df.fillna(method='ffill')
    elif method == 'backward':
        df = df.fillna(method='bfill')
    elif method == 'interpolate':
        df = df.interpolate(method='time')
    
    return df


def remove_outliers(df: pd.DataFrame, columns: list = None, 
                    n_std: float = 5.0) -> pd.DataFrame:
    """Remove outliers using z-score method."""
    df = df.copy()
    
    if columns is None:
        columns = ['Close', 'Volume']
    
    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[np.abs(df[col] - mean) <= n_std * std]
    
    return df.reset_index(drop=True)
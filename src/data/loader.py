import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def load_data(filepath: str, config_path: str = "config.yaml") -> pd.DataFrame:
    """Load and validate the BTC dataset."""
    config = yaml.safe_load(open(config_path))
    
    df = pd.read_csv(filepath)
    
    required_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Validate data quality and return report."""
    report = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'price_stats': df['Close'].describe().to_dict()
    }
    return report


def get_split_indices(df: pd.DataFrame, train_ratio: float = 0.7, 
                      val_ratio: float = 0.15, test_ratio: float = 0.15):
    """Get temporal split indices."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return {
        'train': (0, train_end),
        'val': (train_end, val_end),
        'test': (val_end, n)
    }


def split_data(df: pd.DataFrame, config_path: str = "config.yaml") -> dict:
    """Split data into train/val/test temporally."""
    config = yaml.safe_load(open(config_path))
    
    splits = get_split_indices(
        df,
        config['data']['train_ratio'],
        config['data']['val_ratio'],
        config['data']['test_ratio']
    )
    
    result = {}
    for name, (start, end) in splits.items():
        result[name] = df.iloc[start:end].copy()
        print(f"{name}: {len(result[name])} rows ({start} to {end})")
    
    return result
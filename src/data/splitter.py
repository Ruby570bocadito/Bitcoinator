import pandas as pd
import numpy as np
from typing import Tuple


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally (no data leakage)."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                output_dir: str = "data/splits"):
    """Save train/val/test splits to CSV."""
    train.to_csv(f"{output_dir}/train.csv", index=False)
    val.to_csv(f"{output_dir}/val.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)
    
    print(f"Saved splits: train={len(train)}, val={len(val)}, test={len(test)}")
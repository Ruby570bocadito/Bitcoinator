import pandas as pd
import numpy as np


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features: hour, day of week, month, etc."""
    df = df.copy()
    
    if 'Timestamp' in df.columns:
        df['hour'] = df['Timestamp'].dt.hour
        df['day_of_week'] = df['Timestamp'].dt.dayofweek
        df['day_of_month'] = df['Timestamp'].dt.day
        df['month'] = df['Timestamp'].dt.month
        df['quarter'] = df['Timestamp'].dt.quarter
        df['year'] = df['Timestamp'].dt.year
        
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
    return df
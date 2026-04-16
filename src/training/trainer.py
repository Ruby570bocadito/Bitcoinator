import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


class ScalerWrapper:
    """Wrapper for data scaling."""
    def __init__(self, scaler_type: str = 'standard'):
        self.scaler_type = scaler_type
        self.scaler = None
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        return self.scaler.fit_transform(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)
    
    def save(self, filepath: str):
        joblib.dump(self.scaler, filepath)
    
    def load(self, filepath: str):
        self.scaler = joblib.load(filepath)


class Trainer:
    """Generic trainer for ML models."""
    def __init__(self, model, scaler: str = 'standard'):
        self.model = model
        self.scaler = ScalerWrapper(scaler)
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close'):
        """Prepare features and target."""
        feature_cols = [c for c in df.columns if c not in [target_col, 'Timestamp']]
        X = df[feature_cols].values
        y = df[target_col].values
        return X, y, feature_cols
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train the model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(X_train_scaled, y_train)
            return X_train_scaled, X_val_scaled
        
        self.model.fit(X_train_scaled, y_train)
        return X_train_scaled
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
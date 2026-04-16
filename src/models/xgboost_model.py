import numpy as np
import pandas as pd
import xgboost as xgb
import joblib


class XGBoostModel:
    """XGBoost model for price prediction."""
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, random_state: int = 42,
                 n_jobs: int = -1, subsample: float = 1.0,
                 colsample_bytree: float = 1.0, min_child_weight: int = 1,
                 early_stopping_rounds: int = None):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            early_stopping_rounds=early_stopping_rounds
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        print(f"XGBoost trained with {len(X)} samples")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filepath: str):
        self.model.save_model(filepath)
    
    def load(self, filepath: str):
        self.model.load_model(filepath)
    
    def feature_importance(self) -> dict:
        importance = self.model.feature_importances_
        return dict(zip(range(len(importance)), importance))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib


class RandomForestModel:
    """Random Forest model for price prediction."""
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 random_state: int = 42, n_jobs: int = -1):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        print(f"RandomForest trained with {len(X)} samples")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filepath: str):
        joblib.dump(self.model, filepath)
    
    def load(self, filepath: str):
        self.model = joblib.load(filepath)
    
    def feature_importance(self) -> dict:
        return dict(zip(range(len(self.model.feature_importances_)), 
                        self.model.feature_importances_))
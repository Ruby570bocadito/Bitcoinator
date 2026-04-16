import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class NaiveModel:
    """Predicts previous value as next value."""
    def predict(self, X):
        return X[-1] if len(X) > 0 else 0
    
    def fit(self, X, y):
        pass


class MovingAverageModel:
    """Predicts using moving average."""
    def __init__(self, window: int = 7):
        self.window = window
    
    def predict(self, X):
        if len(X) < self.window:
            return np.mean(X)
        return np.mean(X[-self.window:])
    
    def fit(self, X, y):
        pass


class LinearRegressionModel:
    """Simple linear regression baseline."""
    def __init__(self):
        self.model = LinearRegression()
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
import optuna
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


class HyperparameterTuner:
    """Hyperparameter optimization using Optuna."""
    def __init__(self, model_class, param_space: dict, n_trials: int = 50):
        self.model_class = model_class
        self.param_space = param_space
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = float('inf')
    
    def objective(self, trial):
        params = {}
        for param_name, param_config in self.param_space.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices'])
        
        model = self.model_class(**params)
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            score = np.sqrt(np.mean((pred - y_val) ** 2))
            scores.append(score)
        
        mean_score = np.mean(scores)
        
        if mean_score < self.best_score:
            self.best_score = mean_score
            self.best_params = params
        
        return mean_score
    
    def tune(self, X, y):
        """Run optimization."""
        self.X = X
        self.y = y
        
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        return self.best_params, self.best_score
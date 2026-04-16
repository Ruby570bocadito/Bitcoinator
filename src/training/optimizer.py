"""
Hyperparameter Optimization using Optuna.
Implements efficient hyperparameter search for XGBoost, LightGBM, and other models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable, Tuple, List
from pathlib import Path
import joblib
from src.utils.logger import setup_logger
from src.utils.config import OptunaConfig, get_config
from src.validation.walk_forward import WalkForwardValidator, WalkForwardConfig

logger = setup_logger("optimizer")

try:
    import optuna
    from optuna.pruners import MedianPruner, PercentilePruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Install with: pip install optuna")


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.
    
    Supports:
    - XGBoost
    - LightGBM
    - Random Forest
    - Custom models
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        config: Optional[OptunaConfig] = None,
        walk_forward_config: Optional[WalkForwardConfig] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            model_type: Type of model to optimize ("xgboost", "lightgbm", "random_forest")
            config: OptunaConfig instance
            walk_forward_config: WalkForwardConfig for cross-validation
        """
        self.model_type = model_type.lower()
        self.config = config or OptunaConfig()
        self.walk_forward_config = walk_forward_config or WalkForwardConfig(
            n_splits=3,
            train_size=252,
            test_size=63,
            step_size=63
        )
        self.logger = logger
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")
        
        # Initialize study
        self._init_study()
    
    def _init_study(self):
        """Initialize Optuna study."""
        pruner_map = {
            "median": MedianPruner,
            "percentile": lambda: PercentilePruner(percentile=25),
            "success_halving": SuccessiveHalvingPruner
        }
        
        pruner = pruner_map.get(self.config.pruner, MedianPruner)()
        
        try:
            self.study = optuna.create_study(
                study_name=self.config.study_name,
                storage=f"sqlite:///{self.config.storage}",
                direction="minimize",
                pruner=pruner,
                load_if_exists=True
            )
            self.logger.info(f"Loaded/created study: {self.config.study_name}")
        except Exception as e:
            self.logger.warning(f"Could not use storage: {e}. Using in-memory study.")
            self.study = optuna.create_study(
                study_name=self.config.study_name,
                direction="minimize",
                pruner=pruner
            )
    
    def _suggest_xgboost_params(self, trial: optuna.Trial) -> Dict:
        """Suggest XGBoost hyperparameters."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        }
        
        # Tree method
        params["tree_method"] = trial.suggest_categorical(
            "tree_method", ["hist", "approx"]
        )
        
        return params
    
    def _suggest_lightgbm_params(self, trial: optuna.Trial) -> Dict:
        """Suggest LightGBM hyperparameters."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 300),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        
        return params
    
    def _suggest_random_forest_params(self, trial: optuna.Trial) -> Dict:
        """Suggest Random Forest hyperparameters."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        
        return params
    
    def _get_model_class(self):
        """Get model class based on model type."""
        if self.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        Objective function for optimization.
        
        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        
        Returns:
            RMSE on validation set
        """
        # Get hyperparameters
        if self.model_type == "xgboost":
            params = self._suggest_xgboost_params(trial)
        elif self.model_type == "lightgbm":
            params = self._suggest_lightgbm_params(trial)
        elif self.model_type == "random_forest":
            params = self._suggest_random_forest_params(trial)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create and train model
        model_class = self._get_model_class()
        
        try:
            model = model_class(
                **params,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            # Fit model
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            except TypeError:
                # Some versions don't support early_stopping_rounds
                model.fit(X_train, y_train, verbose=False)
            
            # Predict and calculate RMSE
            preds = model.predict(X_val)
            rmse = np.sqrt(np.mean((preds - y_val) ** 2))
            
            # Report intermediate value for pruning
            trial.report(rmse, trial.number)
            
            # Prune if promising
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return rmse
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return float("inf")
    
    def _objective_walk_forward(
        self,
        trial: optuna.Trial,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "Close"
    ) -> float:
        """
        Objective function using walk-forward validation.
        
        Args:
            trial: Optuna trial
            data: Full dataset
            feature_cols: Feature column names
            target_col: Target column name
        
        Returns:
            Mean RMSE across all folds
        """
        validator = WalkForwardValidator(self.walk_forward_config)
        rmse_scores = []
        
        for train_df, test_df in validator.split(data):
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_test = test_df[feature_cols].values
            y_test = test_df[target_col].values
            
            # Scale data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Get hyperparameters
            if self.model_type == "xgboost":
                params = self._suggest_xgboost_params(trial)
            elif self.model_type == "lightgbm":
                params = self._suggest_lightgbm_params(trial)
            elif self.model_type == "random_forest":
                params = self._suggest_random_forest_params(trial)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Train model
            model_class = self._get_model_class()
            model = model_class(**params, random_state=42, n_jobs=-1, verbose=-1)
            
            try:
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            except TypeError:
                model.fit(X_train_scaled, y_train, verbose=False)
            
            # Calculate RMSE
            preds = model.predict(X_test_scaled)
            rmse = np.sqrt(np.mean((preds - y_test) ** 2))
            rmse_scores.append(rmse)
        
        mean_rmse = np.mean(rmse_scores)
        trial.report(mean_rmse, trial.number)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_rmse
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> Tuple[Dict, optuna.Study]:
        """
        Run hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of trials (overrides config)
            timeout: Timeout in seconds (overrides config)
        
        Returns:
            Tuple of (best_params, study)
        """
        n_trials = n_trials or self.config.n_trials
        timeout = timeout or self.config.timeout
        
        self.logger.info(f"Starting optimization for {self.model_type}...")
        self.logger.info(f"Trials: {n_trials}, Timeout: {timeout}s")
        
        # Create objective function
        objective = lambda trial: self._objective(
            trial, X_train, y_train, X_val, y_val
        )
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        self.logger.info(f"Best RMSE: ${best_value:,.0f}")
        self.logger.info(f"Best params: {best_params}")
        
        return best_params, self.study
    
    def optimize_walk_forward(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "Close",
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> Tuple[Dict, optuna.Study]:
        """
        Run hyperparameter optimization with walk-forward validation.
        
        Args:
            data: Full dataset
            feature_cols: Feature column names
            target_col: Target column name
            n_trials: Number of trials
            timeout: Timeout in seconds
        
        Returns:
            Tuple of (best_params, study)
        """
        n_trials = n_trials or self.config.n_trials
        timeout = timeout or self.config.timeout
        
        self.logger.info(
            f"Starting walk-forward optimization for {self.model_type}..."
        )
        
        objective = lambda trial: self._objective_walk_forward(
            trial, data, feature_cols, target_col
        )
        
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        self.logger.info(f"Best mean RMSE: ${best_value:,.0f}")
        self.logger.info(f"Best params: {best_params}")
        
        return best_params, self.study
    
    def get_best_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Create and return best model with optimal hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
        
        Returns:
            Trained model with best hyperparameters
        """
        best_params = self.study.best_params
        model_class = self._get_model_class()
        
        model = model_class(**best_params, random_state=42, n_jobs=-1, verbose=0)
        model.fit(X_train, y_train)
        
        self.logger.info("Created model with best hyperparameters")
        return model
    
    def save_study(self, filepath: str):
        """Save study to file."""
        joblib.dump(self.study, filepath)
        self.logger.info(f"Study saved to {filepath}")
    
    def load_study(self, filepath: str):
        """Load study from file."""
        self.study = joblib.load(filepath)
        self.logger.info(f"Study loaded from {filepath}")
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if not OPTUNA_AVAILABLE:
            return
        
        fig = optuna.visualization.plot_optimization_history(self.study)
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    
    def plot_param_importances(self, save_path: Optional[str] = None):
        """Plot parameter importances."""
        if not OPTUNA_AVAILABLE:
            return
        
        fig = optuna.visualization.plot_param_importances(self.study)
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    
    def plot_parallel_coordinate(self, save_path: Optional[str] = None):
        """Plot parallel coordinate."""
        if not OPTUNA_AVAILABLE:
            return
        
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        
        if save_path:
            fig.write_image(save_path)
        
        return fig


def optimize_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100
) -> Tuple[Dict, any]:
    """
    Convenience function to optimize a model.
    
    Args:
        model_type: Type of model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of trials
    
    Returns:
        Tuple of (best_params, study)
    """
    optimizer = HyperparameterOptimizer(model_type=model_type)
    return optimizer.optimize(X_train, y_train, X_val, y_val, n_trials=n_trials)

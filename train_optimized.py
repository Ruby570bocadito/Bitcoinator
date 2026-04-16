"""
Bitcoinator - Optimized Training Script

Features:
- Configuration from config.yaml
- Structured logging
- MLflow experiment tracking
- Optuna hyperparameter optimization
- Walk-forward validation
- Backtesting with trading metrics
- Model ensemble support

Usage:
    python train_optimized.py --model xgboost --optimize --walk-forward
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.config import get_config, Config
from src.utils.logger import setup_logger
from src.utils.mlflow_tracker import get_tracker, track_experiment
from src.data.loader import load_data
from src.data.cleaner import clean_data
from src.features.technical import add_technical_indicators
from src.features.temporal import add_temporal_features
from src.features.lags import add_lag_features, add_rolling_features
from src.validation.walk_forward import (
    WalkForwardValidator,
    WalkForwardConfig,
    walk_forward_validation,
    aggregate_walk_forward_results
)
from src.backtesting.backtester import Backtester, BacktestConfig, run_backtest
from src.evaluation.metrics import evaluate_model, print_metrics, evaluate_trades
from src.training.optimizer import HyperparameterOptimizer

# Try to import MLflow and Optuna
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class BitcoinatorTrainer:
    """
    Main trainer class for Bitcoinator project.
    
    Integrates all components:
    - Data loading and preprocessing
    - Feature engineering
    - Model training with hyperparameter optimization
    - Walk-forward validation
    - Backtesting
    - Experiment tracking
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        model_type: str = "xgboost",
        use_mlflow: bool = True,
        use_optuna: bool = True,
        verbose: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
            model_type: Type of model to train
            use_mlflow: Whether to use MLflow tracking
            use_optuna: Whether to use Optuna optimization
            verbose: Whether to print verbose output
        """
        # Load configuration
        self.config = get_config(config_path)
        self.model_type = model_type
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.verbose = verbose
        
        # Setup logging
        self.logger = setup_logger(
            "bitcoinator_trainer",
            log_level="INFO" if verbose else "WARNING",
            log_dir=self.config.paths.logs
        )
        
        # Initialize MLflow tracker
        if self.use_mlflow:
            self.tracker = get_tracker()
            self.logger.info("MLflow tracking enabled")
        else:
            self.tracker = None
            self.logger.info("MLflow tracking disabled")
        
        # State variables
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.results = {}
        
        self.logger.info(f"Training initialized for model: {model_type}")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare data with all features.
        
        Returns:
            DataFrame with all features
        """
        self.logger.info("Loading data...")
        
        # Load raw data
        df = load_data(self.config.paths.raw_data)
        
        # Convert timestamp and resample to daily
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df = df.set_index('Timestamp').resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()
        
        # Filter date range
        df = df[df['Timestamp'] >= '2020-01-01']
        
        # Clean data
        df = clean_data(df)
        
        self.logger.info(f"Loaded {len(df)} rows of daily data")
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features.
        
        Args:
            df: Raw DataFrame
        
        Returns:
            DataFrame with all features
        """
        self.logger.info("Creating features...")
        
        # Technical indicators
        df = add_technical_indicators(df)
        self.logger.info("  - Technical indicators added")
        
        # Temporal features
        df = add_temporal_features(df)
        self.logger.info("  - Temporal features added")
        
        # Lag features
        df = add_lag_features(df, lags=list(range(1, 61)))
        self.logger.info("  - Lag features (1-60) added")
        
        # Rolling features
        df = add_rolling_features(
            df,
            windows=[3, 5, 7, 14, 21, 30, 60, 90]
        )
        self.logger.info("  - Rolling features added")
        
        # Drop NaN values
        df = df.dropna()
        
        self.logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def prepare_data_splits(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare train/validation/test splits.
        
        Args:
            df: DataFrame with features
        
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        self.logger.info("Preparing data splits...")
        
        # Get feature columns
        self.feature_cols = [
            c for c in df.columns
            if c not in [self.config.data.target_column, 'Timestamp']
        ]
        
        # Split data
        n = len(df)
        train_size = int(n * self.config.data.train_ratio)
        val_size = int(n * self.config.data.val_ratio)
        
        train = df.iloc[:train_size]
        val = df.iloc[train_size:train_size + val_size]
        test = df.iloc[train_size + val_size:]
        
        # Prepare features and targets
        X_train = train[self.feature_cols].values
        y_train = train[self.config.data.target_column].values
        X_val = val[self.feature_cols].values
        y_val = val[self.config.data.target_column].values
        X_test = test[self.feature_cols].values
        y_test = test[self.config.data.target_column].values
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, test
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Optional[Dict] = None
    ):
        """
        Train model with given hyperparameters.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            params: Hyperparameters (optional)
        """
        self.logger.info(f"Training {self.model_type} model...")
        
        if self.model_type == "xgboost":
            import xgboost as xgb
            
            default_params = {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'min_child_weight': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'tree_method': 'hist',
                'n_jobs': -1,
                'verbosity': 0
            }
            
            if params:
                default_params.update(params)
            
            self.model = xgb.XGBRegressor(**default_params)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            
            default_params = {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            if params:
                default_params.update(params)
            
            self.model = lgb.LGBMRegressor(**default_params)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            
            default_params = {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
            
            if params:
                default_params.update(params)
            
            self.model = RandomForestRegressor(**default_params)
            self.model.fit(X_train, y_train)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.logger.info("Model training completed")
    
    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100
    ) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_trials: Number of trials
        
        Returns:
            Best hyperparameters
        """
        self.logger.info(f"Starting hyperparameter optimization ({n_trials} trials)...")
        
        optimizer = HyperparameterOptimizer(
            model_type=self.model_type,
            config=self.config.optuna
        )
        
        best_params, study = optimizer.optimize(
            X_train, y_train, X_val, y_val,
            n_trials=n_trials
        )
        
        self.logger.info(f"Best RMSE: ${study.best_value:,.0f}")
        self.logger.info(f"Best params: {best_params}")
        
        # Log to MLflow
        if self.use_mlflow and self.tracker:
            self.tracker.log_json(best_params, "best_hyperparameters.json")
        
        return best_params
    
    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test, y_test: Test data
            test_df: Test DataFrame with timestamps
        
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Evaluating model on test set...")
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = evaluate_model(y_test, predictions)
        
        # Run backtest
        self.logger.info("Running backtest...")
        backtest_result = run_backtest(
            test_df,
            predictions,
            self.config.backtest
        )
        
        # Add trading metrics
        metrics.update(backtest_result.metrics)
        
        # Print metrics
        print_metrics(metrics, f"{self.model_type.upper()} Test Set")
        
        # Store results
        self.results = {
            'predictions': predictions,
            'actuals': y_test,
            'metrics': metrics,
            'backtest': backtest_result
        }
        
        # Log to MLflow
        if self.use_mlflow and self.tracker:
            self.tracker.log_metrics(metrics)
        
        return metrics
    
    def run_walk_forward_validation(
        self,
        df: pd.DataFrame,
        n_splits: int = 5
    ) -> Dict:
        """
        Run walk-forward validation.
        
        Args:
            df: Full dataset
            n_splits: Number of splits
        
        Returns:
            Aggregated results
        """
        self.logger.info(f"Running walk-forward validation ({n_splits} splits)...")
        
        # Configure walk-forward
        wf_config = WalkForwardConfig(
            n_splits=n_splits,
            train_size=int(len(df) * 0.6),
            test_size=int(len(df) * 0.1),
            step_size=int(len(df) * 0.05),
            expanding_window=True
        )
        
        # Get model class
        if self.model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                tree_method='hist',
                n_jobs=-1,
                verbosity=0
            )
        else:
            raise NotImplementedError(f"Walk-forward not implemented for {self.model_type}")
        
        # Run walk-forward
        results = walk_forward_validation(
            df,
            model,
            self.feature_cols,
            self.config.data.target_column,
            wf_config,
            self.scaler
        )
        
        # Aggregate results
        aggregated = aggregate_walk_forward_results(results)
        
        self.logger.info("Walk-forward validation complete")
        self.logger.info(f"Mean RMSE: ${aggregated['statistics']['RMSE']['mean']:,.0f}")
        self.logger.info(f"Std RMSE: ${aggregated['statistics']['RMSE']['std']:,.0f}")
        
        return aggregated
    
    def save_model(self, filepath: str = "models/model_best.json"):
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            self.logger.warning("No model to save")
            return
        
        # Create directory if needed
        Path(filepath).parent.mkdir(exist_ok=True)
        
        # Save model
        if self.model_type == "xgboost":
            self.model.save_model(filepath)
        else:
            joblib.dump(self.model, filepath.replace('.json', '.pkl'))
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = filepath.replace('.json', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
        
        # Save feature columns
        if self.feature_cols is not None:
            features_path = filepath.replace('.json', '_features.json')
            with open(features_path, 'w') as f:
                json.dump(self.feature_cols, f, indent=2)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def train(
        self,
        optimize: bool = False,
        n_trials: int = 100,
        walk_forward: bool = False,
        n_wf_splits: int = 5,
        save: bool = True
    ) -> Dict:
        """
        Main training method.
        
        Args:
            optimize: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            walk_forward: Whether to run walk-forward validation
            n_wf_splits: Number of walk-forward splits
            save: Whether to save the model
        
        Returns:
            Training results
        """
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("BITCOINATOR TRAINING")
        self.logger.info("=" * 60)
        
        # Start MLflow run
        if self.use_mlflow and self.tracker:
            run_name = f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.tracker.start_run(run_name=run_name)
            self.tracker.log_params({
                'model_type': self.model_type,
                'optimize': optimize,
                'n_trials': n_trials,
                'walk_forward': walk_forward,
                **self.config.to_dict()
            })
        
        try:
            # Load and prepare data
            df = self.load_and_prepare_data()
            df = self.create_features(df)
            
            # Prepare splits
            X_train, X_val, X_test, y_train, y_val, y_test, test_df = \
                self.prepare_data_splits(df)
            
            # Optimize hyperparameters if requested
            if optimize and self.use_optuna:
                best_params = self.optimize_hyperparameters(
                    X_train, y_train, X_val, y_val,
                    n_trials=n_trials
                )
                self.train_model(X_train, y_train, X_val, y_val, best_params)
            else:
                self.train_model(X_train, y_train, X_val, y_val)
            
            # Evaluate on test set
            metrics = self.evaluate_model(X_test, y_test, test_df)
            
            # Run walk-forward validation if requested
            if walk_forward:
                wf_results = self.run_walk_forward_validation(df, n_wf_splits)
                metrics['walk_forward'] = wf_results
                
                if self.use_mlflow and self.tracker:
                    self.tracker.log_metrics({
                        'wf_mean_rmse': wf_results['statistics']['RMSE']['mean'],
                        'wf_std_rmse': wf_results['statistics']['RMSE']['std']
                    })
            
            # Save model
            if save:
                self.save_model()
            
            # Calculate total time
            total_time = time.time() - start_time
            self.logger.info(f"Training completed in {total_time/60:.1f} minutes")
            
            # End MLflow run
            if self.use_mlflow and self.tracker:
                self.tracker.end_run()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            
            if self.use_mlflow and self.tracker:
                self.tracker.end_run()
            
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bitcoinator Optimized Training"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "random_forest"],
        help="Model type to train"
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization"
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials"
    )
    
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward validation"
    )
    
    parser.add_argument(
        "--n-wf-splits",
        type=int,
        default=5,
        help="Number of walk-forward splits"
    )
    
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking"
    )
    
    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Disable Optuna optimization"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BitcoinatorTrainer(
        config_path=args.config,
        model_type=args.model,
        use_mlflow=not args.no_mlflow,
        use_optuna=not args.no_optuna,
        verbose=args.verbose
    )
    
    # Run training
    results = trainer.train(
        optimize=args.optimize,
        n_trials=args.n_trials,
        walk_forward=args.walk_forward,
        n_wf_splits=args.n_wf_splits,
        save=True
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model.upper()}")
    print(f"Test RMSE: ${results['RMSE']:,.0f}")
    print(f"Test MAE: ${results['MAE']:,.0f}")
    print(f"Directional Accuracy: {results['Directional_Accuracy']:.1f}%")
    
    # Print trading metrics if available
    if 'Sharpe_Ratio' in results:
        print(f"Sharpe Ratio: {results['Sharpe_Ratio']:.2f}")
    if 'Max_Drawdown' in results:
        print(f"Max Drawdown: {results['Max_Drawdown']:.2f}%")
    if 'total_return' in results:
        print(f"Total Return: {results['total_return']:.2f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

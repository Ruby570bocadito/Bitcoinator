"""
Walk-Forward Validation for Time Series.
Implements expanding window and sliding window cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, List, Dict, Optional
from dataclasses import dataclass
from src.utils.logger import setup_logger

logger = setup_logger("walk_forward")


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    n_splits: int = 5
    train_size: int = 252  # ~1 year of daily data
    test_size: int = 63    # ~3 months of daily data
    step_size: int = 63    # Step forward by 3 months
    gap: int = 0           # Gap between train and test
    min_train_size: int = 126  # Minimum training samples
    expanding_window: bool = True  # True = expanding, False = sliding


class WalkForwardValidator:
    """
    Walk-Forward Validation for time series data.
    
    This class provides methods to split time series data in a way that
    respects temporal ordering and prevents look-ahead bias.
    """
    
    def __init__(self, config: Optional[WalkForwardConfig] = None):
        """
        Initialize walk-forward validator.
        
        Args:
            config: WalkForwardConfig instance or None for defaults
        """
        self.config = config or WalkForwardConfig()
        self.logger = logger
    
    def split(
        self,
        data: pd.DataFrame,
        date_column: str = "Timestamp"
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate walk-forward splits.
        
        Args:
            data: DataFrame with time series data
            date_column: Name of the date column
        
        Yields:
            Tuple of (train_df, test_df) for each split
        """
        data = data.copy()
        
        # Sort by date
        if date_column in data.columns:
            data = data.sort_values(date_column).reset_index(drop=True)
        
        n_samples = len(data)
        
        # Calculate initial train end
        train_end = self.config.train_size
        test_end = train_end + self.config.test_size
        
        split_num = 0
        
        while test_end <= n_samples:
            # Define train indices
            if self.config.expanding_window:
                train_start = 0
            else:
                train_start = max(0, train_end - self.config.train_size)
            
            train_start_idx = train_start
            train_end_idx = train_end - self.config.gap
            
            # Define test indices
            test_start_idx = train_end
            test_end_idx = min(test_end, n_samples)
            
            # Check minimum train size
            if train_end_idx - train_start_idx < self.config.min_train_size:
                self.logger.warning(
                    f"Split {split_num}: Train size {train_end_idx - train_start_idx} "
                    f"below minimum {self.config.min_train_size}"
                )
                train_end += self.config.step_size
                test_end = train_end + self.config.test_size
                continue
            
            # Yield split
            train_df = data.iloc[train_start_idx:train_end_idx].copy()
            test_df = data.iloc[test_start_idx:test_end_idx].copy()
            
            self.logger.info(
                f"Split {split_num}: Train={len(train_df)}, Test={len(test_df)}, "
                f"Date range: {train_df[date_column].min() if date_column in train_df.columns else 'N/A'} "
                f"to {test_df[date_column].max() if date_column in test_df.columns else 'N/A'}"
            )
            
            yield train_df, test_df
            
            # Move forward
            train_end += self.config.step_size
            test_end = train_end + self.config.test_size
            split_num += 1
        
        self.logger.info(f"Generated {split_num} walk-forward splits")
    
    def split_indices(
        self,
        n_samples: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate walk-forward split indices.
        
        Args:
            n_samples: Total number of samples
        
        Yields:
            Tuple of (train_indices, test_indices) for each split
        """
        train_end = self.config.train_size
        test_end = train_end + self.config.test_size
        
        split_num = 0
        
        while test_end <= n_samples:
            if self.config.expanding_window:
                train_start = 0
            else:
                train_start = max(0, train_end - self.config.train_size)
            
            train_start_idx = train_start
            train_end_idx = train_end - self.config.gap
            test_start_idx = train_end
            test_end_idx = min(test_end, n_samples)
            
            if train_end_idx - train_start_idx < self.config.min_train_size:
                train_end += self.config.step_size
                test_end = train_end + self.config.test_size
                continue
            
            train_indices = np.arange(train_start_idx, train_end_idx)
            test_indices = np.arange(test_start_idx, test_end_idx)
            
            yield train_indices, test_indices
            
            train_end += self.config.step_size
            test_end = train_end + self.config.test_size
            split_num += 1
    
    def get_n_splits(self, n_samples: int) -> int:
        """
        Get number of splits for given sample size.
        
        Args:
            n_samples: Total number of samples
        
        Returns:
            Number of walk-forward splits
        """
        if n_samples < self.config.train_size + self.config.test_size:
            return 0
        
        n_splits = 0
        train_end = self.config.train_size
        test_end = train_end + self.config.test_size
        
        while test_end <= n_samples:
            train_start = 0 if self.config.expanding_window else max(0, train_end - self.config.train_size)
            if train_end - train_start >= self.config.min_train_size:
                n_splits += 1
            
            train_end += self.config.step_size
            test_end = train_end + self.config.test_size
        
        return n_splits


def walk_forward_validation(
    data: pd.DataFrame,
    model,
    feature_cols: List[str],
    target_col: str = "Close",
    config: Optional[WalkForwardConfig] = None,
    scaler=None
) -> Dict[str, List]:
    """
    Perform walk-forward validation with a model.
    
    Args:
        data: DataFrame with features and target
        model: Model instance with fit() and predict() methods
        feature_cols: List of feature column names
        target_col: Name of target column
        config: WalkForwardConfig instance
        scaler: Optional scaler for feature transformation
    
    Returns:
        Dictionary with lists of predictions, actuals, and metrics per split
    """
    validator = WalkForwardValidator(config)
    
    results = {
        "predictions": [],
        "actuals": [],
        "dates": [],
        "metrics": [],
        "split_number": []
    }
    
    for split_num, (train_df, test_df) in enumerate(validator.split(data)):
        logger.info(f"Processing split {split_num}...")
        
        # Prepare data
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        
        # Scale if scaler provided
        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Train and predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Store results
        results["predictions"].extend(predictions)
        results["actuals"].extend(y_test)
        
        if "Timestamp" in test_df.columns:
            results["dates"].extend(test_df["Timestamp"].values)
        else:
            results["dates"].extend(range(len(predictions)))
        
        results["split_number"].extend([split_num] * len(predictions))
        
        # Calculate metrics for this split
        from src.evaluation.metrics import evaluate_model
        metrics = evaluate_model(y_test, predictions)
        metrics["split"] = split_num
        results["metrics"].append(metrics)
        
        logger.info(f"Split {split_num} RMSE: ${metrics['RMSE']:,.0f}")
    
    return results


def aggregate_walk_forward_results(results: Dict[str, List]) -> Dict:
    """
    Aggregate results from walk-forward validation.
    
    Args:
        results: Dictionary from walk_forward_validation()
    
    Returns:
        Dictionary with aggregated metrics
    """
    from src.evaluation.metrics import evaluate_model
    import numpy as np
    
    all_predictions = np.array(results["predictions"])
    all_actuals = np.array(results["actuals"])
    
    # Overall metrics
    overall_metrics = evaluate_model(all_actuals, all_predictions)
    
    # Per-split metrics
    split_metrics = results["metrics"]
    
    # Calculate statistics across splits
    metric_names = list(split_metrics[0].keys())
    metric_stats = {}
    
    for metric in metric_names:
        if metric == "split":
            continue
        values = [m[metric] for m in split_metrics]
        metric_stats[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "values": values
        }
    
    return {
        "overall": overall_metrics,
        "per_split": split_metrics,
        "statistics": metric_stats,
        "n_splits": len(split_metrics)
    }

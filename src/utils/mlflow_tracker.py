"""
MLflow Experiment Tracking for Bitcoinator.
Tracks experiments, parameters, metrics, and artifacts.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import pandas as pd
import numpy as np

from src.utils.logger import setup_logger
from src.utils.config import MLflowConfig, get_config

logger = setup_logger("mlflow_tracker")

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Install with: pip install mlflow")


class MLflowTracker:
    """
    MLflow experiment tracker for Bitcoinator.
    
    Features:
    - Automatic experiment tracking
    - Parameter logging
    - Metric logging
    - Artifact storage
    - Model registry
    """
    
    def __init__(self, config: Optional[MLflowConfig] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            config: MLflowConfig instance or None for defaults
        """
        self.config = config or MLflowConfig()
        self.logger = logger
        self.experiment_name = self.config.experiment_name
        self.experiment_id = None
        self.run_id = None
        self._setup()
    
    def _setup(self):
        """Set up MLflow tracking."""
        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow not available. Tracking disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Use SQLite backend (recommended for Windows and MLflow 2026+)
        tracking_path = Path(self.config.tracking_uri)
        
        # Ensure directory exists
        tracking_path.parent.mkdir(exist_ok=True)
        
        # Use SQLite database
        tracking_uri = f"sqlite:///{tracking_path.parent}/mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)
        
        self.logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
            self.logger.info(f"Created experiment: {self.experiment_name}")
        else:
            self.experiment_id = experiment.experiment_id
            self.logger.info(f"Using existing experiment: {self.experiment_name}")
        
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Optional tags dictionary
        """
        if not self.enabled:
            return
        
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
        self.run_id = self._run.info.run_id
        
        self.logger.info(f"Started MLflow run: {run_name} ({self.run_id})")
        
        # Log default tags
        mlflow.set_tag("project", "bitcoinator")
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
    
    def end_run(self):
        """End current MLflow run."""
        if not self.enabled:
            return
        
        mlflow.end_run()
        self.logger.info(f"Ended MLflow run: {self.run_id}")
        self.run_id = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.
        
        Args:
            params: Dictionary of parameters
        """
        if not self.enabled:
            return
        
        # Convert numpy types to Python types
        cleaned_params = {}
        for key, value in params.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                cleaned_params[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                cleaned_params[key] = float(value)
            elif isinstance(value, np.ndarray):
                cleaned_params[key] = value.tolist()
            elif isinstance(value, (list, dict, str, int, float, bool)):
                cleaned_params[key] = value
            else:
                cleaned_params[key] = str(value)
        
        mlflow.log_params(cleaned_params)
        self.logger.debug(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if not self.enabled:
            return
        
        # Convert numpy types
        cleaned_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                cleaned_metrics[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                cleaned_metrics[key] = float(value)
            elif isinstance(value, (int, float)):
                cleaned_metrics[key] = value
            else:
                continue
        
        mlflow.log_metrics(cleaned_metrics, step=step)
        self.logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        if not self.enabled:
            return
        mlflow.log_metric(name, value, step=step)
    
    def log_artifact(self, filepath: str, artifact_path: Optional[str] = None):
        """
        Log an artifact.
        
        Args:
            filepath: Path to the file
            artifact_path: Optional subdirectory in artifacts
        """
        if not self.enabled:
            return
        
        if not Path(filepath).exists():
            self.logger.warning(f"Artifact not found: {filepath}")
            return
        
        mlflow.log_artifact(filepath, artifact_path)
        self.logger.debug(f"Logged artifact: {filepath}")
    
    def log_artifacts(self, folder: str, artifact_path: Optional[str] = None):
        """
        Log all artifacts in a folder.
        
        Args:
            folder: Path to the folder
            artifact_path: Optional subdirectory in artifacts
        """
        if not self.enabled:
            return
        
        if not Path(folder).exists():
            self.logger.warning(f"Artifact folder not found: {folder}")
            return
        
        mlflow.log_artifacts(folder, artifact_path)
        self.logger.debug(f"Logged artifacts from: {folder}")
    
    def log_model(
        self,
        model,
        model_type: str = "sklearn",
        artifact_path: str = "model",
        signature=None,
        input_example=None
    ):
        """
        Log a model.
        
        Args:
            model: Model instance
            model_type: Type of model ("sklearn", "xgboost", "lightgbm")
            artifact_path: Path in artifacts
            signature: Model signature
            input_example: Input example
        """
        if not self.enabled:
            return
        
        if not self.config.log_models:
            self.logger.info("Model logging disabled in config")
            return
        
        try:
            if model_type == "xgboost":
                mlflow.xgboost.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example
                )
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example
                )
            
            self.logger.info(f"Logged model to {artifact_path}")
        except Exception as e:
            self.logger.warning(f"Could not log model: {e}")
    
    def log_figure(self, fig, name: str):
        """
        Log a matplotlib/plotly figure.
        
        Args:
            fig: Figure object
            name: Name for the figure
        """
        if not self.enabled:
            return
        
        try:
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name
            
            if hasattr(fig, 'savefig'):
                # Matplotlib figure
                fig.savefig(temp_path, bbox_inches='tight')
            elif hasattr(fig, 'write_image'):
                # Plotly figure
                fig.write_image(temp_path)
            else:
                self.logger.warning(f"Unknown figure type: {type(fig)}")
                return
            
            self.log_artifact(temp_path, artifact_path="figures")
            
            # Clean up
            os.unlink(temp_path)
            
            self.logger.info(f"Logged figure: {name}")
            
        except Exception as e:
            self.logger.warning(f"Could not log figure: {e}")
    
    def log_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        max_rows: int = 10000
    ):
        """
        Log a DataFrame as CSV artifact.
        
        Args:
            df: DataFrame to log
            name: Name for the file
            max_rows: Maximum rows to log
        """
        if not self.enabled:
            return
        
        try:
            import tempfile
            
            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode='w'
            ) as f:
                temp_path = f.name
                df.head(max_rows).to_csv(f, index=False)
            
            self.log_artifact(temp_path, artifact_path="data")
            os.unlink(temp_path)
            
            self.logger.info(f"Logged DataFrame: {name} ({len(df)} rows)")
            
        except Exception as e:
            self.logger.warning(f"Could not log DataFrame: {e}")
    
    def log_json(self, data: Dict, name: str):
        """
        Log a dictionary as JSON artifact.
        
        Args:
            data: Dictionary to log
            name: Name for the file
        """
        if not self.enabled:
            return
        
        try:
            import tempfile
            import json
            
            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode='w'
            ) as f:
                temp_path = f.name
                json.dump(data, f, indent=2, default=str)
            
            self.log_artifact(temp_path, artifact_path="data")
            os.unlink(temp_path)
            
            self.logger.info(f"Logged JSON: {name}")
            
        except Exception as e:
            self.logger.warning(f"Could not log JSON: {e}")
    
    def set_tags(self, tags: Dict[str, str]):
        """Set run tags."""
        if not self.enabled:
            return
        
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    def get_run_data(self, run_id: Optional[str] = None) -> Dict:
        """
        Get run data.
        
        Args:
            run_id: Run ID (uses current run if None)
        
        Returns:
            Dictionary with run data
        """
        if not self.enabled:
            return {}
        
        run_id = run_id or self.run_id
        
        if run_id is None:
            self.logger.warning("No run ID provided")
            return {}
        
        run = mlflow.get_run(run_id)
        
        return {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags
        }
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs
        
        Returns:
            DataFrame with comparison
        """
        if not self.enabled:
            return pd.DataFrame()
        
        runs = []
        for run_id in run_ids:
            run_data = self.get_run_data(run_id)
            runs.append(run_data)
        
        df = pd.DataFrame(runs)
        return df
    
    def get_best_run(self, metric: str = "rmse", lower_is_better: bool = True):
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric name to optimize
            lower_is_better: Whether lower values are better
        
        Returns:
            Best run data
        """
        if not self.enabled:
            return None
        
        experiment = mlflow.get_experiment(self.experiment_id)
        
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if lower_is_better else 'DESC'}"]
        )
        
        if len(runs) == 0:
            self.logger.warning("No runs found")
            return None
        
        best_run = runs.iloc[0]
        self.logger.info(
            f"Best run: {best_run['run_id']} with {metric}={best_run[f'metrics.{metric}']}"
        )
        
        return best_run


# Global tracker instance
_tracker: Optional[MLflowTracker] = None


def get_tracker() -> MLflowTracker:
    """Get global MLflow tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker()
    return _tracker


def track_experiment(
    params: Dict,
    metrics: Dict,
    model=None,
    model_type: str = "sklearn",
    artifacts: Optional[List[str]] = None,
    run_name: Optional[str] = None
):
    """
    Convenience function to track an experiment.
    
    Args:
        params: Parameters dictionary
        metrics: Metrics dictionary
        model: Optional model to log
        model_type: Type of model
        artifacts: List of artifact paths
        run_name: Name for the run
    """
    tracker = get_tracker()
    
    tracker.start_run(run_name=run_name)
    tracker.log_params(params)
    tracker.log_metrics(metrics)
    
    if model is not None:
        tracker.log_model(model, model_type=model_type)
    
    if artifacts:
        for artifact_path in artifacts:
            tracker.log_artifact(artifact_path)
    
    tracker.end_run()

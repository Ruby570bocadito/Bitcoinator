"""
Configuration management for Bitcoinator project.
Loads and validates configuration from YAML file.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """Path configuration."""
    raw_data: str = "data/raw/btcusd_1-min_data.csv"
    processed_data: str = "data/processed/"
    splits: str = "data/splits/"
    models: str = "models/"
    experiments: str = "experiments/"
    reports: str = "reports/"
    logs: str = "logs/"


@dataclass
class DataConfig:
    """Data processing configuration."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    target_column: str = "Close"
    date_column: str = "Timestamp"


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    technical_indicators: list = field(default_factory=lambda: [
        "RSI", "MACD", "Bollinger_Bands", "EMA_7", "EMA_21",
        "EMA_50", "EMA_200", "ATR", "VWAP"
    ])
    lag_features: list = field(default_factory=lambda: [1, 2, 7, 14, 30])
    temporal_features: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    random_seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    early_stopping_patience: int = 10
    n_jobs: int = -1


@dataclass
class ModelConfig:
    """Model configuration."""
    baseline: list = field(default_factory=lambda: ["Naive", "MovingAverage", "LinearRegression"])
    ml: list = field(default_factory=lambda: ["RandomForest", "XGBoost", "LightGBM"])
    deep_learning: list = field(default_factory=lambda: ["LSTM", "GRU"])


@dataclass
class MetricConfig:
    """Evaluation metrics configuration."""
    primary: str = "RMSE"
    secondary: list = field(default_factory=lambda: ["MAE", "MAPE", "R2", "DirectionalAccuracy"])


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    title: str = "Bitcoin Price Prediction Dashboard"
    port: int = 8501


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    position_size: float = 0.1  # 10% of capital per trade


@dataclass
class OptunaConfig:
    """Optuna hyperparameter tuning configuration."""
    n_trials: int = 100
    timeout: int = 3600  # 1 hour
    study_name: str = "bitcoinator_optimization"
    storage: str = "sqlite:///optuna.db"
    pruner: str = "median"  # median, percentile, success_halving


@dataclass
class MLflowConfig:
    """MLflow experiment tracking configuration."""
    tracking_uri: str = "mlruns"
    experiment_name: str = "bitcoinator"
    log_artifacts: bool = True
    log_models: bool = True


@dataclass
class Config:
    """Main configuration class."""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        config = cls()
        
        if "paths" in config_dict:
            config.paths = PathConfig(**config_dict["paths"])
        
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        
        if "features" in config_dict:
            config.features = FeatureConfig(**config_dict["features"])
        
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        
        if "models" in config_dict:
            config.models = ModelConfig(**config_dict["models"])
        
        if "metrics" in config_dict:
            config.metrics = MetricConfig(**config_dict["metrics"])
        
        if "dashboard" in config_dict:
            config.dashboard = DashboardConfig(**config_dict["dashboard"])
        
        # Load extended configs if available
        if "backtest" in config_dict:
            config.backtest = BacktestConfig(**config_dict["backtest"])
        
        if "optuna" in config_dict:
            config.optuna = OptunaConfig(**config_dict["optuna"])
        
        if "mlflow" in config_dict:
            config.mlflow = MLflowConfig(**config_dict["mlflow"])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "paths": self.paths.__dict__,
            "data": self.data.__dict__,
            "features": self.features.__dict__,
            "training": self.training.__dict__,
            "models": self.models.__dict__,
            "metrics": self.metrics.__dict__,
            "dashboard": self.dashboard.__dict__,
            "backtest": self.backtest.__dict__,
            "optuna": self.optuna.__dict__,
            "mlflow": self.mlflow.__dict__,
        }
    
    def save(self, config_path: str = "config.yaml"):
        """Save configuration to YAML file."""
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_config(config_path: str = "config.yaml") -> Config:
    """Get configuration from file or create default."""
    return Config.from_yaml(config_path)


# Global config instance
_config: Optional[Config] = None


def get_global_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config.from_yaml()
    return _config

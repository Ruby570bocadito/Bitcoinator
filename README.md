# Bitcoinator - Bitcoin Price Prediction ML Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-latest-red.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-latest-orange.svg)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-latest-purple.svg)](https://optuna.org/)

> Advanced Machine Learning platform for Bitcoin price prediction with backtesting, hyperparameter optimization, and comprehensive analytics.

## 🚀 Features

### Core Features
- **Multiple ML Models**: XGBoost, LightGBM, Random Forest
- **Hyperparameter Optimization**: Optuna for efficient hyperparameter search
- **Walk-Forward Validation**: Time-series cross-validation to prevent look-ahead bias
- **Backtesting Engine**: Full trading simulation with costs, slippage, and metrics
- **Experiment Tracking**: MLflow integration for tracking all experiments
- **Interactive Dashboard**: Streamlit-based analytics and visualization

### Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- EMA (7, 21, 50, 200)
- ATR (Average True Range)
- VWAP (Volume Weighted Average Price)

### Feature Engineering
- Lag features (1-60 days)
- Rolling statistics (mean, std)
- Temporal features (hour, day, month, quarter)
- Price-based features

### Trading Metrics
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Win Rate
- Profit Factor
- Expectancy

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
cd Bitcoinator

# Install dependencies
pip install -r requirements.txt

# Train a model with optimization
python train_optimized.py --model xgboost --optimize --n-trials 100

# Run the dashboard
streamlit run dashboard/app.py
```

## 🎯 Usage

### Training Models

#### Basic Training
```bash
# Train XGBoost model
python train_optimized.py --model xgboost

# Train LightGBM model
python train_optimized.py --model lightgbm

# Train Random Forest
python train_optimized.py --model random_forest
```

#### With Hyperparameter Optimization
```bash
# Optimize for 100 trials
python train_optimized.py --model xgboost --optimize --n-trials 100

# Optimize for 200 trials with 2 hour timeout
python train_optimized.py --model xgboost --optimize --n-trials 200
```

#### With Walk-Forward Validation
```bash
# Train with walk-forward validation (5 splits)
python train_optimized.py --model xgboost --walk-forward --n-wf-splits 5

# Combine optimization and walk-forward
python train_optimized.py --model xgboost --optimize --walk-forward
```

#### Command Line Options
```
--model          Model type: xgboost, lightgbm, random_forest
--optimize       Enable hyperparameter optimization
--n-trials       Number of optimization trials (default: 100)
--walk-forward   Enable walk-forward validation
--n-wf-splits    Number of walk-forward splits (default: 5)
--no-mlflow      Disable MLflow tracking
--no-optuna      Disable Optuna optimization
--config         Path to configuration file
--verbose        Enable verbose output
```

### Running the Dashboard

```bash
# Main dashboard
streamlit run dashboard/app.py

# Backtesting page
streamlit run dashboard/pages/2_Backtesting.py

# Model analysis page
streamlit run dashboard/pages/3_Model_Analysis.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_features.py -v
```

## 📁 Project Structure

```
Bitcoinator/
├── archive/              # Raw data storage
│   └── btcusd_1-min_data.csv
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── Makefile             # Useful commands
├── train_optimized.py   # Main training script
│
├── src/
│   ├── backtesting/     # Backtesting engine
│   │   └── backtester.py
│   ├── data/            # Data loading & processing
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   └── splitter.py
│   ├── evaluation/      # Metrics & evaluation
│   │   └── metrics.py
│   ├── features/        # Feature engineering
│   │   ├── technical.py
│   │   ├── temporal.py
│   │   └── lags.py
│   ├── models/          # ML model wrappers
│   │   └── xgboost_model.py
│   ├── training/        # Training utilities
│   │   ├── trainer.py
│   │   ├── tuner.py
│   │   └── optimizer.py  # Optuna optimization
│   ├── utils/           # Utilities
│   │   ├── config.py     # Configuration management
│   │   ├── logger.py     # Logging utilities
│   │   └── mlflow_tracker.py  # MLflow integration
│   └── validation/      # Validation strategies
│       └── walk_forward.py  # Walk-forward validation
│
├── dashboard/
│   ├── app.py           # Main dashboard
│   └── pages/
│       ├── 2_Backtesting.py    # Backtesting analysis
│       └── 3_Model_Analysis.py # Model diagnostics
│
├── models/              # Trained models
├── logs/                # Log files
├── tests/               # Unit tests
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_backtester.py
└── mlruns/              # MLflow experiments
```

## 📊 Configuration

Edit `config.yaml` to customize:

```yaml
# Data splits
data:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

# Feature engineering
features:
  lag_features: [1, 2, 7, 14, 30]
  technical_indicators:
    - RSI
    - MACD
    - Bollinger_Bands

# Backtesting
backtest:
  initial_capital: 10000.0
  commission: 0.001  # 0.1%
  slippage: 0.0005   # 0.05%
  position_size: 0.1 # 10% per trade

# Hyperparameter optimization
optuna:
  n_trials: 100
  timeout: 3600

# MLflow tracking
mlflow:
  experiment_name: "bitcoinator"
  log_models: true
```

## 📈 Dashboard Features

### Main Dashboard (`/`)
- Real-time price charts (candlestick)
- Model predictions vs actual
- Performance metrics (RMSE, MAE, MAPE)
- Error distribution
- Quick navigation

### Backtesting (`/2_Backtesting`)
- Equity curve visualization
- Drawdown analysis
- Trade-by-trade breakdown
- Trading metrics (Sharpe, Sortino, etc.)
- PnL distribution
- Export trade summary

### Model Analysis (`/3_Model_Analysis`)
- Feature importance rankings
- Correlation heatmaps
- Error analysis
- Partial dependence plots
- Model diagnostics

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src

# Generate HTML coverage report
pytest tests/ -v --cov=src --cov-report=html
```

## 🔬 Advanced Usage

### Custom Model Training

```python
from src.training.optimizer import HyperparameterOptimizer
from src.backtesting.backtester import Backtester
from src.validation.walk_forward import WalkForwardValidator

# Hyperparameter optimization
optimizer = HyperparameterOptimizer(model_type="xgboost")
best_params, study = optimizer.optimize(X_train, y_train, X_val, y_val)

# Walk-forward validation
validator = WalkForwardValidator()
for train_df, test_df in validator.split(data):
    # Train and evaluate on each split
    pass

# Backtesting
backtester = Backtester()
result = backtester.run(test_df, predictions)
print(result.metrics)
```

### MLflow Integration

All experiments are automatically tracked with MLflow:

```bash
# View MLflow UI
mlflow ui

# Access at http://localhost:5000
```

### Optuna Visualization

```python
from src.training.optimizer import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(model_type="xgboost")
# ... run optimization ...

# Plot optimization history
fig = optimizer.plot_optimization_history()
fig.show()

# Plot parameter importances
fig = optimizer.plot_param_importances()
fig.show()
```

## 📋 Requirements

See `requirements.txt` for full list:

**Core:**
- pandas, numpy
- scikit-learn
- xgboost, lightgbm

**Optimization:**
- optuna

**Tracking:**
- mlflow

**Visualization:**
- plotly, matplotlib, seaborn
- streamlit

**Testing:**
- pytest, pytest-cov

## 📝 License

This project is for educational and research purposes.

## ⚠️ Disclaimer

**This is NOT financial advice.** The models and backtests are for educational purposes only. Past performance does not guarantee future results. Always do your own research before making investment decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

Built with ❤️ for the Bitcoin community

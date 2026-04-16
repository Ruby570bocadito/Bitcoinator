# Bitcoinator Makefile
# Useful commands for development, training, and deployment

# =============================================================================
# Configuration
# =============================================================================

PYTHON = python
PIP = pip
STREAMLIT = streamlit
PYTEST = pytest
MLFLOW = mlflow

# =============================================================================
# Setup & Installation
# =============================================================================

.PHONY: install install-dev requirements clean

install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Installation complete!"

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy
	@echo "Development installation complete!"

requirements:
	@echo "Updating requirements.txt..."
	$(PIP) freeze > requirements.txt

clean:
	@echo "Cleaning project..."
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/*/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf logs/*.log
	rm -rf mlruns/*
	@echo "Clean complete!"

# =============================================================================
# Data
# =============================================================================

.PHONY: download-data process-data

download-data:
	@echo "Downloading data..."
	# Add your data download command here
	@echo "Data download complete!"

process-data:
	@echo "Processing data..."
	$(PYTHON) -c "from src.data.loader import load_data; print('Data processed!')"
	@echo "Data processing complete!"

# =============================================================================
# Training
# =============================================================================

.PHONY: train train-xgboost train-lightgbm train-rf train-optimized train-with-optuna

train:
	@echo "Training model..."
	$(PYTHON) train_optimized.py --model xgboost

train-xgboost:
	@echo "Training XGBoost model..."
	$(PYTHON) train_optimized.py --model xgboost --verbose

train-lightgbm:
	@echo "Training LightGBM model..."
	$(PYTHON) train_optimized.py --model lightgbm --verbose

train-rf:
	@echo "Training Random Forest..."
	$(PYTHON) train_optimized.py --model random_forest --verbose

train-optimized:
	@echo "Training with optimization..."
	$(PYTHON) train_optimized.py --model xgboost --optimize --n-trials 100 --verbose

train-with-optuna:
	@echo "Training with Optuna hyperparameter optimization..."
	$(PYTHON) train_optimized.py --model xgboost --optimize --n-trials $(or $(TRIALS),100) --verbose

train-walk-forward:
	@echo "Training with walk-forward validation..."
	$(PYTHON) train_optimized.py --model xgboost --walk-forward --n-wf-splits $(or $(SPLITS),5) --verbose

train-full:
	@echo "Full training pipeline with optimization and walk-forward..."
	$(PYTHON) train_optimized.py --model xgboost --optimize --walk-forward --verbose

# =============================================================================
# Dashboard
# =============================================================================

.PHONY: dashboard dashboard-main dashboard-backtest dashboard-analysis

dashboard:
	@echo "Starting main dashboard..."
	$(STREAMLIT) run dashboard/app.py

dashboard-main:
	@echo "Starting main dashboard..."
	$(STREAMLIT) run dashboard/app.py

dashboard-backtest:
	@echo "Starting backtesting dashboard..."
	$(STREAMLIT) run dashboard/pages/2_Backtesting.py

dashboard-analysis:
	@echo "Starting model analysis dashboard..."
	$(STREAMLIT) run dashboard/pages/3_Model_Analysis.py

dashboard-all:
	@echo "Starting dashboard on port 8501..."
	$(STREAMLIT) run dashboard/app.py --server.port 8501

# =============================================================================
# Testing
# =============================================================================

.PHONY: test test-verbose test-cov test-single

test:
	@echo "Running tests..."
	$(PYTEST) tests/ -v

test-verbose:
	@echo "Running tests with verbose output..."
	$(PYTEST) tests/ -vv -s

test-cov:
	@echo "Running tests with coverage..."
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

test-cov-html:
	@echo "Running tests with HTML coverage report..."
	$(PYTEST) tests/ -v --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

test-single:
	@echo "Running single test file..."
	$(PYTEST) tests/$(or $(FILE),test_features.py) -v

# =============================================================================
# Linting & Formatting
# =============================================================================

.PHONY: lint format check-types

lint:
	@echo "Running linter..."
	flake8 src/ tests/ --max-line-length=100

format:
	@echo "Formatting code..."
	black src/ tests/ --line-length 100

check-types:
	@echo "Checking types..."
	mypy src/ --ignore-missing-imports

# =============================================================================
# MLflow
# =============================================================================

.PHONY: mlflow-ui mlflow-clean

mlflow-ui:
	@echo "Starting MLflow UI..."
	$(MLFLOW) ui --port 5000

mlflow-clean:
	@echo "Cleaning MLflow data..."
	rm -rf mlruns/*

# =============================================================================
# Optuna
# =============================================================================

.PHONY: optuna-clean

optuna-clean:
	@echo "Cleaning Optuna database..."
	rm -f optuna.db

# =============================================================================
# Logs
# =============================================================================

.PHONY: logs logs-clean

logs:
	@echo "Recent logs:"
	tail -f logs/*.log 2>/dev/null || echo "No log files found"

logs-clean:
	@echo "Cleaning logs..."
	rm -f logs/*.log

# =============================================================================
# All-in-One Commands
# =============================================================================

.PHONY: all setup run

setup: install-dev clean
	@echo "Setup complete!"

run: train dashboard
	@echo "Training and dashboard started!"

all: setup test train dashboard
	@echo "All tasks complete!"

# =============================================================================
# Help
# =============================================================================

.PHONY: help

help:
	@echo "Bitcoinator Makefile Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          - Install dependencies"
	@echo "  make install-dev      - Install dev dependencies"
	@echo "  make clean            - Clean project"
	@echo ""
	@echo "Training:"
	@echo "  make train            - Train default model"
	@echo "  make train-xgboost    - Train XGBoost"
	@echo "  make train-lightgbm   - Train LightGBM"
	@echo "  make train-rf         - Train Random Forest"
	@echo "  make train-optimized  - Train with optimization"
	@echo "  make train-with-optuna TRIALS=200 - Train with N trials"
	@echo "  make train-walk-forward SPLITS=5 - Train with WF validation"
	@echo "  make train-full       - Full pipeline (optimize + WF)"
	@echo ""
	@echo "Dashboard:"
	@echo "  make dashboard        - Start main dashboard"
	@echo "  make dashboard-backtest - Start backtesting dashboard"
	@echo "  make dashboard-analysis - Start model analysis dashboard"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests"
	@echo "  make test-cov         - Run tests with coverage"
	@echo "  make test-cov-html    - Generate HTML coverage report"
	@echo "  make test-single FILE=test_metrics.py - Run single test"
	@echo ""
	@echo "MLflow:"
	@echo "  make mlflow-ui        - Start MLflow UI"
	@echo "  make mlflow-clean     - Clean MLflow data"
	@echo ""
	@echo "Other:"
	@echo "  make lint             - Run linter"
	@echo "  make format           - Format code"
	@echo "  make logs             - View logs"
	@echo "  make help             - Show this help"

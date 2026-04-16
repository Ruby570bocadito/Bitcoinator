"""
Evaluation Metrics for Bitcoinator.
Includes regression metrics, classification metrics, and trading metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from src.utils.logger import setup_logger

logger = setup_logger("metrics")


# =============================================================================
# Regression Metrics
# =============================================================================

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def calculate_max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum error."""
    return np.max(np.abs(y_true - y_pred))


# =============================================================================
# Directional/Cla ssification Metrics
# =============================================================================

def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of correct directional predictions."""
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return np.mean(true_direction == pred_direction) * 100


def calculate_directional_accuracy_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Directional accuracy with threshold.
    Only considers predictions where change is above threshold.
    """
    true_change = np.diff(y_true)
    pred_change = np.diff(y_pred)
    
    mask = np.abs(true_change) > threshold
    if np.sum(mask) == 0:
        return np.nan
    
    true_direction = true_change[mask] > 0
    pred_direction = pred_change[mask] > 0
    
    return np.mean(true_direction == pred_direction) * 100


def calculate_hit_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hit ratio: percentage of predictions within a certain range."""
    mean_true = np.mean(y_true)
    threshold = 0.02 * mean_true  # 2% of mean price
    
    hits = np.abs(y_true - y_pred) < threshold
    return np.mean(hits) * 100


# =============================================================================
# Trading Metrics
# =============================================================================

def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    annualization: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        annualization: Number of periods per year
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / annualization)
    return np.mean(excess_returns) / np.std(returns) * np.sqrt(annualization)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    annualization: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        annualization: Number of periods per year
    
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / annualization)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return np.mean(excess_returns) * np.sqrt(annualization) / 0.0001
    
    downside_std = np.std(downside_returns)
    return np.mean(excess_returns) / downside_std * np.sqrt(annualization)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        equity_curve: Array of equity values
    
    Returns:
        Maximum drawdown as percentage
    """
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    
    if len(drawdown) == 0:
        return 0.0
    
    return np.min(drawdown) * 100


def calculate_calmar_ratio(
    returns: np.ndarray,
    annualization: int = 252
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Args:
        returns: Array of returns
        annualization: Number of periods per year
    
    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate equity curve from returns
    equity_curve = np.cumprod(1 + returns)
    
    max_dd = calculate_max_drawdown(equity_curve)
    
    if max_dd == 0 or np.isnan(max_dd):
        return 0.0
    
    annual_return = np.mean(returns) * annualization
    
    return annual_return / abs(max_dd)


def calculate_profit_factor(
    profits: np.ndarray,
    losses: np.ndarray
) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        profits: Array of profitable trades
        losses: Array of losing trades (as positive values)
    
    Returns:
        Profit factor
    """
    gross_profit = np.sum(profits) if len(profits) > 0 else 0
    gross_loss = np.sum(losses) if len(losses) > 0 else 0
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_win_rate(trades_pnl: np.ndarray) -> float:
    """
    Calculate win rate.
    
    Args:
        trades_pnl: Array of trade PnL values
    
    Returns:
        Win rate as percentage
    """
    if len(trades_pnl) == 0:
        return 0.0
    
    wins = np.sum(trades_pnl > 0)
    return (wins / len(trades_pnl)) * 100


def calculate_expectancy(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate trading expectancy.
    
    Args:
        win_rate: Win rate (0-1)
        avg_win: Average winning trade
        avg_loss: Average losing trade (as positive value)
    
    Returns:
        Expectancy per trade
    """
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)


def calculate_recovery_factor(
    total_profit: float,
    max_drawdown: float
) -> float:
    """
    Calculate recovery factor.
    
    Args:
        total_profit: Total net profit
        max_drawdown: Maximum drawdown (as positive value)
    
    Returns:
        Recovery factor
    """
    if max_drawdown == 0:
        return 0.0
    
    return total_profit / max_drawdown


def calculate_ulcer_index(equity_curve: np.ndarray) -> float:
    """
    Calculate Ulcer Index (measure of downside risk).
    
    Args:
        equity_curve: Array of equity values
    
    Returns:
        Ulcer Index
    """
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak * 100
    
    if len(drawdown) == 0:
        return 0.0
    
    return np.sqrt(np.mean(drawdown ** 2))


def calculate_serenity_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray
) -> float:
    """
    Calculate Serenity Ratio (Sharpe / Ulcer Index).
    
    Args:
        returns: Array of returns
        equity_curve: Array of equity values
    
    Returns:
        Serenity Ratio
    """
    sharpe = calculate_sharpe_ratio(returns)
    ulcer = calculate_ulcer_index(equity_curve)
    
    if ulcer == 0:
        return sharpe
    
    return sharpe / ulcer


# =============================================================================
# Comprehensive Evaluation
# =============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_trading: bool = False,
    equity_curve: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        include_trading: Whether to include trading metrics
        equity_curve: Optional equity curve for trading metrics
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        # Regression metrics
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'SMAPE': calculate_smape(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred),
        'MSE': calculate_mse(y_true, y_pred),
        'Max_Error': calculate_max_error(y_true, y_pred),
        
        # Directional metrics
        'Directional_Accuracy': calculate_directional_accuracy(y_true, y_pred),
        'Hit_Ratio': calculate_hit_ratio(y_true, y_pred),
    }
    
    if include_trading and equity_curve is not None:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics.update({
            'Sharpe_Ratio': calculate_sharpe_ratio(returns),
            'Sortino_Ratio': calculate_sortino_ratio(returns),
            'Max_Drawdown': calculate_max_drawdown(equity_curve),
            'Calmar_Ratio': calculate_calmar_ratio(returns),
            'Ulcer_Index': calculate_ulcer_index(equity_curve),
        })
    
    return metrics


def evaluate_trades(trades_pnl: np.ndarray) -> Dict[str, float]:
    """
    Evaluate trading performance from trade PnL.
    
    Args:
        trades_pnl: Array of trade PnL values
    
    Returns:
        Dictionary of trading metrics
    """
    if len(trades_pnl) == 0:
        return {}
    
    wins = trades_pnl[trades_pnl > 0]
    losses = np.abs(trades_pnl[trades_pnl < 0])
    
    metrics = {
        'Total_Trades': len(trades_pnl),
        'Winning_Trades': len(wins),
        'Losing_Trades': len(losses),
        'Win_Rate': calculate_win_rate(trades_pnl),
        'Profit_Factor': calculate_profit_factor(wins, losses),
        'Avg_Win': np.mean(wins) if len(wins) > 0 else 0,
        'Avg_Loss': np.mean(losses) if len(losses) > 0 else 0,
        'Win_Loss_Ratio': (np.mean(wins) / np.mean(losses)) if len(losses) > 0 else 0,
        'Total_PnL': np.sum(trades_pnl),
        'Expectancy': calculate_expectancy(
            calculate_win_rate(trades_pnl) / 100,
            np.mean(wins) if len(wins) > 0 else 0,
            np.mean(losses) if len(losses) > 0 else 0
        ),
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{model_name} Performance")
    print(f"{'='*60}")
    
    # Group metrics by type
    regression_metrics = ['RMSE', 'MAE', 'MAPE', 'SMAPE', 'R2', 'MSE', 'Max_Error']
    directional_metrics = ['Directional_Accuracy', 'Hit_Ratio']
    trading_metrics = ['Sharpe_Ratio', 'Sortino_Ratio', 'Max_Drawdown', 
                       'Calmar_Ratio', 'Ulcer_Index']
    
    print("\n📊 Regression Metrics:")
    for metric in regression_metrics:
        if metric in metrics:
            value = metrics[metric]
            if metric in ['MAPE', 'SMAPE']:
                print(f"  {metric:20s}: {value:.2f}%")
            elif metric == 'R2':
                print(f"  {metric:20s}: {value:.4f}")
            elif metric in ['RMSE', 'MAE', 'MSE', 'Max_Error']:
                print(f"  {metric:20s}: ${value:,.2f}")
            else:
                print(f"  {metric:20s}: {value:.4f}")
    
    print("\n🎯 Directional Metrics:")
    for metric in directional_metrics:
        if metric in metrics:
            print(f"  {metric:20s}: {metrics[metric]:.2f}%")
    
    print("\n💰 Trading Metrics:")
    for metric in trading_metrics:
        if metric in metrics:
            if metric == 'Max_Drawdown':
                print(f"  {metric:20s}: {metrics[metric]:.2f}%")
            else:
                print(f"  {metric:20s}: {metrics[metric]:.2f}")
    
    print(f"{'='*60}\n")


def compare_models(
    results: Dict[str, Dict[str, float]],
    primary_metric: str = "RMSE"
) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
        primary_metric: Metric to sort by
    
    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(results).T
    
    if primary_metric in df.columns:
        # Sort by primary metric (lower is better for RMSE, MAE, etc.)
        if primary_metric in ['R2', 'Directional_Accuracy', 'Hit_Ratio', 
                              'Sharpe_Ratio', 'Sortino_Ratio', 'Calmar_Ratio']:
            df = df.sort_values(primary_metric, ascending=False)
        else:
            df = df.sort_values(primary_metric, ascending=True)
    
    return df

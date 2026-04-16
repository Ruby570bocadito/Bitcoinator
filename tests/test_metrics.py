"""
Unit Tests for Evaluation Metrics Module.
Tests regression, directional, and trading metrics.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_smape,
    calculate_r2,
    calculate_directional_accuracy,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_profit_factor,
    evaluate_model,
    evaluate_trades
)


class TestRegressionMetrics:
    """Tests for regression metrics."""
    
    def test_rmse_perfect_prediction(self):
        """RMSE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        rmse = calculate_rmse(y_true, y_pred)
        assert rmse == 0.0
    
    def test_rmse_positive(self):
        """RMSE should always be positive."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.2])
        
        rmse = calculate_rmse(y_true, y_pred)
        assert rmse > 0
    
    def test_mae_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        mae = calculate_mae(y_true, y_pred)
        assert mae == 0.0
    
    def test_mae_calculation(self):
        """MAE should be calculated correctly."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        
        mae = calculate_mae(y_true, y_pred)
        expected = (0 + 0 + 1) / 3
        assert np.isclose(mae, expected)
    
    def test_mape_perfect_prediction(self):
        """MAPE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        mape = calculate_mape(y_true, y_pred)
        assert mape == 0.0
    
    def test_mape_with_zeros(self):
        """MAPE should handle zeros in y_true."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 2, 3])
        
        mape = calculate_mape(y_true, y_pred)
        assert np.isnan(mape)
    
    def test_r2_perfect_prediction(self):
        """R2 should be 1 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        r2 = calculate_r2(y_true, y_pred)
        assert np.isclose(r2, 1.0)
    
    def test_r2_constant_prediction(self):
        """R2 should be 0 for constant predictions (mean)."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])  # Mean of y_true
        
        r2 = calculate_r2(y_true, y_pred)
        assert np.isclose(r2, 0.0)
    
    def test_r2_can_be_negative(self):
        """R2 can be negative for worse than mean predictions."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([10, 10, 10])
        
        r2 = calculate_r2(y_true, y_pred)
        assert r2 < 0


class TestDirectionalMetrics:
    """Tests for directional metrics."""
    
    def test_directional_accuracy_perfect(self):
        """Directional accuracy should be 100 for perfect direction predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        acc = calculate_directional_accuracy(y_true, y_pred)
        assert acc == 100.0
    
    def test_directional_accuracy_all_wrong(self):
        """Directional accuracy should be 0 for all wrong predictions."""
        y_true = np.array([1, 2, 3, 4, 5])  # Always increasing
        y_pred = np.array([5, 4, 3, 2, 1])  # Always decreasing
        
        acc = calculate_directional_accuracy(y_true, y_pred)
        assert acc == 0.0
    
    def test_directional_accuracy_50_percent(self):
        """Directional accuracy should be 50 for random predictions."""
        np.random.seed(42)
        y_true = np.random.randn(100).cumsum()
        y_pred = np.random.randn(100).cumsum()
        
        acc = calculate_directional_accuracy(y_true, y_pred)
        # Should be around 50% for random
        assert 30 < acc < 70


class TestTradingMetrics:
    """Tests for trading metrics."""
    
    def test_sharpe_ratio_constant_returns(self):
        """Sharpe ratio should be 0 for constant returns."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0
    
    def test_sharpe_ratio_positive_returns(self):
        """Sharpe ratio should be positive for positive returns."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01 + 0.001  # Positive mean
        
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe > 0
    
    def test_sortino_ratio_positive(self):
        """Sortino ratio should be calculated correctly."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        
        sortino = calculate_sortino_ratio(returns)
        assert not np.isnan(sortino)
    
    def test_max_drawdown_positive(self):
        """Max drawdown should be negative or zero."""
        equity_curve = np.array([100, 110, 120, 115, 130, 125, 140])
        
        max_dd = calculate_max_drawdown(equity_curve)
        assert max_dd <= 0
    
    def test_max_drawdown_calculation(self):
        """Max drawdown should be calculated correctly."""
        equity_curve = np.array([100, 100, 100])  # No drawdown
        
        max_dd = calculate_max_drawdown(equity_curve)
        assert max_dd == 0.0
    
    def test_max_drawdown_significant(self):
        """Max drawdown should detect significant drops."""
        equity_curve = np.array([100, 50, 25, 10])  # 90% drawdown
        
        max_dd = calculate_max_drawdown(equity_curve)
        assert max_dd < -80  # More than 80% drawdown
    
    def test_calmar_ratio(self):
        """Calmar ratio should be calculated correctly."""
        returns = np.array([0.01, 0.02, -0.01, 0.03])
        
        calmar = calculate_calmar_ratio(returns)
        assert not np.isnan(calmar)
    
    def test_win_rate_calculation(self):
        """Win rate should be calculated correctly."""
        trades_pnl = np.array([100, -50, 200, -100, 300])  # 3 wins, 2 losses
        
        win_rate = calculate_win_rate(trades_pnl)
        assert np.isclose(win_rate, 60.0)
    
    def test_win_rate_all_wins(self):
        """Win rate should be 100 for all winning trades."""
        trades_pnl = np.array([100, 200, 300])
        
        win_rate = calculate_win_rate(trades_pnl)
        assert win_rate == 100.0
    
    def test_win_rate_all_losses(self):
        """Win rate should be 0 for all losing trades."""
        trades_pnl = np.array([-100, -200, -300])
        
        win_rate = calculate_win_rate(trades_pnl)
        assert win_rate == 0.0
    
    def test_profit_factor_calculation(self):
        """Profit factor should be calculated correctly."""
        profits = np.array([100, 200, 300])
        losses = np.array([50, 100])
        
        pf = calculate_profit_factor(profits, losses)
        assert np.isclose(pf, 4.0)  # 600 / 150 = 4
    
    def test_profit_factor_no_losses(self):
        """Profit factor should be inf for no losses."""
        profits = np.array([100, 200, 300])
        losses = np.array([])
        
        pf = calculate_profit_factor(profits, losses)
        assert np.isinf(pf)


class TestEvaluateModel:
    """Tests for comprehensive model evaluation."""
    
    def test_evaluate_model_returns_dict(self):
        """evaluate_model should return a dictionary."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.2])
        
        metrics = evaluate_model(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R2' in metrics
    
    def test_evaluate_model_with_trading(self):
        """evaluate_model should include trading metrics when requested."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.2])
        equity_curve = np.array([100, 110, 120, 115, 130])
        
        metrics = evaluate_model(
            y_true, y_pred,
            include_trading=True,
            equity_curve=equity_curve
        )
        
        assert 'Sharpe_Ratio' in metrics
        assert 'Max_Drawdown' in metrics
    
    def test_evaluate_model_values_reasonable(self):
        """Metrics should have reasonable values."""
        np.random.seed(42)
        y_true = np.random.randn(100).cumsum() + 100
        y_pred = y_true + np.random.randn(100) * 0.5
        
        metrics = evaluate_model(y_true, y_pred)
        
        assert metrics['RMSE'] > 0
        assert metrics['MAE'] > 0
        assert metrics['R2'] <= 1.0
        assert 0 <= metrics['Directional_Accuracy'] <= 100


class TestEvaluateTrades:
    """Tests for trade evaluation."""
    
    def test_evaluate_trades_returns_dict(self):
        """evaluate_trades should return a dictionary."""
        trades_pnl = np.array([100, -50, 200, -100, 300])
        
        metrics = evaluate_trades(trades_pnl)
        
        assert isinstance(metrics, dict)
        assert 'Total_Trades' in metrics
        assert 'Win_Rate' in metrics
        assert 'Profit_Factor' in metrics
    
    def test_evaluate_trades_empty(self):
        """evaluate_trades should handle empty input."""
        trades_pnl = np.array([])
        
        metrics = evaluate_trades(trades_pnl)
        
        assert metrics == {}
    
    def test_evaluate_trades_all_wins(self):
        """evaluate_trades should handle all winning trades."""
        trades_pnl = np.array([100, 200, 300])
        
        metrics = evaluate_trades(trades_pnl)
        
        assert metrics['Win_Rate'] == 100.0
        assert metrics['Losing_Trades'] == 0


# Run with: pytest tests/test_metrics.py -v

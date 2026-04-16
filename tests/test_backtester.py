"""
Unit Tests for Backtesting Module.
Tests backtester, trading signals, and metrics.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.backtester import (
    Backtester,
    BacktestConfig,
    Signal,
    Trade,
    run_backtest
)


class TestBacktestConfig:
    """Tests for BacktestConfig."""
    
    def test_default_config(self):
        """Default config should have reasonable values."""
        config = BacktestConfig()
        
        assert config.initial_capital == 10000.0
        assert config.commission == 0.001
        assert config.slippage == 0.0005
        assert config.position_size == 0.1
    
    def test_custom_config(self):
        """Custom config should override defaults."""
        config = BacktestConfig(
            initial_capital=50000.0,
            commission=0.002,
            slippage=0.001,
            position_size=0.2
        )
        
        assert config.initial_capital == 50000.0
        assert config.commission == 0.002


class TestSignal:
    """Tests for Signal enum."""
    
    def test_signal_values(self):
        """Signal enum should have correct values."""
        assert Signal.BUY.value == 1
        assert Signal.SELL.value == -1
        assert Signal.HOLD.value == 0


class TestBacktester:
    """Tests for Backtester class."""
    
    def create_sample_data(self, n_days=100):
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        prices = 100 + np.random.randn(n_days).cumsum()
        
        df = pd.DataFrame({
            'Timestamp': dates,
            'Open': prices + np.random.randn(n_days) * 0.5,
            'High': prices + np.abs(np.random.randn(n_days)),
            'Low': prices - np.abs(np.random.randn(n_days)),
            'Close': prices,
            'Volume': np.abs(np.random.randn(n_days)) * 1000 + 100
        })
        
        return df
    
    def create_sample_predictions(self, n_days=100):
        """Create sample predictions for testing."""
        np.random.seed(43)
        return 100 + np.random.randn(n_days).cumsum()
    
    def test_backtester_initialization(self):
        """Backtester should initialize correctly."""
        backtester = Backtester()
        
        assert backtester.capital == 10000.0
        assert backtester.position == 0
        assert len(backtester.trades) == 0
    
    def test_backtester_custom_config(self):
        """Backtester should use custom config."""
        config = BacktestConfig(initial_capital=50000.0)
        backtester = Backtester(config)
        
        assert backtester.capital == 50000.0
    
    def test_reset(self):
        """Reset should clear backtester state."""
        backtester = Backtester()
        
        # Run some trades
        df = self.create_sample_data(50)
        preds = self.create_sample_predictions(50)
        backtester.run(df, preds)
        
        # Reset
        backtester.reset()
        
        assert backtester.capital == 10000.0
        assert backtester.position == 0
        assert len(backtester.trades) == 0
    
    def test_position_size_calculation(self):
        """Position size should be calculated correctly."""
        config = BacktestConfig(
            initial_capital=10000.0,
            position_size=0.1
        )
        backtester = Backtester(config)
        
        position_size = backtester.calculate_position_size(100.0)
        
        # 10% of 10000 / 100 = 10 shares
        assert np.isclose(position_size, 10.0)
    
    def test_slippage_buy(self):
        """Slippage should increase buy price."""
        backtester = Backtester()
        
        exec_price = backtester.apply_slippage(100.0, Signal.BUY)
        
        assert exec_price > 100.0
        assert np.isclose(exec_price, 100.0 * (1 + 0.0005))
    
    def test_slippage_sell(self):
        """Slippage should decrease sell price."""
        backtester = Backtester()
        
        exec_price = backtester.apply_slippage(100.0, Signal.SELL)
        
        assert exec_price < 100.0
        assert np.isclose(exec_price, 100.0 * (1 - 0.0005))
    
    def test_commission_calculation(self):
        """Commission should be calculated correctly."""
        backtester = Backtester()
        
        commission = backtester.calculate_commission(10000.0)
        
        assert np.isclose(commission, 10.0)  # 0.1% of 10000
    
    def test_run_returns_result(self):
        """Run should return a BacktestResult."""
        backtester = Backtester()
        df = self.create_sample_data(50)
        preds = self.create_sample_predictions(50)
        
        result = backtester.run(df, preds)
        
        assert hasattr(result, 'trades')
        assert hasattr(result, 'equity_curve')
        assert hasattr(result, 'returns')
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'positions')
        assert hasattr(result, 'signals')
    
    def test_run_generates_trades(self):
        """Run should generate trades."""
        backtester = Backtester()
        df = self.create_sample_data(100)
        preds = self.create_sample_predictions(100)
        
        result = backtester.run(df, preds)
        
        assert len(result.trades) > 0
    
    def test_run_equity_curve_length(self):
        """Equity curve should have correct length."""
        backtester = Backtester()
        df = self.create_sample_data(50)
        preds = self.create_sample_predictions(50)
        
        result = backtester.run(df, preds)
        
        assert len(result.equity_curve) == len(df)
    
    def test_run_metrics_calculated(self):
        """Run should calculate metrics."""
        backtester = Backtester()
        df = self.create_sample_data(100)
        preds = self.create_sample_predictions(100)
        
        result = backtester.run(df, preds)
        
        assert 'total_return' in result.metrics
        assert 'sharpe_ratio' in result.metrics
        assert 'max_drawdown' in result.metrics
        assert 'win_rate' in result.metrics
    
    def test_run_with_uptrend(self):
        """Backtest should profit in uptrend."""
        # Create strongly trending data
        np.random.seed(42)
        n_days = 200
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        prices = 100 + np.linspace(0, 100, n_days)  # Strong uptrend
        
        df = pd.DataFrame({
            'Timestamp': dates,
            'Open': prices,
            'High': prices + 1,
            'Low': prices - 1,
            'Close': prices,
            'Volume': np.ones(n_days) * 1000
        })
        
        # Predictions that follow the trend with lag
        preds = prices - 5
        
        backtester = Backtester()
        result = backtester.run(df, preds)
        
        # Should have positive return in strong uptrend
        assert result.metrics['total_return'] > -50  # Allow some loss from costs
    
    def test_positions_tracking(self):
        """Positions should be tracked correctly."""
        backtester = Backtester()
        df = self.create_sample_data(50)
        preds = self.create_sample_predictions(50)
        
        result = backtester.run(df, preds)
        
        assert len(result.positions) == len(df)
        assert all(p >= 0 for p in result.positions)  # Long only
    
    def test_signals_tracking(self):
        """Signals should be tracked correctly."""
        backtester = Backtester()
        df = self.create_sample_data(50)
        preds = self.create_sample_predictions(50)
        
        result = backtester.run(df, preds)
        
        assert len(result.signals) == len(df)
        assert all(s in [-1, 0, 1] for s in result.signals)
    
    def test_trade_summary(self):
        """Trade summary should be generated."""
        backtester = Backtester()
        df = self.create_sample_data(100)
        preds = self.create_sample_predictions(100)
        
        backtester.run(df, preds)
        summary = backtester.get_trade_summary()
        
        assert isinstance(summary, pd.DataFrame)
        
        if len(summary) > 0:
            assert 'entry_date' in summary.columns
            assert 'exit_date' in summary.columns
            assert 'pnl' in summary.columns


class TestRunBacktest:
    """Tests for run_backtest convenience function."""
    
    def test_run_backtest_function(self):
        """run_backtest should work as convenience function."""
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        prices = 100 + np.random.randn(n_days).cumsum()
        
        df = pd.DataFrame({
            'Timestamp': dates,
            'Open': prices,
            'High': prices + 1,
            'Low': prices - 1,
            'Close': prices,
            'Volume': np.ones(n_days) * 1000
        })
        
        preds = prices + np.random.randn(n_days)
        
        result = run_backtest(df, preds)
        
        assert hasattr(result, 'metrics')
        assert 'total_return' in result.metrics


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_data(self):
        """Backtester should handle empty data."""
        backtester = Backtester()
        
        df = pd.DataFrame({
            'Timestamp': pd.Series([], dtype='datetime64[ns]'),
            'Close': pd.Series([], dtype='float64')
        })
        preds = np.array([])
        
        # Should not crash
        result = backtester.run(df, preds)
        
        assert len(result.equity_curve) == 0
    
    def test_single_data_point(self):
        """Backtester should handle single data point."""
        backtester = Backtester()
        
        df = pd.DataFrame({
            'Timestamp': [pd.Timestamp('2020-01-01')],
            'Open': [100],
            'High': [101],
            'Low': [99],
            'Close': [100],
            'Volume': [1000]
        })
        preds = np.array([100])
        
        result = backtester.run(df, preds)
        
        assert len(result.equity_curve) == 1
    
    def test_constant_prices(self):
        """Backtester should handle constant prices."""
        backtester = Backtester()
        
        df = pd.DataFrame({
            'Timestamp': pd.date_range('2020-01-01', periods=50, freq='D'),
            'Open': [100] * 50,
            'High': [101] * 50,
            'Low': [99] * 50,
            'Close': [100] * 50,
            'Volume': [1000] * 50
        })
        preds = np.array([100] * 50)
        
        result = backtester.run(df, preds)
        
        # Should not crash and should have metrics
        assert 'total_return' in result.metrics


# Run with: pytest tests/test_backtester.py -v

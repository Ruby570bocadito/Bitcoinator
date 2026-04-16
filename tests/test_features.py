"""
Unit Tests for Feature Engineering Module.
Tests technical indicators, temporal features, and lag features.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.technical import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_atr,
    calculate_vwap,
    add_technical_indicators
)
from src.features.temporal import add_temporal_features
from src.features.lags import add_lag_features, add_rolling_features


class TestRSI:
    """Tests for RSI calculation."""
    
    def test_rsi_range(self):
        """RSI should be between 0 and 100."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        rsi = calculate_rsi(prices, 14)
        
        # Ignore NaN values at the beginning
        valid_rsi = rsi.dropna()
        
        assert (valid_rsi >= 0).all(), "RSI should be >= 0"
        assert (valid_rsi <= 100).all(), "RSI should be <= 100"
    
    def test_rsi_constant_prices(self):
        """RSI should be 50 for constant prices."""
        prices = pd.Series([100] * 50)
        rsi = calculate_rsi(prices, 14)
        
        # After warmup period, RSI should be around 50
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert np.allclose(valid_rsi.iloc[-1], 50, atol=1)
    
    def test_rsi_length(self):
        """RSI should have same length as input."""
        prices = pd.Series(np.random.randn(100))
        rsi = calculate_rsi(prices, 14)
        
        assert len(rsi) == len(prices)


class TestMACD:
    """Tests for MACD calculation."""
    
    def test_macd_length(self):
        """MACD components should have same length as input."""
        prices = pd.Series(np.random.randn(100))
        macd, signal, hist = calculate_macd(prices)
        
        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(hist) == len(prices)
    
    def test_macd_relationship(self):
        """MACD histogram should be MACD - Signal."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        macd, signal, hist = calculate_macd(prices)
        
        # Check relationship where values are not NaN
        mask = ~(macd.isna() | signal.isna() | hist.isna())
        if mask.sum() > 0:
            calculated_hist = macd[mask] - signal[mask]
            assert np.allclose(calculated_hist, hist[mask], rtol=1e-5)


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""
    
    def test_bands_ordering(self):
        """Upper band should be >= middle >= lower band."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        upper, middle, lower = calculate_bollinger_bands(prices)
        
        # Check where values are not NaN
        mask = ~(upper.isna() | middle.isna() | lower.isna())
        if mask.sum() > 0:
            assert (upper[mask] >= middle[mask]).all()
            assert (middle[mask] >= lower[mask]).all()
    
    def test_bands_length(self):
        """Bollinger bands should have same length as input."""
        prices = pd.Series(np.random.randn(100))
        upper, middle, lower = calculate_bollinger_bands(prices)
        
        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)


class TestEMA:
    """Tests for EMA calculation."""
    
    def test_ema_smoothness(self):
        """EMA should be smoother than raw prices."""
        np.random.seed(42)
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        ema = calculate_ema(prices, 20)
        
        # EMA should have lower variance in differences
        price_diff_std = prices.diff().std()
        ema_diff_std = ema.diff().std()
        
        assert ema_diff_std < price_diff_std
    
    def test_ema_length(self):
        """EMA should have same length as input."""
        prices = pd.Series(np.random.randn(100))
        ema = calculate_ema(prices, 20)
        
        assert len(ema) == len(prices)


class TestATR:
    """Tests for ATR calculation."""
    
    def test_atr_positive(self):
        """ATR should always be positive."""
        high = pd.Series(np.random.randn(100).cumsum() + 100)
        low = high - np.abs(np.random.randn(100))
        close = pd.Series(np.random.randn(100).cumsum() + 100)
        
        atr = calculate_atr(high, low, close)
        
        # Check where values are not NaN
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
    
    def test_atr_length(self):
        """ATR should have same length as input."""
        high = pd.Series(np.random.randn(100))
        low = high - 1
        close = pd.Series(np.random.randn(100))
        
        atr = calculate_atr(high, low, close)
        assert len(atr) == len(high)


class TestVWAP:
    """Tests for VWAP calculation."""
    
    def test_vwap_range(self):
        """VWAP should be within the high-low range."""
        np.random.seed(42)
        high = pd.Series(np.random.randn(100).cumsum() + 105)
        low = high - np.abs(np.random.randn(100)) - 5
        close = (high + low) / 2
        volume = pd.Series(np.abs(np.random.randn(100)) * 1000 + 100)
        
        vwap = calculate_vwap(high, low, close, volume)
        
        # Check where values are not NaN
        mask = ~(vwap.isna() | high.isna() | low.isna())
        if mask.sum() > 0:
            assert (vwap[mask] >= low[mask]).all()
            assert (vwap[mask] <= high[mask]).all()
    
    def test_vwap_length(self):
        """VWAP should have same length as input."""
        high = pd.Series(np.random.randn(100))
        low = high - 1
        close = pd.Series(np.random.randn(100))
        volume = pd.Series(np.abs(np.random.randn(100)))
        
        vwap = calculate_vwap(high, low, close, volume)
        assert len(vwap) == len(high)


class TestAddTechnicalIndicators:
    """Tests for add_technical_indicators function."""
    
    def test_all_indicators_added(self):
        """All technical indicators should be added."""
        df = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105,
            'Low': np.random.randn(100).cumsum() + 95,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.abs(np.random.randn(100)) * 1000 + 100
        })
        
        result = add_technical_indicators(df)
        
        expected_columns = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'EMA_7', 'EMA_21', 'EMA_50', 'EMA_200',
            'ATR', 'VWAP'
        ]
        
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_original_columns_preserved(self):
        """Original columns should be preserved."""
        df = pd.DataFrame({
            'Open': np.random.randn(100),
            'High': np.random.randn(100),
            'Low': np.random.randn(100),
            'Close': np.random.randn(100),
            'Volume': np.abs(np.random.randn(100))
        })
        
        result = add_technical_indicators(df)
        
        for col in df.columns:
            assert col in result.columns


class TestTemporalFeatures:
    """Tests for temporal features."""
    
    def test_temporal_columns_added(self):
        """Temporal features should be added."""
        df = pd.DataFrame({
            'Timestamp': pd.date_range('2020-01-01', periods=100, freq='D'),
            'Close': np.random.randn(100).cumsum() + 100
        })
        
        result = add_temporal_features(df)
        
        expected_columns = ['year', 'month', 'day', 'dayofweek', 'hour', 'quarter']
        
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_temporal_values_valid(self):
        """Temporal features should have valid values."""
        df = pd.DataFrame({
            'Timestamp': pd.date_range('2020-01-01', periods=100, freq='D'),
            'Close': np.random.randn(100)
        })
        
        result = add_temporal_features(df)
        
        assert (result['month'] >= 1).all()
        assert (result['month'] <= 12).all()
        assert (result['dayofweek'] >= 0).all()
        assert (result['dayofweek'] <= 6).all()


class TestLagFeatures:
    """Tests for lag features."""
    
    def test_lag_columns_added(self):
        """Lag features should be added."""
        df = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100
        })
        
        result = add_lag_features(df, lags=[1, 2, 3])
        
        for lag in [1, 2, 3]:
            assert f'lag_{lag}' in result.columns
    
    def test_lag_values_correct(self):
        """Lag values should be correct."""
        df = pd.DataFrame({
            'Close': [1, 2, 3, 4, 5]
        })
        
        result = add_lag_features(df, lags=[1, 2])
        
        # Check lag_1
        assert result['lag_1'].iloc[1] == 1
        assert result['lag_1'].iloc[2] == 2
        
        # Check lag_2
        assert result['lag_2'].iloc[2] == 1
        assert result['lag_2'].iloc[3] == 2


class TestRollingFeatures:
    """Tests for rolling features."""
    
    def test_rolling_columns_added(self):
        """Rolling features should be added."""
        df = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100
        })
        
        result = add_rolling_features(df, windows=[3, 5, 7])
        
        for window in [3, 5, 7]:
            assert f'rolling_mean_{window}' in result.columns
            assert f'rolling_std_{window}' in result.columns
    
    def test_rolling_mean_correct(self):
        """Rolling mean should be correct."""
        df = pd.DataFrame({
            'Close': [1, 2, 3, 4, 5]
        })
        
        result = add_rolling_features(df, windows=[3])
        
        # Check rolling mean at index 3 (0-indexed)
        expected_mean = (2 + 3 + 4) / 3
        assert np.isclose(result['rolling_mean_3'].iloc[3], expected_mean)


# Run with: pytest tests/test_features.py -v

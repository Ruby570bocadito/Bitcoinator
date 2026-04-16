import pandas as pd
import numpy as np


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                   signal: int = 9) -> tuple:
    """Calculate MACD, signal line and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                             num_std: float = 2.0) -> tuple:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return prices.ewm(span=period, adjust=False).mean()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                 period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, 
                   volume: pd.Series) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to dataframe."""
    df = df.copy()
    
    df['RSI'] = calculate_rsi(df['Close'], 14)
    
    macd, signal, hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Middle'] = bb_middle
    df['BB_Lower'] = bb_lower
    
    for period in [7, 21, 50, 200]:
        df[f'EMA_{period}'] = calculate_ema(df['Close'], period)
    
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['VWAP'] = calculate_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    return df
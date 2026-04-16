"""
Backtesting Module for Bitcoinator.
Implements backtesting with transaction costs, slippage, and trading metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from src.utils.logger import setup_logger
from src.utils.config import BacktestConfig

logger = setup_logger("backtester")


class Signal(Enum):
    """Trading signals."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position: int  # 1 for long, -1 for short
    size: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    holding_period: Optional[int] = None


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    metrics: Dict[str, float]
    positions: pd.Series
    signals: pd.Series


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    
    Features:
    - Transaction costs (commission)
    - Slippage modeling
    - Position sizing
    - Comprehensive metrics
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.
        
        Args:
            config: BacktestConfig instance or None for defaults
        """
        self.config = config or BacktestConfig()
        self.logger = logger
        
        # State variables
        self.capital = self.config.initial_capital
        self.position = 0
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.positions: List[int] = []
        self.signals: List[int] = []
    
    def reset(self):
        """Reset backtester state."""
        self.capital = self.config.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        self.positions = []
        self.signals = []
    
    def calculate_position_size(self, price: float) -> float:
        """
        Calculate position size based on configuration.
        
        Args:
            price: Current asset price
        
        Returns:
            Number of units to trade
        """
        position_value = self.capital * self.config.position_size
        return position_value / price
    
    def apply_slippage(self, price: float, signal: Signal) -> float:
        """
        Apply slippage to execution price.
        
        Args:
            price: Original price
            signal: Trading signal
        
        Returns:
            Price with slippage applied
        """
        if signal == Signal.BUY:
            return price * (1 + self.config.slippage)
        elif signal == Signal.SELL:
            return price * (1 - self.config.slippage)
        return price
    
    def calculate_commission(self, value: float) -> float:
        """
        Calculate transaction commission.
        
        Args:
            value: Trade value
        
        Returns:
            Commission amount
        """
        return value * self.config.commission
    
    def execute_signal(
        self,
        signal: Signal,
        price: float,
        date: pd.Timestamp
    ) -> Optional[Trade]:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal
            price: Current price
            date: Current date
        
        Returns:
            Trade object if executed, None otherwise
        """
        exec_price = self.apply_slippage(price, signal)
        trade = None
        
        if signal == Signal.BUY and self.position == 0:
            # Open long position
            size = self.calculate_position_size(exec_price)
            commission = self.calculate_commission(size * exec_price)
            
            self.capital -= (size * exec_price + commission)
            self.position = size
            
            trade = Trade(
                entry_date=date,
                exit_date=None,
                entry_price=exec_price,
                exit_price=None,
                position=1,
                size=size,
                commission=commission,
                slippage=(exec_price - price) * size
            )
            self.trades.append(trade)
            
        elif signal == Signal.SELL and self.position > 0:
            # Close long position
            size = self.position
            commission = self.calculate_commission(size * exec_price)
            pnl = (exec_price - self.trades[-1].entry_price) * size - commission
            pnl_pct = ((exec_price - self.trades[-1].entry_price) / 
                      self.trades[-1].entry_price) * 100
            
            self.capital += (size * exec_price - commission)
            self.position = 0
            
            # Update the last trade
            self.trades[-1].exit_date = date
            self.trades[-1].exit_price = exec_price
            self.trades[-1].pnl = pnl
            self.trades[-1].pnl_pct = pnl_pct
            self.trades[-1].commission = (
                self.trades[-1].commission + commission
            )
            self.trades[-1].holding_period = len([
                t for t in self.trades if t.exit_date is None
            ])
            
            trade = self.trades[-1]
        
        return trade
    
    def run(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        price_column: str = "Close",
        date_column: str = "Timestamp"
    ) -> BacktestResult:
        """
        Run backtest on historical data with predictions.
        
        Args:
            data: DataFrame with price data
            predictions: Array of predicted prices
            price_column: Name of price column
            date_column: Name of date column
        
        Returns:
            BacktestResult with all results
        """
        self.reset()
        
        data = data.copy().reset_index(drop=True)
        prices = data[price_column].values
        dates = pd.to_datetime(data[date_column]).values
        
        self.logger.info(f"Running backtest on {len(data)} bars...")
        
        # Generate signals from predictions
        # Buy if predicted price > current price, sell if predicted < current
        for i in range(len(data)):
            if i == 0:
                self.signals.append(0)
                continue
            
            pred_change = predictions[i] - prices[i-1]
            
            if pred_change > 0:
                signal = Signal.BUY
            elif pred_change < 0:
                signal = Signal.SELL
            else:
                signal = Signal.HOLD
            
            self.signals.append(signal.value)
            
            # Execute signal
            self.execute_signal(signal, prices[i], dates[i])
            
            # Track equity
            current_equity = self.capital
            if self.position > 0:
                current_equity += self.position * prices[i]
            self.equity_curve.append(current_equity)
            self.positions.append(self.position)
        
        # Close any open positions at the end
        if self.position > 0:
            self.execute_signal(
                Signal.SELL,
                prices[-1],
                dates[-1]
            )
            current_equity = self.capital
            self.equity_curve.append(current_equity)
        else:
            self.equity_curve.append(self.capital)
        
        self.positions.append(self.position)
        
        # Create result objects
        equity_series = pd.Series(self.equity_curve, index=dates[:len(self.equity_curve)])
        returns = equity_series.pct_change().fillna(0)
        positions_series = pd.Series(self.positions, index=dates[:len(self.positions)])
        signals_series = pd.Series(self.signals, index=dates[:len(self.signals)])
        
        # Calculate metrics
        metrics = self.calculate_metrics(equity_series, returns)
        
        self.logger.info(f"Backtest complete. Total trades: {len(self.trades)}")
        self.logger.info(f"Total return: {metrics['total_return']:.2f}%")
        self.logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_series,
            returns=returns,
            metrics=metrics,
            positions=positions_series,
            signals=signals_series
        )
    
    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate comprehensive trading metrics.
        
        Args:
            equity_curve: Equity curve series
            returns: Returns series
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics["initial_capital"] = self.config.initial_capital
        metrics["final_equity"] = equity_curve.iloc[-1]
        metrics["total_return"] = (
            (equity_curve.iloc[-1] - self.config.initial_capital) /
            self.config.initial_capital * 100
        )
        metrics["total_trades"] = len([t for t in self.trades if t.exit_date is not None])
        
        # Win rate
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]
        
        if self.trades:
            metrics["win_rate"] = len(winning_trades) / len(self.trades) * 100
        else:
            metrics["win_rate"] = 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        metrics["profit_factor"] = gross_profit / max(gross_loss, 0.01)
        
        # Average win/loss
        metrics["avg_win"] = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        metrics["avg_loss"] = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        metrics["win_loss_ratio"] = (
            metrics["avg_win"] / abs(metrics["avg_loss"])
            if metrics["avg_loss"] != 0 else 0
        )
        
        # Risk metrics
        metrics["volatility"] = returns.std() * np.sqrt(252) * 100  # Annualized
        metrics["max_drawdown"] = self.calculate_max_drawdown(equity_curve)
        
        # Risk-adjusted returns
        risk_free_rate = 0.02  # 2% annual
        excess_returns = returns - (risk_free_rate / 252)
        
        if returns.std() != 0:
            metrics["sharpe_ratio"] = (
                excess_returns.mean() / returns.std() * np.sqrt(252)
            )
            metrics["sortino_ratio"] = self.calculate_sortino_ratio(returns, risk_free_rate)
        else:
            metrics["sharpe_ratio"] = 0
            metrics["sortino_ratio"] = 0
        
        metrics["calmar_ratio"] = (
            metrics["total_return"] / abs(metrics["max_drawdown"])
            if metrics["max_drawdown"] != 0 else 0
        )
        
        # Transaction costs
        total_commission = sum(t.commission for t in self.trades if t.commission)
        total_slippage = sum(t.slippage for t in self.trades if t.slippage)
        metrics["total_commission"] = total_commission
        metrics["total_slippage"] = total_slippage
        metrics["total_costs"] = total_commission + total_slippage
        
        return metrics
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Equity curve series
        
        Returns:
            Maximum drawdown as percentage
        """
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min() * 100
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Returns series
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sortino ratio
        """
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        downside_std = downside_returns.std() * np.sqrt(252)
        return excess_returns.mean() * 252 / downside_std
    
    def get_trade_summary(self) -> pd.DataFrame:
        """
        Get summary of all trades as DataFrame.
        
        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            if trade.exit_date is not None:
                trade_data.append({
                    "entry_date": trade.entry_date,
                    "exit_date": trade.exit_date,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "position": "Long" if trade.position == 1 else "Short",
                    "size": trade.size,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                    "commission": trade.commission,
                    "slippage": trade.slippage,
                    "holding_period": trade.holding_period
                })
        
        return pd.DataFrame(trade_data)


def run_backtest(
    data: pd.DataFrame,
    predictions: np.ndarray,
    config: Optional[BacktestConfig] = None
) -> BacktestResult:
    """
    Convenience function to run a backtest.
    
    Args:
        data: DataFrame with price data
        predictions: Array of predicted prices
        config: BacktestConfig instance
    
    Returns:
        BacktestResult with all results
    """
    backtester = Backtester(config)
    return backtester.run(data, predictions)

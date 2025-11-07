"""
Backtester
Simulates trading strategies and calculates performance metrics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class Backtester:
    """Backtests trading strategies based on model predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtester.
        
        Args:
            config: Configuration dictionary with backtesting parameters
        """
        self.config = config
        self.initial_capital = config.get('initial_capital', 100000)
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.slippage = config.get('slippage', 0.0005)
        self.confidence_threshold = config.get('strategy', {}).get('confidence_threshold', 0.6)
        
        self.trades = []
        self.portfolio_history = []
    
    def run_backtest(self, predictions_df: pd.DataFrame, 
                    price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest simulation.
        
        Args:
            predictions_df: DataFrame with predictions, confidences, and probabilities
            price_data: DataFrame with historical prices
        
        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info("Running backtest simulation")
        
        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            slippage=self.slippage
        )
        
        # Align data
        predictions_df = predictions_df.copy()
        
        # Iterate through dates
        dates = predictions_df.index.get_level_values('date').unique().sort_values()
        
        for date in dates:
            # Get predictions for this date
            day_predictions = predictions_df.xs(date, level='date')
            
            # Get prices for this date
            if isinstance(price_data.index, pd.MultiIndex):
                day_prices = price_data.xs(date, level='date')
            else:
                day_prices = price_data.loc[date]
            
            # Generate trading signals
            signals = self._generate_signals(day_predictions, day_prices)
            
            # Execute trades
            for ticker, signal in signals.items():
                if signal == 'BUY':
                    portfolio.buy(ticker, day_prices.loc[ticker, 'close'], date)
                elif signal == 'SELL':
                    portfolio.sell(ticker, day_prices.loc[ticker, 'close'], date)
            
            # Update portfolio value
            portfolio.update_value(day_prices['close'].to_dict(), date)
        
        # Calculate metrics
        metrics = self._calculate_metrics(portfolio)
        
        logger.info(f"Backtest complete. Total return: {metrics['total_return']:.2%}")
        
        return {
            'metrics': metrics,
            'portfolio_history': portfolio.history,
            'trades': portfolio.trades,
            'final_portfolio_value': portfolio.cash + portfolio.holdings_value
        }
    
    def _generate_signals(self, predictions: pd.DataFrame, 
                         prices: pd.DataFrame) -> Dict[str, str]:
        """Generate trading signals from predictions."""
        
        signals = {}
        
        for ticker in predictions.index:
            if ticker not in prices.index:
                continue
            
            pred = predictions.loc[ticker]
            
            # Only act if confident enough
            if pred['confidence'] < self.confidence_threshold:
                continue
            
            prediction_class = pred['prediction']
            
            if prediction_class == 2:  # BUY
                signals[ticker] = 'BUY'
            elif prediction_class == 0:  # SELL
                signals[ticker] = 'SELL'
            # prediction_class == 1 (HOLD) -> no action
        
        return signals
    
    def _calculate_metrics(self, portfolio: Portfolio) -> Dict[str, float]:
        """Calculate performance metrics."""
        
        if not portfolio.history:
            return {}
        
        history_df = pd.DataFrame(portfolio.history)
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df = history_df.set_index('date').sort_index()
        
        # Total return
        initial_value = self.initial_capital
        final_value = history_df['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Annualized return
        n_years = (history_df.index[-1] - history_df.index[0]).days / 365.25
        if n_years > 0:
            annualized_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annualized_return = 0
        
        # Daily returns
        history_df['daily_return'] = history_df['total_value'].pct_change()
        
        # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        daily_returns = history_df['daily_return'].dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = daily_returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # Maximum drawdown
        cumulative_returns = (1 + history_df['daily_return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        if portfolio.trades:
            winning_trades = sum(1 for t in portfolio.trades if t.get('profit', 0) > 0)
            win_rate = winning_trades / len(portfolio.trades)
        else:
            win_rate = 0
        
        # Profit factor
        if portfolio.trades:
            gross_profit = sum(t.get('profit', 0) for t in portfolio.trades if t.get('profit', 0) > 0)
            gross_loss = abs(sum(t.get('profit', 0) for t in portfolio.trades if t.get('profit', 0) < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            profit_factor = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(portfolio.trades),
            'final_value': final_value,
        }
        
        return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Backtester module ready")

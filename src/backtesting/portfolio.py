"""
Portfolio Manager
Manages portfolio state, positions, and trades.
"""

import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class Portfolio:
    """Manages portfolio state and executions."""
    
    def __init__(self, initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 max_positions: int = 10):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction of trade value
            slippage: Slippage as fraction of price
            max_positions: Maximum number of concurrent positions
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_positions = max_positions
        
        # Positions: {ticker: {'shares': n, 'avg_price': p}}
        self.positions = {}
        
        # Holdings value
        self.holdings_value = 0
        
        # Trade history
        self.trades = []
        
        # Portfolio value history
        self.history = []
    
    def buy(self, ticker: str, price: float, date) -> bool:
        """
        Buy stock.
        
        Args:
            ticker: Stock ticker
            price: Purchase price
            date: Transaction date
        
        Returns:
            True if trade executed, False otherwise
        """
        # Check if we can add more positions
        if ticker not in self.positions and len(self.positions) >= self.max_positions:
            logger.debug(f"Cannot buy {ticker}: max positions reached")
            return False
        
        # Apply slippage (price moves against us)
        actual_price = price * (1 + self.slippage)
        
        # Calculate position size (equal weight across max positions)
        target_position_value = self.cash / (self.max_positions - len(self.positions))
        shares = int(target_position_value / actual_price)
        
        if shares == 0:
            logger.debug(f"Cannot buy {ticker}: insufficient cash")
            return False
        
        # Calculate cost
        trade_value = shares * actual_price
        cost = trade_value * self.transaction_cost
        total_cost = trade_value + cost
        
        if total_cost > self.cash:
            # Adjust shares to fit available cash
            total_cost = self.cash * 0.99  # Leave small buffer
            trade_value = total_cost / (1 + self.transaction_cost)
            shares = int(trade_value / actual_price)
            
            if shares == 0:
                return False
            
            trade_value = shares * actual_price
            cost = trade_value * self.transaction_cost
            total_cost = trade_value + cost
        
        # Execute trade
        self.cash -= total_cost
        
        if ticker in self.positions:
            # Add to existing position
            old_shares = self.positions[ticker]['shares']
            old_avg_price = self.positions[ticker]['avg_price']
            
            new_shares = old_shares + shares
            new_avg_price = (old_shares * old_avg_price + shares * actual_price) / new_shares
            
            self.positions[ticker] = {
                'shares': new_shares,
                'avg_price': new_avg_price
            }
        else:
            # New position
            self.positions[ticker] = {
                'shares': shares,
                'avg_price': actual_price
            }
        
        # Record trade
        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': actual_price,
            'value': trade_value,
            'cost': cost,
        })
        
        logger.debug(f"Bought {shares} shares of {ticker} at ${actual_price:.2f}")
        
        return True
    
    def sell(self, ticker: str, price: float, date) -> bool:
        """
        Sell stock.
        
        Args:
            ticker: Stock ticker
            price: Sale price
            date: Transaction date
        
        Returns:
            True if trade executed, False otherwise
        """
        if ticker not in self.positions:
            logger.debug(f"Cannot sell {ticker}: no position")
            return False
        
        # Apply slippage (price moves against us)
        actual_price = price * (1 - self.slippage)
        
        # Sell entire position
        shares = self.positions[ticker]['shares']
        avg_price = self.positions[ticker]['avg_price']
        
        # Calculate proceeds
        trade_value = shares * actual_price
        cost = trade_value * self.transaction_cost
        proceeds = trade_value - cost
        
        # Calculate profit
        profit = (actual_price - avg_price) * shares - cost
        
        # Execute trade
        self.cash += proceeds
        
        # Remove position
        del self.positions[ticker]
        
        # Record trade
        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': actual_price,
            'value': trade_value,
            'cost': cost,
            'profit': profit,
            'return': profit / (avg_price * shares) if avg_price > 0 else 0,
        })
        
        logger.debug(f"Sold {shares} shares of {ticker} at ${actual_price:.2f}, profit: ${profit:.2f}")
        
        return True
    
    def update_value(self, current_prices: Dict[str, float], date) -> None:
        """
        Update portfolio value based on current prices.
        
        Args:
            current_prices: Dictionary of {ticker: price}
            date: Current date
        """
        # Calculate holdings value
        self.holdings_value = 0
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                self.holdings_value += position['shares'] * current_prices[ticker]
        
        total_value = self.cash + self.holdings_value
        
        # Record history
        self.history.append({
            'date': date,
            'cash': self.cash,
            'holdings_value': self.holdings_value,
            'total_value': total_value,
            'num_positions': len(self.positions),
        })
    
    def get_positions_summary(self) -> pd.DataFrame:
        """Get summary of current positions."""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for ticker, pos in self.positions.items():
            data.append({
                'ticker': ticker,
                'shares': pos['shares'],
                'avg_price': pos['avg_price'],
                'value': pos['shares'] * pos['avg_price']
            })
        
        return pd.DataFrame(data)
    
    def get_trades_summary(self) -> pd.DataFrame:
        """Get summary of all trades."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test portfolio
    portfolio = Portfolio(initial_capital=100000)
    
    portfolio.buy('AAPL', 150, '2023-01-01')
    portfolio.buy('MSFT', 300, '2023-01-02')
    portfolio.sell('AAPL', 160, '2023-01-03')
    
    portfolio.update_value({'MSFT': 310}, '2023-01-04')
    
    print("\nPositions:")
    print(portfolio.get_positions_summary())
    
    print("\nTrades:")
    print(portfolio.get_trades_summary())
    
    print(f"\nCash: ${portfolio.cash:.2f}")
    print(f"Holdings: ${portfolio.holdings_value:.2f}")

"""
Target Generator
Creates target variables for Buy/Hold/Sell classification.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class TargetGenerator:
    """Generates target variables for stock trading signals."""
    
    def __init__(self, horizon: int = 5, 
                 buy_threshold: float = 0.02,
                 sell_threshold: float = -0.02,
                 target_type: str = 'classification'):
        """
        Initialize target generator.
        
        Args:
            horizon: Number of days ahead to predict
            buy_threshold: Return threshold for BUY signal (e.g., 0.02 = 2%)
            sell_threshold: Return threshold for SELL signal (e.g., -0.02 = -2%)
            target_type: 'classification' or 'regression'
        """
        self.horizon = horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.target_type = target_type
        
        if target_type not in ['classification', 'regression']:
            raise ValueError("target_type must be 'classification' or 'regression'")
    
    def generate_targets(self, df: pd.DataFrame, 
                        price_col: str = 'close') -> pd.DataFrame:
        """
        Generate target variables.
        
        Args:
            df: DataFrame with price data (MultiIndex or single ticker)
            price_col: Column name containing prices
        
        Returns:
            DataFrame with added target columns:
                - future_return: Forward return
                - target: Classification label (0=SELL, 1=HOLD, 2=BUY) or regression value
        """
        logger.info(f"Generating targets with {self.horizon}-day horizon")
        
        result = df.copy()
        is_multi_ticker = isinstance(result.index, pd.MultiIndex)
        
        if is_multi_ticker:
            # Process each ticker separately
            tickers = result.index.get_level_values('ticker').unique()
            target_dfs = []
            
            for ticker in tickers:
                ticker_data = result.xs(ticker, level='ticker')
                ticker_data = self._generate_targets_single_ticker(ticker_data, price_col)
                ticker_data['ticker'] = ticker
                target_dfs.append(ticker_data)
            
            result = pd.concat(target_dfs, axis=0)
            result = result.reset_index()
            result = result.set_index(['date', 'ticker']) if 'date' in result.columns else result
        else:
            result = self._generate_targets_single_ticker(result, price_col)
        
        # Log class distribution for classification
        if self.target_type == 'classification' and 'target' in result.columns:
            target_dist = result['target'].value_counts().sort_index()
            logger.info(f"Target distribution:\n{target_dist}")
            
            # Calculate percentages
            target_pct = (target_dist / target_dist.sum() * 100).round(2)
            logger.info(f"Target percentages:\n{target_pct}")
        
        return result
    
    def _generate_targets_single_ticker(self, df: pd.DataFrame, 
                                       price_col: str) -> pd.DataFrame:
        """Generate targets for a single ticker."""
        
        result = df.copy()
        
        if price_col not in result.columns:
            raise ValueError(f"Price column '{price_col}' not found in DataFrame")
        
        # Calculate future return
        future_price = result[price_col].shift(-self.horizon)
        current_price = result[price_col]
        
        result['future_return'] = (future_price - current_price) / current_price
        
        if self.target_type == 'classification':
            # Create classification labels
            # 0 = SELL, 1 = HOLD, 2 = BUY
            result['target'] = 1  # Default to HOLD
            
            result.loc[result['future_return'] >= self.buy_threshold, 'target'] = 2  # BUY
            result.loc[result['future_return'] <= self.sell_threshold, 'target'] = 0  # SELL
            
            # Convert to int
            result['target'] = result['target'].astype('Int64')
            
        else:
            # Regression: use raw future return as target
            result['target'] = result['future_return']
        
        # Additional target variations (optional)
        # Max return within horizon
        result['max_return_horizon'] = result[price_col].rolling(
            window=self.horizon, min_periods=1
        ).max().shift(-self.horizon) / result[price_col] - 1
        
        # Min return within horizon
        result['min_return_horizon'] = result[price_col].rolling(
            window=self.horizon, min_periods=1
        ).min().shift(-self.horizon) / result[price_col] - 1
        
        return result
    
    def split_features_targets(self, df: pd.DataFrame,
                              drop_future_leak_cols: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features (X) and targets (y).
        
        Args:
            df: DataFrame with features and targets
            drop_future_leak_cols: Whether to drop columns that leak future information
        
        Returns:
            X (features), y (targets)
        """
        
        # Identify target column
        if 'target' not in df.columns:
            raise ValueError("Target column not found. Run generate_targets first.")
        
        # Columns to drop
        cols_to_drop = ['target', 'future_return', 'max_return_horizon', 'min_return_horizon']
        
        # Drop columns that might leak future information
        if drop_future_leak_cols:
            # Look for any column that might contain future data
            future_leak_patterns = ['future_', 'forward_', 'ahead_']
            for col in df.columns:
                if any(pattern in col.lower() for pattern in future_leak_patterns):
                    if col not in cols_to_drop:
                        cols_to_drop.append(col)
        
        # Keep only columns that exist
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        # Features
        X = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Target
        y = df['target']
        
        # Remove rows with NaN targets (last `horizon` rows won't have targets)
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Split into features (X): {X.shape}, targets (y): {y.shape}")
        
        return X, y
    
    def create_time_based_split(self, df: pd.DataFrame,
                               train_size: float = 0.7,
                               val_size: float = 0.15,
                               test_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/val/test splits.
        Important: Never shuffle time series data!
        
        Args:
            df: DataFrame with datetime index
            train_size: Fraction for training
            val_size: Fraction for validation
            test_size: Fraction for testing
        
        Returns:
            train_df, val_df, test_df
        """
        
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0")
        
        # Handle multi-ticker data
        is_multi_ticker = isinstance(df.index, pd.MultiIndex)
        
        if is_multi_ticker:
            # Get unique dates
            dates = df.index.get_level_values('date').unique().sort_values()
        else:
            if isinstance(df.index, pd.DatetimeIndex):
                dates = df.index.unique().sort_values()
            else:
                # Assume there's a 'date' column
                dates = pd.to_datetime(df['date']).unique()
                dates = pd.Series(dates).sort_values().values
        
        n_dates = len(dates)
        
        # Calculate split points
        train_end_idx = int(n_dates * train_size)
        val_end_idx = int(n_dates * (train_size + val_size))
        
        train_end_date = dates[train_end_idx]
        val_end_date = dates[val_end_idx]
        
        logger.info(f"Time-based split:")
        logger.info(f"  Train: {dates[0]} to {train_end_date}")
        logger.info(f"  Val:   {train_end_date} to {val_end_date}")
        logger.info(f"  Test:  {val_end_date} to {dates[-1]}")
        
        # Split data
        if is_multi_ticker:
            train_df = df[df.index.get_level_values('date') < train_end_date]
            val_df = df[(df.index.get_level_values('date') >= train_end_date) & 
                       (df.index.get_level_values('date') < val_end_date)]
            test_df = df[df.index.get_level_values('date') >= val_end_date]
        else:
            if isinstance(df.index, pd.DatetimeIndex):
                train_df = df[df.index < train_end_date]
                val_df = df[(df.index >= train_end_date) & (df.index < val_end_date)]
                test_df = df[df.index >= val_end_date]
            else:
                df_copy = df.copy()
                df_copy['date'] = pd.to_datetime(df_copy['date'])
                train_df = df_copy[df_copy['date'] < train_end_date]
                val_df = df_copy[(df_copy['date'] >= train_end_date) & 
                                (df_copy['date'] < val_end_date)]
                test_df = df_copy[df_copy['date'] >= val_end_date]
        
        logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 500),
        'rsi': np.random.uniform(30, 70, 500),
    })
    df = df.set_index('date')
    
    # Generate targets
    generator = TargetGenerator(horizon=5, buy_threshold=0.02, sell_threshold=-0.02)
    df_with_targets = generator.generate_targets(df)
    
    print(df_with_targets.head(10))
    print(f"\nTarget distribution:\n{df_with_targets['target'].value_counts()}")
    
    # Split features and targets
    X, y = generator.split_features_targets(df_with_targets)
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Time-based split
    train, val, test = generator.create_time_based_split(df_with_targets)
    print(f"\nTrain: {len(train)}, Val: {len(val)}, Test: {len(test)}")

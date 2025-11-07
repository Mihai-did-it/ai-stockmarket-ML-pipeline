"""
Feature Merger
Combines price, technical, fundamental, and sentiment data with time alignment.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FeatureMerger:
    """Merges different data sources into a unified feature set."""
    
    def __init__(self, forward_fill_fundamentals: bool = True,
                 max_ffill_days: int = 90):
        """
        Initialize feature merger.
        
        Args:
            forward_fill_fundamentals: Whether to forward-fill fundamental data
            max_ffill_days: Maximum days to forward-fill missing data
        """
        self.forward_fill_fundamentals = forward_fill_fundamentals
        self.max_ffill_days = max_ffill_days
    
    def merge_all_features(self, 
                          price_with_technicals: pd.DataFrame,
                          fundamentals: Optional[pd.DataFrame] = None,
                          sentiment: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge all feature sources with time alignment.
        
        Args:
            price_with_technicals: DataFrame with price and technical indicators
                                   (MultiIndex: date, ticker OR single index: date)
            fundamentals: DataFrame with fundamental metrics
            sentiment: DataFrame with sentiment scores
        
        Returns:
            Unified DataFrame with all features
        """
        logger.info("Merging all features")
        
        # Start with price and technical data
        result = price_with_technicals.copy()
        
        # Determine if we have multi-ticker data
        is_multi_ticker = isinstance(result.index, pd.MultiIndex)
        
        if is_multi_ticker:
            # Process each ticker separately
            tickers = result.index.get_level_values('ticker').unique()
            merged_dfs = []
            
            for ticker in tickers:
                ticker_data = result.xs(ticker, level='ticker')
                
                # Merge fundamentals
                if fundamentals is not None and not fundamentals.empty:
                    ticker_fundamentals = fundamentals[fundamentals['ticker'] == ticker]
                    ticker_data = self._merge_fundamentals(ticker_data, ticker_fundamentals)
                
                # Merge sentiment
                if sentiment is not None and not sentiment.empty:
                    ticker_sentiment = sentiment[sentiment['ticker'] == ticker]
                    ticker_data = self._merge_sentiment(ticker_data, ticker_sentiment)
                
                ticker_data['ticker'] = ticker
                merged_dfs.append(ticker_data)
            
            result = pd.concat(merged_dfs, axis=0)
            result = result.reset_index()
            result = result.set_index(['date', 'ticker']) if 'date' in result.columns else result
            
        else:
            # Single ticker
            if fundamentals is not None and not fundamentals.empty:
                result = self._merge_fundamentals(result, fundamentals)
            
            if sentiment is not None and not sentiment.empty:
                result = self._merge_sentiment(result, sentiment)
        
        logger.info(f"Merged features: {result.shape[1]} columns, {result.shape[0]} rows")
        
        return result
    
    def _merge_fundamentals(self, price_data: pd.DataFrame, 
                           fundamentals: pd.DataFrame) -> pd.DataFrame:
        """Merge fundamental data with price data."""
        
        if fundamentals.empty:
            return price_data
        
        # Prepare fundamentals
        fund = fundamentals.copy()
        
        # Remove ticker column if present
        if 'ticker' in fund.columns:
            fund = fund.drop(columns=['ticker'])
        
        # Ensure date is datetime
        if 'date' in fund.columns:
            fund['date'] = pd.to_datetime(fund['date'])
            fund = fund.set_index('date')
        
        # Merge with price data
        result = price_data.copy()
        result_index = result.index
        
        # Merge using asof (backward fill from most recent fundamental data)
        if isinstance(result.index, pd.DatetimeIndex):
            result = result.reset_index()
            result = pd.merge_asof(
                result.sort_values('date' if 'date' in result.columns else result.index.name),
                fund.reset_index().sort_values('date'),
                on='date' if 'date' in result.columns else result.index.name,
                direction='backward',
                tolerance=pd.Timedelta(days=self.max_ffill_days)
            )
            result = result.set_index(result_index.name if result_index.name else 'date')
        else:
            # If no explicit date column, try to merge on index
            result = result.join(fund, how='left')
            
            if self.forward_fill_fundamentals:
                # Forward fill fundamentals
                fundamental_cols = fund.columns
                result[fundamental_cols] = result[fundamental_cols].fillna(method='ffill', limit=self.max_ffill_days)
        
        return result
    
    def _merge_sentiment(self, price_data: pd.DataFrame, 
                        sentiment: pd.DataFrame) -> pd.DataFrame:
        """Merge sentiment data with price data."""
        
        if sentiment.empty:
            return price_data
        
        # Prepare sentiment data
        sent = sentiment.copy()
        
        # Remove ticker column if present
        if 'ticker' in sent.columns:
            sent = sent.drop(columns=['ticker'])
        
        # Ensure date is in the right format
        if 'date' in sent.columns:
            sent['date'] = pd.to_datetime(sent['date'])
            sent = sent.set_index('date')
        
        # Merge with price data
        result = price_data.copy()
        
        if isinstance(result.index, pd.DatetimeIndex):
            result = result.join(sent, how='left')
        else:
            result = result.reset_index()
            result['date'] = pd.to_datetime(result['date'])
            result = result.set_index('date')
            result = result.join(sent, how='left')
            result = result.reset_index()
        
        # Fill missing sentiment with neutral (0)
        sentiment_cols = [col for col in sent.columns if 'sentiment' in col or 'news' in col]
        for col in sentiment_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)
        
        return result
    
    def add_lag_features(self, df: pd.DataFrame, 
                        feature_cols: List[str],
                        lags: List[int]) -> pd.DataFrame:
        """
        Add lagged features (past values).
        
        Args:
            df: Input DataFrame
            feature_cols: Columns to create lags for
            lags: List of lag periods (e.g., [1, 2, 3, 5])
        
        Returns:
            DataFrame with additional lag columns
        """
        logger.info(f"Adding lag features: {len(feature_cols)} features x {len(lags)} lags")
        
        result = df.copy()
        is_multi_ticker = isinstance(result.index, pd.MultiIndex)
        
        if is_multi_ticker:
            # Process each ticker separately
            tickers = result.index.get_level_values('ticker').unique()
            lagged_dfs = []
            
            for ticker in tickers:
                ticker_data = result.xs(ticker, level='ticker')
                ticker_data = self._add_lags_single_ticker(ticker_data, feature_cols, lags)
                ticker_data['ticker'] = ticker
                lagged_dfs.append(ticker_data)
            
            result = pd.concat(lagged_dfs, axis=0)
            result = result.reset_index()
            result = result.set_index(['date', 'ticker']) if 'date' in result.columns else result
        else:
            result = self._add_lags_single_ticker(result, feature_cols, lags)
        
        return result
    
    def _add_lags_single_ticker(self, df: pd.DataFrame, 
                               feature_cols: List[str],
                               lags: List[int]) -> pd.DataFrame:
        """Add lags for a single ticker."""
        result = df.copy()
        
        for col in feature_cols:
            if col not in result.columns:
                continue
            
            for lag in lags:
                result[f'{col}_lag{lag}'] = result[col].shift(lag)
        
        return result
    
    def add_rolling_features(self, df: pd.DataFrame,
                           feature_cols: List[str],
                           windows: List[int]) -> pd.DataFrame:
        """
        Add rolling window aggregations (mean, std, min, max).
        
        Args:
            df: Input DataFrame
            feature_cols: Columns to create rolling features for
            windows: List of window sizes (e.g., [5, 10, 20])
        
        Returns:
            DataFrame with additional rolling features
        """
        logger.info(f"Adding rolling features: {len(feature_cols)} features x {len(windows)} windows")
        
        result = df.copy()
        is_multi_ticker = isinstance(result.index, pd.MultiIndex)
        
        if is_multi_ticker:
            tickers = result.index.get_level_values('ticker').unique()
            rolling_dfs = []
            
            for ticker in tickers:
                ticker_data = result.xs(ticker, level='ticker')
                ticker_data = self._add_rolling_single_ticker(ticker_data, feature_cols, windows)
                ticker_data['ticker'] = ticker
                rolling_dfs.append(ticker_data)
            
            result = pd.concat(rolling_dfs, axis=0)
            result = result.reset_index()
            result = result.set_index(['date', 'ticker']) if 'date' in result.columns else result
        else:
            result = self._add_rolling_single_ticker(result, feature_cols, windows)
        
        return result
    
    def _add_rolling_single_ticker(self, df: pd.DataFrame,
                                   feature_cols: List[str],
                                   windows: List[int]) -> pd.DataFrame:
        """Add rolling features for a single ticker."""
        result = df.copy()
        
        for col in feature_cols:
            if col not in result.columns:
                continue
            
            for window in windows:
                # Mean
                result[f'{col}_rolling_mean_{window}'] = result[col].rolling(window).mean()
                # Std
                result[f'{col}_rolling_std_{window}'] = result[col].rolling(window).std()
                # Min
                result[f'{col}_rolling_min_{window}'] = result[col].rolling(window).min()
                # Max
                result[f'{col}_rolling_max_{window}'] = result[col].rolling(window).max()
        
        return result
    
    def clean_features(self, df: pd.DataFrame, 
                      drop_na_threshold: float = 0.5) -> pd.DataFrame:
        """
        Clean feature data: handle NaN, inf, etc.
        
        Args:
            df: Input DataFrame
            drop_na_threshold: Drop columns with more than this fraction of NaN
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning features")
        
        result = df.copy()
        initial_shape = result.shape
        
        # Replace inf with NaN
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with too many NaN
        na_fraction = result.isna().mean()
        cols_to_drop = na_fraction[na_fraction > drop_na_threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{drop_na_threshold*100}% NaN")
            result = result.drop(columns=cols_to_drop)
        
        # Forward fill remaining NaN (for time series)
        result = result.fillna(method='ffill')
        
        # Backward fill for any remaining NaN at the start
        result = result.fillna(method='bfill')
        
        # Fill any remaining NaN with 0
        result = result.fillna(0)
        
        logger.info(f"Cleaning complete: {initial_shape} -> {result.shape}")
        
        return result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    tickers = ['AAPL', 'MSFT']
    
    # Price data
    price_data = []
    for ticker in tickers:
        for date in dates:
            price_data.append({
                'date': date,
                'ticker': ticker,
                'close': np.random.randn() + 100,
                'volume': np.random.randint(1000000, 10000000),
                'rsi': np.random.uniform(30, 70),
            })
    
    price_df = pd.DataFrame(price_data)
    price_df = price_df.set_index(['date', 'ticker'])
    
    # Fundamentals (quarterly)
    fund_dates = pd.date_range('2020-01-01', periods=4, freq='Q')
    fundamentals = []
    for ticker in tickers:
        for date in fund_dates:
            fundamentals.append({
                'date': date,
                'ticker': ticker,
                'pe_ratio': np.random.uniform(15, 30),
                'revenue': np.random.uniform(1e9, 1e10),
            })
    
    fund_df = pd.DataFrame(fundamentals)
    
    # Merge
    merger = FeatureMerger()
    result = merger.merge_all_features(price_df, fund_df)
    
    # Add lag features
    result = merger.add_lag_features(result, ['close', 'rsi'], [1, 2, 5])
    
    # Clean
    result = merger.clean_features(result)
    
    print(result.head())
    print(f"\nShape: {result.shape}")
    print(f"\nColumns: {result.columns.tolist()}")

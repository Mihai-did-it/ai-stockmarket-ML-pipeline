"""
Price Data Fetcher
Retrieves historical OHLCV data from various sources.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class PriceDataFetcher:
    """Fetches historical price data for stocks."""
    
    def __init__(self, cache_dir: str = 'data/raw', use_cache: bool = True, 
                 cache_expiry_days: int = 7):
        """
        Initialize the price data fetcher.
        
        Args:
            cache_dir: Directory to cache downloaded data
            use_cache: Whether to use cached data
            cache_expiry_days: Days before cache expires
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.cache_expiry_days = cache_expiry_days
        
    def fetch_data(self, tickers: List[str], start_date: str, 
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical price data for multiple tickers.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with multi-index (date, ticker) and OHLCV columns
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        all_data = []
        
        for ticker in tickers:
            logger.info(f"Fetching price data for {ticker}")
            
            # Check cache
            if self.use_cache:
                cached_data = self._load_from_cache(ticker, start_date, end_date)
                if cached_data is not None:
                    logger.info(f"Using cached data for {ticker}")
                    all_data.append(cached_data)
                    continue
            
            # Fetch from API
            try:
                df = self._fetch_from_yfinance(ticker, start_date, end_date)
                if df is not None and not df.empty:
                    df['ticker'] = ticker
                    all_data.append(df)
                    
                    # Save to cache
                    if self.use_cache:
                        self._save_to_cache(df, ticker, start_date, end_date)
                        
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No data fetched for any ticker")
        
        # Combine all tickers
        combined_df = pd.concat(all_data, axis=0)
        combined_df = combined_df.set_index(['Date', 'ticker'])
        combined_df = combined_df.sort_index()
        
        logger.info(f"Successfully fetched data for {len(all_data)} tickers")
        return combined_df
    
    def _fetch_from_yfinance(self, ticker: str, start_date: str, 
                            end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            # Standardize column names
            df = df.reset_index()
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Keep only essential columns
            essential_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in essential_cols if col in df.columns]]
            
            return df
            
        except Exception as e:
            logger.error(f"yfinance error for {ticker}: {str(e)}")
            return None
    
    def _get_cache_path(self, ticker: str, start_date: str, end_date: str) -> Path:
        """Generate cache file path."""
        filename = f"{ticker}_{start_date}_{end_date}.parquet"
        return self.cache_dir / filename
    
    def _load_from_cache(self, ticker: str, start_date: str, 
                        end_date: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        cache_path = self._get_cache_path(ticker, start_date, end_date)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age.days > self.cache_expiry_days:
            logger.info(f"Cache expired for {ticker}")
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            logger.warning(f"Error loading cache for {ticker}: {str(e)}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, ticker: str, 
                      start_date: str, end_date: str) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(ticker, start_date, end_date)
        try:
            df.to_parquet(cache_path, index=False)
            logger.debug(f"Cached data for {ticker}")
        except Exception as e:
            logger.warning(f"Error caching data for {ticker}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    fetcher = PriceDataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    df = fetcher.fetch_data(tickers, '2020-01-01', '2023-12-31')
    print(df.head())
    print(f"\nShape: {df.shape}")

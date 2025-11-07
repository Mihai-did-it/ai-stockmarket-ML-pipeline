"""
Fundamentals Data Fetcher
Retrieves company financial statements and metrics.
"""

import pandas as pd
import yfinance as yf
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class FundamentalsDataFetcher:
    """Fetches fundamental financial data for stocks."""
    
    def __init__(self, cache_dir: str = 'data/raw', use_cache: bool = True):
        """
        Initialize the fundamentals data fetcher.
        
        Args:
            cache_dir: Directory to cache downloaded data
            use_cache: Whether to use cached data
        """
        self.cache_dir = Path(cache_dir) / 'fundamentals'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
    
    def fetch_fundamentals(self, tickers: List[str], 
                          start_date: str = None) -> pd.DataFrame:
        """
        Fetch fundamental data for multiple tickers.
        
        Args:
            tickers: List of stock tickers
            start_date: Optional start date for filtering
            
        Returns:
            DataFrame with fundamental metrics per ticker per quarter
        """
        all_fundamentals = []
        
        for ticker in tickers:
            logger.info(f"Fetching fundamentals for {ticker}")
            
            try:
                fund_data = self._fetch_ticker_fundamentals(ticker)
                if fund_data is not None and not fund_data.empty:
                    fund_data['ticker'] = ticker
                    all_fundamentals.append(fund_data)
                    
            except Exception as e:
                logger.error(f"Error fetching fundamentals for {ticker}: {str(e)}")
                continue
        
        if not all_fundamentals:
            logger.warning("No fundamental data fetched")
            return pd.DataFrame()
        
        # Combine all tickers
        combined_df = pd.concat(all_fundamentals, axis=0, ignore_index=True)
        
        # Filter by date if specified
        if start_date and 'date' in combined_df.columns:
            combined_df = combined_df[combined_df['date'] >= pd.to_datetime(start_date)]
        
        logger.info(f"Successfully fetched fundamentals for {len(all_fundamentals)} tickers")
        return combined_df
    
    def _fetch_ticker_fundamentals(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch fundamental data for a single ticker using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get key financial metrics
            info = stock.info
            quarterly_financials = stock.quarterly_financials
            quarterly_balance_sheet = stock.quarterly_balance_sheet
            quarterly_cashflow = stock.quarterly_cashflow
            
            # Extract key metrics
            fundamentals_list = []
            
            # Get dates from quarterly financials
            if quarterly_financials is not None and not quarterly_financials.empty:
                dates = quarterly_financials.columns
                
                for date in dates:
                    metrics = {'date': pd.to_datetime(date)}
                    
                    # Valuation metrics (from info - these are current/latest)
                    metrics['pe_ratio'] = info.get('trailingPE', None)
                    metrics['forward_pe'] = info.get('forwardPE', None)
                    metrics['pb_ratio'] = info.get('priceToBook', None)
                    metrics['ps_ratio'] = info.get('priceToSalesTrailing12Months', None)
                    metrics['peg_ratio'] = info.get('pegRatio', None)
                    metrics['market_cap'] = info.get('marketCap', None)
                    metrics['enterprise_value'] = info.get('enterpriseValue', None)
                    
                    # Income statement metrics
                    if quarterly_financials is not None and date in quarterly_financials.columns:
                        qf = quarterly_financials[date]
                        metrics['revenue'] = qf.get('Total Revenue', None)
                        metrics['gross_profit'] = qf.get('Gross Profit', None)
                        metrics['operating_income'] = qf.get('Operating Income', None)
                        metrics['net_income'] = qf.get('Net Income', None)
                        metrics['ebitda'] = qf.get('EBITDA', None)
                    
                    # Balance sheet metrics
                    if quarterly_balance_sheet is not None and date in quarterly_balance_sheet.columns:
                        qb = quarterly_balance_sheet[date]
                        metrics['total_assets'] = qb.get('Total Assets', None)
                        metrics['total_liabilities'] = qb.get('Total Liabilities Net Minority Interest', None)
                        metrics['stockholder_equity'] = qb.get('Stockholders Equity', None)
                        metrics['total_debt'] = qb.get('Total Debt', None)
                        metrics['cash'] = qb.get('Cash And Cash Equivalents', None)
                        metrics['current_assets'] = qb.get('Current Assets', None)
                        metrics['current_liabilities'] = qb.get('Current Liabilities', None)
                    
                    # Cash flow metrics
                    if quarterly_cashflow is not None and date in quarterly_cashflow.columns:
                        qc = quarterly_cashflow[date]
                        metrics['operating_cash_flow'] = qc.get('Operating Cash Flow', None)
                        metrics['free_cash_flow'] = qc.get('Free Cash Flow', None)
                        metrics['capex'] = qc.get('Capital Expenditure', None)
                    
                    # Calculate derived metrics
                    if metrics['revenue'] and metrics['net_income']:
                        metrics['profit_margin'] = metrics['net_income'] / metrics['revenue']
                    
                    if metrics['gross_profit'] and metrics['revenue']:
                        metrics['gross_margin'] = metrics['gross_profit'] / metrics['revenue']
                    
                    if metrics['operating_income'] and metrics['revenue']:
                        metrics['operating_margin'] = metrics['operating_income'] / metrics['revenue']
                    
                    if metrics['net_income'] and metrics['stockholder_equity']:
                        metrics['roe'] = metrics['net_income'] / metrics['stockholder_equity']
                    
                    if metrics['net_income'] and metrics['total_assets']:
                        metrics['roa'] = metrics['net_income'] / metrics['total_assets']
                    
                    if metrics['total_debt'] and metrics['stockholder_equity']:
                        metrics['debt_to_equity'] = metrics['total_debt'] / metrics['stockholder_equity']
                    
                    if metrics['current_assets'] and metrics['current_liabilities']:
                        metrics['current_ratio'] = metrics['current_assets'] / metrics['current_liabilities']
                    
                    # Quick ratio (current assets - inventory) / current liabilities
                    # Simplified version without inventory
                    if metrics['cash'] and metrics['current_liabilities']:
                        metrics['quick_ratio'] = metrics['cash'] / metrics['current_liabilities']
                    
                    metrics['dividend_yield'] = info.get('dividendYield', None)
                    
                    fundamentals_list.append(metrics)
            
            if not fundamentals_list:
                # If no quarterly data, create single entry with current info
                metrics = {
                    'date': pd.to_datetime('today'),
                    'pe_ratio': info.get('trailingPE', None),
                    'forward_pe': info.get('forwardPE', None),
                    'pb_ratio': info.get('priceToBook', None),
                    'ps_ratio': info.get('priceToSalesTrailing12Months', None),
                    'market_cap': info.get('marketCap', None),
                    'dividend_yield': info.get('dividendYield', None),
                }
                fundamentals_list.append(metrics)
            
            df = pd.DataFrame(fundamentals_list)
            df = df.sort_values('date')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {str(e)}")
            return None
    
    def forward_fill_fundamentals(self, price_data: pd.DataFrame, 
                                  fundamentals: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill fundamental data to match daily price data.
        Fundamental data is typically quarterly, so we carry values forward.
        
        Args:
            price_data: Daily price data with date index
            fundamentals: Quarterly fundamental data
            
        Returns:
            DataFrame with daily frequency and forward-filled fundamentals
        """
        if fundamentals.empty:
            return price_data
        
        # Get unique tickers
        tickers = fundamentals['ticker'].unique()
        
        result_dfs = []
        
        for ticker in tickers:
            # Get ticker-specific data
            ticker_fundamentals = fundamentals[fundamentals['ticker'] == ticker].copy()
            ticker_fundamentals = ticker_fundamentals.set_index('date').sort_index()
            
            # Remove ticker column before reindexing
            ticker_fundamentals = ticker_fundamentals.drop(columns=['ticker'], errors='ignore')
            
            # Get date range from price data for this ticker
            if isinstance(price_data.index, pd.MultiIndex):
                ticker_dates = price_data.xs(ticker, level='ticker').index
            else:
                ticker_dates = price_data.index
            
            # Reindex to daily frequency and forward fill
            ticker_fundamentals_daily = ticker_fundamentals.reindex(
                ticker_dates, method='ffill'
            )
            
            # Add ticker column back
            ticker_fundamentals_daily['ticker'] = ticker
            ticker_fundamentals_daily = ticker_fundamentals_daily.reset_index()
            ticker_fundamentals_daily = ticker_fundamentals_daily.rename(columns={'index': 'date'})
            
            result_dfs.append(ticker_fundamentals_daily)
        
        result = pd.concat(result_dfs, axis=0, ignore_index=True)
        return result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    fetcher = FundamentalsDataFetcher()
    tickers = ['AAPL', 'MSFT']
    df = fetcher.fetch_fundamentals(tickers)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")

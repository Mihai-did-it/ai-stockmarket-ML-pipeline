"""
Data Ingestion Module
Fetches historical price data, fundamentals, and news sentiment.
"""

from .price_data import PriceDataFetcher
from .fundamentals import FundamentalsDataFetcher
from .sentiment import SentimentDataFetcher

__all__ = [
    'PriceDataFetcher',
    'FundamentalsDataFetcher',
    'SentimentDataFetcher'
]

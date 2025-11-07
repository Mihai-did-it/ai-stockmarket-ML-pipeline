"""
Sentiment Data Fetcher
Retrieves and analyzes news sentiment using NLP models.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
from pathlib import Path
import requests

# Sentiment analysis
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available, sentiment analysis will be limited")

# News APIs
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


class SentimentDataFetcher:
    """Fetches news articles and computes sentiment scores."""
    
    def __init__(self, news_api_key: Optional[str] = None, 
                 cache_dir: str = 'data/raw',
                 sentiment_model: str = 'distilbert-base-uncased-finetuned-sst-2-english'):
        """
        Initialize sentiment data fetcher.
        
        Args:
            news_api_key: API key for NewsAPI
            cache_dir: Directory to cache data
            sentiment_model: HuggingFace model for sentiment analysis
        """
        self.cache_dir = Path(cache_dir) / 'sentiment'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize news API
        if news_api_key and NEWSAPI_AVAILABLE:
            self.news_client = NewsApiClient(api_key=news_api_key)
        else:
            self.news_client = None
            logger.warning("NewsAPI not initialized - using mock data")
        
        # Initialize sentiment analyzer
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading sentiment model: {sentiment_model}")
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=sentiment_model,
                    truncation=True,
                    max_length=512
                )
            except Exception as e:
                logger.error(f"Error loading sentiment model: {str(e)}")
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
    
    def fetch_sentiment_data(self, tickers: List[str], 
                            start_date: str, 
                            end_date: Optional[str] = None,
                            lookback_days: int = 30) -> pd.DataFrame:
        """
        Fetch news sentiment data for tickers.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date
            end_date: End date (default: today)
            lookback_days: Days to look back for news
            
        Returns:
            DataFrame with sentiment scores per ticker per day
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        all_sentiment = []
        
        for ticker in tickers:
            logger.info(f"Fetching sentiment for {ticker}")
            
            try:
                ticker_sentiment = self._fetch_ticker_sentiment(
                    ticker, start_date, end_date, lookback_days
                )
                
                if ticker_sentiment is not None and not ticker_sentiment.empty:
                    ticker_sentiment['ticker'] = ticker
                    all_sentiment.append(ticker_sentiment)
                    
            except Exception as e:
                logger.error(f"Error fetching sentiment for {ticker}: {str(e)}")
                continue
        
        if not all_sentiment:
            logger.warning("No sentiment data fetched, returning empty DataFrame")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_sentiment, axis=0, ignore_index=True)
        logger.info(f"Successfully fetched sentiment for {len(all_sentiment)} tickers")
        
        return combined_df
    
    def _fetch_ticker_sentiment(self, ticker: str, start_date: str, 
                               end_date: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Fetch and analyze sentiment for a single ticker."""
        
        # Fetch news articles
        articles = self._fetch_news_articles(ticker, start_date, end_date)
        
        if not articles:
            logger.warning(f"No articles found for {ticker}")
            return self._create_empty_sentiment_df(start_date, end_date)
        
        # Analyze sentiment for each article
        sentiment_data = []
        
        for article in articles:
            try:
                title = article.get('title', '')
                description = article.get('description', '')
                published_at = article.get('publishedAt', article.get('published_at', ''))
                
                # Combine title and description for analysis
                text = f"{title}. {description}".strip()
                
                if not text or len(text) < 10:
                    continue
                
                # Analyze sentiment
                sentiment = self._analyze_sentiment(text)
                
                sentiment_data.append({
                    'date': pd.to_datetime(published_at).date() if published_at else None,
                    'title': title,
                    'sentiment_label': sentiment['label'],
                    'sentiment_score': sentiment['score'],
                })
                
            except Exception as e:
                logger.debug(f"Error processing article: {str(e)}")
                continue
        
        if not sentiment_data:
            return self._create_empty_sentiment_df(start_date, end_date)
        
        # Convert to DataFrame
        df = pd.DataFrame(sentiment_data)
        df = df.dropna(subset=['date'])
        
        # Aggregate by date
        daily_sentiment = self._aggregate_daily_sentiment(df, start_date, end_date, lookback_days)
        
        return daily_sentiment
    
    def _fetch_news_articles(self, ticker: str, start_date: str, 
                            end_date: str) -> List[Dict]:
        """Fetch news articles from API."""
        
        if self.news_client is None:
            # Return mock data for demonstration
            logger.info(f"Using mock sentiment data for {ticker}")
            return self._generate_mock_articles(ticker, start_date, end_date)
        
        try:
            # Fetch from NewsAPI
            # Note: Free tier has limitations on date ranges and number of requests
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            all_articles = []
            
            # NewsAPI free tier limits to 1 month, so we need to chunk
            current_date = start_dt
            
            while current_date < end_dt:
                chunk_end = min(current_date + timedelta(days=30), end_dt)
                
                response = self.news_client.get_everything(
                    q=ticker,
                    from_param=current_date.strftime('%Y-%m-%d'),
                    to=chunk_end.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='publishedAt',
                    page_size=100
                )
                
                articles = response.get('articles', [])
                all_articles.extend(articles)
                
                current_date = chunk_end + timedelta(days=1)
            
            logger.info(f"Fetched {len(all_articles)} articles for {ticker}")
            return all_articles
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return []
    
    def _generate_mock_articles(self, ticker: str, start_date: str, 
                               end_date: str) -> List[Dict]:
        """Generate mock articles for demonstration."""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Sample headlines (positive, neutral, negative)
        headlines_pool = [
            f"{ticker} reports strong earnings, beats expectations",
            f"{ticker} stock rises on positive outlook",
            f"{ticker} announces new product launch",
            f"{ticker} shares steady amid market volatility",
            f"{ticker} trading flat in midday session",
            f"{ticker} maintains market position",
            f"{ticker} faces headwinds in challenging market",
            f"{ticker} stock falls on weak guidance",
            f"Analysts downgrade {ticker} amid concerns",
        ]
        
        articles = []
        
        # Generate 1-3 articles per week
        for date in date_range[::7]:  # Weekly
            num_articles = np.random.randint(1, 4)
            for _ in range(num_articles):
                articles.append({
                    'title': np.random.choice(headlines_pool),
                    'description': f"News about {ticker} stock performance and market conditions.",
                    'publishedAt': date.isoformat(),
                })
        
        return articles
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text."""
        if self.sentiment_analyzer is None:
            # Simple fallback: rule-based sentiment
            return self._simple_sentiment(text)
        
        try:
            result = self.sentiment_analyzer(text)[0]
            
            # Convert to standard format
            label = result['label'].upper()
            score = result['score']
            
            # Normalize: POSITIVE -> 1, NEGATIVE -> -1
            if 'POS' in label:
                sentiment_value = score
            else:
                sentiment_value = -score
            
            return {
                'label': label,
                'score': sentiment_value
            }
            
        except Exception as e:
            logger.debug(f"Sentiment analysis error: {str(e)}")
            return self._simple_sentiment(text)
    
    def _simple_sentiment(self, text: str) -> Dict:
        """Simple rule-based sentiment as fallback."""
        text_lower = text.lower()
        
        positive_words = ['beat', 'strong', 'rise', 'gain', 'up', 'growth', 'positive', 'success']
        negative_words = ['miss', 'weak', 'fall', 'loss', 'down', 'decline', 'negative', 'concern']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {'label': 'POSITIVE', 'score': 0.6}
        elif neg_count > pos_count:
            return {'label': 'NEGATIVE', 'score': -0.6}
        else:
            return {'label': 'NEUTRAL', 'score': 0.0}
    
    def _aggregate_daily_sentiment(self, df: pd.DataFrame, start_date: str,
                                   end_date: str, lookback_days: int) -> pd.DataFrame:
        """Aggregate sentiment by date with rolling windows."""
        
        # Group by date
        daily = df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).reset_index()
        
        daily.columns = ['date', 'sentiment_mean', 'sentiment_std', 'news_count']
        
        # Create full date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        full_df = pd.DataFrame({'date': date_range})
        full_df['date'] = full_df['date'].dt.date
        
        # Merge
        result = full_df.merge(daily, on='date', how='left')
        
        # Fill missing values
        result['news_count'] = result['news_count'].fillna(0)
        result['sentiment_mean'] = result['sentiment_mean'].fillna(0)
        result['sentiment_std'] = result['sentiment_std'].fillna(0)
        
        # Calculate rolling features
        result[f'news_volume_{lookback_days}d'] = result['news_count'].rolling(
            window=lookback_days, min_periods=1
        ).sum()
        
        result['sentiment_mean_7d'] = result['sentiment_mean'].rolling(
            window=7, min_periods=1
        ).mean()
        
        result['sentiment_mean_30d'] = result['sentiment_mean'].rolling(
            window=30, min_periods=1
        ).mean()
        
        return result
    
    def _create_empty_sentiment_df(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Create empty sentiment DataFrame with zeros."""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        df = pd.DataFrame({
            'date': [d.date() for d in date_range],
            'sentiment_mean': 0.0,
            'sentiment_std': 0.0,
            'news_count': 0,
            'news_volume_30d': 0,
            'sentiment_mean_7d': 0.0,
            'sentiment_mean_30d': 0.0,
        })
        
        return df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Get API key from environment
    api_key = os.getenv('NEWS_API_KEY')
    
    fetcher = SentimentDataFetcher(news_api_key=api_key)
    tickers = ['AAPL', 'TSLA']
    df = fetcher.fetch_sentiment_data(tickers, '2023-01-01', '2023-03-31')
    
    print(df.head(20))
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")

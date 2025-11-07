"""
Main Pipeline Orchestrator
Coordinates the entire ML pipeline from data ingestion to backtesting.

Personal project by Mihai Lache
Developed as part of Master's degree in AI and Machine Learning
"""

import argparse
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd

# Import pipeline components
from src.data_ingestion import PriceDataFetcher, FundamentalsDataFetcher, SentimentDataFetcher
from src.feature_engineering import TechnicalIndicatorCalculator, FeatureMerger, TargetGenerator
from src.modeling import ModelTrainer, ModelPredictor
from src.backtesting import Backtester
from src.utils import setup_logger, plot_feature_importance, plot_confusion_matrix, plot_backtest_results


class StockMLPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logger
        log_config = self.config.get('logging', {})
        self.logger = setup_logger(
            name='stock_ml_pipeline',
            level=log_config.get('level', 'INFO'),
            log_dir=log_config.get('log_dir', 'logs')
        )
        
        self.logger.info("=" * 80)
        self.logger.info("Stock Market ML Pipeline Initialized")
        self.logger.info("=" * 80)
        
        # Initialize components
        self.price_fetcher = None
        self.fundamentals_fetcher = None
        self.sentiment_fetcher = None
        self.indicator_calculator = None
        self.feature_merger = None
        self.target_generator = None
        self.model_trainer = None
        
    def run_full_pipeline(self) -> None:
        """Run the complete pipeline end-to-end."""
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING FULL PIPELINE")
        self.logger.info("=" * 80 + "\n")
        
        # Step 1: Data Ingestion
        self.logger.info("STEP 1: Data Ingestion")
        price_data = self.fetch_price_data()
        fundamentals = self.fetch_fundamentals()
        sentiment = self.fetch_sentiment()
        
        # Step 2: Feature Engineering
        self.logger.info("\nSTEP 2: Feature Engineering")
        features_df = self.engineer_features(price_data, fundamentals, sentiment)
        
        # Step 3: Target Generation
        self.logger.info("\nSTEP 3: Target Generation")
        data_with_targets = self.generate_targets(features_df)
        
        # Step 4: Train/Val/Test Split
        self.logger.info("\nSTEP 4: Data Splitting")
        train_df, val_df, test_df = self.split_data(data_with_targets)
        
        # Step 5: Model Training
        self.logger.info("\nSTEP 5: Model Training")
        self.train_model(train_df, val_df)
        
        # Step 6: Model Evaluation
        self.logger.info("\nSTEP 6: Model Evaluation")
        metrics = self.evaluate_model(test_df)
        
        # Step 7: Backtesting
        self.logger.info("\nSTEP 7: Backtesting")
        backtest_results = self.run_backtest(test_df, price_data)
        
        # Step 8: Save Results
        self.logger.info("\nSTEP 8: Saving Results")
        self.save_results(metrics, backtest_results)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PIPELINE COMPLETE!")
        self.logger.info("=" * 80 + "\n")
    
    def fetch_price_data(self) -> pd.DataFrame:
        """Fetch historical price data."""
        
        data_config = self.config['data']
        
        self.price_fetcher = PriceDataFetcher(
            cache_dir=data_config.get('cache_dir', 'data/raw'),
            use_cache=data_config.get('use_cache', True),
            cache_expiry_days=data_config.get('cache_expiry_days', 7)
        )
        
        tickers = data_config['tickers']
        start_date = data_config['start_date']
        end_date = data_config.get('end_date', None)
        
        self.logger.info(f"Fetching price data for {len(tickers)} tickers")
        self.logger.info(f"Date range: {start_date} to {end_date or 'today'}")
        
        price_data = self.price_fetcher.fetch_data(tickers, start_date, end_date)
        
        self.logger.info(f"Price data shape: {price_data.shape}")
        
        # Save raw data
        output_path = Path('data/processed/price_data.parquet')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        price_data.to_parquet(output_path)
        self.logger.info(f"Price data saved to {output_path}")
        
        return price_data
    
    def fetch_fundamentals(self) -> pd.DataFrame:
        """Fetch fundamental data."""
        
        data_config = self.config['data']
        
        self.fundamentals_fetcher = FundamentalsDataFetcher(
            cache_dir=data_config.get('cache_dir', 'data/raw'),
            use_cache=data_config.get('use_cache', True)
        )
        
        tickers = data_config['tickers']
        start_date = data_config['start_date']
        
        self.logger.info(f"Fetching fundamentals for {len(tickers)} tickers")
        
        fundamentals = self.fundamentals_fetcher.fetch_fundamentals(tickers, start_date)
        
        self.logger.info(f"Fundamentals shape: {fundamentals.shape}")
        
        # Save
        if not fundamentals.empty:
            output_path = Path('data/processed/fundamentals.parquet')
            fundamentals.to_parquet(output_path)
            self.logger.info(f"Fundamentals saved to {output_path}")
        
        return fundamentals
    
    def fetch_sentiment(self) -> pd.DataFrame:
        """Fetch sentiment data."""
        
        data_config = self.config['data']
        
        # Get API key from environment
        news_api_key = os.getenv('NEWS_API_KEY')
        
        self.sentiment_fetcher = SentimentDataFetcher(
            news_api_key=news_api_key,
            cache_dir=data_config.get('cache_dir', 'data/raw')
        )
        
        tickers = data_config['tickers']
        start_date = data_config['start_date']
        end_date = data_config.get('end_date', None)
        
        self.logger.info(f"Fetching sentiment for {len(tickers)} tickers")
        
        sentiment = self.sentiment_fetcher.fetch_sentiment_data(tickers, start_date, end_date)
        
        self.logger.info(f"Sentiment shape: {sentiment.shape}")
        
        # Save
        if not sentiment.empty:
            output_path = Path('data/processed/sentiment.parquet')
            sentiment.to_parquet(output_path)
            self.logger.info(f"Sentiment saved to {output_path}")
        
        return sentiment
    
    def engineer_features(self, price_data: pd.DataFrame,
                         fundamentals: pd.DataFrame,
                         sentiment: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data."""
        
        # Calculate technical indicators
        self.logger.info("Calculating technical indicators")
        self.indicator_calculator = TechnicalIndicatorCalculator(use_talib=False)
        price_with_technicals = self.indicator_calculator.calculate_all_indicators(price_data)
        
        # Merge all features
        self.logger.info("Merging all features")
        self.feature_merger = FeatureMerger()
        merged_features = self.feature_merger.merge_all_features(
            price_with_technicals, fundamentals, sentiment
        )
        
        # Add lag features if configured
        feature_config = self.config['features']
        if feature_config.get('lag_features', {}).get('enabled', False):
            lags = feature_config['lag_features']['lags']
            # Select key columns for lagging
            lag_cols = ['close', 'volume', 'rsi_14', 'macd']
            lag_cols = [c for c in lag_cols if c in merged_features.columns]
            
            self.logger.info(f"Adding lag features: {len(lag_cols)} features x {len(lags)} lags")
            merged_features = self.feature_merger.add_lag_features(merged_features, lag_cols, lags)
        
        # Add rolling features if configured
        if feature_config.get('rolling_features', {}).get('enabled', False):
            windows = feature_config['rolling_features']['windows']
            rolling_cols = ['close', 'volume']
            rolling_cols = [c for c in rolling_cols if c in merged_features.columns]
            
            self.logger.info(f"Adding rolling features: {len(rolling_cols)} features x {len(windows)} windows")
            merged_features = self.feature_merger.add_rolling_features(merged_features, rolling_cols, windows)
        
        # Clean features
        self.logger.info("Cleaning features")
        merged_features = self.feature_merger.clean_features(merged_features)
        
        self.logger.info(f"Final feature set shape: {merged_features.shape}")
        
        # Save
        output_path = Path('data/processed/features.parquet')
        merged_features.to_parquet(output_path)
        self.logger.info(f"Features saved to {output_path}")
        
        return merged_features
    
    def generate_targets(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate target variables."""
        
        target_config = self.config['features']['target']
        
        self.target_generator = TargetGenerator(
            horizon=target_config['horizon'],
            buy_threshold=target_config['threshold_buy'],
            sell_threshold=target_config['threshold_sell'],
            target_type=target_config['type']
        )
        
        data_with_targets = self.target_generator.generate_targets(features_df)
        
        self.logger.info(f"Data with targets shape: {data_with_targets.shape}")
        
        # Save
        output_path = Path('data/processed/data_with_targets.parquet')
        data_with_targets.to_parquet(output_path)
        self.logger.info(f"Data with targets saved to {output_path}")
        
        return data_with_targets
    
    def split_data(self, data_with_targets: pd.DataFrame):
        """Split data into train/val/test sets."""
        
        model_config = self.config['model']
        
        train_df, val_df, test_df = self.target_generator.create_time_based_split(
            data_with_targets,
            train_size=model_config['train_size'],
            val_size=model_config['validation_size'],
            test_size=model_config['test_size']
        )
        
        self.logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        
        return train_df, val_df, test_df
    
    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Train the model."""
        
        # Split features and targets
        X_train, y_train = self.target_generator.split_features_targets(train_df)
        X_val, y_val = self.target_generator.split_features_targets(val_df)
        
        self.logger.info(f"Training features: {X_train.shape}, targets: {y_train.shape}")
        
        # Initialize trainer
        self.model_trainer = ModelTrainer(self.config['model'])
        
        # Train
        self.model_trainer.train(X_train, y_train, X_val, y_val)
        
        # Save model
        model_path = Path('models/stock_model.pkl')
        self.model_trainer.save_model(str(model_path))
        
        # Save feature importance plot
        if self.model_trainer.feature_importance is not None:
            plot_path = Path('results/feature_importance.png')
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_feature_importance(self.model_trainer.feature_importance, save_path=str(plot_path))
    
    def evaluate_model(self, test_df: pd.DataFrame) -> dict:
        """Evaluate the model."""
        
        X_test, y_test = self.target_generator.split_features_targets(test_df)
        
        self.logger.info(f"Test features: {X_test.shape}, targets: {y_test.shape}")
        
        # Evaluate
        metrics = self.model_trainer.evaluate(X_test, y_test)
        
        # Save confusion matrix plot
        plot_path = Path('results/confusion_matrix.png')
        plot_confusion_matrix(metrics['confusion_matrix'], save_path=str(plot_path))
        
        # Save metrics
        metrics_df = pd.DataFrame([{k: v for k, v in metrics.items() 
                                   if not isinstance(v, (dict, list, type(None)))}])
        metrics_df.to_csv('results/test_metrics.csv', index=False)
        
        return metrics
    
    def run_backtest(self, test_df: pd.DataFrame, price_data: pd.DataFrame) -> dict:
        """Run backtesting."""
        
        X_test, y_test = self.target_generator.split_features_targets(test_df)
        
        # Get predictions
        predictions, probabilities = self.model_trainer.predict(X_test)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'prediction': predictions,
            'confidence': probabilities.max(axis=1),
            'prob_sell': probabilities[:, 0],
            'prob_hold': probabilities[:, 1],
            'prob_buy': probabilities[:, 2],
        }, index=X_test.index)
        
        # Run backtest
        backtester = Backtester(self.config['backtesting'])
        
        # Get test period price data
        if isinstance(price_data.index, pd.MultiIndex):
            test_dates = X_test.index.get_level_values('date').unique()
            test_price_data = price_data[price_data.index.get_level_values('date').isin(test_dates)]
        else:
            test_price_data = price_data.loc[X_test.index]
        
        results = backtester.run_backtest(predictions_df, test_price_data)
        
        # Plot results
        plot_path = Path('results/backtest_results.png')
        plot_backtest_results(results['portfolio_history'], save_path=str(plot_path))
        
        # Save metrics
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv('results/backtest_metrics.csv', index=False)
        
        return results
    
    def save_results(self, metrics: dict, backtest_results: dict) -> None:
        """Save final results summary."""
        
        summary = {
            'Model Performance': {
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'F1 Score': f"{metrics['f1']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}" if metrics.get('roc_auc') else 'N/A',
            },
            'Backtest Performance': {
                'Total Return': f"{backtest_results['metrics']['total_return']:.2%}",
                'Sharpe Ratio': f"{backtest_results['metrics']['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{backtest_results['metrics']['max_drawdown']:.2%}",
                'Win Rate': f"{backtest_results['metrics']['win_rate']:.2%}",
            }
        }
        
        # Save as YAML
        with open('results/summary.yaml', 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        self.logger.info("\nResults Summary:")
        self.logger.info(yaml.dump(summary, default_flow_style=False))


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Stock Market ML Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'train', 'predict', 'backtest'],
                       help='Pipeline mode')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = StockMLPipeline(config_path=args.config)
    
    if args.mode == 'full':
        pipeline.run_full_pipeline()
    else:
        print(f"Mode '{args.mode}' not fully implemented yet. Run with --mode full")


if __name__ == "__main__":
    main()

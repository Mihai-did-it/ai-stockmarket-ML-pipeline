# Setup & Getting Started Guide

This guide will help you get the Stock Market ML Pipeline up and running quickly.

## Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher
- Git installed
- 8GB+ RAM recommended
- Internet connection for data fetching

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Mihai-did-it/ai-stockmarket-ML-pipeline.git
cd ai-stockmarket-ML-pipeline
```

### 2. Install System Dependencies (Optional but Recommended)

**TA-Lib** provides faster technical indicator calculations:

**On macOS:**
```bash
brew install ta-lib
```

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libta-lib0-dev
```

**On Windows:**
Download the appropriate wheel file from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and install:
```bash
pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl
```

### 3. Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including:
- pandas, numpy, scikit-learn (data processing & ML)
- lightgbm, xgboost, catboost (ML models)
- yfinance (data fetching)
- transformers, torch (NLP sentiment analysis)
- matplotlib, seaborn, plotly (visualization)
- optuna (hyperparameter tuning)
- and more...

**Note**: If TA-Lib installation fails, the system will fall back to pandas_ta automatically.

### 5. Configure API Keys

The system can work without API keys (using mock data), but for real data you'll need:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

**Get Free API Keys:**

1. **Alpha Vantage** (Optional - for alternative data sources)
   - Sign up at: https://www.alphavantage.co/support/#api-key
   - Free tier: 5 API calls per minute, 500 per day

2. **Finnhub** (Optional - for fundamentals)
   - Sign up at: https://finnhub.io/register
   - Free tier: 60 API calls per minute

3. **NewsAPI** (Optional - for news sentiment)
   - Sign up at: https://newsapi.org/register
   - Free tier: 100 requests per day, 1 month of historical data

**Don't have API keys?** No problem! The system will use yfinance (no key required) and generate mock sentiment data for demonstration.

### 6. Verify Installation

Run a quick test:

```bash
python -c "import pandas; import lightgbm; import yfinance; print('✅ All core dependencies installed successfully!')"
```

## Quick Start

### Test Run (Recommended for First Time)

Start with a small test to verify everything works:

1. Edit `config.yaml` to use fewer tickers:
```yaml
data:
  tickers: [AAPL, MSFT]  # Just 2 stocks for testing
  start_date: "2022-01-01"  # Short time period
```

2. Disable hyperparameter tuning for faster testing:
```yaml
model:
  hyperparameter_tuning:
    enabled: false  # Change to false
```

3. Run the pipeline:
```bash
python main.py --mode full
```

**Expected Time**: 5-10 minutes for 2 tickers, 2 years of data

### Full Production Run

Once the test run succeeds, you can run the full pipeline:

1. Restore full configuration in `config.yaml`:
```yaml
data:
  tickers: [AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, BAC, WMT]
  start_date: "2015-01-01"

model:
  hyperparameter_tuning:
    enabled: true
    n_trials: 100
```

2. Run:
```bash
python main.py --mode full
```

**Expected Time**: 1-3 hours depending on:
- Number of tickers
- Date range
- Hyperparameter tuning trials
- Your hardware

## What to Expect

The pipeline will execute these steps:

1. **Data Ingestion** (5-10 min)
   - Fetches historical price data from yfinance
   - Downloads fundamental data
   - Collects news sentiment (or uses mock data)
   - Saves to `data/raw/` and `data/processed/`

2. **Feature Engineering** (5-10 min)
   - Calculates 60+ technical indicators
   - Merges all data sources with time alignment
   - Creates lag and rolling features
   - Generates Buy/Hold/Sell targets
   - Saves to `data/processed/features.parquet`

3. **Model Training** (30-120 min depending on tuning)
   - Splits data into train/val/test sets (70/15/15)
   - Trains LightGBM model
   - Optionally tunes hyperparameters with Optuna
   - Calibrates probabilities
   - Saves model to `models/stock_model.pkl`

4. **Model Evaluation** (1-2 min)
   - Calculates metrics on test set
   - Generates confusion matrix
   - Saves feature importance
   - Outputs to `results/`

5. **Backtesting** (2-5 min)
   - Simulates trading on test period
   - Calculates portfolio performance
   - Generates performance charts
   - Saves metrics to `results/`

## Reviewing Results

After completion, check the `results/` directory:

```
results/
├── feature_importance.png       # Top features
├── confusion_matrix.png         # Model predictions
├── backtest_results.png         # Portfolio performance over time
├── test_metrics.csv            # Classification metrics
├── backtest_metrics.csv        # Trading performance
└── summary.yaml                # Overall summary
```

Key metrics to look at:
- **Test Accuracy**: Should be > 50% (better than random)
- **F1 Score**: Balanced performance across classes
- **Sharpe Ratio**: Risk-adjusted returns (> 1 is good)
- **Max Drawdown**: Worst peak-to-trough loss
- **Win Rate**: Percentage of profitable trades

## Troubleshooting

### Issue: TA-Lib installation fails

**Solution**: The system will automatically fall back to pandas_ta. It's slower but works fine.

### Issue: "API rate limit exceeded"

**Solution**: 
- Reduce number of tickers
- Use shorter date range
- Enable caching (should be on by default)
- Wait and retry (rate limits reset after time)

### Issue: Out of memory errors

**Solution**:
- Reduce number of tickers
- Reduce date range
- Disable some rolling features
- Close other applications

### Issue: News sentiment returns empty data

**Solution**: This is expected if you don't have a NewsAPI key. The system will use mock sentiment data, which is fine for demonstration.

### Issue: Training is very slow

**Solution**:
- Disable hyperparameter tuning for faster training
- Reduce `n_trials` in config
- Use fewer features (disable lag/rolling features)

## Next Steps

1. **Experiment with Configuration**
   - Try different tickers (tech stocks, value stocks, ETFs)
   - Adjust buy/sell thresholds
   - Change prediction horizon (5 days vs 10 days)
   - Test different ML algorithms (XGBoost, CatBoost)

2. **Add Custom Features**
   - Edit `src/feature_engineering/technical_indicators.py`
   - Add your own technical indicators
   - They'll automatically be included in the model

3. **Improve Model Performance**
   - Tune hyperparameters more aggressively
   - Try different class imbalance strategies
   - Experiment with feature selection

4. **Explore the Code**
   - Each module has detailed docstrings
   - Run individual components for testing
   - Check out the modular architecture

## Getting Help

If you encounter issues:

1. Check the logs in `logs/` directory
2. Review error messages carefully
3. Ensure all dependencies are installed correctly
4. Try the test run with minimal configuration first
5. Open an issue on GitHub with:
   - Error message
   - Your configuration
   - Steps to reproduce

## Tips for Success

✅ **Do:**
- Start with test run on limited data
- Review logs for warnings/errors
- Experiment with configuration
- Read the code comments and docstrings
- Check intermediate outputs in `data/processed/`

❌ **Don't:**
- Run on too many tickers initially (start with 2-3)
- Use data before 2010 (quality issues)
- Shuffle time-series data (this breaks temporal dependencies)
- Trade real money based on this model without proper validation

## Resources

- [Main README](README.md) - Full documentation
- [Configuration Guide](config.yaml) - All settings explained
- [TA-Lib Documentation](https://mrjbq7.github.io/ta-lib/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)

---

Happy coding! 🚀 If this helps you land your dream job, give the repo a star! ⭐

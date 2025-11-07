# 🚀 AI Stock Market Analysis & Prediction System

A production-grade machine learning pipeline for stock market analysis that predicts **Buy / Hold / Sell** signals using comprehensive data sources including price history, technical indicators, fundamental metrics, and news sentiment analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Model Performance](#model-performance)
- [Extending the System](#extending-the-system)
- [Interview Talking Points](#interview-talking-points)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

This project demonstrates **production-ready machine learning engineering** for financial markets. It showcases:

- **End-to-end ML pipeline**: Data ingestion → Feature engineering → Model training → Backtesting
- **Multi-source data integration**: Price data, fundamentals, news sentiment with NLP
- **Time-series best practices**: No look-ahead bias, time-based splitting, proper validation
- **Production-grade code**: Modular design, comprehensive logging, error handling, caching
- **Advanced ML techniques**: Ensemble models (LightGBM/XGBoost/CatBoost), hyperparameter optimization, model calibration

**Target Audience**: This is designed as a portfolio project for ML/AI engineers, data scientists, and quantitative analysts to demonstrate real-world ML skills in financial applications.

---

## ✨ Key Features

### Data Ingestion
- **Multi-source fetching**: Yahoo Finance (via yfinance), Alpha Vantage, Finnhub, NewsAPI
- **Smart caching**: Parquet-based caching with configurable expiry
- **Robust error handling**: Graceful degradation, retry logic

### Feature Engineering
- **60+ Technical Indicators**:
  - Trend: SMA, EMA, MACD, ADX, Aroon, Parabolic SAR
  - Momentum: RSI, Stochastic, CCI, Williams %R, ROC
  - Volatility: Bollinger Bands, ATR, Keltner Channels
  - Volume: OBV, VWAP, CMF, MFI, Force Index
- **Fundamental Metrics**: P/E, EPS, ROE, ROA, debt ratios, margins, cash flow
- **NLP Sentiment Analysis**: Transformer-based (DistilBERT) news sentiment scoring
- **Time-series features**: Lags, rolling aggregations (mean, std, min, max)
- **No look-ahead bias**: Careful alignment ensures no future data leaks

### Machine Learning
- **Multiple Algorithms**: LightGBM (default), XGBoost, CatBoost
- **Hyperparameter Tuning**: Optuna-based Bayesian optimization
- **Class Imbalance Handling**: SMOTE, undersampling, class weights
- **Model Calibration**: Isotonic/sigmoid calibration for better probability estimates
- **Feature Selection**: Importance-based pruning
- **Cross-validation**: Time-series aware CV

### Backtesting
- **Realistic simulation**: Transaction costs, slippage, position limits
- **Portfolio management**: Equal weighting, risk parity options
- **Comprehensive metrics**: 
  - Returns: Total, annualized, CAGR
  - Risk: Sharpe, Sortino, max drawdown, Calmar ratio
  - Trading: Win rate, profit factor, trade count

### Production Features
- **Modular architecture**: Easy to extend, test, and maintain
- **Comprehensive logging**: Detailed execution tracking
- **Configuration-driven**: YAML config for easy experimentation
- **Visualization**: Feature importance, confusion matrices, backtest charts
- **Type hints**: Clean, documented code throughout

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA INGESTION                          │
├─────────────────────────────────────────────────────────────┤
│  Price Data    │  Fundamentals  │  News Sentiment (NLP)    │
│  (yfinance)    │  (yfinance)    │  (NewsAPI + DistilBERT)  │
└──────┬──────────────────┬─────────────────┬────────────────┘
       │                  │                 │
       v                  v                 v
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                        │
├─────────────────────────────────────────────────────────────┤
│  • Technical Indicators (60+ features)                       │
│  • Time-series features (lags, rolling stats)                │
│  • Feature alignment & forward-filling                       │
│  • Target generation (Buy=2, Hold=1, Sell=0)                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           v
┌─────────────────────────────────────────────────────────────┐
│                    TIME-BASED SPLIT                          │
├─────────────────────────────────────────────────────────────┤
│  Train (70%)  │  Validation (15%)  │  Test (15%)            │
│  Never shuffle time series data!                             │
└──────┬──────────────────┬─────────────────┬────────────────┘
       │                  │                 │
       v                  v                 v
┌─────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                            │
├─────────────────────────────────────────────────────────────┤
│  • Algorithm: LightGBM / XGBoost / CatBoost                  │
│  • Hyperparameter tuning (Optuna)                            │
│  • Class imbalance handling (SMOTE)                          │
│  • Calibration (Isotonic)                                    │
│  • Feature importance analysis                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           v
┌─────────────────────────────────────────────────────────────┐
│                      EVALUATION                              │
├─────────────────────────────────────────────────────────────┤
│  • Classification metrics (Accuracy, F1, ROC AUC)            │
│  • Per-class precision/recall                                │
│  • Confusion matrix visualization                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           v
┌─────────────────────────────────────────────────────────────┐
│                     BACKTESTING                              │
├─────────────────────────────────────────────────────────────┤
│  • Portfolio simulation (transaction costs, slippage)        │
│  • Performance metrics (Sharpe, drawdown, win rate)          │
│  • Trade history & visualization                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Installation

### Prerequisites
- Python 3.8+
- macOS / Linux / Windows

### Step 1: Clone Repository
```bash
git clone https://github.com/Mihai-did-it/ai-stockmarket-ML-pipeline.git
cd ai-stockmarket-ML-pipeline
```

### Step 2: Install TA-Lib (Optional but Recommended)

**macOS:**
```bash
brew install ta-lib
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libta-lib0-dev
```

**Windows:**
Download pre-built wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

### Step 3: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Setup API Keys
Copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
```

Edit `.env`:
```bash
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

**Free API Keys:**
- [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
- [Finnhub](https://finnhub.io/)
- [NewsAPI](https://newsapi.org/)

---

## 🚀 Quick Start

### Run Full Pipeline
```bash
python main.py --mode full
```

This will:
1. Fetch data for all tickers in `config.yaml`
2. Calculate technical indicators and features
3. Generate Buy/Hold/Sell targets
4. Train a LightGBM model with hyperparameter tuning
5. Evaluate on test set
6. Run backtest simulation
7. Save results to `results/` directory

### Results Location
- `results/feature_importance.png` - Top features
- `results/confusion_matrix.png` - Model predictions
- `results/backtest_results.png` - Portfolio performance
- `results/test_metrics.csv` - Classification metrics
- `results/backtest_metrics.csv` - Trading performance
- `results/summary.yaml` - Overall summary

---

## 📁 Project Structure

```
ai-stockmarket-ML-pipeline/
├── config.yaml                  # Main configuration file
├── main.py                      # Pipeline orchestrator
├── requirements.txt             # Python dependencies
├── .env.example                 # API keys template
├── .gitignore
│
├── data/
│   ├── raw/                     # Cached raw data
│   └── processed/               # Processed datasets
│
├── models/                      # Saved trained models
│
├── results/                     # Outputs (plots, metrics)
│
├── logs/                        # Execution logs
│
├── src/
│   ├── data_ingestion/
│   │   ├── price_data.py        # Historical OHLCV fetching
│   │   ├── fundamentals.py      # Financial statements
│   │   └── sentiment.py         # News + NLP sentiment
│   │
│   ├── feature_engineering/
│   │   ├── technical_indicators.py  # 60+ indicators
│   │   ├── feature_merger.py        # Data alignment
│   │   └── target_generator.py      # Label creation
│   │
│   ├── modeling/
│   │   ├── trainer.py           # Model training & tuning
│   │   └── predictor.py         # Inference
│   │
│   ├── backtesting/
│   │   ├── backtester.py        # Simulation engine
│   │   └── portfolio.py         # Portfolio management
│   │
│   └── utils/
│       ├── logger.py            # Logging setup
│       └── visualizations.py    # Plotting functions
│
└── notebooks/                   # Jupyter notebooks (exploration)
```

---

## 🔧 Pipeline Components

### 1. Data Ingestion (`src/data_ingestion/`)

**Price Data Fetcher** (`price_data.py`)
- Fetches OHLCV data from yfinance
- Implements caching to avoid re-downloading
- Handles multiple tickers efficiently

**Fundamentals Fetcher** (`fundamentals.py`)
- Retrieves quarterly financial statements
- Calculates derived metrics (margins, ratios)
- Forward-fills to daily frequency

**Sentiment Analyzer** (`sentiment.py`)
- Fetches news articles from NewsAPI
- Uses DistilBERT for sentiment classification
- Aggregates sentiment scores daily

### 2. Feature Engineering (`src/feature_engineering/`)

**Technical Indicator Calculator** (`technical_indicators.py`)
- Works with TA-Lib (fast) or pandas_ta (fallback)
- Calculates 60+ indicators across 4 categories
- Handles multi-ticker DataFrames

**Feature Merger** (`feature_merger.py`)
- Time-aligns different data frequencies
- Adds lag and rolling features
- Cleans NaN and inf values

**Target Generator** (`target_generator.py`)
- Creates labels based on future returns
- Configurable thresholds for Buy/Sell
- Time-based train/val/test splitting

### 3. Modeling (`src/modeling/`)

**Model Trainer** (`trainer.py`)
- Supports LightGBM, XGBoost, CatBoost
- Optuna hyperparameter optimization
- SMOTE for class imbalance
- Probability calibration
- Feature importance extraction

**Model Predictor** (`predictor.py`)
- Loads trained models
- Makes predictions with confidence scores
- Filters low-confidence predictions

### 4. Backtesting (`src/backtesting/`)

**Backtester** (`backtester.py`)
- Simulates trading on test data
- Calculates performance metrics
- Realistic cost modeling

**Portfolio Manager** (`portfolio.py`)
- Tracks positions and cash
- Executes buy/sell orders
- Records trade history

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Data Settings
```yaml
data:
  tickers: [AAPL, MSFT, GOOGL, ...]  # Stocks to analyze
  start_date: "2015-01-01"
  end_date: null  # null = today
```

### Model Settings
```yaml
model:
  algorithm: lightgbm  # lightgbm, xgboost, catboost
  hyperparameter_tuning:
    enabled: true
    n_trials: 100
  handle_imbalance:
    method: smote  # smote, undersampling, class_weights
```

### Backtesting Settings
```yaml
backtesting:
  initial_capital: 100000
  transaction_cost: 0.001  # 0.1%
  confidence_threshold: 0.6  # Only trade if model is confident
```

---

## 📊 Usage Examples

### Example 1: Train on Custom Tickers
Edit `config.yaml`:
```yaml
data:
  tickers: [NVDA, AMD, INTC]
  start_date: "2020-01-01"
```

Run:
```bash
python main.py --mode full
```

### Example 2: Use Different Model
Edit `config.yaml`:
```yaml
model:
  algorithm: xgboost
```

### Example 3: Adjust Trading Strategy
Edit `config.yaml`:
```yaml
backtesting:
  strategy:
    confidence_threshold: 0.7  # More conservative
    max_positions: 5  # Fewer concurrent positions
```

---

## 📈 Model Performance

**Expected Metrics** (based on typical results):

| Metric | Value |
|--------|-------|
| Accuracy | 55-65% |
| F1 Score | 0.50-0.60 |
| ROC AUC | 0.60-0.70 |
| Sharpe Ratio | 0.5-1.5 |
| Max Drawdown | -15% to -25% |

**Note**: Stock market prediction is inherently noisy. These metrics demonstrate the model learns meaningful patterns, but perfect prediction is impossible.

---

## 🔨 Extending the System

### Add New Data Source
1. Create new fetcher in `src/data_ingestion/`
2. Implement `fetch_data()` method
3. Update `main.py` to include new data
4. Merge in `FeatureMerger`

### Add New Technical Indicator
1. Add calculation method in `TechnicalIndicatorCalculator`
2. Call it in `calculate_all_indicators()`
3. Indicator automatically included in features

### Add New Model Algorithm
1. Add algorithm case in `ModelTrainer._get_default_params()`
2. Add training logic in `ModelTrainer._train_model()`
3. Set `model.algorithm` in config

### Implement Different Trading Strategy
1. Modify `Backtester._generate_signals()` for custom logic
2. Adjust position sizing in `Portfolio.buy()`

---

## 💼 Interview Talking Points

When discussing this project in interviews, emphasize:

### 1. **Production-Grade Engineering**
- "I designed this with modular, maintainable code following SOLID principles"
- "Implemented comprehensive logging, error handling, and caching for production use"
- "Used configuration files to separate code from parameters for easy experimentation"

### 2. **Time-Series Best Practices**
- "Ensured no look-ahead bias through careful feature alignment and time-based splitting"
- "Used time-series cross-validation instead of random K-fold"
- "Forward-filled fundamental data appropriately to match daily prices"

### 3. **Advanced ML Techniques**
- "Implemented Bayesian hyperparameter optimization with Optuna"
- "Handled class imbalance with SMOTE and explored multiple strategies"
- "Calibrated model probabilities for better confidence estimates"
- "Used ensemble methods (LightGBM) for robust predictions"

### 4. **Multi-Modal Data Integration**
- "Combined structured data (prices, fundamentals) with unstructured data (news sentiment)"
- "Applied NLP transformer models (DistilBERT) for sentiment analysis"
- "Engineered 60+ technical indicators covering trend, momentum, volatility, and volume"

### 5. **Realistic Evaluation**
- "Built a backtesting framework with realistic transaction costs and slippage"
- "Calculated comprehensive performance metrics beyond simple accuracy"
- "Acknowledged that stock prediction is noisy - focused on learning patterns, not perfect prediction"

### 6. **Scalability & Extensibility**
- "Designed for easy addition of new data sources, features, and models"
- "Can scale to hundreds of stocks with efficient caching"
- "Modular architecture allows testing individual components"

---

## 🚀 Future Improvements

Ideas to discuss for extending the project:

1. **Deep Learning Models**
   - LSTM/GRU for sequential pattern learning
   - Transformer models for time-series (Temporal Fusion Transformer)
   - Combine with gradient boosting in ensemble

2. **Alternative Data Sources**
   - Social media sentiment (Twitter, Reddit)
   - Options flow data
   - Insider trading data
   - Economic indicators

3. **Advanced Strategies**
   - Reinforcement learning (PPO, A3C) for trading
   - Portfolio optimization (Markowitz, Black-Litterman)
   - Risk parity position sizing
   - Stop-loss and take-profit optimization

4. **Production Deployment**
   - Real-time data streaming (Kafka)
   - Model serving (FastAPI, BentoML)
   - MLOps pipeline (MLflow, Kubeflow)
   - Monitoring and retraining automation

5. **Feature Enhancement**
   - Sector/industry relative features
   - Macroeconomic indicators
   - Cross-asset correlations
   - Order book data (for liquid stocks)

6. **Risk Management**
   - Value at Risk (VaR) calculations
   - Position sizing based on Kelly criterion
   - Drawdown-based stop-outs
   - Correlation-aware diversification

---

## 📚 References & Learning Resources

- **Books**:
  - "Advances in Financial Machine Learning" by Marcos López de Prado
  - "Machine Learning for Algorithmic Trading" by Stefan Jansen
  - "Quantitative Trading" by Ernest P. Chan

- **Papers**:
  - "XGBoost: A Scalable Tree Boosting System"
  - "SMOTE: Synthetic Minority Over-sampling Technique"
  - "Calibrating Predictions to Decisions"

- **Libraries Used**:
  - [LightGBM](https://lightgbm.readthedocs.io/)
  - [Optuna](https://optuna.readthedocs.io/)
  - [Transformers (Hugging Face)](https://huggingface.co/docs/transformers/)
  - [TA-Lib](https://mrjbq7.github.io/ta-lib/)

---

## 📝 License

This project is for educational and portfolio purposes. Not financial advice.

---

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome! Feel free to:
- Open issues for bugs or ideas
- Submit pull requests with improvements
- Star the repo if you find it useful!

---

## 👤 Author

**Mihai Acherman**
- Building this to demonstrate ML engineering skills for my AI/ML Master's degree
- Focus: Production-grade ML systems for finance
- Contact: [Add your contact info]

---

## ⚠️ Disclaimer

This software is for **educational purposes only**. It is not financial advice and should not be used for actual trading without proper risk assessment. Past performance does not guarantee future results. Always do your own research and consult with financial professionals before making investment decisions.

---

**Good luck with your interviews! 🚀**
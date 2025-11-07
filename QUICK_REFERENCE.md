# ⚡ Quick Reference Card

> **Project by Mihai Lache** - Personal ML project for stock market analysis

## Essential Commands

### Setup
```bash
# Clone and setup
git clone <repo>
cd ai-stockmarket-ML-pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Install TA-Lib (optional)
brew install ta-lib  # macOS
```

### Run Pipeline
```bash
# Full pipeline
python main.py --mode full

# Test run (fast)
# Edit config.yaml: tickers=[AAPL], hyperparameter_tuning.enabled=false
python main.py --mode full
```

### Check Results
```bash
# View results
ls results/
cat results/summary.yaml

# View logs
tail -f logs/*.log
```

---

## Project Structure (Quick Map)

```
📦 Project Root
├── 📜 main.py              → Run this!
├── ⚙️ config.yaml          → Edit settings here
├── 📋 requirements.txt     
├── 🔐 .env.example         → Copy to .env, add API keys
│
├── 📂 src/                 → Source code
│   ├── data_ingestion/     → Fetch data
│   ├── feature_engineering/→ Create features  
│   ├── modeling/           → Train models
│   ├── backtesting/        → Test strategy
│   └── utils/              → Helpers
│
├── 📂 data/                → Cached data
├── 📂 models/              → Saved models
├── 📂 results/             → Outputs (CHECK HERE!)
└── 📂 logs/                → Debug info
```

---

## Configuration Cheat Sheet

### Quick Edit: `config.yaml`

**Change tickers:**
```yaml
data:
  tickers: [AAPL, MSFT, GOOGL]  # Your stocks
```

**Faster testing:**
```yaml
data:
  start_date: "2022-01-01"  # Shorter period

model:
  hyperparameter_tuning:
    enabled: false           # Skip tuning
    n_trials: 20             # Or fewer trials
```

**More conservative trading:**
```yaml
backtesting:
  strategy:
    confidence_threshold: 0.7  # Higher = fewer trades
```

---

## Key Metrics Explained

### Model Metrics
| Metric | Good | Meaning |
|--------|------|---------|
| **Accuracy** | >55% | % correct predictions |
| **F1 Score** | >0.50 | Balance of precision/recall |
| **ROC AUC** | >0.60 | Discrimination ability |

### Trading Metrics
| Metric | Good | Meaning |
|--------|------|---------|
| **Sharpe Ratio** | >1.0 | Risk-adjusted returns |
| **Max Drawdown** | <-20% | Worst loss from peak |
| **Win Rate** | >50% | % profitable trades |
| **Total Return** | >10%/yr | Overall profit |

---

## Troubleshooting Quick Fixes

**Problem: Out of memory**
```yaml
# In config.yaml, reduce:
data:
  tickers: [AAPL, MSFT]  # Fewer stocks
features:
  lag_features:
    enabled: false       # Disable lags
```

**Problem: API rate limit**
```yaml
# Enable caching:
data:
  use_cache: true
  cache_expiry_days: 30  # Cache longer
```

**Problem: Training too slow**
```yaml
model:
  hyperparameter_tuning:
    enabled: false
  # Or reduce trials:
    n_trials: 20
```

**Problem: Import errors**
```bash
# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## File Outputs Reference

After running, check these files:

**Essential:**
- `results/summary.yaml` - Overall results
- `results/feature_importance.png` - Top features
- `results/backtest_results.png` - Portfolio chart

**Detailed:**
- `results/test_metrics.csv` - Model performance
- `results/backtest_metrics.csv` - Trading stats
- `results/confusion_matrix.png` - Prediction matrix

**Data:**
- `data/processed/features.parquet` - All features
- `models/stock_model.pkl` - Trained model

**Debug:**
- `logs/*.log` - Execution logs

---

## Code Navigation

### To modify data sources:
→ `src/data_ingestion/price_data.py`

### To add indicators:
→ `src/feature_engineering/technical_indicators.py`
→ Add method in `_calculate_indicators_single_ticker()`

### To change model:
→ `config.yaml` - set `model.algorithm`
→ Or edit `src/modeling/trainer.py`

### To adjust strategy:
→ `src/backtesting/backtester.py`
→ Modify `_generate_signals()`

---

## Python API Quick Start

```python
# Use components programmatically

from src.data_ingestion import PriceDataFetcher
from src.feature_engineering import TechnicalIndicatorCalculator
from src.modeling import ModelTrainer

# Fetch data
fetcher = PriceDataFetcher()
data = fetcher.fetch_data(['AAPL'], '2020-01-01')

# Calculate indicators
calc = TechnicalIndicatorCalculator(use_talib=False)
features = calc.calculate_all_indicators(data)

# Train model (need X, y)
config = {'algorithm': 'lightgbm', ...}
trainer = ModelTrainer(config)
trainer.train(X_train, y_train, X_val, y_val)
```

---

## Git Workflow

```bash
# Initial commit
git add .
git commit -m "Initial AI stock market ML pipeline"

# Create .gitignore (already included!)
# Excludes: data/, models/, results/, logs/, venv/

# Push to GitHub
git remote add origin <your-repo-url>
git push -u origin main
```

---

## Interview Prep Checklist

**5 Minutes Before Interview:**
- [ ] Open project in IDE
- [ ] Have `results/` folder ready
- [ ] Know your accuracy & Sharpe ratio numbers
- [ ] Review architecture diagram
- [ ] Practice explaining one technical decision

**Key Numbers to Memorize:**
- Accuracy: ~60%
- Sharpe Ratio: ~1.2
- 60+ technical indicators
- 3 ML algorithms (LightGBM, XGBoost, CatBoost)
- 70/15/15 train/val/test split
- ~3000 lines of code

---

## Common Questions & Quick Answers

**Q: How long to run?**
A: Test (2 stocks): 5-10 min | Full (10 stocks): 1-3 hours

**Q: Do I need API keys?**
A: No for testing (uses yfinance + mock data) | Yes for production

**Q: What if TA-Lib fails?**
A: Auto falls back to pandas_ta (slower but works)

**Q: Can I use for real trading?**
A: NO! Educational only. Not financial advice.

**Q: GPU needed?**
A: No, runs on CPU (LightGBM is CPU-optimized)

---

## Extensions Priority

**Easy (1-2 hours):**
1. Add more tickers
2. Change buy/sell thresholds  
3. Try different ML algorithm
4. Adjust prediction horizon

**Medium (1-2 days):**
1. Add LSTM model
2. Implement portfolio optimization
3. Create dashboard
4. Add risk management rules

**Hard (1-2 weeks):**
1. Real-time API
2. MLOps pipeline
3. Reinforcement learning
4. Deep learning models

---

## Key Files to Read

**Start Here:**
1. `README.md` - Full documentation
2. `SETUP.md` - Installation guide
3. `main.py` - See how it all connects

**Deep Dive:**
4. `src/modeling/trainer.py` - ML training
5. `src/feature_engineering/technical_indicators.py` - Features
6. `config.yaml` - All settings

**Advanced:**
7. `PROJECT_SUMMARY.md` - Interview prep
8. `IMPROVEMENTS.md` - Extension ideas

---

## Performance Expectations

**Realistic:**
- Accuracy: 55-65% (vs 33% random for 3-class)
- Sharpe: 0.5-1.5 (vs ~0.3-0.5 for S&P 500)
- Drawdown: -15% to -25%

**Why not 95% accuracy?**
- Markets are noisy (efficient market hypothesis)
- Many factors are unpredictable
- We're learning patterns, not future-seeing

**This is GOOD!**
- Shows model learns something real
- Realistic expectations (not overpromising)
- Demonstrates understanding of limitations

---

## License & Disclaimer

📝 **License**: Educational/Portfolio use

⚠️ **Disclaimer**: NOT FINANCIAL ADVICE
- For learning purposes only
- Do not use for real trading
- Past performance ≠ future results
- Consult professionals before investing

---

## Quick Links

- 📚 Full README: [README.md](README.md)
- 🔧 Setup Guide: [SETUP.md](SETUP.md)
- 🎯 Project Summary: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- 💡 Improvements: [IMPROVEMENTS.md](IMPROVEMENTS.md)
- 🐛 Issues: GitHub Issues
- ⭐ Star: github.com/Mihai-did-it/ai-stockmarket-ML-pipeline

---

## Emergency Help

**Something broken?**
1. Check logs: `tail -f logs/*.log`
2. Read error message carefully
3. Google the error + library name
4. Check GitHub issues
5. Ask for help (provide error + config)

**Need to start over?**
```bash
rm -rf data/ models/ results/ logs/
python main.py --mode full
```

---

**Print this for quick reference during interviews! 📄**

Good luck! 🚀
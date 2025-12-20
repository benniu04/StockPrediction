# StockPrediction

Advanced stock prediction and trading system using machine learning with walk-forward validation and multi-stock training.

## 🚀 Features

- **80+ Normalized Features**: Return-based indicators that generalize across equities.
- **Walk-Forward Validation**: Simulates real trading conditions to prevent overfitting.
- **Multi-Stock Training**: Leverages patterns across multiple tickers (e.g., SPY, QQQ, AAPL).
- **Financial Metrics**: Optimizes for Sharpe ratio, win rate, and profit factor.
- **Model Ensemble**: Combines Random Forest, XGBoost, LightGBM, and GBM with a Logistic Regression meta-model.
- **Interactive Dashboard**: Real-time backtesting, metrics visualization, and next-day forecasts.

## 📁 Project Structure

- `features_engineering.py`: Technical indicators and return-based feature generation.
- `modeling.py`: Walk-forward validator and ensemble model architecture.
- `training.py`: Script for training on single or multiple stocks.
- `streamlit_dashboard.py`: Streamlit-based user interface.
- `models/`: Directory for saved `.pkl` model files and metrics.

## 🎓 Quick Start

### 1. Install Dependencies
```bash
brew install ta-lib # Required for TA-Lib
pip install -r requirements.txt
```

### 2. Set Up Environment
Create a `.env` file with your API keys:
```env
ALPHA_VANTAGE_API_KEY=your_key_here
```

### 3. Train and Launch
```bash
# Train on major ETFs
python training.py --mode multi --symbols SPY QQQ IWM DIA

# Launch the dashboard
streamlit run streamlit_dashboard.py
```

## 📈 Technical Pipeline

1. **Preprocessing**: Data download via `yfinance` and sentiment fetching via Alpha Vantage.
2. **Feature Engineering**: 80+ normalized features including volatility, momentum, and volume ratios.
3. **Training**: Expanding-window walk-forward validation with a 5-day gap to eliminate look-ahead bias.
4. **Ensemble**: Weighted stacking of multiple gradient-boosted models.
5. **Backtesting**: Transaction cost modeling and monthly performance analysis.

---

## ⚠️ Disclaimer
Educational and research purposes only. Trading stocks involves risk. Past performance does not guarantee future results.

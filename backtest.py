import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import talib
from datetime import datetime
from xgboost import XGBClassifier

# Parameters
SYMBOL = 'WMT'
start_date = '2010-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Backtesting period
backtest_start_date = '2024-01-01'
backtest_end_date = datetime.today().strftime('%Y-%m-%d')

# Download stock data
stock_data = yf.download(SYMBOL, start=start_date, end=end_date)

# Feature Engineering
stock_data['Prev Close'] = stock_data['Adj Close'].shift(1)
stock_data['EMA_100'] = talib.EMA(stock_data['Adj Close'], timeperiod=100)
stock_data['MACD'], stock_data['MACD_signal'], _ = talib.MACD(stock_data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
stock_data['Low_Close_Diff'] = stock_data['Adj Close'] - stock_data['Low']
stock_data['SMA_Low_50'] = talib.SMA(stock_data['Low'], timeperiod=50)
upper_band, middle_band, lower_band = talib.BBANDS(stock_data['Low'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
stock_data['Lower_BB'] = lower_band
stock_data['Upper_BB'] = upper_band
stock_data['Low_Close_Diff_Volume'] = stock_data['Low_Close_Diff'] * stock_data['Volume']

# Create "Tomorrow" column and classify direction
stock_data['Tomorrow_Open'] = stock_data['Open'].shift(-1)
stock_data['Direction'] = (stock_data['Tomorrow_Open'] > stock_data['Open']).astype(int)

# Drop rows with NaN values
stock_data.dropna(inplace=True)

# Filter stock_data for backtesting period
backtest_data = stock_data[(stock_data.index >= backtest_start_date) & (stock_data.index <= backtest_end_date)]

# Define features (X) and the target variable (y)
X = backtest_data[['Open', 'High', 'Low', 'Volume', 'Prev Close', 'Low_Close_Diff', 'SMA_Low_50', 'Lower_BB', 'Upper_BB', 'EMA_100', 'Low_Close_Diff_Volume']]
y = backtest_data['Direction']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Load the trained scaler
scaler = joblib.load('scaler.pkl')
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train models
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Stack predictions for meta-model
rf_predictions_test = rf_model.predict_proba(X_test)[:, 1]
xgb_predictions_test = xgb_model.predict_proba(X_test)[:, 1]
stacked_test = np.column_stack((rf_predictions_test, xgb_predictions_test))

# Load the trained meta-model
meta_model = joblib.load('meta_model.pkl')

# Get final predictions using the meta-model
meta_model_probs = meta_model.predict_proba(stacked_test)[:, 1]
custom_threshold = 0.7
stacked_pred_custom = (meta_model_probs > custom_threshold).astype(int)

# Evaluate the model
precision = precision_score(y_test, stacked_pred_custom)
recall = recall_score(y_test, stacked_pred_custom)
f1 = f1_score(y_test, stacked_pred_custom)
accuracy = accuracy_score(y_test, stacked_pred_custom)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Accuracy: {accuracy:.2f}')

# Implement backtesting logic
initial_capital = 10000
cash = initial_capital
positions = 0
portfolio_value = [initial_capital]

# Backtesting loop
for i in range(len(y_test)):
    current_open_price = backtest_data.iloc[i]['Open']

    if stacked_pred_custom[i] == 1 and cash >= current_open_price:  # Buy
        shares_to_buy = cash // current_open_price
        cash -= shares_to_buy * current_open_price
        positions += shares_to_buy

    elif stacked_pred_custom[i] == 0 and positions > 0:  # Sell
        cash += positions * current_open_price
        positions = 0

    current_portfolio_value = cash + positions * current_open_price
    portfolio_value.append(current_portfolio_value)

# Plot portfolio value over time
plt.plot(backtest_data.index[-len(portfolio_value):], portfolio_value, label='Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

# Summary of backtest results
total_return = (portfolio_value[-1] - initial_capital) / initial_capital
annualized_return = ((1 + total_return) ** (252 / len(y_test))) - 1  # 252 trading days in a year
max_drawdown = min(portfolio_value) / initial_capital - 1

print(f'Total Return: {total_return * 100:.2f}%')
print(f'Annualized Return: {annualized_return * 100:.2f}%')
print(f'Maximum Drawdown: {max_drawdown * 100:.2f}%')
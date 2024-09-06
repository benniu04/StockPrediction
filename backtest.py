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

# Backtesting period (adjust these dates for different backtesting periods)
backtest_start_date = '2024-01-01'
backtest_end_date = datetime.today().strftime('%Y-%m-%d')

# Download stock data for the entire period
pd.set_option('display.max_columns', None)
stock_data = yf.download(SYMBOL, start=start_date, end=end_date)

# Feature Engineering
stock_data['Prev Close'] = stock_data['Adj Close'].shift(1)
stock_data['EMA_100'] = talib.EMA(stock_data['Adj Close'], timeperiod=100)
stock_data['MACD'], stock_data['MACD_signal'], _ = talib.MACD(stock_data['Adj Close'], fastperiod=12, slowperiod=26,
                                                              signalperiod=9)
stock_data['Low_Close_Diff'] = stock_data['Adj Close'] - stock_data['Low']
stock_data['SMA_Low_50'] = talib.SMA(stock_data['Low'], timeperiod=50)
upper_band, middle_band, lower_band = talib.BBANDS(stock_data['Low'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
stock_data['Low_Close_Diff_Volume'] = stock_data['Low_Close_Diff'] * stock_data['Volume']
stock_data['Stoch_K'], stock_data['Stoch_D'] = talib.STOCH(stock_data['High'], stock_data['Low'],
                                                           stock_data['Adj Close'], fastk_period=14, slowk_period=3,
                                                           slowd_period=3)
stock_data['Lag_10'] = stock_data['Adj Close'].shift(10)

stock_data['Lower_BB'] = lower_band
stock_data['Upper_BB'] = upper_band

# Create the "Tomorrow" column and classify the direction
stock_data['Tomorrow_Open'] = stock_data['Open'].shift(-1)
stock_data['Direction'] = (stock_data['Tomorrow_Open'] > stock_data['Open']).astype(int)

# Drop rows with NaN values (caused by shifting)
stock_data.dropna(inplace=True)

# Filter stock_data for backtesting period
backtest_data = stock_data[(stock_data.index >= backtest_start_date) & (stock_data.index <= backtest_end_date)]

# Define features (X) and the target variable (y) AFTER dropping NaN values
X = backtest_data[
    ['Open', 'High', 'Low', 'Volume', 'Prev Close', 'Low_Close_Diff', 'SMA_Low_50', 'Lower_BB', 'Upper_BB',
     'Low_Close_Diff_Volume', 'EMA_100', 'Lag_10']]
y = backtest_data['Direction']

# Train-test split (using a subset of backtest_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Load the trained scaler from the .pkl file
scaler = joblib.load('scaler.pkl')
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)

# Train the XGB model
xgb_model = XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Get predictions from both models on the training data
rf_predictions_train = rf_model.predict_proba(X_train)[:, 1]
xgb_predictions_train = xgb_model.predict_proba(X_train)[:, 1]

# Stack predictions as features for the meta-model
stacked_train = np.column_stack((rf_predictions_train, xgb_predictions_train))

# Load the trained meta-model from the .pkl file
meta_model = joblib.load('meta_model.pkl')

# Get predictions from both models on the test data
rf_predictions_test = rf_model.predict_proba(X_test)[:, 1]
xgb_predictions_test = xgb_model.predict_proba(X_test)[:, 1]

# Stack predictions as features for the meta-model
stacked_test = np.column_stack((rf_predictions_test, xgb_predictions_test))

# Get probability predictions from the meta-model
meta_model_probs = meta_model.predict_proba(stacked_test)[:, 1]

# Use a custom threshold to make the final predictions
custom_threshold = 0.7  # Adjust this threshold based on desired precision-recall trade-off
stacked_pred_custom = (meta_model_probs > custom_threshold).astype(int)

# Evaluate the model using the custom threshold
precision = precision_score(y_test, stacked_pred_custom)
recall = recall_score(y_test, stacked_pred_custom)
f1 = f1_score(y_test, stacked_pred_custom)
accuracy = accuracy_score(y_test, stacked_pred_custom)

print(f'Precision Score (Custom Threshold): {precision}')
print(f'Recall Score (Custom Threshold): {recall}')
print(f'F1 Score (Custom Threshold): {f1}')
print(f'Accuracy Score (Custom Threshold): {accuracy}')

# Implement backtesting logic with the selected backtesting period
initial_capital = 10000  # Starting with $10,000
cash = initial_capital
positions = 0  # Number of shares held
portfolio_value = [initial_capital]  # Track portfolio value over time

# Backtesting loop
for i in range(len(y_test)):
    current_open_price = backtest_data.iloc[i]['Open']

    # Buy signal (model predicts price will go up)
    if stacked_pred_custom[i] == 1 and cash >= current_open_price:
        shares_to_buy = cash // current_open_price
        cash -= shares_to_buy * current_open_price
        positions += shares_to_buy

    # Sell signal (model predicts price will go down) and if we have positions
    elif stacked_pred_custom[i] == 0 and positions > 0:
        cash += positions * current_open_price
        positions = 0

    # Calculate current portfolio value
    current_portfolio_value = cash + positions * current_open_price
    portfolio_value.append(current_portfolio_value)

# Plot portfolio value over time
plt.figure(figsize=(10, 6))
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


def sharpe_ratio(returns, risk_free_rate=0.01):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return (mean_return - risk_free_rate) / std_return


daily_returns = np.diff(portfolio_value) / portfolio_value[:-1]
sharpe = sharpe_ratio(daily_returns)
print(f'Sharpe Ratio: {sharpe}')

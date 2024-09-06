import requests
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, \
    ConfusionMatrixDisplay
import talib
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from config import ALPHA_VANTAGE_API_KEY, ALPACA_API_KEY, ALPACA_END_POINT, ALPACA_SECRET_KEY
from alpaca_trade_api import REST

# Stocks I'm trading: AAPL, TSLA, and PLTR

# Daily Routine:

# End of Trading Day: Run your model 15 min before market close.
# Make Decision: Based on the model’s prediction, decide whether to buy, sell, or short before the market closes.


# initialize Alpaca API
alpaca_api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_END_POINT)

# Parameters
SYMBOL = input("Enter the stock symbol (e.g., 'MARA'): ")
start_date = '2010-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Download stock data
pd.set_option('display.max_columns', None)
stock_data = yf.download(SYMBOL, start=start_date, end=end_date)


def get_sentiment_analysis(symbol, start_date, end_date):
    base_url = 'https://www.alphavantage.co/query'
    function = 'NEWS_SENTIMENT'

    start_date_formatted = pd.to_datetime(start_date).strftime('%Y%m%dT%H%M')
    end_date_formatted = pd.to_datetime(end_date).strftime('%Y%m%dT%H%M')

    params = {
        'function': function,
        'tickers': symbol,
        'time_from': start_date_formatted,
        'time_to': end_date_formatted,
        'sort': 'LATEST',
        'apikey': ALPHA_VANTAGE_API_KEY
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    return data


def process_sentiment_analysis(data):
    sentiments = []

    for articles in data.get('articles', []):
        published_date = pd.to_datetime(articles['publishedAt']).date()
        sentiment_score = articles['sentiment_score']
        sentiments.append((published_date, sentiment_score))

    sentiment_df = pd.DataFrame(sentiments, columns=['Date', 'Sentiment'])

    # Aggregate by date to get daily average sentiment score
    daily_sentiment = sentiment_df.groupby('Date').mean()

    return daily_sentiment


# Ensure the index is a DateTimeIndex
stock_data.index = pd.to_datetime(stock_data.index)

# Fetch and process sentiment data
sentiment_data = get_sentiment_analysis(SYMBOL, start_date, end_date)
daily_sentiment = process_sentiment_analysis(sentiment_data)

# Merge sentiment data using index directly, no need to create a separate 'Date' column
stock_data = stock_data.merge(daily_sentiment, how='left', left_index=True, right_index=True)

pd.set_option('future.no_silent_downcasting', True)

# Then perform your fillna operation
stock_data['Sentiment'] = stock_data['Sentiment'].fillna(0)

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

# Define features (X) and the target variable (y) AFTER dropping NaN values
X = stock_data[['Open', 'High', 'Low', 'Volume', 'Prev Close', 'Low_Close_Diff', 'SMA_Low_50', 'Lower_BB', 'Upper_BB',
                'Low_Close_Diff_Volume',
                'EMA_100', 'Lag_10', 'Sentiment']]
y = stock_data['Direction']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)

# Train the XGB model
xgb_model = XGBClassifier(scale_pos_weight=2, n_estimators=200, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Get predictions from both models on the training data
rf_predictions_train = rf_model.predict_proba(X_train)[:, 1]
xgb_predictions_train = xgb_model.predict_proba(X_train)[:, 1]

# Stack predictions as features for the meta-model
stacked_train = np.column_stack((rf_predictions_train, xgb_predictions_train))

# Train meta-model (Logistic Regression)
meta_model = LogisticRegression(random_state=42, penalty='l2', C=0.5)
meta_model.fit(stacked_train, y_train)

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

joblib.dump(meta_model, 'meta_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate the model using the custom threshold
precision = precision_score(y_test, stacked_pred_custom)
recall = recall_score(y_test, stacked_pred_custom)
f1 = f1_score(y_test, stacked_pred_custom)
accuracy = accuracy_score(y_test, stacked_pred_custom)

print(f'Precision Score (Custom Threshold): {precision}')
print(f'Recall Score (Custom Threshold): {recall}')
print(f'F1 Score (Custom Threshold): {f1}')
print(f'Accuracy Score (Custom Threshold): {accuracy}')

# Plot actual vs. predicted direction using custom threshold
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, color='blue', label='Actual Direction (Up=1, Down=0)')
plt.plot(y_test.index, stacked_pred_custom, color='red', label='Predicted Direction (Custom Threshold)')
plt.title('Next Day Opening Price Direction Prediction Using Stacked Model with Custom Threshold')
plt.xlabel('Date')
plt.ylabel('Direction')
plt.legend()
plt.show()

# Plot feature importances for Gradient Boosting
feature_importances = rf_model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances, color='green')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Gradient Boosting Important Features')
plt.show()

cm = confusion_matrix(y_test, stacked_pred_custom)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Use the last available day's data to predict tomorrow's direction
latest_data = stock_data.iloc[-1:]

# Select the features for the latest data point
X_latest = latest_data[
    ['Open', 'High', 'Low', 'Volume', 'Prev Close', 'Low_Close_Diff', 'SMA_Low_50', 'Lower_BB', 'Upper_BB',
     'Low_Close_Diff_Volume',
     'EMA_100', 'Lag_10', 'Sentiment']]

# Scale the features using the same scaler fitted on the training data
X_latest_scaled = scaler.transform(X_latest)

# Predict the probability of the direction for tomorrow
tomorrow_prob = meta_model.predict_proba(np.column_stack((
    rf_model.predict_proba(X_latest_scaled)[:, 1],
    xgb_model.predict_proba(X_latest_scaled)[:, 1]
)))[:, 1]

# Use the custom threshold to predict tomorrow's direction
if tomorrow_prob[0] > custom_threshold:
    alpaca_api.submit_order(
        symbol=SYMBOL,
        qty=20,
        side='buy',
        type='market',
        time_in_force='gtc'
    )
    print(f"Executed buy order for {SYMBOL}")
else:
    alpaca_api.submit_order(
        symbol=SYMBOL,
        qty=20,
        side='sell',
        type='market',
        time_in_force='gtc'
    )
    print(f"Executed sell order for {SYMBOL}")

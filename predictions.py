import requests
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import talib
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from config import ALPHA_VANTAGE_API_KEY
import streamlit as st

st.title('Live Stock Trading Dashboard')
st.sidebar.header('Trading Configuration')

# Parameters
SYMBOL = st.sidebar.text_input("Enter the stock symbol (e.g., 'AAPL')", value='AMD')
start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 1, 1))
end_date = datetime.today().strftime('%Y-%m-%d')

# Download stock data
st.write(f"Loading data for {SYMBOL}...")
stock_data = yf.download(SYMBOL, start=start_date, end=end_date)
st.write(f"Data loaded for {SYMBOL}")

# Show stock data in a table
st.subheader('Stock Data')
st.dataframe(stock_data.tail())


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

# Show evaluation metrics
st.subheader("Model Evaluation Metrics")
precision = precision_score(y_test, stacked_pred_custom)
recall = recall_score(y_test, stacked_pred_custom)
f1 = f1_score(y_test, stacked_pred_custom)
accuracy = accuracy_score(y_test, stacked_pred_custom)
st.write(f'Precision: {precision}')
st.write(f'Recall: {recall}')
st.write(f'F1 Score: {f1}')
st.write(f'Accuracy: {accuracy}')

# Show confusion matrix
cm = confusion_matrix(y_test, stacked_pred_custom)
st.subheader("Confusion Matrix")
st.write(cm)

# Feature Importances Plot
st.subheader("Feature Importances")
feature_importances = rf_model.feature_importances_
fig, ax = plt.subplots()
ax.barh(X.columns, feature_importances, color='green')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.set_title('Random Forest Feature Importances')
st.pyplot(fig)

# Use the last available day's data to predict tomorrow's direction
latest_data = stock_data.iloc[-1:].copy()

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


st.subheader('Prediction for Tomorrow')
if tomorrow_prob[0] > custom_threshold:
    st.write("The model predicts the stock will go up tomorrow.")
else:
    st.write("The model predicts the stock will go down tomorrow.")


# StockPrediction

## Overview

This project is a Stock Price Prediction and Trading Automation System that combines machine learning models with real-time stock trading via the Alpaca API. The system predicts stock price direction for the next day’s market open and can automatically execute buy or sell trades based on those predictions.

The pipeline leverages historical stock data, sentiment analysis, and technical indicators to make informed trading decisions.

## Features

∙ Real-Time Stock Data: Fetch historical stock data using Yahoo Finance (finance).

∙ Sentiment Analysis: Incorporate news sentiment data from Alpha Vantage API to enhance model predictions.

∙ Technical Indicators: Utilize TA-Lib to compute key technical indicators like Exponential Moving Averages (EMA), MACD, Bollinger Bands, and Stochastic Oscillators.

∙ Machine Learning Models: Train Random Forest and XGBoost models, then stack their predictions using Logistic Regression for a meta-model.

∙ Trading Automation: Automatically place buy/sell orders based on the model’s predictions via the Alpaca API.

∙ Custom Threshold: Adjust the prediction threshold to fine-tune the precision-recall trade-off.

∙ Model Evaluation: Evaluate the model using precision, recall, F1 score, accuracy, and confusion matrix.

## Usage

### 1. Input Stock Symbol

The system will prompt you to enter the stock symbol (e.g., AAPL, TSLA) for which you want to predict the next day’s price direction.

### 2. Download Historical Stock Data

The script automatically downloads historical stock data from Yahoo Finance for the specified stock symbol.

### 3. Perform Sentiment Analysis

News sentiment data for the selected stock is fetched using Alpha Vantage's API and integrated into the model’s features.

### 4. Feature Engineering

The system computes technical indicators like EMA, MACD, Bollinger Bands, and others using TA-Lib and incorporates them into the training features.

### 5. Make Predictions

The trained meta-model is used to predict the direction of the stock price for the next day’s market open.

### 6. Automated Trading via Alpaca API

Based on the prediction and a custom threshold (default: 0.7), the system places either a buy or sell order via Alpaca.

### 7. Model Evaluation

The script outputs evaluation metrics such as precision, recall, F1 score, accuracy, and plots a confusion matrix to assess the model’s performance.

### 8. Predict Tomorrow’s Direction

The system predicts the stock's direction for the next day based on the most recent available data and the trained model.

## File Structure

∙ trading.py: The main Python script for stock prediction and trading automation.

∙ predictions.py: The live dashboard for interaction and experimentation

∙ backtest.py: The script for backtesting the model on different time periods

## Requirements

∙ Python 3.7+

∙ Alpaca Trade API

∙ Scikit-learn

∙ XGBoost

∙ TA-Lib

∙ Yahoo Finance (finance)

∙ Alpha Vantage API


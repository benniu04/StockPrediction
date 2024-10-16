# StockPrediction

## Overview

This project is a Stock Price Prediction and Trading Automation System that combines machine learning models with real-time stock trading via the Alpaca API. The system predicts stock price direction for the next day’s market open and can automatically execute buy or sell trades based on those predictions. The model buys if it predicts the stock is going up the next day and sells if it predicts the stock is going down. It is best to run script 15mins before market close for best predictions.


## Program Interface

![main-dashboard](https://github.com/user-attachments/assets/d51847f0-a2af-4b8c-8edb-011bb4aca3c8)

![dashboard-1](https://github.com/user-attachments/assets/5772c200-6644-40a4-a67b-3e19ff2331d7)
![dashboard-2](https://github.com/user-attachments/assets/41e8425f-0613-4816-9f0a-ac7f77ee2114)

## Results

![alpaca-results](https://github.com/user-attachments/assets/f79abf3d-2359-452b-b3bf-39a287e1bcc1)


## Features

∙ Real-Time Stock Data: Fetch historical stock data using Yahoo Finance (finance).

∙ Sentiment Analysis: Incorporate news sentiment data from Alpha Vantage API to enhance model predictions.

∙ Technical Indicators: Utilize TA-Lib to compute key technical indicators like Exponential Moving Averages (EMA), MACD, Bollinger Bands, and Stochastic Oscillators.

∙ Machine Learning Models: Train Random Forest and XGBoost models, then stack their predictions using Logistic Regression for a meta-model.

∙ Trading Automation: Automatically place buy/sell orders based on the model’s predictions via the Alpaca API.

∙ Custom Threshold: Adjust the prediction threshold to fine-tune the precision-recall trade-off.

∙ Model Evaluation: Evaluate the model using precision, recall, F1 score, accuracy, and confusion matrix.


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


# StockPrediction

## Overview

http://52.201.215.144:8501/ (not working right now due to cost reasons)

This project is a Stock Price Prediction and Trading Automation System that combines machine learning models with real-time stock trading via the Alpaca API. The system predicts stock price direction for the next day’s market open and can automatically execute buy or sell trades based on those predictions. The model buys if it predicts the stock is going up the next day and sells if it predicts the stock is going down.

Note: try and run the script 15min before the market closes to get the most recent market data for predictions.


## Program Interface

![main-dashboard](https://github.com/user-attachments/assets/d51847f0-a2af-4b8c-8edb-011bb4aca3c8)

![trading dashboard](https://github.com/user-attachments/assets/5873f76d-44e3-48e1-b47d-1434f1d4c589)


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


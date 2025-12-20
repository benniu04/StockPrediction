"""
Training script for improved stock prediction model
Can train on single or multiple stocks
"""

import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv

from improved_features import engineer_all_features, get_feature_columns
from improved_modeling import ImprovedStockPredictor, MultiStockPredictor, ModelMetrics

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


def download_stock_data(
    symbol: str,
    start_date: str = '2010-01-01',
    end_date: str = None
) -> pd.DataFrame:
    """Download stock price data"""
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    print(f"Downloading {symbol} from {start_date} to {end_date}...")
    
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    
    # Handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Ensure Adj Close exists
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    
    return df


def fetch_sentiment_data(
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Fetch sentiment data from Alpha Vantage"""
    if not ALPHA_VANTAGE_API_KEY:
        print(f"No Alpha Vantage API key found, skipping sentiment for {symbol}")
        return pd.DataFrame(columns=['Sentiment'], index=pd.DatetimeIndex([], name='Date'))
    
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': symbol,
        'time_from': pd.to_datetime(start_date).strftime('%Y%m%dT%H%M'),
        'time_to': end_date,
        'sort': 'LATEST',
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        sentiments = []
        for article in data.get('articles', []):
            date = pd.to_datetime(article['publishedAt']).date()
            score = article['sentiment_score']
            sentiments.append((date, score))
        
        if not sentiments:
            return pd.DataFrame(columns=['Sentiment'], index=pd.DatetimeIndex([], name='Date'))
        
        df = pd.DataFrame(sentiments, columns=['Date', 'Sentiment'])
        df = df.groupby('Date').mean()
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        
        return df
    
    except Exception as e:
        print(f"Failed to fetch sentiment for {symbol}: {e}")
        return pd.DataFrame(columns=['Sentiment'], index=pd.DatetimeIndex([], name='Date'))


def prepare_stock_data(
    symbol: str,
    start_date: str = '2010-01-01',
    end_date: str = None,
    include_sentiment: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Download and prepare data for a single stock
    
    Returns:
        X: Features
        y: Target (direction)
        returns: Actual returns
    """
    # Download price data
    price_df = download_stock_data(symbol, start_date, end_date)
    
    # Download sentiment data
    if include_sentiment:
        sentiment_df = fetch_sentiment_data(symbol, start_date, end_date or datetime.today().strftime('%Y-%m-%d'))
    else:
        sentiment_df = None
    
    # Engineer features
    full_df = engineer_all_features(price_df, sentiment_df)
    
    # Extract features, target, and returns
    feature_cols = get_feature_columns()
    X = full_df[feature_cols]
    y = full_df['Direction']
    returns = full_df['Target_Return']
    
    print(f"  {symbol}: {len(X)} samples, {len(feature_cols)} features")
    print(f"  Class balance: {y.mean():.2%} positive")
    
    return X, y, returns


def train_single_stock_model(
    symbol: str,
    start_date: str = '2010-01-01',
    end_date: str = None,
    save_path: str = 'models/',
    verbose: bool = True
) -> Tuple[ImprovedStockPredictor, ModelMetrics]:
    """
    Train model on a single stock
    """
    print(f"\n{'='*60}")
    print(f"Training model for {symbol}")
    print(f"{'='*60}")
    
    # Prepare data
    X, y, returns = prepare_stock_data(symbol, start_date, end_date)
    
    # Create and train model
    model = ImprovedStockPredictor(
        use_walk_forward=True,
        n_cv_splits=5,
        transaction_cost=0.0005,  # 5 basis points
        random_state=42
    )
    
    model.fit(X, y, returns, verbose=verbose)
    
    # Evaluate
    metrics = model.evaluate(X, y, returns)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {symbol}")
    print(f"{'='*60}")
    print(f"Classification Metrics:")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1 Score:  {metrics.f1:.4f}")
    print(f"  ROC AUC:   {metrics.roc_auc:.4f}")
    print(f"\nTrading Metrics:")
    print(f"  Sharpe Ratio:   {metrics.sharpe_ratio:.4f}")
    print(f"  Max Drawdown:   {metrics.max_drawdown:.2%}")
    print(f"  Win Rate:       {metrics.win_rate:.2%}")
    print(f"  Profit Factor:  {metrics.profit_factor:.4f}")
    print(f"  Total Return:   {metrics.total_return:.2%}")
    print(f"\nStability (across CV splits):")
    print(f"  Accuracy: {np.mean(metrics.split_accuracies):.4f} ± {np.std(metrics.split_accuracies):.4f}")
    print(f"  Sharpe:   {np.mean(metrics.split_sharpes):.4f} ± {np.std(metrics.split_sharpes):.4f}")
    
    # Save model
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, f'{symbol}_model.pkl')
    metrics_file = os.path.join(save_path, f'{symbol}_metrics.pkl')
    
    joblib.dump(model, model_file)
    joblib.dump(metrics, metrics_file)
    
    print(f"\nModel saved to {model_file}")
    
    return model, metrics


def train_multi_stock_model(
    symbols: List[str],
    start_date: str = '2010-01-01',
    end_date: str = None,
    save_path: str = 'models/',
    verbose: bool = True
) -> Tuple[ImprovedStockPredictor, Dict[str, ModelMetrics]]:
    """
    Train a single model on multiple stocks
    This creates a more generalized model
    """
    print(f"\n{'='*60}")
    print(f"Training unified model on {len(symbols)} stocks")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"{'='*60}")
    
    # Prepare data for all stocks
    stock_data = {}
    for symbol in symbols:
        try:
            X, y, returns = prepare_stock_data(symbol, start_date, end_date)
            stock_data[symbol] = (X, y, returns)
        except Exception as e:
            print(f"Failed to prepare {symbol}: {e}")
            continue
    
    if not stock_data:
        raise ValueError("No valid stock data loaded")
    
    # Create base predictor
    base_model = ImprovedStockPredictor(
        use_walk_forward=True,
        n_cv_splits=5,
        transaction_cost=0.0005,
        random_state=42
    )
    
    # Create multi-stock predictor
    multi_model = MultiStockPredictor(base_model, list(stock_data.keys()))
    
    # Train
    multi_model.fit(stock_data, verbose=verbose)
    
    # Evaluate on each stock
    print(f"\n{'='*60}")
    print(f"Evaluation Results by Stock")
    print(f"{'='*60}")
    
    stock_metrics = {}
    for symbol, (X, y, returns) in stock_data.items():
        metrics = multi_model.base_predictor.evaluate(X, y, returns)
        stock_metrics[symbol] = metrics
        
        print(f"\n{symbol}:")
        print(f"  Accuracy: {metrics.accuracy:.4f}, Sharpe: {metrics.sharpe_ratio:.4f}")
        print(f"  Return: {metrics.total_return:.2%}, Max DD: {metrics.max_drawdown:.2%}")
    
    # Calculate average metrics
    avg_accuracy = np.mean([m.accuracy for m in stock_metrics.values()])
    avg_sharpe = np.mean([m.sharpe_ratio for m in stock_metrics.values()])
    avg_return = np.mean([m.total_return for m in stock_metrics.values()])
    
    print(f"\n{'='*60}")
    print(f"Average Performance Across Stocks")
    print(f"{'='*60}")
    print(f"  Accuracy: {avg_accuracy:.4f}")
    print(f"  Sharpe:   {avg_sharpe:.4f}")
    print(f"  Return:   {avg_return:.2%}")
    
    # Save model
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, 'multi_stock_model.pkl')
    metrics_file = os.path.join(save_path, 'multi_stock_metrics.pkl')
    
    joblib.dump(multi_model.base_predictor, model_file)
    joblib.dump(stock_metrics, metrics_file)
    
    print(f"\nModel saved to {model_file}")
    
    return multi_model.base_predictor, stock_metrics


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train improved stock prediction model')
    parser.add_argument('--mode', choices=['single', 'multi'], default='multi',
                        help='Train on single stock or multiple stocks')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM', 'DIA'],
                        help='Stock symbols to train on')
    parser.add_argument('--start-date', default='2010-01-01',
                        help='Start date for training data')
    parser.add_argument('--save-path', default='models/',
                        help='Path to save models')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Train separate model for each stock
        for symbol in args.symbols:
            try:
                train_single_stock_model(
                    symbol=symbol,
                    start_date=args.start_date,
                    save_path=args.save_path,
                    verbose=True
                )
            except Exception as e:
                print(f"Failed to train {symbol}: {e}")
    else:
        # Train unified model on all stocks
        train_multi_stock_model(
            symbols=args.symbols,
            start_date=args.start_date,
            save_path=args.save_path,
            verbose=True
        )


if __name__ == '__main__':
    main()


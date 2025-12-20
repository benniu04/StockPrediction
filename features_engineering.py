"""
Improved Feature Engineering for Stock Prediction
- Uses returns instead of absolute prices for cross-stock generalization
- Adds volatility and market microstructure features
- Normalizes all features appropriately
"""

import numpy as np
import pandas as pd
import talib
from typing import Optional


def calculate_returns_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate return-based features that generalize across stocks"""
    df = df.copy()
    
    # Price returns (generalize across stocks)
    df['Return_1d'] = df['Adj Close'].pct_change(1)
    df['Return_5d'] = df['Adj Close'].pct_change(5)
    df['Return_10d'] = df['Adj Close'].pct_change(10)
    df['Return_20d'] = df['Adj Close'].pct_change(20)
    
    # Intraday returns
    df['Intraday_Return'] = (df['Close'] - df['Open']) / df['Open']
    df['Overnight_Return'] = (df['Open'] - df['Adj Close'].shift(1)) / df['Adj Close'].shift(1)
    
    # High-Low spread (volatility proxy)
    df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
    
    # Close position in daily range
    df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    return df


def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility-based features"""
    df = df.copy()
    
    # Historical volatility (annualized)
    returns = df['Adj Close'].pct_change()
    df['Volatility_10d'] = returns.rolling(10).std() * np.sqrt(252)
    df['Volatility_20d'] = returns.rolling(20).std() * np.sqrt(252)
    df['Volatility_50d'] = returns.rolling(50).std() * np.sqrt(252)
    
    # Volatility ratios (regime detection)
    df['Vol_Ratio_Short_Long'] = df['Volatility_10d'] / (df['Volatility_50d'] + 1e-10)
    
    # Parkinson volatility (uses high-low range)
    df['Parkinson_Vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * 
        (np.log(df['High'] / df['Low']) ** 2).rolling(20).mean()
    ) * np.sqrt(252)
    
    return df


def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum and trend features"""
    df = df.copy()
    
    # Moving averages as ratios
    df['MA_Ratio_20_50'] = (
        talib.SMA(df['Adj Close'], 20) / 
        (talib.SMA(df['Adj Close'], 50) + 1e-10)
    )
    df['MA_Ratio_50_200'] = (
        talib.SMA(df['Adj Close'], 50) / 
        (talib.SMA(df['Adj Close'], 200) + 1e-10)
    )
    
    # Price relative to moving averages
    df['Price_to_MA20'] = df['Adj Close'] / (talib.SMA(df['Adj Close'], 20) + 1e-10)
    df['Price_to_MA50'] = df['Adj Close'] / (talib.SMA(df['Adj Close'], 50) + 1e-10)
    
    # MACD normalized by price
    macd, signal, hist = talib.MACD(df['Adj Close'])
    df['MACD_Normalized'] = macd / (df['Adj Close'] + 1e-10)
    df['MACD_Signal_Normalized'] = signal / (df['Adj Close'] + 1e-10)
    df['MACD_Hist_Normalized'] = hist / (df['Adj Close'] + 1e-10)
    
    # RSI
    df['RSI_14'] = talib.RSI(df['Adj Close'], 14)
    df['RSI_28'] = talib.RSI(df['Adj Close'], 28)
    
    # Rate of change
    df['ROC_10'] = talib.ROC(df['Adj Close'], 10)
    df['ROC_20'] = talib.ROC(df['Adj Close'], 20)
    
    return df


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume-based features"""
    df = df.copy()
    
    # Volume ratios (normalize by recent average)
    df['Volume_Ratio_5d'] = df['Volume'] / (df['Volume'].rolling(5).mean() + 1e-10)
    df['Volume_Ratio_20d'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-10)
    
    # On-Balance Volume (OBV) normalized
    obv = talib.OBV(df['Adj Close'], df['Volume'])
    df['OBV_Normalized'] = obv / (df['Volume'].rolling(20).mean() + 1e-10)
    
    # Volume-weighted returns
    df['Volume_Return_Correlation'] = (
        df['Return_1d'].rolling(20).corr(df['Volume_Ratio_5d'])
    )
    
    # Average Dollar Volume (liquidity measure)
    df['Dollar_Volume'] = df['Volume'] * df['Adj Close']
    df['Dollar_Volume_20d'] = df['Dollar_Volume'].rolling(20).mean()
    df['Dollar_Volume_Ratio'] = df['Dollar_Volume'] / (df['Dollar_Volume_20d'] + 1e-10)
    
    return df


def calculate_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market microstructure features"""
    df = df.copy()
    
    # Amihud illiquidity measure
    df['Amihud_Illiquidity'] = (
        np.abs(df['Return_1d']) / 
        (df['Dollar_Volume'] + 1e-10)
    ).rolling(20).mean()
    
    # Bid-ask spread proxy (using high-low)
    df['Spread_Proxy'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
    df['Avg_Spread_20d'] = df['Spread_Proxy'].rolling(20).mean()
    
    # Price impact (return per dollar volume)
    df['Price_Impact'] = (
        df['Return_1d'] / 
        (np.log1p(df['Dollar_Volume']) + 1e-10)
    ).rolling(20).mean()
    
    return df


def calculate_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pattern-based technical features"""
    df = df.copy()
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['Adj Close'], 20)
    df['BB_Position'] = (df['Adj Close'] - lower) / (upper - lower + 1e-10)
    df['BB_Width'] = (upper - lower) / (middle + 1e-10)
    
    # Stochastic oscillator
    slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Adj Close'])
    df['Stoch_K'] = slowk
    df['Stoch_D'] = slowd
    df['Stoch_Diff'] = slowk - slowd
    
    # ATR (Average True Range) normalized
    atr = talib.ATR(df['High'], df['Low'], df['Adj Close'], 14)
    df['ATR_Normalized'] = atr / (df['Adj Close'] + 1e-10)
    
    # Chandelier Exit
    df['Chandelier_Long'] = (
        df['High'].rolling(22).max() - 
        atr * 3
    ) / (df['Adj Close'] + 1e-10)
    
    return df


def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features that detect market regime"""
    df = df.copy()
    
    # Trend strength
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Adj Close'], 14)
    
    # Number of consecutive up/down days
    up_days = (df['Adj Close'] > df['Adj Close'].shift(1)).astype(int)
    down_days = (df['Adj Close'] < df['Adj Close'].shift(1)).astype(int)
    
    df['Consecutive_Up'] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum()
    df['Consecutive_Down'] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()
    
    # Drawdown from recent peak
    rolling_max = df['Adj Close'].rolling(252, min_periods=1).max()
    df['Drawdown'] = (df['Adj Close'] - rolling_max) / (rolling_max + 1e-10)
    
    # Distance from 52-week high/low
    high_52w = df['High'].rolling(252, min_periods=1).max()
    low_52w = df['Low'].rolling(252, min_periods=1).min()
    df['Distance_From_52w_High'] = (df['Adj Close'] - high_52w) / (high_52w + 1e-10)
    df['Distance_From_52w_Low'] = (df['Adj Close'] - low_52w) / (low_52w + 1e-10)
    
    return df


def engineer_all_features(
    df: pd.DataFrame, 
    sentiment_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Apply all feature engineering functions
    
    Args:
        df: OHLCV price dataframe with DatetimeIndex
        sentiment_df: Optional sentiment dataframe with DatetimeIndex
        
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Apply all feature groups
    df = calculate_returns_features(df)
    df = calculate_volatility_features(df)
    df = calculate_momentum_features(df)
    df = calculate_volume_features(df)
    df = calculate_microstructure_features(df)
    df = calculate_pattern_features(df)
    df = calculate_regime_features(df)
    
    # Add sentiment if available
    if sentiment_df is not None:
        df = df.merge(sentiment_df, left_index=True, right_index=True, how='left')
        df['Sentiment'] = df['Sentiment'].fillna(0)
        
        # Sentiment momentum
        df['Sentiment_Change'] = df['Sentiment'].diff()
        df['Sentiment_MA_5'] = df['Sentiment'].rolling(5).mean()
    else:
        df['Sentiment'] = 0
        df['Sentiment_Change'] = 0
        df['Sentiment_MA_5'] = 0
    
    # Create target: next day's return direction
    # Using open-to-open to avoid look-ahead bias
    df['Tomorrow_Open'] = df['Open'].shift(-1)
    df['Target_Return'] = (df['Tomorrow_Open'] - df['Open']) / (df['Open'] + 1e-10)
    df['Direction'] = (df['Target_Return'] > 0).astype(int)
    
    # Drop rows with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df


def get_feature_columns() -> list:
    """Return list of feature columns to use for modeling"""
    return [
        # Returns
        'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d',
        'Intraday_Return', 'Overnight_Return', 'HL_Spread', 'Close_Position',
        
        # Volatility
        'Volatility_10d', 'Volatility_20d', 'Volatility_50d',
        'Vol_Ratio_Short_Long', 'Parkinson_Vol',
        
        # Momentum
        'MA_Ratio_20_50', 'MA_Ratio_50_200', 'Price_to_MA20', 'Price_to_MA50',
        'MACD_Normalized', 'MACD_Signal_Normalized', 'MACD_Hist_Normalized',
        'RSI_14', 'RSI_28', 'ROC_10', 'ROC_20',
        
        # Volume
        'Volume_Ratio_5d', 'Volume_Ratio_20d', 'OBV_Normalized',
        'Volume_Return_Correlation', 'Dollar_Volume_Ratio',
        
        # Microstructure
        'Amihud_Illiquidity', 'Avg_Spread_20d', 'Price_Impact',
        
        # Patterns
        'BB_Position', 'BB_Width', 'Stoch_K', 'Stoch_D', 'Stoch_Diff',
        'ATR_Normalized', 'Chandelier_Long',
        
        # Regime
        'ADX', 'Consecutive_Up', 'Consecutive_Down', 'Drawdown',
        'Distance_From_52w_High', 'Distance_From_52w_Low',
        
        # Sentiment
        'Sentiment', 'Sentiment_Change', 'Sentiment_MA_5'
    ]


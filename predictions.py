import os
from datetime import datetime
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
import talib
import yfinance as yf
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from plotly import express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

st.set_page_config(page_title="📈 Live Stock Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("📈 Live Stock Trading Dashboard + Backtest (TS CV)")


symbol = st.sidebar.text_input("Stock symbol", value="AMD").upper()
start_date = st.sidebar.date_input("Start date", value=datetime(2010, 1, 1))
end_date_str = datetime.today().strftime("%Y-%m-%d")
custom_threshold = st.sidebar.slider("Probability threshold", 0.50, 0.95, 0.70, 0.01)
trans_cost_bps = st.sidebar.number_input("Transaction cost (bps)", 0.0, 100.0, 5.0, 0.5)

trans_cost = trans_cost_bps / 10_000


@st.cache_data(ttl=60 * 60)
def download_prices(sym: str, start: datetime, end: str) -> pd.DataFrame:
    df = yf.download(sym, start=start, end=end)
    if df.empty:
        raise ValueError("No price data returned.")
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(sym, axis=1, level=1)
        except KeyError:
            df.columns = df.columns.droplevel(0)
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df.get("Close")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


@st.cache_data(ttl=60 * 60)
def fetch_sentiment(sym: str, start: datetime, end: str) -> pd.DataFrame:
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": sym,
        "time_from": pd.to_datetime(start).strftime("%Y%m%dT%H%M"),
        "time_to": end,
        "sort": "LATEST",
        "apikey": ALPHA_VANTAGE_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame(columns=["Sentiment"], index=pd.DatetimeIndex([], name="Date"))
    rows = [(pd.to_datetime(a["publishedAt"]).date(), a["sentiment_score"]) for a in r.json().get("articles", [])]
    if not rows:
        return pd.DataFrame(columns=["Sentiment"], index=pd.DatetimeIndex([], name="Date"))
    df = pd.DataFrame(rows, columns=["Date", "Sentiment"]).groupby("Date").mean()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df["Prev Close"] = df["Adj Close"].shift(1)
    df["EMA_100"] = talib.EMA(df["Adj Close"], 100)
    df["MACD"], df["MACD_signal"], _ = talib.MACD(df["Adj Close"])
    df["Low_Close_Diff"] = df["Adj Close"] - df["Low"]
    df["SMA_Low_50"] = talib.SMA(df["Low"], 50)
    up, _, low = talib.BBANDS(df["Low"], 20)
    df["Lower_BB"], df["Upper_BB"] = low, up
    df["Low_Close_Diff_Volume"] = df["Low_Close_Diff"] * df["Volume"]
    df["Stoch_K"], df["Stoch_D"] = talib.STOCH(df["High"], df["Low"], df["Adj Close"])
    df["Lag_10"] = df["Adj Close"].shift(10)
    df["Tomorrow_Open"] = df["Open"].shift(-1)
    df["Direction"] = (df["Tomorrow_Open"] > df["Open"]).astype(int)
    df.dropna(inplace=True)
    return df


def backtest(prices: pd.Series, signals: pd.Series, cost: float) -> Tuple[pd.Series, pd.Series]:
    ret = prices.pct_change().fillna(0)
    pos = signals.shift(1).fillna(0)
    trades = pos.diff().abs().fillna(0)
    strat_ret = pos * ret - trades * cost
    return (1 + strat_ret).cumprod(), (1 + ret).cumprod()


with st.spinner("Loading data …"):
    price_df = download_prices(symbol, start_date, end_date_str)
    sent_df = fetch_sentiment(symbol, start_date, end_date_str)
    full_df = price_df.merge(sent_df, left_index=True, right_index=True, how="left")
    full_df["Sentiment"].fillna(0, inplace=True)
    feat_df = engineer(full_df)

FEATURES = [
    "Open", "High", "Low", "Volume", "Prev Close", "Low_Close_Diff", "SMA_Low_50",
    "Lower_BB", "Upper_BB", "Low_Close_Diff_Volume", "EMA_100", "Lag_10", "Sentiment",
]

X_all = feat_df[FEATURES].values
y_all = feat_df["Direction"].values


tscv = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(tscv.split(X_all))[-1]
X_train, X_test = X_all[train_idx], X_all[test_idx]
y_train, y_test = y_all[train_idx], y_all[test_idx]
dates_test = feat_df.index[test_idx]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train_s, y_train)

xgb = XGBClassifier(scale_pos_weight=2, n_estimators=200, random_state=42, eval_metric="logloss")
xgb.fit(X_train_s, y_train)

rf_tr = rf.predict_proba(X_train_s)[:, 1]
xgb_tr = xgb.predict_proba(X_train_s)[:, 1]
meta = LogisticRegression(random_state=42)
meta.fit(np.column_stack((rf_tr, xgb_tr)), y_train)

rf_te = rf.predict_proba(X_test_s)[:, 1]
xgb_te = xgb.predict_proba(X_test_s)[:, 1]
meta_prob_test = meta.predict_proba(np.column_stack((rf_te, xgb_te)))[:, 1]
preds_test = (meta_prob_test > custom_threshold).astype(int)

metrics = {
    "precision": precision_score(y_test, preds_test, zero_division=0),
    "recall": recall_score(y_test, preds_test, zero_division=0),
    "f1": f1_score(y_test, preds_test, zero_division=0),
    "accuracy": accuracy_score(y_test, preds_test),
}

tab_metrics, tab_backtest, tab_prediction = st.tabs(["🧮 Metrics", "📊 Backtest", "🔮 Prediction"])

with tab_metrics:
    st.subheader("Evaluation metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{metrics['precision']:.3f}")
    c2.metric("Recall", f"{metrics['recall']:.3f}")
    c3.metric("F1", f"{metrics['f1']:.3f}")
    c4.metric("Accuracy", f"{metrics['accuracy']:.3f}")

    cm = confusion_matrix(y_test, preds_test)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    fig_imp, ax_imp = plt.subplots(figsize=(6, 8))
    ax_imp.barh(FEATURES, rf.feature_importances_)
    ax_imp.set_title("Random Forest Feature Importance")
    st.pyplot(fig_imp)

with tab_backtest:
    st.subheader("Equity curve vs. Buy‑and‑Hold")

    X_all_s = scaler.transform(X_all)
    rf_all = rf.predict_proba(X_all_s)[:, 1]
    xgb_all = xgb.predict_proba(X_all_s)[:, 1]
    meta_all = meta.predict_proba(np.column_stack((rf_all, xgb_all)))[:, 1]
    signals = (meta_all > custom_threshold).astype(int)
    signals_series = pd.Series(signals, index=feat_df.index, name="Signal")

    aligned_prices = price_df["Adj Close"].loc[signals_series.index]
    strat_curve, bh_curve = backtest(aligned_prices, signals_series, trans_cost)
    equity_df = pd.DataFrame({"Strategy": strat_curve, "Buy&Hold": bh_curve})
    fig_curve = px.line(equity_df, labels={"value": "Equity", "Date": "Date"}, title="Strategy Performance")
    st.plotly_chart(fig_curve, use_container_width=True)

    st.metric("Strategy Total Return", f"{(strat_curve.iloc[-1]-1)*100:.2f}%")
    st.metric("Buy&Hold Total Return", f"{(bh_curve.iloc[-1]-1)*100:.2f}%")

with tab_prediction:
    st.subheader("Next‑Day Direction Forecast")

    latest_feat = feat_df[FEATURES].iloc[[-1]]

    latest_scaled = scaler.transform(latest_feat.values)
    rf_p = rf.predict_proba(latest_scaled)[:, 1]
    xgb_p = xgb.predict_proba(latest_scaled)[:, 1]
    meta_p = meta.predict_proba(np.column_stack((rf_p, xgb_p)))[:, 1][0]

    st.metric("P(up tomorrow)", f"{meta_p*100:.2f}%")

    if meta_p > custom_threshold:
        st.markdown(
            f"**Model says:** _go long_  (threshold = {custom_threshold:.2f})"
        )
    else:
        st.markdown(
            f"**Model says:** _stay flat / go short_  (threshold = {custom_threshold:.2f})"
        )


joblib.dump(meta, "meta_model.pkl")
joblib.dump(scaler, "scaler.pkl")

with open("meta_model.pkl", "rb") as f:
    st.sidebar.download_button("Download meta model", f, "meta_model.pkl")

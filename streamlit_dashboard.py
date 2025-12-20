"""
Improved Streamlit Dashboard for Stock Prediction
- Better metrics display
- Walk-forward validation results
- Multi-stock comparison
- Robust backtesting
"""

import os
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from dotenv import load_dotenv

from improved_features import engineer_all_features, get_feature_columns
from improved_modeling import ImprovedStockPredictor

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Page config
st.set_page_config(
    page_title="📈 Advanced Stock Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.big-metric {
    font-size: 24px;
    font-weight: bold;
    color: #1f77b4;
}
.positive {
    color: #2ecc71;
}
.negative {
    color: #e74c3c;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Advanced Stock Trading Dashboard")
st.markdown("*Powered by ML with Walk-Forward Validation*")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    
    # Mode selection
    mode = st.radio(
        "Model Mode",
        options=['Single Stock', 'Multi-Stock (Generalized)'],
        help="Single stock: Train on one stock. Multi-stock: Use generalized model trained on multiple stocks."
    )
    
    if mode == 'Single Stock':
        symbol = st.text_input("Stock Symbol", value="SPY").upper()
        symbols = [symbol]
    else:
        symbols_input = st.text_input(
            "Stock Symbols (comma-separated)",
            value="SPY,QQQ,IWM,DIA"
        )
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        symbol = st.selectbox("Primary Symbol for Display", symbols)
    
    start_date = st.date_input("Start Date", value=datetime(2015, 1, 1))
    custom_threshold = st.slider(
        "Prediction Threshold",
        0.45, 0.80, 0.55, 0.01,
        help="Probability threshold for going long"
    )
    trans_cost_bps = st.number_input(
        "Transaction Cost (bps)",
        0.0, 50.0, 5.0, 0.5,
        help="Transaction costs in basis points"
    )
    
    train_new_model = st.checkbox("Train New Model", value=False)
    use_walk_forward = st.checkbox("Use Walk-Forward Validation", value=True)
    
    trans_cost = trans_cost_bps / 10_000


@st.cache_data(ttl=60*60)
def download_prices(sym: str, start: datetime, end: str) -> pd.DataFrame:
    """Download price data"""
    df = yf.download(sym, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No price data for {sym}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    
    return df


@st.cache_data(ttl=60*60)
def fetch_sentiment(sym: str, start: datetime, end: str) -> pd.DataFrame:
    """Fetch sentiment data"""
    if not ALPHA_VANTAGE_API_KEY:
        return pd.DataFrame(columns=['Sentiment'], index=pd.DatetimeIndex([], name='Date'))
    
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
        
        rows = [
            (pd.to_datetime(a["publishedAt"]).date(), a["sentiment_score"])
            for a in r.json().get("articles", [])
        ]
        
        if not rows:
            return pd.DataFrame(columns=['Sentiment'], index=pd.DatetimeIndex([], name='Date'))
        
        df = pd.DataFrame(rows, columns=["Date", "Sentiment"]).groupby("Date").mean()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        
        return df
    
    except Exception:
        return pd.DataFrame(columns=['Sentiment'], index=pd.DatetimeIndex([], name='Date'))


def calculate_backtest_metrics(
    prices: pd.Series,
    signals: pd.Series,
    cost: float
) -> dict:
    """Calculate comprehensive backtest metrics"""
    ret = prices.pct_change().fillna(0)
    pos = signals.shift(1).fillna(0)
    trades = pos.diff().abs().fillna(0)
    
    # Strategy returns
    strat_ret = pos * ret - trades * cost
    strat_curve = (1 + strat_ret).cumprod()
    bh_curve = (1 + ret).cumprod()
    
    # Calculate metrics
    total_return = (strat_curve.iloc[-1] - 1) * 100
    bh_return = (bh_curve.iloc[-1] - 1) * 100
    
    # Sharpe ratio
    if strat_ret.std() > 0:
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    running_max = strat_curve.expanding().max()
    drawdown = (strat_curve - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    # Win rate
    winning_trades = (strat_ret[trades > 0] > 0).sum()
    total_trades = (trades > 0).sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Profit factor
    gross_profit = strat_ret[strat_ret > 0].sum()
    gross_loss = abs(strat_ret[strat_ret < 0].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    
    return {
        'strat_curve': strat_curve,
        'bh_curve': bh_curve,
        'total_return': total_return,
        'bh_return': bh_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': total_trades
    }


# Main execution
try:
    with st.spinner("Loading data..."):
        end_date_str = datetime.today().strftime("%Y-%m-%d")
        
        # Load data for primary symbol
        price_df = download_prices(symbol, start_date, end_date_str)
        sent_df = fetch_sentiment(symbol, start_date, end_date_str)
        
        # Engineer features
        full_df = engineer_all_features(price_df, sent_df)
        feature_cols = get_feature_columns()
        
        X = full_df[feature_cols]
        y = full_df['Direction']
        returns = full_df['Target_Return']
        
        st.success(f"✓ Loaded {len(X)} samples for {symbol}")
    
    # Train or load model
    model_path = f'models/{"multi_stock" if mode == "Multi-Stock (Generalized)" else symbol}_model.pkl'
    
    if train_new_model or not os.path.exists(model_path):
        with st.spinner("Training model... This may take a few minutes."):
            model = ImprovedStockPredictor(
                use_walk_forward=use_walk_forward,
                n_cv_splits=5,
                transaction_cost=trans_cost,
                random_state=42
            )
            
            if mode == 'Multi-Stock (Generalized)':
                # Load data for all symbols
                stock_data = {}
                progress_bar = st.progress(0)
                
                for i, sym in enumerate(symbols):
                    try:
                        price = download_prices(sym, start_date, end_date_str)
                        sent = fetch_sentiment(sym, start_date, end_date_str)
                        feat_df = engineer_all_features(price, sent)
                        
                        stock_data[sym] = (
                            feat_df[feature_cols],
                            feat_df['Direction'],
                            feat_df['Target_Return']
                        )
                        
                        progress_bar.progress((i + 1) / len(symbols))
                    except Exception as e:
                        st.warning(f"Failed to load {sym}: {e}")
                
                # Combine data
                X_combined = pd.concat([d[0] for d in stock_data.values()])
                y_combined = pd.concat([d[1] for d in stock_data.values()])
                returns_combined = pd.concat([d[2] for d in stock_data.values()])
                
                model.fit(X_combined, y_combined, returns_combined, verbose=False)
                st.success(f"✓ Trained on {len(symbols)} stocks ({len(X_combined)} total samples)")
            else:
                model.fit(X, y, returns, verbose=False)
                st.success("✓ Model trained successfully")
            
            # Save model
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, model_path)
    else:
        with st.spinner("Loading model..."):
            model = joblib.load(model_path)
            st.success("✓ Model loaded")
    
    # Make predictions
    y_proba = model.predict_proba(X)
    y_pred = (y_proba >= custom_threshold).astype(int)
    
    # Evaluate
    metrics = model.evaluate(X, y, returns)
    
    # Create tabs
    tab_overview, tab_metrics, tab_backtest, tab_predictions, tab_analysis = st.tabs([
        "📊 Overview",
        "📈 Model Metrics",
        "💰 Backtest",
        "🔮 Predictions",
        "🔬 Analysis"
    ])
    
    with tab_overview:
        st.subheader(f"Performance Summary: {symbol}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = "positive" if metrics.accuracy > 0.5 else "negative"
            st.markdown(f"**Accuracy**")
            st.markdown(f'<p class="big-metric {color}">{metrics.accuracy:.1%}</p>', unsafe_allow_html=True)
        
        with col2:
            color = "positive" if metrics.sharpe_ratio > 0 else "negative"
            st.markdown(f"**Sharpe Ratio**")
            st.markdown(f'<p class="big-metric {color}">{metrics.sharpe_ratio:.2f}</p>', unsafe_allow_html=True)
        
        with col3:
            color = "positive" if metrics.total_return > 0 else "negative"
            st.markdown(f"**Total Return**")
            st.markdown(f'<p class="big-metric {color}">{metrics.total_return:.1%}</p>', unsafe_allow_html=True)
        
        with col4:
            color = "negative" if metrics.max_drawdown < -0.1 else "positive"
            st.markdown(f"**Max Drawdown**")
            st.markdown(f'<p class="big-metric {color}">{metrics.max_drawdown:.1%}</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick backtest
        signals = pd.Series(y_pred, index=full_df.index)
        aligned_prices = price_df['Adj Close'].reindex(signals.index)
        bt_metrics = calculate_backtest_metrics(aligned_prices, signals, trans_cost)
        
        # Plot equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bt_metrics['strat_curve'].index,
            y=bt_metrics['strat_curve'].values,
            name='Strategy',
            line=dict(color='#2ecc71', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=bt_metrics['bh_curve'].index,
            y=bt_metrics['bh_curve'].values,
            name='Buy & Hold',
            line=dict(color='#3498db', width=2)
        ))
        fig.update_layout(
            title=f"{symbol} - Strategy vs Buy & Hold",
            xaxis_title="Date",
            yaxis_title="Equity ($1 initial)",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Key stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Strategy Return", f"{bt_metrics['total_return']:.2f}%")
        col2.metric("B&H Return", f"{bt_metrics['bh_return']:.2f}%")
        col3.metric("Outperformance", f"{bt_metrics['total_return'] - bt_metrics['bh_return']:.2f}%")
    
    with tab_metrics:
        st.subheader("Model Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Classification Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
                'Value': [
                    f"{metrics.accuracy:.4f}",
                    f"{metrics.precision:.4f}",
                    f"{metrics.recall:.4f}",
                    f"{metrics.f1:.4f}",
                    f"{metrics.roc_auc:.4f}"
                ]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### Trading Metrics")
            trading_df = pd.DataFrame({
                'Metric': ['Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor', 'Total Return'],
                'Value': [
                    f"{metrics.sharpe_ratio:.4f}",
                    f"{metrics.max_drawdown:.2%}",
                    f"{metrics.win_rate:.2%}",
                    f"{metrics.profit_factor:.4f}",
                    f"{metrics.total_return:.2%}"
                ]
            })
            st.dataframe(trading_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### Cross-Validation Stability")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Accuracy Across Folds**")
            acc_df = pd.DataFrame({
                'Fold': range(1, len(metrics.split_accuracies) + 1),
                'Accuracy': metrics.split_accuracies
            })
            fig = px.bar(acc_df, x='Fold', y='Accuracy', 
                        title=f"Mean: {np.mean(metrics.split_accuracies):.3f} ± {np.std(metrics.split_accuracies):.3f}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Sharpe Ratio Across Folds**")
            sharpe_df = pd.DataFrame({
                'Fold': range(1, len(metrics.split_sharpes) + 1),
                'Sharpe': metrics.split_sharpes
            })
            fig = px.bar(sharpe_df, x='Fold', y='Sharpe',
                        title=f"Mean: {np.mean(metrics.split_sharpes):.3f} ± {np.std(metrics.split_sharpes):.3f}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction distribution
        st.markdown("### Prediction Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=y_proba[y == 0],
            name='Actual Down',
            opacity=0.7,
            nbinsx=50
        ))
        fig.add_trace(go.Histogram(
            x=y_proba[y == 1],
            name='Actual Up',
            opacity=0.7,
            nbinsx=50
        ))
        fig.add_vline(x=custom_threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Threshold: {custom_threshold}")
        fig.update_layout(
            title="Predicted Probability Distribution",
            xaxis_title="Predicted Probability (Up)",
            yaxis_title="Count",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab_backtest:
        st.subheader("Detailed Backtest Analysis")
        
        # Calculate backtest
        signals = pd.Series(y_pred, index=full_df.index)
        aligned_prices = price_df['Adj Close'].reindex(signals.index)
        bt_metrics = calculate_backtest_metrics(aligned_prices, signals, trans_cost)
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Return", f"{bt_metrics['total_return']:.2f}%")
        col2.metric("Sharpe Ratio", f"{bt_metrics['sharpe']:.2f}")
        col3.metric("Max Drawdown", f"{bt_metrics['max_dd']:.2f}%")
        col4.metric("Win Rate", f"{bt_metrics['win_rate']:.1f}%")
        col5.metric("Total Trades", f"{bt_metrics['total_trades']}")
        
        # Equity curve with drawdown
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=("Equity Curve", "Drawdown")
        )
        
        fig.add_trace(
            go.Scatter(x=bt_metrics['strat_curve'].index, y=bt_metrics['strat_curve'].values,
                      name='Strategy', line=dict(color='#2ecc71')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=bt_metrics['bh_curve'].index, y=bt_metrics['bh_curve'].values,
                      name='Buy & Hold', line=dict(color='#3498db')),
            row=1, col=1
        )
        
        # Drawdown
        running_max = bt_metrics['strat_curve'].expanding().max()
        drawdown = (bt_metrics['strat_curve'] - running_max) / running_max * 100
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      name='Drawdown', fill='tozeroy', line=dict(color='#e74c3c')),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Equity", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_layout(height=700, hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly returns heatmap
        st.markdown("### Monthly Returns (%)")
        strat_returns = bt_metrics['strat_curve'].pct_change()
        monthly_returns = strat_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        monthly_pivot = monthly_returns.to_frame('Return')
        monthly_pivot['Year'] = monthly_pivot.index.year
        monthly_pivot['Month'] = monthly_pivot.index.month
        pivot_table = monthly_pivot.pivot(index='Year', columns='Month', values='Return')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=[datetime(2000, m, 1).strftime('%b') for m in pivot_table.columns],
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0,
            text=pivot_table.values.round(1),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Return %")
        ))
        fig.update_layout(title="Monthly Returns Heatmap", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab_predictions:
        st.subheader("Next-Day Prediction")
        
        # Latest prediction
        latest_proba = y_proba[-1]
        latest_pred = y_pred[-1]
        latest_date = full_df.index[-1]
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown(f"### Prediction for {(latest_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}")
            
            color = "#2ecc71" if latest_pred == 1 else "#e74c3c"
            direction = "UP ↗" if latest_pred == 1 else "DOWN ↘"
            st.markdown(f'<h1 style="color: {color};">{direction}</h1>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Probability")
            st.markdown(f'<h1 class="big-metric">{latest_proba:.1%}</h1>', unsafe_allow_html=True)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=latest_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2ecc71" if latest_proba > custom_threshold else "#e74c3c"},
                    'steps': [
                        {'range': [0, custom_threshold * 100], 'color': "lightgray"},
                        {'range': [custom_threshold * 100, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': custom_threshold * 100
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("### Action")
            if latest_pred == 1:
                st.success("✅ GO LONG")
            else:
                st.warning("⚠️ STAY FLAT")
        
        st.markdown("---")
        
        # Recent predictions
        st.markdown("### Recent Predictions (Last 20 Days)")
        recent_df = pd.DataFrame({
            'Date': full_df.index[-20:],
            'Actual': ['Up' if y == 1 else 'Down' for y in y.values[-20:]],
            'Predicted': ['Up' if p == 1 else 'Down' for p in y_pred[-20:]],
            'Probability': [f"{p:.1%}" for p in y_proba[-20:]],
            'Correct': ['✅' if y == p else '❌' for y, p in zip(y.values[-20:], y_pred[-20:])]
        })
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
        
        accuracy_recent = (recent_df['Correct'] == '✅').sum() / len(recent_df)
        st.metric("Recent Accuracy (20 days)", f"{accuracy_recent:.1%}")
    
    with tab_analysis:
        st.subheader("Feature Analysis")
        
        # Feature importance (from Random Forest in ensemble)
        if hasattr(model.models.get('rf'), 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.models['rf'].feature_importances_
            }).sort_values('Importance', ascending=False).head(20)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Top 20 Most Important Features")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation with returns
        st.markdown("### Feature Correlation with Returns")
        correlations = pd.DataFrame({
            'Feature': feature_cols,
            'Correlation': [full_df[feat].corr(returns) for feat in feature_cols]
        }).sort_values('Correlation', key=abs, ascending=False).head(20)
        
        fig = px.bar(correlations, x='Correlation', y='Feature', orientation='h',
                    title="Features Most Correlated with Returns",
                    color='Correlation',
                    color_continuous_scale='RdYlGn')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {str(e)}")
    import traceback
    st.code(traceback.format_exc())


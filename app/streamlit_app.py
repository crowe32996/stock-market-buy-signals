import streamlit as st
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from utils import (
    bucket_probabilities_quantile,
    add_logo_html,
    render_logo_table,
    plot_bucket_curves_plotly,
    compute_spy_forward_returns,
    compute_forward_returns,
    summarize_buckets,
    HORIZONS,
    COLOR_MAP
)

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # app/ -> project_root
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

SIGNALS_CSV = os.path.join(OUTPUT_DIR, "stock_buy_signals_ML.csv")
SPY_CSV = os.path.join(OUTPUT_DIR, "SPY_data.csv")

USE_CSV = os.getenv("USE_CSV", "1") == "1"  # set to "0" to use Postgres

@st.cache_data(ttl=3600)
def load_data():
    if USE_CSV and os.path.exists(SIGNALS_CSV) and os.path.exists(SPY_CSV):
        # Load from CSV
        df_all = pd.read_csv(SIGNALS_CSV)
        df_all['date'] = pd.to_datetime(df_all['date'])
        
        spy_df = pd.read_csv(SPY_CSV)
        spy_df['date'] = pd.to_datetime(spy_df['date'])
    else:
        # Load from Postgres, for once deployed to EC2
        from sqlalchemy import create_engine
        engine = create_engine(
            f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@"
            f"{os.environ['POSTGRES_HOST']}:{os.environ.get('POSTGRES_PORT', 5432)}/{os.environ['POSTGRES_DB']}"
        )
        df_all = pd.read_sql("SELECT * FROM stock_data WHERE symbol!='SPY' ORDER BY date ASC", engine)
        df_all.columns = df_all.columns.str.strip()
        df_all['date'] = pd.to_datetime(df_all['date'])
        
        spy_df = pd.read_sql("SELECT date, close_price FROM stock_data WHERE symbol='SPY' ORDER BY date ASC", engine)
        spy_df['date'] = pd.to_datetime(spy_df['date'])
    
    return df_all, spy_df

@st.cache_data(ttl=3600)
def get_bucket_summary(df, prob_col, bucket_col, max_days):
    df = bucket_probabilities_quantile(df.copy(), prob_col, bucket_col)
    df = compute_forward_returns(df, max_days)
    summary = summarize_buckets(df, bucket_col, max_days)
    return summary

def get_summary_for_tab(df, horizon_key, signal_col, bucket_col, max_days, selected_symbol, precomputed_summaries):
    if selected_symbol == "All":
        return precomputed_summaries[horizon_key]
    else:
        df_filtered = df.copy()
        if 'date' in df.columns and horizon_key == '2023':
            df_filtered = df_filtered[df_filtered['date'].dt.year == 2023]
        df_filtered = df_filtered[df_filtered['symbol'] == selected_symbol]
        return get_bucket_summary(df_filtered, signal_col, bucket_col, max_days)

df_all, spy_df = load_data()

for horizon_name in ['short', 'medium', 'long']:
    prob_col = f"buy_prob_ml_{horizon_name}"
    bucket_col = f'signal_bucket_{horizon_name}'
    df_all = bucket_probabilities_quantile(df_all, prob_col, bucket_col)  # assign directly

# Precompute SPY forward returns
spy_summary = compute_spy_forward_returns(spy_df, max_days=150)
spy_summary_2023 = compute_spy_forward_returns(spy_df[spy_df['date'].dt.year == 2023], max_days=150)

# -----------------------------
# Compute buckets and summaries
# -----------------------------
bucket_summaries = {}
bucket_summaries_2023 = {}

for horizon_name, params in HORIZONS.items():
    prob_col = params['prob_col']
    max_days = params['max_days']
    bucket_col = f'signal_bucket_{horizon_name}'

    # 2024-current
    bucket_summaries[horizon_name] = get_bucket_summary(df_all, prob_col, bucket_col, max_days)

    # 2023 historical
    df_2023 = df_all[df_all['date'].dt.year == 2023].copy()
    bucket_summaries_2023[horizon_name] = get_bucket_summary(df_2023, prob_col, bucket_col, max_days)

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="Stock Buy Signal Analysis", layout="wide")

# Sidebar filters
symbol_options = sorted(df_all['symbol'].unique())
symbol_options.insert(0, "All")
selected_symbol = st.sidebar.selectbox("Select Symbol", symbol_options, index=0)

horizon = st.sidebar.radio("Select Horizon", ["Short", "Medium", "Long"])
horizon_map = {
    "Short": {"prob_col": "buy_prob_ml_short", "max_days": 10},
    "Medium": {"prob_col": "buy_prob_ml_medium", "max_days": 100},
    "Long": {"prob_col": "buy_prob_ml_long", "max_days": 150}
}
selected_horizon = horizon_map[horizon]
signal_col = selected_horizon['prob_col']
horizon_key = horizon.lower()
bucket_col = f'signal_bucket_{horizon_key}'
max_days = selected_horizon['max_days']

# Filter dataframe for selected symbol
if selected_symbol != "All":
    df_filtered = df_all[df_all['symbol'] == selected_symbol].copy()
else:
    df_filtered = df_all.copy()
latest_date = df_filtered['date'].max()
selected_date = st.sidebar.date_input(
    "Select Date",
    value=latest_date,
    min_value=df_filtered['date'].min(),
    max_value=latest_date
)
df_filtered = df_filtered[df_filtered['date'] == pd.to_datetime(selected_date)]
df_filtered['Date'] = df_filtered['date'].dt.strftime("%m/%d/%Y")

# -----------------------------
# Streamlit Tabs
# -----------------------------
st.title("📈 Stock Buy Signal Analysis")
st.markdown(
    "Evaluating the performance of buy signals determined by Random Forest models "
    "across short, medium, and long horizons."
)

tab_recommend, tab_overall, tab_2023, tab_methodology = st.tabs(
    ["Stock Recommendations", "Signal Performance (2024-current)", "2023 Historical Data", "Methodology"]
)

# --- Tab 1: Daily Stock Signals ---
with tab_recommend:
    st.subheader(f"📅 Daily Stock Signals ({horizon} Term)")

    df_sorted = df_all.sort_values(['symbol','date'])
    latest_close_per_symbol = df_sorted.groupby('symbol')['close_price'].last().to_dict()
    df_filtered['Most_Recent_Close'] = df_filtered['symbol'].map(latest_close_per_symbol)

    def render_bucket_table(df, bucket_label):
        bucket_df = df[df[bucket_col] == bucket_label].copy()
        if bucket_df.empty:
            st.write(f"No {bucket_label.capitalize()} signals for this day.")
            return

        bucket_df = bucket_df[['Date','symbol','close_price','Most_Recent_Close']].copy()
        bucket_df = bucket_df.rename(columns={
            'symbol':'Symbol',
            'close_price':'Selected Day Close',
            'Most_Recent_Close':'Most Recent Close'
        })
        bucket_df['Selected Day Close'] = bucket_df['Selected Day Close'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "NA")
        bucket_df['Most Recent Close'] = bucket_df['Most Recent Close'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "NA")
        bucket_df = add_logo_html(bucket_df, symbol_col="Symbol")
        render_logo_table(bucket_df, symbol_col="Symbol", numeric_cols=['Selected Day Close','Most Recent Close'])

    st.markdown("### ✅ Buy Signals")
    render_bucket_table(df_filtered, 'buy')
    st.markdown("### ⚪ Hold Signals")
    render_bucket_table(df_filtered, 'hold')
    st.markdown("### ❌ Sell Signals")
    render_bucket_table(df_filtered, 'sell')

# --- Tab 2: Signal Performance (2024-current) ---
with tab_overall:
    st.subheader(f"Signal Performance 2024-current ({horizon} Horizon)")
    # Filter for selected symbol (or keep all)
    # Use precomputed summaries
    summary_df_filtered = get_summary_for_tab(
        df_all,
        horizon_key,
        signal_col,
        bucket_col,
        max_days,
        selected_symbol,
        bucket_summaries
    )

    fig_avg_return = plot_bucket_curves_plotly(
        summary_df_filtered,
        bucket_col=bucket_col,
        title="Average Return by Signal",
        y_col='avg_return',
        y_label="Average Return",
        spy_summary=spy_summary,
        color_map= COLOR_MAP
    )
    fig_winrate = plot_bucket_curves_plotly(
        summary_df_filtered,
        bucket_col=bucket_col,
        title="Win Rate by Signal",
        y_col='win_rate',
        y_label="Win Rate",
        spy_summary=spy_summary,
        color_map= COLOR_MAP
    )
    st.plotly_chart(fig_avg_return, use_container_width=True)
    st.plotly_chart(fig_winrate, use_container_width=True)

# --- Tab 3: 2023 Historical ---
with tab_2023:
    st.subheader(f"2023 Historical Data ({horizon} Horizon)")
    summary_df_2023_filtered = get_summary_for_tab(
        df_all,
        horizon_key,
        signal_col,
        bucket_col,
        max_days,
        selected_symbol,
        bucket_summaries_2023
    )

    fig_avg_return_2023 = plot_bucket_curves_plotly(
        summary_df_2023_filtered,
        bucket_col=bucket_col,
        title="Average Return by Signal (2023)",
        y_col='avg_return',
        y_label="Average Return",
        spy_summary=spy_summary_2023,
        color_map= COLOR_MAP
    )
    fig_winrate_2023 = plot_bucket_curves_plotly(
        summary_df_2023_filtered,
        bucket_col=bucket_col,
        title="Win Rate by Signal (2023)",
        y_col='win_rate',
        y_label="Win Rate",
        spy_summary=spy_summary_2023,
        color_map= COLOR_MAP
    )
    st.plotly_chart(fig_avg_return_2023, use_container_width=True)
    st.plotly_chart(fig_winrate_2023, use_container_width=True)

with tab_methodology:
    st.title("Methodology")
    
    st.subheader("Random Forest Model")
    st.markdown("""
    We use a **Random Forest classifier** to predict whether a stock will generate a **buy signal** over a given horizon. 

    - A **Random Forest** is an ensemble of decision trees, each trained on random subsets of features and data.
    - Each tree outputs a prediction (buy / not buy), and the final prediction is the **majority vote** of all trees.
    - By training separate models for short, medium, and long horizons, we create different trading strategies for different timeframes.
    - The model outputs a **probability of a buy signal**, which is then bucketed into `buy`, `hold`, or `sell` categories.

    **Advantages:**
    - Handles large numbers of features well.
    - Strong in ruling out noisy indicators.
    - Provides probability/confidence levels, not just binary predictions.
    """)
    
    st.subheader("Feature Inputs in Random Forest Model")
    st.markdown("""
    Determining the confidence of a buy signal required a combination of **technical indicators**, **price/volume metrics**, and **momentum/volatility indicators** to predict short-term, medium-term, and long-term stock movements. The features calculated separately with different timelines for each horizon:

    - **Price & Volume**: the stock's close price and number of trades
    - **Simple Moving Averages (SMA)**: measures average price over different periods to identify trends. Short-term SMAs (3,5,10) capture recent trends, medium (10,20,50) capture mid-term trends, and long-term (20,50,100) capture broader trends.
    - **Exponential Moving Averages (EMA)**: similar to SMA but gives more weight to recent prices.
    - **MACD (Moving Average Convergence Divergence) & Signal Line**: difference between fast and slow EMAs, used to measure momentum.
    - **RSI (Relative Strength Index)**: momentum indicator measured from 0–100. Values less than 30 indicate oversold, and values over 70 indicate overbought. 
    - **Momentum & Volatility Features**: measures of the speed and variability of price changes.
    """)

    st.subheader("Buy Signal Definition")
    st.markdown("""
    Probabilities of buy signal by the Random Forest classifier are bucketed as follows:
    - **Buy**: top 20% probability
    - **Hold**: middle 60%
    - **Sell**: bottom 20%
    """)

    st.subheader("Model Effectiveness")
    st.markdown("""
    The different signals are plotted over time to show how each signal for each time horizon fares. 
    - Note the "Signal Performance" tab to see this plot, and note that this data is all out-of-sample (the model is not learned on any data from this range, this is purely evaluation of the model effectiveness). 
    - The "2023 Historical Data" includes data that is part of the trained range (2021-2023), so returns may show higher due to overfitting.
    """)

    st.markdown("---")
    st.markdown("""
    ⚠️ **Disclaimer**

    The information presented in this application is for **educational and informational purposes only** and **does not constitute financial, investment, or trading advice**. Past performance does not guarantee future results. Users should perform their own due diligence or consult a licensed financial advisor before making any investment decisions.
    """)

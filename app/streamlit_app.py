import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# --- Paths ---
#csv_path = "../output/debug_csvs/stock_buy_signals_ML.csv"  # if running locally
csv_path = "output/stock_buy_signals.csv"
# Map symbols to company domains for logos
BUCKET_THRESHOLDS = {
    'short': {'sell': 0.3, 'hold': 0.65, 'buy': 1.0},
    'medium': {'sell': 0.3, 'hold': 0.65, 'buy': 1.0},
    'long': {'sell': 0.3, 'hold': 0.65, 'buy': 1.0}
}
symbol_to_domain = {
    # Large-cap tech / growth
    'AAPL': 'apple.com',
    'MSFT': 'microsoft.com',
    'GOOGL': 'abc.xyz',
    'AMZN': 'amazon.com',
    'NVDA': 'nvidia.com',
    'META': 'meta.com',
    'TSLA': 'tesla.com',
    'ADBE': 'adobe.com',
    'CRM': 'salesforce.com',
    'NFLX': 'netflix.com',
    'PYPL': 'paypal.com',
    'SHOP': 'shopify.com',
    'SQ': 'block.com',  # Square -> block.com

    # Broad ETFs / indexes (usually no logo for SPY, etc.)
    'SPY': '',
    'QQQ': '',
    'DIA': '',
    'IWM': '',
    'VTI': '',
    'VOO': '',

    # Consumer staples
    'KO': 'coca-cola.com',
    'PEP': 'pepsico.com',
    'PG': 'pg.com',
    'CL': 'clorox.com',
    'MDLZ': 'mondelezinternational.com',
    'COST': 'costco.com',
    'WMT': 'walmart.com',
    'MCD': 'mcdonalds.com',
    'SBUX': 'starbucks.com',
    'YUM': 'yum.com',

    # Financials
    'JPM': 'jpmorganchase.com',
    'BAC': 'bankofamerica.com',
    'C': 'citigroup.com',
    'WFC': 'wellsfargo.com',
    'GS': 'goldmansachs.com',
    'MS': 'morganstanley.com',
    'SCHW': 'schwab.com',
    'AXP': 'americanexpress.com',
    'V': 'visa.com',
    'MA': 'mastercard.com',

    # Healthcare & biotech
    'JNJ': 'jnj.com',
    'PFE': 'pfizer.com',
    'MRK': 'merck.com',
    'ABBV': 'abbvie.com',
    'LLY': 'lilly.com',
    'BMY': 'bms.com',
    'GILD': 'gilead.com',
    'AMGN': 'amgen.com',
    'REGN': 'regeneron.com',
    'VRTX': 'vrtx.com',
    'CVS': 'cvs.com',
    'UNH': 'uhc.com',

    # Industrials & transportation
    'GE': 'ge.com',
    'BA': 'boeing.com',
    'CAT': 'caterpillar.com',
    'DE': 'deere.com',
    'UPS': 'ups.com',
    'FDX': 'fedex.com',
    'LMT': 'lockheedmartin.com',
    'NOC': 'northropgrumman.com',
    'HON': 'honeywell.com',
    'RTX': 'raytheon.com',
    'TM': 'toyota-global.com',
    'GM': 'gm.com',
    'F': 'ford.com',

    # Energy & materials
    'XOM': 'exxonmobil.com',
    'CVX': 'chevron.com',
    'COP': 'conocophillips.com',
    'SLB': 'schlumberger.com',
    'HAL': 'haliburton.com',
    'PSX': 'phillips66.com',
    'MPC': 'marathonpetroleum.com',
    'EOG': 'eogresources.com',
    'PXD': 'pioneerdrilling.com',  # approximation
    'VLO': 'valero.com',
    'BHP': 'bhp.com',
    'RIO': 'riotinto.com',

    # Communication / Media
    'T': 'att.com',
    'VZ': 'verizon.com',
    'CMCSA': 'comcast.com',
    'DIS': 'disney.com',
    'PINS': 'pinterest.com',
    'SNAP': 'snap.com',
    'TWTR': 'twitter.com',
    'BABA': 'alibaba.com',
    'JD': 'jd.com',
    'TCEHY': 'tencent.com',
    'SONY': 'sony.com',

    # REITs / real estate
    'AMT': 'americantower.com',
    'PLD': 'prologis.com',
    'SPG': 'simpsonsproperty.com',  # approximate / could adjust
    'O': 'realtyincome.com',
    'VNQ': 'vanguard.com',
    'DLR': 'digitalrealty.com',
    'EQIX': 'equinix.com',

    # Emerging / midcaps / underperformers
    'UBER': 'uber.com',
    'LYFT': 'lyft.com',
    'ABNB': 'airbnb.com',
    'COIN': 'coinbase.com',
    'RBLX': 'roblox.com',
    'ZM': 'zoom.us',
    'NET': 'cloudflare.com',
    'DOCU': 'docusign.com',
    'ROKU': 'roku.com',
    'PTON': 'onepeloton.com',
    'FSLY': 'fastly.com'
}
PLACEHOLDER_LOGOS = {
    'ADBE': 'https://via.placeholder.com/24?text=ADBE',
    'BAC': 'https://via.placeholder.com/24?text=BAC',
    # You can add more as needed
}
HORIZONS = {
    'short': {'prob_col': 'buy_prob_ml_short', 'max_days': 10},
    'medium': {'prob_col': 'buy_prob_ml_medium', 'max_days': 100},
    'long': {'prob_col': 'buy_prob_ml_long', 'max_days': 150}
}

def add_probability_buckets(df):
    """
    Assign 'sell', 'hold', 'buy' labels based on ML probability columns and BUCKET_THRESHOLDS.
    """
    for horizon in HORIZONS:
        prob_col = f"buy_prob_ml_{horizon}"
        bucket_col = f"signal_bucket_{horizon}"  # new column with sell/hold/buy

        def bucket_label(p):
            if pd.isna(p):
                return np.nan
            if p < BUCKET_THRESHOLDS[horizon]['sell']:
                return 'sell'
            elif p < BUCKET_THRESHOLDS[horizon]['hold']:
                return 'hold'
            else:
                return 'buy'

        df[bucket_col] = df[prob_col].apply(bucket_label)

    return df

def add_logo_html(df, symbol_col="Symbol"):
    """Add HTML for company logos."""
    def logo_html(symbol):
        domain = symbol_to_domain.get(symbol, "")
        if domain:
            url = f"https://logo.clearbit.com/{domain}"
            return f'<img src="{url}" width="24" style="vertical-align:middle;margin-right:4px">{symbol}'
        else:
            return symbol
    df[symbol_col] = df[symbol_col].apply(logo_html)
    return df

def render_logo_table(df, symbol_col="Symbol", numeric_cols=None, max_height=400):
    """Render HTML table with company logos."""
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if c != symbol_col]

    html = f'<div style="max-height:{max_height}px;overflow-y:auto;">'
    html += '<table style="width:100%; border-collapse: collapse;">'
    html += f"<tr><th style='text-align:left'>{symbol_col}</th>"
    for col in numeric_cols:
        html += f"<th style='text-align:right'>{col}</th>"
    html += "</tr>"

    for _, row in df.iterrows():
        html += "<tr>"
        html += f"<td>{row[symbol_col]}</td>"
        for col in numeric_cols:
            val = row[col]
            if isinstance(val, float):
                html += f"<td style='text-align:right'>{val:,.2f}</td>"
            else:
                html += f"<td style='text-align:right'>{val}</td>"
        html += "</tr>"
    html += "</table></div>"

    st.markdown(html, unsafe_allow_html=True)

def bucket_probabilities_quantile(df, prob_col, bucket_col, buy_pct=0.2, sell_pct=0.2):
    """
    Assign buckets so that the top buy_pct fraction is 'buy',
    the bottom sell_pct fraction is 'sell', and the rest is 'hold'.
    """
    df[bucket_col] = 'hold'
    sell_thresh = df[prob_col].quantile(sell_pct)
    buy_thresh = df[prob_col].quantile(1 - buy_pct)
    
    df.loc[df[prob_col] <= sell_thresh, bucket_col] = 'sell'
    df.loc[df[prob_col] >= buy_thresh, bucket_col] = 'buy'
    return df

# --- Load Data --- #
if not os.path.exists(csv_path):
    st.error(f"{csv_path} not found. Make sure the DAG has run and generated the CSV.")
    st.stop()

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date'])

# --- Add quantile-based bucket columns for each horizon --- #
for horizon_name in ['short', 'medium', 'long']:
    prob_col = f"buy_prob_ml_{horizon_name}"
    bucket_col = f"signal_bucket_{horizon_name}"
    df = bucket_probabilities_quantile(df, prob_col, bucket_col, buy_pct=0.2, sell_pct=0.2)

st.set_page_config(page_title="Stock Buy Signal Analysis", layout="wide")
# --- Sidebar filters --- #
# Symbol filter: single select or all
symbol_options = sorted(df['symbol'].unique())
symbol_options.insert(0, "All")  # option to select all symbols
selected_symbol = st.sidebar.selectbox("Select Symbol", symbol_options, index=0)

# Sidebar: Horizon selection
horizon = st.sidebar.radio("Select Horizon", ["Short", "Medium", "Long"])
horizon_map = {
    "Short": {"prob_col": "buy_prob_ml_short", "signal_col": "buy_ml_short", "max_days": 10},
    "Medium": {"prob_col": "buy_prob_ml_medium", "signal_col": "buy_ml_medium", "max_days": 100},
    "Long": {"prob_col": "buy_prob_ml_long", "signal_col": "buy_ml_long", "max_days": 150}
}
selected_horizon = horizon_map[horizon]
signal_col = selected_horizon['signal_col']
prob_col = selected_horizon['prob_col']

if selected_symbol != "All":
    df_filtered = df[df['symbol'] == selected_symbol].copy()
else:
    df_filtered = df.copy()

# --- Date filter: single date picker --- #
latest_date = df_filtered['date'].max()
selected_date = st.sidebar.date_input(
    "Select Date",
    value=latest_date,
    min_value=df_filtered['date'].min(),
    max_value=latest_date
)

# Filter dataframe by the selected date
df_filtered = df_filtered[df_filtered['date'] == pd.to_datetime(selected_date)]
df_filtered['Date'] = df_filtered['date'].dt.strftime("%m/%d/%Y")

if selected_symbol != "All":
    df = df[df['symbol'] == selected_symbol]
else:
    df = df.copy()

# --- Streamlit App ---
st.title("📈 Stock Buy Signal Analysis")
st.markdown("Evaluating the performance of buy signals determined by Random Forest models trained by technical indicators across short, medium, and long horizons.")

tab_recommend, tab_overall, tab_2023, tab_methodology = st.tabs(
    ["Stock Recommendations", "Signal Performance (2024-current)", "2023 Backtest", "Methodology"]
)

# --- Stock Recommendations Tab --- #
with tab_recommend:
    st.subheader(f"📅 Daily Stock Signals ({horizon.capitalize()} Term)")

    # Horizon-specific bucket column
    bucket_col = f"signal_bucket_{horizon.lower()}"

    # Define function to render each bucket table
    def render_bucket_table(df, bucket_label, bucket_col):
        bucket_df = df[df[bucket_col] == bucket_label].copy()
        if bucket_df.empty:
            st.write(f"No {bucket_label.capitalize()} signals for this day.")
            return

        rec_cols = [bucket_col, 'symbol', 'Date', 'open_price', 'close_price']
        bucket_df = bucket_df[rec_cols]
        bucket_df = bucket_df.rename(columns={
            bucket_col: 'Signal',
            'symbol': 'Symbol',
            'open_price': 'Open',
            'close_price': 'Close',
        })

        # Format currency columns
        currency_cols = ['Open', 'Close']
        for c in currency_cols:
            bucket_df[c] = bucket_df[c].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "NA")

        # Add logos and render table
        bucket_df = add_logo_html(bucket_df, symbol_col="Symbol")
        numeric_cols = ['Date', 'Open', 'Close']
        render_logo_table(bucket_df, symbol_col="Symbol", numeric_cols=numeric_cols)

    # Render Buy, Hold, Sell tables
    st.markdown("### ✅ Buy Signals")
    render_bucket_table(df_filtered, 'buy', bucket_col)

    st.markdown("### ⚪ Hold Signals")
    render_bucket_table(df_filtered, 'hold', bucket_col)

    st.markdown("### ❌ Sell Signals")
    render_bucket_table(df_filtered, 'sell', bucket_col)

with tab_overall:
    st.subheader(f"Signal Performance 2024-current ({horizon.capitalize()} Horizon)")
    
    avg_return_file = f"../output/avg_return_buckets_{horizon.lower()}.png"
    winrate_file = f"../output/winrate_buckets_{horizon.lower()}.png"
    
    st.markdown("### Average Return by Signal")
    st.image(avg_return_file, use_container_width=True)

    st.markdown("### Win Rate by Signal")
    st.image(winrate_file, use_container_width=True)

# -------------------------
# Tab 3: 2023 Backtest
# -------------------------
with tab_2023:
    st.subheader(f"2023 Backtest ({horizon.capitalize()} Horizon)")
    
    avg_return_2023_file = f"../output/avg_return_buckets_{horizon.lower()}_2023.png"
    winrate_2023_file = f"../output/winrate_buckets_{horizon.lower()}_2023.png"
    
    st.markdown("### Average Return by Signal (2023)")
    st.image(avg_return_2023_file, use_container_width=True)

    st.markdown("### Win Rate by Signal (2023)")
    st.image(winrate_2023_file, use_container_width=True)

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
    This approach captures both **high-confidence buys** and **uncertain positions**.
    """)

    st.subheader("Model Effectiveness")
    st.markdown("""
    The different signals are plotted over time to show how each signal for each time horizon fares. 
    - Note the "Signal Performance" tab to see this plot, but keep in mind that the Random Classifier requires training data which can result in overfitting. 
    - The "2023 Backtest" tab looks at a time range completely outside of the training dataset to further evaluate performance of the signals.
    - Future versions will increase the backtesting window for further evaluation.
    """)

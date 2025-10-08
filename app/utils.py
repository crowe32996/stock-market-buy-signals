from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# -----------------------------
# Config
# -----------------------------
load_dotenv()

SYMBOL_TO_DOMAIN = {
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

HORIZONS = {
    'short': {'prob_col': 'buy_prob_ml_short', 'max_days': 10},
    'medium': {'prob_col': 'buy_prob_ml_medium', 'max_days': 100},
    'long': {'prob_col': 'buy_prob_ml_long', 'max_days': 150}
}


BUCKET_LABELS = {
    'buy': 'Buy',
    'hold': 'Hold',
    'sell': 'Sell',
    'SPY':'SPY'
}

COLOR_MAP = {'buy': 'green', 'hold': 'blue', 'sell': 'red'}


def add_logo_html(df, symbol_col="Symbol"):
    """Add HTML for company logos."""
    def logo_html(symbol):
        domain = SYMBOL_TO_DOMAIN.get(symbol, "")
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

def compute_spy_forward_returns(spy_df, max_days):
    spy_returns = []
    for h in range(1, max_days + 1):
        spy_df[f'spy_return_{h}'] = spy_df['close_price'].pct_change(periods=h).shift(-h)
        avg_return = spy_df[f'spy_return_{h}'].mean()
        win_rate = (spy_df[f'spy_return_{h}'] > 0).mean()  # <= here is 0/1 for negative returns
        spy_returns.append({'days': h, 'avg_return': avg_return, 'win_rate': win_rate})
    return pd.DataFrame(spy_returns)

def compute_forward_returns(df, max_days):
    """Compute forward returns for 1 - max_days"""
    for h in range(1, max_days + 1):
        df[f'forward_return_{h}'] = df.groupby('symbol')['close_price'].transform(
            lambda x: x.pct_change(periods=h).shift(-h)
        )
    return df


def summarize_buckets(df, bucket_col, max_days):
    """Compute avg return and win rate for each bucket across 1 - max_days"""
    summary_frames = []
    for h in range(1, max_days + 1):
        col = f'forward_return_{h}'
        valid = df.dropna(subset=[col])
        summary = valid.groupby(bucket_col).agg(
            avg_return=(col, 'mean'),
            win_rate=(col, lambda x: (x>0).mean()),
            count=(col, 'size')
        ).reset_index()
        summary['days'] = h
        summary_frames.append(summary)
    return pd.concat(summary_frames, ignore_index=True)

def plot_bucket_curves_plotly(summary_df, bucket_col, title, y_col, spy_summary=None, color_map=None):
    """
    Creates an interactive Plotly chart for bucketed performance.
    
    summary_df: dataframe with bucket performance (avg_return or win_rate)
    bucket_col: column name for bucket labels
    title: chart title
    y_col: column to plot ('avg_return' or 'win_rate')
    spy_summary: optional dataframe for SPY reference line
    """
    fig = go.Figure()
    max_days = summary_df['days'].max()

    # Plot each bucket
    for bucket in summary_df[bucket_col].unique():
        df_plot = summary_df[summary_df[bucket_col] == bucket]
        fig.add_trace(go.Scatter(
            x=df_plot['days'],
            y=df_plot[y_col],
            mode='lines+markers',
            name=BUCKET_LABELS.get(bucket, bucket),
            text=[f"N={int(c)}" for c in df_plot['count']],  # hover text
            hovertemplate="Day %{x}<br>" + y_col + ": %{y}<br>%{text}<extra></extra>",
            line=dict(color=color_map.get(bucket, 'grey')) if color_map else None
        ))

    # Add SPY line if provided
    if spy_summary is not None:
        spy_plot = spy_summary[spy_summary['days'] <= max_days]
        fig.add_trace(go.Scatter(
            x=spy_plot['days'],
            y=spy_plot[y_col],
            mode='lines',
            name='SPY',
            line=dict(color='black', dash='dash', width=2)
        ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Holding Period (Days)",
        yaxis_title=y_col,
        template="plotly_white"
    )

    return fig
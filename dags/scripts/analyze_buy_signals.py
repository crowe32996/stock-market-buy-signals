import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import psycopg2
import plotly.graph_objects as go
from dotenv import load_dotenv

# -----------------------------
# Load ML outputs
# -----------------------------
load_dotenv()

# -----------------------------
# Functions
# -----------------------------
def compute_spy_forward_returns(spy_df, max_days):
    spy_returns = []
    for h in range(1, max_days + 1):
        spy_df[f'spy_return_{h}'] = spy_df['close_price'].pct_change(periods=h).shift(-h)
        avg_return = spy_df[f'spy_return_{h}'].mean()
        win_rate = (spy_df[f'spy_return_{h}'] > 0).mean()  # <= here is 0/1 for negative returns
        spy_returns.append({'days': h, 'avg_return': avg_return, 'win_rate': win_rate})
    return pd.DataFrame(spy_returns)

def bucket_probabilities(df, prob_col, bucket_col, thresholds):
    df[bucket_col] = 'hold'  # default
    df.loc[df[prob_col] < thresholds['sell'], bucket_col] = 'sell'
    df.loc[df[prob_col] >= thresholds['hold'], bucket_col] = 'buy'
    return df

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

def plot_bucket_curves_plotly(summary_df, bucket_col, title, y_col, spy_summary=None):
    """
    Creates an interactive Plotly chart for bucketed performance.
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
            hovertemplate="Day %{x}<br>" + y_col + ": %{y}<br>%{text}<extra></extra>"
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
# -----------------------------
# Process each horizon
# -----------------------------

# -----------------------------
# Parameters
# -----------------------------
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

# Connect to Postgres
conn = psycopg2.connect(
    dbname=os.environ["POSTGRES_DB"],
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
    host=os.environ["POSTGRES_HOST"],
    port=os.environ["POSTGRES_PORT"]
)

# Load SPY prices
spy_df = pd.read_sql("SELECT date, close_price FROM stock_data WHERE symbol='SPY' ORDER BY date ASC", conn)
spy_df['date'] = pd.to_datetime(spy_df['date'])
spy_df = spy_df.sort_values('date').reset_index(drop=True)

df_all = pd.read_sql("SELECT * FROM stock_data WHERE symbol!='SPY' ORDER BY date ASC", conn)
df_all.columns = df_all.columns.str.strip()
df_all['date'] = pd.to_datetime(df_all['date'])

conn.close()

spy_summary = compute_spy_forward_returns(spy_df, max_days=150)

# Create SPY summary for 2023 backtest window
spy_summary_2023 = spy_df[spy_df['date'].dt.year == 2023].copy()
spy_summary_2023 = compute_spy_forward_returns(spy_summary_2023, max_days=150)

for horizon_name, horizon_params in HORIZONS.items():
    prob_col = horizon_params['prob_col']
    max_days = horizon_params['max_days']
    bucket_col = f'signal_{horizon_name}_bucket'
    
    # Assign buckets
    df_all = bucket_probabilities_quantile(df_all, prob_col, bucket_col, buy_pct=0.20, sell_pct=0.20)
    
    # Compute forward returns if not already done
    df_all = compute_forward_returns(df_all, max_days)
    
    # Summarize bucket performance
    summary_df = summarize_buckets(df_all, bucket_col, max_days)
    
    # --- Filter for 2023-only backtest ---
    df_2023 = df_all[df_all['date'].dt.year == 2023]
    # Assign buckets for 2023 using quantiles
    df_2023 = bucket_probabilities_quantile(df_2023, prob_col, bucket_col, buy_pct=0.20, sell_pct=0.20)
    df_2023 = compute_forward_returns(df_2023, max_days)
    summary_df_2023 = summarize_buckets(df_2023, bucket_col, max_days)
    
    # Plot avg return
    plot_bucket_curves(
        summary_df, bucket_col,
        title=f"Return by Signal ({horizon_name.capitalize()} Horizon)",
        y_col='avg_return',
        ylabel="Average Return",
        output_file=f"/opt/airflow/volumes/output/avg_return_buckets_{horizon_name}.png",
        spy_summary=spy_summary
    )

    # Plot win rate
    plot_bucket_curves(
        summary_df, bucket_col,
        title=f"Win Rate by Signal ({horizon_name.capitalize()} Horizon)",
        y_col='win_rate',
        ylabel="Win Rate",
        output_file=f"/opt/airflow/volumes/output/winrate_buckets_{horizon_name}.png",
        spy_summary=spy_summary
    )

    # Plot avg return (2023 backtest)
    plot_bucket_curves(
        summary_df_2023, bucket_col,
        title=f"Return by Signal - 2023 Backtest ({horizon_name.capitalize()} Horizon)",
        y_col='avg_return',
        output_file=f"/opt/airflow/volumes/output/avg_return_buckets_{horizon_name}_2023.png",
        spy_summary=spy_summary_2023
    )

    # Plot win rate (2023 backtest)
    plot_bucket_curves(
        summary_df_2023, bucket_col,
        title=f"Win Rate by Signal - 2023 Backtest ({horizon_name.capitalize()} Horizon)",
        y_col='win_rate',
        output_file=f"/opt/airflow/volumes/output/winrate_buckets_{horizon_name}_2023.png",
        spy_summary=spy_summary_2023
    )

print("✅ All horizons processed. Plots and CSVs saved to /opt/airflow/volumes/output/")
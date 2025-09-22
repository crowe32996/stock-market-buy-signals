import pandas as pd
import plotly.express as px
import os

os.makedirs("output", exist_ok=True)

# Load data
csv_path = "/opt/airflow/volumes/output/stock_buy_signals.csv"  # adjust if needed
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found. Make sure the DAG has run and generated the CSV.")

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'])

# Sort ascending by date within each symbol
df = df.sort_values(['symbol', 'date'])

# Define return horizons
horizons = [1, 5, 10, 20, 30]
return_frames = []

for h in horizons:
    col = f'return_{h}d'
    df[col] = df.groupby('symbol')['close_price'].transform(
        lambda x: x.pct_change(periods=h).shift(-h)
    )

    valid = df.dropna(subset=[col])
    valid = valid[valid['buy_signal'].isin([True, False])]

    summary = valid.groupby('buy_signal').agg(
        count=(col, 'size'),
        avg_return=(col, 'mean'),
        median_return=(col, 'median'),
        win_rate=(col, lambda x: (x > 0).mean())
    ).reset_index()

    summary['return_period'] = col
    return_frames.append(summary)

summary = pd.concat(return_frames, ignore_index=True)

# Add numeric days column for sorting and sort
summary['days'] = summary['return_period'].str.extract(r'(\d+)').astype(int)
summary = summary.sort_values(by=['days', 'buy_signal']).reset_index(drop=True)

summary['buy_signal_str'] = summary['buy_signal'].map({True: 'Buy Signal = TRUE', False: 'Buy Signal = FALSE'})

x_order = [f'return_{d}d' for d in horizons]

fig_avg = px.bar(
    summary,
    x='return_period',
    y='avg_return',
    color='buy_signal_str',
    barmode='group',
    category_orders={'return_period': x_order, 'buy_signal_str': ['Buy Signal = FALSE', 'Buy Signal = TRUE']},
    labels={
        'return_period': 'Return Period',
        'avg_return': 'Average Return',
        'buy_signal_str': 'Buy Signal'
    },
    title='Average Return by Buy Signal and Return Period',
    text='count' 
)

fig_avg.update_traces(texttemplate='n=%{text}', textposition='outside')
fig_avg.update_layout(
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    xaxis_tickangle=-45,
    yaxis_tickformat=".4f",
    yaxis=dict(tickformat=".2%"),
    legend_title_text='Buy Signal'
)

fig_avg.write_image("/opt/airflow/volumes/output/avg_return_by_buy_signal.png")

# --- Plot Win Rate ---
fig_win = px.bar(
    summary,
    x='return_period',
    y='win_rate',
    color='buy_signal_str',
    barmode='group',
    category_orders={'return_period': x_order, 'buy_signal_str': ['Buy Signal = FALSE', 'Buy Signal = TRUE']},
    labels={
        'return_period': 'Return Period',
        'win_rate': 'Win Rate',
        'buy_signal_str': 'Buy Signal'
    },
    title='Win Rate by Buy Signal and Return Period',
    text='count'  # show counts on bars
)

fig_win.update_traces(texttemplate='n=%{text}', textposition='outside')
fig_win.update_layout(
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    xaxis_tickangle=-45,
    yaxis=dict(range=[0,1], tickformat=".0%"),
    legend_title_text='Buy Signal'
)

fig_win.write_image("/opt/airflow/volumes/output/win_rate_by_buy_signal.png")

print("✅ Plotly analysis complete. Files saved to output/")

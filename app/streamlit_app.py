import streamlit as st
import pandas as pd
import plotly.express as px
import os


# --- Paths ---
#csv_path = "../output/stock_buy_signals.csv"  # if running locally
csv_path = "output/stock_buy_signals.csv"
# Map symbols to company domains for logos
symbol_to_domain = {
    'MSFT': 'microsoft.com', 'TSLA': 'tesla.com', 'NVDA': 'nvidia.com',
    'META': 'meta.com', 'GOOGL': 'abc.xyz', 'AMZN': 'amazon.com',
    'AMD': 'amd.com', 'UBER': 'uber.com', 'PLTR': 'palantir.com',
    'SHOP': 'shopify.com', 'CRM': 'salesforce.com', 'AAPL': 'apple.com',
    'NFLX': 'netflix.com', 'ABNB': 'airbnb.com', 'COIN': 'coinbase.com',
    'LYFT': 'lyft.com', 'DIS': 'disney.com', 'BAC': 'bankofamerica.com',
    'INTC': 'intel.com', 'KO': 'coca-cola.com', 'PEP': 'pepsico.com',
    'CSCO': 'cisco.com', 'XOM': 'exxonmobil.com', 'WMT': 'walmart.com',
    'PG': 'pg.com', 'PFE': 'pfizer.com'
}

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


# --- Load Data ---
if not os.path.exists(csv_path):
    st.error(f"{csv_path} not found. Make sure the DAG has run and generated the CSV.")
    st.stop()

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date'])

st.set_page_config(page_title="Stock Buy Signal Analysis", layout="wide")

# --- Sidebar filters --- #
# Symbol filter: single select or all
symbol_options = sorted(df['symbol'].unique())
symbol_options.insert(0, "All")  # option to select all symbols
selected_symbol = st.sidebar.selectbox("Select Symbol", symbol_options, index=0)

if selected_symbol != "All":
    df = df[df['symbol'] == selected_symbol]
else:
    df = df.copy()

# Date filter: range selector
min_date = df['date'].min()
max_date = df['date'].max()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Filter dataframe by selected date range
df = df[(df['date'] >= pd.to_datetime(start_date)) &
                          (df['date'] <= pd.to_datetime(end_date))]

# --- Define return horizons ---
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
summary['days'] = summary['return_period'].str.extract(r'(\d+)').astype(int)
summary = summary.sort_values(by=['days', 'buy_signal']).reset_index(drop=True)
summary['buy_signal_str'] = summary['buy_signal'].map({True: 'Buy Signal = TRUE', False: 'Buy Signal = FALSE'})

return_name_map = {
    'return_1d': 'Return L1',
    'return_5d': 'Return L5',
    'return_10d': 'Return L10',
    'return_20d': 'Return L20',
    'return_30d': 'Return L30'
}

summary['return_period_display'] = summary['return_period'].map(return_name_map)

# Update category order for plots
x_order_display = list(return_name_map.values())

# --- Streamlit App ---
st.title("📈 Stock Buy Signal Analysis")
st.markdown("Visualizing average returns, win rates, and raw data by buy signals.")

tab_recommend, tab_avg, tab_win, tab_data = st.tabs(["Stock Recommendations", "Average Return", "Win Rate", "Raw Data"])

# --- Stock Recommendations Tab --- #
with tab_recommend:
    st.subheader("✅ Most Recent Buy Signal for Each Stock:")

    buy_df = df[df['buy_signal'] == True].copy()

    # Keep only the most recent buy signal per stock
    buy_df = buy_df.sort_values(by=['symbol', 'date'], ascending=[True, False])
    buy_df = buy_df.groupby('symbol').head(1).reset_index(drop=True)
    buy_df = buy_df.sort_values(by='date', ascending=False).reset_index(drop=True)
    buy_df['Date'] = buy_df['date'].dt.strftime("%m/%d/%Y")

    # Select simplified columns and rename
    rec_cols = ['buy_signal', 'symbol', 'Date', 'open_price', 'close_price',
                'SMA_10', 'SMA_50', 'RSI', 'MACD', 'Signal',
                'return_1d', 'return_10d', 'return_30d']
    buy_df = buy_df[rec_cols]
    buy_df = buy_df.rename(columns={
        'buy_signal': 'Buy Signal',
        'symbol': 'Symbol',
        'open_price': 'Open',
        'close_price': 'Close',
        'SMA_10': 'SMA L10',
        'SMA_50': 'SMA L50',
        'RSI': 'RSI',
        'MACD': 'MACD',
        'Signal': 'Signal'
    })

    # Reorder columns
    buy_df = buy_df[['Buy Signal', 'Symbol', 'Date', 'Open', 'Close',
                     'SMA L10', 'SMA L50', 'RSI', 'MACD', 'Signal']]

    # Format currency columns
    currency_cols = ['Open', 'Close', 'SMA L10', 'SMA L50']
    for c in currency_cols:
        buy_df[c] = buy_df[c].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "NA")

    # Round RSI, MACD, Signal to 2 decimals
    round_cols = ['RSI', 'MACD', 'Signal']
    for c in round_cols:
        buy_df[c] = buy_df[c].round(2)

    # --- Add logos and render HTML table ---
    buy_df = add_logo_html(buy_df, symbol_col="Symbol")
    numeric_cols = ['Date', 'Open', 'Close', 'SMA L10', 'SMA L50', 'RSI', 'MACD', 'Signal']
    render_logo_table(buy_df, symbol_col="Symbol", numeric_cols=numeric_cols)

# --- Avg Return Plot ---
with tab_avg:
    fig_avg = px.bar(
        summary,
        x='return_period_display',  # use display names
        y='avg_return',
        color='buy_signal_str',
        barmode='group',
        category_orders={
            'return_period_display': x_order_display,  # updated
            'buy_signal_str': ['Buy Signal = FALSE', 'Buy Signal = TRUE']
        },
        labels={
            'return_period_display': 'Return Period',
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
        yaxis_tickformat=".2%",
        legend_title_text='Buy Signal'
    )
    st.plotly_chart(fig_avg, use_container_width=True)

# --- Win Rate Plot ---
with tab_win:
    fig_win = px.bar(
        summary,
        x='return_period_display',  # use new display names
        y='win_rate',
        color='buy_signal_str',
        barmode='group',
        category_orders={'return_period_display': x_order_display,
                         'buy_signal_str': ['Buy Signal = FALSE', 'Buy Signal = TRUE']},
        labels={'return_period_display': 'Return Period',
                'win_rate': 'Win Rate',
                'buy_signal_str': 'Buy Signal'},
        title='Win Rate by Buy Signal and Return Period',
        text='count'
    )
    fig_win.update_traces(texttemplate='n=%{text}', textposition='outside')
    fig_win.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        xaxis_tickangle=-45,
        yaxis=dict(range=[0,1], tickformat=".0%"),
        legend_title_text='Buy Signal'
    )
    st.plotly_chart(fig_win, use_container_width=True)

# --- Raw Data Table ---
with tab_data:
    # Copy df to avoid modifying original
    df_display = df.copy()
    df_display = df_display.drop(columns=['id'])

    # Rename columns
    df_display = df_display.rename(columns={
        'symbol': 'Symbol',
        'open_price': 'Open',
        'close_price': 'Close',
        'high_price': 'High',
        'low_price': 'Low',
        'volume': 'Volume',
        'date': 'Date',
        'SMA_10': 'SMA L10',
        'SMA_50': 'SMA L50',
        'RSI': 'RSI',
        'MACD': 'MACD',
        'Signal': 'Signal',
        'buy_signal': 'Buy Signal',
        'return_1d': 'Return L1',
        'return_5d': 'Return L5',
        'return_10d': 'Return L10',
        'return_20d': 'Return L20',
        'return_30d': 'Return L30'
    })

    # Reorder columns
    cols_order = [
        'Symbol', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume',
        'SMA L10', 'SMA L50', 'RSI', 'MACD', 'Signal', 'Buy Signal',
        'Return L1', 'Return L5', 'Return L10', 'Return L20', 'Return L30'
    ]
    df_display = df_display[cols_order]

    # Format currency columns, show 'NA' if value is missing
    currency_cols = ['Open', 'Close', 'High', 'Low', 'SMA L10', 'SMA L50']
    for c in currency_cols:
        df_display[c] = df_display[c].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "NA")

    # Round RSI, MACD, Signal to 2 decimals
    round_cols = ['RSI', 'MACD', 'Signal']
    for c in round_cols:
        df_display[c] = df_display[c].round(2)

    # Format returns as percentages, show 'NA' if value is missing
    return_cols = ['Return L1', 'Return L5', 'Return L10', 'Return L20', 'Return L30']
    for c in return_cols:
        df_display[c] = df_display[c].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "NA")
    
    df_display = df_display.sort_values(by=['Date', 'Symbol'], ascending=[False, True]).reset_index(drop=True)

    # Display in Streamlit
    st.dataframe(df_display, use_container_width=True)

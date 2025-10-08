import os
import psycopg2
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

# -----------------------------
# Config
# -----------------------------
load_dotenv()
OUTPUT_DIR = "/opt/airflow/volumes/output"
OUTPUT_ML_CSV = os.path.join(OUTPUT_DIR, "stock_buy_signals_ML.csv")

POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)

HORIZONS = {
    'short': 5,    # days forward
    'medium': 20,  # days forward
    'long': 60     # days forward
}

FEATURES_BY_HORIZON = {
    'short': [
        'close_price', 'volume',
        'SMA_3','SMA_5','SMA_10',
        'EMA_3','EMA_5','EMA_10',
        'MACD_short','Signal_short',
        'momentum_short_1','momentum_short_3',
        'volatility_short_3','volatility_short_5',
        'RSI_short_3','RSI_short_7',
        'score','buy_signal'
    ],
    'medium': [
        'close_price', 'volume',
        'SMA_10','SMA_20','SMA_50',
        'EMA_10','EMA_26',
        'MACD_medium','Signal_medium',
        'momentum_medium_5','momentum_medium_10',
        'volatility_medium_10','volatility_medium_20',
        'RSI_medium_14',
        'score','buy_signal'
    ],
    'long': [
        'close_price', 'volume',
        'SMA_20','SMA_50','SMA_100',
        'EMA_26','EMA_50',
        'MACD_long','Signal_long',
        'momentum_long_20','momentum_long_50',
        'volatility_long_50','volatility_long_100',
        'RSI_long_28',
        'score','buy_signal'
    ]
}

# -----------------------------
# Database connection
# -----------------------------
conn = psycopg2.connect(
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT
)
cursor = conn.cursor()

def compute_indicators(df):
    df = df.sort_values("date").copy()

    # -----------------------------
    # Moving averages (common)
    # -----------------------------
    for span in [3,5,10,20,26,50,100]:
        df[f'SMA_{span}'] = df['close_price'].rolling(span).mean()
        df[f'EMA_{span}'] = df['close_price'].ewm(span=span, adjust=False).mean()

    # -----------------------------
    # Horizon-specific MACD & Signal
    # -----------------------------
    MACD_PARAMS = {
        'short': {'fast': 3, 'slow': 8, 'signal': 3},
        'medium': {'fast': 12, 'slow': 26, 'signal': 9},
        'long': {'fast': 26, 'slow': 100, 'signal': 18}
    }

    for horizon, params in MACD_PARAMS.items():
        df[f'MACD_{horizon}'] = df['close_price'].ewm(span=params['fast'], adjust=False).mean() - \
                                df['close_price'].ewm(span=params['slow'], adjust=False).mean()
        df[f'Signal_{horizon}'] = df[f'MACD_{horizon}'].ewm(span=params['signal'], adjust=False).mean()

    # -----------------------------
    # Momentum (diffs) by horizon
    # -----------------------------
    MOM_PERIODS = {'short':[1,3], 'medium':[5,10], 'long':[20,50]}
    for horizon, periods in MOM_PERIODS.items():
        for p in periods:
            df[f'momentum_{horizon}_{p}'] = df['close_price'].diff(p)

    # -----------------------------
    # Volatility (rolling std)
    # -----------------------------
    VOL_PERIODS = {'short':[3,5], 'medium':[10,20], 'long':[50,100]}
    for horizon, periods in VOL_PERIODS.items():
        for p in periods:
            df[f'volatility_{horizon}_{p}'] = df['close_price'].rolling(p).std()

    # -----------------------------
    # RSI by horizon
    # -----------------------------
    RSI_PERIODS = {'short':[3,7], 'medium':[14], 'long':[28]}
    delta = df['close_price'].diff()
    gain = np.where(delta>0, delta, 0)
    loss = np.where(delta<0, -delta, 0)

    for horizon, periods in RSI_PERIODS.items():
        for p in periods:
            avg_gain = pd.Series(gain).rolling(p).mean()
            avg_loss = pd.Series(loss).rolling(p).mean()
            df[f'RSI_{horizon}_{p}'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # -----------------------------
    # Score & buy_signal (existing)
    # -----------------------------
    df['uptrend'] = (df['SMA_10'] > df['SMA_20'])
    df['vol_confirm'] = df['volume'] > df['volume'].rolling(5).mean()
    df['score'] = df[['uptrend','vol_confirm']].astype(int).sum(axis=1)
    df['buy_signal'] = df['score'] >= 2

    return df

def train_ml(df, horizon_days, spy_df, horizon_name):
    """
    Train RF on 2024–2025 data and backtest on 2023, but predict ML probabilities for all years.
    """
    # Remove duplicate SPY columns
    df = df.drop(columns=[col for col in df.columns if 'spy_close' in col], errors='ignore')

    # Merge SPY
    spy_df = spy_df.drop_duplicates(subset=['date']).copy()
    spy_df = spy_df.rename(columns={'close_price': 'spy_close'})
    df = df.merge(spy_df[['date', 'spy_close']], on='date', how='left')

    # Compute forward returns (aligned)
    df['stock_forward_return'] = df.groupby('symbol')['close_price'].shift(-horizon_days) / df['close_price'] - 1
    df['spy_forward_return'] = df['spy_close'].shift(-horizon_days) / df['spy_close'] - 1
    df['outperformed'] = (df['stock_forward_return'] > df['spy_forward_return']).astype(int)

    df['date'] = pd.to_datetime(df['date'])

    # Training on 2024+ only
    train_df = df[df['date'] >= '2024-01-01']
    # Evaluation on 2023 only
    test_eval_df = df[(df['date'] >= '2023-01-01') & (df['date'] < '2024-01-01')]

    # Features for ML
    features = [f for f in FEATURES_BY_HORIZON[horizon_name] if f not in ('score','buy_signal')]

    # Drop NA
    train_df = train_df.dropna(subset=features + ['outperformed'])
    if train_df.empty:
        print(f"⚠️ No training data for {horizon_name} horizon.")
        df[f'buy_prob_ml_{horizon_name}'] = np.nan
        return df, None

    # Train RandomForest
    X_train, y_train = train_df[features], train_df['outperformed']
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)

    # ✅ Predict ML probabilities for all rows (after dropping NA features)
    predict_df = df.dropna(subset=features)
    df[f'buy_prob_ml_{horizon_name}'] = np.nan
    if not predict_df.empty:
        X_pred = predict_df[features]
        df.loc[predict_df.index, f'buy_prob_ml_{horizon_name}'] = rf.predict_proba(X_pred)[:, 1]

    # df is now ready for signals across all years
    return df, rf


def evaluate_signals(df, horizon_name):
    """
    Evaluate buy signals vs SPY using date-aligned forward returns.
    """
    forward_return_col = f'{horizon_name}_forward_return'
    # Compute aligned forward return for stocks
    df[forward_return_col] = df.groupby('symbol')['close_price'].shift(-HORIZONS[horizon_name]) / df['close_price'] - 1
    # SPY forward return (already merged)
    df['spy_forward_return'] = df['spy_close'].shift(-HORIZONS[horizon_name]) / df['spy_close'] - 1

    signal_returns = df.loc[df['buy_signal']==True, forward_return_col]
    spy_returns = df['spy_forward_return']

    results = {
        'horizon': horizon_name,
        'n_signals': signal_returns.count(),
        'avg_signal_return': signal_returns.mean(),
        'avg_spy_return': spy_returns.mean()
    }
    return results

def main(symbols):
    all_dfs = []
    
    # Fetch SPY
    cursor.execute("SELECT date, close_price FROM stock_data WHERE symbol='SPY' ORDER BY date ASC")
    spy_rows = cursor.fetchall()
    spy_df = pd.DataFrame(spy_rows, columns=['date','close_price'])
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    
    for symbol in symbols:
        cursor.execute("SELECT * FROM stock_data WHERE symbol=%s ORDER BY date ASC", (symbol,))
        rows = cursor.fetchall()
        if not rows:
            print(f"No data for {symbol}, skipping.")
            continue
        
        df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
        df['date'] = pd.to_datetime(df['date'])
        
        # Explicitly add 'symbol' column
        df['symbol'] = symbol
        
        df = compute_indicators(df)
        all_dfs.append(df)
    
    # Debug: check if any DataFrames were fetched
    if not all_dfs:
        print("⚠️ No stock data fetched for any symbol. all_dfs is empty!")
    else:
        print(f"Fetched {len(all_dfs)} stock DataFrames, row counts per symbol:")
        for df in all_dfs:
            symbol = df['symbol'].iloc[0] if not df.empty else 'UNKNOWN'
            print(f"  {symbol}: {len(df)} rows, columns: {list(df.columns)}")

    df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    print(f"After concat, df_all shape: {df_all.shape}, columns: {list(df_all.columns)}")
    
    eval_results = []
    for name in HORIZONS:
        df_all, rf_model = train_ml(df_all, HORIZONS[name], spy_df, horizon_name=name)
        result = evaluate_signals(df_all, horizon_name=name)
        eval_results.append(result)
    
    print("Evaluation results vs SPY:")
    print(pd.DataFrame(eval_results))
    
    # Write CSVs
    df_all.to_csv(OUTPUT_ML_CSV, index=False)
    print(f"✅ CSV written: {OUTPUT_ML_CSV}")
    
    return df_all, eval_results

def add_threshold_signals_and_forward_returns(df, threshold=0.75, max_days=150):
    """
    Compute forward returns 1..max_days and generate binary buy signals
    based on ML probabilities for each horizon.
    """
    # Compute forward returns for 1..max_days
    for horizon in range(1, max_days + 1):
        df[f'forward_return_{horizon}'] = df['close_price'].pct_change(horizon).shift(-horizon)

    # Compute binary buy signals for each ML horizon
    for ml_horizon in HORIZONS:
        prob_col = f'buy_prob_ml_{ml_horizon}'
        signal_col = f'buy_ml_{ml_horizon}'
        df[signal_col] = df[prob_col] >= threshold

    return df

def update_stock_data(df, cursor, conn):
    update_sql = """
        UPDATE stock_data
        SET
            buy_prob_ml_short = %s,
            buy_prob_ml_medium = %s,
            buy_prob_ml_long = %s,
            buy_ml_short = %s,
            buy_ml_medium = %s,
            buy_ml_long = %s
        WHERE symbol = %s AND date = %s
    """
    for _, row in df.iterrows():
        cursor.execute(update_sql, (
            row['buy_prob_ml_short'],
            row['buy_prob_ml_medium'],
            row['buy_prob_ml_long'],
            row['buy_ml_short'],
            row['buy_ml_medium'],
            row['buy_ml_long'],
            row['symbol'],
            row['date']
        ))
    conn.commit()

if __name__ == "__main__":
    cursor.execute("SELECT DISTINCT symbol FROM stock_data WHERE symbol != 'SPY'")
    symbols = [row[0] for row in cursor.fetchall()]
    df_all, eval_results = main(symbols)
    # Add threshold-based buy signals and forward returns
    threshold = 0.75
    df_all = add_threshold_signals_and_forward_returns(df_all, threshold=threshold, max_days=150)
    update_stock_data(df_all, cursor, conn)

    query = """
        SELECT *
        FROM stock_data
        WHERE symbol != 'SPY'
        ORDER BY symbol, date
    """
    df_all = pd.read_sql(query, conn)  # uses SQLAlchemy or psycopg2 + pandas
    df_all['date'] = pd.to_datetime(df_all['date'])

    # Export to CSV
    df_all.to_csv(OUTPUT_ML_CSV, index=False)
    print(f"✅ CSV written from DB: {OUTPUT_ML_CSV}")

    cursor.close()
    conn.close()
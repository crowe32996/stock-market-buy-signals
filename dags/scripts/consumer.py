import psycopg2
import pandas as pd
import numpy as np
from kafka import KafkaConsumer
import json
from datetime import datetime
import time
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import time
import os
from dotenv import load_dotenv

load_dotenv()


for attempt in range(10):
    try:
        consumer = KafkaConsumer(
            'stock_data',
            group_id='stock_consumer_group',
            bootstrap_servers=['kafka:9092'],
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            consumer_timeout_ms=10000,  # Exit if no new messages in 10 sec
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        print("Connected to Kafka!")
        break
    except NoBrokersAvailable:
        print(f"Kafka not available yet (attempt {attempt+1}), retrying...")
        time.sleep(3)
else:
    raise Exception("Failed to connect to Kafka after 10 attempts")

POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)

conn = psycopg2.connect(
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT
)
cursor = conn.cursor()

def store_stock_data(record):
    try:
        timestamp = datetime.strptime(record['date'], '%Y-%m-%d')

        cursor.execute("""
            SELECT 1 FROM stock_data WHERE symbol = %s AND date = %s
        """, (record['symbol'], timestamp))
        existing = cursor.fetchone()

        if existing:
            print(f"Skipping {record['symbol']} for {record['date']} (already exists).")
            return False

        cursor.execute("""
            INSERT INTO stock_data (symbol, open_price, close_price, high_price, low_price, volume, date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            record['symbol'], record['open_price'], record['close_price'],
            record['high_price'], record['low_price'], record['volume'], timestamp
        ))
        conn.commit()
        print(f"Stored record for {record['symbol']} at {record['date']}")
        return True
    except Exception as e:
        print(f"Error inserting record: {record}. Error: {e}")
        conn.rollback()
        return False

def compute_and_update_indicators(symbol):
    query = """
        SELECT id, close_price, volume, date FROM stock_data
        WHERE symbol = %s ORDER BY date ASC
    """
    cursor.execute(query, (symbol,))
    data = cursor.fetchall()
    print(f"Fetched {len(data)} rows for {symbol}")

    if not data or len(data) < 50:
        print(f"Not enough data to compute indicators for {symbol}. Skipping...")
        return

    df = pd.DataFrame(data, columns=['id', 'close_price', 'volume', 'timestamp'])

    df['SMA_10'] = df['close_price'].rolling(window=10).mean()
    df['SMA_50'] = df['close_price'].rolling(window=50).mean()

    delta = df['close_price'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    df['EMA_12'] = df['close_price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close_price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Add buy_signal logic
    df['buy_signal'] = (
        (df['MACD'] > df['Signal']) &
        (df['RSI'] < 65) &
        (df['SMA_10'] > df['SMA_50']) &
        (df['close_price'] > df['SMA_10'])
    )

    for _, row in df.iterrows():
        cursor.execute("""
            UPDATE stock_data
            SET SMA_10 = %s, SMA_50 = %s, RSI = %s, MACD = %s, Signal = %s, buy_signal = %s
            WHERE id = %s
        """, (
            row['SMA_10'], row['SMA_50'], row['RSI'], row['MACD'],
            row['Signal'], row['buy_signal'], row['id']
        ))

    conn.commit()
    print(f"Indicators + buy_signal updated for {symbol}")

if __name__ == "__main__":
    print("Consumer is starting...")
    processed_symbols = set()
    messages_consumed = 0
    max_messages = 1000

    try:
        for message in consumer:
            record = message.value
            print(f"Consuming: {record}")

            if record.get('end_of_data'):
                print("End-of-data signal received. Exiting loop.")
                break

            success = store_stock_data(record)
            if success:
                processed_symbols.add(record['symbol'])
                messages_consumed += 1
                print(f"Processed symbols: {processed_symbols} | Messages consumed: {messages_consumed}")

            if messages_consumed >= max_messages:
                print(f"Reached maximum message limit ({max_messages}). Exiting loop.")
                break

        for symbol in processed_symbols:
            print(f"Computing indicators for {symbol}...")
            compute_and_update_indicators(symbol)

    except Exception as e:
        print(f"Error consuming message: {e}")
    finally:
        print("Closing consumer...")
        cursor.close()
        conn.close()
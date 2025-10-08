import psycopg2
from kafka import KafkaConsumer
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Kafka config
KAFKA_BROKER = 'kafka:9092'
TOPIC = 'stock_data'

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    group_id='stock_consumer_batch_3',
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    consumer_timeout_ms=10000,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Connected to Kafka!")

# PostgreSQL config
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

BATCH_SIZE = 5000
batch = []

def store_stock_data_batch(record):
    try:
        timestamp = datetime.strptime(record['date'], '%Y-%m-%d')
        batch.append((
            record['symbol'], record['open_price'], record['close_price'],
            record['high_price'], record['low_price'], record['volume'], timestamp
        ))

        # Insert batch if full
        if len(batch) >= BATCH_SIZE:
            insert_batch(batch)
            batch.clear()

        return True
    except Exception as e:
        print(f"Error appending record: {e}")
        return False

def insert_batch(batch_to_insert):
    try:
        cursor.executemany("""
            INSERT INTO stock_data (symbol, open_price, close_price, high_price, low_price, volume, date)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (symbol, date) DO NOTHING
        """, batch_to_insert)
        conn.commit()
        print(f"Inserted batch of {len(batch_to_insert)} records")
    except Exception as e:
        print(f"Error inserting batch: {e}")
        conn.rollback()

if __name__ == "__main__":
    messages_consumed = 0
    try:
        for message in consumer:
            record = message.value
            store_stock_data_batch(record)
            messages_consumed += 1

        # Insert remaining records
        if batch:
            insert_batch(batch)

    except Exception as e:
        print(f"Error consuming messages: {e}")
    finally:
        cursor.close()
        conn.close()
        print(f"Consumer finished. Total messages consumed: {messages_consumed}")
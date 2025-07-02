import os
import time
import json
import requests
from kafka import KafkaProducer

# Configuration
API_KEY = 'M1D7S2117PSVU5OW'  # Ensure you set your API key as an environment variable
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
KAFKA_BROKER = '172.31.38.186:9092'  # Your Kafka broker
TOPIC = 'stock_data'  # Kafka topic where the stock data will be sent
MAX_REQUESTS_PER_DAY = 25  # Alpha Vantage free limit

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Variable to track the number of requests made today
request_count = 0

# Load the current request count from a file (optional, can be persistent)
def load_request_count():
    global request_count
    try:
        with open("request_count.txt", "r") as f:
            request_count = int(f.read())
        print(f"Loaded request count: {request_count}")  # Debugging output
    except FileNotFoundError:
        request_count = 0
        print(f"Request count file not found. Starting from 0.")  # Debugging output

# Save the request count to a file (optional, to persist across script runs)
def save_request_count():
    global request_count
    with open("request_count.txt", "w") as f:
        f.write(str(request_count))
    print(f"Saved request count: {request_count}")  # Debugging output

def can_make_request():
    global request_count
    print(f"[DEBUG] Request count before check: {request_count}")
    #if request_count >= MAX_REQUESTS_PER_DAY:
        #print(f"Daily API request limit ({MAX_REQUESTS_PER_DAY}) reached. Skipping API call.")
        #return False
    return True

def fetch_stock_data(symbol):
    global request_count

    if not can_make_request():
        return None

    request_count += 1

    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(ALPHA_VANTAGE_URL, params=params)
    data = response.json()

    if "Time Series (Daily)" in data:
        save_request_count()
        return data["Time Series (Daily)"]
    else:
        print(f"Error fetching data for {symbol}: {data}")
        return None

def produce_stock_data(symbol):
    stock_data = fetch_stock_data(symbol)

    if stock_data:
        #recent_dates = sorted(stock_data.keys(), reverse = True)[:8]
        #for date in recent_dates:
        for date, values in stock_data.items():
            #values = stock_data[date]
            record = {
                'symbol': symbol,
                'date': date,
                'open_price': float(values['1. open']),
                'close_price': float(values['4. close']),
                'high_price': float(values['2. high']),
                'low_price': float(values['3. low']),
                'volume': int(values['5. volume']),
            }
            try:
                producer.send(TOPIC, record)
                print(f"Sent data for {symbol} on {date}")
            except Exception as e:
                print(f"Error sending data for {symbol} on {date}: {e}")
            time.sleep(0.5)  # To respect API rate limits
if __name__ == "__main__":
    load_request_count()
    symbols = ['MSFT','TSLA','NVDA']  # Example stock symbols

    # Add a loop to go through each symbol
    for symbol in symbols:
        if not can_make_request():
            print(f"Max API requests reached. Stopping producer.")
            break
        produce_stock_data(symbol)
    # Only send end-of-data after ALL symbols are produced
    time.sleep(15)
    producer.send(TOPIC, {'end_of_data': True})
    print("Sent final end-of-data signal.")
    producer.flush()
    producer.close()

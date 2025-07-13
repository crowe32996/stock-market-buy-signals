import os
import time
import json
import requests
from kafka import KafkaProducer
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import time
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
KAFKA_BROKER = 'kafka:9092' 
TOPIC = 'stock_data'  
MAX_REQUESTS_PER_DAY = 25  # Alpha Vantage free limit

for attempt in range(10):  
    try:
        producer = KafkaProducer(bootstrap_servers=[KAFKA_BROKER], value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        print("Connected to Kafka!")
        break
    except NoBrokersAvailable:
        print(f"Kafka not available yet (attempt {attempt+1}), retrying...")
        time.sleep(3)
else:
    raise Exception("Failed to connect to Kafka after 10 attempts")

# Track number of API requests
request_count = 0

def load_request_count():
    global request_count
    try:
        with open("request_count.txt", "r") as f:
            request_count = int(f.read())
        print(f"Loaded request count: {request_count}")  # Debugging output
    except FileNotFoundError:
        request_count = 0
        print(f"Request count file not found. Starting from 0.")  # Debugging output

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
    time.sleep(12) # max of 5 API calls per minute

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
            time.sleep(0.05)  # To manage API rate limits
if __name__ == "__main__":
    load_request_count()
    symbols = ['MSFT','TSLA','NVDA','META','GOOGL','AMZN','AMD','UBER','PLTR','SHOP']  # Example stock symbols
    #symbols = ['MSFT','TSLA','NVDA']
    #symbols = ['MSFT','TSLA','NVDA','META','GOOGL']

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

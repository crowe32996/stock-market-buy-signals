import time
import json
from kafka import KafkaProducer
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import time
from dotenv import load_dotenv
import yfinance as yf

load_dotenv()
KAFKA_BROKER = 'kafka:9092' 
TOPIC = 'stock_data'  
MAX_REQUESTS_PER_DAY = 25  # Alpha Vantage free limit
LOCAL_STORAGE_FILE = "produced_messages.jsonl"  # jsonl = one JSON object per line

def save_message_locally(record):
    with open(LOCAL_STORAGE_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")  # append as JSON line

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

def fetch_stock_data_yf(symbol, start_date="2023-01-01", end_date=None):
    """
    Fetch daily historical data for a stock using yfinance.
    Returns a dictionary in the same format as Alpha Vantage TIME_SERIES_DAILY.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Get last 100 trading days; you can adjust period as needed
        df = ticker.history(start=start_date, end=end_date, interval="1d")  # fetch from start_date to today if end_date is None
        if df.empty:
            print(f"No data for {symbol}")
            return None
        
        # Convert to dict in same structure as Alpha Vantage
        stock_data = {}
        for date, row in df.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            stock_data[date_str] = {
                '1. open': str(row['Open']),
                '2. high': str(row['High']),
                '3. low': str(row['Low']),
                '4. close': str(row['Close']),
                '5. volume': str(int(row['Volume']))
            }
        return stock_data
    except Exception as e:
        print(f"Error fetching {symbol} from yfinance: {e}")
        return None


def produce_stock_data_yf(symbol):
    stock_data = fetch_stock_data_yf(symbol)
    if stock_data:
        for date, values in stock_data.items():
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
                # Send asynchronously (non-blocking)
                producer.send(TOPIC, record)

                # Save locally
                save_message_locally(record)

                # Print status once
                print(f"Queued data for {symbol} on {date}")

            except Exception as e:
                print(f"Failed sending data for {symbol} on {date}: {e}", file=sys.stderr)
                raise
            time.sleep(0.001)  # Small pause to avoid overwhelming Kafka

if __name__ == "__main__":
    symbols = [
        # Large-cap tech / growth
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'ADBE', 'CRM', 'NFLX', 'PYPL', 'SHOP', 'SQ',
        # Broad ETFs / indexes
        'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO',
        # Consumer staples
        'KO', 'PEP', 'PG', 'CL', 'MDLZ', 'COST', 'WMT', 'MCD', 'SBUX', 'YUM',
        # Financials
        'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'SCHW', 'AXP', 'V', 'MA',
        # Healthcare & biotech
        'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'BMY', 'GILD', 'AMGN', 'REGN', 'VRTX', 'CVS', 'UNH',
        # Industrials & transportation
        'GE', 'BA', 'CAT', 'DE', 'UPS', 'FDX', 'LMT', 'NOC', 'HON', 'RTX', 'TM', 'GM', 'F',
        # Energy & materials
        'XOM', 'CVX', 'COP', 'SLB', 'HAL', 'PSX', 'MPC', 'EOG', 'PXD', 'VLO', 'BHP', 'RIO',
        # Communication / Media
        'T', 'VZ', 'CMCSA', 'DIS', 'PINS', 'SNAP', 'TWTR', 'BABA', 'JD', 'TCEHY', 'SONY',
        # REITs / real estate
        'AMT', 'PLD', 'SPG', 'O', 'VNQ', 'DLR', 'EQIX',
        # Emerging / midcaps / underperformers
        'UBER', 'LYFT', 'ABNB', 'COIN', 'RBLX', 'ZM', 'NET', 'DOCU', 'ROKU', 'PTON', 'FSLY'
    ]
    
    for symbol in symbols:
        produce_stock_data_yf(symbol)
    
    # Send end-of-data after all symbols are produced
    #time.sleep(2)  # shorter pause is fine
    producer.send(TOPIC, {'end_of_data': True})
    print("Sent final end-of-data signal.")
    producer.flush()
    producer.close()

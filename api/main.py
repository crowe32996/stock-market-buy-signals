from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import os
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

app = FastAPI(title="Stock Market ML Signals API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ["https://crowedata.com"] to restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection function
def get_db_connection():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    return conn

# Pydantic model for stock data response
class StockData(BaseModel):
    symbol: str
    date: str
    open_price: Optional[float]
    close_price: Optional[float]
    high_price: Optional[float]
    low_price: Optional[float]
    volume: Optional[int]
    SMA_10: Optional[float]
    SMA_50: Optional[float]
    RSI: Optional[float]
    MACD: Optional[float]
    Signal: Optional[float]
    buy_signal: Optional[bool]

# Root endpoint
@app.get("/")
def home():
    return {"message": "FastAPI is up and running"}

# Test database connection
@app.get("/test_db")
def test_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT NOW();")
        result = cur.fetchone()
        cur.close()
        conn.close()
        return {"db_time": result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get all stocks (limit optional)
@app.get("/stocks", response_model=List[StockData])
def get_all_stocks(limit: int = 100):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT symbol, date, open_price, close_price, high_price, low_price, volume, 
               SMA_10, SMA_50, RSI, MACD, Signal, buy_signal
        FROM stock_data
        ORDER BY date DESC
        LIMIT {limit}
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    stocks = [StockData(
        symbol=row[0],
        date=row[1].isoformat(),
        open_price=row[2],
        close_price=row[3],
        high_price=row[4],
        low_price=row[5],
        volume=row[6],
        SMA_10=row[7],
        SMA_50=row[8],
        RSI=row[9],
        MACD=row[10],
        Signal=row[11],
        buy_signal=row[12]
    ) for row in rows]
    
    return stocks

# Get stock by symbol
@app.get("/stocks/{symbol}", response_model=List[StockData])
def get_stock_by_symbol(symbol: str, limit: int = 100):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT symbol, date, open_price, close_price, high_price, low_price, volume, 
               SMA_10, SMA_50, RSI, MACD, Signal, buy_signal
        FROM stock_data
        WHERE symbol = %s
        ORDER BY date DESC
        LIMIT %s
    """, (symbol.upper(), limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    if not rows:
        raise HTTPException(status_code=404, detail="Stock not found")
    
    stocks = [StockData(
        symbol=row[0],
        date=row[1].isoformat(),
        open_price=row[2],
        close_price=row[3],
        high_price=row[4],
        low_price=row[5],
        volume=row[6],
        SMA_10=row[7],
        SMA_50=row[8],
        RSI=row[9],
        MACD=row[10],
        Signal=row[11],
        buy_signal=row[12]
    ) for row in rows]
    
    return stocks

# Get only buy signals
@app.get("/buy_signals", response_model=List[StockData])
def get_buy_signals(limit: int = 100):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT symbol, date, open_price, close_price, high_price, low_price, volume, 
               SMA_10, SMA_50, RSI, MACD, Signal, buy_signal
        FROM stock_data
        WHERE buy_signal = TRUE
        ORDER BY date DESC
        LIMIT {limit}
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    stocks = [StockData(
        symbol=row[0],
        date=row[1].isoformat(),
        open_price=row[2],
        close_price=row[3],
        high_price=row[4],
        low_price=row[5],
        volume=row[6],
        SMA_10=row[7],
        SMA_50=row[8],
        RSI=row[9],
        MACD=row[10],
        Signal=row[11],
        buy_signal=row[12]
    ) for row in rows]
    
    return stocks

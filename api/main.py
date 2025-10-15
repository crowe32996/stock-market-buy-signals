from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import psycopg2
import os
from dotenv import load_dotenv
from typing import List, Optional
from datetime import datetime

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

# Minimal response model
class SignalResponse(BaseModel):
    date: str
    symbol: str
    horizon: str
    buy_prob: Optional[float]
    buy_signal: Optional[bool]

@app.get("/signals", response_model=List[SignalResponse])
def get_signals(
    date: str,
    horizon: str = Query(..., description="short, medium, or long"),
    symbol: Optional[str] = None
):
    conn = get_db_connection()
    cur = conn.cursor()

    # Map horizon to column
    horizon_map = {
        "short": "buy_prob_ml_short",
        "medium": "buy_prob_ml_medium",
        "long": "buy_prob_ml_long"
    }
    if horizon not in horizon_map:
        raise HTTPException(status_code=400, detail="Invalid horizon")

    prob_col = horizon_map[horizon]

    # Build query
    query = f"""
        SELECT date, symbol, {prob_col}, 
               CASE WHEN {prob_col} >= 0.5 THEN TRUE ELSE FALSE END AS buy_signal
        FROM stock_data
        WHERE date = %s
            AND {prob_col} IS NOT NULL

    """
    params = [date]

    if symbol:
        query += " AND symbol = %s"
        params.append(symbol.upper())

    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        SignalResponse(
            date=row[0].isoformat() if isinstance(row[0], (datetime,)) else row[0],
            symbol=row[1],
            horizon=horizon,
            buy_prob=row[2],
            buy_signal=row[3]
        ) for row in rows
    ]


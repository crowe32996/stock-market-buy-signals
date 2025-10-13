from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path


# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

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

# FastAPI app
app = FastAPI(title="Stock Signal API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Stock Signal API is running!"}

@app.get("/signals")
def get_latest_signal(symbol: str):
    query = """
        SELECT * FROM stock_data
        WHERE symbol = %s
        ORDER BY date DESC
        LIMIT 1;
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    return df.to_dict(orient="records")

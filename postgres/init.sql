CREATE TABLE IF NOT EXISTS stock_data (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    open_price FLOAT,
    close_price FLOAT,
    high_price FLOAT,
    low_price FLOAT,
    volume BIGINT,
    date TIMESTAMP NOT NULL,
    SMA_10 FLOAT,
    SMA_50 FLOAT,
    RSI FLOAT,
    MACD FLOAT,
    Signal FLOAT,
    buy_signal BOOLEAN
);

"""Forex data loading module. Supports yfinance download and CSV import."""

import pandas as pd
import yfinance as yf


# Major forex pairs available via yfinance (quoted as USD crosses)
FOREX_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "USDCAD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
}


def fetch_pair(pair_name: str, period: str = "6mo", interval: str = "1h") -> pd.DataFrame:
    """Download forex data from yfinance.

    Args:
        pair_name: Human-readable pair like 'EUR/USD'
        period: yfinance period string (1mo, 3mo, 6mo, 1y, 2y, 5y)
        interval: candle interval (1m, 5m, 15m, 1h, 1d)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    ticker = FOREX_PAIRS.get(pair_name, pair_name)
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {pair_name} ({ticker})")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only OHLCV
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df.index.name = "Datetime"
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load forex data from a CSV file.

    Expects columns: Datetime (or Date), Open, High, Low, Close.
    """
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index.name = "Datetime"
    return df

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
    _validate_ohlc(df)
    return df


def load_csv(path_or_buffer) -> pd.DataFrame:
    """Load forex data from a CSV file or file-like object.

    Expects columns: Datetime (or Date), Open, High, Low, Close.
    """
    df = pd.read_csv(path_or_buffer, parse_dates=True, index_col=0)
    df.index.name = "Datetime"
    _validate_ohlc(df)
    return df


def _validate_ohlc(df: pd.DataFrame) -> None:
    """Validate that the DataFrame has required OHLC columns and sane data."""
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}. "
                         f"Found: {', '.join(df.columns.tolist())}")
    if len(df) < 30:
        raise ValueError(f"Insufficient data: got {len(df)} rows, need at least 30")
    if df[list(required)].isna().all(axis=None):
        raise ValueError("All OHLC values are NaN — check your data source")

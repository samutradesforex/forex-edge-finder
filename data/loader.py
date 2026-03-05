"""Forex data loading module. Supports yfinance download, CSV import, and local caching."""

import hashlib
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

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

CACHE_DIR = Path("data/.cache")

# How long cached data is considered fresh (seconds)
CACHE_TTL = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 3600,
    "1d": 14400,  # 4 hours
}


def _cache_key(pair_name: str, period: str, interval: str) -> Path:
    """Generate a cache file path for a given query."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_pair = pair_name.replace("/", "_")
    return CACHE_DIR / f"{safe_pair}_{period}_{interval}.parquet"


def _is_cache_fresh(path: Path, interval: str) -> bool:
    """Check if a cached file is still fresh based on its interval."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    ttl = CACHE_TTL.get(interval, 3600)
    return age < ttl


def fetch_pair(pair_name: str, period: str = "6mo", interval: str = "1h",
               use_cache: bool = True) -> pd.DataFrame:
    """Download forex data from yfinance with local disk caching.

    Args:
        pair_name: Human-readable pair like 'EUR/USD'
        period: yfinance period string (1mo, 3mo, 6mo, 1y, 2y, 5y)
        interval: candle interval (1m, 5m, 15m, 1h, 1d)
        use_cache: Whether to use local cache (default True)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    cache_path = _cache_key(pair_name, period, interval)

    # Return cached data if fresh
    if use_cache and _is_cache_fresh(cache_path, interval):
        try:
            df = pd.read_parquet(cache_path)
            df.index.name = "Datetime"
            logger.debug("Cache hit for %s %s %s", pair_name, period, interval)
            return df
        except Exception:
            pass  # Fall through to download

    ticker = FOREX_PAIRS.get(pair_name, pair_name)
    logger.info("Downloading %s (%s) period=%s interval=%s", pair_name, ticker, period, interval)
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

    # Save to cache
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            logger.debug("Cached %s %s %s (%d rows)", pair_name, period, interval, len(df))
        except Exception as e:
            logger.warning("Failed to cache data: %s", e)

    return df


def clear_cache():
    """Remove all cached data files."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
        logger.info("Cache cleared")


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

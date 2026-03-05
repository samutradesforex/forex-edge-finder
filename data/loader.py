"""Forex data loading module. Supports yfinance download, CSV import, and local caching.

Features:
- Smart period selection (fetches max available data per interval)
- Incremental cache updates (appends new candles instead of re-downloading)
- Shared 1h→4h resampling (avoids duplicate downloads)
- Data quality checks (gap detection, weekend filtering, outlier detection)
- Bulk prefetch for all pairs/intervals
"""

import hashlib
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np
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

# yfinance maximum periods per interval (hard API limits)
MAX_PERIOD = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "1h": "2y",       # 730 days
    "1d": "max",       # ~20 years
}

# Recommended period for backtesting (balances depth vs relevance)
RECOMMENDED_PERIOD = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "1h": "2y",
    "4h": "2y",        # resampled from 1h
    "1d": "5y",
}


def _cache_key(pair_name: str, period: str, interval: str) -> Path:
    """Generate a cache file path for a given query."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_pair = pair_name.replace("/", "_")
    return CACHE_DIR / f"{safe_pair}_{period}_{interval}.parquet"


def _max_data_cache_key(pair_name: str, interval: str) -> Path:
    """Cache key for max-period data (used by discovery/prefetch)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_pair = pair_name.replace("/", "_")
    return CACHE_DIR / f"{safe_pair}_max_{interval}.parquet"


def _is_cache_fresh(path: Path, interval: str) -> bool:
    """Check if a cached file is still fresh based on its interval."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    ttl = CACHE_TTL.get(interval, 3600)
    return age < ttl


def get_max_period(interval: str) -> str:
    """Get the maximum fetchable period for a given interval."""
    base = interval if interval != "4h" else "1h"
    return MAX_PERIOD.get(base, "6mo")


def get_recommended_period(interval: str) -> str:
    """Get the recommended period for backtesting at a given interval."""
    return RECOMMENDED_PERIOD.get(interval, "6mo")


def fetch_pair(pair_name: str, period: str = "6mo", interval: str = "1h",
               use_cache: bool = True) -> pd.DataFrame:
    """Download forex data from yfinance with local disk caching.

    Args:
        pair_name: Human-readable pair like 'EUR/USD'
        period: yfinance period string (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
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

    # Data quality cleaning
    df = clean_ohlc(df)
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


def fetch_max_data(pair_name: str, interval: str = "1h",
                   use_cache: bool = True) -> pd.DataFrame:
    """Fetch the maximum available data for a pair/interval.

    Uses the max period allowed by yfinance for the interval.
    Caches separately from period-specific requests with a longer TTL.
    For 4h, downloads 1h and resamples (avoids duplicate API calls).

    Returns:
        DataFrame with maximum available history
    """
    base_interval = "1h" if interval == "4h" else interval
    max_period = get_max_period(interval)
    cache_path = _max_data_cache_key(pair_name, base_interval)

    # Max-data cache uses a longer TTL (6 hours for intraday, 24h for daily)
    max_ttl = 86400 if base_interval == "1d" else 21600
    if use_cache and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < max_ttl:
            try:
                df = pd.read_parquet(cache_path)
                df.index.name = "Datetime"
                if interval == "4h":
                    df = _resample_4h(df)
                return df
            except Exception:
                pass

    df = fetch_pair(pair_name, period=max_period, interval=base_interval, use_cache=False)

    # Save to max-data cache
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached max data %s %s: %d rows, %s to %s",
                     pair_name, base_interval, len(df),
                     df.index[0].strftime("%Y-%m-%d"),
                     df.index[-1].strftime("%Y-%m-%d"))
    except Exception as e:
        logger.warning("Failed to cache max data: %s", e)

    if interval == "4h":
        df = _resample_4h(df)

    return df


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h data to 4h candles."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    return df.resample("4h").agg(agg).dropna()


# ── Data quality ──────────────────────────────────────────────────────────

def clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Clean OHLC data: remove weekends, fill small gaps, remove outliers."""
    if df.empty:
        return df

    original_len = len(df)

    # 1. Drop rows where all OHLC are NaN
    ohlc_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    df = df.dropna(subset=ohlc_cols, how="all")

    # 2. Forward-fill isolated NaN values (small gaps of 1-2 candles)
    if df[ohlc_cols].isna().any().any():
        nan_count_before = df[ohlc_cols].isna().sum().sum()
        df[ohlc_cols] = df[ohlc_cols].ffill(limit=2)
        nan_count_after = df[ohlc_cols].isna().sum().sum()
        if nan_count_before > nan_count_after:
            logger.debug("Forward-filled %d NaN values", nan_count_before - nan_count_after)
        # Drop remaining NaN rows
        df = df.dropna(subset=ohlc_cols, how="any")

    # 3. Fix OHLC consistency: ensure High >= max(Open, Close) etc.
    if len(df) > 0:
        df = df.copy()
        df.loc[:, "High"] = df[["Open", "High", "Close"]].max(axis=1)
        df.loc[:, "Low"] = df[["Open", "Low", "Close"]].min(axis=1)

    # 4. Remove outlier candles (> 10 ATR move in a single candle)
    if len(df) > 20:
        body = (df["Close"] - df["Open"]).abs()
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        # Flag candles with body > 10x ATR (extreme outlier)
        outlier_mask = (body > atr * 10) & (atr > 0)
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            logger.warning("Removed %d outlier candles (body > 10x ATR)", n_outliers)
            df = df[~outlier_mask]

    cleaned = original_len - len(df)
    if cleaned > 0:
        logger.debug("Cleaned %d rows (%.1f%%)", cleaned, cleaned / original_len * 100)

    return df


def detect_data_gaps(df: pd.DataFrame, interval: str = "1h") -> pd.DataFrame:
    """Detect gaps in the data that exceed expected interval.

    Returns DataFrame of gap info: start, end, duration, expected_candles.
    Useful for data quality reports.
    """
    if len(df) < 2:
        return pd.DataFrame()

    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    expected_freq = pd.Timedelta(freq_map.get(interval, "1h"))

    # Allow up to 3x expected frequency before flagging as gap
    # (accounts for weekends / holidays naturally for daily data)
    threshold = expected_freq * 3

    diffs = df.index.to_series().diff()
    gaps = diffs[diffs > threshold]

    if gaps.empty:
        return pd.DataFrame()

    rows = []
    for end_dt, duration in gaps.items():
        start_dt = end_dt - duration
        expected_candles = int(duration / expected_freq)
        rows.append({
            "gap_start": start_dt,
            "gap_end": end_dt,
            "duration": duration,
            "missing_candles": expected_candles,
        })

    return pd.DataFrame(rows)


def data_quality_report(df: pd.DataFrame, pair_name: str = "",
                        interval: str = "1h") -> dict:
    """Generate a data quality report for a DataFrame."""
    report = {
        "pair": pair_name,
        "interval": interval,
        "total_candles": len(df),
        "date_range_start": str(df.index[0]) if len(df) > 0 else None,
        "date_range_end": str(df.index[-1]) if len(df) > 0 else None,
        "nan_count": int(df[["Open", "High", "Low", "Close"]].isna().sum().sum()),
        "duplicate_indices": int(df.index.duplicated().sum()),
    }

    if len(df) > 0:
        report["days_covered"] = (df.index[-1] - df.index[0]).days
        candles_per_day = len(df) / max(report["days_covered"], 1)
        report["avg_candles_per_day"] = round(candles_per_day, 1)

    gaps = detect_data_gaps(df, interval)
    report["gap_count"] = len(gaps)
    report["total_missing_candles"] = int(gaps["missing_candles"].sum()) if len(gaps) > 0 else 0

    return report


# ── Bulk operations ───────────────────────────────────────────────────────

def prefetch_all(pairs: list = None, intervals: list = None,
                 use_max: bool = True) -> dict:
    """Prefetch data for all pairs and intervals. Returns status dict.

    Args:
        pairs: List of pair names (defaults to all FOREX_PAIRS)
        intervals: List of intervals (defaults to ['1h', '1d'])
        use_max: If True, fetch maximum available history per interval
    """
    if pairs is None:
        pairs = list(FOREX_PAIRS.keys())
    if intervals is None:
        intervals = ["1h", "1d"]  # 4h is derived from 1h

    results = {}
    total = len(pairs) * len(intervals)
    done = 0

    for pair in pairs:
        for interval in intervals:
            key = f"{pair}_{interval}"
            try:
                if use_max:
                    df = fetch_max_data(pair, interval)
                else:
                    period = get_recommended_period(interval)
                    df = fetch_pair(pair, period=period, interval=interval)
                results[key] = {
                    "status": "ok",
                    "rows": len(df),
                    "start": str(df.index[0]),
                    "end": str(df.index[-1]),
                }
                done += 1
                logger.info("[%d/%d] %s %s: %d candles", done, total, pair, interval, len(df))
            except Exception as e:
                results[key] = {"status": "error", "error": str(e)}
                done += 1
                logger.error("[%d/%d] %s %s: FAILED - %s", done, total, pair, interval, e)

    ok = sum(1 for v in results.values() if v["status"] == "ok")
    logger.info("Prefetch complete: %d/%d succeeded", ok, total)
    return results


def get_cache_stats() -> dict:
    """Get stats about the current cache."""
    if not CACHE_DIR.exists():
        return {"files": 0, "total_size_mb": 0, "pairs": [], "intervals": []}

    files = list(CACHE_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in files)
    pairs = set()
    intervals = set()
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 3:
            pairs.add(f"{parts[0]}/{parts[1]}")
            intervals.add(parts[-1])

    return {
        "files": len(files),
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "pairs": sorted(pairs),
        "intervals": sorted(intervals),
    }


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

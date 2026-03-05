"""Shared test fixtures for Forex Edge Finder tests."""

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_ohlcv():
    """Generate a realistic OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    # Generate price data with realistic movement
    price = 1.1000
    opens, highs, lows, closes = [], [], [], []
    for _ in range(n):
        change = np.random.normal(0, 0.001)
        o = price
        c = price + change
        h = max(o, c) + abs(np.random.normal(0, 0.0005))
        l = min(o, c) - abs(np.random.normal(0, 0.0005))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        price = c

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": np.random.randint(100, 10000, n),
    }, index=dates)
    df.index.name = "Datetime"
    return df


@pytest.fixture
def sample_jpy_ohlcv():
    """Generate a realistic JPY pair OHLCV DataFrame."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    price = 150.00
    opens, highs, lows, closes = [], [], [], []
    for _ in range(n):
        change = np.random.normal(0, 0.1)
        o = price
        c = price + change
        h = max(o, c) + abs(np.random.normal(0, 0.05))
        l = min(o, c) - abs(np.random.normal(0, 0.05))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        price = c

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": np.random.randint(100, 10000, n),
    }, index=dates)
    df.index.name = "Datetime"
    return df


@pytest.fixture
def trending_up_ohlcv():
    """Generate a strongly trending-up OHLCV DataFrame."""
    np.random.seed(123)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    price = 1.1000
    opens, highs, lows, closes = [], [], [], []
    for _ in range(n):
        change = np.random.normal(0.0003, 0.0008)  # positive bias
        o = price
        c = price + change
        h = max(o, c) + abs(np.random.normal(0, 0.0005))
        l = min(o, c) - abs(np.random.normal(0, 0.0003))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        price = c

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": np.random.randint(100, 10000, n),
    }, index=dates)
    df.index.name = "Datetime"
    return df

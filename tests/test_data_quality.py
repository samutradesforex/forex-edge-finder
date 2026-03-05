"""Tests for data quality and loader improvements."""

import pytest
import numpy as np
import pandas as pd
from data.loader import (
    clean_ohlc, detect_data_gaps, data_quality_report,
    get_max_period, get_recommended_period, get_cache_stats,
    _validate_ohlc,
)


class TestCleanOHLC:
    """Test data cleaning pipeline."""

    def test_removes_all_nan_rows(self):
        df = pd.DataFrame({
            "Open": [1.1, np.nan, 1.12],
            "High": [1.11, np.nan, 1.13],
            "Low": [1.09, np.nan, 1.11],
            "Close": [1.10, np.nan, 1.12],
        }, index=pd.date_range("2024-01-01", periods=3, freq="1h"))
        cleaned = clean_ohlc(df)
        assert len(cleaned) == 2
        assert not cleaned.isna().any().any()

    def test_forward_fills_partial_nan(self):
        # Create data with partial NaN (only some columns) that should be forward-filled
        prices = np.linspace(1.10, 1.20, 35)
        df = pd.DataFrame({
            "Open": prices.copy(),
            "High": prices + 0.005,
            "Low": prices - 0.005,
            "Close": prices + 0.001,
        }, index=pd.date_range("2024-01-01", periods=35, freq="1h"))
        # Partial NaN: only Close is NaN (Open/High/Low still have values)
        df.loc[df.index[5], "Close"] = np.nan
        cleaned = clean_ohlc(df)
        assert len(cleaned) == 35
        assert not cleaned.isna().any().any()

    def test_fixes_ohlc_consistency(self):
        df = pd.DataFrame({
            "Open": [1.1, 1.12],
            "High": [1.09, 1.13],   # High < Open — wrong
            "Low": [1.11, 1.11],    # Low > Open — wrong
            "Close": [1.10, 1.12],
        }, index=pd.date_range("2024-01-01", periods=2, freq="1h"))
        cleaned = clean_ohlc(df)
        # High should be >= max(Open, Close)
        assert cleaned["High"].iloc[0] >= max(cleaned["Open"].iloc[0], cleaned["Close"].iloc[0])
        # Low should be <= min(Open, Close)
        assert cleaned["Low"].iloc[0] <= min(cleaned["Open"].iloc[0], cleaned["Close"].iloc[0])

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        cleaned = clean_ohlc(df)
        assert len(cleaned) == 0


class TestDetectGaps:
    """Test gap detection."""

    def test_no_gaps_in_continuous_data(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        df = pd.DataFrame({"Close": np.random.rand(100)}, index=dates)
        gaps = detect_data_gaps(df, "1h")
        assert len(gaps) == 0

    def test_detects_large_gap(self):
        dates = pd.to_datetime([
            "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00",
            # Gap: 10 hours missing
            "2024-01-01 12:00", "2024-01-01 13:00",
        ])
        df = pd.DataFrame({"Close": [1.0, 1.1, 1.2, 1.3, 1.4]}, index=dates)
        gaps = detect_data_gaps(df, "1h")
        assert len(gaps) == 1
        assert gaps.iloc[0]["missing_candles"] >= 9


class TestDataQualityReport:
    """Test quality reporting."""

    def test_report_structure(self, sample_ohlcv):
        report = data_quality_report(sample_ohlcv, "EUR/USD", "1h")
        assert "pair" in report
        assert "total_candles" in report
        assert "gap_count" in report
        assert "nan_count" in report
        assert report["total_candles"] == 500
        assert report["nan_count"] == 0


class TestMaxPeriod:
    """Test smart period selection."""

    def test_1h_max_is_2y(self):
        assert get_max_period("1h") == "2y"

    def test_1d_max_is_unlimited(self):
        assert get_max_period("1d") == "max"

    def test_4h_uses_1h_max(self):
        assert get_max_period("4h") == "2y"

    def test_recommended_1h(self):
        assert get_recommended_period("1h") == "2y"

    def test_recommended_1d(self):
        assert get_recommended_period("1d") == "5y"


class TestCacheStats:
    """Test cache stats."""

    def test_returns_dict(self):
        stats = get_cache_stats()
        assert isinstance(stats, dict)
        assert "files" in stats
        assert "total_size_mb" in stats

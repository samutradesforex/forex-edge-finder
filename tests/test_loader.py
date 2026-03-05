"""Tests for the data loader module."""

import pytest
import pandas as pd
from pathlib import Path
from data.loader import _validate_ohlc, _cache_key, _is_cache_fresh, CACHE_DIR, load_csv
import io


class TestValidation:
    def test_valid_ohlc(self, sample_ohlcv):
        _validate_ohlc(sample_ohlcv)  # should not raise

    def test_missing_column(self, sample_ohlcv):
        df = sample_ohlcv.drop(columns=["High"])
        with pytest.raises(ValueError, match="Missing required columns"):
            _validate_ohlc(df)

    def test_insufficient_rows(self):
        df = pd.DataFrame({
            "Open": [1.1] * 10,
            "High": [1.2] * 10,
            "Low": [1.0] * 10,
            "Close": [1.1] * 10,
        })
        with pytest.raises(ValueError, match="Insufficient data"):
            _validate_ohlc(df)

    def test_all_nan(self):
        import numpy as np
        df = pd.DataFrame({
            "Open": [np.nan] * 50,
            "High": [np.nan] * 50,
            "Low": [np.nan] * 50,
            "Close": [np.nan] * 50,
        })
        with pytest.raises(ValueError, match="NaN"):
            _validate_ohlc(df)


class TestCaching:
    def test_cache_key_format(self):
        path = _cache_key("EUR/USD", "6mo", "1h")
        assert "EUR_USD" in str(path)
        assert path.suffix == ".parquet"

    def test_fresh_cache(self, tmp_path, monkeypatch):
        import data.loader as loader
        monkeypatch.setattr(loader, "CACHE_DIR", tmp_path)

        path = tmp_path / "test.parquet"
        path.write_text("dummy")
        assert _is_cache_fresh(path, "1h")

    def test_stale_cache(self, tmp_path, monkeypatch):
        import data.loader as loader
        import os
        monkeypatch.setattr(loader, "CACHE_DIR", tmp_path)

        path = tmp_path / "test.parquet"
        path.write_text("dummy")
        # Make it old
        old_time = path.stat().st_mtime - 7200
        os.utime(path, (old_time, old_time))
        assert not _is_cache_fresh(path, "1h")

    def test_missing_cache(self, tmp_path):
        path = tmp_path / "nonexistent.parquet"
        assert not _is_cache_fresh(path, "1h")


class TestCSVLoader:
    def test_load_valid_csv(self):
        csv_data = """Datetime,Open,High,Low,Close,Volume
2024-01-01 00:00:00,1.1000,1.1050,1.0950,1.1020,1000
""" + "\n".join(
            f"2024-01-01 {i:02d}:00:00,{1.1+i*0.001:.4f},{1.105+i*0.001:.4f},"
            f"{1.095+i*0.001:.4f},{1.102+i*0.001:.4f},{1000+i}"
            for i in range(1, 50)
        )
        df = load_csv(io.StringIO(csv_data))
        assert len(df) == 50
        assert "Open" in df.columns

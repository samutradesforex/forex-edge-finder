"""Tests for market structure analysis."""

import pytest
from engine.structure import compute_structure, detect_structure_breaks, get_bias_at
from engine.liquidity import find_swing_points


class TestStructureBreaks:
    def test_compute_structure(self, sample_ohlcv):
        swings, breaks, bias_series = compute_structure(sample_ohlcv)
        assert len(swings) > 0
        assert len(bias_series) == len(sample_ohlcv)

    def test_bias_values(self, sample_ohlcv):
        _, _, bias_series = compute_structure(sample_ohlcv)
        valid_biases = {"bullish", "bearish", "neutral"}
        for val in bias_series.unique():
            assert val in valid_biases

    def test_breaks_have_valid_types(self, sample_ohlcv):
        swings = find_swing_points(sample_ohlcv, lookback=5)
        breaks = detect_structure_breaks(sample_ohlcv, swings)
        for b in breaks:
            assert b.kind in ("bos", "choch")
            assert b.direction in ("bullish", "bearish")

    def test_get_bias_at(self, sample_ohlcv):
        swings = find_swing_points(sample_ohlcv, lookback=5)
        breaks = detect_structure_breaks(sample_ohlcv, swings)
        bias = get_bias_at(breaks, 0)
        assert bias in ("bullish", "bearish", "neutral")

    def test_trending_data_has_breaks(self, trending_up_ohlcv):
        swings, breaks, _ = compute_structure(trending_up_ohlcv)
        # Trending data should produce some structure breaks
        assert len(breaks) > 0

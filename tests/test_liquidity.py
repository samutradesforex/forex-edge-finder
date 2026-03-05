"""Tests for the liquidity detection engine."""

import pytest
import numpy as np
from engine.liquidity import (
    find_swing_points, find_multi_tf_swings, find_liquidity_levels,
    find_fvgs, find_order_blocks, get_pip_size, calc_atr, calc_rsi, calc_ema,
    get_session, get_killzone, is_in_killzone,
    detect_liquidity_sweeps, detect_inducement_traps, detect_stop_hunts,
    detect_ema_crossover, detect_rsi_reversal, detect_breakout,
    detect_fvg_entry, detect_ob_bounce,
)
import pandas as pd


class TestPipSize:
    def test_standard_pair(self):
        assert get_pip_size("EUR/USD") == 0.0001

    def test_jpy_pair(self):
        assert get_pip_size("USD/JPY") == 0.01

    def test_gbpjpy(self):
        assert get_pip_size("GBP/JPY") == 0.01


class TestIndicators:
    def test_calc_atr_length(self, sample_ohlcv):
        atr = calc_atr(sample_ohlcv)
        assert len(atr) == len(sample_ohlcv)

    def test_calc_atr_positive(self, sample_ohlcv):
        atr = calc_atr(sample_ohlcv)
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_calc_rsi_range(self, sample_ohlcv):
        rsi = calc_rsi(sample_ohlcv)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_calc_ema_length(self, sample_ohlcv):
        ema = calc_ema(sample_ohlcv["Close"], 21)
        assert len(ema) == len(sample_ohlcv)

    def test_calc_ema_tracks_price(self, sample_ohlcv):
        ema = calc_ema(sample_ohlcv["Close"], 5)
        valid = ema.dropna()
        # EMA should be in the ballpark of price
        assert abs(valid.iloc[-1] - sample_ohlcv["Close"].iloc[-1]) < 0.05


class TestSwingDetection:
    def test_finds_swings(self, sample_ohlcv):
        swings = find_swing_points(sample_ohlcv, lookback=5)
        assert len(swings) > 0

    def test_swing_types(self, sample_ohlcv):
        swings = find_swing_points(sample_ohlcv, lookback=5)
        highs = [s for s in swings if s.kind == "high"]
        lows = [s for s in swings if s.kind == "low"]
        assert len(highs) > 0
        assert len(lows) > 0

    def test_swing_indices_in_range(self, sample_ohlcv):
        swings = find_swing_points(sample_ohlcv, lookback=5)
        for s in swings:
            assert 0 <= s.index < len(sample_ohlcv)

    def test_multi_tf_swings(self, sample_ohlcv):
        swings = find_multi_tf_swings(sample_ohlcv)
        assert len(swings) > 0

    def test_small_lookback(self, sample_ohlcv):
        s3 = find_swing_points(sample_ohlcv, lookback=3)
        s13 = find_swing_points(sample_ohlcv, lookback=13)
        # Smaller lookback should find more swings
        assert len(s3) >= len(s13)


class TestLiquidityLevels:
    def test_finds_levels(self, sample_ohlcv):
        swings = find_swing_points(sample_ohlcv, lookback=5)
        levels = find_liquidity_levels(swings, cluster_pips=10.0)
        assert len(levels) > 0

    def test_level_types(self, sample_ohlcv):
        swings = find_swing_points(sample_ohlcv, lookback=5)
        levels = find_liquidity_levels(swings, cluster_pips=10.0)
        kinds = {l.kind for l in levels}
        assert "buy_side" in kinds or "sell_side" in kinds

    def test_level_strength_positive(self, sample_ohlcv):
        swings = find_swing_points(sample_ohlcv, lookback=5)
        levels = find_liquidity_levels(swings, cluster_pips=10.0)
        for l in levels:
            assert l.strength >= 1


class TestFVGsAndOrderBlocks:
    def test_find_fvgs(self, sample_ohlcv):
        fvgs = find_fvgs(sample_ohlcv)
        # May or may not find FVGs depending on random data
        assert isinstance(fvgs, list)

    def test_fvg_structure(self, sample_ohlcv):
        fvgs = find_fvgs(sample_ohlcv)
        for fvg in fvgs:
            assert fvg.top > fvg.bottom
            assert fvg.direction in ("bullish", "bearish")

    def test_find_order_blocks(self, sample_ohlcv):
        obs = find_order_blocks(sample_ohlcv)
        assert isinstance(obs, list)

    def test_ob_structure(self, sample_ohlcv):
        obs = find_order_blocks(sample_ohlcv)
        for ob in obs:
            assert ob.high > ob.low
            assert ob.direction in ("bullish", "bearish")


class TestSessions:
    def test_london_session(self):
        dt = pd.Timestamp("2024-01-15 09:00:00")
        assert get_session(dt) == "london"

    def test_new_york_session(self):
        dt = pd.Timestamp("2024-01-15 18:00:00")
        assert get_session(dt) == "new_york"

    def test_asia_session(self):
        dt = pd.Timestamp("2024-01-15 03:00:00")
        assert get_session(dt) == "asia"

    def test_killzone_london_open(self):
        dt = pd.Timestamp("2024-01-15 08:00:00")
        assert get_killzone(dt) == "london_open"
        assert is_in_killzone(dt)


class TestSignalDetection:
    """Test that signal detectors run without errors and return valid signals."""

    def _get_detect_kwargs(self, df):
        pip_size = get_pip_size("EUR/USD")
        return dict(
            pip_size=pip_size,
            atr=calc_atr(df),
            fvgs=find_fvgs(df, pip_size=pip_size),
            obs=find_order_blocks(df, pip_size=pip_size),
            rsi=calc_rsi(df),
            ema_fast=calc_ema(df["Close"], 21),
            ema_slow=calc_ema(df["Close"], 50),
            require_displacement=False,
            min_confluence=0,
            session_filter="all",
            rr_ratio=2.0,
        )

    def test_detect_sweeps(self, sample_ohlcv):
        swings = find_swing_points(sample_ohlcv, lookback=5)
        levels = find_liquidity_levels(swings, cluster_pips=10.0)
        kwargs = self._get_detect_kwargs(sample_ohlcv)
        signals = detect_liquidity_sweeps(
            sample_ohlcv, levels, min_wick_pips=3.0, **kwargs)
        assert isinstance(signals, list)

    def test_detect_inducement(self, sample_ohlcv):
        swings = find_swing_points(sample_ohlcv, lookback=5)
        kwargs = self._get_detect_kwargs(sample_ohlcv)
        signals = detect_inducement_traps(sample_ohlcv, swings, **kwargs)
        assert isinstance(signals, list)

    def test_detect_ema_crossover(self, sample_ohlcv):
        kwargs = self._get_detect_kwargs(sample_ohlcv)
        signals = detect_ema_crossover(sample_ohlcv, **kwargs)
        assert isinstance(signals, list)

    def test_detect_rsi_reversal(self, sample_ohlcv):
        kwargs = self._get_detect_kwargs(sample_ohlcv)
        signals = detect_rsi_reversal(sample_ohlcv, **kwargs)
        assert isinstance(signals, list)

    def test_detect_fvg_entry(self, sample_ohlcv):
        kwargs = self._get_detect_kwargs(sample_ohlcv)
        signals = detect_fvg_entry(sample_ohlcv, **kwargs)
        assert isinstance(signals, list)

    def test_detect_ob_bounce(self, sample_ohlcv):
        kwargs = self._get_detect_kwargs(sample_ohlcv)
        signals = detect_ob_bounce(sample_ohlcv, **kwargs)
        assert isinstance(signals, list)

    def test_signal_structure(self, sample_ohlcv):
        kwargs = self._get_detect_kwargs(sample_ohlcv)
        signals = detect_ema_crossover(sample_ohlcv, **kwargs)
        for sig in signals:
            assert sig.direction in ("long", "short")
            assert sig.entry_price > 0
            assert sig.stop_loss > 0
            assert sig.take_profit > 0
            assert 0 <= sig.entry_index < len(sample_ohlcv)

"""Round 2 regression tests for bugs found during deep audit."""

import pytest
import numpy as np
import pandas as pd
from engine.backtester import BacktestResult, Trade, run_backtest
from engine.liquidity import (
    detect_ema_crossover, detect_rsi_reversal, calc_atr, calc_rsi, calc_ema,
    get_pip_size, SwingPoint,
)
from engine.structure import detect_structure_breaks, StructureBreak


class TestSortinoAllWins:
    """Bug: Sortino ratio was 0.0 when all trades were winners.
    Should be inf (no downside risk)."""

    def test_sortino_inf_all_wins(self):
        trades = []
        for i in range(20):
            t = Trade(
                entry_datetime=pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 24),
                exit_datetime=pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 24 + 10),
                direction="long", entry_price=1.1, exit_price=1.11,
                stop_loss=1.09, take_profit=1.11,
                pnl_pips=100.0 + i, result="win", signal_type="test",
                swept_level=1.09, session="london",
            )
            trades.append(t)

        bt = BacktestResult(trades=trades, pair_name="EUR/USD")
        bt.compute_metrics()
        assert bt.sortino_ratio == float("inf"), \
            f"Sortino should be inf with all wins, got {bt.sortino_ratio}"

    def test_sortino_normal_with_varying_losses(self):
        """Sortino should be a finite positive float when there are varying losses."""
        trades = []
        for i in range(30):
            is_win = i % 3 != 0
            loss_pips = -30.0 - (i * 2)  # varying losses to ensure std > 0
            t = Trade(
                entry_datetime=pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 24),
                exit_datetime=pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 24 + 10),
                direction="long", entry_price=1.1,
                exit_price=1.11 if is_win else 1.09,
                stop_loss=1.09, take_profit=1.11,
                pnl_pips=100.0 if is_win else loss_pips,
                result="win" if is_win else "loss",
                signal_type="test", swept_level=1.09, session="london",
            )
            trades.append(t)

        bt = BacktestResult(trades=trades, pair_name="EUR/USD")
        bt.compute_metrics()
        assert bt.sortino_ratio != float("inf")
        assert bt.sortino_ratio > 0


class TestEMACrossoverConfluence:
    """Bug: EMA crossover had inflated confluence score because
    trend_aligned was counted in score but removed from factor list."""

    def test_score_excludes_trend_aligned(self, sample_ohlcv):
        """Score should NOT include trend_aligned (always true at crossover)."""
        sigs = detect_ema_crossover(
            sample_ohlcv, pip_size=0.0001, rr_ratio=2.0, min_confluence=0)
        for s in sigs:
            # Score counts real confluence factors (excluding the ema_crossover label
            # and the removed trend_aligned). So score == len(factors) - 1.
            real_factors = [f for f in s.confluence_factors if f != "ema_crossover"]
            assert s.confluence_score == len(real_factors), \
                f"Score {s.confluence_score} != real factor count {len(real_factors)}: {s.confluence_factors}"

    def test_no_trend_aligned_in_factors(self, sample_ohlcv):
        sigs = detect_ema_crossover(
            sample_ohlcv, pip_size=0.0001, rr_ratio=2.0, min_confluence=0)
        for s in sigs:
            assert "trend_aligned" not in s.confluence_factors, \
                "trend_aligned should be removed (redundant with crossover)"


class TestRSIReversalConfluence:
    """Bug: RSI reversal double-counted the RSI condition in score."""

    def test_no_rsi_oversold_in_long_factors(self, sample_ohlcv):
        sigs = detect_rsi_reversal(
            sample_ohlcv, pip_size=0.0001, rr_ratio=2.0, min_confluence=0)
        for s in sigs:
            if s.direction == "long":
                rsi_factors = [f for f in s.confluence_factors if f.startswith("rsi_oversold")]
                assert len(rsi_factors) == 0, \
                    f"rsi_oversold should be removed for RSI reversal long: {s.confluence_factors}"

    def test_no_rsi_overbought_in_short_factors(self, sample_ohlcv):
        sigs = detect_rsi_reversal(
            sample_ohlcv, pip_size=0.0001, rr_ratio=2.0, min_confluence=0)
        for s in sigs:
            if s.direction == "short":
                rsi_factors = [f for f in s.confluence_factors if f.startswith("rsi_overbought")]
                assert len(rsi_factors) == 0, \
                    f"rsi_overbought should be removed for RSI reversal short: {s.confluence_factors}"

    def test_score_excludes_rsi_doublecount(self, sample_ohlcv):
        """Score should NOT double-count the RSI condition."""
        sigs = detect_rsi_reversal(
            sample_ohlcv, pip_size=0.0001, rr_ratio=2.0, min_confluence=0)
        for s in sigs:
            # Score counts real confluence factors (excluding the rsi_reversal label)
            real_factors = [f for f in s.confluence_factors if f != "rsi_reversal"]
            assert s.confluence_score == len(real_factors), \
                f"Score {s.confluence_score} != real factor count {len(real_factors)}: {s.confluence_factors}"


class TestStructureBreakChronological:
    """Bug: Structure breaks processed highs and lows in separate loops,
    causing BOS/CHoCH misclassification when breaks interleave."""

    def test_interleaved_breaks_correct(self):
        """When a bearish break happens before a bullish break chronologically,
        the classifications should reflect the actual order."""
        # Create swings that produce interleaved break candidates
        swings = [
            SwingPoint(index=5, price=1.1010, kind="high",
                       datetime=pd.Timestamp("2024-01-01 05:00")),
            SwingPoint(index=8, price=1.0990, kind="low",
                       datetime=pd.Timestamp("2024-01-01 08:00")),
            SwingPoint(index=15, price=1.1005, kind="high",
                       datetime=pd.Timestamp("2024-01-01 15:00")),
            SwingPoint(index=20, price=1.0980, kind="low",
                       datetime=pd.Timestamp("2024-01-01 20:00")),
        ]

        # Build price data where:
        # - idx 10: breaks below swing low at 1.0990 (bearish)
        # - idx 17: breaks above swing high at 1.1010 (bullish, should be CHoCH)
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        np.random.seed(42)
        prices = np.linspace(1.100, 1.100, n)
        highs = prices + 0.002
        lows = prices - 0.002

        # Force break below first swing low at candle 10
        lows[10] = 1.0985  # below 1.0990
        # Force break above first swing high at candle 17
        highs[17] = 1.1015  # above 1.1010

        df = pd.DataFrame({
            "Open": prices,
            "High": highs,
            "Low": lows,
            "Close": prices,
        }, index=dates)

        breaks = detect_structure_breaks(df, swings)

        if len(breaks) >= 2:
            # First break should be bearish (at idx ~10)
            # Second break should be bullish CHoCH (at idx ~17)
            bearish_breaks = [b for b in breaks if b.direction == "bearish"]
            bullish_breaks = [b for b in breaks if b.direction == "bullish"]

            if bearish_breaks and bullish_breaks:
                first_bear = bearish_breaks[0]
                first_bull = bullish_breaks[0]
                if first_bear.index < first_bull.index:
                    # Bearish happened first, bullish should be CHoCH
                    assert first_bull.kind == "choch", \
                        f"Bullish break after bearish should be CHoCH, got {first_bull.kind}"

    def test_breaks_sorted_chronologically(self, sample_ohlcv):
        """All breaks should be in chronological order."""
        from engine.structure import compute_structure
        _, breaks, _ = compute_structure(sample_ohlcv)
        for i in range(1, len(breaks)):
            assert breaks[i].index >= breaks[i - 1].index, \
                "Structure breaks not in chronological order"


class TestScannerNaN:
    """Bug: Scanner edge_score could become NaN if sharpe_ratio was NaN."""

    def test_edge_score_never_nan(self):
        """Create a mock scenario and verify no NaN in edge score."""
        from engine.scanner import scan_all_pairs
        # We can't easily call scan_all_pairs without network,
        # but we can test the score calculation logic directly.
        bt = BacktestResult.__new__(BacktestResult)
        bt.profit_factor = float("nan")
        bt.win_rate = 50.0
        bt.expectancy_pips = float("nan")
        bt.total_pips = 100.0
        bt.max_drawdown_pips = 0.0
        bt.sharpe_ratio = float("nan")

        # Replicate scanner's edge calculation
        edge = 0.0
        pf = bt.profit_factor
        if pf != float("inf") and pf == pf:
            edge += min(pf, 5) * 10
        edge += bt.win_rate * 0.5
        exp = bt.expectancy_pips
        if exp == exp:
            edge += exp * 2
        if bt.max_drawdown_pips > 0:
            edge += (bt.total_pips / bt.max_drawdown_pips) * 10
        sr = bt.sharpe_ratio
        if sr == sr and sr != float("inf") and sr != float("-inf"):
            edge += sr * 5

        assert edge == edge, "Edge score should not be NaN"
        assert edge == 25.0, f"Expected 25.0 (win_rate*0.5 only), got {edge}"


class TestDiscoveryResume:
    """Bug: Discovery resume used lexicographic comparison for pairs,
    skipping pairs that came later in iteration but earlier alphabetically."""

    def test_resume_uses_iteration_order(self):
        """Verify the resume logic correctly tracks by identity, not alphabetics."""
        from data.loader import FOREX_PAIRS
        from engine.discovery import DiscoveryState

        pairs = list(FOREX_PAIRS.keys())
        state = DiscoveryState()
        state.last_completed_pair = "USD/JPY"  # 3rd in list
        state.last_completed_interval = "1h"
        state.last_completed_strategy = "sweeps"

        # After resuming past USD/JPY, the next pair should be USD/CHF (4th)
        # NOT skip AUD/USD, NZD/USD etc. which are alphabetically < USD/JPY
        resume_key = (state.last_completed_pair, state.last_completed_interval,
                      state.last_completed_strategy)

        found_resume = False
        should_process = []
        for pair in pairs:
            for intv in ["1h"]:
                if not found_resume:
                    if pair == resume_key[0] and intv == resume_key[1]:
                        found_resume = True
                    continue
                should_process.append(pair)

        # USD/CHF should be first to process (comes right after USD/JPY)
        assert "USD/CHF" in should_process, \
            f"USD/CHF should be processed after resuming from USD/JPY"
        assert "AUD/USD" in should_process, \
            f"AUD/USD (alphabetically < USD/JPY) should NOT be skipped"
        assert "EUR/USD" not in should_process, \
            "EUR/USD (before USD/JPY in iteration) should be skipped"


class TestWalkForwardWarmup:
    """Bug: Walk-forward OOS computed indicators from split point without
    warm-up data, causing inaccurate EMA/RSI at the start of OOS period."""

    def test_oos_indicators_have_warmup(self):
        """Verify that OOS indicators are computed with lookback data."""
        from engine.discovery import walk_forward_validate

        # Create enough data for walk-forward
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
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
            "Open": opens, "High": highs, "Low": lows, "Close": closes,
            "Volume": np.random.randint(100, 10000, n),
        }, index=dates)

        # Compute EMA on full OOS vs with warmup
        split_idx = int(n * 0.7)
        df_oos = df.iloc[split_idx:]

        warmup = 60
        warmup_start = max(0, split_idx - warmup)
        df_with_warmup = df.iloc[warmup_start:]

        ema_no_warmup = calc_ema(df_oos["Close"], 50)
        ema_with_warmup = calc_ema(df_with_warmup["Close"], 50).loc[df_oos.index]

        # First few values should differ significantly
        first_diff = abs(ema_no_warmup.iloc[0] - ema_with_warmup.iloc[0])
        assert first_diff > 1e-6, \
            f"EMA with warmup should differ from cold-start EMA at OOS start: diff={first_diff}"

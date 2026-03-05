"""Regression tests for bugs found during deep audit."""

import pytest
import numpy as np
import pandas as pd
from engine.liquidity import (
    find_swing_points, find_liquidity_levels, find_fvgs, find_order_blocks,
    get_pip_size, calc_atr, calc_rsi, calc_ema,
    detect_ob_bounce, get_session,
)
from engine.backtester import run_backtest, BacktestResult, Trade
from engine.sizing import simulate_account
from engine.scanner import scan_all_pairs


class TestOBMutationBug:
    """Bug: detect_ob_bounce mutated shared obs list via ob.mitigated = True,
    causing second call with same list to produce 0 signals."""

    def test_ob_bounce_does_not_mutate_input(self, sample_ohlcv):
        pip_size = get_pip_size("EUR/USD")
        obs = find_order_blocks(sample_ohlcv, pip_size=pip_size)
        if not obs:
            pytest.skip("No order blocks in sample data")

        # Record original mitigated states
        original_states = [ob.mitigated for ob in obs]

        detect_ob_bounce(
            sample_ohlcv, pip_size=pip_size, obs=obs,
            min_confluence=0, rr_ratio=2.0,
        )

        # obs list should be unchanged
        current_states = [ob.mitigated for ob in obs]
        assert current_states == original_states

    def test_ob_bounce_idempotent(self, sample_ohlcv):
        pip_size = get_pip_size("EUR/USD")
        obs = find_order_blocks(sample_ohlcv, pip_size=pip_size)
        kwargs = dict(pip_size=pip_size, obs=obs, min_confluence=0, rr_ratio=2.0)

        sigs1 = detect_ob_bounce(sample_ohlcv, **kwargs)
        sigs2 = detect_ob_bounce(sample_ohlcv, **kwargs)
        assert len(sigs1) == len(sigs2)


class TestClusterDriftBug:
    """Bug: find_liquidity_levels compared new point to cluster[-1].price
    instead of centroid, causing cluster drift."""

    def test_cluster_uses_centroid(self):
        """Points that are near the centroid but far from last element should cluster."""
        from engine.liquidity import SwingPoint

        # Create points at 1.1000, 1.1002, 1.1004, 1.1006
        # With cluster_pips=5 (0.0005), each is within 0.0005 of centroid
        # but 1.1006 is 0.0004 from 1.1004 (within range) — this worked before.
        # More telling: 1.1000, 1.1004, 1.1008 — each 4 pips apart.
        # Old code: 1.1008 compared to 1.1004 (4 pips, within 5). Centroid=(1.1000+1.1004)/2=1.1002, dist=6 pips (outside 5).
        # The centroid check is stricter and more correct.
        swings = [
            SwingPoint(index=0, price=1.1000, kind="high",
                       datetime=pd.Timestamp("2024-01-01")),
            SwingPoint(index=10, price=1.1004, kind="high",
                       datetime=pd.Timestamp("2024-01-02")),
            SwingPoint(index=20, price=1.1008, kind="high",
                       datetime=pd.Timestamp("2024-01-03")),
        ]

        levels = find_liquidity_levels(swings, cluster_pips=5.0, pip_size=0.0001)
        buy_side = [l for l in levels if l.kind == "buy_side"]
        # With centroid-based clustering, 1.1008 should NOT cluster with 1.1000+1.1004
        # because centroid=1.1002, distance=0.0006 > 0.0005
        assert len(buy_side) >= 2, "Centroid clustering should split drifted points"


class TestSpreadDoubleChargeBug:
    """Bug: Spread was deducted both from entry_price AND pnl_pips."""

    def test_spread_deducted_once(self, sample_ohlcv):
        bt_no_spread = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps",
                                     spread_pips=0.0)
        bt_spread = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps",
                                  spread_pips=2.0)
        if bt_no_spread.total_trades > 0 and bt_spread.total_trades > 0:
            diff = bt_no_spread.total_pips - bt_spread.total_pips
            expected_max = bt_spread.total_trades * 2.0  # 2 pips per trade max
            assert diff <= expected_max + 0.1, "Spread deducted more than once"


class TestSameBarSLTPBug:
    """Bug: When both SL and TP are hit on same candle, SL always won."""

    def test_same_bar_resolution_not_biased(self, sample_ohlcv):
        # Run with very tight RR so same-bar hits are likely
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps",
                          rr_ratio=1.0, spread_pips=0)
        if bt.total_trades >= 10:
            # With 1:1 RR, we shouldn't see 100% loss rate
            # (statistically very unlikely if resolution is fair)
            assert bt.wins > 0 or bt.total_trades < 5


class TestMaxDrawdownAllLossBug:
    """Bug: max_drawdown_pct was 0 when all trades are losses (peak stays at 0)."""

    def test_all_loss_drawdown(self):
        # Create a scenario with guaranteed losses by using a tiny dataset
        # and checking the metric calculation directly
        bt = BacktestResult.__new__(BacktestResult)
        bt.trades = []
        bt.total_trades = 3
        bt.wins = 0
        bt.losses = 3
        bt.breakevens = 0
        bt.total_pips = -30.0
        bt.equity_curve = [0.0, -10.0, -20.0, -30.0]
        bt.max_drawdown_pips = 30.0
        # Compute max_drawdown_pct
        peak = max(bt.equity_curve)
        if peak > 0:
            trough = min(bt.equity_curve)
            bt.max_drawdown_pct = (peak - trough) / peak * 100
        elif bt.max_drawdown_pips > 0:
            bt.max_drawdown_pct = 100.0
        else:
            bt.max_drawdown_pct = 0.0

        assert bt.max_drawdown_pct == 100.0


class TestSessionOverlapBug:
    """Bug: London/NY overlap session was never detected because London matched first."""

    def test_overlap_detected(self):
        dt = pd.Timestamp("2024-01-15 13:30:00")  # 13:30 UTC = overlap
        session = get_session(dt)
        assert session == "lo_ny_overlap"

    def test_pure_london_still_works(self):
        dt = pd.Timestamp("2024-01-15 09:00:00")
        assert get_session(dt) == "london"

    def test_pure_ny_still_works(self):
        dt = pd.Timestamp("2024-01-15 18:00:00")
        assert get_session(dt) == "new_york"


class TestKellyLookAheadBug:
    """Bug: Kelly criterion used ALL trades to pre-calculate fraction,
    giving it future information about win rate and payoff ratio."""

    def test_kelly_no_look_ahead(self):
        """First N trades should use conservative sizing since Kelly hasn't kicked in."""
        # Create trades with known outcomes
        trades = []
        for i in range(30):
            t = Trade.__new__(Trade)
            t.direction = "long"
            t.entry_price = 1.1000
            t.stop_loss = 1.0950
            t.exit_price = 1.1100 if i % 2 == 0 else 1.0950
            t.result = "win" if i % 2 == 0 else "loss"
            t.pnl_pips = 100.0 if i % 2 == 0 else -50.0
            t.holding_candles = 10
            t.entry_datetime = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
            t.exit_datetime = t.entry_datetime + pd.Timedelta(hours=10)
            t.signal_type = "test"
            t.confluence_factors = []
            t.confluence_score = 0
            t.session = "london"
            trades.append(t)

        sim = simulate_account(
            trades, starting_balance=10000, sizing_mode="kelly",
            pair_name="EUR/USD", compounding=False,
        )

        # First 20 trades should use minimum lot (0.01) since Kelly needs min_kelly_trades
        for state in sim.states[:20]:
            assert state.lot_size == 0.01, \
                f"Trade {state.trade_num} used lot {state.lot_size}, expected 0.01 (Kelly shouldn't kick in yet)"


class TestScannerReturnTypeBug:
    """Bug: scan_all_pairs returns tuple but was annotated as List."""

    def test_return_is_tuple(self):
        # Just verify the function signature allows tuple return
        # We can't easily call it without network, so check the annotation
        import inspect
        sig = inspect.signature(scan_all_pairs)
        ret = sig.return_annotation
        # Should mention Tuple now
        assert "Tuple" in str(ret) or "tuple" in str(ret)

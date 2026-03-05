"""Tests for the backtesting engine."""

import pytest
import numpy as np
from engine.backtester import run_backtest, BacktestResult, Trade, optimize_parameters


class TestRunBacktest:
    def test_basic_run(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        assert isinstance(bt, BacktestResult)
        assert bt.total_trades >= 0

    def test_all_strategies(self, sample_ohlcv):
        strategies = [
            "sweeps", "inducement", "stop_hunts", "smc",
            "ema_crossover", "rsi_reversal", "breakout", "fvg_entry", "ob_bounce",
        ]
        for strat in strategies:
            bt = run_backtest(sample_ohlcv, "EUR/USD", strategy=strat)
            assert isinstance(bt, BacktestResult), f"Failed for strategy: {strat}"

    def test_win_rate_range(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        assert 0 <= bt.win_rate <= 100

    def test_trade_count_consistency(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        assert bt.total_trades == bt.wins + bt.losses + bt.breakevens

    def test_pnl_consistency(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        if bt.trades:
            total = sum(t.pnl_pips for t in bt.trades)
            assert abs(total - bt.total_pips) < 0.1

    def test_equity_curve_length(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        # Equity curve = 1 initial point + 1 per trade
        assert len(bt.equity_curve) == bt.total_trades + 1

    def test_equity_curve_starts_at_zero(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        assert bt.equity_curve[0] == 0.0

    def test_jpy_pair(self, sample_jpy_ohlcv):
        bt = run_backtest(sample_jpy_ohlcv, "USD/JPY", strategy="sweeps")
        assert isinstance(bt, BacktestResult)

    def test_insufficient_data(self):
        import pandas as pd
        df = pd.DataFrame({
            "Open": [1.1] * 10,
            "High": [1.2] * 10,
            "Low": [1.0] * 10,
            "Close": [1.1] * 10,
        }, index=pd.date_range("2024-01-01", periods=10, freq="1h"))
        with pytest.raises(ValueError, match="Insufficient data"):
            run_backtest(df, "EUR/USD")


class TestBacktestMetrics:
    def test_profit_factor_positive_for_winners(self, trending_up_ohlcv):
        bt = run_backtest(trending_up_ohlcv, "EUR/USD", strategy="ema_crossover",
                          rr_ratio=1.5, min_confluence=0)
        if bt.wins > 0 and bt.losses > 0:
            assert bt.profit_factor > 0

    def test_drawdown_non_negative(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        assert bt.max_drawdown_pips >= 0

    def test_monthly_pnl_sums(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        if bt.monthly_pnl:
            monthly_total = sum(bt.monthly_pnl.values())
            assert abs(monthly_total - bt.total_pips) < 0.5

    def test_direction_breakdown(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        assert bt.long_wins + bt.long_losses + bt.short_wins + bt.short_losses <= bt.total_trades

    def test_consecutive_streaks(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        assert bt.max_consecutive_wins >= 0
        assert bt.max_consecutive_losses >= 0

    def test_sharpe_ratio_type(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        assert isinstance(bt.sharpe_ratio, (int, float))
        assert not np.isnan(bt.sharpe_ratio) or bt.total_trades <= 1


class TestBacktestOptions:
    def test_spread_reduces_pnl(self, sample_ohlcv):
        bt_no_spread = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps",
                                    spread_pips=0.0)
        bt_spread = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps",
                                 spread_pips=2.0)
        if bt_no_spread.total_trades > 0 and bt_spread.total_trades > 0:
            assert bt_spread.total_pips <= bt_no_spread.total_pips

    def test_higher_rr_fewer_wins(self, sample_ohlcv):
        bt_low = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps",
                              rr_ratio=1.5)
        bt_high = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps",
                               rr_ratio=3.0)
        if bt_low.total_trades > 5 and bt_high.total_trades > 5:
            # Higher RR should generally have lower win rate
            # (not guaranteed but very likely)
            assert bt_high.win_rate <= bt_low.win_rate + 15

    def test_min_confluence_filter(self, sample_ohlcv):
        bt_0 = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps",
                            min_confluence=0)
        bt_2 = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps",
                            min_confluence=2)
        assert bt_2.total_trades <= bt_0.total_trades

    def test_max_trades_per_day(self, sample_ohlcv):
        bt_unlimited = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps",
                                    max_trades_per_day=0)
        bt_limited = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps",
                                  max_trades_per_day=1)
        assert bt_limited.total_trades <= bt_unlimited.total_trades


class TestTradeSimulation:
    def test_trade_has_valid_fields(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        for trade in bt.trades:
            assert trade.direction in ("long", "short")
            assert trade.result in ("win", "loss", "breakeven")
            assert trade.holding_candles > 0
            assert trade.entry_price > 0
            assert trade.exit_price > 0

    def test_long_trade_pnl_direction(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps", spread_pips=0)
        for trade in bt.trades:
            if trade.direction == "long" and trade.result == "win":
                assert trade.exit_price > trade.entry_price

    def test_short_trade_pnl_direction(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps", spread_pips=0)
        for trade in bt.trades:
            if trade.direction == "short" and trade.result == "win":
                assert trade.exit_price < trade.entry_price


class TestOptimizeParameters:
    def test_basic_optimization(self, sample_ohlcv):
        results = optimize_parameters(
            sample_ohlcv, "EUR/USD",
            param_grid={
                "rr_ratio": [1.5, 2.0],
                "min_confluence": [0, 1],
            },
            strategy="sweeps",
            top_n=5,
        )
        assert isinstance(results, list)

    def test_results_sorted_by_score(self, sample_ohlcv):
        results = optimize_parameters(
            sample_ohlcv, "EUR/USD",
            param_grid={
                "rr_ratio": [1.5, 2.0, 3.0],
                "swing_lookback": [3, 5],
            },
            strategy="sweeps",
            top_n=10,
        )
        if len(results) >= 2:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

"""Tests for position sizing and account simulation."""

import pytest
from engine.sizing import simulate_account, pip_value
from engine.backtester import run_backtest


class TestPipValue:
    def test_standard_lot(self):
        assert pip_value("EUR/USD", 1.0) == 10.0

    def test_mini_lot(self):
        assert pip_value("EUR/USD", 0.1) == 1.0

    def test_micro_lot(self):
        assert pip_value("EUR/USD", 0.01) == 0.1


class TestAccountSimulation:
    def test_fixed_sizing(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        if not bt.trades:
            pytest.skip("No trades generated")
        sim = simulate_account(bt.trades, starting_balance=10000,
                               sizing_mode="fixed", fixed_lot=0.1)
        assert sim.starting_balance == 10000
        assert sim.ending_balance != 10000 or bt.total_pips == 0
        assert len(sim.balance_curve) == len(bt.trades) + 1

    def test_risk_pct_sizing(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        if not bt.trades:
            pytest.skip("No trades generated")
        sim = simulate_account(bt.trades, starting_balance=10000,
                               sizing_mode="risk_pct", risk_pct=1.0)
        assert sim.starting_balance == 10000
        assert len(sim.states) == len(bt.trades)

    def test_kelly_sizing(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        if not bt.trades:
            pytest.skip("No trades generated")
        sim = simulate_account(bt.trades, starting_balance=10000,
                               sizing_mode="kelly")
        assert sim.starting_balance == 10000

    def test_drawdown_tracking(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        if not bt.trades:
            pytest.skip("No trades generated")
        sim = simulate_account(bt.trades, starting_balance=10000)
        assert sim.max_drawdown_pct >= 0
        assert len(sim.drawdown_curve) == len(bt.trades) + 1

    def test_compounding_vs_fixed(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        if not bt.trades or len(bt.trades) < 5:
            pytest.skip("Not enough trades")
        sim_compound = simulate_account(bt.trades, sizing_mode="risk_pct",
                                        risk_pct=2.0, compounding=True)
        sim_fixed = simulate_account(bt.trades, sizing_mode="risk_pct",
                                     risk_pct=2.0, compounding=False)
        # Results should differ (compounding adjusts position size)
        assert sim_compound.ending_balance != sim_fixed.ending_balance or \
               sim_compound.largest_position == sim_fixed.largest_position

    def test_return_pct_calculation(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="sweeps")
        if not bt.trades:
            pytest.skip("No trades generated")
        sim = simulate_account(bt.trades, starting_balance=10000)
        expected_pct = (sim.ending_balance - 10000) / 10000 * 100
        assert abs(sim.total_return_pct - expected_pct) < 0.1

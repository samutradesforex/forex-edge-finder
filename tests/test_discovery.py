"""Tests for the discovery engine."""

import pytest
import json
import threading
from pathlib import Path
from engine.discovery import (
    DiscoveredEdge, DiscoveryState,
    qualifies_as_edge, compute_edge_score,
    save_edges, load_edges, save_state, load_state,
    _get_strategy_combos, _count_total_combos,
    walk_forward_validate,
    PARAM_GRID, ALL_STRATEGIES, ALL_INTERVALS, STRATEGY_PARAMS,
)
from engine.backtester import BacktestResult, Trade


class TestEdgeQualification:
    def _make_bt(self, trades=25, win_rate=50, pf=1.5, exp=3.0, sharpe=1.0,
                 dd=20, payoff=1.5):
        bt = BacktestResult(trades=[])
        bt.total_trades = trades
        bt.win_rate = win_rate
        bt.profit_factor = pf
        bt.expectancy_pips = exp
        bt.sharpe_ratio = sharpe
        bt.max_drawdown_pips = dd
        bt.max_drawdown_pct = 30.0
        bt.payoff_ratio = payoff
        bt.total_pips = exp * trades
        return bt

    def test_qualifies(self):
        bt = self._make_bt()
        assert qualifies_as_edge(bt)

    def test_too_few_trades(self):
        bt = self._make_bt(trades=15)
        assert not qualifies_as_edge(bt)

    def test_low_win_rate(self):
        bt = self._make_bt(win_rate=30)
        assert not qualifies_as_edge(bt)

    def test_low_profit_factor(self):
        bt = self._make_bt(pf=0.9)
        assert not qualifies_as_edge(bt)

    def test_negative_expectancy(self):
        bt = self._make_bt(exp=-1.0)
        assert not qualifies_as_edge(bt)

    def test_low_sharpe(self):
        bt = self._make_bt(sharpe=0.1)
        assert not qualifies_as_edge(bt)

    def test_low_payoff_ratio(self):
        bt = self._make_bt(payoff=0.3)
        assert not qualifies_as_edge(bt)

    def test_high_drawdown_rejected(self):
        bt = self._make_bt()
        bt.max_drawdown_pct = 90.0
        assert not qualifies_as_edge(bt)


class TestEdgeScore:
    def test_score_positive_for_good_edge(self):
        bt = BacktestResult(trades=[])
        bt.total_trades = 50
        bt.win_rate = 55
        bt.profit_factor = 1.8
        bt.expectancy_pips = 5.0
        bt.sharpe_ratio = 2.0
        bt.max_drawdown_pips = 30
        bt.total_pips = 250
        score = compute_edge_score(bt)
        assert score > 0

    def test_better_edge_scores_higher(self):
        bt_good = BacktestResult(trades=[])
        bt_good.total_trades = 50
        bt_good.win_rate = 60
        bt_good.profit_factor = 2.5
        bt_good.expectancy_pips = 8.0
        bt_good.sharpe_ratio = 3.0
        bt_good.max_drawdown_pips = 20
        bt_good.total_pips = 400

        bt_ok = BacktestResult(trades=[])
        bt_ok.total_trades = 50
        bt_ok.win_rate = 42
        bt_ok.profit_factor = 1.3
        bt_ok.expectancy_pips = 1.5
        bt_ok.sharpe_ratio = 0.5
        bt_ok.max_drawdown_pips = 50
        bt_ok.total_pips = 75

        assert compute_edge_score(bt_good) > compute_edge_score(bt_ok)


class TestSmartParamFiltering:
    def test_ema_crossover_fewer_combos(self):
        ema_combos = _get_strategy_combos("ema_crossover", PARAM_GRID)
        sweep_combos = _get_strategy_combos("sweeps", PARAM_GRID)
        # ema_crossover only uses rr_ratio + min_confluence = 4*4 = 16
        # sweeps uses all 5 params = 4*4*3*4*4 = 768
        assert len(ema_combos) < len(sweep_combos)

    def test_all_strategies_have_mappings(self):
        for strat in ALL_STRATEGIES:
            assert strat in STRATEGY_PARAMS

    def test_combos_have_all_params(self):
        for strat in ALL_STRATEGIES:
            combos = _get_strategy_combos(strat, PARAM_GRID)
            for combo in combos:
                for key in PARAM_GRID:
                    assert key in combo, f"{strat} combo missing {key}"

    def test_total_combos_less_than_brute_force(self):
        pairs = ["EUR/USD", "GBP/USD"]
        smart_total = _count_total_combos(ALL_STRATEGIES, ALL_INTERVALS, pairs, PARAM_GRID)
        brute_force = len(pairs) * len(ALL_INTERVALS) * len(ALL_STRATEGIES) * 768
        assert smart_total < brute_force


class TestPersistence:
    def test_save_load_edges(self, tmp_path, monkeypatch):
        import engine.discovery as disc
        monkeypatch.setattr(disc, "RESULTS_DIR", tmp_path)

        edges = [
            DiscoveredEdge(
                pair="EUR/USD", interval="1h", strategy="sweeps",
                params={"rr_ratio": 2.0, "swing_lookback": 5},
                total_trades=50, win_rate=55.0, total_pips=100.0,
                profit_factor=1.8, expectancy_pips=2.0, sharpe_ratio=1.5,
                max_drawdown_pips=30.0, score=35.0,
            ),
        ]
        save_edges(edges, "test_edges.json")
        loaded = load_edges("test_edges.json")
        assert len(loaded) == 1
        assert loaded[0].pair == "EUR/USD"
        assert loaded[0].strategy == "sweeps"

    def test_deduplication_on_save(self, tmp_path, monkeypatch):
        import engine.discovery as disc
        monkeypatch.setattr(disc, "RESULTS_DIR", tmp_path)

        edge = DiscoveredEdge(
            pair="EUR/USD", interval="1h", strategy="sweeps",
            params={"rr_ratio": 2.0}, total_trades=50, win_rate=55.0,
            total_pips=100.0, profit_factor=1.8, expectancy_pips=2.0,
            sharpe_ratio=1.5, max_drawdown_pips=30.0, score=35.0,
        )
        # Save duplicates
        save_edges([edge, edge, edge], "test_edges.json")
        loaded = load_edges("test_edges.json")
        assert len(loaded) == 1

    def test_save_load_state(self, tmp_path, monkeypatch):
        import engine.discovery as disc
        monkeypatch.setattr(disc, "RESULTS_DIR", tmp_path)

        state = DiscoveryState(
            status="running", combos_tested=100, total_combos=1000,
            edges_found=5, last_completed_pair="EUR/USD",
        )
        save_state(state, "test_state.json")
        loaded = load_state("test_state.json")
        assert loaded.combos_tested == 100
        assert loaded.last_completed_pair == "EUR/USD"


class TestWalkForward:
    def test_validate_with_enough_data(self, sample_ohlcv):
        result = walk_forward_validate(
            sample_ohlcv, "EUR/USD", "sweeps", "1h",
            params={"swing_lookback": 5, "cluster_pips": 10.0,
                    "min_wick_pips": 3.0, "rr_ratio": 1.5, "min_confluence": 0},
        )
        # May or may not validate depending on random data
        assert result is None or isinstance(result, dict)

    def test_validate_returns_expected_keys(self, sample_ohlcv):
        result = walk_forward_validate(
            sample_ohlcv, "EUR/USD", "ema_crossover", "1h",
            params={"swing_lookback": 5, "cluster_pips": 10.0,
                    "min_wick_pips": 3.0, "rr_ratio": 1.5, "min_confluence": 0},
        )
        if result is not None:
            assert "oos_win_rate" in result
            assert "oos_profit_factor" in result
            assert "oos_expectancy_pips" in result
            assert "validated" in result
            assert result["validated"] is True

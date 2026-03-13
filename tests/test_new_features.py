"""Tests for new features: strategies, charts, CLI."""

import pytest
import numpy as np
from engine.backtester import run_backtest, BacktestResult
from engine.strategies import registry as strategy_registry


class TestNewStrategies:
    """Test that the 3 new strategies register and run without errors."""

    def test_macd_divergence_registered(self):
        meta = strategy_registry.get("macd_divergence")
        assert meta is not None
        assert meta.display_name == "MACD Divergence"
        assert meta.category == "momentum"

    def test_bollinger_bands_registered(self):
        meta = strategy_registry.get("bollinger_bands")
        assert meta is not None
        assert meta.display_name == "Bollinger Band Reversion"
        assert meta.category == "reversal"

    def test_supply_demand_registered(self):
        meta = strategy_registry.get("supply_demand")
        assert meta is not None
        assert meta.display_name == "Supply/Demand Zone Bounce"
        assert meta.category == "smc"

    def test_macd_divergence_backtest(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="macd_divergence")
        assert isinstance(bt, BacktestResult)
        assert bt.total_trades >= 0

    def test_bollinger_bands_backtest(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="bollinger_bands")
        assert isinstance(bt, BacktestResult)
        assert bt.total_trades >= 0

    def test_supply_demand_backtest(self, sample_ohlcv):
        bt = run_backtest(sample_ohlcv, "EUR/USD", strategy="supply_demand")
        assert isinstance(bt, BacktestResult)
        assert bt.total_trades >= 0

    def test_all_strategies_run(self, sample_ohlcv):
        """Every registered strategy should run without error."""
        for name in strategy_registry.list():
            bt = run_backtest(sample_ohlcv, "EUR/USD", strategy=name)
            assert isinstance(bt, BacktestResult), f"Failed for {name}"

    def test_strategy_count(self):
        """We should have at least 11 strategies now."""
        assert len(strategy_registry.list()) >= 11

    def test_new_strategies_in_discovery_params(self):
        """New strategies should be included in discovery param mapping."""
        from engine.discovery import ALL_STRATEGIES, STRATEGY_PARAMS
        for name in ["macd_divergence", "bollinger_bands", "supply_demand"]:
            assert name in ALL_STRATEGIES, f"{name} not in ALL_STRATEGIES"
            assert name in STRATEGY_PARAMS, f"{name} not in STRATEGY_PARAMS"


class TestChartComponents:
    """Test the new chart helper functions."""

    def test_edge_heatmap_empty(self):
        from ui.components import edge_heatmap
        fig = edge_heatmap([], metric="score")
        # Should return an empty figure without error
        assert fig is not None

    def test_edge_heatmap_with_edges(self):
        from ui.components import edge_heatmap
        from engine.discovery import DiscoveredEdge

        edges = [
            DiscoveredEdge(
                pair="EUR/USD", interval="1h", strategy="sweeps",
                params={}, total_trades=50, win_rate=55.0,
                total_pips=100.0, profit_factor=1.8,
                expectancy_pips=2.0, sharpe_ratio=1.5,
                max_drawdown_pips=30.0, score=45.0,
                discovered_at="2024-01-01",
            ),
            DiscoveredEdge(
                pair="GBP/USD", interval="4h", strategy="ema_crossover",
                params={}, total_trades=30, win_rate=50.0,
                total_pips=50.0, profit_factor=1.3,
                expectancy_pips=1.5, sharpe_ratio=0.8,
                max_drawdown_pips=20.0, score=30.0,
                discovered_at="2024-01-01",
            ),
        ]
        fig = edge_heatmap(edges, metric="score")
        assert fig is not None
        assert len(fig.data) > 0

    def test_monte_carlo_chart_empty(self):
        from ui.components import monte_carlo_chart
        fig = monte_carlo_chart([0.0, 1.0], n_simulations=10)
        assert fig is not None

    def test_monte_carlo_chart_real(self):
        from ui.components import monte_carlo_chart
        # Simulate a real equity curve
        np.random.seed(42)
        pnls = np.random.normal(0.5, 5.0, 50)
        equity = np.concatenate([[0], np.cumsum(pnls)])
        fig = monte_carlo_chart(equity, n_simulations=100)
        assert fig is not None
        # Should have 6 traces: 2 CI bands + 2 IQR bands + median + actual
        assert len(fig.data) == 6

    def test_correlation_matrix_insufficient(self):
        from ui.components import correlation_matrix_chart
        fig = correlation_matrix_chart([], height=300)
        assert fig is not None

    def test_correlation_matrix_with_edges(self):
        from ui.components import correlation_matrix_chart
        from engine.discovery import DiscoveredEdge

        edges = []
        for i, pair in enumerate(["EUR/USD", "GBP/USD", "USD/JPY"]):
            np.random.seed(i)
            ec = np.concatenate([[0], np.cumsum(np.random.normal(0.3, 3, 30))])
            e = DiscoveredEdge(
                pair=pair, interval="1h", strategy="sweeps",
                params={}, total_trades=30, win_rate=55.0,
                total_pips=float(ec[-1]), profit_factor=1.5,
                expectancy_pips=1.0, sharpe_ratio=1.0,
                max_drawdown_pips=10.0, score=40.0,
                discovered_at="2024-01-01",
            )
            e.equity_curve = ec.tolist()
            edges.append(e)

        fig = correlation_matrix_chart(edges, height=400)
        assert fig is not None
        assert len(fig.data) > 0


class TestCLI:
    """Test CLI argument parsing."""

    def test_cli_imports(self):
        import cli
        assert hasattr(cli, "main")
        assert hasattr(cli, "cmd_discover")
        assert hasattr(cli, "cmd_edges")
        assert hasattr(cli, "cmd_backtest")
        assert hasattr(cli, "cmd_strategies")
        assert hasattr(cli, "cmd_pairs")

    def test_strategies_command(self, capsys):
        import cli
        args = type("Args", (), {})()
        cli.cmd_strategies(args)
        captured = capsys.readouterr()
        assert "sweeps" in captured.out.lower() or "Sweeps" in captured.out
        assert "macd" in captured.out.lower() or "MACD" in captured.out

    def test_pairs_command(self, capsys):
        import cli
        args = type("Args", (), {})()
        cli.cmd_pairs(args)
        captured = capsys.readouterr()
        assert "EUR/USD" in captured.out
        assert "GBP/USD" in captured.out

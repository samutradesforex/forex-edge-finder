"""Continuous strategy discovery engine.

Runs in the background, sweeping across all pairs × strategies × timeframes
× parameter combos to find profitable edges. Persists results to disk so
they survive restarts. Designed to be driven by a Streamlit background thread
or a standalone CLI runner.
"""

import json
import os
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Dict, Optional, Callable

import pandas as pd

from data.loader import FOREX_PAIRS, fetch_pair
from engine.backtester import run_backtest, optimize_parameters, BacktestResult
from engine.liquidity import (
    get_pip_size, calc_atr, calc_rsi, calc_ema, find_fvgs, find_order_blocks,
)


# ── Configuration ──────────────────────────────────────────────────────────

RESULTS_DIR = Path("discovery_results")

ALL_STRATEGIES = [
    "sweeps", "inducement", "stop_hunts", "smc",
    "ema_crossover", "rsi_reversal", "breakout", "fvg_entry", "ob_bounce",
]

ALL_INTERVALS = ["1h", "4h", "1d"]

PARAM_GRID = {
    "swing_lookback": [3, 5, 8, 13],
    "cluster_pips": [5.0, 10.0, 15.0, 20.0],
    "min_wick_pips": [2.0, 3.0, 5.0],
    "rr_ratio": [1.5, 2.0, 2.5, 3.0],
    "min_confluence": [0, 1, 2, 3],
}

# Minimum thresholds for a strategy to be considered an "edge"
EDGE_THRESHOLDS = {
    "min_trades": 10,
    "min_win_rate": 40.0,
    "min_profit_factor": 1.2,
    "min_expectancy_pips": 1.0,
    "min_sharpe": 0.3,
}


@dataclass
class DiscoveredEdge:
    """A profitable strategy/param combo found by the discovery engine."""
    pair: str
    interval: str
    strategy: str
    params: Dict
    total_trades: int
    win_rate: float
    total_pips: float
    profit_factor: float
    expectancy_pips: float
    sharpe_ratio: float
    max_drawdown_pips: float
    score: float
    discovered_at: str = ""
    period: str = "6mo"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class DiscoveryState:
    """Tracks progress of an ongoing discovery run."""
    status: str = "idle"  # "idle", "running", "paused", "completed"
    current_pair: str = ""
    current_strategy: str = ""
    current_interval: str = ""
    combos_tested: int = 0
    total_combos: int = 0
    edges_found: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""
    edges: List[DiscoveredEdge] = field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        if self.total_combos == 0:
            return 0.0
        return min(100.0, self.combos_tested / self.total_combos * 100)


# ── Persistence ────────────────────────────────────────────────────────────

def _ensure_dir():
    RESULTS_DIR.mkdir(exist_ok=True)


def save_edges(edges: List[DiscoveredEdge], filename: str = "edges.json"):
    _ensure_dir()
    path = RESULTS_DIR / filename
    data = [e.to_dict() for e in edges]
    path.write_text(json.dumps(data, indent=2, default=str))


def load_edges(filename: str = "edges.json") -> List[DiscoveredEdge]:
    path = RESULTS_DIR / filename
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return [DiscoveredEdge.from_dict(d) for d in data]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


def save_state(state: DiscoveryState, filename: str = "state.json"):
    _ensure_dir()
    path = RESULTS_DIR / filename
    d = {
        "status": state.status,
        "current_pair": state.current_pair,
        "current_strategy": state.current_strategy,
        "current_interval": state.current_interval,
        "combos_tested": state.combos_tested,
        "total_combos": state.total_combos,
        "edges_found": state.edges_found,
        "elapsed_seconds": state.elapsed_seconds,
        "error": state.error,
    }
    path.write_text(json.dumps(d, indent=2))


def load_state(filename: str = "state.json") -> DiscoveryState:
    path = RESULTS_DIR / filename
    if not path.exists():
        return DiscoveryState()
    try:
        d = json.loads(path.read_text())
        return DiscoveryState(**{k: v for k, v in d.items()
                                 if k != "edges"})
    except (json.JSONDecodeError, TypeError, KeyError):
        return DiscoveryState()


# ── Edge qualification ─────────────────────────────────────────────────────

def qualifies_as_edge(bt: BacktestResult, thresholds: dict = None) -> bool:
    """Check if a backtest result qualifies as a tradeable edge."""
    t = thresholds or EDGE_THRESHOLDS
    if bt.total_trades < t["min_trades"]:
        return False
    if bt.win_rate < t["min_win_rate"]:
        return False
    if bt.profit_factor < t["min_profit_factor"]:
        return False
    if bt.expectancy_pips < t["min_expectancy_pips"]:
        return False
    if bt.sharpe_ratio < t["min_sharpe"]:
        return False
    return True


def compute_edge_score(bt: BacktestResult) -> float:
    """Compute a composite score for ranking discovered edges."""
    pf = min(bt.profit_factor, 5.0) if bt.profit_factor != float("inf") else 5.0
    rf = 0.0
    if bt.max_drawdown_pips > 0:
        rf = min(bt.total_pips / bt.max_drawdown_pips, 10.0)

    norm_exp = min(max(bt.expectancy_pips, -10), 10) * 5
    norm_pf = pf * 10
    norm_sharpe = min(max(bt.sharpe_ratio, -2), 5) * 10
    norm_wr = bt.win_rate
    norm_rf = rf * 5

    score = (
        norm_exp * 0.25 +
        norm_pf * 0.25 +
        norm_sharpe * 0.20 +
        norm_wr * 0.15 +
        norm_rf * 0.15
    )
    return round(score, 2)


# ── Core discovery loop ───────────────────────────────────────────────────

def _fetch_data(pair: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetch data for a pair/interval, handling the 4h resample."""
    try:
        yf_interval = "1h" if interval == "4h" else interval
        df = fetch_pair(pair, period=period, interval=yf_interval)
        if interval == "4h":
            agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            if "Volume" in df.columns:
                agg["Volume"] = "sum"
            df = df.resample("4h").agg(agg).dropna()
        if len(df) < 50:
            return None
        return df
    except Exception:
        return None


def _precompute_indicators(df: pd.DataFrame, pair: str) -> dict:
    """Compute indicators once per pair/interval combo."""
    pip_size = get_pip_size(pair)
    return {
        "atr": calc_atr(df),
        "rsi": calc_rsi(df),
        "ema_fast": calc_ema(df["Close"], 21),
        "ema_slow": calc_ema(df["Close"], 50),
        "fvgs": find_fvgs(df, pip_size=pip_size),
        "obs": find_order_blocks(df, pip_size=pip_size),
    }


def run_discovery(
    pairs: List[str] = None,
    strategies: List[str] = None,
    intervals: List[str] = None,
    period: str = "6mo",
    param_grid: dict = None,
    thresholds: dict = None,
    state: DiscoveryState = None,
    stop_event: threading.Event = None,
    on_progress: Callable = None,
) -> DiscoveryState:
    """Run a full discovery sweep.

    Iterates across pairs × intervals × strategies × param combos.
    Saves edges to disk incrementally.

    Args:
        pairs: Currency pairs to scan (defaults to all)
        strategies: Strategies to test (defaults to all)
        intervals: Timeframes to test (defaults to 1h, 4h, 1d)
        period: Data period for yfinance
        param_grid: Parameter grid to sweep
        thresholds: Edge qualification thresholds
        state: Existing state to resume from
        stop_event: Threading event to signal stop
        on_progress: Callback(state) called after each pair/strategy combo
    """
    if pairs is None:
        pairs = list(FOREX_PAIRS.keys())
    if strategies is None:
        strategies = ALL_STRATEGIES
    if intervals is None:
        intervals = ALL_INTERVALS
    if param_grid is None:
        param_grid = PARAM_GRID
    if state is None:
        state = DiscoveryState()
    if stop_event is None:
        stop_event = threading.Event()

    # Compute total work
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combos = list(product(*param_values))
    num_param_combos = len(param_combos)
    state.total_combos = len(pairs) * len(intervals) * len(strategies) * num_param_combos

    state.status = "running"
    state.edges = load_edges()
    t0 = time.time()

    existing_keys = {
        (e.pair, e.interval, e.strategy,
         tuple(sorted(e.params.items())))
        for e in state.edges
    }

    for pair in pairs:
        for intv in intervals:
            if stop_event.is_set():
                state.status = "paused"
                save_state(state)
                return state

            state.current_pair = pair
            state.current_interval = intv

            df = _fetch_data(pair, period, intv)
            if df is None:
                state.combos_tested += len(strategies) * num_param_combos
                continue

            # Pre-compute indicators once per pair/interval
            precomputed = _precompute_indicators(df, pair)

            for strat in strategies:
                if stop_event.is_set():
                    state.status = "paused"
                    save_state(state)
                    return state

                state.current_strategy = strat

                for combo in param_combos:
                    params = dict(zip(param_names, combo))

                    # Skip already-discovered combos
                    key = (pair, intv, strat, tuple(sorted(params.items())))
                    if key in existing_keys:
                        state.combos_tested += 1
                        continue

                    try:
                        bt = run_backtest(
                            df, pair, strategy=strat, interval=intv,
                            _precomputed=precomputed, **params,
                        )

                        if qualifies_as_edge(bt, thresholds):
                            edge = DiscoveredEdge(
                                pair=pair,
                                interval=intv,
                                strategy=strat,
                                params=params,
                                total_trades=bt.total_trades,
                                win_rate=round(bt.win_rate, 1),
                                total_pips=round(bt.total_pips, 1),
                                profit_factor=round(min(bt.profit_factor, 99.9), 2),
                                expectancy_pips=round(bt.expectancy_pips, 1),
                                sharpe_ratio=round(bt.sharpe_ratio, 2),
                                max_drawdown_pips=round(bt.max_drawdown_pips, 1),
                                score=compute_edge_score(bt),
                                discovered_at=datetime.now().isoformat(),
                                period=period,
                            )
                            state.edges.append(edge)
                            state.edges_found += 1
                            existing_keys.add(key)

                            # Save incrementally every 10 edges
                            if state.edges_found % 10 == 0:
                                save_edges(state.edges)

                    except Exception:
                        pass

                    state.combos_tested += 1

                state.elapsed_seconds = time.time() - t0
                save_state(state)
                if on_progress:
                    on_progress(state)

    # Final save
    state.status = "completed"
    state.elapsed_seconds = time.time() - t0
    state.edges.sort(key=lambda e: e.score, reverse=True)
    save_edges(state.edges)
    save_state(state)
    return state


# ── Background runner for Streamlit ────────────────────────────────────────

_bg_thread: Optional[threading.Thread] = None
_bg_stop_event = threading.Event()
_bg_state = DiscoveryState()
_bg_lock = threading.Lock()


def start_discovery_background(
    pairs: List[str] = None,
    strategies: List[str] = None,
    intervals: List[str] = None,
    period: str = "6mo",
    param_grid: dict = None,
    thresholds: dict = None,
) -> bool:
    """Start discovery in a background thread. Returns True if started."""
    global _bg_thread, _bg_stop_event, _bg_state

    with _bg_lock:
        if _bg_thread is not None and _bg_thread.is_alive():
            return False  # already running

        _bg_stop_event.clear()
        _bg_state = DiscoveryState()

        def _run():
            global _bg_state
            _bg_state = run_discovery(
                pairs=pairs, strategies=strategies, intervals=intervals,
                period=period, param_grid=param_grid, thresholds=thresholds,
                state=_bg_state, stop_event=_bg_stop_event,
            )

        _bg_thread = threading.Thread(target=_run, daemon=True)
        _bg_thread.start()
        return True


def stop_discovery_background():
    """Signal the background discovery to stop."""
    _bg_stop_event.set()


def get_discovery_state() -> DiscoveryState:
    """Get current state of background discovery."""
    with _bg_lock:
        return _bg_state


def is_discovery_running() -> bool:
    """Check if discovery is currently running."""
    with _bg_lock:
        return _bg_thread is not None and _bg_thread.is_alive()


# ── CLI runner ─────────────────────────────────────────────────────────────

def main():
    """Run discovery from the command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Forex Edge Discovery")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Pairs to scan (default: all)")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Strategies to test (default: all)")
    parser.add_argument("--intervals", nargs="+", default=None,
                        help="Timeframes (default: 1h 4h 1d)")
    parser.add_argument("--period", default="6mo",
                        help="Data period (default: 6mo)")
    args = parser.parse_args()

    def _progress(state):
        pct = state.progress_pct
        eta = ""
        if state.combos_tested > 0 and pct > 0:
            remaining = state.elapsed_seconds / pct * (100 - pct)
            eta = f" | ETA: {remaining/60:.0f}m"
        print(
            f"\r[{pct:5.1f}%] {state.current_pair} {state.current_interval} "
            f"{state.current_strategy} | "
            f"{state.edges_found} edges found | "
            f"{state.combos_tested}/{state.total_combos}{eta}",
            end="", flush=True,
        )

    print("Starting Forex Edge Discovery...")
    print(f"Pairs: {args.pairs or 'all'}")
    print(f"Strategies: {args.strategies or 'all'}")
    print(f"Intervals: {args.intervals or ALL_INTERVALS}")
    print(f"Period: {args.period}")
    print()

    state = run_discovery(
        pairs=args.pairs,
        strategies=args.strategies,
        intervals=args.intervals,
        period=args.period,
        on_progress=_progress,
    )

    print(f"\n\nDiscovery complete!")
    print(f"Tested: {state.combos_tested} combos in {state.elapsed_seconds:.0f}s")
    print(f"Edges found: {state.edges_found}")

    if state.edges:
        print(f"\nTop 10 edges:")
        print(f"{'Pair':<10} {'TF':<4} {'Strategy':<16} {'Trades':>6} {'WR':>6} "
              f"{'Pips':>8} {'PF':>6} {'Sharpe':>7} {'Score':>7}")
        print("-" * 80)
        for e in state.edges[:10]:
            print(f"{e.pair:<10} {e.interval:<4} {e.strategy:<16} "
                  f"{e.total_trades:>6} {e.win_rate:>5.1f}% "
                  f"{e.total_pips:>+7.1f} {e.profit_factor:>6.2f} "
                  f"{e.sharpe_ratio:>7.2f} {e.score:>7.1f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

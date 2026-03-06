"""Continuous strategy discovery engine.

Runs in the background, sweeping across all pairs × strategies × timeframes
× parameter combos to find profitable edges. Persists results to disk so
they survive restarts. Designed to be driven by a Streamlit background thread
or a standalone CLI runner.

Features:
- Walk-forward validation (in-sample train → out-of-sample test)
- Edge deduplication
- Resume from crash (tracks last completed position)
- Smart param filtering (skips irrelevant params per strategy)
- Proper logging instead of silent exception swallowing
"""

import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Dict, Optional, Callable

import pandas as pd
import numpy as np

from data.loader import FOREX_PAIRS, fetch_pair, fetch_max_data
from engine.backtester import run_backtest, BacktestResult
from engine.liquidity import (
    get_pip_size, calc_atr, calc_rsi, calc_ema, find_fvgs, find_order_blocks,
)
from engine.strategies import registry as strategy_registry

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

RESULTS_DIR = Path("discovery_results")

# Derive strategy list from registry (auto-discovers new strategies)
ALL_STRATEGIES = strategy_registry.list()

ALL_INTERVALS = ["1h", "4h", "1d"]

PARAM_GRID = {
    "swing_lookback": [3, 5, 8, 13],
    "cluster_pips": [5.0, 10.0, 15.0, 20.0],
    "min_wick_pips": [2.0, 3.0, 5.0],
    "rr_ratio": [1.5, 2.0, 2.5, 3.0],
    "min_confluence": [0, 1, 2, 3],
}

# Derive relevant params from registry (auto-discovers new strategies)
STRATEGY_PARAMS = {
    name: strategy_registry.relevant_params_for(name)
    for name in ALL_STRATEGIES
}

# Minimum thresholds for a strategy to be considered an "edge"
EDGE_THRESHOLDS = {
    "min_trades": 10,
    "min_win_rate": 40.0,
    "min_profit_factor": 1.2,
    "min_expectancy_pips": 1.0,
    "min_sharpe": 0.3,
}

# Walk-forward validation settings
WALK_FORWARD_SPLIT = 0.7  # 70% in-sample, 30% out-of-sample
WALK_FORWARD_DECAY = 0.5  # OOS must retain at least 50% of in-sample performance


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
    # Walk-forward validation fields
    oos_win_rate: float = 0.0
    oos_profit_factor: float = 0.0
    oos_expectancy_pips: float = 0.0
    oos_total_trades: int = 0
    oos_sharpe_ratio: float = 0.0
    validated: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        # Handle edges saved before validation fields existed
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


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
    edges_validated: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""
    edges: List[DiscoveredEdge] = field(default_factory=list)
    # Resume tracking
    last_completed_pair: str = ""
    last_completed_interval: str = ""
    last_completed_strategy: str = ""

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
    # Deduplicate before saving
    seen = set()
    unique = []
    for e in edges:
        key = (e.pair, e.interval, e.strategy, tuple(sorted(e.params.items())))
        if key not in seen:
            seen.add(key)
            unique.append(e)
    data = [e.to_dict() for e in unique]
    path.write_text(json.dumps(data, indent=2, default=str))


def load_edges(filename: str = "edges.json") -> List[DiscoveredEdge]:
    path = RESULTS_DIR / filename
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return [DiscoveredEdge.from_dict(d) for d in data]
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("Failed to load edges from %s: %s", path, e)
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
        "edges_validated": state.edges_validated,
        "elapsed_seconds": state.elapsed_seconds,
        "error": state.error,
        "last_completed_pair": state.last_completed_pair,
        "last_completed_interval": state.last_completed_interval,
        "last_completed_strategy": state.last_completed_strategy,
    }
    path.write_text(json.dumps(d, indent=2))


def load_state(filename: str = "state.json") -> DiscoveryState:
    path = RESULTS_DIR / filename
    if not path.exists():
        return DiscoveryState()
    try:
        d = json.loads(path.read_text())
        return DiscoveryState(**{k: v for k, v in d.items()
                                 if k != "edges" and k in {
                                     f.name for f in DiscoveryState.__dataclass_fields__.values()
                                 }})
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("Failed to load state: %s", e)
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


# ── Walk-forward validation ──────────────────────────────────────────────

def walk_forward_validate(
    df: pd.DataFrame, pair: str, strategy: str, interval: str,
    params: dict, precomputed_is: dict = None,
    split: float = WALK_FORWARD_SPLIT,
    decay: float = WALK_FORWARD_DECAY,
    thresholds: dict = None,
) -> Optional[Dict]:
    """Run walk-forward validation: train on first portion, test on rest.

    Returns dict with OOS metrics if validated, None if failed.
    """
    n = len(df)
    split_idx = int(n * split)

    if split_idx < 50 or (n - split_idx) < 30:
        return None

    # Out-of-sample data — include a lookback window before the split point
    # so that EMA/RSI/ATR have proper warm-up data
    indicator_warmup = 60  # candles of lookback for indicator calculation
    warmup_start = max(0, split_idx - indicator_warmup)
    df_oos_with_warmup = df.iloc[warmup_start:].copy()
    df_oos = df.iloc[split_idx:].copy()
    if len(df_oos) < 30:
        return None

    try:
        # Compute indicators on data WITH warm-up, then slice to OOS portion
        pip_size = get_pip_size(pair)
        atr_full = calc_atr(df_oos_with_warmup)
        rsi_full = calc_rsi(df_oos_with_warmup)
        ema_fast_full = calc_ema(df_oos_with_warmup["Close"], 21)
        ema_slow_full = calc_ema(df_oos_with_warmup["Close"], 50)

        # Slice indicators to match the OOS DataFrame index
        oos_precomputed = {
            "atr": atr_full.loc[df_oos.index],
            "rsi": rsi_full.loc[df_oos.index],
            "ema_fast": ema_fast_full.loc[df_oos.index],
            "ema_slow": ema_slow_full.loc[df_oos.index],
            "fvgs": find_fvgs(df_oos, pip_size=pip_size),
            "obs": find_order_blocks(df_oos, pip_size=pip_size),
        }

        bt_oos = run_backtest(
            df_oos, pair, strategy=strategy, interval=interval,
            _precomputed=oos_precomputed, **params,
        )

        if bt_oos.total_trades < 5:
            return None

        # Check OOS performance meets minimum thresholds
        t = thresholds or EDGE_THRESHOLDS
        oos_ok = (
            bt_oos.win_rate >= t["min_win_rate"] * decay and
            bt_oos.profit_factor >= max(t["min_profit_factor"] * decay, 1.0) and
            bt_oos.expectancy_pips > 0
        )

        if not oos_ok:
            return None

        return {
            "oos_win_rate": round(bt_oos.win_rate, 1),
            "oos_profit_factor": round(min(bt_oos.profit_factor, 99.9), 2),
            "oos_expectancy_pips": round(bt_oos.expectancy_pips, 1),
            "oos_total_trades": bt_oos.total_trades,
            "oos_sharpe_ratio": round(bt_oos.sharpe_ratio, 2),
            "validated": True,
        }

    except Exception as e:
        logger.debug("Walk-forward validation error for %s %s %s: %s",
                     pair, strategy, interval, e)
        return None


# ── Smart param grid ─────────────────────────────────────────────────────

def _get_strategy_combos(strategy: str, param_grid: dict) -> List[Dict]:
    """Get only the relevant parameter combinations for a strategy."""
    relevant_params = STRATEGY_PARAMS.get(strategy, list(param_grid.keys()))
    param_names = [p for p in relevant_params if p in param_grid]
    param_values = [param_grid[p] for p in param_names]

    # For params not relevant to this strategy, use defaults
    default_values = {
        "swing_lookback": 5,
        "cluster_pips": 10.0,
        "min_wick_pips": 3.0,
        "rr_ratio": 2.0,
        "min_confluence": 0,
    }

    combos = []
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        # Fill in defaults for non-relevant params
        for key in param_grid:
            if key not in params:
                params[key] = default_values.get(key, param_grid[key][0])
        combos.append(params)

    return combos


def _count_total_combos(strategies: List[str], intervals: List[str],
                        pairs: List[str], param_grid: dict) -> int:
    """Count total combos accounting for smart param filtering."""
    total = 0
    for strat in strategies:
        combos = _get_strategy_combos(strat, param_grid)
        total += len(combos) * len(intervals) * len(pairs)
    return total


# ── Core discovery loop ───────────────────────────────────────────────────

def _fetch_data(pair: str, period: str, interval: str,
                use_max: bool = False) -> Optional[pd.DataFrame]:
    """Fetch data for a pair/interval, handling the 4h resample.

    Args:
        use_max: If True, fetch maximum available history (better for discovery)
    """
    try:
        if use_max:
            df = fetch_max_data(pair, interval)
        else:
            yf_interval = "1h" if interval == "4h" else interval
            df = fetch_pair(pair, period=period, interval=yf_interval)
            if interval == "4h":
                agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
                if "Volume" in df.columns:
                    agg["Volume"] = "sum"
                df = df.resample("4h").agg(agg).dropna()
        if len(df) < 50:
            logger.warning("Insufficient data for %s %s: %d rows", pair, interval, len(df))
            return None
        return df
    except Exception as e:
        logger.error("Failed to fetch data for %s %s: %s", pair, interval, e)
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
    validate: bool = True,
    resume: bool = False,
) -> DiscoveryState:
    """Run a full discovery sweep.

    Iterates across pairs × intervals × strategies × param combos.
    Saves edges to disk incrementally. Uses smart param filtering
    and walk-forward validation.

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
        validate: Whether to run walk-forward validation on edges
        resume: Whether to resume from last saved position
    """
    if pairs is None:
        pairs = list(FOREX_PAIRS.keys())
    if strategies is None:
        strategies = ALL_STRATEGIES
    if intervals is None:
        intervals = ALL_INTERVALS
    if param_grid is None:
        param_grid = PARAM_GRID
    if stop_event is None:
        stop_event = threading.Event()

    # Resume from saved state if requested
    if resume and state is None:
        state = load_state()
        if state.status not in ("paused", "running"):
            state = DiscoveryState()
    if state is None:
        state = DiscoveryState()

    # Compute total work with smart filtering
    state.total_combos = _count_total_combos(strategies, intervals, pairs, param_grid)
    state.status = "running"
    state.edges = load_edges()
    t0 = time.time()

    existing_keys = {
        (e.pair, e.interval, e.strategy,
         tuple(sorted(e.params.items())))
        for e in state.edges
    }

    # Determine resume position using (pair, interval, strategy) tuple indices
    # to avoid lexicographic comparison bugs across iteration order
    should_skip = resume and state.last_completed_pair
    resume_key = None
    if should_skip:
        resume_key = (state.last_completed_pair, state.last_completed_interval,
                      state.last_completed_strategy)

    found_resume_point = not should_skip  # True if not resuming

    for pair in pairs:
        for intv in intervals:
            if stop_event.is_set():
                state.status = "paused"
                save_state(state)
                return state

            state.current_pair = pair
            state.current_interval = intv

            # Resume: skip already-completed pair/interval combos
            if not found_resume_point:
                if pair != resume_key[0] or intv != resume_key[1]:
                    for strat in strategies:
                        combos = _get_strategy_combos(strat, param_grid)
                        state.combos_tested += len(combos)
                    continue

            df = _fetch_data(pair, period, intv, use_max=True)
            if df is None:
                for strat in strategies:
                    combos = _get_strategy_combos(strat, param_grid)
                    state.combos_tested += len(combos)
                if not found_resume_point:
                    found_resume_point = True  # resume point pair/intv matched but no data
                continue

            # For walk-forward: split data
            split_idx = int(len(df) * WALK_FORWARD_SPLIT)
            df_is = df.iloc[:split_idx].copy() if validate and split_idx >= 50 else df

            # Pre-compute indicators for in-sample portion
            precomputed = _precompute_indicators(df_is, pair)

            for strat in strategies:
                if stop_event.is_set():
                    state.status = "paused"
                    save_state(state)
                    return state

                # Resume: skip completed strategies within the resume pair/interval
                if not found_resume_point:
                    combos = _get_strategy_combos(strat, param_grid)
                    state.combos_tested += len(combos)
                    if strat == resume_key[2]:
                        found_resume_point = True  # Done skipping
                    continue

                state.current_strategy = strat

                # Smart param filtering: only test relevant params
                strategy_combos = _get_strategy_combos(strat, param_grid)

                for params in strategy_combos:
                    # Skip already-discovered combos
                    key = (pair, intv, strat, tuple(sorted(params.items())))
                    if key in existing_keys:
                        state.combos_tested += 1
                        continue

                    try:
                        bt = run_backtest(
                            df_is, pair, strategy=strat, interval=intv,
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

                            # Walk-forward validation
                            if validate and len(df) > len(df_is) + 30:
                                oos_result = walk_forward_validate(
                                    df, pair, strat, intv, params,
                                    thresholds=thresholds,
                                )
                                if oos_result:
                                    edge.oos_win_rate = oos_result["oos_win_rate"]
                                    edge.oos_profit_factor = oos_result["oos_profit_factor"]
                                    edge.oos_expectancy_pips = oos_result["oos_expectancy_pips"]
                                    edge.oos_total_trades = oos_result["oos_total_trades"]
                                    edge.oos_sharpe_ratio = oos_result["oos_sharpe_ratio"]
                                    edge.validated = True
                                    state.edges_validated += 1
                                    logger.info(
                                        "VALIDATED edge: %s %s %s WR=%.1f%% PF=%.2f "
                                        "OOS_WR=%.1f%% OOS_PF=%.2f",
                                        pair, intv, strat, edge.win_rate,
                                        edge.profit_factor, edge.oos_win_rate,
                                        edge.oos_profit_factor,
                                    )

                            state.edges.append(edge)
                            state.edges_found += 1
                            existing_keys.add(key)

                            # Save incrementally every 10 edges
                            if state.edges_found % 10 == 0:
                                save_edges(state.edges)

                    except Exception as e:
                        logger.debug("Backtest error %s %s %s: %s", pair, strat, intv, e)

                    state.combos_tested += 1

                # Track completion for resume
                state.last_completed_pair = pair
                state.last_completed_interval = intv
                state.last_completed_strategy = strat
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
    continuous: bool = True,
    restart_delay: int = 300,
    validate: bool = True,
) -> bool:
    """Start discovery in a background thread. Returns True if started.

    Args:
        continuous: If True, automatically restart after each full sweep
        restart_delay: Seconds to wait between sweeps (default 5 min)
        validate: Whether to run walk-forward validation
    """
    global _bg_thread, _bg_stop_event, _bg_state

    with _bg_lock:
        if _bg_thread is not None and _bg_thread.is_alive():
            return False  # already running

        _bg_stop_event.clear()
        _bg_state = DiscoveryState()

        def _run():
            global _bg_state
            sweep_num = 0
            while not _bg_stop_event.is_set():
                sweep_num += 1
                try:
                    _bg_state = run_discovery(
                        pairs=pairs, strategies=strategies, intervals=intervals,
                        period=period, param_grid=param_grid, thresholds=thresholds,
                        state=DiscoveryState(), stop_event=_bg_stop_event,
                        validate=validate,
                    )
                except Exception as e:
                    logger.error("Discovery sweep %d failed: %s", sweep_num, e)

                if not continuous or _bg_stop_event.is_set():
                    break

                # Wait between sweeps, checking stop_event every second
                _bg_state.status = "waiting"
                save_state(_bg_state)
                for _ in range(restart_delay):
                    if _bg_stop_event.is_set():
                        break
                    time.sleep(1)

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
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved position")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip walk-forward validation")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    last_line_len = 0

    def _progress(state):
        nonlocal last_line_len
        pct = state.progress_pct
        eta = ""
        if state.combos_tested > 0 and pct > 0:
            remaining = state.elapsed_seconds / pct * (100 - pct)
            eta = f" | ETA: {remaining/60:.0f}m"

        # Use \n for log-file compatibility instead of \r
        line = (
            f"[{pct:5.1f}%] {state.current_pair} {state.current_interval} "
            f"{state.current_strategy} | "
            f"{state.edges_found} edges ({state.edges_validated} validated) | "
            f"{state.combos_tested}/{state.total_combos}{eta}"
        )

        if os.isatty(1):
            # Terminal: use \r for clean output
            padding = max(0, last_line_len - len(line))
            print(f"\r{line}{' ' * padding}", end="", flush=True)
            last_line_len = len(line)
        else:
            # File/pipe: use newlines
            print(line, flush=True)

    print("Starting Forex Edge Discovery...")
    print(f"Pairs: {args.pairs or 'all'}")
    print(f"Strategies: {args.strategies or 'all'}")
    print(f"Intervals: {args.intervals or ALL_INTERVALS}")
    print(f"Period: {args.period}")
    print(f"Walk-forward validation: {'OFF' if args.no_validate else 'ON'}")
    print(f"Resume: {'YES' if args.resume else 'NO'}")
    print()

    state = run_discovery(
        pairs=args.pairs,
        strategies=args.strategies,
        intervals=args.intervals,
        period=args.period,
        on_progress=_progress,
        validate=not args.no_validate,
        resume=args.resume,
    )

    print(f"\n\nDiscovery complete!")
    print(f"Tested: {state.combos_tested} combos in {state.elapsed_seconds:.0f}s")
    print(f"Edges found: {state.edges_found} ({state.edges_validated} validated)")

    if state.edges:
        validated = [e for e in state.edges if e.validated]
        unvalidated = [e for e in state.edges if not e.validated]

        if validated:
            print(f"\nTop 10 VALIDATED edges:")
            print(f"{'Pair':<10} {'TF':<4} {'Strategy':<16} {'Trades':>6} {'WR':>6} "
                  f"{'PF':>6} {'OOS_WR':>7} {'OOS_PF':>7} {'Score':>7}")
            print("-" * 80)
            for e in sorted(validated, key=lambda x: x.score, reverse=True)[:10]:
                print(f"{e.pair:<10} {e.interval:<4} {e.strategy:<16} "
                      f"{e.total_trades:>6} {e.win_rate:>5.1f}% "
                      f"{e.profit_factor:>6.2f} {e.oos_win_rate:>6.1f}% "
                      f"{e.oos_profit_factor:>7.2f} {e.score:>7.1f}")

        if unvalidated:
            print(f"\nTop 5 unvalidated edges (use with caution):")
            print(f"{'Pair':<10} {'TF':<4} {'Strategy':<16} {'Trades':>6} {'WR':>6} "
                  f"{'Pips':>8} {'PF':>6} {'Score':>7}")
            print("-" * 70)
            for e in sorted(unvalidated, key=lambda x: x.score, reverse=True)[:5]:
                print(f"{e.pair:<10} {e.interval:<4} {e.strategy:<16} "
                      f"{e.total_trades:>6} {e.win_rate:>5.1f}% "
                      f"{e.total_pips:>+7.1f} {e.profit_factor:>6.2f} "
                      f"{e.score:>7.1f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

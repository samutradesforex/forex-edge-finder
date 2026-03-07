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
- Network retry with exponential backoff
- Auto-start on app load (no button click needed)
- Crash recovery with automatic thread restart
- File logging for unattended operation
- Thread health monitoring
"""

import json
import logging
import os
import time
import threading
import traceback
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

# ── File logging for unattended operation ─────────────────────────────────
_log_configured = False

def _setup_file_logging():
    """Configure rotating file logging so discovery runs are traceable unattended.

    Uses RotatingFileHandler to prevent discovery.log from growing unboundedly.
    Max 5 MB per file, keeps 3 backup files (20 MB total max).
    """
    global _log_configured
    if _log_configured:
        return
    _log_configured = True
    from logging.handlers import RotatingFileHandler
    log_path = Path("discovery.log")
    handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, mode="a",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    # Only add if no file handler already exists
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(handler)
        root.setLevel(logging.INFO)

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
    "min_trades": 20,
    "min_win_rate": 45.0,
    "min_profit_factor": 1.3,
    "min_expectancy_pips": 2.0,
    "min_sharpe": 0.5,
    "min_payoff_ratio": 0.8,  # Min avg_win/avg_loss (RR)
    "max_drawdown_pct": 80.0,  # Reject edges with extreme drawdown
}

# Walk-forward validation settings
WALK_FORWARD_SPLIT = 0.7  # 70% in-sample, 30% out-of-sample
WALK_FORWARD_DECAY = 0.7  # OOS must retain at least 70% of in-sample performance

# Edge lifecycle
EDGE_MAX_AGE_DAYS = 90  # Edges older than this are marked stale
MONTE_CARLO_RUNS = 100  # Permutation test iterations


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
    payoff_ratio: float = 0.0  # Risk:Reward (avg win / avg loss)
    max_drawdown_pct: float = 0.0
    discovered_at: str = ""
    period: str = "6mo"
    # Walk-forward validation fields
    oos_win_rate: float = 0.0
    oos_profit_factor: float = 0.0
    oos_expectancy_pips: float = 0.0
    oos_total_trades: int = 0
    oos_sharpe_ratio: float = 0.0
    oos_folds_passed: int = 0
    oos_folds_total: int = 0
    validated: bool = False
    mc_pvalue: float = 1.0  # Monte Carlo p-value (lower = more significant)
    confidence_grade: str = "D"  # A/B/C/D confidence grade

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


_file_lock = threading.Lock()


def _atomic_write(path: Path, content: str):
    """Write file atomically via temp file + rename to prevent corruption."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.replace(path)  # atomic on POSIX


MAX_EDGES = 5000  # Retention limit — prune lowest-scoring edges beyond this


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
    # Remove stale edges (older than EDGE_MAX_AGE_DAYS)
    before_stale = len(unique)
    unique = [e for e in unique if not is_edge_stale(e)]
    stale_removed = before_stale - len(unique)
    if stale_removed:
        logger.info("Removed %d stale edges (older than %d days)",
                     stale_removed, EDGE_MAX_AGE_DAYS)

    # Retention policy: keep top MAX_EDGES by score to prevent unbounded growth
    if len(unique) > MAX_EDGES:
        unique.sort(key=lambda e: e.score, reverse=True)
        pruned = len(unique) - MAX_EDGES
        unique = unique[:MAX_EDGES]
        logger.info("Pruned %d low-scoring edges (retention limit: %d)", pruned, MAX_EDGES)
    data = [e.to_dict() for e in unique]
    with _file_lock:
        _atomic_write(path, json.dumps(data, indent=2, default=str))


def load_edges(filename: str = "edges.json") -> List[DiscoveredEdge]:
    path = RESULTS_DIR / filename
    if not path.exists():
        return []
    try:
        with _file_lock:
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
    with _file_lock:
        _atomic_write(path, json.dumps(d, indent=2))


def load_state(filename: str = "state.json") -> DiscoveryState:
    path = RESULTS_DIR / filename
    if not path.exists():
        return DiscoveryState()
    try:
        with _file_lock:
            d = json.loads(path.read_text())
        return DiscoveryState(**{k: v for k, v in d.items()
                                 if k != "edges" and k in {
                                     f.name for f in DiscoveryState.__dataclass_fields__.values()
                                 }})
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("Failed to load state: %s", e)
        return DiscoveryState()


def _clamp_float(v: float, lo: float = -99.9, hi: float = 99.9) -> float:
    """Clamp a float, converting inf/NaN to safe values for JSON serialization."""
    if v != v:  # NaN
        return 0.0
    if v == float("inf"):
        return hi
    if v == float("-inf"):
        return lo
    return max(lo, min(hi, v))


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
    # Check payoff ratio (RR) — reject edges with terrible reward relative to risk
    min_rr = t.get("min_payoff_ratio", 0.8)
    pr = bt.payoff_ratio
    # Handle inf (no losses = infinite payoff, which is fine)
    if pr != float("inf") and pr != float("-inf") and pr == pr:  # not NaN
        if pr < min_rr:
            return False
    # Reject edges with extreme drawdown relative to profits
    max_dd_pct = t.get("max_drawdown_pct", 80.0)
    if bt.max_drawdown_pct > max_dd_pct:
        return False
    return True


def compute_edge_score(bt: BacktestResult) -> float:
    """Compute a composite score for ranking discovered edges.

    Factors: expectancy, profit factor, Sharpe, win rate, recovery factor,
    drawdown penalty, and sample size bonus.
    """
    pf = min(bt.profit_factor, 5.0) if bt.profit_factor != float("inf") else 5.0
    rf = 0.0
    if bt.max_drawdown_pips > 0:
        rf = min(bt.total_pips / bt.max_drawdown_pips, 10.0)

    # Payoff ratio (RR) — cap at 5.0 for normalization, handle inf
    pr = bt.payoff_ratio
    if pr != pr or pr == float("inf") or pr == float("-inf"):
        pr = 5.0 if (bt.avg_win_pips > 0 and bt.avg_loss_pips == 0) else 0.0
    pr = min(pr, 5.0)

    norm_exp = min(max(bt.expectancy_pips, -10), 10) * 5
    norm_pf = pf * 10
    norm_sharpe = min(max(bt.sharpe_ratio, -2), 5) * 10
    norm_wr = bt.win_rate
    norm_rf = rf * 5
    norm_rr = pr * 10  # RR of 2.0 → 20 points

    # Drawdown penalty: penalize edges with >50% drawdown
    dd_penalty = 1.0
    if bt.max_drawdown_pct > 50:
        dd_penalty = max(0.5, 1.0 - (bt.max_drawdown_pct - 50) / 100)

    # Sample size bonus: more trades = more confidence (log scale)
    # 20 trades = 1.0x, 50 trades = 1.13x, 100 trades = 1.23x
    import math
    sample_bonus = min(1.3, math.log(max(bt.total_trades, 1)) / math.log(20))

    score = (
        norm_exp * 0.15 +
        norm_pf * 0.15 +
        norm_sharpe * 0.15 +
        norm_wr * 0.15 +
        norm_rf * 0.10 +
        norm_rr * 0.15 +   # RR contributes 15%
        (100 - bt.max_drawdown_pct) * 0.15  # Low drawdown bonus
    ) * dd_penalty * sample_bonus
    return round(score, 2)


# ── Walk-forward validation ──────────────────────────────────────────────

def _run_oos_backtest(df: pd.DataFrame, df_oos: pd.DataFrame,
                      pair: str, strategy: str, interval: str,
                      params: dict, indicator_warmup: int = 60,
                      ) -> Optional[BacktestResult]:
    """Run a single OOS backtest with proper indicator warm-up."""
    try:
        pip_size = get_pip_size(pair)
        # Find warmup start in the full df (handle duplicate indices)
        oos_start_loc = df.index.get_loc(df_oos.index[0])
        if isinstance(oos_start_loc, slice):
            oos_start_loc = oos_start_loc.start or 0
        elif hasattr(oos_start_loc, '__len__'):
            import numpy as np
            oos_start_loc = int(np.where(oos_start_loc)[0][0])
        warmup_start = max(0, oos_start_loc - indicator_warmup)
        df_warmup = df.iloc[warmup_start:oos_start_loc + len(df_oos)].copy()

        atr_full = calc_atr(df_warmup)
        rsi_full = calc_rsi(df_warmup)
        ema_fast_full = calc_ema(df_warmup["Close"], 21)
        ema_slow_full = calc_ema(df_warmup["Close"], 50)

        oos_precomputed = {
            "atr": atr_full.loc[df_oos.index],
            "rsi": rsi_full.loc[df_oos.index],
            "ema_fast": ema_fast_full.loc[df_oos.index],
            "ema_slow": ema_slow_full.loc[df_oos.index],
            "fvgs": find_fvgs(df_oos, pip_size=pip_size),
            "obs": find_order_blocks(df_oos, pip_size=pip_size),
        }

        return run_backtest(
            df_oos, pair, strategy=strategy, interval=interval,
            _precomputed=oos_precomputed, **params,
        )
    except Exception:
        return None


def walk_forward_validate(
    df: pd.DataFrame, pair: str, strategy: str, interval: str,
    params: dict, precomputed_is: dict = None,
    split: float = WALK_FORWARD_SPLIT,
    decay: float = WALK_FORWARD_DECAY,
    thresholds: dict = None,
    n_folds: int = 3,
) -> Optional[Dict]:
    """Run rolling walk-forward validation with multiple folds.

    Uses n_folds rolling windows instead of a single train/test split.
    Each fold trains on 70% and tests on the next 30%.
    An edge must pass ALL folds to be validated.

    Returns dict with averaged OOS metrics if validated, None if failed.
    """
    n = len(df)
    min_oos_candles = 80  # Require meaningful OOS period

    # Fall back to single-fold for smaller datasets
    if n < 400:
        n_folds = 1
        min_oos_candles = 40

    t = thresholds or EDGE_THRESHOLDS

    # Generate rolling fold boundaries
    # Each fold shifts forward by step_size, with OOS being the last 30% of remaining data
    step_size = max(1, n // (n_folds * 2))  # Reasonable step between folds
    folds_passed = 0
    oos_results = []

    for fold in range(n_folds):
        if n_folds == 1:
            split_idx = int(n * split)
            oos_start = split_idx
            oos_end = n
        else:
            # Anchored walk-forward: train start advances, OOS is always the trailing portion
            train_start = fold * step_size
            remaining = n - train_start
            split_idx = train_start + int(remaining * split)
            oos_start = split_idx
            oos_end = n  # Always test through the end for maximum OOS data

        if split_idx < 50 or (oos_end - oos_start) < min_oos_candles:
            continue

        df_oos = df.iloc[oos_start:oos_end].copy()
        if len(df_oos) < min_oos_candles:
            continue

        bt_oos = _run_oos_backtest(df, df_oos, pair, strategy, interval, params)
        if bt_oos is None or bt_oos.total_trades < 5:
            continue

        # Check OOS performance with stricter decay
        oos_ok = (
            bt_oos.win_rate >= t["min_win_rate"] * decay and
            bt_oos.profit_factor >= max(t["min_profit_factor"] * decay, 1.05) and
            bt_oos.expectancy_pips > 0 and
            bt_oos.sharpe_ratio > 0
        )

        if oos_ok:
            folds_passed += 1
            oos_results.append(bt_oos)

    # Must pass majority of folds (all for small n_folds)
    min_folds_required = max(1, n_folds if n_folds <= 2 else n_folds - 1)
    if folds_passed < min_folds_required or not oos_results:
        return None

    # Average OOS metrics across all passing folds
    avg_wr = np.mean([r.win_rate for r in oos_results])
    avg_pf = np.mean([min(r.profit_factor, 99.9) for r in oos_results])
    avg_exp = np.mean([r.expectancy_pips for r in oos_results])
    total_trades = sum(r.total_trades for r in oos_results)
    avg_sharpe = np.mean([_clamp_float(r.sharpe_ratio, -10, 99.9) for r in oos_results])

    return {
        "oos_win_rate": round(avg_wr, 1),
        "oos_profit_factor": round(min(avg_pf, 99.9), 2),
        "oos_expectancy_pips": round(avg_exp, 1),
        "oos_total_trades": total_trades,
        "oos_sharpe_ratio": round(avg_sharpe, 2),
        "oos_folds_passed": folds_passed,
        "oos_folds_total": n_folds,
        "validated": True,
    }


# ── Monte Carlo permutation test ──────────────────────────────────────────

def monte_carlo_test(trades_pnl: List[float], observed_expectancy: float,
                     n_runs: int = None) -> float:
    """Run Monte Carlo permutation test to check if edge is statistically significant.

    Tests null hypothesis: "the observed expectancy could have arisen by chance."
    Randomly flips the sign of each trade's P&L n_runs times (simulating a
    world where the strategy has no directional edge — wins and losses are
    equally likely). Computes the fraction of random sign-flips that produce
    an equal or greater mean P&L than observed.

    Returns p-value (0.0 to 1.0). Values < 0.05 are statistically significant.
    """
    if n_runs is None:
        n_runs = MONTE_CARLO_RUNS
    if len(trades_pnl) < 10:
        return 1.0  # Not enough data

    pnl_arr = np.array(trades_pnl)
    n_trades = len(pnl_arr)

    # Seed from the data for reproducibility without global state
    seed = int(abs(np.sum(pnl_arr) * 1000 + n_trades * 7)) % (2**31)
    rng = np.random.default_rng(seed)

    # Vectorized sign-flip permutation test:
    # For each run, randomly flip the sign of each trade (+1 or -1)
    # This simulates a strategy with no directional edge
    signs = rng.choice([-1, 1], size=(n_runs, n_trades))
    permuted = pnl_arr * signs  # shape: (n_runs, n_trades)
    permuted_means = np.mean(permuted, axis=1)

    # p-value: fraction of permutations with mean >= observed
    beat_count = np.sum(permuted_means >= observed_expectancy)
    return float(beat_count / n_runs)


# ── Edge lifecycle ────────────────────────────────────────────────────────

def compute_edge_confidence(edge) -> str:
    """Compute a confidence grade for an edge based on validation strength.

    Returns: 'A' (highest), 'B', 'C', or 'D' (lowest).
    """
    score = 0

    # Validation status
    if getattr(edge, 'validated', False):
        score += 2
        # Multi-fold validation bonus
        folds_passed = getattr(edge, 'oos_folds_passed', 1)
        folds_total = getattr(edge, 'oos_folds_total', 1)
        if folds_passed >= folds_total:
            score += 1  # Passed all folds

    # OOS performance quality
    if getattr(edge, 'oos_profit_factor', 0) >= 1.5:
        score += 1
    if getattr(edge, 'oos_win_rate', 0) >= 50:
        score += 1

    # Monte Carlo significance
    mc_pvalue = getattr(edge, 'mc_pvalue', 1.0)
    if mc_pvalue < 0.05:
        score += 2
    elif mc_pvalue < 0.10:
        score += 1

    # Sample size
    if edge.total_trades >= 50:
        score += 1
    elif edge.total_trades >= 30:
        score += 0.5

    # Good RR (payoff ratio)
    rr = getattr(edge, 'payoff_ratio', 0)
    if rr and rr >= 1.5:
        score += 1
    elif rr and rr >= 1.0:
        score += 0.5

    if score >= 6:
        return 'A'
    elif score >= 4:
        return 'B'
    elif score >= 2:
        return 'C'
    return 'D'


def is_edge_stale(edge, max_age_days: int = None) -> bool:
    """Check if an edge is stale (too old to be reliable)."""
    if max_age_days is None:
        max_age_days = EDGE_MAX_AGE_DAYS
    if not edge.discovered_at:
        return True  # Undated edges are treated as stale
    try:
        discovered = datetime.fromisoformat(edge.discovered_at)
        age_days = (datetime.now() - discovered).days
        return age_days > max_age_days
    except (ValueError, TypeError):
        return False


def get_edge_age_days(edge) -> int:
    """Get edge age in days. Returns -1 if unknown."""
    if not edge.discovered_at:
        return -1
    try:
        discovered = datetime.fromisoformat(edge.discovered_at)
        return (datetime.now() - discovered).days
    except (ValueError, TypeError):
        return -1


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


# ── Market hours filtering ─────────────────────────────────────────────────

# Low-liquidity hours to exclude (UTC). These typically have wider spreads
# and unreliable price action.
LOW_LIQUIDITY_HOURS_UTC = {21, 22, 23, 0}  # Late NY to early Sydney


def filter_market_hours(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Remove low-liquidity hours from intraday data.

    Only applies to 1h and 4h data. Daily data is unaffected.
    Filters out late NY / early Sydney sessions (21:00-00:59 UTC)
    where spreads widen and price action is unreliable.
    """
    if interval not in ("1h", "4h"):
        return df
    if not hasattr(df.index, 'hour'):
        return df
    mask = ~df.index.hour.isin(LOW_LIQUIDITY_HOURS_UTC)
    filtered = df[mask]
    if len(filtered) < 50:
        return df  # Don't filter if it would leave too little data
    return filtered


# ── Core discovery loop ───────────────────────────────────────────────────

def _fetch_data_with_retry(pair: str, period: str, interval: str,
                           use_max: bool = False,
                           max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch data for a pair/interval with exponential backoff retry.

    Retries on network errors and rate limits, which are critical for
    24/7 unattended operation where yfinance may temporarily fail.
    """
    last_error = None
    for attempt in range(max_retries + 1):
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
                logger.warning("Insufficient data for %s %s: %d rows",
                               pair, interval, len(df))
                return None
            return df
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                logger.warning(
                    "Fetch attempt %d/%d failed for %s %s: %s. Retrying in %ds...",
                    attempt + 1, max_retries + 1, pair, interval, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "All %d fetch attempts failed for %s %s: %s",
                    max_retries + 1, pair, interval, last_error,
                )
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
    # Load edges from disk unless caller already populated them (avoids double-load)
    if not state.edges:
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

            df = _fetch_data_with_retry(pair, period, intv, use_max=True)
            if df is None:
                for strat in strategies:
                    combos = _get_strategy_combos(strat, param_grid)
                    state.combos_tested += len(combos)
                if not found_resume_point:
                    found_resume_point = True  # resume point pair/intv matched but no data
                continue

            # Filter out low-liquidity hours for cleaner signals
            df = filter_market_hours(df, intv)

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
                                sharpe_ratio=round(_clamp_float(bt.sharpe_ratio, -10, 99.9), 2),
                                max_drawdown_pips=round(bt.max_drawdown_pips, 1),
                                score=compute_edge_score(bt),
                                payoff_ratio=round(_clamp_float(bt.payoff_ratio, 0, 99.9), 2),
                                max_drawdown_pct=round(bt.max_drawdown_pct, 1),
                                discovered_at=datetime.now().isoformat(),
                                period=period,
                            )

                            # Monte Carlo significance test
                            trade_pnls = [t.pnl_pips for t in bt.trades]
                            mc_pval = monte_carlo_test(
                                trade_pnls, bt.expectancy_pips,
                                n_runs=MONTE_CARLO_RUNS,
                            )
                            edge.mc_pvalue = round(mc_pval, 4)

                            # Walk-forward validation (rolling multi-fold)
                            if validate and len(df) > len(df_is) + 50:
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
                                    edge.oos_folds_passed = oos_result.get("oos_folds_passed", 1)
                                    edge.oos_folds_total = oos_result.get("oos_folds_total", 1)
                                    edge.validated = True
                                    state.edges_validated += 1
                                    logger.info(
                                        "VALIDATED edge: %s %s %s WR=%.1f%% PF=%.2f "
                                        "OOS_WR=%.1f%% OOS_PF=%.2f MC_p=%.3f "
                                        "folds=%d/%d",
                                        pair, intv, strat, edge.win_rate,
                                        edge.profit_factor, edge.oos_win_rate,
                                        edge.oos_profit_factor, mc_pval,
                                        edge.oos_folds_passed,
                                        edge.oos_folds_total,
                                    )

                            # Compute confidence grade
                            edge.confidence_grade = compute_edge_confidence(edge)

                            state.edges.append(edge)
                            state.edges_found += 1
                            existing_keys.add(key)

                            # Save after every edge to prevent loss on crash
                            save_edges(state.edges)

                    except Exception as e:
                        _bt_error_count = getattr(state, '_bt_error_count', 0) + 1
                        state._bt_error_count = _bt_error_count
                        # Log first 5 errors per strategy at WARNING, rest at DEBUG
                        if _bt_error_count <= 5:
                            logger.warning("Backtest error %s %s %s: %s",
                                           pair, strat, intv, e)
                        elif _bt_error_count == 6:
                            logger.warning(
                                "Suppressing further backtest errors (5+ logged). "
                                "Check strategy %s for bugs.", strat)

                    state.combos_tested += 1

                # Reset per-strategy error count
                state._bt_error_count = 0

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
_bg_crash_count = 0
_bg_last_heartbeat = 0.0
_bg_sweep_count = 0
_bg_total_edges_lifetime = 0


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

    Features:
    - Crash recovery: auto-restarts on unhandled exceptions (up to 10 times)
    - Heartbeat: updates timestamp so UI can detect zombie threads
    - State accumulation: edges persist correctly across sweeps
    - File logging: all activity logged to discovery.log
    """
    global _bg_thread, _bg_stop_event, _bg_state
    global _bg_crash_count, _bg_last_heartbeat, _bg_sweep_count

    # Ensure file logging is active for unattended operation
    _setup_file_logging()

    with _bg_lock:
        if _bg_thread is not None and _bg_thread.is_alive():
            return False  # already running

        _bg_stop_event.clear()
        _bg_crash_count = 0
        _bg_last_heartbeat = time.time()

        # Load existing state if available (supports resume across app restarts)
        existing_edges = load_edges()
        _bg_state = DiscoveryState()
        _bg_state.edges = existing_edges
        _bg_state.edges_found = len(existing_edges)
        _bg_state.edges_validated = sum(1 for e in existing_edges if e.validated)

        def _run():
            global _bg_state, _bg_crash_count, _bg_last_heartbeat
            global _bg_sweep_count, _bg_total_edges_lifetime
            max_crashes = 10
            crash_backoff = 30  # seconds to wait after crash before retry

            logger.info(
                "Discovery background thread started. continuous=%s, "
                "restart_delay=%ds, validate=%s, existing_edges=%d",
                continuous, restart_delay, validate, len(_bg_state.edges),
            )

            while not _bg_stop_event.is_set():
                _bg_sweep_count += 1
                sweep_start = time.time()
                _bg_last_heartbeat = time.time()

                try:
                    logger.info(
                        "Starting sweep #%d (crashes so far: %d)",
                        _bg_sweep_count, _bg_crash_count,
                    )

                    # Create fresh state for this sweep but carry forward edges
                    sweep_state = DiscoveryState()
                    sweep_state.edges = load_edges()  # Always load latest from disk
                    sweep_state.edges_found = 0  # Count new edges this sweep
                    sweep_state.edges_validated = 0

                    _bg_state = run_discovery(
                        pairs=pairs, strategies=strategies, intervals=intervals,
                        period=period, param_grid=param_grid, thresholds=thresholds,
                        state=sweep_state, stop_event=_bg_stop_event,
                        validate=validate,
                    )

                    _bg_last_heartbeat = time.time()
                    _bg_total_edges_lifetime += _bg_state.edges_found
                    _bg_crash_count = 0  # Reset crash counter on successful sweep

                    elapsed = time.time() - sweep_start
                    logger.info(
                        "Sweep #%d complete in %.0fs. %d new edges (%d validated). "
                        "Total edges on disk: %d",
                        _bg_sweep_count, elapsed,
                        _bg_state.edges_found, _bg_state.edges_validated,
                        len(_bg_state.edges),
                    )

                except Exception as e:
                    _bg_crash_count += 1
                    _bg_last_heartbeat = time.time()
                    logger.error(
                        "Discovery sweep #%d CRASHED (crash #%d/%d): %s\n%s",
                        _bg_sweep_count, _bg_crash_count, max_crashes,
                        e, traceback.format_exc(),
                    )
                    _bg_state.error = f"Crash #{_bg_crash_count}: {e}"
                    save_state(_bg_state)

                    if _bg_crash_count >= max_crashes:
                        logger.critical(
                            "Discovery exceeded max crashes (%d). Stopping.",
                            max_crashes,
                        )
                        _bg_state.status = "crashed"
                        _bg_state.error = (
                            f"Stopped after {max_crashes} consecutive crashes. "
                            f"Last error: {e}"
                        )
                        save_state(_bg_state)
                        break

                    # Exponential backoff on crash (30s, 60s, 120s, ...)
                    wait = crash_backoff * (2 ** (_bg_crash_count - 1))
                    wait = min(wait, 600)  # Cap at 10 minutes
                    logger.info("Waiting %ds before restart...", wait)
                    for _ in range(int(wait)):
                        if _bg_stop_event.is_set():
                            break
                        time.sleep(1)
                        _bg_last_heartbeat = time.time()
                    continue  # Retry the sweep

                if not continuous or _bg_stop_event.is_set():
                    break

                # Wait between sweeps, checking stop_event every second
                _bg_state.status = "waiting"
                save_state(_bg_state)
                logger.info(
                    "Waiting %ds before next sweep...", restart_delay,
                )
                for i in range(restart_delay):
                    if _bg_stop_event.is_set():
                        break
                    time.sleep(1)
                    _bg_last_heartbeat = time.time()

            logger.info("Discovery background thread exiting.")

        _bg_thread = threading.Thread(target=_run, daemon=True, name="discovery-bg")
        _bg_thread.start()
        logger.info("Discovery background thread launched (thread=%s).", _bg_thread.name)
        return True


def stop_discovery_background():
    """Signal the background discovery to stop."""
    _bg_stop_event.set()
    logger.info("Discovery stop signal sent.")


def get_discovery_state() -> DiscoveryState:
    """Get current state of background discovery."""
    with _bg_lock:
        return _bg_state


def is_discovery_running() -> bool:
    """Check if discovery is currently running."""
    with _bg_lock:
        return _bg_thread is not None and _bg_thread.is_alive()


def get_discovery_health() -> Dict:
    """Get health info for the discovery thread. Used by dashboard for monitoring."""
    with _bg_lock:
        alive = _bg_thread is not None and _bg_thread.is_alive()
    return {
        "alive": alive,
        "crash_count": _bg_crash_count,
        "sweep_count": _bg_sweep_count,
        "total_edges_lifetime": _bg_total_edges_lifetime,
        "last_heartbeat": _bg_last_heartbeat,
        "seconds_since_heartbeat": time.time() - _bg_last_heartbeat if _bg_last_heartbeat > 0 else -1,
        "status": _bg_state.status,
        "error": _bg_state.error,
    }


def auto_start_discovery() -> bool:
    """Auto-start discovery if not already running. Called on app load.

    This is the key function that makes the system truly autonomous —
    discovery begins immediately when the app starts, no button click needed.
    Returns True if discovery was started, False if already running.
    """
    if is_discovery_running():
        return False
    logger.info("Auto-starting discovery engine...")
    return start_discovery_background(continuous=True)


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

"""Advanced backtesting engine for liquidity inducement strategies.

Features:
- Multiple exit strategies (fixed RR, trailing SL, break-even, partial TP)
- Spread simulation
- Max trades per day limit
- Consecutive loss protection
- Advanced metrics (Sharpe, Sortino, monthly breakdown, streaks, drawdown duration)
- Parameter optimization via grid search
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from itertools import product
from engine.liquidity import (
    InducementSignal,
    find_swing_points,
    find_multi_tf_swings,
    find_liquidity_levels,
    find_fvgs,
    find_order_blocks,
    detect_liquidity_sweeps,
    detect_inducement_traps,
    detect_stop_hunts,
    detect_ema_crossover,
    detect_rsi_reversal,
    detect_breakout,
    detect_fvg_entry,
    detect_ob_bounce,
    get_pip_size,
    calc_atr,
    calc_rsi,
    calc_ema,
)


@dataclass
class Trade:
    """A completed trade."""
    entry_datetime: pd.Timestamp
    exit_datetime: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl_pips: float
    result: str  # "win", "loss", "breakeven"
    signal_type: str
    swept_level: float
    confluence_score: int = 0
    confluence_factors: List[str] = field(default_factory=list)
    session: str = ""
    holding_candles: int = 0
    max_favorable_pips: float = 0.0
    max_adverse_pips: float = 0.0


@dataclass
class BacktestResult:
    """Full backtest results and metrics."""
    trades: List[Trade]
    pair_name: str = ""
    interval: str = ""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    win_rate: float = 0.0
    total_pips: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    max_drawdown_pips: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0  # in trades
    profit_factor: float = 0.0
    expectancy_pips: float = 0.0
    best_trade_pips: float = 0.0
    worst_trade_pips: float = 0.0
    long_wins: int = 0
    long_losses: int = 0
    short_wins: int = 0
    short_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_holding_candles: float = 0.0
    avg_confluence_score: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0
    payoff_ratio: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    monthly_pnl: Dict[str, float] = field(default_factory=dict)
    session_breakdown: Dict[str, Dict] = field(default_factory=dict)
    signal_type_breakdown: Dict[str, Dict] = field(default_factory=dict)
    confluence_breakdown: Dict[int, Dict] = field(default_factory=dict)
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    hourly_pnl: Dict[int, float] = field(default_factory=dict)

    def compute_metrics(self):
        if not self.trades:
            return

        self.total_trades = len(self.trades)
        self.wins = sum(1 for t in self.trades if t.result == "win")
        self.losses = sum(1 for t in self.trades if t.result == "loss")
        self.breakevens = sum(1 for t in self.trades if t.result == "breakeven")
        self.win_rate = (self.wins / self.total_trades * 100) if self.total_trades else 0

        pnls = [t.pnl_pips for t in self.trades]
        self.total_pips = sum(pnls)
        self.best_trade_pips = max(pnls) if pnls else 0
        self.worst_trade_pips = min(pnls) if pnls else 0

        win_pnls = [t.pnl_pips for t in self.trades if t.result == "win"]
        loss_pnls = [t.pnl_pips for t in self.trades if t.result == "loss"]
        self.avg_win_pips = np.mean(win_pnls) if win_pnls else 0
        self.avg_loss_pips = np.mean(loss_pnls) if loss_pnls else 0

        gross_profit = sum(win_pnls) if win_pnls else 0
        gross_loss = abs(sum(loss_pnls)) if loss_pnls else 0
        self.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        self.expectancy_pips = np.mean(pnls) if pnls else 0
        self.payoff_ratio = (self.avg_win_pips / abs(self.avg_loss_pips)) if self.avg_loss_pips != 0 else float("inf")

        # Direction breakdown
        self.long_wins = sum(1 for t in self.trades if t.result == "win" and t.direction == "long")
        self.long_losses = sum(1 for t in self.trades if t.result == "loss" and t.direction == "long")
        self.short_wins = sum(1 for t in self.trades if t.result == "win" and t.direction == "short")
        self.short_losses = sum(1 for t in self.trades if t.result == "loss" and t.direction == "short")

        # Consecutive wins/losses
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for t in self.trades:
            if t.result == "win":
                if streak > 0:
                    streak += 1
                else:
                    streak = 1
                max_win_streak = max(max_win_streak, streak)
            elif t.result == "loss":
                if streak < 0:
                    streak -= 1
                else:
                    streak = -1
                max_loss_streak = max(max_loss_streak, abs(streak))
            else:
                streak = 0
        self.max_consecutive_wins = max_win_streak
        self.max_consecutive_losses = max_loss_streak

        # Equity curve, drawdown, drawdown duration
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        curve = [0.0]
        dd_duration = 0
        max_dd_duration = 0
        for t in self.trades:
            equity += t.pnl_pips
            curve.append(equity)
            if equity > peak:
                peak = equity
                dd_duration = 0
            else:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            dd = peak - equity
            max_dd = max(max_dd, dd)
        self.equity_curve = curve
        self.max_drawdown_pips = max_dd
        self.max_drawdown_duration = max_dd_duration
        if peak > 0:
            self.max_drawdown_pct = (max_dd / peak) * 100
        self.recovery_factor = (self.total_pips / max_dd) if max_dd > 0 else float("inf")

        # Sharpe and Sortino (annualized assuming ~252 trading days)
        if len(pnls) > 1:
            pnl_arr = np.array(pnls)
            mean_pnl = np.mean(pnl_arr)
            std_pnl = np.std(pnl_arr, ddof=1)
            if std_pnl > 0:
                self.sharpe_ratio = (mean_pnl / std_pnl) * np.sqrt(252)
            downside = pnl_arr[pnl_arr < 0]
            if len(downside) > 0:
                downside_std = np.std(downside, ddof=1)
                if downside_std > 0:
                    self.sortino_ratio = (mean_pnl / downside_std) * np.sqrt(252)
        if max_dd > 0:
            self.calmar_ratio = self.total_pips / max_dd

        # Average holding time and confluence
        holdings = [t.holding_candles for t in self.trades]
        self.avg_holding_candles = np.mean(holdings) if holdings else 0
        confs = [t.confluence_score for t in self.trades]
        self.avg_confluence_score = np.mean(confs) if confs else 0

        # Monthly PnL
        for t in self.trades:
            month_key = t.entry_datetime.strftime("%Y-%m")
            self.monthly_pnl[month_key] = self.monthly_pnl.get(month_key, 0) + t.pnl_pips

        # Daily PnL (day of week)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for t in self.trades:
            day_key = day_names[t.entry_datetime.dayofweek]
            self.daily_pnl[day_key] = self.daily_pnl.get(day_key, 0) + t.pnl_pips

        # Hourly PnL
        for t in self.trades:
            hour = t.entry_datetime.hour
            self.hourly_pnl[hour] = self.hourly_pnl.get(hour, 0) + t.pnl_pips

        # Session breakdown
        for t in self.trades:
            s = t.session or "unknown"
            if s not in self.session_breakdown:
                self.session_breakdown[s] = {"trades": 0, "wins": 0, "pips": 0}
            self.session_breakdown[s]["trades"] += 1
            if t.result == "win":
                self.session_breakdown[s]["wins"] += 1
            self.session_breakdown[s]["pips"] += t.pnl_pips

        # Signal type breakdown
        for t in self.trades:
            st = t.signal_type
            if st not in self.signal_type_breakdown:
                self.signal_type_breakdown[st] = {"trades": 0, "wins": 0, "pips": 0}
            self.signal_type_breakdown[st]["trades"] += 1
            if t.result == "win":
                self.signal_type_breakdown[st]["wins"] += 1
            self.signal_type_breakdown[st]["pips"] += t.pnl_pips

        # Confluence score breakdown
        for t in self.trades:
            cs = t.confluence_score
            if cs not in self.confluence_breakdown:
                self.confluence_breakdown[cs] = {"trades": 0, "wins": 0, "pips": 0}
            self.confluence_breakdown[cs]["trades"] += 1
            if t.result == "win":
                self.confluence_breakdown[cs]["wins"] += 1
            self.confluence_breakdown[cs]["pips"] += t.pnl_pips


# ── Trade simulation ───────────────────────────────────────────────────────

def simulate_trade(signal: InducementSignal, df: pd.DataFrame,
                   pip_size: float, spread_pips: float = 0.0,
                   trailing_sl: bool = False, trailing_activation_rr: float = 1.0,
                   break_even_rr: float = 0.0,
                   partial_tp_rr: float = 0.0, partial_tp_pct: float = 0.5,
                   _high_arr: np.ndarray = None, _low_arr: np.ndarray = None,
                   ) -> Optional[Trade]:
    """Simulate a single trade with advanced exit logic.

    Args:
        signal: The entry signal
        df: Price data
        pip_size: Pip size for the pair
        spread_pips: Spread cost in pips
        trailing_sl: Enable trailing stop loss
        trailing_activation_rr: RR multiple to activate trailing
        break_even_rr: RR multiple to move SL to break-even (0 = disabled)
        partial_tp_rr: RR multiple for partial take profit (0 = disabled)
        partial_tp_pct: Percentage of position to close at partial TP
        _high_arr: Pre-extracted High values array (performance optimization)
        _low_arr: Pre-extracted Low values array (performance optimization)
    """
    spread = spread_pips * pip_size
    entry_price = signal.entry_price
    sl = signal.stop_loss
    tp = signal.take_profit

    # Adjust entry for spread
    if signal.direction == "long":
        entry_price += spread / 2
    else:
        entry_price -= spread / 2

    initial_risk = abs(entry_price - sl)
    if initial_risk == 0:
        return None

    current_sl = sl
    max_favorable = 0.0
    max_adverse = 0.0
    partial_closed = False
    effective_pnl_multiplier = 1.0

    # Use pre-extracted arrays for speed (avoid .iloc per candle)
    high_arr = _high_arr if _high_arr is not None else df["High"].values
    low_arr = _low_arr if _low_arr is not None else df["Low"].values
    n = len(df)

    for j in range(signal.entry_index + 1, n):
        candle_high = high_arr[j]
        candle_low = low_arr[j]

        if signal.direction == "long":
            favorable = candle_high - entry_price
            adverse = entry_price - candle_low
            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

            # Break-even logic
            if break_even_rr > 0 and favorable >= initial_risk * break_even_rr:
                current_sl = max(current_sl, entry_price + pip_size)

            # Trailing stop logic
            if trailing_sl and favorable >= initial_risk * trailing_activation_rr:
                trail_level = candle_high - initial_risk
                current_sl = max(current_sl, trail_level)

            # Partial TP
            if partial_tp_rr > 0 and not partial_closed:
                if favorable >= initial_risk * partial_tp_rr:
                    partial_closed = True
                    effective_pnl_multiplier = 1.0 - partial_tp_pct

            # Check SL
            if candle_low <= current_sl:
                pnl = ((current_sl - entry_price) / pip_size) * effective_pnl_multiplier
                if partial_closed:
                    partial_pnl = (initial_risk * partial_tp_rr / pip_size) * partial_tp_pct
                    pnl += partial_pnl
                pnl -= spread_pips
                result = "win" if pnl > 0 else ("breakeven" if pnl == 0 else "loss")
                return Trade(
                    entry_datetime=signal.entry_datetime,
                    exit_datetime=df.index[j],
                    direction="long", entry_price=signal.entry_price,
                    exit_price=current_sl, stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    pnl_pips=round(pnl, 1), result=result,
                    signal_type=signal.signal_type,
                    swept_level=signal.swept_level,
                    confluence_score=signal.confluence_score,
                    confluence_factors=signal.confluence_factors,
                    session=signal.session,
                    holding_candles=j - signal.entry_index,
                    max_favorable_pips=round(max_favorable / pip_size, 1),
                    max_adverse_pips=round(max_adverse / pip_size, 1),
                )

            # Check TP
            if candle_high >= tp:
                pnl = ((tp - entry_price) / pip_size) * effective_pnl_multiplier
                if partial_closed:
                    partial_pnl = (initial_risk * partial_tp_rr / pip_size) * partial_tp_pct
                    pnl += partial_pnl
                pnl -= spread_pips
                return Trade(
                    entry_datetime=signal.entry_datetime,
                    exit_datetime=df.index[j],
                    direction="long", entry_price=signal.entry_price,
                    exit_price=tp, stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    pnl_pips=round(pnl, 1), result="win",
                    signal_type=signal.signal_type,
                    swept_level=signal.swept_level,
                    confluence_score=signal.confluence_score,
                    confluence_factors=signal.confluence_factors,
                    session=signal.session,
                    holding_candles=j - signal.entry_index,
                    max_favorable_pips=round(max_favorable / pip_size, 1),
                    max_adverse_pips=round(max_adverse / pip_size, 1),
                )

        elif signal.direction == "short":
            favorable = entry_price - candle_low
            adverse = candle_high - entry_price
            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

            # Break-even
            if break_even_rr > 0 and favorable >= initial_risk * break_even_rr:
                current_sl = min(current_sl, entry_price - pip_size)

            # Trailing stop
            if trailing_sl and favorable >= initial_risk * trailing_activation_rr:
                trail_level = candle_low + initial_risk
                current_sl = min(current_sl, trail_level)

            # Partial TP
            if partial_tp_rr > 0 and not partial_closed:
                if favorable >= initial_risk * partial_tp_rr:
                    partial_closed = True
                    effective_pnl_multiplier = 1.0 - partial_tp_pct

            # Check SL
            if candle_high >= current_sl:
                pnl = ((entry_price - current_sl) / pip_size) * effective_pnl_multiplier
                if partial_closed:
                    partial_pnl = (initial_risk * partial_tp_rr / pip_size) * partial_tp_pct
                    pnl += partial_pnl
                pnl -= spread_pips
                result = "win" if pnl > 0 else ("breakeven" if pnl == 0 else "loss")
                return Trade(
                    entry_datetime=signal.entry_datetime,
                    exit_datetime=df.index[j],
                    direction="short", entry_price=signal.entry_price,
                    exit_price=current_sl, stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    pnl_pips=round(pnl, 1), result=result,
                    signal_type=signal.signal_type,
                    swept_level=signal.swept_level,
                    confluence_score=signal.confluence_score,
                    confluence_factors=signal.confluence_factors,
                    session=signal.session,
                    holding_candles=j - signal.entry_index,
                    max_favorable_pips=round(max_favorable / pip_size, 1),
                    max_adverse_pips=round(max_adverse / pip_size, 1),
                )

            # Check TP
            if candle_low <= tp:
                pnl = ((entry_price - tp) / pip_size) * effective_pnl_multiplier
                if partial_closed:
                    partial_pnl = (initial_risk * partial_tp_rr / pip_size) * partial_tp_pct
                    pnl += partial_pnl
                pnl -= spread_pips
                return Trade(
                    entry_datetime=signal.entry_datetime,
                    exit_datetime=df.index[j],
                    direction="short", entry_price=signal.entry_price,
                    exit_price=tp, stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    pnl_pips=round(pnl, 1), result="win",
                    signal_type=signal.signal_type,
                    swept_level=signal.swept_level,
                    confluence_score=signal.confluence_score,
                    confluence_factors=signal.confluence_factors,
                    session=signal.session,
                    holding_candles=j - signal.entry_index,
                    max_favorable_pips=round(max_favorable / pip_size, 1),
                    max_adverse_pips=round(max_adverse / pip_size, 1),
                )

    return None


# ── Main backtest runner ───────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    pair_name: str,
    swing_lookback: int = 5,
    cluster_pips: float = 10.0,
    min_wick_pips: float = 3.0,
    strategy: str = "all",
    rr_ratio: float = 2.0,
    spread_pips: float = 1.0,
    require_displacement: bool = False,
    min_confluence: int = 0,
    session_filter: str = "all",
    trailing_sl: bool = False,
    trailing_activation_rr: float = 1.0,
    break_even_rr: float = 0.0,
    partial_tp_rr: float = 0.0,
    partial_tp_pct: float = 0.5,
    max_trades_per_day: int = 0,
    max_consecutive_losses: int = 0,
    use_multi_tf_swings: bool = False,
    interval: str = "",
) -> BacktestResult:
    """Run a full backtest with all advanced features."""
    pip_size = get_pip_size(pair_name)

    # Pre-compute indicators once
    atr = calc_atr(df)
    rsi = calc_rsi(df)
    ema_fast = calc_ema(df["Close"], 21)
    ema_slow = calc_ema(df["Close"], 50)
    fvgs = find_fvgs(df, pip_size=pip_size)
    obs = find_order_blocks(df, pip_size=pip_size)

    # Detect structure
    if use_multi_tf_swings:
        swings = find_multi_tf_swings(df)
    else:
        swings = find_swing_points(df, lookback=swing_lookback)
    levels = find_liquidity_levels(swings, cluster_pips=cluster_pips, pip_size=pip_size)

    # Shared kwargs for signal detection
    detect_kwargs = dict(
        pip_size=pip_size, atr=atr, fvgs=fvgs, obs=obs, rsi=rsi,
        ema_fast=ema_fast, ema_slow=ema_slow,
        require_displacement=require_displacement,
        min_confluence=min_confluence,
        session_filter=session_filter,
        rr_ratio=rr_ratio,
    )

    # Generate signals
    signals = []

    # SMC / Liquidity strategies
    if strategy in ("sweeps", "both", "all", "smc"):
        signals.extend(detect_liquidity_sweeps(
            df, levels, min_wick_pips=min_wick_pips, **detect_kwargs))
    if strategy in ("inducement", "both", "all", "smc"):
        signals.extend(detect_inducement_traps(df, swings, **detect_kwargs))
    if strategy in ("stop_hunts", "all", "smc"):
        sh_kwargs = {k: v for k, v in detect_kwargs.items()
                     if k != "require_displacement"}
        signals.extend(detect_stop_hunts(df, levels, **sh_kwargs))

    # Common strategies
    if strategy in ("ema_crossover", "all"):
        signals.extend(detect_ema_crossover(df, **detect_kwargs))
    if strategy in ("rsi_reversal", "all"):
        signals.extend(detect_rsi_reversal(df, **detect_kwargs))
    if strategy in ("breakout", "all"):
        signals.extend(detect_breakout(df, swings, **detect_kwargs))
    if strategy in ("fvg_entry", "all"):
        signals.extend(detect_fvg_entry(df, **detect_kwargs))
    if strategy in ("ob_bounce", "all"):
        signals.extend(detect_ob_bounce(df, **detect_kwargs))

    # Sort by time
    signals.sort(key=lambda s: s.entry_index)

    # Pre-extract arrays for simulate_trade performance
    _high_arr = df["High"].values
    _low_arr = df["Low"].values

    # Filter: one trade at a time, max per day, consecutive loss protection
    trades = []
    last_exit_idx = -1
    daily_trade_count: Dict[str, int] = {}
    consecutive_losses = 0

    for sig in signals:
        if sig.entry_index <= last_exit_idx:
            continue

        # Max trades per day
        if max_trades_per_day > 0:
            day_key = sig.entry_datetime.strftime("%Y-%m-%d")
            if daily_trade_count.get(day_key, 0) >= max_trades_per_day:
                continue

        # Consecutive loss protection
        if max_consecutive_losses > 0 and consecutive_losses >= max_consecutive_losses:
            # Skip until next day
            if trades:
                last_day = trades[-1].entry_datetime.date()
                if sig.entry_datetime.date() == last_day:
                    continue
                else:
                    consecutive_losses = 0

        trade = simulate_trade(
            sig, df, pip_size,
            spread_pips=spread_pips,
            trailing_sl=trailing_sl,
            trailing_activation_rr=trailing_activation_rr,
            break_even_rr=break_even_rr,
            partial_tp_rr=partial_tp_rr,
            partial_tp_pct=partial_tp_pct,
            _high_arr=_high_arr,
            _low_arr=_low_arr,
        )

        if trade:
            trades.append(trade)
            exit_idx = df.index.get_loc(trade.exit_datetime)
            last_exit_idx = exit_idx

            day_key = sig.entry_datetime.strftime("%Y-%m-%d")
            daily_trade_count[day_key] = daily_trade_count.get(day_key, 0) + 1

            if trade.result == "loss":
                consecutive_losses += 1
            else:
                consecutive_losses = 0

    result = BacktestResult(trades=trades, pair_name=pair_name, interval=interval)
    result.compute_metrics()
    return result


# ── Parameter optimization ─────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Result of a single parameter combination."""
    params: Dict
    total_pips: float
    win_rate: float
    profit_factor: float
    total_trades: int
    max_drawdown: float
    sharpe_ratio: float
    expectancy: float
    score: float = 0.0  # combined optimization score


def optimize_parameters(
    df: pd.DataFrame,
    pair_name: str,
    param_grid: Dict[str, List] = None,
    optimize_for: str = "expectancy",
    top_n: int = 10,
) -> List[OptimizationResult]:
    """Grid search over parameter combinations.

    Args:
        df: Price data
        pair_name: Currency pair
        param_grid: Dict of parameter names to lists of values to test
        optimize_for: Metric to optimize ("expectancy", "profit_factor",
                      "sharpe", "total_pips", "win_rate")
        top_n: Number of top results to return
    """
    if param_grid is None:
        param_grid = {
            "swing_lookback": [3, 5, 8],
            "cluster_pips": [5.0, 10.0, 15.0],
            "min_wick_pips": [2.0, 3.0, 5.0],
            "rr_ratio": [1.5, 2.0, 3.0],
            "min_confluence": [0, 1, 2],
        }

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    def _safe_val(v, cap=100.0):
        """Clamp inf/NaN to a safe numeric value."""
        if v != v or v == float("inf") or v == float("-inf"):  # NaN or inf
            return 0.0
        return min(v, cap)

    results = []
    for combo in combinations:
        params = dict(zip(param_names, combo))

        try:
            bt_result = run_backtest(df, pair_name, **params)

            if bt_result.total_trades < 5:
                continue

            pf = _safe_val(bt_result.profit_factor, 10.0)
            rf = _safe_val(bt_result.recovery_factor, 20.0)
            pr = _safe_val(bt_result.payoff_ratio, 10.0)

            # Compute optimization score
            if optimize_for == "expectancy":
                score = bt_result.expectancy_pips
            elif optimize_for == "profit_factor":
                score = pf
            elif optimize_for == "sharpe":
                score = bt_result.sharpe_ratio
            elif optimize_for == "total_pips":
                score = bt_result.total_pips
            elif optimize_for == "win_rate":
                score = bt_result.win_rate
            elif optimize_for == "combined":
                # Normalized balanced score (all components on 0-100 scale)
                norm_exp = min(max(bt_result.expectancy_pips, -10), 10) * 5  # -50 to 50
                norm_pf = min(pf, 5) * 10  # 0 to 50
                norm_sharpe = min(max(bt_result.sharpe_ratio, -2), 5) * 10  # -20 to 50
                norm_wr = bt_result.win_rate  # 0 to 100
                norm_rf = min(rf, 10) * 5  # 0 to 50
                score = (
                    norm_exp * 0.25 +
                    norm_pf * 0.25 +
                    norm_sharpe * 0.20 +
                    norm_wr * 0.15 +
                    norm_rf * 0.15
                )
            else:
                score = bt_result.expectancy_pips

            # Guard against NaN score
            if score != score:  # NaN check
                continue

            results.append(OptimizationResult(
                params=params,
                total_pips=bt_result.total_pips,
                win_rate=bt_result.win_rate,
                profit_factor=bt_result.profit_factor,
                total_trades=bt_result.total_trades,
                max_drawdown=bt_result.max_drawdown_pips,
                sharpe_ratio=bt_result.sharpe_ratio,
                expectancy=bt_result.expectancy_pips,
                score=round(score, 2),
            ))
        except (ValueError, KeyError, IndexError):
            continue

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]

"""Backtesting engine for liquidity inducement strategies."""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List
from engine.liquidity import (
    InducementSignal,
    find_swing_points,
    find_liquidity_levels,
    detect_liquidity_sweeps,
    detect_inducement_traps,
    get_pip_size,
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


@dataclass
class BacktestResult:
    """Full backtest results and metrics."""
    trades: List[Trade]
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pips: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    max_drawdown_pips: float = 0.0
    profit_factor: float = 0.0
    expectancy_pips: float = 0.0
    best_trade_pips: float = 0.0
    worst_trade_pips: float = 0.0
    long_wins: int = 0
    short_wins: int = 0
    equity_curve: List[float] = field(default_factory=list)

    def compute_metrics(self):
        if not self.trades:
            return

        self.total_trades = len(self.trades)
        self.wins = sum(1 for t in self.trades if t.result == "win")
        self.losses = sum(1 for t in self.trades if t.result == "loss")
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

        self.long_wins = sum(1 for t in self.trades if t.result == "win" and t.direction == "long")
        self.short_wins = sum(1 for t in self.trades if t.result == "win" and t.direction == "short")

        # Equity curve and drawdown
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        curve = [0.0]
        for t in self.trades:
            equity += t.pnl_pips
            curve.append(equity)
            peak = max(peak, equity)
            dd = peak - equity
            max_dd = max(max_dd, dd)
        self.equity_curve = curve
        self.max_drawdown_pips = max_dd


def simulate_trade(signal: InducementSignal, df: pd.DataFrame,
                   pip_size: float) -> Trade | None:
    """Simulate a single trade from entry to SL or TP hit.

    Walks forward from the signal candle checking if SL or TP is hit first.
    """
    for j in range(signal.entry_index + 1, len(df)):
        candle_high = df["High"].iloc[j]
        candle_low = df["Low"].iloc[j]

        if signal.direction == "long":
            # Check stop loss first (worst case)
            if candle_low <= signal.stop_loss:
                pnl = (signal.stop_loss - signal.entry_price) / pip_size
                return Trade(
                    entry_datetime=signal.entry_datetime,
                    exit_datetime=df.index[j],
                    direction="long",
                    entry_price=signal.entry_price,
                    exit_price=signal.stop_loss,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    pnl_pips=round(pnl, 1),
                    result="loss",
                    signal_type=signal.signal_type,
                    swept_level=signal.swept_level,
                )
            if candle_high >= signal.take_profit:
                pnl = (signal.take_profit - signal.entry_price) / pip_size
                return Trade(
                    entry_datetime=signal.entry_datetime,
                    exit_datetime=df.index[j],
                    direction="long",
                    entry_price=signal.entry_price,
                    exit_price=signal.take_profit,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    pnl_pips=round(pnl, 1),
                    result="win",
                    signal_type=signal.signal_type,
                    swept_level=signal.swept_level,
                )

        elif signal.direction == "short":
            # Check stop loss first
            if candle_high >= signal.stop_loss:
                pnl = (signal.entry_price - signal.stop_loss) / pip_size
                return Trade(
                    entry_datetime=signal.entry_datetime,
                    exit_datetime=df.index[j],
                    direction="short",
                    entry_price=signal.entry_price,
                    exit_price=signal.stop_loss,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    pnl_pips=round(pnl, 1),
                    result="loss",
                    signal_type=signal.signal_type,
                    swept_level=signal.swept_level,
                )
            if candle_low <= signal.take_profit:
                pnl = (signal.entry_price - signal.take_profit) / pip_size
                return Trade(
                    entry_datetime=signal.entry_datetime,
                    exit_datetime=df.index[j],
                    direction="short",
                    entry_price=signal.entry_price,
                    exit_price=signal.take_profit,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    pnl_pips=round(pnl, 1),
                    result="win",
                    signal_type=signal.signal_type,
                    swept_level=signal.swept_level,
                )

    return None  # Trade still open at end of data


def run_backtest(
    df: pd.DataFrame,
    pair_name: str,
    swing_lookback: int = 5,
    cluster_pips: float = 10.0,
    min_wick_pips: float = 3.0,
    strategy: str = "both",
) -> BacktestResult:
    """Run a full backtest on the given data.

    Args:
        df: OHLCV DataFrame
        pair_name: Pair name for pip size lookup
        swing_lookback: Candles on each side to confirm swing points
        cluster_pips: Distance to cluster swing points into levels
        min_wick_pips: Minimum wick beyond level for sweep detection
        strategy: "sweeps", "inducement", or "both"

    Returns:
        BacktestResult with all trades and computed metrics
    """
    pip_size = get_pip_size(pair_name)

    # Detect structure
    swings = find_swing_points(df, lookback=swing_lookback)
    levels = find_liquidity_levels(swings, cluster_pips=cluster_pips, pip_size=pip_size)

    # Generate signals
    signals = []
    if strategy in ("sweeps", "both"):
        signals.extend(detect_liquidity_sweeps(df, levels, min_wick_pips=min_wick_pips,
                                                pip_size=pip_size))
    if strategy in ("inducement", "both"):
        signals.extend(detect_inducement_traps(df, swings, pip_size=pip_size))

    # Sort signals by time
    signals.sort(key=lambda s: s.entry_index)

    # Remove overlapping signals (only one trade at a time)
    filtered_signals = []
    last_exit_idx = -1
    for sig in signals:
        if sig.entry_index > last_exit_idx:
            trade = simulate_trade(sig, df, pip_size)
            if trade:
                filtered_signals.append(sig)
                # Find exit index
                exit_idx = df.index.get_loc(trade.exit_datetime)
                last_exit_idx = exit_idx

    # Simulate all filtered trades
    trades = []
    for sig in filtered_signals:
        trade = simulate_trade(sig, df, pip_size)
        if trade:
            trades.append(trade)

    result = BacktestResult(trades=trades)
    result.compute_metrics()
    return result

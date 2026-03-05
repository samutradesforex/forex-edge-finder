"""Liquidity inducement detection engine.

Identifies liquidity inducement patterns:
- Swing highs/lows that act as liquidity pools
- Inducement levels (minor structure breaks that lure traders)
- Liquidity sweeps (price wicks beyond key levels before reversing)
- Stop hunt patterns (false breakouts that grab liquidity)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class SwingPoint:
    """A swing high or low point."""
    index: int
    datetime: pd.Timestamp
    price: float
    kind: str  # "high" or "low"


@dataclass
class LiquidityLevel:
    """A detected liquidity level."""
    price: float
    kind: str  # "buy_side" (above highs) or "sell_side" (below lows)
    strength: int  # how many times price has respected this level
    formed_at: pd.Timestamp


@dataclass
class InducementSignal:
    """A liquidity inducement trading signal."""
    entry_index: int
    entry_datetime: pd.Timestamp
    entry_price: float
    direction: str  # "long" or "short"
    stop_loss: float
    take_profit: float
    swept_level: float
    signal_type: str  # "sweep_reversal", "inducement_trap", "stop_hunt"


def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> List[SwingPoint]:
    """Detect swing highs and swing lows.

    A swing high has the highest high within `lookback` candles on each side.
    A swing low has the lowest low within `lookback` candles on each side.
    """
    highs = df["High"].values
    lows = df["Low"].values
    swings = []

    for i in range(lookback, len(df) - lookback):
        # Swing high: highest high in the window
        window_highs = highs[i - lookback : i + lookback + 1]
        if highs[i] == window_highs.max() and np.sum(window_highs == highs[i]) == 1:
            swings.append(SwingPoint(
                index=i,
                datetime=df.index[i],
                price=highs[i],
                kind="high",
            ))

        # Swing low: lowest low in the window
        window_lows = lows[i - lookback : i + lookback + 1]
        if lows[i] == window_lows.min() and np.sum(window_lows == lows[i]) == 1:
            swings.append(SwingPoint(
                index=i,
                datetime=df.index[i],
                price=lows[i],
                kind="low",
            ))

    return swings


def find_liquidity_levels(swings: List[SwingPoint], cluster_pips: float = 10.0,
                          pip_size: float = 0.0001) -> List[LiquidityLevel]:
    """Cluster swing points into liquidity levels.

    Multiple swing highs/lows near the same price form stronger liquidity pools.
    """
    cluster_dist = cluster_pips * pip_size
    levels = []

    # Process highs and lows separately
    for kind, liq_kind in [("high", "buy_side"), ("low", "sell_side")]:
        points = sorted([s for s in swings if s.kind == kind], key=lambda s: s.price)
        if not points:
            continue

        cluster = [points[0]]
        for p in points[1:]:
            if abs(p.price - cluster[-1].price) <= cluster_dist:
                cluster.append(p)
            else:
                avg_price = np.mean([c.price for c in cluster])
                levels.append(LiquidityLevel(
                    price=avg_price,
                    kind=liq_kind,
                    strength=len(cluster),
                    formed_at=cluster[0].datetime,
                ))
                cluster = [p]

        # Last cluster
        avg_price = np.mean([c.price for c in cluster])
        levels.append(LiquidityLevel(
            price=avg_price,
            kind=liq_kind,
            strength=len(cluster),
            formed_at=cluster[0].datetime,
        ))

    return levels


def detect_liquidity_sweeps(df: pd.DataFrame, levels: List[LiquidityLevel],
                            min_wick_pips: float = 3.0,
                            pip_size: float = 0.0001) -> List[InducementSignal]:
    """Detect liquidity sweep patterns.

    A sweep occurs when price wicks beyond a liquidity level then closes back
    inside — indicating the level was raided and a reversal is likely.
    """
    signals = []
    min_wick = min_wick_pips * pip_size

    for level in levels:
        for i in range(1, len(df)):
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]
            close = df["Close"].iloc[i]
            open_price = df["Open"].iloc[i]

            if level.kind == "buy_side":
                # Price wicks above the buy-side liquidity, then closes below
                sweep_distance = high - level.price
                if sweep_distance > min_wick and close < level.price:
                    # Bearish sweep — short signal
                    sl = high + (5 * pip_size)
                    risk = sl - close
                    tp = close - (risk * 2)  # 2:1 RR
                    signals.append(InducementSignal(
                        entry_index=i,
                        entry_datetime=df.index[i],
                        entry_price=close,
                        direction="short",
                        stop_loss=sl,
                        take_profit=tp,
                        swept_level=level.price,
                        signal_type="sweep_reversal",
                    ))

            elif level.kind == "sell_side":
                # Price wicks below sell-side liquidity, then closes above
                sweep_distance = level.price - low
                if sweep_distance > min_wick and close > level.price:
                    # Bullish sweep — long signal
                    sl = low - (5 * pip_size)
                    risk = close - sl
                    tp = close + (risk * 2)  # 2:1 RR
                    signals.append(InducementSignal(
                        entry_index=i,
                        entry_datetime=df.index[i],
                        entry_price=close,
                        direction="long",
                        stop_loss=sl,
                        take_profit=tp,
                        swept_level=level.price,
                        signal_type="sweep_reversal",
                    ))

    return signals


def detect_inducement_traps(df: pd.DataFrame, swings: List[SwingPoint],
                            pip_size: float = 0.0001) -> List[InducementSignal]:
    """Detect inducement trap patterns.

    Inducement occurs when a minor swing high/low is taken out to lure traders
    into breakout trades, only for price to reverse to the major structure.
    Pattern: minor swing broken -> immediate reversal candle.
    """
    signals = []
    swing_highs = [s for s in swings if s.kind == "high"]
    swing_lows = [s for s in swings if s.kind == "low"]

    for i in range(2, len(df)):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        close = df["Close"].iloc[i]
        prev_close = df["Close"].iloc[i - 1]

        # Check for bearish inducement trap
        # Recent minor high broken, then strong bearish close
        for sh in swing_highs:
            if sh.index < i - 20 or sh.index >= i:
                continue
            if high > sh.price and close < prev_close and close < sh.price:
                sl = high + (5 * pip_size)
                risk = sl - close
                tp = close - (risk * 2)
                signals.append(InducementSignal(
                    entry_index=i,
                    entry_datetime=df.index[i],
                    entry_price=close,
                    direction="short",
                    stop_loss=sl,
                    take_profit=tp,
                    swept_level=sh.price,
                    signal_type="inducement_trap",
                ))
                break

        # Check for bullish inducement trap
        for sl_point in swing_lows:
            if sl_point.index < i - 20 or sl_point.index >= i:
                continue
            if low < sl_point.price and close > prev_close and close > sl_point.price:
                sl = low - (5 * pip_size)
                risk = close - sl
                tp = close + (risk * 2)
                signals.append(InducementSignal(
                    entry_index=i,
                    entry_datetime=df.index[i],
                    entry_price=close,
                    direction="long",
                    stop_loss=sl,
                    take_profit=tp,
                    swept_level=sl_point.price,
                    signal_type="inducement_trap",
                ))
                break

    return signals


def get_pip_size(pair_name: str) -> float:
    """Return pip size for a currency pair."""
    jpy_pairs = ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY", "CAD/JPY", "NZD/JPY"]
    if pair_name in jpy_pairs:
        return 0.01
    return 0.0001

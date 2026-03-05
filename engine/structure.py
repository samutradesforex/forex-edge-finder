"""Market structure analysis — Break of Structure (BOS) and Change of Character (CHoCH).

Determines the current market bias (bullish/bearish/ranging) by tracking
how price interacts with swing highs and lows over time. This is used
as a directional filter for the liquidity signals.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List
from engine.liquidity import SwingPoint, find_swing_points


@dataclass
class StructureBreak:
    """A Break of Structure (BOS) or Change of Character (CHoCH)."""
    index: int
    datetime: pd.Timestamp
    price: float
    kind: str  # "bos" or "choch"
    direction: str  # "bullish" or "bearish"
    broken_swing: SwingPoint


def detect_structure_breaks(df: pd.DataFrame, swings: List[SwingPoint]
                            ) -> List[StructureBreak]:
    """Detect BOS (trend continuation) and CHoCH (trend reversal) events.

    BOS = break of structure in the direction of the existing trend
    CHoCH = change of character, break against the existing trend

    Logic:
    - In an uptrend (higher highs, higher lows):
      - Break above previous swing high = BOS (bullish continuation)
      - Break below previous swing low = CHoCH (bearish reversal)
    - In a downtrend (lower lows, lower highs):
      - Break below previous swing low = BOS (bearish continuation)
      - Break above previous swing high = CHoCH (bullish reversal)
    """
    breaks = []
    if len(swings) < 4:
        return breaks

    swing_highs = [s for s in swings if s.kind == "high"]
    swing_lows = [s for s in swings if s.kind == "low"]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return breaks

    # Determine initial bias from first few swings
    bias = "neutral"

    for i in range(1, len(swing_highs)):
        prev_sh = swing_highs[i - 1]
        curr_sh = swing_highs[i]

        # Find candles between these swing highs that break above
        for j in range(prev_sh.index + 1, min(curr_sh.index + 1, len(df))):
            if df["High"].iloc[j] > prev_sh.price:
                if bias == "bearish":
                    kind = "choch"
                else:
                    kind = "bos"
                bias = "bullish"
                breaks.append(StructureBreak(
                    index=j, datetime=df.index[j], price=prev_sh.price,
                    kind=kind, direction="bullish", broken_swing=prev_sh,
                ))
                break

    for i in range(1, len(swing_lows)):
        prev_sl = swing_lows[i - 1]
        curr_sl = swing_lows[i]

        for j in range(prev_sl.index + 1, min(curr_sl.index + 1, len(df))):
            if df["Low"].iloc[j] < prev_sl.price:
                if bias == "bullish":
                    kind = "choch"
                else:
                    kind = "bos"
                bias = "bearish"
                breaks.append(StructureBreak(
                    index=j, datetime=df.index[j], price=prev_sl.price,
                    kind=kind, direction="bearish", broken_swing=prev_sl,
                ))
                break

    breaks.sort(key=lambda b: b.index)
    return breaks


def get_bias_at(breaks: List[StructureBreak], index: int) -> str:
    """Get the market bias at a given candle index.

    Returns "bullish", "bearish", or "neutral".
    """
    bias = "neutral"
    for b in breaks:
        if b.index > index:
            break
        bias = b.direction
    return bias


def get_bias_series(df: pd.DataFrame, breaks: List[StructureBreak]) -> pd.Series:
    """Create a series of market bias for each candle."""
    bias_arr = ["neutral"] * len(df)
    current_bias = "neutral"

    break_idx = 0
    for i in range(len(df)):
        while break_idx < len(breaks) and breaks[break_idx].index <= i:
            current_bias = breaks[break_idx].direction
            break_idx += 1
        bias_arr[i] = current_bias

    return pd.Series(bias_arr, index=df.index, name="bias")


def compute_structure(df: pd.DataFrame, swing_lookback: int = 5):
    """Compute full market structure analysis.

    Returns:
        swings: List of swing points
        breaks: List of structure breaks
        bias_series: Series with bias at each candle
    """
    swings = find_swing_points(df, lookback=swing_lookback)
    breaks = detect_structure_breaks(df, swings)
    bias_series = get_bias_series(df, breaks)
    return swings, breaks, bias_series

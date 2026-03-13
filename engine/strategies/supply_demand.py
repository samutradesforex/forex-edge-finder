"""Supply/Demand Zone Bounce strategy — enters on price reaction from supply/demand zones.

Detects impulse moves (3+ consecutive directional candles totalling > 2x ATR),
marks the last opposing candle body as a zone, and signals when price returns to
touch the zone without closing through it.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from engine.strategies import register_strategy
from engine.liquidity import (
    InducementSignal,
    LiquidityLevel,
    FairValueGap,
    OrderBlock,
    calc_atr,
    calc_ema,
    calc_rsi,
    get_session,
    is_in_killzone,
    score_confluence,
    _build_fvg_index,
    _build_ob_index,
)


# ── Zone dataclass (internal) ──────────────────────────────────────────────

class _Zone:
    """A supply or demand zone defined by the body of a candle."""
    __slots__ = ("top", "bottom", "direction", "origin_index", "mitigated")

    def __init__(self, top: float, bottom: float, direction: str, origin_index: int):
        self.top = top            # upper edge of zone (max of open/close)
        self.bottom = bottom      # lower edge of zone (min of open/close)
        self.direction = direction  # "demand" or "supply"
        self.origin_index = origin_index
        self.mitigated = False


# ── Detection function ──────────────────────────────────────────────────────

def detect_supply_demand(df: pd.DataFrame, pip_size: float = 0.0001,
                         atr: pd.Series = None, fvgs: List[FairValueGap] = None,
                         obs: List[OrderBlock] = None, rsi: pd.Series = None,
                         ema_fast: pd.Series = None, ema_slow: pd.Series = None,
                         require_displacement: bool = False,
                         min_confluence: int = 0, session_filter: str = "all",
                         rr_ratio: float = 2.0,
                         **kwargs) -> List[InducementSignal]:
    """Detect supply and demand zone bounce signals.

    1. Scan for *impulse moves*: 3+ consecutive candles in one direction whose
       total range exceeds 2x the ATR at that point.
    2. The zone is the body (open-close range) of the last opposing candle
       immediately before the impulse.
    3. When a later candle's wick touches the zone but the close does not
       penetrate through it, fire a signal.

    Returns a list of ``InducementSignal`` objects.
    """
    signals: List[InducementSignal] = []

    if len(df) < 10:
        return signals

    # --- defaults for optional indicators --------------------------------
    if atr is None:
        atr = calc_atr(df)
    if ema_fast is None:
        ema_fast = calc_ema(df["Close"], 21)
    if ema_slow is None:
        ema_slow = calc_ema(df["Close"], 50)
    if rsi is None:
        rsi = calc_rsi(df)
    if fvgs is None:
        fvgs = []
    if obs is None:
        obs = []

    now = pd.Timestamp.now()

    # Pre-extract numpy arrays for speed
    open_vals = df["Open"].values
    high_vals = df["High"].values
    low_vals = df["Low"].values
    close_vals = df["Close"].values
    atr_vals = atr.values
    rsi_vals = rsi.values
    ema_f_vals = ema_fast.values
    ema_s_vals = ema_slow.values
    fvg_indices = _build_fvg_index(fvgs)
    ob_indices = _build_ob_index(obs)
    dates = df.index

    conf_kw = dict(
        _close_arr=close_vals, _open_arr=open_vals,
        _atr_arr=atr_vals, _rsi_arr=rsi_vals,
        _ema_f_arr=ema_f_vals, _ema_s_arr=ema_s_vals,
        _fvg_indices=fvg_indices, _ob_indices=ob_indices,
    )

    min_impulse_candles = 3

    # ── Step 1: identify all zones ──────────────────────────────────────
    zones: List[_Zone] = []

    i = min_impulse_candles  # start after enough room for impulse check
    while i < len(df):
        atr_val = atr_vals[i]
        if np.isnan(atr_val) or atr_val == 0:
            i += 1
            continue

        # --- check for bullish impulse ending at candle i ----------------
        bull_count = 0
        j = i
        while j >= 1 and close_vals[j] > open_vals[j]:
            bull_count += 1
            j -= 1

        if bull_count >= min_impulse_candles:
            impulse_start = j + 1
            impulse_end = i
            total_move = close_vals[impulse_end] - open_vals[impulse_start]
            atr_at_start = atr_vals[impulse_start]
            if np.isnan(atr_at_start) or atr_at_start == 0:
                atr_at_start = atr_val

            if total_move > 2.0 * atr_at_start:
                # The zone candle is the last bearish candle before the impulse
                zone_idx = impulse_start - 1
                if zone_idx >= 0:
                    z_open = open_vals[zone_idx]
                    z_close = close_vals[zone_idx]
                    zone_top = max(z_open, z_close)
                    zone_bottom = min(z_open, z_close)
                    zones.append(_Zone(
                        top=zone_top, bottom=zone_bottom,
                        direction="demand", origin_index=zone_idx,
                    ))
                    # skip past the impulse to avoid overlapping zones
                    i = impulse_end + 1
                    continue

        # --- check for bearish impulse ending at candle i ----------------
        bear_count = 0
        j = i
        while j >= 1 and close_vals[j] < open_vals[j]:
            bear_count += 1
            j -= 1

        if bear_count >= min_impulse_candles:
            impulse_start = j + 1
            impulse_end = i
            total_move = open_vals[impulse_start] - close_vals[impulse_end]
            atr_at_start = atr_vals[impulse_start]
            if np.isnan(atr_at_start) or atr_at_start == 0:
                atr_at_start = atr_val

            if total_move > 2.0 * atr_at_start:
                zone_idx = impulse_start - 1
                if zone_idx >= 0:
                    z_open = open_vals[zone_idx]
                    z_close = close_vals[zone_idx]
                    zone_top = max(z_open, z_close)
                    zone_bottom = min(z_open, z_close)
                    zones.append(_Zone(
                        top=zone_top, bottom=zone_bottom,
                        direction="supply", origin_index=zone_idx,
                    ))
                    i = impulse_end + 1
                    continue

        i += 1

    # ── Step 2: scan for touches / bounces off zones ────────────────────
    for zone in zones:
        # Only look at candles after the impulse that created the zone
        scan_start = zone.origin_index + min_impulse_candles + 1
        if scan_start >= len(df):
            continue

        for i in range(scan_start, len(df)):
            if zone.mitigated:
                break

            atr_val = atr_vals[i]
            if np.isnan(atr_val) or atr_val == 0:
                continue

            dt = dates[i]

            # Session filter
            if session_filter != "all":
                session = get_session(dt)
                if session != session_filter and session_filter != "killzones":
                    continue
                if session_filter == "killzones" and not is_in_killzone(dt):
                    continue

            close = close_vals[i]
            low = low_vals[i]
            high = high_vals[i]

            if zone.direction == "demand":
                # Price wick touches the zone from above but close stays above zone top
                touches_zone = low <= zone.top
                closes_through = close < zone.bottom
                valid_bounce = touches_zone and close > zone.top and not closes_through

                if closes_through:
                    zone.mitigated = True
                    break

                if not valid_bounce:
                    continue

                # Displacement check: require the bounce candle to be bullish
                if require_displacement and close <= open_vals[i]:
                    continue

                sl = zone.bottom - (atr_val * 0.25)
                risk = close - sl
                if risk <= 0:
                    continue
                tp = close + (risk * rr_ratio)

                level = LiquidityLevel(
                    price=zone.bottom, kind="sell_side", strength=1, formed_at=now)
                conf_score, conf_factors = score_confluence(
                    df, i, "long", level, atr, fvgs, obs, rsi,
                    ema_fast, ema_slow, pip_size, **conf_kw)
                conf_factors.insert(0, "demand_zone_bounce")
                conf_score += 1

                if conf_score < min_confluence:
                    continue

                signals.append(InducementSignal(
                    entry_index=i,
                    entry_datetime=dt,
                    entry_price=close,
                    direction="long",
                    stop_loss=sl,
                    take_profit=tp,
                    swept_level=zone.bottom,
                    signal_type="supply_demand",
                    confluence_score=conf_score,
                    confluence_factors=conf_factors,
                    session=get_session(dt),
                ))
                # One signal per zone
                zone.mitigated = True
                break

            else:  # supply zone
                # Price wick touches the zone from below but close stays below zone bottom
                touches_zone = high >= zone.bottom
                closes_through = close > zone.top
                valid_bounce = touches_zone and close < zone.bottom and not closes_through

                if closes_through:
                    zone.mitigated = True
                    break

                if not valid_bounce:
                    continue

                # Displacement check: require the bounce candle to be bearish
                if require_displacement and close >= open_vals[i]:
                    continue

                sl = zone.top + (atr_val * 0.25)
                risk = sl - close
                if risk <= 0:
                    continue
                tp = close - (risk * rr_ratio)

                level = LiquidityLevel(
                    price=zone.top, kind="buy_side", strength=1, formed_at=now)
                conf_score, conf_factors = score_confluence(
                    df, i, "short", level, atr, fvgs, obs, rsi,
                    ema_fast, ema_slow, pip_size, **conf_kw)
                conf_factors.insert(0, "supply_zone_bounce")
                conf_score += 1

                if conf_score < min_confluence:
                    continue

                signals.append(InducementSignal(
                    entry_index=i,
                    entry_datetime=dt,
                    entry_price=close,
                    direction="short",
                    stop_loss=sl,
                    take_profit=tp,
                    swept_level=zone.top,
                    signal_type="supply_demand",
                    confluence_score=conf_score,
                    confluence_factors=conf_factors,
                    session=get_session(dt),
                ))
                zone.mitigated = True
                break

    return signals


# ── Register the strategy ───────────────────────────────────────────────────

register_strategy(
    name="supply_demand",
    display_name="Supply/Demand Zone Bounce",
    category="smc",
    relevant_params=["rr_ratio", "min_confluence"],
    description="Enters on price bounce from supply/demand zones formed by impulse moves",
)(detect_supply_demand)

"""Liquidity inducement detection engine.

Identifies liquidity inducement patterns:
- Swing highs/lows that act as liquidity pools
- Inducement levels (minor structure breaks that lure traders)
- Liquidity sweeps (price wicks beyond key levels before reversing)
- Stop hunt patterns (false breakouts that grab liquidity)
- Displacement confirmation (strong momentum after sweep)
- Fair Value Gap (FVG) confluence
- Order block detection
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


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
    last_touched: Optional[pd.Timestamp] = None
    times_tested: int = 0
    still_valid: bool = True


@dataclass
class FairValueGap:
    """A Fair Value Gap (imbalance)."""
    index: int
    datetime: pd.Timestamp
    top: float
    bottom: float
    direction: str  # "bullish" or "bearish"
    filled: bool = False


@dataclass
class OrderBlock:
    """An order block (last opposing candle before displacement)."""
    index: int
    datetime: pd.Timestamp
    high: float
    low: float
    direction: str  # "bullish" (demand) or "bearish" (supply)
    mitigated: bool = False


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
    confluence_score: int = 0  # 0-5 how many confluence factors align
    confluence_factors: List[str] = field(default_factory=list)
    session: str = ""  # "london", "new_york", "asia", "overlap"


# ── Session / Killzone helpers ──────────────────────────────────────────────

SESSIONS = {
    "asia": (0, 8),       # 00:00 - 08:00 UTC
    "london": (7, 16),    # 07:00 - 16:00 UTC
    "new_york": (12, 21), # 12:00 - 21:00 UTC
    "lo_ny_overlap": (12, 16),  # London-NY overlap
}

KILLZONES = {
    "london_open": (7, 10),    # 07:00 - 10:00 UTC
    "new_york_open": (12, 15), # 12:00 - 15:00 UTC
    "london_close": (15, 17),  # 15:00 - 17:00 UTC
    "asia_open": (0, 3),       # 00:00 - 03:00 UTC
}


def get_session(dt: pd.Timestamp) -> str:
    """Determine which trading session a timestamp falls in."""
    hour = dt.hour
    for name, (start, end) in SESSIONS.items():
        if start <= hour < end:
            return name
    return "off_session"


def get_killzone(dt: pd.Timestamp) -> str:
    """Determine which killzone a timestamp falls in."""
    hour = dt.hour
    for name, (start, end) in KILLZONES.items():
        if start <= hour < end:
            return name
    return "none"


def is_in_killzone(dt: pd.Timestamp) -> bool:
    """Check if timestamp is within any killzone."""
    return get_killzone(dt) != "none"


# ── Swing detection ─────────────────────────────────────────────────────────

def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> List[SwingPoint]:
    """Detect swing highs and swing lows using vectorized approach."""
    highs = df["High"].values
    lows = df["Low"].values
    swings = []
    n = len(df)

    for i in range(lookback, n - lookback):
        window_highs = highs[i - lookback : i + lookback + 1]
        if highs[i] == window_highs.max() and np.sum(window_highs == highs[i]) == 1:
            swings.append(SwingPoint(
                index=i, datetime=df.index[i], price=highs[i], kind="high",
            ))

        window_lows = lows[i - lookback : i + lookback + 1]
        if lows[i] == window_lows.min() and np.sum(window_lows == lows[i]) == 1:
            swings.append(SwingPoint(
                index=i, datetime=df.index[i], price=lows[i], kind="low",
            ))

    return swings


def find_multi_tf_swings(df: pd.DataFrame, lookbacks: List[int] = None) -> List[SwingPoint]:
    """Find swing points across multiple lookback periods for stronger levels."""
    if lookbacks is None:
        lookbacks = [3, 5, 8, 13]

    all_swings = []
    for lb in lookbacks:
        all_swings.extend(find_swing_points(df, lookback=lb))
    return all_swings


# ── Liquidity level detection ───────────────────────────────────────────────

def find_liquidity_levels(swings: List[SwingPoint], cluster_pips: float = 10.0,
                          pip_size: float = 0.0001) -> List[LiquidityLevel]:
    """Cluster swing points into liquidity levels with strength scoring."""
    cluster_dist = cluster_pips * pip_size
    levels = []

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
                    last_touched=cluster[-1].datetime,
                    times_tested=len(cluster),
                ))
                cluster = [p]

        avg_price = np.mean([c.price for c in cluster])
        levels.append(LiquidityLevel(
            price=avg_price, kind=liq_kind, strength=len(cluster),
            formed_at=cluster[0].datetime, last_touched=cluster[-1].datetime,
            times_tested=len(cluster),
        ))

    return levels


# ── Fair Value Gap detection ────────────────────────────────────────────────

def find_fvgs(df: pd.DataFrame, min_gap_pips: float = 2.0,
              pip_size: float = 0.0001) -> List[FairValueGap]:
    """Detect Fair Value Gaps (3-candle imbalances)."""
    fvgs = []
    min_gap = min_gap_pips * pip_size

    for i in range(2, len(df)):
        # Bullish FVG: candle[i] low > candle[i-2] high
        gap = df["Low"].iloc[i] - df["High"].iloc[i - 2]
        if gap > min_gap:
            fvgs.append(FairValueGap(
                index=i - 1, datetime=df.index[i - 1],
                top=df["Low"].iloc[i], bottom=df["High"].iloc[i - 2],
                direction="bullish",
            ))

        # Bearish FVG: candle[i-2] low > candle[i] high
        gap = df["Low"].iloc[i - 2] - df["High"].iloc[i]
        if gap > min_gap:
            fvgs.append(FairValueGap(
                index=i - 1, datetime=df.index[i - 1],
                top=df["Low"].iloc[i - 2], bottom=df["High"].iloc[i],
                direction="bearish",
            ))

    return fvgs


# ── Order Block detection ──────────────────────────────────────────────────

def find_order_blocks(df: pd.DataFrame, displacement_pips: float = 15.0,
                      pip_size: float = 0.0001) -> List[OrderBlock]:
    """Detect order blocks (last opposing candle before strong displacement)."""
    obs = []
    min_disp = displacement_pips * pip_size

    for i in range(2, len(df)):
        body_i = df["Close"].iloc[i] - df["Open"].iloc[i]
        body_prev = df["Close"].iloc[i - 1] - df["Open"].iloc[i - 1]

        # Bullish OB: bearish candle followed by strong bullish displacement
        if body_prev < 0 and body_i > min_disp:
            obs.append(OrderBlock(
                index=i - 1, datetime=df.index[i - 1],
                high=df["High"].iloc[i - 1], low=df["Low"].iloc[i - 1],
                direction="bullish",
            ))

        # Bearish OB: bullish candle followed by strong bearish displacement
        if body_prev > 0 and body_i < -min_disp:
            obs.append(OrderBlock(
                index=i - 1, datetime=df.index[i - 1],
                high=df["High"].iloc[i - 1], low=df["Low"].iloc[i - 1],
                direction="bearish",
            ))

    return obs


# ── Momentum & displacement helpers ────────────────────────────────────────

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def has_displacement(df: pd.DataFrame, index: int, direction: str,
                     atr: pd.Series, multiplier: float = 1.5) -> bool:
    """Check if there is strong displacement (momentum) at the given candle."""
    if index < 1 or index >= len(df):
        return False
    body = abs(df["Close"].iloc[index] - df["Open"].iloc[index])
    atr_val = atr.iloc[index]
    if pd.isna(atr_val) or atr_val == 0:
        return False
    return body > (atr_val * multiplier)


def is_engulfing(df: pd.DataFrame, index: int, direction: str) -> bool:
    """Check if candle at index is an engulfing pattern."""
    if index < 1:
        return False
    curr_open = df["Open"].iloc[index]
    curr_close = df["Close"].iloc[index]
    prev_open = df["Open"].iloc[index - 1]
    prev_close = df["Close"].iloc[index - 1]

    if direction == "long":
        return (curr_close > curr_open and prev_close < prev_open and
                curr_close > prev_open and curr_open < prev_close)
    else:
        return (curr_close < curr_open and prev_close > prev_open and
                curr_close < prev_open and curr_open > prev_close)


# ── Confluence scoring ──────────────────────────────────────────────────────

def score_confluence(df: pd.DataFrame, index: int, direction: str,
                     level: LiquidityLevel, atr: pd.Series,
                     fvgs: List[FairValueGap], obs: List[OrderBlock],
                     rsi: pd.Series, ema_fast: pd.Series, ema_slow: pd.Series,
                     pip_size: float) -> tuple:
    """Score confluence factors for a signal. Returns (score, factors list)."""
    score = 0
    factors = []

    # 1. Killzone timing
    dt = df.index[index]
    if is_in_killzone(dt):
        score += 1
        factors.append(f"killzone:{get_killzone(dt)}")

    # 2. Displacement confirmation
    if has_displacement(df, index, direction, atr, multiplier=1.5):
        score += 1
        factors.append("displacement")

    # 3. Engulfing candle
    if is_engulfing(df, index, direction):
        score += 1
        factors.append("engulfing")

    # 4. FVG confluence (nearby FVG in same direction)
    price = df["Close"].iloc[index]
    for fvg in fvgs:
        if fvg.index >= index:
            continue
        if direction == "long" and fvg.direction == "bullish":
            if fvg.bottom <= price <= fvg.top:
                score += 1
                factors.append("fvg_confluence")
                break
        elif direction == "short" and fvg.direction == "bearish":
            if fvg.bottom <= price <= fvg.top:
                score += 1
                factors.append("fvg_confluence")
                break

    # 5. Order block confluence
    for ob in obs:
        if ob.index >= index:
            continue
        if direction == "long" and ob.direction == "bullish":
            if ob.low <= price <= ob.high:
                score += 1
                factors.append("order_block")
                break
        elif direction == "short" and ob.direction == "bearish":
            if ob.low <= price <= ob.high:
                score += 1
                factors.append("order_block")
                break

    # 6. EMA trend alignment
    if index < len(ema_fast) and index < len(ema_slow):
        ema_f = ema_fast.iloc[index]
        ema_s = ema_slow.iloc[index]
        if not pd.isna(ema_f) and not pd.isna(ema_s):
            if direction == "long" and ema_f > ema_s:
                score += 1
                factors.append("trend_aligned")
            elif direction == "short" and ema_f < ema_s:
                score += 1
                factors.append("trend_aligned")

    # 7. RSI confirmation (oversold for longs, overbought for shorts during sweep)
    if index < len(rsi) and not pd.isna(rsi.iloc[index]):
        rsi_val = rsi.iloc[index]
        if direction == "long" and rsi_val < 35:
            score += 1
            factors.append(f"rsi_oversold:{rsi_val:.0f}")
        elif direction == "short" and rsi_val > 65:
            score += 1
            factors.append(f"rsi_overbought:{rsi_val:.0f}")

    # 8. Level strength
    if level.strength >= 3:
        score += 1
        factors.append(f"strong_level:{level.strength}x")

    return score, factors


# ── Signal detection ────────────────────────────────────────────────────────

def detect_liquidity_sweeps(df: pd.DataFrame, levels: List[LiquidityLevel],
                            min_wick_pips: float = 3.0, pip_size: float = 0.0001,
                            atr: pd.Series = None, fvgs: List[FairValueGap] = None,
                            obs: List[OrderBlock] = None, rsi: pd.Series = None,
                            ema_fast: pd.Series = None, ema_slow: pd.Series = None,
                            require_displacement: bool = False,
                            min_confluence: int = 0,
                            session_filter: str = "all",
                            rr_ratio: float = 2.0,
                            ) -> List[InducementSignal]:
    """Detect liquidity sweep patterns with confluence scoring."""
    signals = []
    min_wick = min_wick_pips * pip_size

    if atr is None:
        atr = calc_atr(df)
    if fvgs is None:
        fvgs = []
    if obs is None:
        obs = []
    if rsi is None:
        rsi = calc_rsi(df)
    if ema_fast is None:
        ema_fast = calc_ema(df["Close"], 21)
    if ema_slow is None:
        ema_slow = calc_ema(df["Close"], 50)

    for level in levels:
        for i in range(1, len(df)):
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]
            close = df["Close"].iloc[i]
            dt = df.index[i]

            # Session filter
            if session_filter != "all":
                session = get_session(dt)
                if session != session_filter and session_filter != "killzones":
                    continue
                if session_filter == "killzones" and not is_in_killzone(dt):
                    continue

            if level.kind == "buy_side":
                sweep_distance = high - level.price
                if sweep_distance > min_wick and close < level.price:
                    if require_displacement and not has_displacement(df, i, "short", atr):
                        continue

                    sl = high + (5 * pip_size)
                    risk = sl - close
                    tp = close - (risk * rr_ratio)

                    conf_score, conf_factors = score_confluence(
                        df, i, "short", level, atr, fvgs, obs, rsi,
                        ema_fast, ema_slow, pip_size)

                    if conf_score < min_confluence:
                        continue

                    signals.append(InducementSignal(
                        entry_index=i, entry_datetime=dt,
                        entry_price=close, direction="short",
                        stop_loss=sl, take_profit=tp,
                        swept_level=level.price,
                        signal_type="sweep_reversal",
                        confluence_score=conf_score,
                        confluence_factors=conf_factors,
                        session=get_session(dt),
                    ))

            elif level.kind == "sell_side":
                sweep_distance = level.price - low
                if sweep_distance > min_wick and close > level.price:
                    if require_displacement and not has_displacement(df, i, "long", atr):
                        continue

                    sl = low - (5 * pip_size)
                    risk = close - sl
                    tp = close + (risk * rr_ratio)

                    conf_score, conf_factors = score_confluence(
                        df, i, "long", level, atr, fvgs, obs, rsi,
                        ema_fast, ema_slow, pip_size)

                    if conf_score < min_confluence:
                        continue

                    signals.append(InducementSignal(
                        entry_index=i, entry_datetime=dt,
                        entry_price=close, direction="long",
                        stop_loss=sl, take_profit=tp,
                        swept_level=level.price,
                        signal_type="sweep_reversal",
                        confluence_score=conf_score,
                        confluence_factors=conf_factors,
                        session=get_session(dt),
                    ))

    return signals


def detect_inducement_traps(df: pd.DataFrame, swings: List[SwingPoint],
                            pip_size: float = 0.0001,
                            atr: pd.Series = None, fvgs: List[FairValueGap] = None,
                            obs: List[OrderBlock] = None, rsi: pd.Series = None,
                            ema_fast: pd.Series = None, ema_slow: pd.Series = None,
                            require_displacement: bool = False,
                            min_confluence: int = 0,
                            session_filter: str = "all",
                            rr_ratio: float = 2.0,
                            lookback_candles: int = 20,
                            ) -> List[InducementSignal]:
    """Detect inducement trap patterns with confluence scoring."""
    signals = []
    swing_highs = [s for s in swings if s.kind == "high"]
    swing_lows = [s for s in swings if s.kind == "low"]

    if atr is None:
        atr = calc_atr(df)
    if fvgs is None:
        fvgs = []
    if obs is None:
        obs = []
    if rsi is None:
        rsi = calc_rsi(df)
    if ema_fast is None:
        ema_fast = calc_ema(df["Close"], 21)
    if ema_slow is None:
        ema_slow = calc_ema(df["Close"], 50)

    # Build a dummy level for confluence scoring
    dummy_level = LiquidityLevel(price=0, kind="", strength=1,
                                  formed_at=pd.Timestamp.now())

    for i in range(2, len(df)):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        close = df["Close"].iloc[i]
        prev_close = df["Close"].iloc[i - 1]
        dt = df.index[i]

        if session_filter != "all":
            session = get_session(dt)
            if session != session_filter and session_filter != "killzones":
                continue
            if session_filter == "killzones" and not is_in_killzone(dt):
                continue

        # Bearish inducement trap
        for sh in swing_highs:
            if sh.index < i - lookback_candles or sh.index >= i:
                continue
            if high > sh.price and close < prev_close and close < sh.price:
                if require_displacement and not has_displacement(df, i, "short", atr):
                    continue

                sl = high + (5 * pip_size)
                risk = sl - close
                tp = close - (risk * rr_ratio)

                dummy_level.price = sh.price
                dummy_level.kind = "buy_side"
                conf_score, conf_factors = score_confluence(
                    df, i, "short", dummy_level, atr, fvgs, obs, rsi,
                    ema_fast, ema_slow, pip_size)

                if conf_score < min_confluence:
                    continue

                signals.append(InducementSignal(
                    entry_index=i, entry_datetime=dt,
                    entry_price=close, direction="short",
                    stop_loss=sl, take_profit=tp,
                    swept_level=sh.price,
                    signal_type="inducement_trap",
                    confluence_score=conf_score,
                    confluence_factors=conf_factors,
                    session=get_session(dt),
                ))
                break

        # Bullish inducement trap
        for sl_point in swing_lows:
            if sl_point.index < i - lookback_candles or sl_point.index >= i:
                continue
            if low < sl_point.price and close > prev_close and close > sl_point.price:
                if require_displacement and not has_displacement(df, i, "long", atr):
                    continue

                sl = low - (5 * pip_size)
                risk = close - sl
                tp = close + (risk * rr_ratio)

                dummy_level.price = sl_point.price
                dummy_level.kind = "sell_side"
                conf_score, conf_factors = score_confluence(
                    df, i, "long", dummy_level, atr, fvgs, obs, rsi,
                    ema_fast, ema_slow, pip_size)

                if conf_score < min_confluence:
                    continue

                signals.append(InducementSignal(
                    entry_index=i, entry_datetime=dt,
                    entry_price=close, direction="long",
                    stop_loss=sl, take_profit=tp,
                    swept_level=sl_point.price,
                    signal_type="inducement_trap",
                    confluence_score=conf_score,
                    confluence_factors=conf_factors,
                    session=get_session(dt),
                ))
                break

    return signals


def detect_stop_hunts(df: pd.DataFrame, levels: List[LiquidityLevel],
                      pip_size: float = 0.0001,
                      atr: pd.Series = None, fvgs: List[FairValueGap] = None,
                      obs: List[OrderBlock] = None, rsi: pd.Series = None,
                      ema_fast: pd.Series = None, ema_slow: pd.Series = None,
                      min_confluence: int = 0,
                      session_filter: str = "all",
                      rr_ratio: float = 2.0,
                      ) -> List[InducementSignal]:
    """Detect stop hunt patterns — rapid spike beyond level + full reversal.

    Unlike sweep reversals, stop hunts require:
    - Wick > 2x the candle body (spike characteristic)
    - Close back within previous candle range (full rejection)
    """
    signals = []

    if atr is None:
        atr = calc_atr(df)
    if fvgs is None:
        fvgs = []
    if obs is None:
        obs = []
    if rsi is None:
        rsi = calc_rsi(df)
    if ema_fast is None:
        ema_fast = calc_ema(df["Close"], 21)
    if ema_slow is None:
        ema_slow = calc_ema(df["Close"], 50)

    for level in levels:
        for i in range(2, len(df)):
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]
            close = df["Close"].iloc[i]
            open_price = df["Open"].iloc[i]
            prev_high = df["High"].iloc[i - 1]
            prev_low = df["Low"].iloc[i - 1]
            dt = df.index[i]

            if session_filter != "all":
                session = get_session(dt)
                if session != session_filter and session_filter != "killzones":
                    continue
                if session_filter == "killzones" and not is_in_killzone(dt):
                    continue

            body = abs(close - open_price)
            if body == 0:
                continue

            if level.kind == "buy_side":
                upper_wick = high - max(close, open_price)
                if (high > level.price and close < level.price and
                        close < prev_high and upper_wick > body * 2):
                    sl = high + (5 * pip_size)
                    risk = sl - close
                    tp = close - (risk * rr_ratio)

                    conf_score, conf_factors = score_confluence(
                        df, i, "short", level, atr, fvgs, obs, rsi,
                        ema_fast, ema_slow, pip_size)

                    if conf_score < min_confluence:
                        continue

                    signals.append(InducementSignal(
                        entry_index=i, entry_datetime=dt,
                        entry_price=close, direction="short",
                        stop_loss=sl, take_profit=tp,
                        swept_level=level.price,
                        signal_type="stop_hunt",
                        confluence_score=conf_score,
                        confluence_factors=conf_factors,
                        session=get_session(dt),
                    ))

            elif level.kind == "sell_side":
                lower_wick = min(close, open_price) - low
                if (low < level.price and close > level.price and
                        close > prev_low and lower_wick > body * 2):
                    sl = low - (5 * pip_size)
                    risk = close - sl
                    tp = close + (risk * rr_ratio)

                    conf_score, conf_factors = score_confluence(
                        df, i, "long", level, atr, fvgs, obs, rsi,
                        ema_fast, ema_slow, pip_size)

                    if conf_score < min_confluence:
                        continue

                    signals.append(InducementSignal(
                        entry_index=i, entry_datetime=dt,
                        entry_price=close, direction="long",
                        stop_loss=sl, take_profit=tp,
                        swept_level=level.price,
                        signal_type="stop_hunt",
                        confluence_score=conf_score,
                        confluence_factors=conf_factors,
                        session=get_session(dt),
                    ))

    return signals


def get_pip_size(pair_name: str) -> float:
    """Return pip size for a currency pair."""
    jpy_pairs = ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY", "CAD/JPY", "NZD/JPY",
                 "CHF/JPY"]
    if pair_name in jpy_pairs:
        return 0.01
    return 0.0001

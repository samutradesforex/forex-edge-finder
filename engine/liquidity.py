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

import bisect

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
    """Detect swing highs and swing lows using vectorized rolling max/min."""
    highs = df["High"].values
    lows = df["Low"].values
    dates = df.index
    swings = []
    n = len(df)

    win = 2 * lookback + 1
    if n < win:
        return swings

    # Vectorized: compute rolling max/min once instead of per-candle slicing
    h_series = pd.Series(highs)
    l_series = pd.Series(lows)
    roll_max = h_series.rolling(win, center=True).max().values
    roll_min = l_series.rolling(win, center=True).min().values

    for i in range(lookback, n - lookback):
        h_val = highs[i]
        if h_val >= roll_max[i] and np.count_nonzero(highs[i - lookback : i + lookback + 1] == h_val) == 1:
            swings.append(SwingPoint(index=i, datetime=dates[i], price=h_val, kind="high"))

        l_val = lows[i]
        if l_val <= roll_min[i] and np.count_nonzero(lows[i - lookback : i + lookback + 1] == l_val) == 1:
            swings.append(SwingPoint(index=i, datetime=dates[i], price=l_val, kind="low"))

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
    low_arr = df["Low"].values
    high_arr = df["High"].values
    dates = df.index

    # Vectorized gap detection
    bull_gaps = low_arr[2:] - high_arr[:-2]
    bear_gaps = low_arr[:-2] - high_arr[2:]

    for i in range(len(bull_gaps)):
        idx = i + 2
        if bull_gaps[i] > min_gap:
            fvgs.append(FairValueGap(
                index=idx - 1, datetime=dates[idx - 1],
                top=low_arr[idx], bottom=high_arr[idx - 2],
                direction="bullish",
            ))
        if bear_gaps[i] > min_gap:
            fvgs.append(FairValueGap(
                index=idx - 1, datetime=dates[idx - 1],
                top=low_arr[idx - 2], bottom=high_arr[idx],
                direction="bearish",
            ))

    return fvgs


# ── Order Block detection ──────────────────────────────────────────────────

def find_order_blocks(df: pd.DataFrame, displacement_pips: float = 15.0,
                      pip_size: float = 0.0001) -> List[OrderBlock]:
    """Detect order blocks (last opposing candle before strong displacement)."""
    obs = []
    min_disp = displacement_pips * pip_size
    close_arr = df["Close"].values
    open_arr = df["Open"].values
    high_arr = df["High"].values
    low_arr = df["Low"].values
    dates = df.index

    # Vectorized body calculations
    body = close_arr - open_arr
    body_curr = body[2:]
    body_prev = body[1:-1]

    bull_mask = (body_prev < 0) & (body_curr > min_disp)
    bear_mask = (body_prev > 0) & (body_curr < -min_disp)

    for i in np.where(bull_mask)[0]:
        idx = i + 1  # original df index of the OB candle
        obs.append(OrderBlock(
            index=idx, datetime=dates[idx],
            high=high_arr[idx], low=low_arr[idx],
            direction="bullish",
        ))
    for i in np.where(bear_mask)[0]:
        idx = i + 1
        obs.append(OrderBlock(
            index=idx, datetime=dates[idx],
            high=high_arr[idx], low=low_arr[idx],
            direction="bearish",
        ))

    # Sort by index to maintain order
    obs.sort(key=lambda o: o.index)
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

def _build_fvg_index(fvgs: List[FairValueGap]) -> List[int]:
    """Build sorted index list for bisect lookups on FVGs."""
    return [fvg.index for fvg in fvgs]


def _build_ob_index(obs: List[OrderBlock]) -> List[int]:
    """Build sorted index list for bisect lookups on order blocks."""
    return [ob.index for ob in obs]


def score_confluence(df: pd.DataFrame, index: int, direction: str,
                     level: LiquidityLevel, atr: pd.Series,
                     fvgs: List[FairValueGap], obs: List[OrderBlock],
                     rsi: pd.Series, ema_fast: pd.Series, ema_slow: pd.Series,
                     pip_size: float,
                     _close_arr=None, _open_arr=None,
                     _atr_arr=None, _rsi_arr=None,
                     _ema_f_arr=None, _ema_s_arr=None,
                     _fvg_indices=None, _ob_indices=None,
                     ) -> tuple:
    """Score confluence factors for a signal. Returns (score, factors list)."""
    score = 0
    factors = []

    # Use pre-extracted arrays when available
    close_arr = _close_arr if _close_arr is not None else df["Close"].values
    open_arr = _open_arr if _open_arr is not None else df["Open"].values
    atr_arr = _atr_arr if _atr_arr is not None else atr.values
    rsi_arr = _rsi_arr if _rsi_arr is not None else rsi.values
    ema_f_arr = _ema_f_arr if _ema_f_arr is not None else ema_fast.values
    ema_s_arr = _ema_s_arr if _ema_s_arr is not None else ema_slow.values

    # 1. Killzone timing
    dt = df.index[index]
    if is_in_killzone(dt):
        score += 1
        factors.append(f"killzone:{get_killzone(dt)}")

    # 2. Displacement confirmation
    if index >= 1 and index < len(df):
        body = abs(close_arr[index] - open_arr[index])
        atr_val = atr_arr[index]
        if not np.isnan(atr_val) and atr_val != 0 and body > (atr_val * 1.5):
            score += 1
            factors.append("displacement")

    # 3. Engulfing candle
    if is_engulfing(df, index, direction):
        score += 1
        factors.append("engulfing")

    # 4. FVG confluence — use bisect for O(log N) lookup
    price = close_arr[index]
    fvg_indices = _fvg_indices if _fvg_indices is not None else _build_fvg_index(fvgs)
    search_start = index - 50
    lo = bisect.bisect_left(fvg_indices, search_start)
    hi = bisect.bisect_left(fvg_indices, index)
    for k in range(hi - 1, lo - 1, -1):
        fvg = fvgs[k]
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

    # 5. Order block confluence — use bisect for O(log N) lookup
    ob_indices = _ob_indices if _ob_indices is not None else _build_ob_index(obs)
    lo = bisect.bisect_left(ob_indices, search_start)
    hi = bisect.bisect_left(ob_indices, index)
    for k in range(hi - 1, lo - 1, -1):
        ob = obs[k]
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
    if index < len(ema_f_arr) and index < len(ema_s_arr):
        ema_f = ema_f_arr[index]
        ema_s = ema_s_arr[index]
        if not np.isnan(ema_f) and not np.isnan(ema_s):
            if direction == "long" and ema_f > ema_s:
                score += 1
                factors.append("trend_aligned")
            elif direction == "short" and ema_f < ema_s:
                score += 1
                factors.append("trend_aligned")

    # 7. RSI confirmation
    if index < len(rsi_arr) and not np.isnan(rsi_arr[index]):
        rsi_val = rsi_arr[index]
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

    # Pre-extract arrays for speed
    high_arr = df["High"].values
    low_arr = df["Low"].values
    close_arr = df["Close"].values
    open_arr = df["Open"].values
    atr_arr = atr.values
    rsi_arr = rsi.values
    ema_f_arr = ema_fast.values
    ema_s_arr = ema_slow.values
    fvg_indices = _build_fvg_index(fvgs)
    ob_indices = _build_ob_index(obs)
    dates = df.index

    conf_kw = dict(_close_arr=close_arr, _open_arr=open_arr,
                   _atr_arr=atr_arr, _rsi_arr=rsi_arr,
                   _ema_f_arr=ema_f_arr, _ema_s_arr=ema_s_arr,
                   _fvg_indices=fvg_indices, _ob_indices=ob_indices)

    for level in levels:
        for i in range(1, len(df)):
            high = high_arr[i]
            low = low_arr[i]
            close = close_arr[i]
            dt = dates[i]

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
                        ema_fast, ema_slow, pip_size, **conf_kw)

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
                        ema_fast, ema_slow, pip_size, **conf_kw)

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

    now = pd.Timestamp.now()

    # Pre-extract arrays
    high_arr = df["High"].values
    low_arr = df["Low"].values
    close_arr = df["Close"].values
    open_arr = df["Open"].values
    atr_arr = atr.values
    rsi_arr = rsi.values
    ema_f_arr = ema_fast.values
    ema_s_arr = ema_slow.values
    fvg_indices = _build_fvg_index(fvgs)
    ob_indices = _build_ob_index(obs)
    dates = df.index

    conf_kw = dict(_close_arr=close_arr, _open_arr=open_arr,
                   _atr_arr=atr_arr, _rsi_arr=rsi_arr,
                   _ema_f_arr=ema_f_arr, _ema_s_arr=ema_s_arr,
                   _fvg_indices=fvg_indices, _ob_indices=ob_indices)

    for i in range(2, len(df)):
        high = high_arr[i]
        low = low_arr[i]
        close = close_arr[i]
        prev_close = close_arr[i - 1]
        dt = dates[i]

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

                level = LiquidityLevel(price=sh.price, kind="buy_side",
                                       strength=1, formed_at=now)
                conf_score, conf_factors = score_confluence(
                    df, i, "short", level, atr, fvgs, obs, rsi,
                    ema_fast, ema_slow, pip_size, **conf_kw)

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

                level = LiquidityLevel(price=sl_point.price, kind="sell_side",
                                       strength=1, formed_at=now)
                conf_score, conf_factors = score_confluence(
                    df, i, "long", level, atr, fvgs, obs, rsi,
                    ema_fast, ema_slow, pip_size, **conf_kw)

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

    # Pre-extract arrays
    high_arr = df["High"].values
    low_arr = df["Low"].values
    close_arr = df["Close"].values
    open_arr = df["Open"].values
    atr_arr = atr.values
    rsi_arr = rsi.values
    ema_f_arr = ema_fast.values
    ema_s_arr = ema_slow.values
    fvg_indices = _build_fvg_index(fvgs)
    ob_indices = _build_ob_index(obs)
    dates = df.index

    conf_kw = dict(_close_arr=close_arr, _open_arr=open_arr,
                   _atr_arr=atr_arr, _rsi_arr=rsi_arr,
                   _ema_f_arr=ema_f_arr, _ema_s_arr=ema_s_arr,
                   _fvg_indices=fvg_indices, _ob_indices=ob_indices)

    for level in levels:
        for i in range(2, len(df)):
            high = high_arr[i]
            low = low_arr[i]
            close = close_arr[i]
            open_price = open_arr[i]
            prev_high = high_arr[i - 1]
            prev_low = low_arr[i - 1]
            dt = dates[i]

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
                        ema_fast, ema_slow, pip_size, **conf_kw)

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
                        ema_fast, ema_slow, pip_size, **conf_kw)

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


# ── Common strategy signal generators ──────────────────────────────────────

def detect_ema_crossover(df: pd.DataFrame, pip_size: float = 0.0001,
                         atr: pd.Series = None, fvgs: List[FairValueGap] = None,
                         obs: List[OrderBlock] = None, rsi: pd.Series = None,
                         ema_fast: pd.Series = None, ema_slow: pd.Series = None,
                         min_confluence: int = 0, session_filter: str = "all",
                         rr_ratio: float = 2.0,
                         require_displacement: bool = False,
                         ) -> List[InducementSignal]:
    """Detect EMA crossover signals. Fast EMA crosses above/below slow EMA."""
    signals = []
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
    # Pre-extract values arrays for faster access
    ema_f_vals = ema_fast.values
    ema_s_vals = ema_slow.values
    atr_vals = atr.values
    close_vals = df["Close"].values
    open_vals = df["Open"].values
    rsi_vals = rsi.values
    fvg_indices = _build_fvg_index(fvgs)
    ob_indices = _build_ob_index(obs)
    dates = df.index

    conf_kw = dict(_close_arr=close_vals, _open_arr=open_vals,
                   _atr_arr=atr_vals, _rsi_arr=rsi_vals,
                   _ema_f_arr=ema_f_vals, _ema_s_arr=ema_s_vals,
                   _fvg_indices=fvg_indices, _ob_indices=ob_indices)

    for i in range(2, len(df)):
        if np.isnan(ema_f_vals[i]) or np.isnan(ema_s_vals[i]):
            continue
        if np.isnan(ema_f_vals[i - 1]) or np.isnan(ema_s_vals[i - 1]):
            continue

        dt = dates[i]
        if session_filter != "all":
            session = get_session(dt)
            if session != session_filter and session_filter != "killzones":
                continue
            if session_filter == "killzones" and not is_in_killzone(dt):
                continue

        close = close_vals[i]
        atr_val = atr_vals[i]
        if np.isnan(atr_val) or atr_val == 0:
            continue

        # Bullish crossover: fast crosses above slow
        if ema_f_vals[i - 1] <= ema_s_vals[i - 1] and ema_f_vals[i] > ema_s_vals[i]:
            sl = close - (atr_val * 1.5)
            risk = close - sl
            tp = close + (risk * rr_ratio)

            level = LiquidityLevel(price=close, kind="sell_side", strength=1, formed_at=now)
            conf_score, conf_factors = score_confluence(
                df, i, "long", level, atr, fvgs, obs, rsi,
                ema_fast, ema_slow, pip_size, **conf_kw)
            conf_factors = [f for f in conf_factors if f != "trend_aligned"]
            conf_factors.insert(0, "ema_crossover")

            if conf_score < min_confluence:
                continue

            signals.append(InducementSignal(
                entry_index=i, entry_datetime=dt,
                entry_price=close, direction="long",
                stop_loss=sl, take_profit=tp,
                swept_level=ema_s_vals[i],
                signal_type="ema_crossover",
                confluence_score=conf_score,
                confluence_factors=conf_factors,
                session=get_session(dt),
            ))

        # Bearish crossover: fast crosses below slow
        elif ema_f_vals[i - 1] >= ema_s_vals[i - 1] and ema_f_vals[i] < ema_s_vals[i]:
            sl = close + (atr_val * 1.5)
            risk = sl - close
            tp = close - (risk * rr_ratio)

            level = LiquidityLevel(price=close, kind="buy_side", strength=1, formed_at=now)
            conf_score, conf_factors = score_confluence(
                df, i, "short", level, atr, fvgs, obs, rsi,
                ema_fast, ema_slow, pip_size, **conf_kw)
            conf_factors = [f for f in conf_factors if f != "trend_aligned"]
            conf_factors.insert(0, "ema_crossover")

            if conf_score < min_confluence:
                continue

            signals.append(InducementSignal(
                entry_index=i, entry_datetime=dt,
                entry_price=close, direction="short",
                stop_loss=sl, take_profit=tp,
                swept_level=ema_s_vals[i],
                signal_type="ema_crossover",
                confluence_score=conf_score,
                confluence_factors=conf_factors,
                session=get_session(dt),
            ))

    return signals


def detect_rsi_reversal(df: pd.DataFrame, pip_size: float = 0.0001,
                        atr: pd.Series = None, fvgs: List[FairValueGap] = None,
                        obs: List[OrderBlock] = None, rsi: pd.Series = None,
                        ema_fast: pd.Series = None, ema_slow: pd.Series = None,
                        min_confluence: int = 0, session_filter: str = "all",
                        rr_ratio: float = 2.0,
                        require_displacement: bool = False,
                        rsi_oversold: float = 30.0, rsi_overbought: float = 70.0,
                        ) -> List[InducementSignal]:
    """Detect RSI reversal signals. RSI exits oversold/overbought with confirmation candle."""
    signals = []
    if atr is None:
        atr = calc_atr(df)
    if rsi is None:
        rsi = calc_rsi(df)
    if ema_fast is None:
        ema_fast = calc_ema(df["Close"], 21)
    if ema_slow is None:
        ema_slow = calc_ema(df["Close"], 50)
    if fvgs is None:
        fvgs = []
    if obs is None:
        obs = []

    now = pd.Timestamp.now()
    rsi_vals = rsi.values

    close_vals = df["Close"].values
    open_vals = df["Open"].values
    low_vals = df["Low"].values
    high_vals = df["High"].values
    atr_vals = atr.values
    ema_f_vals = ema_fast.values
    ema_s_vals = ema_slow.values
    fvg_indices = _build_fvg_index(fvgs)
    ob_indices = _build_ob_index(obs)
    dates = df.index

    conf_kw = dict(_close_arr=close_vals, _open_arr=open_vals,
                   _atr_arr=atr_vals, _rsi_arr=rsi_vals,
                   _ema_f_arr=ema_f_vals, _ema_s_arr=ema_s_vals,
                   _fvg_indices=fvg_indices, _ob_indices=ob_indices)

    for i in range(2, len(df)):
        if np.isnan(rsi_vals[i]) or np.isnan(rsi_vals[i - 1]):
            continue

        dt = dates[i]
        if session_filter != "all":
            session = get_session(dt)
            if session != session_filter and session_filter != "killzones":
                continue
            if session_filter == "killzones" and not is_in_killzone(dt):
                continue

        close = close_vals[i]
        open_price = open_vals[i]
        atr_val = atr_vals[i]
        if np.isnan(atr_val) or atr_val == 0:
            continue

        # Bullish: RSI was oversold, now crossing back up + bullish candle
        if (rsi_vals[i - 1] < rsi_oversold and rsi_vals[i] >= rsi_oversold
                and close > open_price):
            sl = low_vals[i] - (5 * pip_size)
            risk = close - sl
            if risk <= 0:
                continue
            tp = close + (risk * rr_ratio)

            level = LiquidityLevel(price=low_vals[i], kind="sell_side",
                                   strength=1, formed_at=now)
            conf_score, conf_factors = score_confluence(
                df, i, "long", level, atr, fvgs, obs, rsi,
                ema_fast, ema_slow, pip_size, **conf_kw)
            conf_factors.insert(0, "rsi_reversal")

            if conf_score < min_confluence:
                continue

            signals.append(InducementSignal(
                entry_index=i, entry_datetime=dt,
                entry_price=close, direction="long",
                stop_loss=sl, take_profit=tp,
                swept_level=low_vals[i],
                signal_type="rsi_reversal",
                confluence_score=conf_score,
                confluence_factors=conf_factors,
                session=get_session(dt),
            ))

        # Bearish: RSI was overbought, now crossing back down + bearish candle
        elif (rsi_vals[i - 1] > rsi_overbought and rsi_vals[i] <= rsi_overbought
              and close < open_price):
            sl = high_vals[i] + (5 * pip_size)
            risk = sl - close
            if risk <= 0:
                continue
            tp = close - (risk * rr_ratio)

            level = LiquidityLevel(price=high_vals[i], kind="buy_side",
                                   strength=1, formed_at=now)
            conf_score, conf_factors = score_confluence(
                df, i, "short", level, atr, fvgs, obs, rsi,
                ema_fast, ema_slow, pip_size, **conf_kw)
            conf_factors.insert(0, "rsi_reversal")

            if conf_score < min_confluence:
                continue

            signals.append(InducementSignal(
                entry_index=i, entry_datetime=dt,
                entry_price=close, direction="short",
                stop_loss=sl, take_profit=tp,
                swept_level=high_vals[i],
                signal_type="rsi_reversal",
                confluence_score=conf_score,
                confluence_factors=conf_factors,
                session=get_session(dt),
            ))

    return signals


def detect_breakout(df: pd.DataFrame, swings: List[SwingPoint],
                    pip_size: float = 0.0001,
                    atr: pd.Series = None, fvgs: List[FairValueGap] = None,
                    obs: List[OrderBlock] = None, rsi: pd.Series = None,
                    ema_fast: pd.Series = None, ema_slow: pd.Series = None,
                    min_confluence: int = 0, session_filter: str = "all",
                    rr_ratio: float = 2.0,
                    require_displacement: bool = False,
                    ) -> List[InducementSignal]:
    """Detect breakout signals. Close above swing high or below swing low with momentum."""
    signals = []
    if atr is None:
        atr = calc_atr(df)
    if rsi is None:
        rsi = calc_rsi(df)
    if ema_fast is None:
        ema_fast = calc_ema(df["Close"], 21)
    if ema_slow is None:
        ema_slow = calc_ema(df["Close"], 50)
    if fvgs is None:
        fvgs = []
    if obs is None:
        obs = []

    swing_highs = [s for s in swings if s.kind == "high"]
    swing_lows = [s for s in swings if s.kind == "low"]
    now = pd.Timestamp.now()

    # Pre-extract arrays
    close_arr = df["Close"].values
    open_arr = df["Open"].values
    atr_arr = atr.values
    rsi_arr = rsi.values
    ema_f_arr = ema_fast.values
    ema_s_arr = ema_slow.values
    fvg_indices = _build_fvg_index(fvgs)
    ob_indices = _build_ob_index(obs)
    dates = df.index

    conf_kw = dict(_close_arr=close_arr, _open_arr=open_arr,
                   _atr_arr=atr_arr, _rsi_arr=rsi_arr,
                   _ema_f_arr=ema_f_arr, _ema_s_arr=ema_s_arr,
                   _fvg_indices=fvg_indices, _ob_indices=ob_indices)

    for i in range(2, len(df)):
        dt = dates[i]
        if session_filter != "all":
            session = get_session(dt)
            if session != session_filter and session_filter != "killzones":
                continue
            if session_filter == "killzones" and not is_in_killzone(dt):
                continue

        close = close_arr[i]
        prev_close = close_arr[i - 1]
        atr_val = atr_arr[i]
        if np.isnan(atr_val) or atr_val == 0:
            continue

        body = abs(close - open_arr[i])

        # Bullish breakout: close above recent swing high with strong candle
        for sh in swing_highs:
            if sh.index < i - 30 or sh.index >= i - 1:
                continue
            if prev_close <= sh.price and close > sh.price and body > atr_val * 0.8:
                sl = sh.price - (atr_val * 0.5)
                risk = close - sl
                if risk <= 0:
                    continue
                tp = close + (risk * rr_ratio)

                level = LiquidityLevel(price=sh.price, kind="buy_side",
                                       strength=1, formed_at=now)
                conf_score, conf_factors = score_confluence(
                    df, i, "long", level, atr, fvgs, obs, rsi,
                    ema_fast, ema_slow, pip_size, **conf_kw)
                conf_factors.insert(0, "breakout")

                if conf_score < min_confluence:
                    continue

                signals.append(InducementSignal(
                    entry_index=i, entry_datetime=dt,
                    entry_price=close, direction="long",
                    stop_loss=sl, take_profit=tp,
                    swept_level=sh.price,
                    signal_type="breakout",
                    confluence_score=conf_score,
                    confluence_factors=conf_factors,
                    session=get_session(dt),
                ))
                break

        # Bearish breakout: close below recent swing low
        for sl_point in swing_lows:
            if sl_point.index < i - 30 or sl_point.index >= i - 1:
                continue
            if prev_close >= sl_point.price and close < sl_point.price and body > atr_val * 0.8:
                sl = sl_point.price + (atr_val * 0.5)
                risk = sl - close
                if risk <= 0:
                    continue
                tp = close - (risk * rr_ratio)

                level = LiquidityLevel(price=sl_point.price, kind="sell_side",
                                       strength=1, formed_at=now)
                conf_score, conf_factors = score_confluence(
                    df, i, "short", level, atr, fvgs, obs, rsi,
                    ema_fast, ema_slow, pip_size, **conf_kw)
                conf_factors.insert(0, "breakout")

                if conf_score < min_confluence:
                    continue

                signals.append(InducementSignal(
                    entry_index=i, entry_datetime=dt,
                    entry_price=close, direction="short",
                    stop_loss=sl, take_profit=tp,
                    swept_level=sl_point.price,
                    signal_type="breakout",
                    confluence_score=conf_score,
                    confluence_factors=conf_factors,
                    session=get_session(dt),
                ))
                break

    return signals


def detect_fvg_entry(df: pd.DataFrame, pip_size: float = 0.0001,
                     atr: pd.Series = None, fvgs: List[FairValueGap] = None,
                     obs: List[OrderBlock] = None, rsi: pd.Series = None,
                     ema_fast: pd.Series = None, ema_slow: pd.Series = None,
                     min_confluence: int = 0, session_filter: str = "all",
                     rr_ratio: float = 2.0,
                     require_displacement: bool = False,
                     ) -> List[InducementSignal]:
    """Detect FVG fill entries. Price retraces into a Fair Value Gap and reverses."""
    signals = []
    if atr is None:
        atr = calc_atr(df)
    if fvgs is None:
        fvgs = find_fvgs(df, pip_size=pip_size)
    if rsi is None:
        rsi = calc_rsi(df)
    if ema_fast is None:
        ema_fast = calc_ema(df["Close"], 21)
    if ema_slow is None:
        ema_slow = calc_ema(df["Close"], 50)
    if obs is None:
        obs = []

    now = pd.Timestamp.now()

    # Pre-extract arrays
    close_arr = df["Close"].values
    open_arr = df["Open"].values
    low_arr = df["Low"].values
    high_arr = df["High"].values
    atr_arr = atr.values
    rsi_arr = rsi.values
    ema_f_arr = ema_fast.values
    ema_s_arr = ema_slow.values
    fvg_indices = _build_fvg_index(fvgs)
    ob_indices = _build_ob_index(obs)
    dates = df.index

    conf_kw = dict(_close_arr=close_arr, _open_arr=open_arr,
                   _atr_arr=atr_arr, _rsi_arr=rsi_arr,
                   _ema_f_arr=ema_f_arr, _ema_s_arr=ema_s_arr,
                   _fvg_indices=fvg_indices, _ob_indices=ob_indices)

    for fvg in fvgs:
        for i in range(fvg.index + 2, min(fvg.index + 20, len(df))):
            dt = dates[i]
            if session_filter != "all":
                session = get_session(dt)
                if session != session_filter and session_filter != "killzones":
                    continue
                if session_filter == "killzones" and not is_in_killzone(dt):
                    continue

            close = close_arr[i]
            open_price = open_arr[i]
            atr_val = atr_arr[i]
            if np.isnan(atr_val) or atr_val == 0:
                continue

            if fvg.direction == "bullish":
                if low_arr[i] <= fvg.top and close >= fvg.bottom and close > open_price:
                    sl = fvg.bottom - (5 * pip_size)
                    risk = close - sl
                    if risk <= 0:
                        continue
                    tp = close + (risk * rr_ratio)

                    level = LiquidityLevel(price=fvg.bottom, kind="sell_side",
                                           strength=1, formed_at=now)
                    conf_score, conf_factors = score_confluence(
                        df, i, "long", level, atr, fvgs, obs, rsi,
                        ema_fast, ema_slow, pip_size, **conf_kw)
                    conf_factors.insert(0, "fvg_entry")

                    if conf_score < min_confluence:
                        continue

                    signals.append(InducementSignal(
                        entry_index=i, entry_datetime=dt,
                        entry_price=close, direction="long",
                        stop_loss=sl, take_profit=tp,
                        swept_level=fvg.bottom,
                        signal_type="fvg_entry",
                        confluence_score=conf_score,
                        confluence_factors=conf_factors,
                        session=get_session(dt),
                    ))
                    break

            elif fvg.direction == "bearish":
                if high_arr[i] >= fvg.bottom and close <= fvg.top and close < open_price:
                    sl = fvg.top + (5 * pip_size)
                    risk = sl - close
                    if risk <= 0:
                        continue
                    tp = close - (risk * rr_ratio)

                    level = LiquidityLevel(price=fvg.top, kind="buy_side",
                                           strength=1, formed_at=now)
                    conf_score, conf_factors = score_confluence(
                        df, i, "short", level, atr, fvgs, obs, rsi,
                        ema_fast, ema_slow, pip_size, **conf_kw)
                    conf_factors.insert(0, "fvg_entry")

                    if conf_score < min_confluence:
                        continue

                    signals.append(InducementSignal(
                        entry_index=i, entry_datetime=dt,
                        entry_price=close, direction="short",
                        stop_loss=sl, take_profit=tp,
                        swept_level=fvg.top,
                        signal_type="fvg_entry",
                        confluence_score=conf_score,
                        confluence_factors=conf_factors,
                        session=get_session(dt),
                    ))
                    break

    return signals


def detect_ob_bounce(df: pd.DataFrame, pip_size: float = 0.0001,
                     atr: pd.Series = None, fvgs: List[FairValueGap] = None,
                     obs: List[OrderBlock] = None, rsi: pd.Series = None,
                     ema_fast: pd.Series = None, ema_slow: pd.Series = None,
                     min_confluence: int = 0, session_filter: str = "all",
                     rr_ratio: float = 2.0,
                     require_displacement: bool = False,
                     ) -> List[InducementSignal]:
    """Detect Order Block bounce entries. Price returns to an OB zone and reverses."""
    signals = []
    if atr is None:
        atr = calc_atr(df)
    if obs is None:
        obs = find_order_blocks(df, pip_size=pip_size)
    if rsi is None:
        rsi = calc_rsi(df)
    if ema_fast is None:
        ema_fast = calc_ema(df["Close"], 21)
    if ema_slow is None:
        ema_slow = calc_ema(df["Close"], 50)
    if fvgs is None:
        fvgs = []

    now = pd.Timestamp.now()

    # Pre-extract arrays
    close_arr = df["Close"].values
    open_arr = df["Open"].values
    low_arr = df["Low"].values
    high_arr = df["High"].values
    atr_arr = atr.values
    rsi_arr = rsi.values
    ema_f_arr = ema_fast.values
    ema_s_arr = ema_slow.values
    fvg_indices = _build_fvg_index(fvgs)
    ob_indices = _build_ob_index(obs)
    dates = df.index

    conf_kw = dict(_close_arr=close_arr, _open_arr=open_arr,
                   _atr_arr=atr_arr, _rsi_arr=rsi_arr,
                   _ema_f_arr=ema_f_arr, _ema_s_arr=ema_s_arr,
                   _fvg_indices=fvg_indices, _ob_indices=ob_indices)

    for ob in obs:
        if ob.mitigated:
            continue
        for i in range(ob.index + 2, min(ob.index + 30, len(df))):
            dt = dates[i]
            if session_filter != "all":
                session = get_session(dt)
                if session != session_filter and session_filter != "killzones":
                    continue
                if session_filter == "killzones" and not is_in_killzone(dt):
                    continue

            close = close_arr[i]
            open_price = open_arr[i]
            atr_val = atr_arr[i]
            if np.isnan(atr_val) or atr_val == 0:
                continue

            if ob.direction == "bullish":
                if low_arr[i] <= ob.high and close >= ob.low and close > open_price:
                    sl = ob.low - (5 * pip_size)
                    risk = close - sl
                    if risk <= 0:
                        continue
                    tp = close + (risk * rr_ratio)

                    level = LiquidityLevel(price=ob.low, kind="sell_side",
                                           strength=1, formed_at=now)
                    conf_score, conf_factors = score_confluence(
                        df, i, "long", level, atr, fvgs, obs, rsi,
                        ema_fast, ema_slow, pip_size, **conf_kw)
                    conf_factors.insert(0, "ob_bounce")

                    if conf_score < min_confluence:
                        continue

                    signals.append(InducementSignal(
                        entry_index=i, entry_datetime=dt,
                        entry_price=close, direction="long",
                        stop_loss=sl, take_profit=tp,
                        swept_level=ob.low,
                        signal_type="ob_bounce",
                        confluence_score=conf_score,
                        confluence_factors=conf_factors,
                        session=get_session(dt),
                    ))
                    ob.mitigated = True
                    break

            elif ob.direction == "bearish":
                if high_arr[i] >= ob.low and close <= ob.high and close < open_price:
                    sl = ob.high + (5 * pip_size)
                    risk = sl - close
                    if risk <= 0:
                        continue
                    tp = close - (risk * rr_ratio)

                    level = LiquidityLevel(price=ob.high, kind="buy_side",
                                           strength=1, formed_at=now)
                    conf_score, conf_factors = score_confluence(
                        df, i, "short", level, atr, fvgs, obs, rsi,
                        ema_fast, ema_slow, pip_size, **conf_kw)
                    conf_factors.insert(0, "ob_bounce")

                    if conf_score < min_confluence:
                        continue

                    signals.append(InducementSignal(
                        entry_index=i, entry_datetime=dt,
                        entry_price=close, direction="short",
                        stop_loss=sl, take_profit=tp,
                        swept_level=ob.high,
                        signal_type="ob_bounce",
                        confluence_score=conf_score,
                        confluence_factors=conf_factors,
                        session=get_session(dt),
                    ))
                    ob.mitigated = True
                    break

    return signals


def get_pip_size(pair_name: str) -> float:
    """Return pip size for a currency pair."""
    jpy_pairs = ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY", "CAD/JPY", "NZD/JPY",
                 "CHF/JPY"]
    if pair_name in jpy_pairs:
        return 0.01
    return 0.0001

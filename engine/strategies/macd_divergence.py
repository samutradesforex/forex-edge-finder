"""MACD Divergence strategy — detects bullish/bearish MACD divergence signals."""

import numpy as np
import pandas as pd

from engine.strategies import register_strategy
from engine.liquidity import (
    InducementSignal,
    LiquidityLevel,
    calc_atr,
    calc_ema,
    calc_rsi,
    get_session,
    is_in_killzone,
    score_confluence,
    _build_fvg_index,
    _build_ob_index,
)


def _calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _find_swing_lows(values, i, lookback=5):
    """Find the two most recent swing lows before index i.

    A swing low is a point where the value is lower than the surrounding
    ``lookback`` bars on each side.  Returns a list of (index, value) tuples
    for the two most recent swing lows, ordered oldest-first.
    """
    lows = []
    # Need at least lookback bars on each side to confirm a swing
    start = max(lookback, 0)
    end = i - lookback  # last bar that can have lookback bars after it
    for j in range(end, start - 1, -1):
        val = values[j]
        if np.isnan(val):
            continue
        window = values[max(j - lookback, 0): j + lookback + 1]
        if np.any(np.isnan(window)):
            continue
        if val <= np.nanmin(window):
            lows.append((j, val))
            if len(lows) == 2:
                break
    if len(lows) == 2:
        lows.reverse()  # oldest first
    return lows


def _find_swing_highs(values, i, lookback=5):
    """Find the two most recent swing highs before index i.

    Same logic as ``_find_swing_lows`` but for highs.  Returns oldest-first.
    """
    highs = []
    start = max(lookback, 0)
    end = i - lookback
    for j in range(end, start - 1, -1):
        val = values[j]
        if np.isnan(val):
            continue
        window = values[max(j - lookback, 0): j + lookback + 1]
        if np.any(np.isnan(window)):
            continue
        if val >= np.nanmax(window):
            highs.append((j, val))
            if len(highs) == 2:
                break
    if len(highs) == 2:
        highs.reverse()  # oldest first
    return highs


def detect_macd_divergence(df, pip_size, atr, fvgs=None, obs=None, rsi=None, ema_fast=None, ema_slow=None,
                            require_displacement=False, min_confluence=0, session_filter="all", rr_ratio=2.0,
                            **kwargs):
    """Detect MACD divergence signals.

    Bullish divergence: price makes lower lows but MACD makes higher lows.
    Bearish divergence: price makes higher highs but MACD makes lower highs.

    Returns a list of :class:`InducementSignal` objects.
    """
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

    # Compute MACD
    macd_line, signal_line, histogram = _calc_macd(df["Close"], fast=12, slow=26, signal=9)

    now = pd.Timestamp.now()

    # Pre-extract value arrays for faster access
    close_vals = df["Close"].values
    low_vals = df["Low"].values
    high_vals = df["High"].values
    open_vals = df["Open"].values
    atr_vals = atr.values
    rsi_vals = rsi.values
    ema_f_vals = ema_fast.values
    ema_s_vals = ema_slow.values
    macd_vals = macd_line.values
    fvg_indices = _build_fvg_index(fvgs)
    ob_indices = _build_ob_index(obs)
    dates = df.index

    conf_kw = dict(_close_arr=close_vals, _open_arr=open_vals,
                   _atr_arr=atr_vals, _rsi_arr=rsi_vals,
                   _ema_f_arr=ema_f_vals, _ema_s_arr=ema_s_vals,
                   _fvg_indices=fvg_indices, _ob_indices=ob_indices)

    # We need enough bars for MACD to stabilize and to find swing points
    min_bars = 35  # 26 (slow EMA) + 9 (signal) buffer
    lookback = 5

    for i in range(min_bars, len(df)):
        dt = dates[i]

        # Session filter
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
        if np.isnan(macd_vals[i]):
            continue

        # --- Bullish divergence: price lower lows, MACD higher lows ---
        price_lows = _find_swing_lows(low_vals, i, lookback)
        macd_lows = _find_swing_lows(macd_vals, i, lookback)

        if len(price_lows) == 2 and len(macd_lows) == 2:
            price_ll = price_lows[1][1] < price_lows[0][1]  # price made lower low
            macd_hl = macd_lows[1][1] > macd_lows[0][1]     # MACD made higher low

            if price_ll and macd_hl:
                sl = close - (atr_val * 1.5)
                risk = close - sl
                tp = close + (risk * rr_ratio)

                level = LiquidityLevel(price=close, kind="sell_side", strength=1, formed_at=now)
                conf_score, conf_factors = score_confluence(
                    df, i, "long", level, atr, fvgs, obs, rsi,
                    ema_fast, ema_slow, pip_size, **conf_kw)

                # Add the divergence factor itself
                conf_factors.insert(0, "macd_bullish_divergence")
                conf_score += 1

                if conf_score < min_confluence:
                    pass  # skip below
                else:
                    signals.append(InducementSignal(
                        entry_index=i, entry_datetime=dt,
                        entry_price=close, direction="long",
                        stop_loss=sl, take_profit=tp,
                        swept_level=price_lows[1][1],
                        signal_type="macd_divergence",
                        confluence_score=conf_score,
                        confluence_factors=conf_factors,
                        session=get_session(dt),
                    ))
                    continue  # don't check bearish on the same bar

        # --- Bearish divergence: price higher highs, MACD lower highs ---
        price_highs = _find_swing_highs(high_vals, i, lookback)
        macd_highs = _find_swing_highs(macd_vals, i, lookback)

        if len(price_highs) == 2 and len(macd_highs) == 2:
            price_hh = price_highs[1][1] > price_highs[0][1]  # price made higher high
            macd_lh = macd_highs[1][1] < macd_highs[0][1]     # MACD made lower high

            if price_hh and macd_lh:
                sl = close + (atr_val * 1.5)
                risk = sl - close
                tp = close - (risk * rr_ratio)

                level = LiquidityLevel(price=close, kind="buy_side", strength=1, formed_at=now)
                conf_score, conf_factors = score_confluence(
                    df, i, "short", level, atr, fvgs, obs, rsi,
                    ema_fast, ema_slow, pip_size, **conf_kw)

                # Add the divergence factor itself
                conf_factors.insert(0, "macd_bearish_divergence")
                conf_score += 1

                if conf_score >= min_confluence:
                    signals.append(InducementSignal(
                        entry_index=i, entry_datetime=dt,
                        entry_price=close, direction="short",
                        stop_loss=sl, take_profit=tp,
                        swept_level=price_highs[1][1],
                        signal_type="macd_divergence",
                        confluence_score=conf_score,
                        confluence_factors=conf_factors,
                        session=get_session(dt),
                    ))

    return signals


register_strategy(
    name="macd_divergence",
    display_name="MACD Divergence",
    category="momentum",
    relevant_params=["rr_ratio", "min_confluence"],
    description="Detects bullish and bearish MACD divergence with confluence scoring",
)(detect_macd_divergence)

"""Bollinger Band Reversion strategy — mean-reversion entries at band extremes."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

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


def _calc_bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (middle, upper, lower) Bollinger Bands."""
    middle = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper = middle + (rolling_std * std_dev)
    lower = middle - (rolling_std * std_dev)
    return middle, upper, lower


@register_strategy(
    name="bollinger_bands",
    display_name="Bollinger Band Reversion",
    category="reversal",
    relevant_params=["rr_ratio", "min_confluence"],
    description="Enters on Bollinger Band mean-reversion when price re-enters bands after closing outside",
)
def detect_bollinger_bands(
    df: pd.DataFrame,
    pip_size: float = 0.0001,
    atr: pd.Series = None,
    fvgs: List[FairValueGap] = None,
    obs: List[OrderBlock] = None,
    rsi: pd.Series = None,
    ema_fast: pd.Series = None,
    ema_slow: pd.Series = None,
    require_displacement: bool = False,
    min_confluence: int = 0,
    session_filter: str = "all",
    rr_ratio: float = 2.0,
    **kwargs,
) -> List[InducementSignal]:
    """Detect Bollinger Band mean-reversion signals.

    Long:  price closes below lower band (oversold), then next candle closes
           back inside the bands -> entry.
    Short: price closes above upper band (overbought), then next candle closes
           back inside the bands -> entry.
    """
    signals: List[InducementSignal] = []

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

    # Compute Bollinger Bands
    middle, upper, lower = _calc_bollinger_bands(df["Close"], period=20, std_dev=2.0)

    now = pd.Timestamp.now()

    # Pre-extract value arrays for fast access
    close_vals = df["Close"].values
    open_vals = df["Open"].values
    atr_vals = atr.values
    rsi_vals = rsi.values
    ema_f_vals = ema_fast.values
    ema_s_vals = ema_slow.values
    upper_vals = upper.values
    lower_vals = lower.values
    middle_vals = middle.values
    dates = df.index

    fvg_indices = _build_fvg_index(fvgs)
    ob_indices = _build_ob_index(obs)

    conf_kw = dict(
        _close_arr=close_vals,
        _open_arr=open_vals,
        _atr_arr=atr_vals,
        _rsi_arr=rsi_vals,
        _ema_f_arr=ema_f_vals,
        _ema_s_arr=ema_s_vals,
        _fvg_indices=fvg_indices,
        _ob_indices=ob_indices,
    )

    # Need at least index 1 for the previous candle check; start at the
    # Bollinger period so that all band values are valid.
    start = max(2, 20)
    for i in range(start, len(df)):
        if np.isnan(upper_vals[i]) or np.isnan(lower_vals[i]):
            continue
        if np.isnan(upper_vals[i - 1]) or np.isnan(lower_vals[i - 1]):
            continue

        dt = dates[i]

        # Session filtering
        if session_filter != "all":
            session = get_session(dt)
            if session != session_filter and session_filter != "killzones":
                continue
            if session_filter == "killzones" and not is_in_killzone(dt):
                continue

        close_cur = close_vals[i]
        close_prev = close_vals[i - 1]
        atr_val = atr_vals[i]
        if np.isnan(atr_val) or atr_val == 0:
            continue

        # ── Long: prev candle closed below lower band, current closes back inside ──
        if close_prev < lower_vals[i - 1] and close_cur >= lower_vals[i]:
            sl = close_cur - (atr_val * 1.5)
            risk = close_cur - sl
            tp = close_cur + (risk * rr_ratio)

            level = LiquidityLevel(
                price=lower_vals[i], kind="sell_side", strength=1, formed_at=now
            )
            conf_score, conf_factors = score_confluence(
                df, i, "long", level, atr, fvgs, obs, rsi,
                ema_fast, ema_slow, pip_size, **conf_kw,
            )

            # Add Bollinger-specific factor
            conf_factors.insert(0, "bb_lower_reversion")
            conf_score += 1

            if conf_score < min_confluence:
                continue

            signals.append(InducementSignal(
                entry_index=i,
                entry_datetime=dt,
                entry_price=close_cur,
                direction="long",
                stop_loss=sl,
                take_profit=tp,
                swept_level=lower_vals[i],
                signal_type="bollinger_bands",
                confluence_score=conf_score,
                confluence_factors=conf_factors,
                session=get_session(dt),
            ))

        # ── Short: prev candle closed above upper band, current closes back inside ──
        elif close_prev > upper_vals[i - 1] and close_cur <= upper_vals[i]:
            sl = close_cur + (atr_val * 1.5)
            risk = sl - close_cur
            tp = close_cur - (risk * rr_ratio)

            level = LiquidityLevel(
                price=upper_vals[i], kind="buy_side", strength=1, formed_at=now
            )
            conf_score, conf_factors = score_confluence(
                df, i, "short", level, atr, fvgs, obs, rsi,
                ema_fast, ema_slow, pip_size, **conf_kw,
            )

            # Add Bollinger-specific factor
            conf_factors.insert(0, "bb_upper_reversion")
            conf_score += 1

            if conf_score < min_confluence:
                continue

            signals.append(InducementSignal(
                entry_index=i,
                entry_datetime=dt,
                entry_price=close_cur,
                direction="short",
                stop_loss=sl,
                take_profit=tp,
                swept_level=upper_vals[i],
                signal_type="bollinger_bands",
                confluence_score=conf_score,
                confluence_factors=conf_factors,
                session=get_session(dt),
            ))

    return signals

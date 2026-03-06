"""Swing Breakout strategy — enters on break of significant swing levels."""

from engine.strategies import register_strategy
from engine.liquidity import detect_breakout as _detect

register_strategy(
    name="breakout",
    display_name="Swing Breakout",
    category="structure",
    relevant_params=["swing_lookback", "rr_ratio", "min_confluence"],
    description="Enters on break above/below significant swing high/low",
)(_detect)

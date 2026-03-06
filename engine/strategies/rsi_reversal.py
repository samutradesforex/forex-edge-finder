"""RSI Reversal strategy — entries at RSI extremes (30/70)."""

from engine.strategies import register_strategy
from engine.liquidity import detect_rsi_reversal as _detect

register_strategy(
    name="rsi_reversal",
    display_name="RSI Reversal (30/70)",
    category="reversal",
    relevant_params=["rr_ratio", "min_confluence"],
    description="Enters on RSI oversold/overbought reversals with confluence",
)(_detect)

"""Inducement Trap strategy — detects minor structure breaks that lure traders."""

from engine.strategies import register_strategy
from engine.liquidity import detect_inducement_traps as _detect

register_strategy(
    name="inducement",
    display_name="Inducement Traps",
    category="smc",
    relevant_params=["swing_lookback", "rr_ratio", "min_confluence"],
    description="Identifies minor structure breaks designed to trap retail traders",
)(_detect)

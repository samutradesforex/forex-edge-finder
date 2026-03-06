"""Liquidity Sweep strategy — detects price wicks beyond key liquidity levels."""

from engine.strategies import register_strategy
from engine.liquidity import detect_liquidity_sweeps as _detect

register_strategy(
    name="sweeps",
    display_name="Liquidity Sweeps",
    category="smc",
    relevant_params=["swing_lookback", "cluster_pips", "min_wick_pips",
                     "rr_ratio", "min_confluence"],
    description="Detects price wicks beyond key liquidity levels before reversal",
)(_detect)

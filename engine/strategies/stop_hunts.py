"""Stop Hunt strategy — detects false breakouts that grab liquidity."""

from engine.strategies import register_strategy
from engine.liquidity import detect_stop_hunts as _detect

register_strategy(
    name="stop_hunts",
    display_name="Stop Hunts",
    category="smc",
    relevant_params=["swing_lookback", "cluster_pips", "rr_ratio", "min_confluence"],
    description="False breakouts beyond key levels that grab stop-loss liquidity",
)(_detect)

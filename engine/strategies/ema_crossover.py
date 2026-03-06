"""EMA Crossover strategy — 21/50 EMA trend-following entries."""

from engine.strategies import register_strategy
from engine.liquidity import detect_ema_crossover as _detect

register_strategy(
    name="ema_crossover",
    display_name="EMA Crossover (21/50)",
    category="trend",
    relevant_params=["rr_ratio", "min_confluence"],
    description="Enters on 21/50 EMA crossover with trend confirmation",
)(_detect)

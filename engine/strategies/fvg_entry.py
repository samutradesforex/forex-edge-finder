"""FVG Fill Entry strategy — enters when price fills a Fair Value Gap."""

from engine.strategies import register_strategy
from engine.liquidity import detect_fvg_entry as _detect

register_strategy(
    name="fvg_entry",
    display_name="FVG Fill Entry",
    category="structure",
    relevant_params=["rr_ratio", "min_confluence"],
    description="Enters when price retraces to fill a Fair Value Gap (imbalance)",
)(_detect)

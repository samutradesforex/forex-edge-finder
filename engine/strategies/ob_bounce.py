"""Order Block Bounce strategy — enters on reaction from order block zones."""

from engine.strategies import register_strategy
from engine.liquidity import detect_ob_bounce as _detect

register_strategy(
    name="ob_bounce",
    display_name="Order Block Bounce",
    category="structure",
    relevant_params=["rr_ratio", "min_confluence"],
    description="Enters on price bounce from demand/supply order block zones",
)(_detect)

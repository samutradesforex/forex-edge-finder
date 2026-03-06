"""Strategy Registry — plugin architecture for backtesting strategies.

Add a new strategy by creating a file in this package with a class that
inherits from ``Strategy``.  The ``@register`` decorator (or calling
``StrategyRegistry.register``) makes it available everywhere automatically.

Usage::

    from engine.strategies import registry

    # List all strategies
    registry.list()          # -> ["sweeps", "inducement", ...]

    # Get a strategy's detect function
    fn = registry.get("sweeps").detect

    # Iterate
    for name, strat in registry.items():
        ...
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type


# ── Base strategy class ───────────────────────────────────────────────────

@dataclass
class StrategyMeta:
    """Metadata for a registered strategy."""
    name: str
    display_name: str
    category: str  # "smc", "trend", "reversal", "structure"
    relevant_params: List[str]
    detect: Callable  # The actual signal-detection function
    description: str = ""


class StrategyRegistry:
    """Central registry for all strategies."""

    _strategies: Dict[str, StrategyMeta] = {}

    # Strategy groupings for the "smc", "both", "all" meta-selectors
    _groups: Dict[str, List[str]] = {
        "smc": ["sweeps", "inducement", "stop_hunts"],
        "both": ["sweeps", "inducement"],
    }

    @classmethod
    def register(cls, meta: StrategyMeta) -> None:
        cls._strategies[meta.name] = meta

    @classmethod
    def get(cls, name: str) -> Optional[StrategyMeta]:
        return cls._strategies.get(name)

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._strategies.keys())

    @classmethod
    def items(cls):
        return cls._strategies.items()

    @classmethod
    def all_metas(cls) -> List[StrategyMeta]:
        return list(cls._strategies.values())

    @classmethod
    def resolve_names(cls, selector: str) -> List[str]:
        """Resolve a selector like 'all', 'smc', or a single name to a list."""
        if selector == "all":
            return cls.list()
        if selector in cls._groups:
            return cls._groups[selector]
        if selector in cls._strategies:
            return [selector]
        return [selector]

    @classmethod
    def display_options(cls) -> Dict[str, str]:
        """Return {value: label} for UI selectboxes, including group options."""
        opts = {
            "all": "All Strategies",
            "smc": "SMC Only (Sweeps + Inducement + Stop Hunts)",
            "both": "Sweeps + Inducement",
        }
        for name, meta in cls._strategies.items():
            opts[name] = meta.display_name
        return opts

    @classmethod
    def relevant_params_for(cls, strategy_name: str) -> List[str]:
        """Get the params that actually matter for a strategy."""
        meta = cls._strategies.get(strategy_name)
        if meta:
            return meta.relevant_params
        # Default: all params
        return ["swing_lookback", "cluster_pips", "min_wick_pips",
                "rr_ratio", "min_confluence"]

    @classmethod
    def categories(cls) -> Dict[str, List[str]]:
        """Group strategies by category."""
        cats: Dict[str, List[str]] = {}
        for name, meta in cls._strategies.items():
            cats.setdefault(meta.category, []).append(name)
        return cats


# Convenience alias
registry = StrategyRegistry


def register_strategy(
    name: str,
    display_name: str,
    category: str,
    relevant_params: List[str],
    description: str = "",
):
    """Decorator to register a detect function as a strategy."""
    def decorator(fn: Callable) -> Callable:
        registry.register(StrategyMeta(
            name=name,
            display_name=display_name,
            category=category,
            relevant_params=relevant_params,
            detect=fn,
            description=description,
        ))
        return fn
    return decorator


# ── Auto-discover strategy modules ───────────────────────────────────────

def _auto_discover():
    """Import all modules in this package so their @register decorators fire."""
    package_path = __path__
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        importlib.import_module(f".{module_name}", __package__)


_auto_discover()

"""
EP Communication Strategies.

This module contains all strategy implementations for EP communication,
separated by mode:
- normal_strategy.py: All normal mode strategies
- low_latency_strategy.py: All low latency mode strategies
"""

from ..ep_strategy import (
    EPCommStrategy,
    LowLatencyEPCommStrategy,
    NormalEPCommStrategy,
    get_low_latency_strategy,
    get_normal_strategy,
    register_low_latency_strategy,
    register_normal_strategy,
)
from .low_latency_strategy import (
    DefaultLowLatencyCommStrategy,
    OpsLowLatencyCommStrategy,
)
from .normal_strategy import AlltoAllNormalCommStrategy, DefaultNormalCommStrategy

__all__ = [
    # Base classes
    "EPCommStrategy",
    "NormalEPCommStrategy",
    "LowLatencyEPCommStrategy",
    # Registry functions
    "register_normal_strategy",
    "register_low_latency_strategy",
    "get_normal_strategy",
    "get_low_latency_strategy",
    # Normal strategies
    "DefaultNormalCommStrategy",
    "AlltoAllNormalCommStrategy",
    # Low latency strategies
    "DefaultLowLatencyCommStrategy",
    "OpsLowLatencyCommStrategy",
]

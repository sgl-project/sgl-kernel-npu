"""
EP Communication Strategies.

This module contains all strategy implementations for EP communication,
separated by mode:
- normal_strategy.py: All normal mode strategies
- low_latency_strategy.py: All low latency mode strategies
- fused_strategy.py: All fused MoE strategies
"""

from ..ep_strategy import (
    EPCommStrategy,
    FusedEPCommStrategy,
    LowLatencyEPCommStrategy,
    NormalEPCommStrategy,
    get_fused_strategy,
    get_fused_strategy_for_mode,
    get_low_latency_strategy,
    get_normal_strategy,
    register_fused_strategy,
    register_low_latency_strategy,
    register_normal_strategy,
)
from .fused_strategy import (
    DispatchFFNCombineStrategy,
    FusedDeepMoeStrategy,
    MegaMoeStrategy,
)
from .low_latency_strategy import (
    DefaultLowLatencyCommStrategy,
    OpsLowLatencyCommStrategy,
)
from .normal_strategy import AlltoAllNormalCommStrategy, DefaultNormalCommStrategy

__all__ = [
    # Base classes
    "EPCommStrategy",
    "FusedEPCommStrategy",
    "NormalEPCommStrategy",
    "LowLatencyEPCommStrategy",
    # Registry functions
    "register_normal_strategy",
    "register_low_latency_strategy",
    "register_fused_strategy",
    "get_normal_strategy",
    "get_low_latency_strategy",
    "get_fused_strategy",
    "get_fused_strategy_for_mode",
    # Normal strategies
    "DefaultNormalCommStrategy",
    "AlltoAllNormalCommStrategy",
    # Low latency strategies
    "DefaultLowLatencyCommStrategy",
    "OpsLowLatencyCommStrategy",
    # Fused strategies
    "FusedDeepMoeStrategy",
    "DispatchFFNCombineStrategy",
    "MegaMoeStrategy",
]

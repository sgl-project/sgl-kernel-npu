from abc import ABC, abstractmethod
from typing import Tuple, Callable, Optional, List, Union, Dict, Type
import torch
from .utils import EventOverlap


class EPCommStrategy(ABC):
    """Abstract base class for EP communication strategies"""

    def __init__(self, group_name: str, group_size: int, rank: int):
        self.group_name = group_name
        self.group_size = group_size
        self.rank = rank

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the strategy"""
        pass
    
    @abstractmethod
    def get_supported_modes(self) -> List[str]:
        """Get list of supported modes ['normal', 'low_latency']"""
        pass


class NormalEPCommStrategy(EPCommStrategy):
    """Normal EP communication strategies base class."""

    @abstractmethod
    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap
    ]:
        """get dispatch layout"""
        pass
    
    @abstractmethod
    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple],
        num_tokens_per_rank: Optional[torch.Tensor],
        num_tokens_per_rdma_rank: Optional[torch.Tensor],
        is_token_in_rank: Optional[torch.Tensor],
        num_tokens_per_expert: Optional[torch.Tensor],
        topk_idx: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        expert_alignment: int,
        num_worst_tokens: int,
        config,
        previous_event: Optional[EventOverlap],
        async_finish: bool,
        allocate_on_comm_stream: bool,
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor],
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """Normal dispatch"""
        pass
    
    @abstractmethod
    def combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor],
        bias,
        config,
        previous_event: Optional[EventOverlap],
        async_finish: bool,
        allocate_on_comm_stream: bool,
        combine_send_cost_stats: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """Normal combine"""
        pass


class LowLatencyEPCommStrategy(EPCommStrategy):
    """LowLatency EP communication strategies base class."""
    
    @abstractmethod
    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor],
        use_fp8: bool,
        round_scale: bool,
        use_ue8m0: bool,
        async_finish: bool,
        return_recv_hook: bool,
        topk_weights: Optional[torch.Tensor],
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        Tuple,
        EventOverlap,
        Callable,
    ]:
        """LowLatency dispatch"""
        pass
    
    @abstractmethod
    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        zero_copy: bool,
        async_finish: bool,
        return_recv_hook: bool,
        out: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, EventOverlap, Callable]:
        """LowLatency combine"""
        pass


# ==================== Strategy Registry ====================

# Normal mode strategy registry
_NORMAL_STRATEGY_REGISTRY: Dict[str, Type[NormalEPCommStrategy]] = {}

# Low latency mode strategy registry
_LOW_LATENCY_STRATEGY_REGISTRY: Dict[str, Type[LowLatencyEPCommStrategy]] = {}


def register_normal_strategy(name: str):
    """Decorator to register a normal mode strategy."""
    def decorator(cls: Type[NormalEPCommStrategy]):
        _NORMAL_STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def register_low_latency_strategy(name: str):
    """Decorator to register a low latency mode strategy."""
    def decorator(cls: Type[LowLatencyEPCommStrategy]):
        _LOW_LATENCY_STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def get_normal_strategy(name: str) -> Type[NormalEPCommStrategy]:
    """Get a normal mode strategy class by name."""
    if name not in _NORMAL_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown normal strategy: {name}. Available: {list(_NORMAL_STRATEGY_REGISTRY.keys())}")
    return _NORMAL_STRATEGY_REGISTRY[name]


def get_low_latency_strategy(name: str) -> Type[LowLatencyEPCommStrategy]:
    """Get a low latency mode strategy class by name."""
    if name not in _LOW_LATENCY_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown low latency strategy: {name}. Available: {list(_LOW_LATENCY_STRATEGY_REGISTRY.keys())}")
    return _LOW_LATENCY_STRATEGY_REGISTRY[name]

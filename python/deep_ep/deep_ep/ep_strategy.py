from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist

from .utils import EventOverlap


# Normal mode strategy names
class NormalStrategy:
    DEFAULT = "default"
    ALLTOALL = "alltoall"

    @classmethod
    def get_all_strategies(cls) -> list:
        return [cls.DEFAULT, cls.ALLTOALL]


# Low latency mode strategy names
class LowLatencyStrategy:
    DEFAULT = "default"
    OPS = "ops"
    ALLTOALL = "alltoall"

    @classmethod
    def get_all_strategies(cls) -> list:
        return [cls.DEFAULT, cls.OPS, cls.ALLTOALL]


# Normal mode strategy and Low latency mode strategy
class StrategyMap:
    strategy_map = {
        ("default"): (
            NormalStrategy.DEFAULT,
            LowLatencyStrategy.DEFAULT,
        ),
        ("alltoall"): (
            NormalStrategy.ALLTOALL,
            LowLatencyStrategy.ALLTOALL,
        ),
        ("ops"): (
            NormalStrategy.DEFAULT,
            LowLatencyStrategy.OPS,
        ),
    }

    @classmethod
    def get_strategy(
        cls,
        deep_mode: str,
    ):
        key = deep_mode.lower()

        if key not in cls.strategy_map:
            raise ValueError(
                f"Unsupported mode combination: " f"DEEP_USE_MODE={deep_mode}, "
            )

        return cls.strategy_map[key]


class EPCommStrategy(ABC):
    """Abstract base class for EP communication strategies"""

    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self._group_name = None
        self._group_size = None
        self._rank = None

    @property
    def group_name(self) -> str:
        """Get the group name (lazy initialization)"""
        if self._group_name is None:
            try:
                backend = self.group._get_backend(torch.device("npu"))
                self._group_name = backend.get_hccl_comm_name(self.rank)
            except Exception:
                self._group_name = ""
        return self._group_name

    @property
    def group_size(self) -> int:
        """Get the group size"""
        if self._group_size is None:
            self._group_size = self.group.size()
        return self._group_size

    @property
    def rank(self) -> int:
        """Get the rank"""
        if self._rank is None:
            self._rank = self.group.rank()
        return self._rank

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
        use_mxfp4: bool,
        async_finish: bool,
        return_recv_hook: bool,
        topk_weights: Optional[torch.Tensor],
        quant_mode: Optional[str],
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
        raise ValueError(
            f"Unknown normal strategy: {name}. Available: {list(_NORMAL_STRATEGY_REGISTRY.keys())}"
        )
    return _NORMAL_STRATEGY_REGISTRY[name]


def get_low_latency_strategy(name: str) -> Type[LowLatencyEPCommStrategy]:
    """Get a low latency mode strategy class by name."""
    if name not in _LOW_LATENCY_STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown low latency strategy: {name}. Available: {list(_LOW_LATENCY_STRATEGY_REGISTRY.keys())}"
        )
    return _LOW_LATENCY_STRATEGY_REGISTRY[name]

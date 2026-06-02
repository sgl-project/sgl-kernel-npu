"""
Normal mode EP communication strategies.
All normal mode strategy implementations are in this file.
"""

import os
from typing import Callable, Tuple, Optional, List, Union

import torch
import torch_npu
from deep_ep_cpp import EventHandle

from ..ep_strategy import NormalEPCommStrategy, register_normal_strategy
from ..utils import EventOverlap


@register_normal_strategy("default")
class DefaultNormalCommStrategy(NormalEPCommStrategy):
    """
    Normal mode strategy using Custom operator implementation (deep_ep_cpp).
    This is the default and most optimized implementation for normal mode.
    """
    
    def __init__(self, runtime, group_name: str, group_size: int, rank: int):
        super().__init__(group_name, group_size, rank)
        self.runtime = runtime
    
    def get_name(self) -> str:
        return "custom"
    
    def get_supported_modes(self) -> List[str]:
        return ["normal"]
    
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
        self.num_experts = num_experts

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = self.runtime.get_dispatch_layout(
            topk_idx,
            num_experts,
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
        )
        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            EventOverlap(event),
        )
    
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
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config=None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        # Internode dispatch
        if self.runtime.get_num_rdma_ranks() > 1:
            return self._internode_dispatch(
                x, handle, num_tokens_per_rank, num_tokens_per_rdma_rank,
                is_token_in_rank, num_tokens_per_expert, topk_idx, topk_weights,
                expert_alignment, config, previous_event, async_finish,
                allocate_on_comm_stream,
            )
        
        # Intranode dispatch
        return self._intranode_dispatch(
            x, handle, num_tokens_per_rank, is_token_in_rank,
            num_tokens_per_expert, topk_idx, topk_weights, expert_alignment,
            num_worst_tokens, config, previous_event, async_finish,
            allocate_on_comm_stream, dispatch_wait_recv_cost_stats,
        )
    
    def _intranode_dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple],
        num_tokens_per_rank: Optional[torch.Tensor],
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
        if isinstance(x, tuple):
            raise NotImplementedError("Not support fp8")
        
        x_scales = None
        use_quant = os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT") == "1"
        
        if handle is not None:
            raise NotImplementedError("Optional communication handle is not supported yet.")
        
        assert (
            num_tokens_per_rank is not None
            and is_token_in_rank is not None
            and num_tokens_per_expert is not None
        )
        
        (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
            event,
        ) = self.runtime.intranode_dispatch(
            x,
            x_scales,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            0,
            None,
            None,
            dispatch_wait_recv_cost_stats,
            expert_alignment,
            num_worst_tokens,
            config,
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
            use_quant,
        )
        
        handle = (
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            is_token_in_rank,
            send_head,
            topk_idx,
            topk_weights,
        )
        
        return (
            (recv_x, recv_x_scales) if use_quant else recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            EventOverlap(event),
        )
    
    def _internode_dispatch(
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
        config,
        previous_event: Optional[EventOverlap],
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        use_quant = os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT") == "1"
        
        if handle is not None:
            raise NotImplementedError("Optional communication handle is not supported yet.")
        
        assert (
            num_tokens_per_rank is not None
            and is_token_in_rank is not None
            and num_tokens_per_expert is not None
        )
        
        (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            recv_src_idx,
            send_head,
            offset_inner,
            offset_outer,
            count_outer,
            expand_scales,
            event,
        ) = self.runtime.internode_dispatch(
            x,
            x_scales,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            config,
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
            use_quant,
        )
        
        handle = (
            recv_src_idx,
            is_token_in_rank,
            send_head,  # ep_rank_token_cnt
            topk_idx,
            topk_weights,
            offset_inner,
            offset_outer,  # token_server_idx
            count_outer,
            expand_scales,
        )
        
        return (
            (recv_x, recv_x_scales) if use_quant else recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            EventOverlap(event),
        )
    
    def combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor] = None,
        bias=None,
        config=None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        combine_send_cost_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        # Internode combine
        if self.runtime.get_num_rdma_ranks() > 1:
            return self._internode_combine(
                x, handle, topk_weights, bias, config, previous_event,
                async_finish, allocate_on_comm_stream,
            )
        
        # Intranode combine
        return self._intranode_combine(
            x, handle, topk_weights, config, previous_event,
            async_finish, allocate_on_comm_stream, combine_send_cost_stats,
        )
    
    def _intranode_combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor],
        config,
        previous_event: Optional[EventOverlap],
        async_finish: bool,
        allocate_on_comm_stream: bool,
        combine_send_cost_stats: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        (
            rank_prefix_matrix,
            _,
            channel_prefix_matrix,
            src_idx,
            is_in_recv_token_rank,
            send_head,
            topk_idx,
            topk_weights_ori,
        ) = handle
        
        recv_x, recv_topk_weights, event = self.runtime.intranode_combine(
            x, topk_idx, topk_weights_ori, src_idx, send_head, combine_send_cost_stats
        )
        
        return recv_x, recv_topk_weights, EventOverlap(event)
    
    def _internode_combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor],
        bias,
        config,
        previous_event: Optional[EventOverlap],
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        (
            src_idx,
            is_recv_token_in_rank,
            send_head,
            topk_idx,
            topk_weights_ori,
            offset_inner,
            offset_outer,
            count_outer,
            expand_scales,
        ) = handle
        
        recv_x, recv_topk_weights, event = self.runtime.internode_combine(
            x, topk_idx, topk_weights_ori, src_idx, send_head,
            offset_inner, offset_outer, count_outer, expand_scales,
        )
        
        return recv_x, recv_topk_weights, EventOverlap(event)

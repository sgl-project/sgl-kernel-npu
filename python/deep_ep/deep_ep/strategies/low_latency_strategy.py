"""
Low latency mode EP communication strategies.
All low latency mode strategy implementations are in this file.
"""

import os
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu
from deep_ep_cpp import EventHandle

from ..ep_strategy import LowLatencyEPCommStrategy, register_low_latency_strategy
from ..utils import EventOverlap


@register_low_latency_strategy("default")
class DefaultLowLatencyCommStrategy(LowLatencyEPCommStrategy):
    """
    Low latency mode strategy using Custom operator implementation (deep_ep_cpp).
    This is the default implementation for low latency mode.
    """

    def __init__(self, runtime, group: dist.ProcessGroup):
        super().__init__(group)
        self.runtime = runtime

    def get_name(self) -> str:
        return "default"

    def get_supported_modes(self) -> List[str]:
        return ["low_latency"]

    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        topk_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        Tuple,
        EventOverlap,
        Callable,
    ]:
        topk_ids = topk_idx.int()

        (
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            event,
            hook,
        ) = self.runtime.low_latency_dispatch(
            x,
            topk_ids,
            cumulative_local_expert_recv_stats,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8,
            round_scale,
            use_ue8m0,
            async_finish,
            return_recv_hook,
        )

        handle = (
            packed_recv_src_info,
            packed_recv_layout_range,
            num_max_dispatch_tokens_per_rank,
            x.size(1),
            num_experts,
            packed_recv_count,
            None,  # expand_scales
        )

        tensors_to_record = (
            x,
            topk_idx,
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            cumulative_local_expert_recv_stats,
        )

        return (
            (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x,
            packed_recv_count,
            handle,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )

    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, EventOverlap, Callable]:
        topk_ids = topk_idx.int()

        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
            packed_recv_count,
            expand_scales,
        ) = handle

        combined_x, event, hook = self.runtime.low_latency_combine(
            x,
            topk_ids,
            topk_weights,
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            packed_recv_count,
            zero_copy,
            async_finish,
            return_recv_hook,
            out,
        )

        tensors_to_record = (
            x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            combined_x,
        )

        return (
            combined_x,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )


@register_low_latency_strategy("ops")
class OpsLowLatencyCommStrategy(LowLatencyEPCommStrategy):
    """
    Low latency mode strategy using torch_npu ops.
    This strategy uses the ops-transformer op for A3 RoCE.
    """

    def __init__(self, runtime, group: dist.ProcessGroup, comm_alg: str = "hierarchy"):
        super().__init__(group)
        comm_alg_support_list = ["hierarchy", "fullmesh_v1", "fullmesh_v2", "ccu", ""]
        torch._check(
            comm_alg in comm_alg_support_list,
            lambda: (
                f"comm_alg only support {comm_alg_support_list=}, but got {comm_alg=}."
            ),
        )
        self.comm_alg = comm_alg

    def get_name(self) -> str:
        return "ops"

    def get_supported_modes(self) -> List[str]:
        return ["low_latency"]

    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        topk_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        Tuple,
        EventOverlap,
        Callable,
    ]:

        topk_ids = topk_idx.int()
        x_active_mask = torch.zeros(
            num_max_dispatch_tokens_per_rank,
            dtype=torch.bool,
            device=x.device,
        )
        x_active_mask[:x.size(0)] = True
        padding_size = num_max_dispatch_tokens_per_rank - x.size(0)
        if self.comm_alg == "hierarchy":
            assert (
                topk_weights is not None
            ), "When comm_alg='hierarchy', topk_weights can not be None"
        x_padding = torch.empty(
            padding_size,
            x.size(1),
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat((x, x_padding), dim=0)
        topk_padding = torch.empty(
            padding_size,
            topk_ids.size(1),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_ids = torch.cat((topk_ids, topk_padding), dim=0)
        weight_padding = torch.empty(
            padding_size,
            topk_weights.size(1),
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )
        topk_weights = torch.cat((topk_weights, weight_padding), dim=0)

        (
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            expand_scales,
        ) = self._npu_low_latency_dispatch(
            x=x,
            topk_idx=topk_ids,
            num_experts=num_experts,
            quant_mode=3 if use_ue8m0 else 2 if use_fp8 else 0,
            comm_alg=self.comm_alg,
            topk_weights=topk_weights,
            x_active_mask=x_active_mask,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        )

        handle = (
            packed_recv_src_info,
            packed_recv_layout_range,
            num_max_dispatch_tokens_per_rank,
            x.size(1),
            num_experts,
            packed_recv_count,
            expand_scales,
            x_active_mask,
        )

        event = EventOverlap(EventHandle())
        hook = lambda *args, **kwargs: None

        return (
            (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x,
            packed_recv_count,
            handle,
            event,
            hook,
        )

    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, EventOverlap, Callable]:

        topk_ids = topk_idx.int()
        src_num = topk_idx.size(0)
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
            packed_recv_count,
            expand_scales,
            x_active_mask,
        ) = handle
        padding_size = num_max_dispatch_tokens_per_rank - topk_ids.size(0)
        topk_padding = torch.empty(
            padding_size,
            topk_ids.size(1),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_ids = torch.cat((topk_ids, topk_padding), dim=0)
        weight_padding = torch.empty(
            padding_size,
            topk_weights.size(1),
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )
        topk_weights = torch.cat((topk_weights, weight_padding), dim=0)

        combined_x = self._npu_low_latency_combine(
            x=x,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            assist_info_for_combine=src_info,
            ep_send_counts=layout_range,
            num_experts=num_experts,
            comm_alg=self.comm_alg,
            expand_scales=expand_scales,
            x_active_mask=x_active_mask,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        )

        event = EventOverlap(EventHandle())
        hook = lambda *args, **kwargs: None
        combined_x = combined_x[:src_num, :]
        return combined_x, event, hook

    def _npu_low_latency_dispatch(
        self,
        x,
        topk_idx,
        num_experts: int,
        *,
        quant_mode=0,
        comm_alg="",
        x_smooth_scale=None,
        x_active_mask=None,
        topk_weights=None,
        expert_shard_type=0,
        shared_expert_num=0,
        shared_expert_rank_num=0,
        num_max_dispatch_tokens_per_rank=0,
    ):

        shared_expert_rank_num = int(os.getenv("MOE_SHARED_EXPERT_RANK_NUM", 0))
        expert_token_nums_type = int(os.getenv("MOE_EXPERT_TOKEN_NUMS_TYPE", 1))
        if comm_alg == "hierarchy":
            assert (
                shared_expert_num == 0
            ), "When comm_alg='hierarchy', shared_expert_num must be 0."

        (
            expand_x,
            dynamic_scales,
            expand_idx,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
            expand_scales,
        ) = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=topk_idx,
            group_ep=self.group_name,
            ep_world_size=self.group_size,
            ep_rank_id=self.rank,
            moe_expert_num=num_experts,
            scales=x_smooth_scale,
            x_active_mask=x_active_mask,
            expert_scales=topk_weights,
            performance_info=None,
            expert_shard_type=expert_shard_type,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            quant_mode=quant_mode,
            expert_token_nums_type=expert_token_nums_type,
            comm_alg=comm_alg,  # A3: 支持""，"fullmesh_v1"，"fullmesh_v2", "hierarchy"
        )

        return (
            expand_x,
            dynamic_scales,
            expert_token_nums,
            expand_idx,
            ep_recv_counts,
            expand_scales,
        )

    def _npu_low_latency_combine(
        self,
        x,
        topk_idx,
        topk_weights,
        assist_info_for_combine,
        ep_send_counts,
        *,
        num_experts=0,
        comm_alg="",
        comm_quant_mode=0,
        x_active_mask=None,
        expand_scales=None,
        shared_expert_x=None,
        expert_shared_type=0,
        shared_expert_num=0,
        shared_expert_rank_num=0,
        num_max_dispatch_tokens_per_rank=0,
    ):

        shared_expert_rank_num = int(os.getenv("MOE_SHARED_EXPERT_RANK_NUM", 0))
        if comm_alg == "hierarchy":
            assert (
                shared_expert_num == 0
            ), "When comm_alg='hierarchy', shared_expert_num must be 0."

        combine_x = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=x,
            expert_ids=topk_idx,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_send_counts,
            expert_scales=topk_weights,
            group_ep=self.group_name,
            ep_world_size=self.group_size,
            ep_rank_id=self.rank,
            moe_expert_num=num_experts,
            tp_send_counts=None,
            x_active_mask=x_active_mask,
            expand_scales=expand_scales,
            shared_expert_x=shared_expert_x,
            performance_info=None,
            expert_shard_type=expert_shared_type,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            comm_quant_mode=comm_quant_mode,
            comm_alg=comm_alg,  # A3: 支持""，"hierarchy"两种
        )

        return combine_x


@register_low_latency_strategy("alltoall")
class AllToAllLowLatencyCommStrategy(LowLatencyEPCommStrategy):
    """
    Low latency mode strategy using torch_npu alltoall.
    This strategy uses the alltoall for A3 RoCE.
    """

    def __init__(self, runtime, group: dist.ProcessGroup):
        super().__init__(group)

    def get_name(self) -> str:
        return "alltoall"

    def get_supported_modes(self) -> List[str]:
        return ["low_latency"]

    def low_latency_dispatch(
        buffer,
        x,
        topk_idx,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        cumulative_local_expert_recv_stats=None,
        use_fp8=True,
        round_scale=False,
        use_ue8m0=False,
        async_finish=False,
        return_recv_hook=False,
        topk_weights: Optional[torch.Tensor] = None,
    ):
        group = buffer.group
        group_size = buffer.group_size
        num_local_experts = num_experts // group_size
        ep_rank = buffer.rank
        device = x.device
        hidden = x.size(1)
        aligned_num_tokens = num_max_dispatch_tokens_per_rank
        num_tokens = x.size(0)
        padding_size = aligned_num_tokens - num_tokens
        x_padding = torch.zeros(
            (padding_size, hidden),
            dtype=x.dtype,
            device=x.device,
        )
        topk_padding = torch.zeros(
            padding_size,
            topk_idx.size(1),
            dtype=topk_idx.dtype,
            device=topk_idx.device,
        )
        topk_padding = torch.cat((topk_idx, topk_padding), dim=0)
        x_padding = torch.cat((x, x_padding), dim=0)

        topk_idx_int = topk_padding.to(torch.int32)
        expert_capacity = aligned_num_tokens

        (expanded_x, expanded_row_idx, _, _) = torch_npu.npu_moe_init_routing_v2(
            x_padding,
            topk_idx_int,
            quant_mode=-1,
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            row_idx_type=0,
            drop_pad_mode=1,
            expert_capacity=expert_capacity,
            active_expert_range=[0, num_experts],
        )

        expanded_x_2d = expanded_x.reshape(num_experts * expert_capacity, hidden)
        chunk_size = num_local_experts * expert_capacity

        input_list = [
            expanded_x_2d[r * chunk_size : (r + 1) * chunk_size].contiguous()
            for r in range(group_size)
        ]
        output_list = [
            torch.empty(chunk_size, hidden, dtype=expanded_x_2d.dtype, device=device)
            for r in range(group_size)
        ]
        dist.all_to_all(output_list, input_list, group=group)
        recv_x_raw = torch.cat(output_list, dim=0)

        recv_all = recv_x_raw.reshape(
            group_size, num_local_experts, expert_capacity, hidden
        )
        recv_all = recv_all.permute(1, 0, 2, 3).contiguous()
        recv_x = recv_all.reshape(
            num_local_experts * group_size * expert_capacity, hidden
        )
        recv_x_out = (torch_npu.npu_dynamic_quant(recv_x)) if use_fp8 else recv_x

        packed_recv_count = torch.full(
            (num_local_experts,),
            expert_capacity * group_size,
            dtype=torch.int64,
            device=x.device,
        )

        handle_tuple = (
            expanded_row_idx,
            expert_capacity,
            hidden,
            num_tokens,
            num_local_experts,
            group_size,
            packed_recv_count,
        )

        return (
            recv_x_out,
            packed_recv_count,
            handle_tuple,
            EventOverlap(),
            lambda: None,
        )

    def low_latency_combine(
        buffer,
        x,
        topk_idx,
        topk_weights,
        handle,
        zero_copy=False,
        async_finish=False,
        return_recv_hook=False,
        out=None,
    ):
        expanded_row_idx = handle[0]
        expert_capacity = handle[1]
        hidden = handle[2]
        num_tokens = handle[3]
        num_local_experts = handle[4]
        group_size = handle[5]

        device = x.device
        group = buffer.group

        x_reordered = x.reshape(num_local_experts, group_size, expert_capacity, hidden)
        x_reordered = x_reordered.permute(1, 0, 2, 3).contiguous()
        x_reordered = x_reordered.reshape(
            group_size * num_local_experts * expert_capacity, hidden
        )

        chunk_size = num_local_experts * expert_capacity
        input_list = [
            x_reordered[r * chunk_size : (r + 1) * chunk_size].contiguous()
            for r in range(group_size)
        ]
        output_list = [
            torch.empty(chunk_size, hidden, dtype=x.dtype, device=device)
            for r in range(group_size)
        ]
        dist.all_to_all(output_list, input_list, group=group)
        recv_all_raw = torch.cat(output_list, dim=0)

        recv_all_raw = recv_all_raw.reshape(
            group_size * num_local_experts, expert_capacity, hidden
        )
        topk_weights_padding = torch.empty(
            expert_capacity,
            topk_weights.size(1),
            dtype=x.dtype,
            device=x.device,
        )
        topk_weights_padding[:num_tokens].copy_(topk_weights)
        output = torch_npu.npu_moe_finalize_routing(
            expanded_permuted_rows=recv_all_raw,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights_padding,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=3,
        )
        output = output[:num_tokens, :]

        return output, EventOverlap(), lambda: None

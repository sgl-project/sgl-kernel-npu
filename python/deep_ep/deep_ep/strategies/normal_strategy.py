"""
Normal mode EP communication strategies.
All normal mode strategy implementations are in this file.
"""

import os
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from deep_ep_cpp import EventHandle

from ..ep_strategy import (
    VALID_QUANT_MODES,
    NormalEPCommStrategy,
    register_normal_strategy,
)
from ..utils import EventOverlap

# Global variable for communication stream
COMM_STREAM = None


@register_normal_strategy("default")
class DefaultNormalCommStrategy(NormalEPCommStrategy):
    """
    Normal mode strategy using Custom operator implementation (deep_ep_cpp).
    This is the default and most optimized implementation for normal mode.
    """

    def __init__(self, runtime, group: dist.ProcessGroup):
        super().__init__(group)
        self.runtime = runtime

    def get_name(self) -> str:
        return "default"

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
        quant_mode: Optional[str] = None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:

        if self.runtime.get_num_rdma_ranks() > 1:
            return self._internode_dispatch(
                x,
                handle,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                topk_idx,
                topk_weights,
                expert_alignment,
                config,
                previous_event,
                async_finish,
                allocate_on_comm_stream,
            )

        return self._intranode_dispatch(
            x,
            handle,
            num_tokens_per_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            topk_idx,
            topk_weights,
            expert_alignment,
            num_worst_tokens,
            config,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
            dispatch_wait_recv_cost_stats,
            quant_mode,
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
        quant_mode: Optional[str] = None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        # Determine quant type from quant_mode
        if quant_mode is None:
            if isinstance(x, torch.Tensor):
                # BF16 no quant
                data = x
                x_scales = None
                quant_type = "bf16"
                use_quant = False
            elif isinstance(x, tuple) and len(x) == 2:
                data, quant_type_tensor = x
                if quant_type_tensor.dtype == torch.float8_e4m3fn:
                    quant_type = "mx_fp8_e4m3"
                    use_quant = True
                elif quant_type_tensor.dtype == torch.float8_e5m2:
                    quant_type = "mx_fp8_e5m2"
                    use_quant = True
                elif quant_type_tensor.dtype == torch.int8:
                    quant_type = "int8"
                    use_quant = True
                elif quant_type_tensor.dtype == torch.float4_e2m1fn_x2:
                    quant_type = "mx_fp4_e2m1"
                    use_quant = True
                else:
                    raise TypeError(
                        f"Unsupported quantized dtype: {quant_type_tensor.dtype}"
                    )
                x_scales = None
            else:
                raise TypeError(f"Unsupported x type: {type(x)}")

            if not use_quant:
                use_quant = os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT") == "1"
                if use_quant:
                    quant_type = "int8"
        else:
            # New API: explicit quant_mode
            if quant_mode not in VALID_QUANT_MODES:
                raise ValueError(
                    f"Invalid quant_mode: {quant_mode}. Valid options: {VALID_QUANT_MODES}"
                )
            data = x
            x_scales = None
            quant_type = quant_mode
            use_quant = quant_mode != "bf16"

        if handle is not None:
            raise NotImplementedError(
                "Optional communication handle is not supported yet."
            )

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
            data,
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
            quant_type,
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
            raise NotImplementedError(
                "Optional communication handle is not supported yet."
            )

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

        if self.runtime.get_num_rdma_ranks() > 1:
            return self._internode_combine(
                x,
                handle,
                topk_weights,
                bias,
                config,
                previous_event,
                async_finish,
                allocate_on_comm_stream,
            )

        return self._intranode_combine(
            x,
            handle,
            topk_weights,
            config,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
            combine_send_cost_stats,
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
            x,
            topk_idx,
            topk_weights_ori,
            src_idx,
            send_head,
            offset_inner,
            offset_outer,
            count_outer,
            expand_scales,
        )

        return recv_x, recv_topk_weights, EventOverlap(event)


@register_normal_strategy("alltoall")
class AlltoAllNormalCommStrategy(NormalEPCommStrategy):
    """
    Normal mode strategy using alltoallv implementation.
    This strategy uses the alltoall for A3 RoCE.
    Internode and intranode use the same implementation.
    """

    def __init__(self, runtime, group: dist.ProcessGroup):
        super().__init__(group)
        self.runtime = runtime
        self._alltoall_layout = None

    def get_name(self) -> str:
        return "alltoall"

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
        """Get dispatch layout using alltoall"""
        group = self.group
        group_size = self.group_size
        num_local_experts = num_experts // group_size
        ep_rank = self.rank
        device = topk_idx.device

        num_local_tokens_per_expert = torch.histc(
            topk_idx, bins=num_experts, min=0, max=num_experts
        )

        input_splits = (
            num_local_tokens_per_expert.reshape(group_size, num_local_experts)
            .sum(axis=1)
            .cpu()
            .numpy()
            .tolist()
        )

        num_global_tokens_per_expert = self._gather_along_first_dim(
            num_local_tokens_per_expert, group
        ).reshape(group_size, num_experts)

        local_expert_indices_offset = ep_rank * num_local_experts
        local_expert_indices = [
            local_expert_indices_offset + i for i in range(num_local_experts)
        ]

        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, local_expert_indices[0] : local_expert_indices[-1] + 1
        ]

        output_splits = (
            num_global_tokens_per_local_expert.sum(axis=-1).cpu().numpy().tolist()
        )

        num_tokens_per_expert = num_global_tokens_per_local_expert.sum(axis=0)

        expert_ids_per_ep_rank = (
            torch.arange(
                num_experts,
                dtype=torch.int32,
                device=device,
            )
            % num_local_experts
        )

        num_global_tokens_per_local_expert_ravel = (
            num_global_tokens_per_local_expert.ravel()
        )
        if num_local_experts > 1:
            global_tokens_indices = torch.repeat_interleave(
                expert_ids_per_ep_rank,
                num_global_tokens_per_local_expert_ravel,
            )
        else:
            torch.npu.synchronize()
            global_tokens_indices = None

        self._alltoall_layout = {
            "num_local_experts": num_local_experts,
            "input_splits": input_splits,
            "output_splits": output_splits,
            "num_global_tokens_per_local_expert": num_global_tokens_per_local_expert,
            "global_tokens_indices": global_tokens_indices,
            "num_experts": num_experts,
        }

        num_tokens_per_rank = num_local_tokens_per_expert.reshape(
            group_size, num_local_experts
        ).sum(axis=1)
        is_token_in_rank = torch.zeros(
            (topk_idx.size(0), group_size), dtype=torch.bool, device=device
        )

        return (
            num_tokens_per_rank,
            None,
            num_tokens_per_expert,
            is_token_in_rank,
            EventOverlap(),
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
        quant_mode: Optional[str] = None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """Dispatch using alltoall (for internode and intranode)"""

        layout = self._alltoall_layout
        num_local_experts = layout["num_local_experts"]
        input_splits = layout["input_splits"]
        output_splits = layout["output_splits"]
        num_global_tokens_per_local_expert = layout[
            "num_global_tokens_per_local_expert"
        ]
        global_tokens_indices = layout["global_tokens_indices"]
        num_experts = layout["num_experts"]
        topk_idx_int = topk_idx.to(torch.int32)

        # Determine quant type from quant_mode
        VALID_QUANT_MODES = {
            "bf16",
            "int8",
        }
        if quant_mode is None:
            quant_mode = "bf16"
        if quant_mode not in VALID_QUANT_MODES:
            raise NotImplementedError(
                f"quant_mode '{quant_mode}' is not supported by the alltoall strategy. "
                f"Only 'bf16' and 'int8' are supported; use the default strategy for "
                f"FP8/FP4 modes."
            )
        hidden_shape = x.shape

        use_quant = 1 if quant_mode == "int8" else -1
        is_quant_env = os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT")
        if is_quant_env is not None and quant_mode is None:
            use_quant = 1 if is_quant_env == "1" else -1

        (permutated_tokens, reversed_local_mapping, _, dynamic_scale) = (
            torch_npu.npu_moe_init_routing_v2(
                x,
                topk_idx_int,
                quant_mode=use_quant,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                row_idx_type=0,
                active_expert_range=[0, num_experts],
            )
        )

        if use_quant == 1:
            _, dynamic_scale_after_all2all, scale_handle = self._async_all_to_all(
                dynamic_scale, output_splits, input_splits, self.group
            )
            scale_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        _, global_input_tokens, handle_a2a = self._async_all_to_all(
            permutated_tokens,
            output_splits,
            input_splits,
            self.group,
        )
        handle_a2a.wait()
        permutated_tokens.untyped_storage().resize_(0)

        if num_local_experts > 1:
            global_tokens_indices = global_tokens_indices.reshape(
                global_tokens_indices.size(0), 1
            )
            if use_quant == 1:
                dynamic_scale_after_all2all = dynamic_scale_after_all2all.reshape(
                    dynamic_scale_after_all2all.size(0), 1
                )
                (dynamic_scale_after_routing, reversed_global_mapping, _, _) = (
                    torch_npu.npu_moe_init_routing_v2(
                        dynamic_scale_after_all2all,
                        global_tokens_indices,
                        quant_mode=-1,
                        expert_num=num_local_experts,
                        expert_tokens_num_type=1,
                        expert_tokens_num_flag=True,
                        row_idx_type=0,
                        active_expert_range=[0, num_local_experts],
                    )
                )
                dynamic_scale_after_routing = dynamic_scale_after_routing.reshape(
                    dynamic_scale_after_routing.size(0)
                )
            (dispatch_out, reversed_global_mapping, _, _) = (
                torch_npu.npu_moe_init_routing_v2(
                    global_input_tokens,
                    global_tokens_indices,
                    quant_mode=-1,
                    expert_num=num_local_experts,
                    expert_tokens_num_type=1,
                    expert_tokens_num_flag=True,
                    row_idx_type=0,
                    active_expert_range=[0, num_local_experts],
                )
            )
        else:
            dispatch_out = global_input_tokens
            reversed_global_mapping = None

        num_recv_tokens_per_expert_list = (
            num_global_tokens_per_local_expert.sum(axis=0).cpu().numpy().tolist()
        )

        combine_handle = {
            "input_splits": input_splits,
            "output_splits": output_splits,
            "topk_weights": topk_weights,
            "reversed_local_mapping": reversed_local_mapping,
            "reversed_global_mapping": reversed_global_mapping,
            "hidden_shape": hidden_shape,
            "hidden_shape_before_permute": x.shape,
            "num_local_experts": num_local_experts,
        }
        recv_x = (
            (dispatch_out, dynamic_scale_after_routing)
            if use_quant == 1
            else dispatch_out
        )

        return (
            recv_x,
            None,
            None,
            num_recv_tokens_per_expert_list,
            combine_handle,
            EventOverlap(),
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
        """Combine using alltoall (same for internode and intranode)"""

        input_splits = handle["input_splits"]
        output_splits = handle["output_splits"]
        topk_weights = handle["topk_weights"]
        reversed_local_mapping = handle["reversed_local_mapping"]
        reversed_global_mapping = handle["reversed_global_mapping"]
        hidden_shape = handle["hidden_shape"]
        hidden_shape_before_permute = handle["hidden_shape_before_permute"]
        num_local_experts = handle["num_local_experts"]

        if (
            x.shape[0] > 0
            and num_local_experts > 1
            and reversed_global_mapping is not None
        ):
            x = torch_npu.npu_moe_finalize_routing(
                expanded_permuted_rows=x,
                skip1=None,
                skip2=None,
                bias=None,
                scales=None,
                expanded_src_to_dst_row=reversed_global_mapping.to(torch.int32),
                export_for_source_row=None,
                drop_pad_mode=2,
            )

        _, local_tokens, a2a_handle = self._async_all_to_all(
            x,
            input_splits,
            output_splits,
            self.group,
        )
        a2a_handle.wait()
        x.untyped_storage().resize_(0)

        output = torch_npu.npu_moe_finalize_routing(
            expanded_permuted_rows=local_tokens,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=reversed_local_mapping.to(torch.int32),
            export_for_source_row=None,
            drop_pad_mode=2,
        )
        output = output.view(hidden_shape)

        return output, None, EventOverlap()

    def _async_all_to_all(
        self, input_, output_split_sizes, input_split_sizes, group, event=None
    ):
        """Async all-to-all operation"""
        global COMM_STREAM

        if output_split_sizes is None:
            # Equal split (all2all)
            a2a_out = torch.empty_like(input_)
        else:
            # Unequal split (all2all-v)
            a2a_out = input_.new_empty(
                size=[sum(output_split_sizes)] + list(input_.size()[1:]),
                dtype=input_.dtype,
                device=torch.npu.current_device(),
            )

        if event:
            # multi stream wait event
            if COMM_STREAM is None:
                COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
            with torch_npu.npu.stream(COMM_STREAM):
                event.wait()
                handle = dist.all_to_all_single(
                    a2a_out,
                    input_.contiguous(),
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                    async_op=True,
                )
        else:
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )

        return input_, a2a_out, handle

    def _gather_along_first_dim(self, input_, group):
        """Gather tensors along first dimension"""
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input_

        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * world_size
        output = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.npu.current_device()
        )
        torch.distributed.all_gather_into_tensor(
            output, input_.contiguous(), group=group
        )
        return output


@register_normal_strategy("allgather")
class AllGatherNormalCommStrategy(NormalEPCommStrategy):
    """
    Normal mode strategy using AllGather implementation.
    All ranks gather all tokens, each rank processes only its local experts,
    then reduce-scatter combines partial results.

    Padding strategy (mirrors vllm-ascend's EP path):
      1. Before all_gather: pad each rank's input to the global max token
         count (required by all_gather_into_tensor which demands equal
         send sizes).
      2. After all_gather: immediately unpad into a compact layout of
         [sum(local_tokens), ...] so that routing and expert FFN never
         touch padding rows.
      3. Before reduce_scatter: re-pad the compact output back to
         [group_size * max_tokens, ...] (reduce_scatter_tensor also
         requires equal sizes), then trim the scattered result to
         local_num_tokens.
    """

    def __init__(self, runtime, group: dist.ProcessGroup):
        super().__init__(group)
        self.runtime = runtime
        self._allgather_layout = None
        self.use_mx_fp8_quant = int(os.environ.get("USE_MX_FP8_QUANT", "0"))

    def get_name(self) -> str:
        return "allgather"

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
        """Get dispatch layout for AllGather mode."""
        group = self.group
        group_size = self.group_size
        num_local_experts = num_experts // group_size
        ep_rank = self.rank
        device = topk_idx.device

        self._allgather_layout = {
            "num_experts": num_experts,
            "num_local_experts": num_local_experts,
            "first_expert_idx": ep_rank * num_local_experts,
            "last_expert_idx": ep_rank * num_local_experts + num_local_experts,
        }

        num_tokens_per_rank = torch.empty(group_size, dtype=torch.int32, device=device)
        is_token_in_rank = torch.ones(
            (topk_idx.size(0), group_size), dtype=torch.bool, device=device
        )
        num_tokens_per_expert = torch.empty(
            num_local_experts, dtype=torch.int64, device=device
        )

        return (
            num_tokens_per_rank,
            None,
            num_tokens_per_expert,
            is_token_in_rank,
            EventOverlap(),
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
        quant_mode: Optional[str] = None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """Dispatch using AllGather: all ranks get all tokens, process local experts only."""
        layout = self._allgather_layout
        num_experts = layout["num_experts"]
        num_local_experts = layout["num_local_experts"]
        first_expert_idx = layout["first_expert_idx"]
        last_expert_idx = layout["last_expert_idx"]
        group_size = self.group_size
        ep_rank = self.rank

        if isinstance(x, tuple):
            hidden_states = x[0]
        else:
            hidden_states = x

        local_num_tokens = hidden_states.shape[0]
        hidden_shape = hidden_states.shape

        # ----------------------------------------------------------------
        # Step 0: Synchronize token counts across ranks.
        # all_gather_into_tensor requires every rank to send the same number
        # of bytes.  Collect per-rank token counts so we can (a) compute the
        # global max for padding and (b) unpad precisely after the gather.
        # ----------------------------------------------------------------
        if group_size > 1:
            from sglang.srt.layers.utils.cp_utils import MAX_LEN, PER_RANK_ACTUAL_TOKEN
            global MAX_LEN
            max_tokens = MAX_LEN
            global PER_RANK_ACTUAL_TOKEN
            all_num_tokens = PER_RANK_ACTUAL_TOKEN
        else:
            max_tokens = local_num_tokens
            all_num_tokens = [local_num_tokens]
        # ----------------------------------------------------------------
        # Step 1: Quantize hidden_states BEFORE AllGather to halve
        # communication volume.  per-token quant is safe to do locally —
        # each rank quantizes its own tokens, AllGather concatenates, and
        # the numerical result is identical to quantizing after AG.
        # ----------------------------------------------------------------
        print(f"{hidden_states.shape=} {max_tokens=} {all_num_tokens=} {torch.distributed.get_rank()=}", flush=True)
        if self.use_mx_fp8_quant:
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                hidden_states, dst_type=torch.float8_e4m3fn
            )
        else:
            pertoken_scale = None

        # ----------------------------------------------------------------
        # Step 2: Pad + AllGather hidden_states / topk_idx / topk_weights /
        # scale, then immediately UNPAD into a compact layout.
        #
        # After padded all-gather the layout is:
        #   [rank0: max_tokens | rank1: max_tokens | ...]
        # where each rank's chunk has (max_tokens - local_r) zero-filled
        # padding rows at the end.  We compact this to:
        #   [rank0: local_0 | rank1: local_1 | ...]
        # so that downstream routing and expert FFN never see padding.
        # ----------------------------------------------------------------
        if group_size > 1:
            global_hidden_states = self._all_gather(
                hidden_states, max_tokens, all_num_tokens
            )
            global_topk_idx = self._all_gather(
                topk_idx, max_tokens, all_num_tokens
            )
            global_topk_weights = self._all_gather(
                topk_weights, max_tokens, all_num_tokens
            )
            if pertoken_scale is not None:
                global_pertoken_scale = self._all_gather(
                    pertoken_scale, max_tokens, all_num_tokens
                )
            else:
                global_pertoken_scale = None
        else:
            global_hidden_states = hidden_states
            global_topk_idx = topk_idx
            global_topk_weights = topk_weights
            global_pertoken_scale = pertoken_scale

        # Compact layout: [sum(all_num_tokens), ...] — no padding rows.
        global_num_tokens = global_hidden_states.shape[0]

        # ----------------------------------------------------------------
        # Step 3: Mask out non-local expert weights so that after unpermute,
        # only local expert contributions remain on each rank.
        # ----------------------------------------------------------------
        if group_size > 1:
            expert_map = torch.full(
                (num_experts,), -1, dtype=torch.int32, device=topk_idx.device
            )
            expert_map[first_expert_idx:last_expert_idx] = torch.arange(
                num_local_experts, dtype=torch.int32, device=topk_idx.device
            )
            mask = expert_map[global_topk_idx] != -1
            masked_topk_weights = global_topk_weights * mask.to(
                global_topk_weights.dtype
            )
        else:
            masked_topk_weights = global_topk_weights

        # ----------------------------------------------------------------
        # Step 4: Local routing — sort tokens by expert, keep only local
        # expert tokens.  Because the input is compact (no padding), every
        # row is a real token; expert_tokens counts are accurate.
        # ----------------------------------------------------------------
        topk_idx_int = global_topk_idx.to(torch.int32)
        init_routing_kwargs = dict(
            quant_mode=-1,
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            row_idx_type=0,
            active_expert_range=[first_expert_idx, last_expert_idx],
        )
        if global_pertoken_scale is not None:
            init_routing_kwargs["scale"] = global_pertoken_scale
            init_routing_kwargs["x_dtype"] = torch.float8_e4m3fn

        (
            sorted_hidden_states,
            expanded_row_idx,
            expert_tokens,
            routed_scale,
        ) = torch_npu.npu_moe_init_routing_v2(
            global_hidden_states,
            topk_idx_int,
            **init_routing_kwargs,
        )

        num_recv_tokens_per_expert_list = expert_tokens.to(torch.int64)

        combine_handle = {
            "expanded_row_idx": expanded_row_idx,
            "topk_weights": masked_topk_weights,
            "hidden_shape": hidden_shape,
            "local_num_tokens": local_num_tokens,
            "max_tokens": max_tokens,
            "all_num_tokens": all_num_tokens,
            "global_num_tokens": global_num_tokens,
            "group_size": group_size,
        }

        # Return (hidden_states, scale) tuple when quantized — MLP path consumes both.
        recv_x = (
            (sorted_hidden_states, routed_scale)
            if self.use_mx_fp8_quant
            else sorted_hidden_states
        )
        return (
            recv_x,
            None,
            None,
            num_recv_tokens_per_expert_list,
            combine_handle,
            EventOverlap(),
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
        """Combine using AllGather: unpermute, re-pad, then reduce-scatter."""
        expanded_row_idx = handle["expanded_row_idx"]
        probs = handle["topk_weights"]
        hidden_shape = handle["hidden_shape"]
        local_num_tokens = handle["local_num_tokens"]
        max_tokens = handle["max_tokens"]
        all_num_tokens = handle["all_num_tokens"]
        global_num_tokens = handle["global_num_tokens"]
        group_size = handle["group_size"]

        # ----------------------------------------------------------------
        # Step 1: Unpermute — restore global token order (compact layout)
        # and apply routing weights.  Output is [global_num_tokens, ...].
        # ----------------------------------------------------------------
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=x,
            sorted_indices=expanded_row_idx,
            probs=probs,
        )

        # ----------------------------------------------------------------
        # Step 2: Re-pad compact output to [group_size * max_tokens, ...]
        # so reduce_scatter_tensor gets equal-sized chunks per rank.
        # Padding rows are zeros, contributing nothing to the sum.
        # Then reduce-scatter: each rank receives its own [max_tokens, ...]
        # chunk (sum of all ranks' partial results for its tokens).
        # ----------------------------------------------------------------
        if group_size > 1:
            padded_output = self._repad_for_scatter(
                output, all_num_tokens, max_tokens
            )
            scattered_output = torch.empty(
                (max_tokens, *output.shape[1:]),
                dtype=output.dtype,
                device=output.device,
            )
            dist.reduce_scatter_tensor(
                scattered_output, padded_output, group=self.group
            )
            output = scattered_output

        # ----------------------------------------------------------------
        # Step 3: Trim padding rows to restore this rank's original token
        # count, then restore the original hidden shape.
        # ----------------------------------------------------------------
        output = output[:local_num_tokens]
        output = output.view(hidden_shape)

        return output, None, EventOverlap()

    def _sync_token_counts(
        self, local_num_tokens: int
    ) -> Tuple[int, List[int]]:
        """AllGather per-rank token counts.

        Returns:
            max_tokens:    global maximum token count (for padded all-gather)
            all_num_tokens: per-rank token count list (for precise unpad/re-pad)
        """
        group_size = torch.distributed.get_world_size(self.group)
        token_counts = torch.empty(
            group_size, dtype=torch.int32, device=torch.npu.current_device()
        )
        local = torch.tensor(
            [local_num_tokens],
            dtype=torch.int32,
            device=torch.npu.current_device(),
        )
        torch.distributed.all_gather(
            list(torch.split(token_counts, 1)), local, group=self.group
        )
        max_tokens = int(token_counts.max().item())
        all_num_tokens = token_counts.tolist()
        return max_tokens, all_num_tokens

    def _all_gather(
        self,
        input_: torch.Tensor,
        max_tokens: int,
        all_num_tokens: List[int],
    ) -> torch.Tensor:
        """Pad input to *max_tokens*, all-gather, then reorganize into compact layout.

        Mirrors cp_all_gather_reorganized_into_tensor:
        1. Pad input to max_tokens along dim 0 (zero-fill) so every rank
           sends the same number of bytes.
        2. all_gather_into_tensor into a [max_tokens * world_size, ...] buffer.
        3. Split by per-rank max_tokens chunks, slice each down to its
           actual token count, and concatenate — yielding
           [sum(all_num_tokens), ...] with no padding rows.
        """
        world_size = torch.distributed.get_world_size(self.group)
        if world_size == 1:
            return input_

        # Step 1: Pad along dim 0 so every rank sends exactly max_tokens rows.
        pad_size = max_tokens - input_.size(0)
        if pad_size > 0:
            # F.pad expects padding in reverse dimension order; for n-D
            # tensor, pad only the first dim: [0, 0]*(ndim-1) + [0, pad_size]
            padding = [0, 0] * (input_.ndim - 1) + [0, pad_size]
            input_ = F.pad(input_, padding, mode="constant", value=0)

        # Step 2: Allocate output buffer and all-gather.
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * world_size
        output = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.npu.current_device()
        )
        print(f"{dim_size[0]=} {pad_size=} {torch.distributed.get_rank()=}", flush=True)
        torch.distributed.all_gather_into_tensor(
            output, input_.contiguous(), group=self.group
        )

        # Step 3: Reorganize — split into per-rank chunks of max_tokens,
        # slice each to its actual token count, concatenate into compact
        # layout (no padding rows).
        outputs_list_max = list(
            torch.split(output, [max_tokens] * world_size, dim=0)
        )
        result = torch.cat(
            [
                outputs_list_max[index][:per_rank_len]
                for index, per_rank_len in enumerate(all_num_tokens)
            ],
            dim=0,
        )
        return result

    @staticmethod
    def _unpad_compact(
        gathered: torch.Tensor,
        all_num_tokens: List[int],
        max_tokens: int,
    ) -> torch.Tensor:
        """Compact a padded all-gather result into a padding-free layout.

        Input layout (after padded all-gather):
            [rank0: max_tokens | rank1: max_tokens | ...]
        where each rank's chunk has (max_tokens - local_r) zero padding rows.

        Output layout (compact):
            [rank0: local_0 | rank1: local_1 | ...]

        This mirrors vllm-ascend's maybe_all_gather_and_maybe_unpad EP path:
        routing and expert FFN downstream never see padding rows, so
        expert_tokens statistics are accurate and no compute is wasted.
        """
        group_size = len(all_num_tokens)
        global_total = sum(all_num_tokens)
        if global_total == gathered.size(0):
            return gathered  # no padding needed (all ranks equal)

        result = torch.empty(
            (global_total, *gathered.shape[1:]),
            dtype=gathered.dtype,
            device=gathered.device,
        )
        offset_in = 0   # read offset in the gathered tensor
        offset_out = 0  # write offset in the compact result
        for r in range(group_size):
            n = all_num_tokens[r]
            result[offset_out : offset_out + n] = gathered[
                offset_in : offset_in + n
            ]
            offset_in += max_tokens
            offset_out += n
        return result

    @staticmethod
    def _repad_for_scatter(
        compact: torch.Tensor,
        all_num_tokens: List[int],
        max_tokens: int,
    ) -> torch.Tensor:
        """Re-pad a compact tensor for reduce_scatter_tensor.

        reduce_scatter_tensor requires input size = world_size * output_size,
        with equal-sized chunks per rank.  This scatters the compact
        [sum(local), ...] tensor back into [group_size * max_tokens, ...]
        layout (per-rank chunks of max_tokens, zero-padded), so the
        subsequent reduce_scatter gives each rank its own tokens.

        Padding rows are zeros, so they contribute nothing to the
        cross-rank sum inside reduce_scatter.

        Fast path: when every rank already has max_tokens tokens (no
        padding), the compact tensor is already in the target layout,
        so we return it directly and avoid an alloc + copy.
        """
        group_size = len(all_num_tokens)
        if compact.size(0) == group_size * max_tokens:
            return compact

        padded = torch.zeros(
            (group_size * max_tokens, *compact.shape[1:]),
            dtype=compact.dtype,
            device=compact.device,
        )
        offset_in = 0   # read offset in the compact tensor
        for r in range(group_size):
            n = all_num_tokens[r]
            padded[r * max_tokens : r * max_tokens + n] = compact[
                offset_in : offset_in + n
            ]
            offset_in += n
        return padded


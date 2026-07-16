"""
Normal mode EP communication strategies.
All normal mode strategy implementations are in this file.
"""

import os
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu
from deep_ep_cpp import EventHandle

from ..ep_strategy import NormalEPCommStrategy, register_normal_strategy
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
        # Determine quant type
        # New API: quant_mode directly specifies the quantization type
        VALID_QUANT_MODES = {
            "bf16",
            "int8",
            "fp8_e4m3",
            "fp8_e5m2",
            "scalar_fp8_e4m3",
            "scalar_fp8_e5m2",
            "fp4_e2m1",
        }
        if quant_mode is not None:
            if quant_mode not in VALID_QUANT_MODES:
                raise ValueError(
                    f"Invalid quant_mode: {quant_mode}. Valid options: {VALID_QUANT_MODES}"
                )
            data = x[0] if isinstance(x, tuple) else x
            x_scales = None
            quant_type = quant_mode
            use_quant = quant_mode != "bf16"
        elif isinstance(x, torch.Tensor):
            # BF16 no quant (backward compat)
            data = x
            x_scales = None
            quant_type = "bf16"
            use_quant = False
        elif isinstance(x, tuple) and len(x) == 2:
            # Old API: infer from signal tensor dtype (backward compat)
            data, quant_type_tensor = x
            if quant_type_tensor.dtype == torch.float8_e4m3fn:
                quant_type = "fp8_e4m3"
                use_quant = True
            elif quant_type_tensor.dtype == torch.float8_e5m2:
                quant_type = "fp8_e5m2"
                use_quant = True
            elif quant_type_tensor.dtype == torch.int8:
                quant_type = "int8"
                use_quant = True
            elif quant_type_tensor.dtype == torch.float4_e2m1fn_x2:
                quant_type = "fp4_e2m1"
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

        hidden_shape = x.shape
        x = x.view(-1, hidden_shape[-1])

        permutated_tokens, reversed_local_mapping = torch_npu.npu_moe_token_permute(
            tokens=x,
            indices=topk_idx,
            num_out_tokens=topk_idx.numel(),
        )

        input_quant = os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT") == "1"
        if input_quant:
            permutated_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                permutated_tokens
            )
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
            if input_quant:
                dynamic_scale_after_all2all, _ = torch_npu.npu_moe_token_permute(
                    dynamic_scale_after_all2all.unsqueeze(-1), global_tokens_indices
                )
                dynamic_scale_after_all2all = dynamic_scale_after_all2all.squeeze(-1)

            dispatch_out, reversed_global_mapping = torch_npu.npu_moe_token_permute(
                global_input_tokens, global_tokens_indices
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
            (dispatch_out, dynamic_scale_after_all2all) if input_quant else dispatch_out
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
            x = torch_npu.npu_moe_token_unpermute(x, reversed_global_mapping)

        _, local_tokens, a2a_handle = self._async_all_to_all(
            x,
            input_splits,
            output_splits,
            self.group,
        )
        a2a_handle.wait()
        x.untyped_storage().resize_(0)

        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=local_tokens,
            sorted_indices=reversed_local_mapping.to(torch.int32),
            probs=topk_weights,
            restore_shape=hidden_shape_before_permute,
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

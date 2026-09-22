import os
from enum import IntEnum
from typing import Callable, List, Optional, Tuple, Union

import deep_ep_cpp
import torch
import torch.distributed as dist
import torch_npu
from deep_ep_cpp import Config, EventHandle

from .ep_strategy import (
    LowLatencyStrategy,
    NormalStrategy,
    StrategyMap,
    get_fused_strategy,
    get_low_latency_strategy,
    get_normal_strategy,
)
from .utils import EventOverlap, _resolve_quant_mode, log_parameters


class FuseMode(IntEnum):
    FUSED_DEEP_MOE = 1
    DISPATCH_FFN_COMBINE = 2


TensorOrTensors = Union[torch.Tensor, List[torch.Tensor]]


class Buffer:

    num_sms: int = 20
    FuseMode = FuseMode

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 12,
        allow_nvlink_for_low_latency_mode: bool = True,
        allow_mnnvl: bool = False,
        normal_strategy: Union[str, NormalStrategy] = NormalStrategy.DEFAULT,
        low_latency_strategy: Union[
            str, LowLatencyStrategy
        ] = LowLatencyStrategy.DEFAULT,
    ) -> None:
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode HCCS communication. Use this name
                to ensure compatibility with DeepEP.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
            allow_nvlink_for_low_latency_mode: This parameter is deprecated and retained to ensure compatibility with DeepEP.
            allow_mnnvl: This parameter is deprecated and retained to ensure compatibility with DeepEP.
            normal_strategy: the strategy to use for normal mode dispatch/combine, support: default, alltoall.
            low_latency_strategy: the strategy to use for low latency mode dispatch/combine, support: default, ops.
        """

        self.group = group
        self.rank = group.rank()
        self.group_size = group.size()
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        try:
            backend = group._get_backend(torch.device("npu"))
            moe_all_to_all_group_name = backend.get_hccl_comm_name(self.rank)
        except Exception as e:
            print("get_hccl_comm_name failed", e)
            moe_all_to_all_group_name = ""

        self.moe_all_to_all_group_name = moe_all_to_all_group_name

        self.runtime = deep_ep_cpp.Buffer(
            self.rank,
            self.group_size,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode,
            moe_all_to_all_group_name,
        )

        # set strategy by env
        deep_mode = os.getenv("DEEP_USE_MODE")

        if deep_mode is not None:
            normal_strategy, low_latency_strategy = StrategyMap.get_strategy(
                deep_mode.lower()
            )

        # Initialize normal mode strategy
        self._init_normal_strategy(normal_strategy)

        # Initialize low latency mode strategy
        self._init_low_latency_strategy(low_latency_strategy)
        self._init_fused_strategies()

    def _init_normal_strategy(self, strategy: Union[str, NormalStrategy]):
        """Initialize normal mode communication strategy"""
        if isinstance(strategy, NormalStrategy):
            strategy = strategy.value
        strategy_cls = get_normal_strategy(strategy)

        self.normal_strategy = strategy_cls(
            runtime=self.runtime,
            group=self.group,
        )

    def _init_low_latency_strategy(
        self, strategy: Union[str, NormalStrategy], comm_alg: str = "hierarchy"
    ):
        """Initialize low latency mode communication strategy"""
        if isinstance(strategy, LowLatencyStrategy):
            strategy = strategy.value
        strategy_cls = get_low_latency_strategy(strategy)

        # Pass different init kwargs based on strategy type
        init_kwargs = {
            "runtime": self.runtime,
            "group": self.group,
        }
        if strategy == "ops":
            init_kwargs["comm_alg"] = comm_alg

        self.low_latency_strategy = strategy_cls(**init_kwargs)

    def _init_fused_strategies(self):
        self._fused_strategies = {
            "deep_ep": get_fused_strategy("deep_ep")(),
            "mega_moe": get_fused_strategy("mega_moe")(),
        }

    def __del__(self):
        try:
            for strategy in getattr(self, "_fused_strategies", {}).values():
                strategy.destroy()
        except Exception:
            pass

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 24, 256, 6, 128),
            4: Config(Buffer.num_sms, 6, 256, 6, 128),
            8: Config(Buffer.num_sms, 6, 256, 6, 128),
            16: Config(Buffer.num_sms, 36, 288, 20, 128),
            24: Config(Buffer.num_sms, 8, 288, 32, 128),
            32: Config(Buffer.num_sms, 32, 288, 32, 128),
            64: Config(Buffer.num_sms, 20, 288, 28, 128),
            128: Config(Buffer.num_sms, 20, 560, 32, 128),
            144: Config(Buffer.num_sms, 32, 720, 12, 128),
            160: Config(Buffer.num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 10, 256, 6, 128),
            4: Config(Buffer.num_sms, 9, 256, 6, 128),
            8: Config(Buffer.num_sms, 4, 256, 6, 128),
            16: Config(Buffer.num_sms, 4, 288, 12, 128),
            24: Config(Buffer.num_sms, 1, 288, 8, 128),
            32: Config(Buffer.num_sms, 1, 288, 8, 128),
            64: Config(Buffer.num_sms, 1, 288, 20, 128),
            128: Config(Buffer.num_sms, 1, 560, 12, 128),
            144: Config(Buffer.num_sms, 2, 720, 8, 128),
            160: Config(Buffer.num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
        return config_map[num_ranks]

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """

        assert new_num_sms % 2 == 0, "The SM count must be even"
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event: the captured event.
        """
        return EventOverlap(EventHandle())

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        return deep_ep_cpp.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )

    # noinspection PyTypeChecker
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
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            num_experts: the number of experts.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.int`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Delegate to normal strategy
        return self.normal_strategy.get_dispatch_layout(
            topk_idx=topk_idx,
            num_experts=num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

    # internal interface, Only use in test
    def get_notify_send_data(self) -> torch.Tensor:
        """
        Internal interface, we only use it to check the output of get_dispatch_layout.

        Returns:
            notify_send_data: the member variable of buffer, which usually contains the output of get_dispatch_layout.
        """
        notify_send_data = self.runtime.get_notify_send_data()
        return notify_send_data

    def clean_low_latency_buffer(
        self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int
    ) -> None:
        """
        Compatibility hook for cleaning low-latency buffers.

        The current backend implementation is a no-op and does not clear any device/RDMA buffer. This method is kept for
        API compatibility with DeepEP callers that invoke it when switching from normal mode to low-latency mode.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_experts: the number of all experts.
        """
        self.runtime.clean_low_latency_buffer(
            num_max_dispatch_tokens_per_rank, hidden, num_experts
        )

    # noinspection PyTypeChecker
    @log_parameters(["topk_idx"])
    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
        use_fp8: bool = False,
        use_mxfp4: bool = False,
        use_mxfp8: bool = False,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via HCCS.
        Internode kernels require the ranks in a node should be visible via HCCS, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: input tokens, ``torch.Tensor`` with ``torch.bfloat16``, shaped ``[num_tokens, hidden]``.
                The dtype of ``x`` is no longer used for quantization-mode detection; use
                the ``use_fp8`` / ``use_mxfp4`` / ``use_mxfp8`` bool flags instead.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
                will be CUDA-graph compatible. Please also notice that this flag is for intranode only.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.
            dispatch_wait_recv_cost_stats: `[num_ranks]` with `torch.int`, record the time it takes for the dispatch phase
                to receive all tokens from each slave rank in the current rank.
            use_fp8: enable FP8-family quantization. On A5 → ``pertoken_fp8_e4m3``;
                on A2/A3 → ``int8``.
            use_mxfp4: enable MXFP4 per-block quantization → ``mx_fp4_e2m1`` (A5 only).
                Raises ``NotImplementedError`` on A2/A3.
            use_mxfp8: enable MXFP8 per-block quantization → ``mx_fp8_e4m3`` (A5 only).
                Raises ``NotImplementedError`` on A2/A3.

        Returns:
            recv_x: received tokens. The format depends on quantization mode:
                - BF16 (no quantization): a `torch.Tensor` shaped `[received_token_count, hidden]` with `torch.bfloat16`.
                - INT8: a tuple, first element shaped `[received_token_count, hidden]`
                  with `torch.int8`, second element shaped `[received_token_count]` with `torch.float32` (per-token scales).
                - PerToken FP8 (A5): a tuple, first element shaped `[received_token_count, hidden]`
                  with `torch.float8_e4m3fn`, second element shaped `[received_token_count]` with `torch.float32`.
                - MXFP8 (A5): a tuple, first element shaped `[received_token_count, hidden]`
                  with `torch.float8_e4m3fn`, second element shaped
                  `[received_token_count, hidden // 32]` with `torch.float8_e8m0fnu` (per-block E8M0 scales).
                - MXFP4 (A5): a tuple, first element shaped `[received_token_count, hidden / 2]`
                  with `torch.float4_e2m1fn_x2`, second element shaped
                  `[received_token_count, hidden // 32]` with `torch.float8_e8m0fnu`.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                will be empty.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_dispatch_config(self.group_size) if config is None else config

        quant_mode = _resolve_quant_mode(use_fp8, use_mxfp4, use_mxfp8)
        if quant_mode is None:
            is_quant_env = os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT", "0")
            quant_mode = "int8" if is_quant_env == "1" else "bf16"

        # Delegate to normal strategy
        return self.normal_strategy.dispatch(
            x=x,
            handle=handle,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens,
            config=config,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            dispatch_wait_recv_cost_stats=dispatch_wait_recv_cost_stats,
            quant_mode=quant_mode,
        )

    @log_parameters(["topk_idx"])
    def notify_verify(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Default config
        config = self.get_dispatch_config(self.group_size) if config is None else config
        # Launch the kernel with cached or non-cached mode
        x_scales = None
        use_quant = os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT") == "1"

        if handle is not None:
            raise NotImplementedError(
                "Optional communication handle is not supported yet."
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )
            (
                recv_data,
                recv_count,
                recv_offset,
                expert_global_offset,
                srcrank_in_expert_offset,
                C,
                total_recv_token,
                max_bs,
                recv_tokens_per_expert,
            ) = self.runtime.notify_verify(
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
            return (
                recv_data,
                recv_count,
                recv_offset,
                expert_global_offset,
                srcrank_in_expert_offset,
                C,
                total_recv_token,
                max_bs,
                recv_tokens_per_expert,
            )

    @log_parameters()
    def combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        combine_send_cost_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via HCCS.
        Internode kernels require the ranks in a node should be visible via HCCS, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.
            combine_send_cost_stats: `[num_ranks]`: record the time when the current rank sends all tokens to other ranks
                in the combine phase.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_combine_config(self.group_size) if config is None else config

        # Delegate to normal strategy
        return self.normal_strategy.combine(
            x=x,
            handle=handle,
            topk_weights=topk_weights,
            bias=bias,
            config=config,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            combine_send_cost_stats=combine_send_cost_stats,
        )

    def internode_dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.
        """
        return self.normal_strategy._internode_dispatch(
            x=x,
            handle=handle,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            expert_alignment=expert_alignment,
            config=config,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

    def internode_combine(
        self,
        x: torch.Tensor,
        handle: Union[tuple, list],
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
        """
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

        # Launch the kernel
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

    # noinspection PyTypeChecker
    @log_parameters(["topk_idx"])
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
        use_mxfp4: bool = False,
        use_mxfp8: bool = False,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable
    ]:
        """
        A low-latency implementation for dispatch.

        Arguments:
            x: `torch.Tensor` with `torch.bfloat16`, shaped as `[num_tokens, hidden]`, only several hidden shapes are
                supported. The number of tokens to be dispatched must be less than `num_max_dispatch_tokens_per_rank`.
            topk_idx: `torch.Tensor` with `torch.int64`, shaped as `[num_tokens, num_topk]`, only several top-k shapes
                are supported. `-1` indices (not selecting any expert) are supported.
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            num_experts: the number of all experts.
            cumulative_local_expert_recv_stats: a cumulative expert count tensor for statistics, which should have shape
                `[num_local_experts]` and be typed as `torch.int`. This is useful for online service EP load balance
                monitoring.
            use_fp8: selects per-token FP8 on A5 and falls back to INT8 on A2/A3.
            round_scale: whether to round the scaling factors into power of 2.
            use_ue8m0: legacy alias for MXFP8 when `use_fp8=True`.
            use_mxfp4: selects MXFP4 on A5; unsupported on A2/A3.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            use_mxfp8: enable MXFP8 per-block quantization → ``mx_fp8_e4m3`` (A5 only).
                Raises ``NotImplementedError`` on A2/A3.

        Quantization selection priority for the default strategy is `use_mxfp4`, `use_mxfp8` (including the legacy
        `use_fp8=True, use_ue8m0=True` alias), `use_fp8`, the deprecated
        `DEEP_NORMAL_MODE_USE_INT8_QUANT=1` fallback, and finally BF16. Since `use_fp8` defaults to `True`, callers
        must pass `use_fp8=False` to reach the environment-variable or BF16 fallback.

        Returns:
            recv_x: received tokens. The format depends on quantization mode:
                - BF16: a `torch.Tensor` shaped `[num_max_tokens, hidden]` with `torch.bfloat16`.
                - INT8 or scalar FP8: a tuple containing quantized data and one `torch.float32` scale per token.
                - MXFP8: a tuple of two tensors. The first is shaped
                  `[num_max_tokens, hidden]`, the second is shaped
                  `[num_max_tokens * hidden / 32]` with `torch.float8_e8m0fnu` (per-block scales, one scale per
                  32-element block).
                Not all tokens are valid; only the first `recv_count` tokens per expert contain meaningful data.
            recv_count: a tensor shaped `[num_local_experts]` with type `torch.int64`, indicating how many tokens each
                expert receives.
            handle: the communication handle to be used in the `low_latency_combine` function.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        quant_mode = _resolve_quant_mode(use_fp8, use_mxfp4, use_mxfp8)

        return self.low_latency_strategy.low_latency_dispatch(
            x=x,
            topk_idx=topk_idx,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            num_experts=num_experts,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            use_fp8=use_fp8,
            round_scale=round_scale,
            use_ue8m0=use_ue8m0,
            use_mxfp4=use_mxfp4,
            async_finish=async_finish,
            return_recv_hook=return_recv_hook,
            topk_weights=topk_weights,
            quant_mode=quant_mode,
        )

    @log_parameters(["topk_idx"])
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
        """
        A low-latency implementation for combine.

        Arguments:
            x: `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`,
                the local calculated tokens to be sent to this original rank and reduced.
            topk_idx: `[num_combined_tokens, num_topk]` with `torch.int64`, the expert indices selected by the dispatched
                tokens. `-1` indices (not selecting any expert) are supported. Note that, `num_combined_tokens` equals
                to the number of dispatched tokens.
            topk_weights: `[num_combined_tokens, num_topk]` with `torch.float`, the expert weights selected by the dispatched
                tokens. The received tokens will be reduced with the weights in this tensor.
            handle: the communication handle given by the `dispatch` function.
            zero_copy: whether the tensor is already copied into the RDMA buffer, should be cooperative
                with `get_next_low_latency_combine_buffer`.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.
            out: the in-place output tensor, if set, the kernel will write the result to this tensor and return it directly.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        # Delegate to low latency strategy
        return self.low_latency_strategy.low_latency_combine(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            handle=handle,
            zero_copy=zero_copy,
            async_finish=async_finish,
            return_recv_hook=return_recv_hook,
            out=out,
        )

    @staticmethod
    def _validate_activation_clamp(
        activation_clamp: Optional[float],
    ) -> Optional[float]:
        if activation_clamp is None or activation_clamp == 0:
            return None
        if activation_clamp < 0:
            raise ValueError("`activation_clamp` must be None or >= 0.")
        return activation_clamp

    def _resolve_fused_backend(
        self,
        *,
        backend: str,
        activation: str,
        l1_bias: Optional[TensorOrTensors],
        l2_bias: Optional[TensorOrTensors],
    ) -> str:
        if backend not in ("auto", "deep_ep", "mega_moe"):
            raise ValueError(
                f"Unsupported backend {backend!r}. Expected one of "
                "`auto`, `deep_ep`, or `mega_moe`."
            )
        if backend == "auto":
            if activation == "situ" or l1_bias is not None or l2_bias is not None:
                return "mega_moe"
            return "deep_ep"
        return backend

    def fused_deep_moe(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        gmm1_permuted_weight: TensorOrTensors,
        gmm1_permuted_weight_scale: Optional[TensorOrTensors],
        gmm2_weight: TensorOrTensors,
        gmm2_weight_scale: Optional[TensorOrTensors],
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        quant_mode: int = 1,
        fuse_mode: FuseMode = FuseMode.FUSED_DEEP_MOE,
        activation: Optional[str] = "swiglu",
        beta: Optional[float] = 4.0,
        linear_beta: Optional[float] = 25.0,
        profile_enable: bool = False,
        *,
        backend: str = "auto",
        l1_bias: Optional[TensorOrTensors] = None,
        l2_bias: Optional[TensorOrTensors] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused MoE forward entrypoint with backend routing between `deep_ep` and
        `cann_ops_transformer.ops.mega_moe`.

        Arguments:
            x: `[bs, hidden]` token tensor. The hidden dimension defines the
                mega_moe `hidden` parameter.
            topk_idx: `[bs, num_topk]` token-to-expert routing indices. `-1` means
                the token does not select that top-k slot.
            topk_weights: `[bs, num_topk]` routing weights used during combine.
            gmm1_permuted_weight: First-stage expert weights. For `backend="deep_ep"`,
                this preserves the legacy fused kernel layout requirements. For
                `backend="mega_moe"`, this argument is interpreted as mega_moe
                `l1_weights` and must be either a Tensor whose leading dimension is
                the local expert count or a `list[Tensor]` of per-expert weights in
                mega_moe layout `[hidden, 2 * intermediate_hidden]` for
                A16W16/A8W8-INT, or packed INT4 layout exposed as `torch.int32`
                with shape `[hidden, (2 * intermediate_hidden) // 8]` for A8W4-INT.
            gmm1_permuted_weight_scale: First-stage weight scales. Required by the
                deep_ep backend. Optional for mega_moe A16W16, required for mega_moe
                A8W8-INT/A8W4-INT. For mega_moe, accepts either a Tensor with leading
                local-expert dimension or a `list[Tensor]`.
            gmm2_weight: Second-stage expert weights. For `backend="mega_moe"`, this
                is interpreted as mega_moe `l2_weights` and must use layout
                `[intermediate_hidden, hidden]` per local expert for
                A16W16/A8W8-INT, or packed INT4 layout exposed as `torch.int32`
                with shape `[intermediate_hidden, hidden // 8]` for A8W4-INT.
            gmm2_weight_scale: Second-stage weight scales. Same backend and quantized
                scene rules as `gmm1_permuted_weight_scale`.
            num_max_dispatch_tokens_per_rank: Maximum token count participating in EP
                dispatch for each rank. This value is forwarded to either backend and
                is also part of the mega_moe SymmBuffer cache key. On Atlas A3, the
                mega_moe operator supports at most 4096 tokens per rank in a single
                invocation.
            num_experts: Global expert count. For mega_moe, it must be divisible by
                the process-group size so that local expert counts are well-defined.
            quant_mode: Quantization mode. The deep_ep backend consumes this value
                directly. The mega_moe backend maps quantized execution to its
                internal dispatch mode and infers W8/W4 from weights, scales, and bias.
            fuse_mode: Fused execution mode. The deep_ep backend supports both
                `FuseMode.FUSED_DEEP_MOE` and `FuseMode.DISPATCH_FFN_COMBINE`.
                The mega_moe backend supports only `FuseMode.FUSED_DEEP_MOE`.
            activation: Activation name. Supported values are `"swiglu"`,
                `"swiglu_gpt_oss"`, and `"situ"` on the mega_moe backend. The
                deep_ep backend supports `"swiglu"` and `"situ"` (`"situ"`
                only with `FuseMode.FUSED_DEEP_MOE`).
            beta: Optional beta parameter for the `"situ"` activation. Defaults
                to `4.0`.
            linear_beta: Optional linear beta for the `"situ"` activation linear
                branch. Defaults to `25.0`.
            profile_enable: Whether to enable fused-kernel profiling for the deep_ep
                fused backend. Defaults to `False`.
            backend: Keyword-only backend selector. Supported values:
                - `"deep_ep"`: native fused kernels (`aclnnFusedDeepMoe` /
                  `aclnnDispatchFFNCombine`);
                - `"mega_moe"`: `cann_ops_transformer.ops.mega_moe`; supports
                  `"swiglu"`, `"swiglu_gpt_oss"`, and `"situ"`.
                - `"auto"` (default): routes to mega_moe only when
                  `activation == "situ"` or `l1_bias`/`l2_bias` is provided;
                  otherwise keeps the deep_ep path. Note `"swiglu_gpt_oss"`
                  is not auto-routed and needs `backend="mega_moe"`.
            l1_bias: Optional per-expert first-stage bias tensors used for mega_moe
                A8W4-INT compensation. Unsupported on the deep_ep backend.
            l2_bias: Optional per-expert second-stage bias tensors used for mega_moe
                A8W4-INT compensation. Unsupported on the deep_ep backend.

        Returns:
            A tuple `(output, aux)` where `output` is the fused expert output tensor.
            The `aux` tensor is backend-dependent:
            - deep_ep + `FuseMode.FUSED_DEEP_MOE`: `ep_recv_count`,
              shape `[num_local_experts * num_ranks]`
            - deep_ep + `FuseMode.DISPATCH_FFN_COMBINE`: `expert_token_nums`,
              shape `[num_local_experts]`
            - mega_moe: `expert_token_nums`, shape `[num_local_experts]`
        """
        resolved_backend = self._resolve_fused_backend(
            backend=backend,
            activation=activation,
            l1_bias=l1_bias,
            l2_bias=l2_bias,
        )
        strategy = self._fused_strategies[resolved_backend]
        if x.size(0) == 0:
            x = torch.zeros(
                (1, x.size(1)),
                dtype=x.dtype,
                device=x.device,
            )

            topk_idx = torch.arange(
                topk_idx.size(1),
                dtype=topk_idx.dtype,
                device=topk_idx.device,
            ).unsqueeze(0)

            topk_weights = torch.zeros(
                (1, topk_weights.size(1)),
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            )
        strategy_dispatch_quant_mode = (
            2 if resolved_backend == "mega_moe" and quant_mode == 1 else None
        )
        strategy_dispatch_quant_out_dtype = (
            torch.int8 if resolved_backend == "mega_moe" and quant_mode == 1 else None
        )
        output, expert_token_num = strategy.run(
            buffer=self,
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            gmm1_permuted_weight=gmm1_permuted_weight,
            gmm1_permuted_weight_scale=gmm1_permuted_weight_scale,
            gmm2_weight=gmm2_weight,
            gmm2_weight_scale=gmm2_weight_scale,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            num_experts=num_experts,
            quant_mode=quant_mode,
            fuse_mode=fuse_mode,
            activation=activation,
            activation_clamp=None,
            beta=beta,
            linear_beta=linear_beta,
            profile_enable=profile_enable,
            l1_bias=l1_bias,
            l2_bias=l2_bias,
            dispatch_quant_mode=strategy_dispatch_quant_mode,
            dispatch_quant_out_dtype=strategy_dispatch_quant_out_dtype,
            max_recv_token_num=0,
        )
        if x.size(0) == 0:
            output = torch.empty(
                (0, x.size(1)),
                dtype=x.dtype,
                device=x.device,
            )

        return output, expert_token_num

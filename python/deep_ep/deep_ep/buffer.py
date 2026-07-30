import os
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple, Union

import deep_ep_cpp
import torch
import torch.distributed as dist
import torch_npu
from deep_ep_cpp import Config, EventHandle

from .ep_strategy import (
    LowLatencyStrategy,
    NormalStrategy,
    StrategyMap,
    get_low_latency_strategy,
    get_normal_strategy,
)
from .utils import EventOverlap, log_parameters

try:
    from cann_ops_transformer.ops import (
        get_symm_buffer_for_mega_moe as _get_symm_buffer_for_mega_moe,
    )
    from cann_ops_transformer.ops import mega_moe as _mega_moe

    _MEGA_MOE_IMPORT_ERROR = None
except ImportError as exc:
    _get_symm_buffer_for_mega_moe = None
    _mega_moe = None
    _MEGA_MOE_IMPORT_ERROR = exc


class FuseMode(IntEnum):
    FUSED_DEEP_MOE = 1
    DISPATCH_FFN_COMBINE = 2


TensorOrTensors = Union[torch.Tensor, List[torch.Tensor]]


class Buffer:

    num_sms: int = 20

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
        deep_mode = os.getenv("DEEP_USE_MODE", "default").lower()

        normal_strategy, low_latency_strategy = StrategyMap.get_strategy(deep_mode)

        # Initialize normal mode strategy
        self._init_normal_strategy(normal_strategy)

        # Initialize low latency mode strategy
        self._init_low_latency_strategy(low_latency_strategy)
        self._mega_moe_symm_buffer_cache: Dict[tuple, object] = {}

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

    def _destroy_mega_moe_symm_buffers(self) -> None:
        for sym_buffer in self._mega_moe_symm_buffer_cache.values():
            try:
                sym_buffer.destroy()
            except Exception:
                pass
        self._mega_moe_symm_buffer_cache.clear()

    def __del__(self):
        try:
            self._destroy_mega_moe_symm_buffers()
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
        quant_mode: Optional[str] = None,
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
            x: input tokens. Supports two formats:
                - `torch.Tensor` with `torch.bfloat16`, shaped `[num_tokens, hidden]`. Quantization is controlled by
                  the `DEEP_NORMAL_MODE_USE_INT8_QUANT` environment variable (set to `1` for INT8 quantization, **deprecated**).
                - Tuple of two `torch.Tensor`: for MXFP8 quantization, the first element is shaped `[num_tokens, hidden]`
                  with `torch.float8_e4m3fn` (pre-quantized data), the second is shaped `[num_tokens, hidden // 32]`
                  with `torch.float8_e8m0fnu` (per-block E8M0 scales). On NPU, this triggers MXFP8 per-block quantization
                  (quant_mode=3) inside the dispatch kernel.
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

        Returns:
            recv_x: received tokens. The format depends on quantization mode:
                - BF16 (no quantization): a `torch.Tensor` shaped `[received_token_count, hidden]` with `torch.bfloat16`.
                - INT8 (`DEEP_NORMAL_MODE_USE_INT8_QUANT=1`, **deprecated**): a tuple, first element shaped `[received_token_count, hidden]`
                  with `torch.int8`, second element shaped `[received_token_count]` with `torch.float32` (per-token scales).
                - MXFP8 (tuple input with `float8_e4m3fn` + `float8_e8m0fnu`, A5/C310 only): a tuple, first element shaped
                  `[received_token_count, hidden]` with `torch.float8_e4m3fn`, second element shaped
                  `[received_token_count, hidden // 32]` with `torch.float8_e8m0fnu` (per-block E8M0 scales).
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
        x_scales = None
        use_quant = False
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
        use_mxfp4: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        topk_weights: Optional[torch.Tensor] = None,
        quant_mode: Optional[str] = None,
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
            use_fp8: deprecated for the default low-latency strategy and ignored when selecting its quantization mode.
            round_scale: whether to round the scaling factors into power of 2.
            use_ue8m0: deprecated for the default low-latency strategy and ignored when selecting its quantization mode.
            use_mxfp4: deprecated for the default low-latency strategy and ignored when selecting its quantization mode.
            quant_mode: quantization mode used by the default low-latency strategy. Supported values are `None`,
                `int8`, `mx_fp8_e4m3`, `mx_fp8_e5m2`, `pertoken_fp8_e4m3`, `pertoken_fp8_e5m2`, and `mx_fp4_e2m1`.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.

        Returns:
            recv_x: received tokens. The format depends on quantization mode:
                - BF16 (`quant_mode=None`): a `torch.Tensor` shaped `[num_max_tokens, hidden]` with `torch.bfloat16`.
                - INT8 or scalar FP8: a tuple containing quantized data and one `torch.float32` scale per token.
                - MXFP8 (`quant_mode="mx_fp8_e4m3"` or `"mx_fp8_e5m2"`): a tuple of two tensors. The first is shaped
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
        # Preserve the legacy quantization behavior and return structure when callers do not pass quant_mode.
        if quant_mode is None:
            if use_mxfp4:
                quant_mode = "mx_fp4_e2m1"
            elif use_fp8 and use_ue8m0:
                quant_mode = "mx_fp8_e4m3"
            elif use_fp8:
                quant_mode = "int8"

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

    def _require_mega_moe_ops(self) -> Tuple[Callable, Callable]:
        if _get_symm_buffer_for_mega_moe is None or _mega_moe is None:
            raise ImportError(
                "The mega_moe backend requires the optional dependency "
                "`cann_ops_transformer`. Install or expose `cann_ops_transformer.ops` "
                'before calling `Buffer.fused_deep_moe(..., backend="mega_moe")`.'
            ) from _MEGA_MOE_IMPORT_ERROR
        return _get_symm_buffer_for_mega_moe, _mega_moe

    @staticmethod
    def _normalize_expert_param(
        param: Optional[TensorOrTensors],
        name: str,
        expected_num_local_experts: int,
    ) -> Optional[List[torch.Tensor]]:
        if param is None:
            return None
        if isinstance(param, list):
            if len(param) != expected_num_local_experts:
                raise ValueError(
                    f"`{name}` must contain exactly {expected_num_local_experts} "
                    f"local expert tensors, but got {len(param)}."
                )
            return param
        if not isinstance(param, torch.Tensor):
            raise TypeError(
                f"`{name}` must be a Tensor, a list[Tensor], or None, "
                f"but got {type(param)}."
            )
        if param.dim() == 0:
            raise ValueError(f"`{name}` must not be a scalar tensor.")
        if param.size(0) != expected_num_local_experts:
            raise ValueError(
                f"`{name}` must have leading dimension {expected_num_local_experts} "
                f"for local experts, but got shape {tuple(param.shape)}."
            )
        return [param[i] for i in range(expected_num_local_experts)]

    @staticmethod
    def _is_zero_like_linear_beta(linear_beta: Optional[float]) -> bool:
        return linear_beta is None or linear_beta == 0

    @staticmethod
    def _is_default_beta(beta: float) -> bool:
        return beta == 1.0

    @staticmethod
    def _validate_activation_clamp(
        activation_clamp: Optional[float],
    ) -> Optional[float]:
        if activation_clamp is None or activation_clamp == 0:
            return None
        if activation_clamp < 0:
            raise ValueError("`activation_clamp` must be None or >= 0.")
        return activation_clamp

    @staticmethod
    def _infer_mega_moe_quant_config(
        l1_weights_sf: Optional[List[torch.Tensor]],
        l2_weights_sf: Optional[List[torch.Tensor]],
        l1_bias: Optional[List[torch.Tensor]],
        l2_bias: Optional[List[torch.Tensor]],
        dispatch_quant_mode: Optional[int],
        dispatch_quant_out_dtype: Optional[torch.dtype],
    ) -> Tuple[int, Optional[torch.dtype]]:
        has_scales = l1_weights_sf is not None or l2_weights_sf is not None
        has_bias = l1_bias is not None or l2_bias is not None
        if has_scales and (l1_weights_sf is None or l2_weights_sf is None):
            raise ValueError(
                "`gmm1_permuted_weight_scale` and `gmm2_weight_scale` must both be "
                "provided for mega_moe quantized execution."
            )
        if has_bias and (l1_bias is None or l2_bias is None):
            raise ValueError(
                "`l1_bias` and `l2_bias` must both be provided for A8W4-INT "
                "mega_moe execution."
            )

        resolved_quant_mode = (
            2
            if has_scales or has_bias
            else 0 if dispatch_quant_mode is None else dispatch_quant_mode
        )
        if resolved_quant_mode not in (0, 2):
            raise ValueError(
                "`dispatch_quant_mode` only supports 0 (A16W16) or 2 "
                "(A8W8-INT/A8W4-INT) in fused_deep_moe mega_moe backend."
            )
        if resolved_quant_mode == 0:
            if has_scales or has_bias:
                raise ValueError(
                    "Scale and bias tensors are only valid when "
                    "`dispatch_quant_mode=2`."
                )
            if dispatch_quant_out_dtype is not None:
                raise ValueError(
                    "`dispatch_quant_out_dtype` must be None when "
                    "`dispatch_quant_mode=0`."
                )
            return 0, None

        resolved_quant_out_dtype = (
            torch.int8 if dispatch_quant_out_dtype is None else dispatch_quant_out_dtype
        )
        if resolved_quant_out_dtype != torch.int8:
            raise ValueError(
                "`dispatch_quant_out_dtype` must be torch.int8 for "
                "A8W8-INT/A8W4-INT mega_moe execution."
            )
        return resolved_quant_mode, resolved_quant_out_dtype

    def _get_or_create_mega_moe_symm_buffer(
        self,
        *,
        num_experts: int,
        num_max_dispatch_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        max_recv_token_num: int,
        dispatch_quant_mode: int,
        dispatch_quant_out_dtype: Optional[torch.dtype],
        activation: str,
    ):
        get_symm_buffer_for_mega_moe, _ = self._require_mega_moe_ops()
        cache_key = (
            num_experts,
            num_max_dispatch_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            max_recv_token_num,
            dispatch_quant_mode,
            dispatch_quant_out_dtype,
            activation,
        )
        sym_buffer = self._mega_moe_symm_buffer_cache.get(cache_key)
        if sym_buffer is None:
            sym_buffer = get_symm_buffer_for_mega_moe(
                self.group,
                num_experts=num_experts,
                num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                num_topk=num_topk,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
                max_recv_token_num=max_recv_token_num,
                dispatch_quant_mode=dispatch_quant_mode,
                dispatch_quant_out_dtype=dispatch_quant_out_dtype,
            )
            self._mega_moe_symm_buffer_cache[cache_key] = sym_buffer
        return sym_buffer

    def _pad_mega_moe_inputs(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        original_num_tokens = x.size(0)
        if original_num_tokens > num_max_dispatch_tokens_per_rank:
            raise ValueError(
                "The number of input tokens exceeds "
                "`num_max_dispatch_tokens_per_rank`: "
                f"{original_num_tokens} > {num_max_dispatch_tokens_per_rank}."
            )

        x_active_mask = torch.zeros(
            num_max_dispatch_tokens_per_rank,
            dtype=torch.int8,
            device=x.device,
        )
        x_active_mask[:original_num_tokens] = 1

        if original_num_tokens == num_max_dispatch_tokens_per_rank:
            return x, topk_idx, topk_weights, x_active_mask, original_num_tokens

        padding_size = num_max_dispatch_tokens_per_rank - original_num_tokens
        x_padded = torch.cat(
            (x, x.new_zeros((padding_size, x.size(1)))),
            dim=0,
        )
        topk_idx_padded = torch.cat(
            (
                topk_idx,
                topk_idx.new_zeros((padding_size, topk_idx.size(1))),
            ),
            dim=0,
        )
        topk_weights_padded = torch.cat(
            (
                topk_weights,
                topk_weights.new_zeros((padding_size, topk_weights.size(1))),
            ),
            dim=0,
        )
        return (
            x_padded,
            topk_idx_padded,
            topk_weights_padded,
            x_active_mask,
            original_num_tokens,
        )

    def _resolve_fused_backend(
        self,
        *,
        backend: str,
        activation: str,
        l1_bias: Optional[TensorOrTensors],
        l2_bias: Optional[TensorOrTensors],
        dispatch_quant_mode: Optional[int],
    ) -> str:
        if backend not in ("auto", "deep_ep", "mega_moe"):
            raise ValueError(
                f"Unsupported backend {backend!r}. Expected one of "
                "`auto`, `deep_ep`, or `mega_moe`."
            )
        if backend == "auto":
            if activation == "situ":
                return "mega_moe"
            return "deep_ep"
        return backend

    def _fused_deep_moe_with_deep_ep(
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
        quant_mode: int,
        fuse_mode: FuseMode,
        activation: str,
        activation_clamp: Optional[float],
        beta: float,
        linear_beta: Optional[float],
        l1_bias: Optional[TensorOrTensors],
        l2_bias: Optional[TensorOrTensors],
        dispatch_quant_mode: Optional[int],
        dispatch_quant_out_dtype: Optional[torch.dtype],
        max_recv_token_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        activation_clamp = self._validate_activation_clamp(activation_clamp)
        if not self._is_default_beta(beta):
            raise ValueError("`beta` is only supported by the mega_moe backend.")
        if activation != "situ" and not self._is_zero_like_linear_beta(linear_beta):
            raise ValueError('`linear_beta` is only valid when `activation="situ"`.')
        if activation != "swiglu":
            raise ValueError(
                "The deep_ep fused backend only supports activation='swiglu'. "
                "Use backend='mega_moe' for other activation types."
            )
        if activation_clamp is not None:
            raise ValueError(
                "`activation_clamp` is only supported by the mega_moe backend."
            )
        if not self._is_zero_like_linear_beta(linear_beta):
            raise ValueError("`linear_beta` is only supported by the mega_moe backend.")
        if l1_bias is not None or l2_bias is not None:
            raise ValueError(
                "`l1_bias` and `l2_bias` are only supported by the mega_moe backend."
            )
        if dispatch_quant_mode is not None or dispatch_quant_out_dtype is not None:
            raise ValueError(
                "`dispatch_quant_mode` and `dispatch_quant_out_dtype` are only "
                "supported by the mega_moe backend."
            )
        if isinstance(gmm1_permuted_weight, list) or isinstance(gmm2_weight, list):
            raise TypeError(
                "The deep_ep fused backend expects Tensor inputs for "
                "`gmm1_permuted_weight` and `gmm2_weight`."
            )
        if isinstance(gmm1_permuted_weight_scale, list) or isinstance(
            gmm2_weight_scale, list
        ):
            raise TypeError(
                "The deep_ep fused backend expects Tensor inputs for weight scales."
            )
        if gmm1_permuted_weight_scale is None or gmm2_weight_scale is None:
            raise ValueError(
                "The deep_ep fused backend requires both weight scale tensors."
            )

        topk_ids = topk_idx.int()
        if fuse_mode == FuseMode.FUSED_DEEP_MOE:
            gmm1_permuted_weight_scale = gmm1_permuted_weight_scale.float()
            gmm2_weight_scale = gmm2_weight_scale.float()
            output, ep_recv_count = self.runtime.fused_deep_moe(
                x,
                topk_ids,
                gmm1_permuted_weight,
                gmm1_permuted_weight_scale,
                gmm2_weight,
                gmm2_weight_scale,
                topk_weights,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                quant_mode,
            )
            return output, ep_recv_count
        if fuse_mode == FuseMode.DISPATCH_FFN_COMBINE:
            max_output_size = num_max_dispatch_tokens_per_rank
            output, expert_token_nums = self.runtime.dispatch_ffn_combine(
                x,
                topk_ids,
                gmm1_permuted_weight,
                gmm1_permuted_weight_scale,
                gmm2_weight,
                gmm2_weight_scale,
                topk_weights,
                max_output_size,
                num_experts,
                quant_mode,
            )
            return output, expert_token_nums
        raise NotImplementedError(f"Not support fuse_mode:{fuse_mode}")

    def _fused_deep_moe_with_mega_moe(
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
        fuse_mode: FuseMode,
        activation: str,
        activation_clamp: Optional[float],
        beta: float,
        linear_beta: Optional[float],
        l1_bias: Optional[TensorOrTensors],
        l2_bias: Optional[TensorOrTensors],
        dispatch_quant_mode: Optional[int],
        dispatch_quant_out_dtype: Optional[torch.dtype],
        max_recv_token_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, mega_moe = self._require_mega_moe_ops()
        activation_clamp = self._validate_activation_clamp(activation_clamp)
        if activation not in ("swiglu", "swiglu_gpt_oss", "situ"):
            raise ValueError(
                f"Unsupported mega_moe activation {activation!r}. Expected one of "
                "`swiglu`, `swiglu_gpt_oss`, or `situ`."
            )
        if activation != "situ" and not self._is_default_beta(beta):
            raise ValueError('`beta` is only valid when `activation="situ"`.')
        if activation != "situ" and not self._is_zero_like_linear_beta(linear_beta):
            raise ValueError('`linear_beta` is only valid when `activation="situ"`.')
        if fuse_mode != FuseMode.FUSED_DEEP_MOE:
            raise NotImplementedError(
                "The mega_moe backend only supports " "FuseMode.FUSED_DEEP_MOE."
            )
        expected_num_local_experts = num_experts // self.group_size
        if expected_num_local_experts * self.group_size != num_experts:
            raise ValueError(
                "`num_experts` must be divisible by the process-group size when "
                "using the mega_moe backend."
            )

        l1_weights = self._normalize_expert_param(
            gmm1_permuted_weight,
            "gmm1_permuted_weight",
            expected_num_local_experts,
        )
        l2_weights = self._normalize_expert_param(
            gmm2_weight,
            "gmm2_weight",
            expected_num_local_experts,
        )
        l1_weights_sf = self._normalize_expert_param(
            gmm1_permuted_weight_scale,
            "gmm1_permuted_weight_scale",
            expected_num_local_experts,
        )
        l2_weights_sf = self._normalize_expert_param(
            gmm2_weight_scale,
            "gmm2_weight_scale",
            expected_num_local_experts,
        )
        l1_bias_list = self._normalize_expert_param(
            l1_bias,
            "l1_bias",
            expected_num_local_experts,
        )
        l2_bias_list = self._normalize_expert_param(
            l2_bias,
            "l2_bias",
            expected_num_local_experts,
        )

        resolved_dispatch_quant_mode, resolved_dispatch_quant_out_dtype = (
            self._infer_mega_moe_quant_config(
                l1_weights_sf,
                l2_weights_sf,
                l1_bias_list,
                l2_bias_list,
                dispatch_quant_mode,
                dispatch_quant_out_dtype,
            )
        )
        is_a8w4_int = l1_bias_list is not None and l2_bias_list is not None

        hidden = x.size(1)
        if not l2_weights or l2_weights[0].dim() < 2:
            raise ValueError(
                "`gmm2_weight` must contain per-expert 2D tensors in mega_moe layout "
                "[intermediate_hidden, hidden]."
            )
        intermediate_hidden = l2_weights[0].shape[-2]
        expected_l2_last_dim = hidden // 8 if is_a8w4_int else hidden
        expected_l1_last_dim = (
            (intermediate_hidden * 2) // 8 if is_a8w4_int else intermediate_hidden * 2
        )
        inferred_scene = (
            "A8W4-INT"
            if is_a8w4_int
            else "A8W8-INT" if resolved_dispatch_quant_mode == 2 else "A16W16"
        )
        if l2_weights[0].shape[-1] != expected_l2_last_dim:
            raise ValueError(
                "`gmm2_weight` has an invalid mega_moe layout for "
                f"{inferred_scene}. Expected first local expert shape "
                f"({intermediate_hidden}, {expected_l2_last_dim}) "
                f"({'packed INT4 via .view(torch.int32)' if is_a8w4_int else 'unpacked'}) "
                f"but got {tuple(l2_weights[0].shape)} with hidden={hidden}."
            )
        if l1_weights[0].dim() < 2:
            raise ValueError(
                "`gmm1_permuted_weight` must contain per-expert 2D tensors in mega_moe "
                "layout [hidden, 2 * intermediate_hidden]."
            )
        if l1_weights[0].shape[-2] != hidden:
            raise ValueError(
                "`gmm1_permuted_weight` must use mega_moe layout "
                "[hidden, 2 * intermediate_hidden] for the mega_moe backend, "
                f"but got first local expert shape {tuple(l1_weights[0].shape)} "
                f"with hidden={hidden}."
            )
        if l1_weights[0].shape[-1] != expected_l1_last_dim:
            raise ValueError(
                "`gmm1_permuted_weight` has an invalid mega_moe layout for "
                f"{inferred_scene}. Expected first local expert shape "
                f"({hidden}, {expected_l1_last_dim}) "
                f"({'packed INT4 via .view(torch.int32)' if is_a8w4_int else 'unpacked'}) "
                f"but got {tuple(l1_weights[0].shape)}."
            )

        sym_buffer = self._get_or_create_mega_moe_symm_buffer(
            num_experts=num_experts,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            num_topk=topk_idx.size(1),
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            max_recv_token_num=max_recv_token_num,
            dispatch_quant_mode=resolved_dispatch_quant_mode,
            dispatch_quant_out_dtype=resolved_dispatch_quant_out_dtype,
            activation=activation,
        )

        (
            x_padded,
            topk_idx_padded,
            topk_weights_padded,
            x_active_mask,
            original_num_tokens,
        ) = self._pad_mega_moe_inputs(
            x,
            topk_idx,
            topk_weights,
            num_max_dispatch_tokens_per_rank,
        )

        output, expert_token_num = mega_moe(
            x=x_padded,
            topk_ids=topk_idx_padded,
            topk_weights=topk_weights_padded,
            l1_weights=l1_weights,
            l2_weights=l2_weights,
            sym_buffer=sym_buffer,
            l1_weights_sf=l1_weights_sf,
            l2_weights_sf=l2_weights_sf,
            l1_bias=l1_bias_list,
            l2_bias=l2_bias_list,
            x_active_mask=x_active_mask,
            activation=activation,
            activation_clamp=activation_clamp,
            beta=beta,
            linear_beta=linear_beta,
        )
        return output[:original_num_tokens], expert_token_num

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
        backend: str = "auto",
        activation: str = "swiglu",
        activation_clamp: Optional[float] = None,
        beta: float = 1.0,
        linear_beta: Optional[float] = None,
        l1_bias: Optional[TensorOrTensors] = None,
        l2_bias: Optional[TensorOrTensors] = None,
        dispatch_quant_mode: Optional[int] = None,
        dispatch_quant_out_dtype: Optional[torch.dtype] = None,
        max_recv_token_num: int = 0,
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
                is also part of the mega_moe SymmBuffer cache key.
            num_experts: Global expert count. For mega_moe, it must be divisible by
                the process-group size so that local expert counts are well-defined.
            quant_mode: Legacy deep_ep quantization mode. Supported by `backend="deep_ep"`
                only. The mega_moe backend infers its scene from scales/biases and
                `dispatch_quant_mode`.
            fuse_mode: Fused execution mode. The deep_ep backend supports both
                `FuseMode.FUSED_DEEP_MOE` and `FuseMode.DISPATCH_FFN_COMBINE`.
                The mega_moe backend supports only `FuseMode.FUSED_DEEP_MOE`.
            backend: Backend selector. `"auto"` keeps the existing A5 deep_ep fused
                path and routes non-A5 or mega_moe-only features to mega_moe.
            activation: Activation name. Supported values are `"swiglu"`,
                `"swiglu_gpt_oss"`, and `"situ"` on the mega_moe backend. The
                deep_ep backend supports only `"swiglu"`.
            activation_clamp: Optional symmetric clamp value applied by the
                mega_moe backend activation implementation. This is independent
                from `linear_beta`, must be `None` or `>= 0`, and is unsupported
                by the legacy deep_ep backend.
            beta: Optional beta parameter for the `"situ"` activation. Defaults
                to `1.0`. Non-default values are unsupported outside
                `activation="situ"`.
            linear_beta: Optional linear beta for the `"situ"` activation linear
                branch. This is distinct from `activation_clamp` and is
                forwarded to mega_moe as `linear_beta=...`.
            l1_bias: Optional per-expert first-stage bias tensors used for mega_moe
                A8W4-INT compensation. Unsupported on the deep_ep backend.
            l2_bias: Optional per-expert second-stage bias tensors used for mega_moe
                A8W4-INT compensation. Unsupported on the deep_ep backend.
            dispatch_quant_mode: Optional mega_moe dispatch quantization selector.
                Supported values in this wrapper are `0` (A16W16) and `2`
                (A8W8-INT/A8W4-INT). Unsupported on the deep_ep backend.
            dispatch_quant_out_dtype: Optional mega_moe dispatch output dtype.
                This wrapper currently supports only `torch.int8` when
                `dispatch_quant_mode=2`.
            max_recv_token_num: Optional mega_moe max receive token hint. Forwarded to
                mega_moe only.

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
            dispatch_quant_mode=dispatch_quant_mode,
        )
        if resolved_backend == "deep_ep":
            return self._fused_deep_moe_with_deep_ep(
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
                activation_clamp=activation_clamp,
                beta=beta,
                linear_beta=linear_beta,
                l1_bias=l1_bias,
                l2_bias=l2_bias,
                dispatch_quant_mode=dispatch_quant_mode,
                dispatch_quant_out_dtype=dispatch_quant_out_dtype,
                max_recv_token_num=max_recv_token_num,
            )
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
        output, expert_token_num = self._fused_deep_moe_with_mega_moe(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            gmm1_permuted_weight=gmm1_permuted_weight,
            gmm1_permuted_weight_scale=gmm1_permuted_weight_scale,
            gmm2_weight=gmm2_weight,
            gmm2_weight_scale=gmm2_weight_scale,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            num_experts=num_experts,
            fuse_mode=fuse_mode,
            activation=activation,
            activation_clamp=activation_clamp,
            beta=beta,
            linear_beta=linear_beta,
            l1_bias=l1_bias,
            l2_bias=l2_bias,
            dispatch_quant_mode=2 if quant_mode == 1 else dispatch_quant_mode,
            dispatch_quant_out_dtype=(
                torch.int8 if quant_mode == 1 else dispatch_quant_out_dtype
            ),
            max_recv_token_num=max_recv_token_num,
        )
        if x.size(0) == 0:
            output = torch.empty(
                (0, x.size(1)),
                dtype=x.dtype,
                device=x.device,
            )

        return output, expert_token_num

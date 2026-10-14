"""
Ops-Transformer EP communication strategy.

This strategy delegates normal-mode dispatch/combine to ops-transformer's
``ElasticBuffer``, which internally runs a four-operator URMA pipeline:

    moe_ep_dispatch -> moe_ep_dispatch_epilogue
        -> [expert FFN]
    -> moe_ep_combine -> moe_ep_combine_epilogue

Activation
----------
Set ``DEEP_USE_MODE=ops_transformer`` or pass ``normal_strategy="ops_transformer"``
to ``Buffer.__init__``.

Requirements
------------
``cann-ops-transformer`` must be installed and importable as
``cann_ops_transformer``.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu

from ..ep_strategy import NormalEPCommStrategy, register_normal_strategy
from ..utils import EventOverlap

logger = logging.getLogger(__name__)


class _EPEventAdapter(EventOverlap):
    """Adapt ops-transformer ``_EPEpilogueEvent`` to the ``EventOverlap`` interface.

    Reserved for future true-async support.  Currently the strategy always
    runs synchronously (see ``dispatch`` / ``combine`` docstrings), so this
    class is not instantiated at runtime.

    When ``async_with_compute_stream=True`` is used, ``ElasticBuffer`` returns
    an ``_EPEpilogueEvent`` whose ``current_stream_wait()`` submits the
    deferred epilogue kernel and returns the result tensors.  This wrapper
    would bridge that to the ``EventOverlap`` contract used by deep_ep.
    """

    def __init__(self, ep_event):
        super().__init__()
        self._ep_event = ep_event
        self._result = None

    def current_stream_wait(self):
        if self._ep_event is not None and self._result is None:
            self._result = self._ep_event.current_stream_wait()
        return self._result


@register_normal_strategy("ops_transformer")
class OpsTransformerNormalStrategy(NormalEPCommStrategy):
    """Normal-mode strategy backed by ops-transformer ``ElasticBuffer``.

    Instead of calling ``deep_ep_cpp`` single-op kernels, this strategy creates
    an ``ElasticBuffer`` (lazily, on first ``dispatch``) and delegates to its
    ``dispatch`` / ``combine`` methods.  The four-operator pipeline
    (dispatch + epilogue + combine + epilogue) runs entirely inside
    ``ElasticBuffer``; this class is a thin I/O-mapping adapter.
    """

    _SUPPORTED_QUANT_MODES = frozenset({"bf16", "int8"})

    def __init__(self, runtime, group: dist.ProcessGroup):
        super().__init__(group)
        # *runtime* is the deep_ep_cpp.Buffer instance created by Buffer.__init__.
        # It is retained for API compatibility but not used for EP communication.
        self._runtime = runtime
        self._elastic_buf = None
        self._num_experts: Optional[int] = None
        self._hidden: Optional[int] = None
        self._num_topk: Optional[int] = None
        self._num_max_tokens_per_rank: Optional[int] = None

    def get_name(self) -> str:
        return "ops_transformer"

    def get_supported_modes(self) -> List[str]:
        return ["normal"]

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------
    def _ensure_elastic_buffer(
        self,
        hidden: int,
        num_topk: int,
        num_experts: int,
        num_max_tokens_per_rank: int,
    ) -> None:
        """Create ``ElasticBuffer`` on first use.

        ``ElasticBuffer`` requires ``hidden``, ``num_topk`` and
        ``num_max_tokens_per_rank`` at construction time, but these are only
        available at the first ``dispatch`` call.  Subsequent calls reuse the
        same instance.
        """
        if self._elastic_buf is not None:
            return

        try:
            from cann_ops_transformer.mc2 import ElasticBuffer
        except ImportError:
            raise ImportError(
                "cann-ops-transformer is required for the 'ops_transformer' "
                "strategy. Install it and ensure 'cann_ops_transformer' is "
                "importable."
            )

        if num_max_tokens_per_rank is None or num_max_tokens_per_rank <= 0:
            num_max_tokens_per_rank = max(1, 128)

        self._hidden = hidden
        self._num_topk = num_topk
        self._num_experts = num_experts
        self._num_max_tokens_per_rank = num_max_tokens_per_rank

        logger.info(
            "[rank %s] Creating ElasticBuffer(hidden=%d, num_topk=%d, "
            "num_experts=%d, nmt=%d)",
            self.rank,
            hidden,
            num_topk,
            num_experts,
            num_max_tokens_per_rank,
        )
        self._elastic_buf = ElasticBuffer(
            self.group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden,
            num_topk=num_topk,
        )

    # ------------------------------------------------------------------
    # NormalEPCommStrategy interface
    # ------------------------------------------------------------------
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
        """Compute dispatch layout via PyTorch ops.

        ``ElasticBuffer`` computes routing internally, so the returned layout
        is informational only.  However, it is used to determine
        ``num_max_tokens_per_rank`` (global max) for ``ElasticBuffer``
        construction.
        """
        self._num_experts = num_experts
        group_size = self.group_size
        device = topk_idx.device
        num_local_experts = num_experts // group_size

        # Local per-expert histogram
        num_tokens_per_expert_local = torch.histc(
            topk_idx, bins=num_experts, min=0, max=num_experts
        )

        # Global per-expert histogram via all-gather
        num_global_tokens_per_expert = self._gather_along_first_dim(
            num_tokens_per_expert_local, self.group
        )

        # Per-rank token counts (global)
        num_tokens_per_rank = num_global_tokens_per_expert.reshape(
            group_size, num_local_experts
        ).sum(dim=1)

        # Determine num_max_tokens_per_rank (must be consistent across ranks)
        self._num_max_tokens_per_rank = int(num_tokens_per_rank.max().item())

        # is_token_in_rank: [num_tokens, num_ranks] bool
        is_token_in_rank = torch.zeros(
            (topk_idx.size(0), group_size), dtype=torch.bool, device=device
        )
        for r in range(group_size):
            lo = r * num_local_experts
            hi = lo + num_local_experts
            mask = (topk_idx >= lo) & (topk_idx < hi)
            is_token_in_rank[:, r] = mask.any(dim=-1)

        return (
            num_tokens_per_rank,
            None,  # num_tokens_per_rdma_rank — N/A for ops-transformer
            num_global_tokens_per_expert,
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
        if quant_mode is not None and quant_mode not in self._SUPPORTED_QUANT_MODES:
            raise NotImplementedError(
                f"quant_mode '{quant_mode}' is not supported by the ops_transformer "
                f"strategy. Only {self._SUPPORTED_QUANT_MODES} are supported."
            )
        if handle is not None:
            raise NotImplementedError(
                "Cached handle is not supported by the ops_transformer strategy yet."
            )

        # Separate data and scales (for int8 quant)
        data = x[0] if isinstance(x, tuple) else x
        scales = x[1] if isinstance(x, tuple) else None

        hidden = data.shape[1]
        num_topk = topk_idx.shape[1] if topk_idx is not None else 1
        num_experts = self._num_experts
        if num_experts is None:
            if num_tokens_per_expert is not None:
                num_experts = num_tokens_per_expert.shape[0]
            else:
                raise ValueError(
                    "num_experts is unknown. Call get_dispatch_layout() first."
                )

        # Determine num_max_tokens_per_rank (must be same across all ranks)
        nmt = self._num_max_tokens_per_rank
        if nmt is None:
            if num_tokens_per_rank is not None:
                nmt = int(num_tokens_per_rank.max().item())
            else:
                nmt = data.shape[0]

        self._ensure_elastic_buffer(hidden, num_topk, num_experts, nmt)

        # Build x argument: tuple (data, scales) for quant, plain tensor for bf16
        ep_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        if scales is not None:
            ep_x = (data, scales)
        else:
            ep_x = data

        # ----------------------------------------------------------------
        # Call ElasticBuffer.dispatch()
        #
        # ElasticBuffer internally runs moe_ep_dispatch (send + count exchange)
        # and moe_ep_dispatch_epilogue (receive + staging).
        #
        # NOTE: We always use synchronous mode (async_with_compute_stream=False).
        # ElasticBuffer's async mode defers the epilogue kernel until
        # current_stream_wait() is called, which means the output tensors are
        # not populated.  The sgl framework expects populated tensors even in
        # async mode (with an event for synchronization).  True async support
        # would require ElasticBuffer to expose pre-allocated tensors.
        # ----------------------------------------------------------------
        if async_finish:
            logger.warning(
                "[rank %s] async_finish=True is requested but the ops_transformer "
                "strategy currently runs synchronously.",
                self.rank,
            )

        result = self._elastic_buf.dispatch(
            ep_x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=num_experts,
            num_max_tokens_per_rank=nmt,
            expert_alignment=expert_alignment,
            do_cpu_sync=True,
            async_with_compute_stream=False,
        )

        # Synchronous: result is (recv_x, None, recv_topk_weights, EPHandle)
        recv_x, _, recv_topk_weights, ep_handle = result

        # Extract num_recv_tokens_per_expert list from EPHandle
        num_local_experts = num_experts // self.group_size
        num_recv_list: List[int] = [0] * num_local_experts
        if ep_handle is not None and ep_handle.num_recv_tokens_per_expert is not None:
            t = ep_handle.num_recv_tokens_per_expert
            if t.numel() == num_local_experts:
                num_recv_list = [int(v) for v in t.cpu().tolist()]

        handle = {
            "ep_handle": ep_handle,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
        }

        return (
            recv_x,
            None,  # recv_topk_idx (not directly available from ElasticBuffer)
            recv_topk_weights,
            num_recv_list,
            handle,
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
        if bias is not None:
            raise NotImplementedError(
                "bias is not supported by the ops_transformer strategy."
            )

        ep_handle = handle["ep_handle"]
        stored_weights = handle.get("topk_weights")
        weights = topk_weights if topk_weights is not None else stored_weights

        # ----------------------------------------------------------------
        # Call ElasticBuffer.combine()
        #
        # ElasticBuffer internally runs moe_ep_combine (send back to source
        # ranks via URMA) and moe_ep_combine_epilogue (receive + fp32 reduce).
        #
        # NOTE: Always synchronous (see dispatch() for rationale).
        # ----------------------------------------------------------------
        if async_finish:
            logger.warning(
                "[rank %s] async_finish=True is requested but the ops_transformer "
                "strategy currently runs synchronously.",
                self.rank,
            )

        result = self._elastic_buf.combine(
            x,
            handle=ep_handle,
            topk_weights=weights,
            async_with_compute_stream=False,
        )

        # Synchronous: result is (combined_x, combined_topk_weights) or just combined_x
        if isinstance(result, tuple) and len(result) >= 2:
            combined_x = result[0]
            combined_topk_weights = result[1]
        else:
            combined_x = result if not isinstance(result, tuple) else result[0]
            combined_topk_weights = None

        return combined_x, combined_topk_weights, EventOverlap()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _gather_along_first_dim(
        input_: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        """All-gather a 1-D tensor along dimension 0."""
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return input_
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * world_size
        output = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.npu.current_device()
        )
        dist.all_gather_into_tensor(output, input_.contiguous(), group=group)
        return output

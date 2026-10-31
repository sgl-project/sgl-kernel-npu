"""Public D256 QSA prefill main-attention operator."""

import math

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_helpers = None
_torch_native_fn = None

# Local (Q heads, KV heads) layouts the native kernel launches directly. A
# sixteen-query sharing group occupies 16 * (H / KV) rows of the per-stage GM
# workspace block, which holds 128 rows, so one launch needs H / KV <= 8.
NATIVE_HEAD_LAYOUTS = ((16, 2), (8, 1), (6, 1), (3, 1), (2, 1))
# Query heads per launch for layouts above that limit, and the layouts served
# that way. Attention is independent per query head, so these are evaluated as
# several native launches over consecutive query-head slices.
SPLIT_QUERY_HEADS = 6
SPLIT_HEAD_LAYOUTS = ((24, 2), (12, 1))
SUPPORTED_HEAD_LAYOUTS = NATIVE_HEAD_LAYOUTS + SPLIT_HEAD_LAYOUTS
SELECTED_BLOCKS = 512
HEAD_DIM = 256


def _check_inputs(q, k, v, blocks, request_table, request_row, length, base, scale):
    """Reject inputs outside the operator contract before any kernel launch.

    Both implementations share this contract, so it is checked once here
    instead of being duplicated per backend. Only metadata is inspected; no
    device-to-host data read happens.
    """
    if q.ndim != 3 or q.shape[2] != HEAD_DIM:
        raise ValueError("QSA prefill requires Q [R,H,256]")
    if k.ndim != 3 or k.shape[2] != HEAD_DIM or v.shape != k.shape:
        raise ValueError("QSA prefill requires K,V [capacity,KV,256]")
    heads, kv_heads = q.shape[1], k.shape[1]
    if (heads, kv_heads) not in SUPPORTED_HEAD_LAYOUTS:
        expected = ", ".join(f"{h}/{kv}" for h, kv in SUPPORTED_HEAD_LAYOUTS)
        raise ValueError(f"unsupported Q/KV heads: expected one of {expected}")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError("QSA prefill requires BF16 Q/K/V")
    if q.device.type != "npu" or any(
        x.device != q.device for x in (k, v, blocks, request_table, request_row)
    ):
        raise ValueError("all inputs must share an NPU device")
    if any(not x.is_contiguous() for x in (q, k, v, blocks, request_row)):
        raise ValueError("Q/K/V, blocks and request row must be contiguous")
    if blocks.shape != (q.shape[0], SELECTED_BLOCKS) or blocks.dtype != torch.int32:
        raise ValueError("blocks must be int32 [R,512]")
    if (
        request_table.ndim != 2
        or request_table.dtype != torch.int32
        or request_table.stride(1) != 1
        or request_table.shape[0] == 0
    ):
        raise ValueError("request table must be int32 with unit column stride")
    if request_row.shape != (1,) or request_row.dtype != torch.int64:
        raise ValueError("request row must be int64 [1]")
    if (
        not isinstance(length, int)
        or not isinstance(base, int)
        or not 0 <= length <= request_table.shape[1]
    ):
        raise ValueError("invalid CPU length/base metadata")
    if not -(2**31) <= base or base + q.shape[0] >= 2**31 or k.shape[0] >= 2**31:
        raise ValueError("metadata exceeds int32 indexing")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    if length and not k.shape[0]:
        raise ValueError("nonempty sequences require a nonempty cache")


def _pack_and_merge():
    """Import the Triton helpers once; registration must not import them."""
    global _triton_helpers
    if _triton_helpers is None:
        from ...implementation.triton.attention.qsa_prefill import (
            merge_causal_tail,
            pack_queries,
        )

        _triton_helpers = (pack_queries, merge_causal_tail)

    return _triton_helpers


def _native_launch(q, k, v, blocks, request_table, request_row, length, base, scale):
    """Run one native launch for a layout in NATIVE_HEAD_LAYOUTS.

    Query packing, the selection equality check and the per-query causal tail
    merge are Triton kernels; the QK, online-softmax, PV and rescale pipeline
    and the physical run preparation are Ascend C. Tiling is immutable and
    cached natively by full shape, scale and device, including both local head
    counts. No device-to-host metadata read happens here.
    """
    if length == 0 or q.shape[0] == 0:
        # Without visible tokens every row's output is zero, and no packed-Q
        # write is queued against storage the caller may reuse.
        return torch.zeros_like(q)
    pack_queries, merge_causal_tail = _pack_and_merge()
    packed, counts = pack_queries(q, blocks, k.shape[1])
    starts, runs, counts = torch.ops.npu.qsa_prefill_prepare(
        blocks, request_table, request_row, counts, length, base, k.shape[0]
    )
    base_out, lse = torch.ops.npu.qsa_prefill_runs(
        packed, k, v, starts, runs, counts, scale
    )
    return merge_causal_tail(
        q, k, v, request_table, request_row, base_out, lse, length, base, scale
    )


# Only arch22 builds with the A3-family target enabled register the native
# helpers, so this implementation is not declared for arch35.
@register_impl(
    "attention.qsa_prefill", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "attention.qsa_prefill", "ascendc", supported_generations={Generation.ARCH22}
)
def _ascendc_impl(q, k, v, blocks, request_table, request_row, length, base, scale):
    heads, kv_heads = q.shape[1], k.shape[1]
    if (heads, kv_heads) in NATIVE_HEAD_LAYOUTS:
        return _native_launch(
            q, k, v, blocks, request_table, request_row, length, base, scale
        )
    group = heads // kv_heads
    if group % SPLIT_QUERY_HEADS or (SPLIT_QUERY_HEADS, 1) not in NATIVE_HEAD_LAYOUTS:
        raise ValueError(
            f"{heads}/{kv_heads} cannot be split into native launches of "
            f"{SPLIT_QUERY_HEADS} query heads"
        )
    parts = []
    for kv_head in range(kv_heads):
        if kv_heads == 1:
            keys, values = k, v
        else:
            # One copy per KV head, reused by all of its query-head slices.
            keys = k[:, kv_head : kv_head + 1].contiguous()
            values = v[:, kv_head : kv_head + 1].contiguous()
        for start in range(0, group, SPLIT_QUERY_HEADS):
            head = kv_head * group + start
            parts.append(
                _native_launch(
                    q[:, head : head + SPLIT_QUERY_HEADS].contiguous(),
                    keys,
                    values,
                    blocks,
                    request_table,
                    request_row,
                    length,
                    base,
                    scale,
                )
            )
    return torch.cat(parts, dim=1)


@register_impl("attention.qsa_prefill", "torch_native")
def _torch_native_impl(
    q, k, v, blocks, request_table, request_row, length, base, scale
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.qsa_prefill import qsa_prefill

        _torch_native_fn = qsa_prefill

    return _torch_native_fn(
        q, k, v, blocks, request_table, request_row, length, base, scale
    )


@register_operator("attention.qsa_prefill")
def qsa_prefill(
    q,
    k,
    v,
    blocks,
    request_table,
    request_row,
    length,
    base,
    scale=1 / 16,
    *,
    impl="auto",
):
    """The default impl="auto" selects the Ascend C pipeline.

    BF16 D256 ratio-4 prefill main attention over 512 preselected logical
    blocks per query. Every nonnegative selected block expands into four token
    positions, which are mapped to physical slots through one request-table
    row. Each query then independently appends its causal remainder of 0--3
    tokens. Invalid blocks and invalid physical slots are omitted; selection
    order and duplicates are preserved. This operator neither computes indexer
    scores nor selects Top-K.

    Args:
        q (torch.Tensor): Queries of shape [R, H, 256], BF16, contiguous, NPU.
        k (torch.Tensor): Key cache of shape [capacity, KV, 256], BF16,
            contiguous, on q's device.
        v (torch.Tensor): Value cache with k's shape, dtype and device.
        blocks (torch.Tensor): Selected logical blocks, int32 [R, 512],
            contiguous. Negative or out-of-range entries are ignored.
        request_table (torch.Tensor): Logical-to-physical slot table, int32
            [requests, max_length] with unit column stride.
        request_row (torch.Tensor): Contiguous int64 [1] tensor holding the
            caller's valid row index into request_table.
        length (int): Sequence length in tokens; must not exceed
            request_table.shape[1]. CPU metadata, never read from the device.
        base (int): Logical position preceding q's first row, normally
            length - q.shape[0]. CPU metadata.
        scale (float): Positive finite logit multiplier. Defaults to 1/16.
        impl (str): Implementation name: "auto", "ascendc" or "torch_native".
            Defaults to "auto". Keyword-only.

    Returns:
        torch.Tensor: Newly allocated attention output of shape [R, H, 256]
        with q's dtype and device. Rows without visible tokens are zero.
        Inputs are not modified.

    Supported local (H, KV) layouts are NATIVE_HEAD_LAYOUTS, which the kernel
    takes directly, plus SPLIT_HEAD_LAYOUTS, which the Ascend C
    implementation evaluates as several launches over query-head slices and
    therefore copies those slices. The caller handles head sharding and KV
    replication; this operator performs no distributed communication.
    """
    _check_inputs(q, k, v, blocks, request_table, request_row, length, base, scale)
    impl_fn = get_impl_by_name("attention.qsa_prefill", impl)
    return impl_fn(q, k, v, blocks, request_table, request_row, length, base, scale)

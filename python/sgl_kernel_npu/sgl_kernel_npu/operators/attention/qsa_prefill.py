"""Public D256 QSA prefill main-attention operator."""

import math

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_ascendc_fn = None
_torch_native_fn = None

SUPPORTED_HEAD_LAYOUTS = ((16, 2), (8, 1), (6, 1), (2, 1))
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
        raise ValueError("unsupported Q/KV heads: expected 16/2, 8/1, 6/1 or 2/1")
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


# Only arch22 builds with the A3-family target enabled register the native
# helpers, so this implementation is not declared for arch35.
@register_impl(
    "attention.qsa_prefill", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "attention.qsa_prefill", "ascendc", supported_generations={Generation.ARCH22}
)
def _ascendc_impl(q, k, v, blocks, request_table, request_row, length, base, scale):
    global _ascendc_fn
    if _ascendc_fn is None:
        from ...implementation.ascendc.attention.qsa_prefill import qsa_prefill

        _ascendc_fn = qsa_prefill

    return _ascendc_fn(q, k, v, blocks, request_table, request_row, length, base, scale)


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

    Supported local (H, KV) layouts are (16, 2), (8, 1), (6, 1) and (2, 1).
    The caller handles head sharding and KV replication; this operator
    performs no distributed communication.
    """
    _check_inputs(q, k, v, blocks, request_table, request_row, length, base, scale)
    impl_fn = get_impl_by_name("attention.qsa_prefill", impl)
    return impl_fn(q, k, v, blocks, request_table, request_row, length, base, scale)

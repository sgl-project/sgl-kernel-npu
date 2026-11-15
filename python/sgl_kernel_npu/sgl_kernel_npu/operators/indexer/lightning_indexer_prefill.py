"""Public ratio-4 LightningIndexer prefill selection for one causal request."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_native_fn = None

# Fixed contract of the ratio-4 prefill indexer: four query heads of 128
# elements each, 512 selected keys, and a key compression ratio of four.
QUERY_HEADS = 4
HEAD_DIM = 128
SELECTED_KEYS = 512
COMPRESSION_RATIO = 4
# Pages of the paged-attention key layout the CANN operator reads.
PAGE_SIZE = 64
# Four-query batching only pays off once one launch holds enough rows and keys.
# Below either threshold both paths dispatch the same independent-query work,
# so the switch is observable only as measurement noise.
GROUP_MIN_ROWS = 1024
GROUP_MIN_KEYS = 512


def _check_inputs(q, k, sequence_length, group4):
    """Reject inputs outside the operator contract before any kernel launch.

    Both backends share this contract, so it is checked once here instead of
    being duplicated per backend. Only metadata is inspected; no
    device-to-host data read happens.
    """
    if type(sequence_length) is not int or not 0 <= sequence_length < 2**31:
        raise ValueError("sequence_length must be a nonnegative int32 host integer")
    if type(group4) is not bool:
        raise ValueError("group4 must be a bool")
    if (
        q.ndim != 3
        or k.ndim != 3
        or q.shape[1:] != (QUERY_HEADS, HEAD_DIM)
        or k.shape[1:] != (1, HEAD_DIM)
    ):
        raise ValueError(
            f"expected Q [rows,{QUERY_HEADS},{HEAD_DIM}] and K [keys,1,{HEAD_DIM}]"
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("Q and K must be BF16")
    if q.device.type != "npu" or q.device != k.device:
        raise ValueError("Q and K must be on the same NPU")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("Q and K must be contiguous")
    if (
        q.shape[0] > sequence_length
        or k.shape[0] != sequence_length // COMPRESSION_RATIO
    ):
        raise ValueError("Q/K lengths disagree with sequence_length")
    if q.requires_grad or k.requires_grad:
        raise ValueError("lightning_indexer_prefill is inference only")


def visible_key_counts(rows, sequence_length, device):
    """Return the int32 count of compressed keys each query row may attend to.

    Row ``i`` is logical position ``sequence_length - rows + i``, which makes
    ``floor((sequence_length - rows + i + 1) / 4)`` compressed keys causally
    visible. Shared by both backends so the visibility rule has one definition.
    """
    return (
        torch.arange(rows, device=device, dtype=torch.int32)
        + (sequence_length - rows + 1)
    ) // COMPRESSION_RATIO


# npu_lightning_indexer is reachable on every generation, but the grouped
# launch geometry was measured on arch22 only, so auto is declared there.
@register_impl(
    "indexer.lightning_indexer_prefill",
    "auto",
    supported_generations={Generation.ARCH22},
)
@register_impl(
    "indexer.lightning_indexer_prefill",
    "torch_npu",
    supported_generations={Generation.ARCH22},
)
def _torch_npu_impl(q, k, sequence_length, group4):
    """Adapt one causal request onto CANN's batched LightningIndexer.

    For ratio-4 causal prefill up to four adjacent queries see the same
    compressed-key range. Those queries are grouped into the BSND sequence axis
    instead of duplicating one page-table row per query. Every query still
    computes its own scores and its own ordered Top-K; no selection is shared.
    """
    import torch_npu

    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        return torch.full((rows, SELECTED_KEYS), -1, dtype=torch.int32, device=q.device)
    with torch.npu.device(q.device):
        counts = visible_key_counts(rows, sequence_length, q.device)
        grouped = group4 and rows >= GROUP_MIN_ROWS and keys >= GROUP_MIN_KEYS
        # The phase aligns four-query groups so that every real query in a
        # group shares one visible key range: with m = sequence_length - rows + 1
        # and phase = m % 4, row g * 4 - phase + t has (i + m) // 4 == g + m // 4
        # for every t in 0..3.
        phase = (sequence_length - rows + 1) % COMPRESSION_RATIO if grouped else 0
        span = COMPRESSION_RATIO if grouped else 1
        batch = (rows + phase + span - 1) // span
        if grouped:
            query = torch.nn.functional.pad(
                q, (0, 0, 0, 0, phase, batch * span - rows - phase)
            ).reshape(batch, span, QUERY_HEADS, HEAD_DIM)
            leader = (torch.arange(batch, device=q.device) * span - phase).clamp(
                0, rows - 1
            )
            visible = counts[leader]
        else:
            query = q.unsqueeze(1)
            visible = counts
        cache = torch.nn.functional.pad(
            k, (0, 0, 0, 0, 0, (-keys) % PAGE_SIZE)
        ).reshape(-1, PAGE_SIZE, 1, HEAD_DIM)
        table = (
            torch.arange(cache.shape[0], device=q.device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch, -1)
            .contiguous()
        )
        width = min(keys, SELECTED_KEYS)
        result = torch_npu.npu_lightning_indexer(
            query,
            cache,
            torch.ones((batch, span, QUERY_HEADS), device=q.device, dtype=q.dtype),
            actual_seq_lengths_query=torch.full(
                (batch,), span, device=q.device, dtype=torch.int32
            ),
            # CANN requires at least one key in its length metadata; rows with
            # no visible key are masked back to -1 below.
            actual_seq_lengths_key=visible.clamp_min(1),
            block_table=table,
            layout_query="BSND",
            layout_key="PA_BSND",
            sparse_count=width,
            sparse_mode=0,
        )
        indices = (result[0] if isinstance(result, tuple) else result).reshape(
            batch * span, width
        )[phase : phase + rows]
        ranks = torch.arange(width, device=q.device).unsqueeze(0)
        indices = torch.where(ranks < counts.unsqueeze(1), indices, -1)
        if width < SELECTED_KEYS:
            indices = torch.nn.functional.pad(
                indices, (0, SELECTED_KEYS - width), value=-1
            )
        return indices.to(torch.int32)


@register_impl("indexer.lightning_indexer_prefill", "torch_native")
def _native_impl(q, k, sequence_length, group4):
    global _native_fn
    if _native_fn is None:
        from ...implementation.torch_native.indexer.lightning_indexer_prefill import (
            lightning_indexer_prefill,
        )

        _native_fn = lightning_indexer_prefill
    # group4 selects between two equivalent torch_npu launch geometries and has no
    # meaning for the reference, which always scores every query independently.
    return _native_fn(q, k, sequence_length)


@register_operator("indexer.lightning_indexer_prefill")
def lightning_indexer_prefill(
    q, k, *, sequence_length: int, group4: bool = True, impl: str = "auto"
):
    """Select ordered Top-512 compressed keys for one ratio-4 prefill request.

    The default ``impl="auto"`` selects the CANN LightningIndexer adaptation.
    Scores use unit head weights, ``score[i, j] = sum_h relu(q[i, h] . k[j])``,
    and each row keeps the ``floor((sequence_length - rows + i + 1) / 4)`` keys
    that are causally visible to it. Call this once per request.

    Args:
        q (torch.Tensor): Queries of shape ``[rows, 4, 128]``, BF16, contiguous,
            on an NPU. These are the last ``rows`` queries of the request.
        k (torch.Tensor): Compressed keys of shape
            ``[sequence_length // 4, 1, 128]``, BF16, contiguous, on q's device.
        sequence_length (int): Request length in tokens. CPU metadata; no device
            value is read back for eligibility or phase selection. Keyword-only.
        group4 (bool): True batches up to four adjacent queries into one CANN
            sequence axis; False scores every query in its own batch entry. The
            two paths select the same keys and differ only in launch geometry,
            so this is a comparison switch rather than a tuning knob. Grouping
            additionally requires ``rows >= 1024`` and ``keys >= 512``.
            Defaults to True. Keyword-only.
        impl (str): Implementation name: "auto", "torch_npu" or "torch_native".
            Defaults to "auto". Keyword-only.

    Returns:
        torch.Tensor: Newly allocated int32 ``[rows, 512]`` tensor of ordered
        key indices local to ``k``, padded with -1. Rows with fewer than 512
        visible keys are -1 past their visible count. Inputs are not modified.

    Exact tie order is not part of the contract: keys whose scores are equal, or
    differ by one float32 ulp, may be returned in either order because the
    backends accumulate the head sum differently. The selected set and the
    non-increasing score order are stable. Arbitrary masks, learned head
    weights, other shapes, decode and speculative execution are outside this
    operator; callers keep their existing fallback for those.
    """
    _check_inputs(q, k, sequence_length, group4)
    impl_fn = get_impl_by_name("indexer.lightning_indexer_prefill", impl)
    return impl_fn(q, k, sequence_length, group4)

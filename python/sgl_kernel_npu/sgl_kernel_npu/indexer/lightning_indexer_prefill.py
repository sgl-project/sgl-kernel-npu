"""CANN LightningIndexer adapter for one ratio-4 causal prefill request."""

import torch
import torch_npu


def lightning_indexer_prefill(q, k, *, sequence_length: int, group4: bool = True):
    """Return ordered int32 [rows, 512] indices into this request's K.

    Q is contiguous BF16 [rows, 4, 128], K is contiguous BF16
    [sequence_length // 4, 1, 128]. Q contains the last ``rows`` queries
    of the request. Row i sees floor((sequence_length-rows+i+1)/4)
    compressed keys, with uniform head weights. No device metadata is
    read on the host. Call separately for each request; returned indices
    are local to K, with -1 padding. This is not a general mask adapter.

    Four-query batching applies only for rows >= 1024 and keys >= 512;
    group4=False provides the independent-query CANN comparison path.
    Both paths still score and select independently for each query.
    """
    if type(sequence_length) is not int or not 0 <= sequence_length < 2**31:
        raise ValueError("sequence_length must be a nonnegative int32 host integer")
    if type(group4) is not bool:
        raise ValueError("group4 must be a bool")
    if q.ndim != 3 or k.ndim != 3 or q.shape[1:] != (4, 128) or k.shape[1:] != (1, 128):
        raise ValueError("expected Q [rows,4,128] and K [keys,1,128]")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("Q and K must be BF16")
    if q.device.type != "npu" or q.device != k.device:
        raise ValueError("Q and K must be on the same NPU")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("Q and K must be contiguous")
    rows, keys = q.shape[0], k.shape[0]
    if rows > sequence_length or keys != sequence_length // 4:
        raise ValueError("Q/K lengths disagree with sequence_length")
    if not rows or not keys:
        return torch.full((rows, 512), -1, dtype=torch.int32, device=q.device)
    with torch.npu.device(q.device):
        counts = (
            torch.arange(rows, device=q.device, dtype=torch.int32)
            + (sequence_length - rows + 1)
        ) // 4
        grouped = group4 and rows >= 1024 and keys >= 512
        phase = (sequence_length - rows + 1) % 4 if grouped else 0
        span = 4 if grouped else 1
        batch = (rows + phase + span - 1) // span
        if grouped:
            query = torch.nn.functional.pad(
                q, (0, 0, 0, 0, phase, batch * span - rows - phase)
            ).reshape(batch, span, 4, 128)
            src = (torch.arange(batch, device=q.device) * span - phase).clamp(
                0, rows - 1
            )
            visible = counts[src]
        else:
            query = q.unsqueeze(1)
            visible = counts
        cache = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, (-keys) % 64)).reshape(
            -1, 64, 1, 128
        )
        table = (
            torch.arange(cache.shape[0], device=q.device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch, -1)
            .contiguous()
        )
        width = min(keys, 512)
        result = torch_npu.npu_lightning_indexer(
            query,
            cache,
            torch.ones((batch, span, 4), device=q.device, dtype=q.dtype),
            actual_seq_lengths_query=torch.full(
                (batch,), span, device=q.device, dtype=torch.int32
            ),
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
        if width < 512:
            indices = torch.nn.functional.pad(indices, (0, 512 - width), value=-1)
        return indices.to(torch.int32)

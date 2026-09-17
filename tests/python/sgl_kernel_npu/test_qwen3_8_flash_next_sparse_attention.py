"""NPU sparse attention comparisons against the existing Torch reference."""

from typing import Optional

import pytest
import torch
import torch_npu
from sgl_kernel_npu.qwen3_8_flash_next.sparse_attention import (
    can_run_sparse_attention,
    sparse_attention,
)

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason="NPU is required"
)


def qsa_sparse_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Device-agnostic sparse GQA reference."""

    scale = softmax_scale or q.shape[-1] ** -0.5
    if q.shape[0] == 0 or token_slots.shape[1] == 0:
        return torch.zeros_like(q)
    outputs = []
    repeats = q.shape[1] // k_cache.shape[1]
    for row in range(q.shape[0]):
        valid = token_slots[row] >= 0
        # Preserve the fixed width; boolean indexing creates dynamic shapes
        # and requires a device-to-host synchronization on NPU.
        slots = token_slots[row].clamp_min(0).long()
        keys = k_cache.index_select(0, slots).repeat_interleave(repeats, dim=1)
        values = v_cache.index_select(0, slots)
        # Zero invalid values so NaN/Inf padding cannot contaminate the sum.
        values = values.masked_fill(~valid[:, None, None], 0.0)
        values = values.repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float()) * scale
        valid = valid.unsqueeze(0)
        probabilities = torch.softmax(scores.masked_fill(~valid, -float("inf")), dim=-1)
        # softmax of an all-padding row is NaN; its output must instead be zero.
        probabilities = torch.where(valid, probabilities, 0.0)
        outputs.append(
            torch.einsum("hk,khd->hd", probabilities, values.float()).to(q.dtype)
        )
    return torch.stack(outputs)


def make_inputs(rows, heads, kv_heads, dim, width, dtype, strided=False):
    torch.manual_seed(42)
    q = torch.randn(rows, heads, dim, device="npu", dtype=dtype)
    k = torch.randn(3072, kv_heads, dim, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    slots = torch.randint(1, 3072, (rows, width), device="npu", dtype=torch.int32)
    if rows and width:
        slots[0] = -1
        slots[:, ::7] = -1
        if rows > 1:
            slots[1, ::5] = 1  # Repeated valid slots must retain their multiplicity.
    k[0] = float("nan")
    v[0] = float("inf")
    if strided:
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
        k = k.transpose(0, 1).contiguous().transpose(0, 1)
        v = v.transpose(0, 1).contiguous().transpose(0, 1)
        slots = slots.t().contiguous().t()
    return q, k, v, slots


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,heads,kv_heads,dim,width",
    [
        (0, 3, 1, 256, 2051),
        (2, 3, 1, 256, 0),
        (2, 3, 1, 256, 1),
        (3, 4, 2, 64, 33),
        (8, 3, 1, 256, 2051),
        (32, 3, 1, 256, 2051),
        (3, 4, 2, 128, 257),
    ],
)
def test_sparse_attention_reference(rows, heads, kv_heads, dim, width, dtype):
    args = make_inputs(rows, heads, kv_heads, dim, width, dtype)
    before = [x.clone() for x in args]
    expected = qsa_sparse_attention_reference(*args)
    actual = sparse_attention(*args)
    tolerance = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 2e-3}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert torch.isfinite(actual).all()
    for original, saved in zip(args, before):
        torch.testing.assert_close(original, saved, equal_nan=True)


def test_sparse_attention_strides_and_scale():
    args = make_inputs(3, 4, 2, 128, 257, torch.float32, strided=True)
    assert can_run_sparse_attention(*args)
    expected = qsa_sparse_attention_reference(*args, 0.3)
    actual = sparse_attention(*args, 0.3)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_sparse_attention_graph_replay():
    args = make_inputs(8, 3, 1, 256, 2051, torch.bfloat16)
    for _ in range(2):
        sparse_attention(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = sparse_attention(*args)
    for length in (0, 33, 2051):
        args[0].normal_()
        args[3].fill_(-1)
        args[3][:, :length] = 1
        graph.replay()
        torch.npu.synchronize()
        expected = qsa_sparse_attention_reference(*args)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_sparse_attention_unsupported_layout():
    args = make_inputs(2, 3, 1, 256, 33, torch.bfloat16)
    q, k, v, slots = args
    unsupported = (q[..., ::2], k[..., ::2], v[..., ::2], slots)
    assert not can_run_sparse_attention(*unsupported)
    with pytest.raises(ValueError, match="Unsupported NPU"):
        sparse_attention(*unsupported)


def test_sparse_attention_large_batch():
    args = list(make_inputs(128, 3, 1, 256, 2051, torch.bfloat16))
    args[-1] = args[-1].to(torch.int64)
    expected = qsa_sparse_attention_reference(*args)
    actual = sparse_attention(*args)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    "rows,dim,width,dtype",
    [
        (2730, 64, 33, torch.float32),  # Last row count below the split-grid limit.
        (2731, 64, 33, torch.float32),
        (4096, 256, 2051, torch.bfloat16),  # Full prefill chunk for this model.
        (21846, 64, 1, torch.bfloat16),  # Even the unchunked merge grid overflows.
    ],
)
def test_sparse_attention_launch_boundaries(rows, dim, width, dtype):
    q, k, v, slots = make_inputs(4, 3, 1, dim, width, dtype)
    # Repeat distinct rows to check both chunk offsets and the final partial chunk.
    expected = qsa_sparse_attention_reference(q, k, v, slots)
    copies = (rows + 3) // 4
    q = q.repeat(copies, 1, 1)[:rows].transpose(0, 1).contiguous().transpose(0, 1)
    slots = slots.repeat(copies, 1)[:rows].t().contiguous().t()
    actual = sparse_attention(q, k, v, slots)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-3
    torch.testing.assert_close(
        actual, expected.repeat(copies, 1, 1)[:rows], atol=tolerance, rtol=tolerance
    )


def test_sparse_attention_chunked_graph_replay():
    rows = 2731
    q, k, v, slots = make_inputs(4, 3, 1, 64, 33, torch.bfloat16)
    copies = (rows + 3) // 4
    queries = q.repeat(copies, 1, 1)[:rows]
    indices = slots.repeat(copies, 1)[:rows]
    for _ in range(2):
        sparse_attention(queries, k, v, indices)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = sparse_attention(queries, k, v, indices)
    for length in (0, 17, 33):
        q.normal_()
        slots.fill_(-1)
        slots[:, :length] = 1
        queries.copy_(q.repeat(copies, 1, 1)[:rows])
        indices.copy_(slots.repeat(copies, 1)[:rows])
        graph.replay()
        torch.npu.synchronize()
        expected = qsa_sparse_attention_reference(q, k, v, slots)
        torch.testing.assert_close(
            actual, expected.repeat(copies, 1, 1)[:rows], atol=2e-3, rtol=2e-3
        )


def test_sparse_attention_head_count_exceeds_launch_limit():
    q, k, v, slots = make_inputs(1, 8192, 1, 64, 1, torch.bfloat16)
    assert not can_run_sparse_attention(q, k, v, slots)
    with pytest.raises(ValueError, match="Unsupported NPU"):
        sparse_attention(q, k, v, slots)

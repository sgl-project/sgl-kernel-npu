"""Paged NPU MQA comparisons with the unchanged Torch scoring reference."""

import math
from typing import Optional

import pytest
import torch
import torch_npu
from sgl_kernel_npu.qwen3_8_flash_next.mqa import can_run_mqa_decode, mqa_decode

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason="NPU is required"
)


def _validate_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(f"QSA requires q [tokens, heads, head_dim], got {q.shape}")


def _validate_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    _validate_q(q)
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError(
            "QSA decode cache must be [pages, page_size, 1, head_dim], "
            f"got {tuple(k_cache.shape)}"
        )
    if k_cache.shape[-1] != q.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("QSA decode page table must have one row per query")
    if context_lens.numel() != q.shape[0]:
        raise ValueError("QSA decode context lengths must have one entry per query")


def torch_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for variable-length paged decode MQA."""

    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    batch = q.shape[0]
    page_size = k_cache.shape[1]
    total = page_table.shape[1] * page_size
    gathered = k_cache[page_table.long().clamp_min(0).reshape(-1), :, 0].reshape(
        batch, total, q.shape[-1]
    )
    scores = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    scores = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    positions = torch.arange(total, device=q.device).unsqueeze(0)
    scores.masked_fill_(
        positions >= context_lens.to(q.device).reshape(-1, 1), -float("inf")
    )
    logits = torch.full(
        (batch, max_model_len), -float("inf"), dtype=torch.float32, device=q.device
    )
    copy_len = min(total, max_model_len)
    if copy_len:
        logits[:, :copy_len] = scores[:, :copy_len]
    return logits


def make_inputs(rows, heads, dim, page_size, pages, dtype, strided=False):
    torch.manual_seed(73)
    q = torch.randn(rows, heads, dim, device="npu", dtype=dtype)
    cache = torch.randn(23, page_size, 1, dim, device="npu", dtype=dtype)
    table = torch.randint(0, 23, (rows, pages), device="npu", dtype=torch.int64)
    lengths = torch.arange(rows, device="npu", dtype=torch.int32)
    lengths = lengths * (pages * page_size) // max(rows - 1, 1)
    if rows and pages:
        table[-1, 0] = -1  # The Torch reference clamps negative page ids to zero.
    if strided:
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
        cache = cache.transpose(0, 1).contiguous().transpose(0, 1)
        table = table.t().contiguous().t()
        lengths = torch.stack((lengths, lengths), dim=1)[:, 0]
    return q, cache, table, lengths


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,heads,dim,page_size,pages,width",
    [
        (0, 4, 128, 16, 3, 48),
        (3, 4, 128, 16, 3, 0),
        (3, 4, 128, 16, 0, 19),
        (3, 4, 128, 16, 3, 61),
        (8, 4, 128, 64, 7, 257),
        (4, 3, 64, 3, 7, 21),
        (4, 1, 256, 1, 17, 129),
        (32, 4, 128, 16, 129, 2065),
        (128, 8, 128, 16, 9, 65536),
    ],
)
def test_mqa_reference(rows, heads, dim, page_size, pages, width, dtype):
    args = make_inputs(rows, heads, dim, page_size, pages, dtype)
    before = [x.clone() for x in args]
    expected = torch_qsa_mqa_decode(*args, width)
    actual = mqa_decode(*args, width)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for original, saved in zip(args, before):
        torch.testing.assert_close(original, saved)


def test_mqa_padding_and_graph_replay():
    args = make_inputs(4, 4, 128, 16, 9, torch.bfloat16)
    q, cache, table, lengths = args
    width = 161
    for _ in range(2):
        mqa_decode(*args, width)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = mqa_decode(*args, width)
    for length in (0, 1, 129, 144):
        q.normal_()
        lengths.fill_(length)
        table.fill_(1)
        cache[1].normal_()
        cache[0] = float("nan")
        table[:, (length + 15) // 16 :] = 0
        graph.replay()
        torch.npu.synchronize()
        expected = torch_qsa_mqa_decode(*args, width)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_mqa_unsupported_layout():
    args = make_inputs(3, 4, 128, 16, 3, torch.bfloat16)
    q, cache, table, lengths = args
    args = q[..., ::2], cache[..., ::2], table, lengths
    assert not can_run_mqa_decode(*args, 48)
    with pytest.raises(ValueError, match="Unsupported NPU"):
        mqa_decode(*args, 48)


def test_mqa_strides_and_scale():
    args = make_inputs(4, 4, 128, 16, 19, torch.bfloat16, strided=True)
    torch.testing.assert_close(
        mqa_decode(*args, 333, 3.0),
        torch_qsa_mqa_decode(*args, 333, 3.0),
        atol=2e-5,
        rtol=2e-5,
    )

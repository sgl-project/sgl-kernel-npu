"""Exact comparisons for stable NPU block expansion and graph replay."""

import pytest
import torch
import torch_npu
from sgl_kernel_npu.qwen3_8_flash_next.expansion import (
    can_run_block_expansion,
    expand_blocks,
)

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason="NPU is required"
)

# Inclusive bound for exact positional sorting keys in FP32.
_FP32_EXACT_INT_MAX = 2**24


def torch_expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand compressed block indices into fixed-width logical token indices."""

    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError(
            f"expected block indices [M, {block_topk}], got "
            f"{tuple(block_indices.shape)}"
        )
    rows = block_indices.shape[0]
    if query_positions.numel() != rows or sequence_lengths.numel() != rows:
        raise ValueError("query positions and sequence lengths must match top-k rows")

    device = block_indices.device
    blocks = block_indices.long()
    offsets = torch.arange(compress_ratio, device=device, dtype=torch.long)
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]

    query_positions = query_positions.to(device=device, dtype=torch.long)
    sequence_lengths = sequence_lengths.to(device=device, dtype=torch.long)
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=device, dtype=torch.long)
    visible_tokens = query_positions + 1
    tail_start = (
        torch.div(visible_tokens, compress_ratio, rounding_mode="floor")
        * compress_ratio
    )
    tail_count = visible_tokens - tail_start
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat([expanded, tail], dim=1)
    # Keep all valid entries contiguous. This is required by the FA2 packing path.
    order = torch.arange(final_topk, device=device).unsqueeze(0).expand(rows, -1)
    if 2 * final_topk - 1 <= _FP32_EXACT_INT_MAX:
        # NPU integer argsort falls back to AiCpu. Convert only when the
        # largest positional key, including padding, is exact in FP32.
        order = order.float()
    sort_key = torch.where(result >= 0, order, order + final_topk)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "rows,ratio,topk",
    [(0, 4, 17), (1, 1, 1), (8, 3, 17), (32, 4, 2048), (128, 8, 4097), (4, 8, 8185)],
)
def test_exact_expansion(rows, ratio, topk, dtype):
    torch.manual_seed(73)
    width = (topk + ratio - 1) // ratio
    blocks = torch.randint(-3, 100, (rows, width * 2), dtype=dtype)[:, ::2]
    positions = torch.arange(rows * 2, dtype=dtype)[::2] - 5
    lengths = torch.arange(rows * 2, dtype=dtype)[::2] * 7
    # Preserve non-contiguous strides on device, including the column stride.
    blocks = blocks.t().contiguous().to("npu").t()
    positions = torch.stack((positions, positions), dim=1).to("npu")[:, 0]
    lengths = torch.stack((lengths, lengths), dim=1).to("npu")[:, 0]
    args = blocks, positions, lengths, ratio, topk
    assert can_run_block_expansion(*args)
    expected = torch_expand_qsa_block_indices(*args)
    actual = expand_blocks(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_holes_duplicates_large_indices_and_graph():
    blocks = torch.tensor(
        [
            [3, -1, 0, 3, 2],
            [-1, -2, -1, -1, -1],
            [2**30, 0, 1, 2**29, -1],
        ],
        device="npu",
        dtype=torch.int64,
    )
    positions = torch.tensor([19, 7, 2**32], device="npu", dtype=torch.int64)
    lengths = torch.tensor([14, 0, 2**34], device="npu", dtype=torch.int64)
    args = blocks, positions, lengths, 4, 19
    for _ in range(2):
        expand_blocks(*args)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = expand_blocks(*args)
    for length in (0, 1, 14, 2**34):
        lengths.fill_(length)
        positions.sub_(1)
        blocks[0, 1] = 2 if length else -1
        graph.replay()
        torch.npu.synchronize()
        expected = torch_expand_qsa_block_indices(*args)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("ratio", [1, 3, 4, 8])
def test_mixed_integer_types_and_broadcast(ratio):
    blocks = torch.tensor(
        [[0, -1, 2**30, 2**31 - 1, 3]], device="npu", dtype=torch.int32
    ).expand(4, -1)
    positions = torch.tensor([-5, 15, 2**31, 2**34], device="npu")
    lengths = torch.tensor([0, 17, 2**31 - 1, 2**35], device="npu")
    args = blocks, positions, lengths, ratio, 5 * ratio - (ratio > 1)
    torch.testing.assert_close(
        expand_blocks(*args),
        torch_expand_qsa_block_indices(*args),
        atol=0,
        rtol=0,
    )


def test_unsupported_width():
    args = (
        torch.zeros((1, 1025), device="npu", dtype=torch.int32),
        torch.tensor([20], device="npu"),
        torch.tensor([21], device="npu"),
        1,
        1025,
    )
    assert not can_run_block_expansion(*args)
    with pytest.raises(ValueError, match="Unsupported NPU"):
        expand_blocks(*args)

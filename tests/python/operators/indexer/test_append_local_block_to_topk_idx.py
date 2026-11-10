"""Accuracy/performance tests for indexer.append_local_block_to_topk_idx.

Run:
    python -m pytest tests/python/operators/indexer/test_append_local_block_to_topk_idx.py -sv

The DUT (``auto``, i.e. Triton) is compared with the ``torch_native`` reference.
Indices are integers, so the comparison is exact.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Shape (N, B, topk) of the production-like cases that collect performance.
perf_shape = [(1, 1, 16), (1, 16, 16), (1, 256, 16), (1, 48, 16), (2, 16, 16)]


def generate_test_data(shape, num_blocks, seq_lens, rows, seq_dtype=torch.int32):
    """Create NPU candidates and seq_lens; random candidates when rows is None."""
    if rows is None:
        generator = torch.Generator(device="cpu").manual_seed(20261031)
        candidates = torch.randint(
            -1,
            num_blocks + 8,
            shape,
            generator=generator,
            dtype=torch.int32,
            device="cpu",
        )
    else:
        candidates = torch.tensor(rows, dtype=torch.int32, device="cpu")
    return (
        candidates.reshape(shape).contiguous().npu(),
        torch.tensor(seq_lens, dtype=seq_dtype, device="cpu").contiguous().npu(),
    )


# (shape, block_size, num_blocks, seq_lens, rows, expected).
# Dims are borrowed from MiniMax-M3 (N=1, topk=16, block=128): the current K-only
# configs skip this operator, so it is only reachable on the
# disable_index_value=False decode/verify path. Edge cases list candidates and
# the expected result flattened; q_pos = max(seq_len - 1, 0) and
# local = min(q_pos // block_size, num_blocks - 1).
# fmt: off
CASES = [
    pytest.param((1, 1, 16), 128, 8192, [1048576], None, None, id="minimax-m3-b1"),
    pytest.param((1, 16, 16), 128, 8192, [4096 + i * 16384 for i in range(16)], None, None,
                 id="minimax-m3-b16"),
    pytest.param((1, 256, 16), 128, 8192, [4096 + (i % 16) * 8192 for i in range(256)], None, None,
                 id="minimax-m3-b256"),
    pytest.param((1, 48, 16), 128, 8192, [4096 + j for _ in range(16) for j in (1, 2, 3)], None, None,
                 id="minimax-m3-verify"),
    pytest.param((2, 16, 16), 128, 8192, [4096 + i * 16384 for i in range(16)], None, None,
                 id="minimax-m3-kvh2"),
    pytest.param((1, 16, 8), 128, 8192, [8192 * (i + 1) for i in range(16)], None, None, id="topk8"),
    pytest.param((1, 16, 32), 128, 8192, [8192 * (i + 1) for i in range(16)], None, None, id="topk32"),
    pytest.param((1, 16, 16), 64, 16384, [8192 + i * 4096 for i in range(16)], None, None,
                 id="block64"),
    # Edge cases: local dedup / append, candidate filtering, seq_len boundaries.
    pytest.param((1, 4, 3), 128, 8, [513] * 4,
                 [0, 1, 2, 4, 1, 2, 4, 4, 1, 1, 1, 2],
                 [0, 1, 2, 4, 4, 1, 2, -1, 4, 4, 1, -1, 1, 1, 2, 4],
                 id="edge-local-dedup", marks=pytest.mark.extra_checks),
    pytest.param((1, 5, 2), 128, 8, [513] * 5,
                 [-1, 0, 8, 0, 9, 0, 5, 0, -1, -1],
                 [-1, 0, 4] * 4 + [-1, -1, 4],
                 id="edge-candidate-filter", marks=pytest.mark.extra_checks),
    pytest.param((1, 2, 2), 128, 8, [129, 128], [1, 1, 1, 1], [1, 1, -1, -1, -1, 0],
                 id="edge-causal-boundary", marks=pytest.mark.extra_checks),
    pytest.param((1, 6, 4), 128, 8, [0, 1, 127, 128, 129, 2000], [2, 3, 5, 6] * 6,
                 [-1, -1, -1, -1, 0] * 4 + [-1, -1, -1, -1, 1, 2, 3, 5, 6, 7],
                 id="edge-seq-len-boundaries", marks=pytest.mark.extra_checks),
    pytest.param((1, 1, 1), 128, 1, [1], [0], [0, -1],
                 id="edge-min-shape", marks=pytest.mark.extra_checks),
    pytest.param((1, 17, 16), 128, 8192, [128 * (i + 1) for i in range(17)], None, None,
                 id="edge-bsq-tail", marks=pytest.mark.extra_checks),
]
# fmt: on


@pytest.mark.parametrize(
    "shape, block_size, num_blocks, seq_lens, rows, expected", CASES
)
def test_append_local_block_to_topk_idx(
    shape, block_size, num_blocks, seq_lens, rows, expected
):
    from sgl_kernel_npu.operators.indexer import append_local_block_to_topk_idx

    topk_idx, seq = generate_test_data(shape, num_blocks, seq_lens, rows)
    topk_before, seq_before = topk_idx.clone(), seq.clone()
    executor = check_accuracy(
        append_local_block_to_topk_idx, args=(topk_idx, seq, block_size, num_blocks)
    )
    result = executor()
    assert result.dtype == torch.int32
    assert result.shape == (*shape[:2], shape[2] + 1)
    assert result.untyped_storage().data_ptr() != topk_idx.untyped_storage().data_ptr()
    assert torch.equal(topk_idx, topk_before) and torch.equal(seq, seq_before)
    if expected is not None:
        assert result.flatten().tolist() == expected
    check_performance(executor, enable_perf=shape in perf_shape and block_size == 128)


@pytest.mark.extra_checks
def test_int64_seq_lens():
    from sgl_kernel_npu.operators.indexer import append_local_block_to_topk_idx

    topk_idx, seq = generate_test_data(
        (1, 4, 16), 8192, [513] * 4, None, seq_dtype=torch.int64
    )
    executor = check_accuracy(
        append_local_block_to_topk_idx, args=(topk_idx, seq, 128, 8192)
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize("impl", ["triton", "torch_native"])
def test_bad_inputs_are_rejected(impl):
    from sgl_kernel_npu.operators.indexer import append_local_block_to_topk_idx

    topk_idx = torch.zeros((1, 2, 6), dtype=torch.int32).npu()
    seq = torch.zeros(2, dtype=torch.int32).npu()
    with pytest.raises(AssertionError):
        append_local_block_to_topk_idx(topk_idx[:, :, ::2], seq, 128, 8, impl=impl)
    with pytest.raises(AssertionError):
        append_local_block_to_topk_idx(topk_idx, seq[:-1], 128, 8, impl=impl)
    with pytest.raises(AssertionError):
        append_local_block_to_topk_idx(topk_idx[:, :, :0], seq, 128, 8, impl=impl)

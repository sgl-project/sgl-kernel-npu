"""Verify reordered pages, lengths, padding and caller-owned outputs."""

import pytest

torch = pytest.importorskip("torch")


from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# MiniMax NPU sparse prefill passes [total_q, topk+1] page candidates.
# block=128/topk=16 follows minimax_sparse_backend; query counts are synthetic.
# Pad to a power of two for this kernel's tl.arange(0, TOPK1).
perf_shape = [(128, 32, 128), (2048, 32, 128)]


def generate_test_data(shape, mode, reuse_outputs):
    rows, width, block_size = shape
    num_requests, num_blocks = 3, 64
    # Different requests use shuffled physical pages rather than identity mapping.
    physical_pages = torch.randperm(num_requests * num_blocks, dtype=torch.int64)
    pages = (
        (
            physical_pages.reshape(num_requests, num_blocks, 1) * block_size
            + torch.arange(block_size)
        )
        .reshape(num_requests, -1)
        .to(torch.int32)
    )
    requests = (torch.arange(rows) % num_requests).to(torch.int32)
    boundaries = torch.tensor(
        [
            0,
            1,
            block_size - 1,
            block_size,
            block_size + 1,
            16 * block_size,
            16 * block_size + 7,
        ],
        dtype=torch.int32,
    )
    lengths = boundaries[torch.arange(rows) % boundaries.numel()]
    own = (lengths - 1).clamp_min(0) // block_size
    ti = torch.full((rows, width), -1, dtype=torch.int32)
    for row in range(rows):
        current = int(own[row].item())
        if mode == "padding":
            continue
        if mode == "own-only":
            if int(lengths[row].item()) > 0:
                ti[row, 0] = current
            continue
        # Keep one slot available for the unconditional own-page output.
        candidates = torch.randperm(current).tolist()[: min(16, width - 2)]
        if mode == "mixed" and int(lengths[row].item()) > 0:
            candidates.insert(len(candidates) // 2, current)
        if candidates:
            ti[row, : len(candidates)] = torch.tensor(candidates, dtype=torch.int32)
    args = (ti, lengths, requests, pages, block_size)
    return args, rows, width


@pytest.mark.parametrize(
    "shape",
    [(3, 4, 16), (7, 8, 32), (128, 32, 128), (2048, 32, 128)],
    ids=["small", "multi-request", "prefill-128", "prefill-2048"],
)
@pytest.mark.parametrize("mode", ["mixed", "padding", "own-only", "own-missing"])
@pytest.mark.parametrize("reuse_outputs", [False, True])
def test_fia_prep(shape, mode, reuse_outputs):
    from sgl_kernel_npu.operators.attention import fia_prep

    args, rows, width = generate_test_data(shape, mode, reuse_outputs)

    def isolated_prep(*, impl="auto"):
        table = torch.empty((rows, width), dtype=torch.int32) if reuse_outputs else None
        out_lengths = torch.empty((rows,), dtype=torch.int32) if reuse_outputs else None
        actual = fia_prep(*args, table, out_lengths, impl=impl)
        if reuse_outputs:
            assert actual[0] is table and actual[1] is out_lengths
        return actual

    impl, impl_ref = get_test_impls()
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        require_distinct_impls(fia_prep, impl, impl_ref)
        assert_accuracy(isolated_prep(impl=impl), isolated_prep(impl=impl_ref))
    else:
        require_test_impl(fia_prep, impl)
    table = torch.empty((rows, width), dtype=torch.int32)
    out_lengths = torch.empty((rows,), dtype=torch.int32)

    def executor():
        return fia_prep(*args, table, out_lengths, impl=impl)

    executor.func = fia_prep
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    check_performance(
        executor,
        enable_perf=(shape in perf_shape and mode == "mixed" and reuse_outputs),
    )

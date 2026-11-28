import importlib.util
from pathlib import Path

import pytest
import torch

_PLANNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "sgl_kernel_npu"
    / "sgl_kernel_npu"
    / "sparsity_driven_kv_offload"
    / "cache_planner.py"
)
_PLANNER_SPEC = importlib.util.spec_from_file_location(
    "cache_planner_reference", _PLANNER_PATH
)
if _PLANNER_SPEC is None or _PLANNER_SPEC.loader is None:
    raise ImportError(f"cannot load planner reference from {_PLANNER_PATH}")
_PLANNER_MODULE = importlib.util.module_from_spec(_PLANNER_SPEC)
_PLANNER_SPEC.loader.exec_module(_PLANNER_MODULE)
plan_sparse_kv_cache_reference = _PLANNER_MODULE.plan_sparse_kv_cache_reference


def _run(source_map, inverse, req_ids, rows, *, max_context_len, capacity):
    return plan_sparse_kv_cache_reference(
        source_map,
        inverse,
        req_ids,
        rows,
        max_context_len=max_context_len,
        cache_capacity=capacity,
    )


def test_planner_sorts_misses_before_hits_and_preserves_hits():
    source_map = [[0, -1, -1]]
    inverse = [[0, -1, -1]]
    result = _run(source_map, inverse, [0], [[2, 1, 0]], max_context_len=3, capacity=3)

    topk, slots, miss_count, miss_src, miss_slots, request_miss_count = result
    assert topk[0][:3] == [1, 2, 0]
    assert slots[0][:3] == [1, 2, 0]
    assert miss_count == request_miss_count == [2]
    assert miss_src[0][:2] == [1, 2]
    assert miss_slots[0][:2] == [1, 2]
    assert source_map == [[0, 1, 2]]
    assert inverse == [[0, 1, 2]]


def test_planner_uses_free_slots_before_evicting_and_protects_hits():
    source_map = [[0, -1, -1, 2]]
    inverse = [[0, -1, 3]]
    result = _run(source_map, inverse, [0], [[2, 1, 0]], max_context_len=4, capacity=3)

    topk, slots, miss_count, *_ = result
    assert topk[0][:3] == [1, 2, 0]
    assert slots[0][:3] == [1, 2, 0]
    assert miss_count == [2]
    assert inverse == [[0, 1, 2]]
    assert source_map == [[0, 1, 2, -1]]


def test_invalid_request_is_noop_with_sentinel_outputs():
    source_map = [[0, -1, -1]]
    inverse = [[0, -1]]
    before = (source_map[0][:], inverse[0][:])
    result = _run(
        source_map, inverse, [-1, 1], [[0, 1], [0, 1]], max_context_len=3, capacity=2
    )

    topk, slots, miss_count, miss_src, miss_slots, request_miss_count = result
    for row in range(2):
        assert topk[row] == [-1, -1]
        assert slots[row] == [-1, -1]
        assert miss_src[row] == [-1, -1]
        assert miss_slots[row] == [-1, -1]
    assert miss_count == request_miss_count == [0, 0]
    assert (source_map[0], inverse[0]) == before


def test_out_of_range_topk_sources_are_ignored():
    source_map = [[-1] * 5]
    inverse = [[-1] * 4]
    result = _run(
        source_map, inverse, [0], [[-1, 2, 7, 1]], max_context_len=5, capacity=4
    )

    topk, slots, miss_count, *_ = result
    assert topk[0][:2] == [1, 2]
    assert slots[0][:2] == [0, 1]
    assert topk[0][2:] == [-1, -1]
    assert slots[0][2:] == [-1, -1]
    assert miss_count == [2]


def test_first_fill_uses_every_empty_cache_slot():
    capacity = 2048
    source_map = [[-1] * capacity]
    inverse = [[-1] * capacity]
    sources = list(range(capacity))

    result = _run(
        source_map,
        inverse,
        [0],
        [sources],
        max_context_len=capacity,
        capacity=capacity,
    )

    topk, slots, miss_count, miss_src, miss_slots, request_miss_count = result
    assert miss_count == request_miss_count == [capacity]
    assert topk[0] == sources
    assert miss_src[0] == sources
    assert slots[0] == list(range(capacity))
    assert miss_slots[0] == list(range(capacity))
    assert source_map[0] == list(range(capacity))
    assert inverse[0] == list(range(capacity))


def test_npu_first_fill_uses_every_empty_cache_slot():
    try:
        import sgl_kernel_npu  # noqa: F401
        import torch_npu  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"NPU runtime or sgl-kernel-npu extension is unavailable: {exc}")

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required to exercise the native cache planner")

    capacity = 2048
    device = torch.device("npu")
    source_map = torch.full((2, capacity), -1, dtype=torch.int32, device=device)
    inverse = torch.full((2, capacity), -1, dtype=torch.int32, device=device)
    req_ids = torch.tensor([0], dtype=torch.int32, device=device)
    topk_input = torch.arange(capacity, dtype=torch.int32, device=device).view(1, -1)
    topk_src = torch.empty((1, 1, capacity), dtype=torch.int32, device=device)
    topk_dst = torch.empty_like(topk_src)
    miss_count = torch.empty((1,), dtype=torch.int32, device=device)
    miss_src = torch.empty((1, capacity), dtype=torch.int32, device=device)
    miss_dst = torch.empty_like(miss_src)
    miss_counts = torch.empty((1,), dtype=torch.int32, device=device)

    torch.ops.npu.plan_sparse_kv_cache(
        source_map,
        inverse,
        req_ids,
        topk_input,
        topk_src,
        topk_dst,
        miss_count,
        miss_src,
        miss_dst,
        miss_counts,
        capacity,
        capacity,
    )
    torch.npu.synchronize()

    assert miss_count.item() == capacity
    assert miss_counts.item() == capacity
    assert torch.equal(topk_src[0, 0], topk_input[0])
    assert torch.equal(miss_src[0], topk_input[0])
    assert torch.equal(
        topk_dst[0, 0], torch.arange(capacity, dtype=torch.int32, device=device)
    )
    assert torch.equal(
        miss_dst[0], torch.arange(capacity, dtype=torch.int32, device=device)
    )
    assert torch.equal(
        source_map[0], torch.arange(capacity, dtype=torch.int32, device=device)
    )
    assert torch.equal(
        inverse[0], torch.arange(capacity, dtype=torch.int32, device=device)
    )


@pytest.mark.parametrize("batch_size", [3, 16])
def test_npu_dense_count_writes_with_invalid_rows_and_varying_misses(batch_size):
    try:
        import sgl_kernel_npu  # noqa: F401
        import torch_npu  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"NPU runtime or sgl-kernel-npu extension is unavailable: {exc}")

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required to exercise the native cache planner")

    capacity = 2048
    max_context_len = capacity + 128
    device = torch.device("npu")
    source_map = torch.full(
        (batch_size + 1, max_context_len), -1, dtype=torch.int32, device=device
    )
    inverse = torch.full(
        (batch_size + 1, capacity), -1, dtype=torch.int32, device=device
    )
    first_fill_counts = []
    for row in range(batch_size):
        if row == 0:
            count = capacity
        elif row % 5 == 1:
            count = 0
        else:
            count = capacity - 128 * (row % 5)
        first_fill_counts.append(count)
    req_ids_cpu = list(range(batch_size))
    first_fill_rows = [
        list(range(count)) + [-1] * (capacity - count) for count in first_fill_counts
    ]
    for row in range(batch_size):
        if first_fill_counts[row] == 0:
            req_ids_cpu[row] = -1

    def run_phase(rows, expected_counts):
        topk_input = torch.tensor(rows, dtype=torch.int32, device=device)
        topk_src = torch.full(
            (batch_size, 1, capacity), -99, dtype=torch.int32, device=device
        )
        topk_dst = torch.full_like(topk_src, -99)
        topk_miss_count = torch.full(
            (batch_size,), -99, dtype=torch.int32, device=device
        )
        miss_src = torch.full(
            (batch_size, capacity), -99, dtype=torch.int32, device=device
        )
        miss_dst = torch.full_like(miss_src, -99)
        miss_counts = torch.full_like(topk_miss_count, -99)
        torch.ops.npu.plan_sparse_kv_cache(
            source_map,
            inverse,
            torch.tensor(req_ids_cpu, dtype=torch.int32, device=device),
            topk_input,
            topk_src,
            topk_dst,
            topk_miss_count,
            miss_src,
            miss_dst,
            miss_counts,
            max_context_len,
            capacity,
        )
        torch.npu.synchronize()
        expected = torch.tensor(expected_counts, dtype=torch.int32, device=device)
        assert torch.equal(topk_miss_count, expected)
        assert torch.equal(miss_counts, expected)
        for row in range(batch_size):
            if req_ids_cpu[row] < 0:
                assert torch.all(topk_src[row] == -1)
                assert torch.all(topk_dst[row] == -1)
                assert torch.all(miss_src[row] == -1)
                assert torch.all(miss_dst[row] == -1)

    run_phase(first_fill_rows, first_fill_counts)

    partial_rows = []
    for row, count in enumerate(first_fill_counts):
        if req_ids_cpu[row] < 0:
            partial_rows.append([-1] * capacity)
        elif row == 0:
            partial_rows.append(
                list(range(128, capacity)) + list(range(capacity, capacity + 128))
            )
        else:
            partial_rows.append(list(range(count)) + [-1] * (capacity - count))
    expected_partial = [128 if row == 0 else 0 for row in range(batch_size)]
    run_phase(partial_rows, expected_partial)

    expected_all_hit = [0] * batch_size
    run_phase(partial_rows, expected_all_hit)

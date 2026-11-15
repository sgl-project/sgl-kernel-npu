# In-place Operator
"""In-place tail scatter; isolated accuracy copies, reusable performance output."""

from types import SimpleNamespace

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
from tests.python.utils.standards import AccuracyStandard

# W: Q/Nt workload candidates; tail_rows=8 is an independent synthetic value.
HEAD_DIM = 128
perf_shape = [(4096, 1024)]

EXACT_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0, rtol=0),
    "torch.float16": AccuracyStandard(atol=0, rtol=0),
}


def generate_test_data(
    n_rows,
    total_tokens,
    req_pool_size,
    pool_size=4,
    tail_size=4,
    head_dim=128,
    dtype=torch.bfloat16,
    seed=20,
):
    """Return operator kwargs and protected row IDs in the [-1, 128] view."""
    torch.manual_seed(seed)

    chunk_k = torch.empty(total_tokens, head_dim, dtype=dtype).normal_(
        mean=0.0, std=0.5
    )
    chunk_score = torch.empty(total_tokens, head_dim, dtype=dtype).normal_(
        mean=0.0, std=0.5
    )
    tail_k = torch.empty(req_pool_size, tail_size, head_dim, dtype=dtype).normal_(
        mean=0.0, std=0.5
    )
    tail_score = torch.empty(req_pool_size, tail_size, head_dim, dtype=dtype).normal_(
        mean=0.0, std=0.5
    )

    assert req_pool_size >= n_rows
    req_pool_idx = torch.arange(n_rows, dtype=torch.int64)
    dst_logical_start = torch.randint(0, tail_size, (n_rows,), dtype=torch.int32)
    n_write = torch.randint(1, pool_size + 1, (n_rows,), dtype=torch.int32)

    cumsum = torch.cumsum(n_write.to(torch.int64), dim=0)
    chunk_src_start = torch.zeros(n_rows, dtype=torch.int64)
    chunk_src_start[1:] = cumsum[:-1]
    assert chunk_src_start[-1] + n_write[-1] <= total_tokens

    inputs = dict(
        pool=SimpleNamespace(index_kpool=pool_size),
        chunk_k=chunk_k,
        chunk_score=chunk_score,
        tail_k=tail_k,
        tail_score=tail_score,
        req_pool_idx=req_pool_idx,
        dst_logical_start=dst_logical_start,
        chunk_src_start=chunk_src_start,
        n_write=n_write,
    )
    offsets = torch.arange(pool_size)[None, :]
    slots = (dst_logical_start[:, None] + offsets) % tail_size
    requests = req_pool_idx[:, None].expand_as(slots)
    valid = offsets < n_write[:, None]
    written = torch.zeros(req_pool_size, tail_size, dtype=torch.bool)
    written[requests[valid], slots[valid]] = True
    untouched_rows = (~written).flatten().nonzero().flatten()
    return inputs, {name: untouched_rows for name in ("tail_k", "tail_score")}


@pytest.mark.parametrize(
    "case",
    [
        # Case 01: Minimum nonempty scatter: one request and tail4.
        pytest.param(dict(n_rows=1, total_tokens=16, req_pool_size=4), id="minimum"),
        # Case 02: Multiple requests with tail8.
        pytest.param(
            dict(n_rows=8, total_tokens=64, req_pool_size=16, tail_size=8), id="tail8"
        ),
        # Case 03: 128 updates with short tail4.
        pytest.param(
            dict(n_rows=128, total_tokens=640, req_pool_size=256), id="many-updates"
        ),
        # Case 04: FP16 key/score copies.
        pytest.param(
            dict(
                n_rows=4,
                total_tokens=32,
                req_pool_size=8,
                tail_size=8,
                dtype=torch.float16,
            ),
            id="fp16",
        ),
        # Case 05: Typical extend: Q4096, 1024 updates; also profile.
        pytest.param(
            dict(n_rows=1024, total_tokens=4096, req_pool_size=1024, tail_size=8),
            id="W-q4096-nt1024",
        ),
    ],
)
def test_scatter_kpool_tail_updates(case):
    """In-place: copy tail key/score; preserve all unwritten slots exactly."""
    from sgl_kernel_npu.operators.mem_cache import scatter_kpool_tail_updates

    # Inputs: named operator arguments and independent unchanged-row checks.
    inputs, unchanged = generate_test_data(**case)
    state_names = ("tail_k", "tail_score")
    # Accuracy: isolate DUT/reference state for this in-place operator.
    impl, impl_ref = get_test_impls()
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        require_distinct_impls(scatter_kpool_tail_updates, impl, impl_ref)
    else:
        require_test_impl(scatter_kpool_tail_updates, impl)

    if accuracy_checked:
        actual = dict(inputs, **{name: inputs[name].clone() for name in state_names})
        reference = dict(inputs, **{name: inputs[name].clone() for name in state_names})
        scatter_kpool_tail_updates(**reference, impl=impl_ref)
        scatter_kpool_tail_updates(**actual, impl=impl)
        for name in state_names:
            assert_accuracy(actual[name], reference[name], EXACT_STANDARD, path=name)
        # Independent invariant: untouched rows must match the original exactly.
        for name, rows in unchanged.items():
            assert_accuracy(
                actual[name].view(-1, HEAD_DIM).index_select(0, rows),
                inputs[name].view(-1, HEAD_DIM).index_select(0, rows),
                EXACT_STANDARD,
                path=f"{name}.unchanged",
            )

    # Old tail values are not read: repeatedly overwrite the same output buffers.
    # Closure for in place operator. Refresh buffer to keep every iter equivalent
    def executor():
        return scatter_kpool_tail_updates(**inputs, impl=impl)

    executor.func = scatter_kpool_tail_updates
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    shape = (inputs["chunk_k"].shape[0], inputs["req_pool_idx"].numel())
    check_performance(executor, enable_perf=shape in perf_shape)


# Case 06: Reject a noncontiguous in-place tail.
@pytest.mark.extra_checks
def test_scatter_rejects_noncontiguous_tail():
    from sgl_kernel_npu.operators.mem_cache import scatter_kpool_tail_updates

    inputs, _ = generate_test_data(n_rows=1, total_tokens=4, req_pool_size=1)
    tail_k = inputs["tail_k"]
    inputs["tail_k"] = torch.stack((tail_k, tail_k), dim=-1)[..., 0]
    assert not inputs["tail_k"].is_contiguous()
    with pytest.raises(AssertionError, match="tail_k must be contiguous"):
        scatter_kpool_tail_updates(**inputs)

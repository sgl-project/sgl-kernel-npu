# In-place Operator
"""In-place cache assembly; isolated accuracy copies, reusable performance output."""


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

# W: Q candidates; Np is synthetic from full chunks, not a traced value. P/C: K=4.
perf_shape = [(4096, 1024)]

HEAD_DIM = 128
PAGE_SIZE = 128
CACHE_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=1e-2, rtol=1e-2),
}
INT8_CACHE_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=5e-2, rtol=5e-2),
}
FP8_CACHE_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=1e-1, rtol=1e-1),
}
SCALE_STANDARD = {
    "torch.float16": AccuracyStandard(atol=1e-3, rtol=1e-2),
    "torch.float32": AccuracyStandard(atol=1e-6, rtol=1e-2),
}


def dequantize_cache(cache, scale):
    rows = cache.float().reshape(-1, HEAD_DIM)
    return (rows * scale.float().reshape(-1, 1)).reshape(cache.shape).to(torch.bfloat16)


def generate_test_data(
    *,
    seed=100,
    n_pools=5,
    pool_size=8,
    tail_size=13,
    n_from_tail_values=None,
    score_dtype=torch.bfloat16,
    use_write_mask=False,
    total_tokens=None,
    use_noncontiguous_dynamic_inputs=False,
    num_programs=None,
    cache_dtype=torch.bfloat16,
    zero_keys=False,
):
    """Return operator kwargs and protected row IDs in the [-1, 128] view."""
    if n_from_tail_values is None:
        n_from_tail_values = [0] * n_pools
    assert len(n_from_tail_values) == n_pools
    assert all(0 <= value <= pool_size for value in n_from_tail_values)
    assert max(n_from_tail_values) <= tail_size

    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    chunk_src_starts = []
    chunk_rows = 0
    for n_tail in n_from_tail_values:
        chunk_src_starts.append(chunk_rows)
        chunk_rows += pool_size - n_tail

    # No sentinel: full-tail pools may point past the chunk without reading it.
    if total_tokens is not None:
        chunk_rows = total_tokens
        chunk_src_starts = [row * (total_tokens // n_pools) for row in range(n_pools)]
        assert all(
            start + pool_size - n_tail <= total_tokens
            for start, n_tail in zip(chunk_src_starts, n_from_tail_values)
        )
    n_requests = max(4, (n_pools + 1) // 2)
    cache_pages = max(2, (n_pools + PAGE_SIZE) // PAGE_SIZE)
    cache_rows = cache_pages * PAGE_SIZE

    def randn(shape, dtype):
        return torch.randn(shape, dtype=torch.float32).to(dtype)

    if cache_dtype == torch.bfloat16:
        initial_cache = randn((cache_pages, PAGE_SIZE, 1, HEAD_DIM), torch.bfloat16)
        scale_cache = None
    else:
        initial_cache = torch.zeros(
            (cache_pages, PAGE_SIZE, 1, HEAD_DIM), dtype=cache_dtype
        )
        scale_cache = torch.zeros(
            cache_rows,
            dtype=torch.float16 if cache_dtype == torch.int8 else torch.float32,
        )
    chunk_k = randn((chunk_rows, HEAD_DIM), torch.bfloat16)
    tail_k = randn((n_requests, tail_size, HEAD_DIM), torch.bfloat16)
    chunk_score = randn((chunk_rows, HEAD_DIM), score_dtype) * 4
    tail_score = randn((n_requests, tail_size, HEAD_DIM), score_dtype) * 4
    ape = randn((pool_size, HEAD_DIM), torch.float32) * 2
    if zero_keys:
        chunk_k.zero_()
        tail_k.zero_()

    req_pool_idx = torch.tensor(
        [(row * 3 + 1) % n_requests for row in range(n_pools)],
        dtype=torch.int64,
    )
    n_from_tail = torch.tensor(n_from_tail_values, dtype=torch.int32)
    chunk_src_start = torch.tensor(chunk_src_starts, dtype=torch.int64)
    tail_logical_base = torch.tensor(
        [(tail_size - 2 + row * 3) % tail_size for row in range(n_pools)],
        dtype=torch.int32,
    )
    loc = torch.tensor(
        [(row * 7 + 3) % cache_rows for row in range(n_pools)],
        dtype=torch.int64,
    )
    write_mask = None
    if use_write_mask:
        write_mask = torch.tensor(
            [row % 3 != 1 for row in range(n_pools)], dtype=torch.bool
        )

    inputs = {
        "index_k_cache": initial_cache.contiguous(),
        "num_programs": num_programs,
        "chunk_k": chunk_k.contiguous(),
        "chunk_score": chunk_score.contiguous(),
        "tail_k": tail_k.contiguous(),
        "tail_score": tail_score.contiguous(),
        "req_pool_idx": req_pool_idx.contiguous(),
        "n_from_tail": n_from_tail.contiguous(),
        "chunk_src_start": chunk_src_start.contiguous(),
        "tail_logical_base": tail_logical_base.contiguous(),
        "ape": ape.contiguous(),
        "loc": loc.contiguous(),
    }
    if write_mask is not None:
        inputs["write_mask"] = write_mask.contiguous()
    if scale_cache is not None:
        inputs["scale_cache"] = scale_cache.contiguous()
    if use_noncontiguous_dynamic_inputs:
        dynamic_names = ["chunk_k", "chunk_score", "ape", "loc"]
        if "write_mask" in inputs:
            dynamic_names.append("write_mask")
        for name in dynamic_names:
            tensor = inputs[name]
            inputs[name] = (
                torch.stack((tensor, tensor), dim=1)[:, 0]
                if tensor.ndim == 1
                else tensor.transpose(0, 1).contiguous().transpose(0, 1)
            )
            assert not inputs[name].is_contiguous()
    unchanged = {}
    if write_mask is not None:
        unchanged["index_k_cache"] = inputs["loc"][~inputs["write_mask"]]
    return inputs, unchanged


# Each row covers a distinct dtype, state transition, boundary, or workload.
@pytest.mark.parametrize(
    "case",
    [
        # Case 01: Pure chunk pooling with BF16 scores.
        pytest.param(dict(score_dtype=torch.bfloat16), id="bf16"),
        # Case 02: Pure chunk pooling with FP16 scores.
        pytest.param(dict(seed=101, score_dtype=torch.float16), id="fp16"),
        # Case 03: Pure chunk pooling with FP32 scores.
        pytest.param(dict(seed=102, score_dtype=torch.float32), id="fp32"),
        # Case 04: Mixed tail/chunk, ring wraparound, write mask and strided inputs.
        pytest.param(
            dict(
                seed=200,
                n_pools=9,
                tail_size=11,
                n_from_tail_values=[0, 1, 3, 7, 8, 2, 5, 8, 4],
                score_dtype=torch.float32,
                use_write_mask=True,
                use_noncontiguous_dynamic_inputs=True,
                num_programs=4,
            ),
            id="mixed_tail_chunk_wraparound_mask_and_noncontiguous_inputs",
        ),
        # Case 05: 37 pools with 3 programs: exercise the grid-stride loop.
        pytest.param(
            dict(
                seed=300,
                n_pools=37,
                pool_size=16,
                tail_size=19,
                n_from_tail_values=[row * 5 % 17 for row in range(37)],
                num_programs=3,
            ),
            id="grid_stride_when_pools_exceed_programs",
        ),
        # Case 06: All-tail pools with an empty chunk: do not read chunk storage.
        pytest.param(
            dict(
                seed=301,
                n_pools=3,
                pool_size=4,
                tail_size=5,
                n_from_tail_values=[4, 4, 4],
                score_dtype=torch.float32,
            ),
            id="all_tail_with_empty_chunk",
        ),
        # Case 07: Typical extend: Q4096, 1024 pools, pool4; also profile this case.
        pytest.param(
            dict(
                seed=400,
                n_pools=1024,
                pool_size=4,
                tail_size=5,
                n_from_tail_values=[2] * 1024,
                total_tokens=4096,
            ),
            id="W-extend-q4096-np1024",
        ),
        # Case 08: INT8 production extend shape; also profile this case.
        pytest.param(
            dict(
                seed=500,
                cache_dtype=torch.int8,
                n_pools=1024,
                pool_size=4,
                tail_size=5,
                n_from_tail_values=[2] * 1024,
                total_tokens=4096,
            ),
            id="W-extend-q4096-np1024-int8",
        ),
        # Case 09: FP8 production extend shape; also profile this case on A5.
        pytest.param(
            dict(
                seed=501,
                cache_dtype=torch.float8_e4m3fn,
                n_pools=1024,
                pool_size=4,
                tail_size=5,
                n_from_tail_values=[2] * 1024,
                total_tokens=4096,
            ),
            id="W-extend-q4096-np1024-fp8",
        ),
        # Case 10: Zero keys exercise the INT8 zero-scale fallback.
        pytest.param(
            dict(seed=502, cache_dtype=torch.int8, zero_keys=True),
            id="int8-zero-keys",
        ),
        # Case 11: Zero keys exercise the FP8 minimum-scale clamp.
        pytest.param(
            dict(seed=503, cache_dtype=torch.float8_e4m3fn, zero_keys=True),
            id="fp8-zero-keys",
        ),
    ],
)
def test_kpool_assemble_softmax_rotate_write_cache(case):
    """In-place: write cache, preserving mask-disabled rows exactly."""
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_assemble_softmax_rotate_write_cache,
    )

    if case.get("cache_dtype") == torch.float8_e4m3fn:
        from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

        if get_hardware().generation != Generation.ARCH35:
            pytest.skip("FP8 KPool cache coverage targets Ascend A5")

    # Inputs: named operator arguments and independent unchanged-row checks.
    inputs, unchanged = generate_test_data(**case)
    state_names = ["index_k_cache"]
    if "scale_cache" in inputs:
        state_names.append("scale_cache")
    # Accuracy: isolate DUT/reference state for this in-place operator.
    impl, impl_ref = get_test_impls()
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        require_distinct_impls(
            kpool_assemble_softmax_rotate_write_cache, impl, impl_ref
        )
    else:
        require_test_impl(kpool_assemble_softmax_rotate_write_cache, impl)

    if accuracy_checked:
        actual = dict(inputs, **{name: inputs[name].clone() for name in state_names})
        reference = dict(inputs, **{name: inputs[name].clone() for name in state_names})
        kpool_assemble_softmax_rotate_write_cache(**reference, impl=impl_ref)
        kpool_assemble_softmax_rotate_write_cache(**actual, impl=impl)
        cache_dtype = inputs["index_k_cache"].dtype
        if cache_dtype == torch.bfloat16:
            assert_accuracy(
                actual["index_k_cache"],
                reference["index_k_cache"],
                CACHE_STANDARD,
                path="cache",
            )
        else:
            standard = (
                FP8_CACHE_STANDARD
                if cache_dtype == torch.float8_e4m3fn
                else INT8_CACHE_STANDARD
            )
            assert_accuracy(
                dequantize_cache(actual["index_k_cache"], actual["scale_cache"]),
                dequantize_cache(reference["index_k_cache"], reference["scale_cache"]),
                standard,
                path="cache.dequantized",
            )
            assert_accuracy(
                actual["scale_cache"],
                reference["scale_cache"],
                SCALE_STANDARD,
                path="scale_cache",
            )
        # Independent invariant: untouched rows must match the original exactly.
        for name, rows in unchanged.items():
            if rows.numel() == 0:
                continue
            actual_bytes = actual[name].view(-1, HEAD_DIM).view(torch.uint8)
            original_bytes = inputs[name].view(-1, HEAD_DIM).view(torch.uint8)
            assert_accuracy(
                actual_bytes.index_select(0, rows),
                original_bytes.index_select(0, rows),
                path=f"{name}.unchanged",
            )

    # Output-buffer reuse: old cache values are never read by this operator.
    # This benchmark overwrites one buffer, rather than rotating fresh buffers.
    # Closure for in place operator. Refresh buffer to keep every iter equivalent
    def executor():
        return kpool_assemble_softmax_rotate_write_cache(**inputs, impl=impl)

    executor.func = kpool_assemble_softmax_rotate_write_cache
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    shape = (inputs["chunk_k"].shape[0], inputs["n_from_tail"].numel())
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.parametrize("cache_dtype", [torch.int8, torch.float8_e4m3fn])
def test_assemble_quantized_all_masked_preserves_cache(cache_dtype):
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_assemble_softmax_rotate_write_cache,
    )

    if cache_dtype == torch.float8_e4m3fn:
        from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

        if get_hardware().generation != Generation.ARCH35:
            pytest.skip("FP8 KPool cache coverage targets Ascend A5")

    impl, _ = get_test_impls()
    require_test_impl(kpool_assemble_softmax_rotate_write_cache, impl)
    cache = torch.full((4, HEAD_DIM), 7, dtype=torch.float32).to(cache_dtype)
    scale = torch.full(
        (4,), 0.25, dtype=torch.float16 if cache_dtype == torch.int8 else torch.float32
    )
    kpool_assemble_softmax_rotate_write_cache(
        cache,
        torch.randn(4, HEAD_DIM).bfloat16(),
        torch.randn(4, HEAD_DIM),
        torch.zeros(1, 4, HEAD_DIM, dtype=torch.bfloat16),
        torch.zeros(1, 4, HEAD_DIM),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int32),
        torch.zeros(4, HEAD_DIM),
        torch.tensor([2], dtype=torch.int64),
        write_mask=torch.tensor([False]),
        scale_cache=scale,
        impl=impl,
    )
    assert torch.all(cache.float() == 7)
    assert torch.all(scale == 0.25)

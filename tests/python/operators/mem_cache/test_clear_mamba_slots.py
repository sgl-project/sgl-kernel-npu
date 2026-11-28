"""Tests for mem_cache.clear_mamba_slots public API."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance

# Production Qwen3.8-27B SSM and Conv state pool shapes (48 GDN layers)
perf_shape = [
    (48, 32, 48, 128, 128),
    (48, 64, 3, 10240),
]


def generate_data(shape, num_clear, dtype, idx_dtype):
    torch.npu.manual_seed(42)
    buf = torch.randn(shape, dtype=dtype)
    pool_size = shape[1]
    if num_clear == 0:
        indices = torch.empty((0,), dtype=idx_dtype)
    else:
        step = max(1, pool_size // num_clear)
        raw = [(i * step) % pool_size for i in range(num_clear)]
        indices = torch.tensor(raw, dtype=idx_dtype)
    return buf, indices


@pytest.mark.parametrize(
    "shape,num_clear",
    [
        pytest.param((48, 32, 48, 128, 128), 8, id="qwen38-27b-ssm-pool"),
        pytest.param((48, 64, 3, 10240), 16, id="qwen38-27b-conv-pool"),
        pytest.param((2, 10, 3, 17), 3, id="edge-unaligned-tail"),
        pytest.param((4, 8, 16), 0, id="edge-empty-indices"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_clear_mamba_slots(shape, num_clear, dtype, idx_dtype):
    from sgl_kernel_npu.operators.mem_cache import clear_mamba_slots

    if shape == (48, 32, 48, 128, 128) and dtype != torch.bfloat16:
        pytest.skip("Large SSM pool benchmarked in production bfloat16 dtype")

    buf, indices = generate_data(shape, num_clear, dtype, idx_dtype)
    executor = check_accuracy(
        clear_mamba_slots,
        kwargs={"buf": buf, "indices": indices},
        mutable_kwargs=("buf",),
    )
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape and dtype == torch.bfloat16 and idx_dtype == torch.int64
        ),
    )


def test_clear_mamba_slots_large_stride_int64_no_overflow():
    """Verify 64-bit addressing when layer_stride * lid exceeds signed int32."""
    from sgl_kernel_npu.operators.mem_cache import clear_mamba_slots

    # 2 layers * 1_100_000_000 elements -> layer 1 starts past 2^31 - 1 (~2.2B)
    # Allocate compactly via a strided view backed by a small buffer? No, use a
    # real 1.1B-element uint8/bf16 tensor? Even simpler: 3 layers x 1 slot x 4B?
    # On 128 GB HBM, a 2.3B-element bf16 tensor is ~4.6 GB — fits easily.
    # For smaller-HBM devices, test 32 layers x 32 slots x (64*128*128) = 1.07B
    # plus 2x stride via slicing or check free HBM.
    free_bytes, _ = torch.npu.mem_get_info()
    if free_bytes < 10 * (1024**3):
        pytest.skip("Requires >= 10 GB free HBM for >2^31 element int64 overflow test")

    # 3 layers x 2 slots x 400_000_000 elements -> total 2.4B elements (> 2^31 - 1)
    # layer_stride = 800_000_000, lid=2 -> 1.6B, slot=1 -> +400M = 2.0B;
    # use 4 layers x 2 slots x 350_000_000 -> total 2.8B elements (~5.6 GB bf16)
    buf = torch.ones((4, 2, 350_000_000), dtype=torch.bfloat16)
    indices = torch.tensor([1], dtype=torch.int64)
    impl, impl_ref = get_test_impls()
    if not get_skip_acc_check():
        require_distinct_impls(clear_mamba_slots, impl, impl_ref)
    clear_mamba_slots(buf, indices, impl=impl)
    torch.npu.synchronize()
    # Cleared slot 1 must be zero across all layers including layer 3 (offset > 2^31)
    assert buf[:, 1, :16].abs().max().item() == 0.0
    assert buf[:, 1, -16:].abs().max().item() == 0.0
    # Untouched slot 0 must stay 1.0 everywhere
    assert_accuracy(buf[:, 0, :16], torch.ones((4, 16), dtype=torch.bfloat16))
    assert_accuracy(buf[:, 0, -16:], torch.ones((4, 16), dtype=torch.bfloat16))

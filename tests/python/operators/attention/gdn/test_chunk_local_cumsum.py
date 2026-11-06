"""Check scalar chunk sums against an independent torch implementation."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Qwen3-Next TP4 has 8 value heads; 2048 tokens is a synthetic prefill load.
perf_shape = [(1, 2048, 8)]


def generate_test_data(length, reverse, varlen, dtype):
    g = torch.randn(1, length, 8, dtype=dtype)
    offsets = torch.tensor([0, 31, length], dtype=torch.int32) if varlen else None
    return g, offsets


@pytest.mark.parametrize(
    "length", [64, 129, 2048], ids=["single-chunk", "chunk-tail", "prefill-2048"]
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_chunk_local_cumsum(length, reverse, varlen, dtype):
    from sgl_kernel_npu.operators.attention.gdn import chunk_local_cumsum

    g, offsets = generate_test_data(length, reverse, varlen, dtype)

    executor = check_accuracy(
        chunk_local_cumsum,
        args=(g, 64),
        # The existing Triton store requires matching input/output element types.
        kwargs={
            "reverse": reverse,
            "scale": 0.5,
            "cu_seqlens": offsets,
            "output_dtype": None,
        },
    )
    check_performance(
        executor,
        enable_perf=(
            tuple(g.shape) in perf_shape
            and dtype == torch.bfloat16
            and not reverse
            and not varlen
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["none", "explicit"])
@pytest.mark.extra_checks
def test_matching_output_dtype(dtype, mode):
    from sgl_kernel_npu.operators.attention.gdn import chunk_local_cumsum

    g, _ = generate_test_data(64, False, False, dtype)
    output_dtype = None if mode == "none" else dtype
    result = chunk_local_cumsum(g, 64, output_dtype=output_dtype, impl="triton")
    assert result.dtype == dtype


@pytest.mark.parametrize("impl", ["auto", "triton"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.extra_checks
def test_rejects_default_mismatched_output(dtype, impl):
    from sgl_kernel_npu.operators.attention.gdn import chunk_local_cumsum

    g, _ = generate_test_data(64, False, False, dtype)
    with pytest.raises(ValueError, match="output_dtype must match g.dtype"):
        chunk_local_cumsum(g, 64, impl=impl)


@pytest.mark.extra_checks
def test_default_output_dtype_for_float32():
    from sgl_kernel_npu.operators.attention.gdn import chunk_local_cumsum

    g, _ = generate_test_data(64, False, False, torch.float32)
    assert chunk_local_cumsum(g, 64, impl="triton").dtype == torch.float32


@pytest.mark.extra_checks
def test_rejects_explicit_mismatched_output():
    from sgl_kernel_npu.operators.attention.gdn import chunk_local_cumsum

    g, _ = generate_test_data(64, False, False, torch.float32)
    with pytest.raises(ValueError, match="output_dtype must match g.dtype"):
        chunk_local_cumsum(g, 64, output_dtype=torch.float16, impl="triton")

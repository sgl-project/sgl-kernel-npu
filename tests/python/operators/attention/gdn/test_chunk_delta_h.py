"""Test chunk states, updated values and final state against torch."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Qwen3-Next TP4: Hg=4, H=8, K=V=128; T=2048 is synthetic prefill.
perf_shape = [(1, 2048, 4, 8, 128, 128)]


def generate_test_data(length, varlen, dtype, head_dim):
    k = torch.randn(1, length, 4, head_dim, dtype=dtype) * 0.05
    w = torch.randn(1, length, 8, head_dim, dtype=dtype) * 0.05
    u = torch.randn(1, length, 8, head_dim, dtype=dtype) * 0.05
    g = -torch.rand(1, length, 8).cumsum(dim=1) * 0.01
    offsets = torch.tensor([0, 31, length], dtype=torch.int32) if varlen else None
    state = (
        torch.randn(2 if varlen else 1, 8, head_dim, head_dim, dtype=torch.float32)
        * 0.05
    )
    return k, w, u, g, state, offsets


@pytest.mark.parametrize(
    "length", [64, 129, 2048], ids=["single-chunk", "chunk-tail", "prefill-2048"]
)
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "head_dim",
    [
        pytest.param(128, id="model-head-dim-128"),
        pytest.param(64, marks=pytest.mark.extra_checks, id="edge-head-dim-64"),
    ],
)
def test_chunk_delta_h(length, varlen, dtype, head_dim):
    from sgl_kernel_npu.operators.attention.gdn import chunk_delta_h

    k, w, u, g, state, offsets = generate_test_data(length, varlen, dtype, head_dim)

    executor = check_accuracy(
        chunk_delta_h,
        args=(k, w, u, g, state),
        kwargs={
            "cu_seqlens": offsets,
            "output_final_state": True,
        },
        # Compare against CPU FP64 with the same rounding points. Storage in
        # FP32 does not remove the BF16/FP16 error accumulated by final_state.
        standard={
            str(dtype): AccuracyStandard(atol=1e-3, rtol=0),
            "torch.float32": AccuracyStandard(atol=1e-3, rtol=0),
        },
    )
    check_performance(
        executor,
        enable_perf=(
            (1, length, 4, 8, head_dim, head_dim) in perf_shape
            and dtype == torch.bfloat16
            and not varlen
        ),
    )

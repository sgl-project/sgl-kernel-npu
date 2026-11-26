import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

HEAD = 128
# Pure permutation: any mismatch is a layout bug, so compare bit exactly.
EXACT = {
    str(dtype): AccuracyStandard(atol=0.0, rtol=0.0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
}
perf_shape = [(1, 2, 4), (8192, 2, 4)]


def generate_test_data(shape, dtype, ba_dtype=None):
    """Build contiguous inputs; shape is (rows, num_heads_qk, num_heads_v)."""
    rows, num_heads_qk, num_heads_v = shape
    qkvz_width = 2 * num_heads_qk * HEAD + 2 * num_heads_v * HEAD
    torch.npu.manual_seed(42)
    return (
        torch.randn((rows, qkvz_width), dtype=dtype),
        torch.randn((rows, 2 * num_heads_v), dtype=ba_dtype or dtype),
    )


@pytest.mark.parametrize(
    "shape,dtype,ba_dtype",
    [
        # Qwen3.5 shares the Qwen3-Next head config; TP shards 16 key / 32 value.
        pytest.param((1, 2, 4), torch.bfloat16, None, id="decode-tp8-b1-2x4"),
        pytest.param((1, 4, 8), torch.float16, None, id="decode-tp4-b1-4x8"),
        pytest.param((8192, 2, 4), torch.bfloat16, None, id="prefill-tp8-b8192-2x4"),
        pytest.param((8192, 4, 8), torch.float16, None, id="prefill-tp4-b8192-4x8"),
        pytest.param((4, 4, 4), torch.bfloat16, None, id="ratio1-4x4"),
        pytest.param((4, 2, 8), torch.bfloat16, None, id="ratio4-2x8"),
        # Empty batch is defined for this layout and returns empty outputs.
        pytest.param((0, 2, 4), torch.bfloat16, None, id="edge-empty"),
        # b/a follow mixed_ba while mixed_qkv/z follow mixed_qkvz.
        pytest.param((7, 4, 8), torch.bfloat16, torch.float32, id="mixed-dtypes"),
    ],
)
def test_fused_qkvzba_split_reshape_cat_contiguous(shape, dtype, ba_dtype):
    from sgl_kernel_npu.operators.attention.gdn import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )

    mixed_qkvz, mixed_ba = generate_test_data(shape, dtype, ba_dtype)
    executor = check_accuracy(
        fused_qkvzba_split_reshape_cat_contiguous,
        args=(mixed_qkvz, mixed_ba, shape[1], shape[2], HEAD, HEAD),
        standard=EXACT,
    )
    check_performance(
        executor, enable_perf=shape in perf_shape and dtype == torch.bfloat16
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case,match",
    [
        ("not-2d", "2-D"),
        ("noncontig-qkvz", "contiguous"),
        ("noncontig-ba", "contiguous"),
        ("row-count", "row count"),
        ("ratio", "multiple"),
        ("width-qkvz", "row width"),
        ("width-ba", "row width"),
    ],
)
def test_rejects_invalid_inputs(case, match):
    """Each case violates exactly one checked constraint."""
    from sgl_kernel_npu.operators.attention.gdn import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )

    qk, v, rows = 4, 8, 4
    if case == "ratio":
        v = 6
    width = 2 * qk * HEAD + 2 * v * HEAD
    qkvz = torch.randn((rows, width), dtype=torch.bfloat16)
    ba = torch.randn((rows, 2 * v), dtype=torch.bfloat16)
    if case == "not-2d":
        qkvz = qkvz.reshape(rows, qk, -1)
    elif case == "noncontig-qkvz":
        qkvz = torch.randn((rows, width + 1), dtype=torch.bfloat16)[:, :width]
    elif case == "noncontig-ba":
        ba = torch.randn((rows, 2 * v + 1), dtype=torch.bfloat16)[:, : 2 * v]
    elif case == "row-count":
        ba = ba[:2]
    elif case == "width-qkvz":
        qkvz = torch.randn((rows, width - 1), dtype=torch.bfloat16)
    elif case == "width-ba":
        ba = torch.randn((rows, 2 * v - 2), dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match=match):
        fused_qkvzba_split_reshape_cat_contiguous(qkvz, ba, qk, v, HEAD, HEAD)

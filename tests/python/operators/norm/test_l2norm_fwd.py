import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Standards are selected by output dtype, including dtype overrides.
INPUT_STANDARDS = {
    "torch.float32": AccuracyStandard(atol=1e-6, rtol=1e-5),
    "torch.float16": AccuracyStandard(atol=1e-4, rtol=1e-3),
    "torch.bfloat16": AccuracyStandard(atol=1e-3, rtol=8e-3),
}


def generate_test_data(shape, zero, dtype):
    torch.npu.manual_seed(42)
    x = torch.zeros(shape, dtype=dtype) if zero else torch.randn(shape, dtype=dtype)
    return x


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize(
    "shape,eps,output_dtype,zero",
    [
        pytest.param((1, 1), 1e-6, None, False, id="minimum"),
        pytest.param((17, 127), 1e-6, None, False, id="opt-tail"),
        pytest.param(
            (2, 3, 512), 1e-6, torch.float32, False, id="opt-boundary-fp32-output"
        ),
        pytest.param(
            (17, 513), 1e-3, torch.bfloat16, False, id="kernel-tail-bf16-output"
        ),
        pytest.param((3, 1024), 0.0, torch.float16, False, id="kernel-zero-epsilon"),
        pytest.param((3, 128), 1e-6, None, True, id="zero-input"),
        pytest.param((0, 128), 1e-6, None, False, id="empty"),
        # Ascend GDN prefill passes contiguous Q/K as [tokens, key_heads / TP, 128].
        # Qwen3-Next-80B-A3B and Qwen3.5-35B-A3B both have 16 key heads.
        pytest.param(
            (128, 16, 128),
            1e-6,
            None,
            False,
            id="qwen3-next-80b-a3b-prefill-t128-tp1",
        ),
        pytest.param(
            (1024, 16, 128),
            1e-6,
            None,
            False,
            id="qwen3-5-35b-a3b-prefill-t1024-tp1",
        ),
        pytest.param(
            (256, 4, 128),
            1e-6,
            None,
            False,
            id="qwen3-5-35b-a3b-prefill-t256-tp4",
        ),
    ],
)
# Cover both native width branches, epsilon semantics, leading dimensions and output dtype.
def test_l2norm_fwd(shape, eps, output_dtype, zero, dtype):
    from sgl_kernel_npu.operators.norm import l2norm_fwd

    x = generate_test_data(shape, zero, dtype)
    executor = check_accuracy(
        l2norm_fwd,
        args=(x, eps, output_dtype),
        standard=INPUT_STANDARDS,
    )
    result = executor()
    assert result.shape == x.shape
    assert result.dtype == (dtype if output_dtype is None else output_dtype)
    check_performance(executor)

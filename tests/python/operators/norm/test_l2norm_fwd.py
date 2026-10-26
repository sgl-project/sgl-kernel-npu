import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

from ._inputs import seed_inputs


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
            (17, 513), 1e-3, torch.bfloat16, False, id="kernel1-tail-bf16-output"
        ),
        pytest.param((3, 1024), 0.0, torch.float16, False, id="kernel1-zero-epsilon"),
        pytest.param((3, 128), 1e-6, None, True, id="zero-input"),
        pytest.param((0, 128), 1e-6, None, False, id="empty"),
    ],
)
# Cover both native width branches, epsilon semantics, leading dimensions and output dtype.
def test_l2norm_fwd(shape, eps, output_dtype, zero, dtype):
    from sgl_kernel_npu.operators.norm import l2norm_fwd

    seed_inputs()
    x = torch.zeros(shape, dtype=dtype) if zero else torch.randn(shape, dtype=dtype)
    executor = check_accuracy(
        l2norm_fwd, args=(x, eps, output_dtype), impl="triton", impl_ref="torch_native"
    )
    result = executor()
    assert result.shape == x.shape
    assert result.dtype == (dtype if output_dtype is None else output_dtype)
    check_performance(executor)

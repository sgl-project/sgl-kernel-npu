"""Public RMSNorm+bias: each case validates accuracy before profiling."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance


def generate_test_data(shape, dtype, mode, seed=42):
    """Create all inputs directly on NPU; never stage tensors on CPU."""
    torch.npu.manual_seed(seed)
    x = torch.randn(shape, dtype=dtype)
    weight = torch.randn(shape[1], dtype=dtype)
    bias = torch.randn(shape[1], dtype=dtype)
    scale = offset = None
    if mode == "zero":
        x.zero_()
    elif mode == "offset_only":
        offset = torch.full((shape[1],), 3.0, dtype=torch.float32)
    elif mode == "quantized":
        # +/-1 inputs and quarter offsets avoid ambiguous rounding
        # boundaries; also exercise truncation and both saturation endpoints.
        x.fill_(1)
        x[:, 1::2] = -1
        weight.fill_(1)
        bias.zero_()
        scale = torch.tensor(
            [0.5, 2.0, 200.0, 200.0], dtype=torch.float32,
        ).repeat((shape[1] + 3) // 4)[:shape[1]].contiguous()
        offset = torch.full((shape[1],), 0.25, dtype=torch.float32)
    elif mode != "normal":
        raise ValueError(f"Unknown data mode: {mode}")
    return x, weight, bias, scale, offset


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1024), (7, 1536), (32, 4096)])
@pytest.mark.parametrize("mode", ["normal", "zero", "offset_only", "quantized"])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_rmsnorm_bias(
    shape,
    dtype,
    mode,
    eps,
):
    from sgl_kernel_npu.operators.norm import rmsnorm_bias
    x, weight, bias, scale, offset = generate_test_data(shape, dtype, mode)
    executor = check_accuracy(
        rmsnorm_bias,
        args=(x, weight, bias, eps, scale, offset),
    )

    check_performance(executor)

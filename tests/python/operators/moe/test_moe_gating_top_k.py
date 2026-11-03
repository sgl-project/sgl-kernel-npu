"""Version/hardware-constrained MoE gate accuracy and selected performance."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(1, 256, 8), (16, 256, 8)]


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 256, 8), id="synthetic-decode-1x256-k8"),
        pytest.param((16, 256, 8), id="synthetic-batch-16x256-k8"),
        pytest.param((128, 384, 6), id="synthetic-prefill-128x384-k6"),
    ],
)
@pytest.mark.parametrize("norm_type", [0, 1, 2])
@pytest.mark.parametrize("with_bias", [False, True])
def test_moe_gate(shape, norm_type, with_bias):
    from sgl_kernel_npu.operators.moe import moe_gating_top_k
    from sgl_kernel_npu.operators.utils._transformer_compat import cann_backend
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    legacy = cann_backend() == "legacy"
    if legacy and get_hardware().generation == Generation.ARCH22 and norm_type == 2:
        pytest.skip("CANN 9.1 A3 does not support norm_type=2; requires Ascend 950")
    # Deployed legacy custom bindings reject renorm=1. Exercise their supported
    # contract on both DUT and reference; do not silently change production args.
    renorm = 0 if legacy else 1
    rows, experts, k = shape
    torch.manual_seed(51)
    x = torch.randn(rows, experts, device="npu", dtype=torch.float32)
    bias = torch.randn(experts, device="npu") * 0.02 if with_bias else None
    executor = check_accuracy(
        moe_gating_top_k,
        args=(x, k),
        kwargs=dict(
            bias=bias,
            norm_type=norm_type,
            renorm=renorm,
            out_flag=True,
            routed_scaling_factor=2.5,
        ),
    )
    check_performance(
        executor, enable_perf=shape in perf_shape and norm_type == 2 and with_bias
    )


@pytest.mark.parametrize("mode", [0, 1])
def test_grouped_moe_gate(mode):
    from sgl_kernel_npu.operators.moe import moe_gating_top_k

    torch.manual_seed(52)
    x = torch.randn(16, 256, device="npu", dtype=torch.float32)
    bias = torch.randn(256, device="npu") * 0.02
    executor = check_accuracy(
        moe_gating_top_k,
        args=(x, 8),
        kwargs=dict(
            bias=bias, norm_type=1, group_count=8, k_group=4, group_select_mode=mode
        ),
    )
    check_performance(executor, enable_perf=False)

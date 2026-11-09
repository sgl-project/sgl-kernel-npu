"""Version/hardware-constrained MoE gate accuracy and selected performance."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(1, 256, 6), (16, 256, 6)]


@pytest.mark.extra_checks
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
    from sgl_kernel_npu.operators.utils.cann import detect_cann_version
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    cann_9_1 = detect_cann_version().startswith("9.1.")
    if cann_9_1 and get_hardware().generation == Generation.ARCH22 and norm_type == 2:
        pytest.skip("CANN 9.1 A3 does not support norm_type=2; requires Ascend 950")
    # Deployed cann_9_1 custom bindings reject renorm=1. Exercise their supported
    # contract on both DUT and reference; do not silently change production args.
    renorm = 0 if cann_9_1 else 1
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


@pytest.mark.extra_checks
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


def generate_data(tokens):
    """DeepSeek-V4-Flash router logits [T,256] and correction bias [256]."""
    torch.manual_seed(53)
    return (
        torch.randn(tokens, 256, dtype=torch.float32),
        torch.randn(256, dtype=torch.float32) * 0.02,
    )


@pytest.mark.parametrize(
    "tokens",
    [
        pytest.param(1, id="deepseek-v4-flash-decode-t1-e256-k6"),
        pytest.param(16, id="deepseek-v4-flash-decode-t16-e256-k6"),
    ],
)
def test_deepseek_v4_flash_moe_gate(tokens):
    from sgl_kernel_npu.operators.moe import moe_gating_top_k
    from sgl_kernel_npu.operators.utils.cann import detect_cann_version
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if (
        detect_cann_version().startswith("9.1.")
        and get_hardware().generation == Generation.ARCH22
    ):
        pytest.skip(
            "CANN 9.1 A3 kernel does not support the model's sqrtsoftplus mode (norm_type=2)"
        )
    x, bias = generate_data(tokens)
    executor = check_accuracy(
        moe_gating_top_k,
        args=(x, 6),
        kwargs=dict(
            bias=bias,
            norm_type=2,
            renorm=0,
            out_flag=False,
            routed_scaling_factor=1.5,
        ),
    )
    check_performance(executor, enable_perf=(tokens, 256, 6) in perf_shape)

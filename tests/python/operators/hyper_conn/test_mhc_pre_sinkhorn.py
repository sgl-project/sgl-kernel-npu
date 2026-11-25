import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Register the bundled vendor before the device fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Source: sglang/srt/configs/glm5_next.py: hidden_size=4096, hc_mult=4.
# GLM-5.3-Flash cases borrow these dimensions; token counts are illustrative,
# not traced production inputs. Use model widths supported on CANN 9.1/9.2.
perf_shape = [(1, 4096), (128, 4096)]

# CANN projects with HF32; post/comb remain FP32 but differ from the FP32
# reference GEMM. This operator-specific budget is calibrated on Ascend910_9362
# with CANN 9.1 over nine seeds, both activation dtypes and both eps values.
# These budgets check synthetic operator accuracy, not model-level accuracy.
# BF16/FP16 activation budgets allow dtype rounding and near-zero cancellation.
# The same thresholds also passed on Ascend950DT_9582 with CANN 9.1.
PRE_STANDARD = {
    "torch.float32": AccuracyStandard(atol=5e-5, rtol=5e-5),
    "torch.float16": AccuracyStandard(atol=1e-5, rtol=2e-3),
    "torch.bfloat16": AccuracyStandard(atol=1e-5, rtol=1e-2),
}


def generate_data(tokens, hidden, dtype):
    torch.manual_seed(42)
    x = torch.randn(tokens, 4, hidden, dtype=dtype) * 0.1
    phi = torch.randn(24, 4 * hidden, dtype=torch.float32) / (4 * hidden) ** 0.5
    alpha = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    bias = torch.randn(24, dtype=torch.float32) * 0.1
    return x, phi, alpha, bias


@pytest.mark.parametrize(
    "tokens,hidden",
    [
        pytest.param(1, 4096, id="glm53-flash-decode-1x4x4096"),
        pytest.param(128, 4096, id="glm53-flash-extend-128x4x4096"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hc_eps,norm_eps", [(1e-6, 1e-6), (1e-5, 1e-5)])
def test_mhc_pre_sinkhorn(tokens, hidden, dtype, hc_eps, norm_eps):
    from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

    inputs = generate_data(tokens, hidden, dtype)
    executor = check_accuracy(
        mhc_pre_sinkhorn,
        args=inputs,
        kwargs={"hc_eps": hc_eps, "norm_eps": norm_eps},
        standard=PRE_STANDARD,
    )
    check_performance(
        executor,
        enable_perf=(tokens, hidden) in perf_shape
        and dtype == torch.bfloat16
        and hc_eps == 1e-6,
    )


def test_a3_return_pre_graph_replay():
    from sgl_kernel_npu.operators.hyper_conn.mhc_pre_sinkhorn import (
        _has_bundled_v2,
        mhc_pre_sinkhorn,
    )

    if not _has_bundled_v2():
        pytest.skip("requires the bundled mHC V2 vendor")

    x, phi, alpha, bias = (a.npu() for a in generate_data(6, 5120, torch.bfloat16))
    args = (x, phi, alpha, bias)
    kwargs = {"norm_eps": 1e-20}
    old = mhc_pre_sinkhorn(*args, **kwargs)
    new = mhc_pre_sinkhorn(*args, **kwargs, return_pre=True)
    assert len(old) == 3
    assert len(new) == 4
    torch.testing.assert_close(new[0], old[0], atol=0, rtol=0)
    # The backward-output tiling can round FP32 coefficients differently.
    for result, expected in zip(new[1:3], old[1:], strict=True):
        torch.testing.assert_close(result, expected, atol=2e-7, rtol=2e-7)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = mhc_pre_sinkhorn(*args, **kwargs, return_pre=True)
    for factor in (1.5, -0.8):
        x.mul_(factor)
        graph.replay()
        torch.npu.synchronize()
        expected = mhc_pre_sinkhorn(
            *args, **kwargs, return_pre=True, impl="torch_native"
        )
        for result, reference in zip(captured[1:], expected[1:], strict=True):
            torch.testing.assert_close(result, reference, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("tokens", [18, 24])
def test_a3_v2_premix_graph_replay(tokens):
    from sgl_kernel_npu.operators.hyper_conn.mhc_pre_sinkhorn import (
        _has_bundled_v2,
        mhc_pre_sinkhorn,
    )

    if not _has_bundled_v2():
        pytest.skip("requires the bundled mHC V2 vendor")

    with torch.device("cpu"):
        cpu_args = generate_data(tokens, 5120, torch.bfloat16)
        previous = torch.zeros((tokens, 4), dtype=torch.float32)
    args = tuple(t.npu() for t in cpu_args)
    x = args[0]
    previous[:, 0] = 1
    pre_mix = previous.npu()
    kwargs = {"pre_mix": pre_mix, "return_pre": True, "norm_eps": 1e-20}
    eager = mhc_pre_sinkhorn(*args, **kwargs, impl="cann_ops_transformer_v2")
    torch.testing.assert_close(eager[0].cpu(), cpu_args[0][:, 0], atol=0, rtol=0)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = mhc_pre_sinkhorn(*args, **kwargs, impl="cann_ops_transformer_v2")

    generator = torch.Generator().manual_seed(tokens)
    for _ in range(2):
        changed_x = (
            torch.randn(x.shape, dtype=x.dtype, device="cpu", generator=generator) * 0.1
        )
        changed_pre = torch.rand(previous.shape, device="cpu", generator=generator)
        x.copy_(changed_x)
        pre_mix.copy_(changed_pre)
        graph.replay()
        torch.npu.synchronize()
        expected = mhc_pre_sinkhorn(
            changed_x,
            *cpu_args[1:],
            pre_mix=changed_pre,
            return_pre=True,
            norm_eps=1e-20,
            impl="torch_native",
        )
        torch.testing.assert_close(outputs[0].cpu(), expected[0], atol=1e-5, rtol=1e-2)
        for result, reference in zip(outputs[1:], expected[1:], strict=True):
            torch.testing.assert_close(result.cpu(), reference, atol=2e-4, rtol=2e-4)

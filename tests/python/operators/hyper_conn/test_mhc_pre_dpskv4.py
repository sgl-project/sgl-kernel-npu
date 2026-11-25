"""mHC pre accuracy and selected-shape performance through public APIs."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

MC_PRE_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0.03, rtol=0.03),
    "torch.float32": AccuracyStandard(atol=0.03, rtol=0.03),
}
RECIPES_IMPL = "cann_recipes_infer_cann920"

# DeepSeek-V4-Flash: hidden_size=4096, hc_mult=4.
perf_shape = [(1, 4, 4096), (16, 4, 4096)]
mhc_shapes = [
    pytest.param((1, 4, 4096), id="deepseek-v4-flash-decode-t1-hc4-h4096"),
    pytest.param((16, 4, 4096), id="deepseek-v4-flash-decode-t16-hc4-h4096"),
]


def generate_data(shape):
    torch.manual_seed(11)
    _, streams, hidden = shape
    width = streams * (streams + 2)
    x = torch.randn(shape, dtype=torch.bfloat16)
    fn = torch.randn(width, streams * hidden, dtype=torch.float32) * 0.01
    scale = torch.ones(3, dtype=torch.float32)
    base = torch.zeros(width, dtype=torch.float32)
    return (x, fn, scale, base, streams, 20, 1e-6, 1e-6)


@pytest.mark.parametrize("shape", mhc_shapes)
def test_mhc_pre_dpskv4(shape):
    from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

    args = generate_data(shape)
    executor = check_accuracy(mhc_pre_dpskv4, args=args, standard=MC_PRE_STANDARD)
    check_performance(executor, enable_perf=shape in perf_shape)


def _recipes_unavailable():
    dependencies = getattr(sgl_kernel_npu, "_manifest", {}).get("third_party", ())
    return not any("npu_hc_pre_v2" in item["ops"] for item in dependencies)


@pytest.mark.skipif(_recipes_unavailable(), reason="requires bundled HcPre")
@pytest.mark.parametrize("shape", mhc_shapes)
@pytest.mark.parametrize("return_pre", [False, True], ids=["default", "return-pre"])
def test_mhc_pre_dpskv4_dsv41(shape, return_pre):
    from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

    args = generate_data(shape)
    executor = check_accuracy(
        mhc_pre_dpskv4,
        args=args,
        kwargs={"return_pre": return_pre},
        impl=RECIPES_IMPL,
        impl_ref="torch_native",
        standard=MC_PRE_STANDARD,
    )
    check_performance(executor, enable_perf=shape in perf_shape and not return_pre)


@pytest.mark.skipif(_recipes_unavailable(), reason="requires bundled HcPre")
@pytest.mark.parametrize("shape", mhc_shapes)
def test_mhc_pre_dpskv4_dsv41_premix(shape):
    from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

    args = generate_data(shape)
    # pre_mix only weights y; the returned pre is still computed internally.
    pre_mix = torch.sigmoid(torch.randn((shape[0], shape[1]), dtype=torch.float32))
    executor = check_accuracy(
        mhc_pre_dpskv4,
        args=args,
        kwargs={"pre_mix": pre_mix, "return_pre": True},
        impl="auto",
        impl_ref="torch_native",
        standard=MC_PRE_STANDARD,
    )
    check_performance(executor, enable_perf=False)


def test_mhc_pre_dpskv4_pre_outputs_rejected():
    from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

    args = generate_data((1, 4, 4096))
    with pytest.raises(ValueError, match="require impl"):
        mhc_pre_dpskv4(*args, return_pre=True, impl="cann_ops_transformers_cann920")
    with pytest.raises(ValueError, match="require impl"):
        mhc_pre_dpskv4(
            *args, pre_mix=torch.zeros(1, 4), impl="cann_recipes_infer_cann910"
        )


@pytest.mark.skipif(_recipes_unavailable(), reason="requires bundled HcPre")
@pytest.mark.parametrize("tokens", [18, 24, 4096])
def test_mhc_pre_dpskv4_premix_graph_replay(tokens):
    from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

    with torch.device("cpu"):
        x, fn, scale, base, *_ = generate_data((tokens, 4, 5120))
        scale = torch.tensor([0.1, 0.2, 0.3])
        premix = torch.zeros(tokens, 4)
        premix[:, 0] = 1
    x_npu, fn_npu, scale_npu, base_npu, premix_npu = (
        t.npu() for t in (x, fn, scale, base, premix)
    )

    def run():
        return mhc_pre_dpskv4(
            x_npu,
            fn_npu,
            scale_npu,
            base_npu,
            4,
            20,
            1e-6,
            1e-20,
            pre_mix=premix_npu,
            return_pre=True,
        )

    torch.testing.assert_close(run()[0].cpu(), x[:, 0], atol=0, rtol=0)
    torch.npu.synchronize()
    if tokens < 4096:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run()
    for _ in range(2 if tokens < 4096 else 1):
        with torch.device("cpu"):
            x = torch.randn_like(x)
            premix = torch.rand_like(premix)
        x_npu.copy_(x)
        premix_npu.copy_(premix)
        if tokens < 4096:
            graph.replay()
        else:
            output = run()
        torch.npu.synchronize()
        expected = mhc_pre_dpskv4(
            x,
            fn,
            scale,
            base,
            4,
            20,
            1e-6,
            1e-20,
            pre_mix=premix,
            return_pre=True,
            impl="torch_native",
        )
        torch.testing.assert_close(output[0].cpu(), expected[0], atol=2e-4, rtol=8e-3)
        for actual, reference in zip(output[1:], expected[1:], strict=True):
            torch.testing.assert_close(actual.cpu(), reference, atol=2e-4, rtol=2e-4)

"""LayerNorm accuracy through the public normalization interface.

The device operator also returns mean/rstd, but the sglang call site discards
them, so only the normalized output is compared by value; the statistics are
checked for shape and dtype. Inputs are built on CPU because the public
interface fixture sets the default device to NPU.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# One representative inference-shaped call; only this tuple is profiled.
perf_shape = [((2048, 4096), (4096,), "fp32")]

# FP16 output is rounded from FP32 internals, so compare with the wider
# tolerance used by the other half-precision operator tests. FP32 runs on a
# different reduction order, which needs slightly more than the default.
LAYER_NORM_STANDARD = {
    "torch.float32": AccuracyStandard(atol=1e-4, rtol=1e-4),
    "torch.float16": AccuracyStandard(atol=1e-2, rtol=1e-2),
}


def generate_test_data(shape, normalized_shape, dtype, weight_dtype, affine, bias):
    """Build CPU inputs and optional affine parameters, then move them to NPU."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    values = torch.randn(
        shape, generator=generator, dtype=torch.float32, device="cpu"
    ).to(dtype)
    weight = None
    bias_value = None
    if affine:
        weight = (
            1.0
            + 0.125
            * torch.randn(
                normalized_shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
        ).to(weight_dtype)
    if bias:
        bias_value = (
            0.125
            * torch.randn(
                normalized_shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
        ).to(weight_dtype)
    return dict(
        input=values.npu(),
        normalized_shape=list(normalized_shape),
        weight=None if weight is None else weight.npu(),
        bias=None if bias_value is None else bias_value.npu(),
        eps=1e-5,
        impl_mode=0,
    )


def stat_shape(shape, normalized_shape):
    """Legacy mean/rstd shape: kept axes followed by one per normalized axis."""
    begin_axis = len(shape) - len(normalized_shape)
    return tuple(shape[:begin_axis]) + (1,) * len(normalized_shape)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize(
    "shape,normalized_shape,weight_mode,bias",
    [
        # sglang FP32LayerNorm: half input with cached FP32 affine parameters.
        pytest.param((512, 4096), (4096,), "fp32", True, id="fp32-affine"),
        # Same-dtype affine parameters exercise the no-cast path.
        pytest.param((512, 4096), (4096,), "same", True, id="same-dtype-affine"),
        # Weight-only and no-affine branches.
        pytest.param((128, 2048), (2048,), "same", False, id="weight-only"),
        pytest.param((64, 1024), (1024,), None, False, id="no-affine"),
        # Normalizing two trailing axes.
        pytest.param((8, 128, 512), (128, 512), "fp32", True, id="two-axes"),
        pytest.param((2048, 4096), (4096,), "fp32", True, id="perf-tnd"),
    ],
)
def test_layer_norm(shape, normalized_shape, weight_mode, bias, dtype):
    from sgl_kernel_npu.operators.norm import layernorm

    impl, impl_ref = get_test_impls()
    weight_dtype = torch.float32 if weight_mode == "fp32" else dtype
    data = generate_test_data(
        shape, normalized_shape, dtype, weight_dtype, weight_mode is not None, bias
    )
    input_before = data["input"].clone()
    weight_before = None if data["weight"] is None else data["weight"].clone()
    bias_before = None if data["bias"] is None else data["bias"].clone()

    if get_skip_acc_check():
        require_test_impl(layernorm, impl)
        out, mean, rstd = layernorm(**data, impl=impl)
        accuracy_checked = False
    else:
        require_distinct_impls(layernorm, impl, impl_ref)
        result_ref = layernorm(**data, impl=impl_ref)
        result = layernorm(**data, impl=impl)
        out, mean, rstd = result
        for name, actual, expected in zip(("out", "mean", "rstd"), result, result_ref):
            assert actual.shape == expected.shape, f"{name}: shape mismatch"
            assert actual.dtype == expected.dtype, f"{name}: dtype mismatch"
            assert actual.device == expected.device, f"{name}: device mismatch"
        assert_accuracy(out, result_ref[0], standard=LAYER_NORM_STANDARD, path="out")
        accuracy_checked = True

    assert out.shape == data["input"].shape
    assert out.dtype == dtype
    assert out.device == data["input"].device
    expected_stat_shape = stat_shape(shape, normalized_shape)
    assert mean.shape == expected_stat_shape
    assert rstd.shape == expected_stat_shape
    assert mean.dtype == dtype
    assert rstd.dtype == dtype
    assert torch.equal(data["input"], input_before)
    if weight_before is not None:
        assert torch.equal(data["weight"], weight_before)
    if bias_before is not None:
        assert torch.equal(data["bias"], bias_before)

    repeated, _, _ = layernorm(**data, impl=impl)
    assert torch.equal(out, repeated)

    executor = make_executor(
        layernorm,
        kwargs=data,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=accuracy_checked,
    )
    check_performance(
        executor,
        enable_perf=(shape, normalized_shape, weight_mode) in perf_shape
        and bias
        and dtype == torch.float16,
    )

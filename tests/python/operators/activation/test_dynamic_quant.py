"""Per-row dynamic INT8 quantization through the public API.

The INT8 payload can differ by one step across independent rounding paths, so
accuracy is checked after dequantizing with the shared per-row FP32 scales, as
in the dynamic MXFP8 test.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

_PERF_SHAPES = ((4096, 128), (16, 7168))

CASES = [
    # GLM-5.3-Flash DSA indexer query shapes after reshape(-1, 128).
    pytest.param((64, 128), torch.bfloat16, "normal", id="indexer-64x128-bf16"),
    pytest.param((4096, 128), torch.bfloat16, "normal", id="indexer-4096x128-bf16"),
    pytest.param((2, 8, 128), torch.bfloat16, "normal", id="indexer-3d-bf16"),
    pytest.param((16, 7168), torch.bfloat16, "normal", id="wide-7168-bf16"),
    pytest.param((64, 128), torch.float16, "normal", id="indexer-fp16"),
    pytest.param((64, 128), torch.float32, "normal", id="indexer-fp32"),
    pytest.param((64, 128), torch.bfloat16, "zero", id="zero-bf16"),
    # Boundary shapes stay out of the default run and only execute under
    # ``-m extra_checks``. Unlike MXFP8 there is no block-size constraint.
    pytest.param(
        (1, 128),
        torch.bfloat16,
        "normal",
        id="single-row",
        marks=pytest.mark.extra_checks,
    ),
    pytest.param(
        (7, 96),
        torch.bfloat16,
        "normal",
        id="odd-rows",
        marks=pytest.mark.extra_checks,
    ),
    pytest.param(
        (7, 96),
        torch.bfloat16,
        "zero",
        id="odd-rows-zero",
        marks=pytest.mark.extra_checks,
    ),
]


def generate_test_data(shape, dtype, mode, seed=7):
    """Indexer-style input on the NPU for each value mode."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "zero":
        x = torch.zeros(shape, dtype=torch.float32)
    else:
        x = torch.randn(shape, generator=generator, dtype=torch.float32) * 2
    return x.to(dtype).to("npu")


def dequantize(payload, scale):
    """Apply the per-row FP32 scales back to the INT8 payload."""
    return payload.float() * scale.unsqueeze(-1)


@pytest.mark.parametrize("shape,dtype,mode", CASES)
def test_dynamic_quant(shape, dtype, mode):
    from sgl_kernel_npu.operators.activation import dynamic_quant

    x = generate_test_data(shape, dtype, mode)
    x_copy = x.clone()
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(dynamic_quant, impl)
    else:
        require_distinct_impls(dynamic_quant, impl, impl_ref)
        ref_payload, ref_scale = dynamic_quant(x, impl=impl_ref)

    payload, scale = dynamic_quant(x, impl=impl)

    assert payload.shape == x.shape
    assert payload.dtype == torch.int8
    assert scale.shape == x.shape[:-1]
    assert scale.dtype == torch.float32
    # Byte-wise comparison so the input is verified unchanged.
    assert torch.equal(
        x.view(torch.uint8), x_copy.view(torch.uint8)
    ), "x must not be modified"

    if not skip_accuracy:
        actual = dequantize(payload, scale)
        expected = dequantize(ref_payload, ref_scale)
        # One INT8 step is the per-row scale; allow a couple of steps for the
        # independent rounding paths (randn*2 inputs keep amax bounded).
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.15)

    executor = make_executor(
        dynamic_quant,
        args=(x,),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    check_performance(
        executor,
        enable_perf=(
            shape in _PERF_SHAPES and dtype == torch.bfloat16 and mode == "normal"
        ),
    )

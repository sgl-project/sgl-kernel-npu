"""A5-only dynamic MXFP8 quantization through the public API.

The E4M3FN payload cannot be compared bitwise across independent rounding
paths, so accuracy is checked after dequantizing with the shared E8M0 scales,
as in the fused SiTU + MXFP8 test.
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

_MX_BLOCK = 32
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
    # Boundary shapes and special values stay out of the default run and only
    # execute under ``-m extra_checks``.
    pytest.param(
        (1, 32),
        torch.bfloat16,
        "normal",
        id="single-block",
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
    pytest.param(
        (7, 100),
        torch.bfloat16,
        "normal",
        id="partial-block",
        marks=pytest.mark.extra_checks,
    ),
    pytest.param(
        (8, 128),
        torch.bfloat16,
        "special",
        id="special-values",
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
    if mode == "special":
        rows = x.reshape(-1, shape[-1])
        # Exact powers of two exercise the floor(log2) corrections.
        rows[0] = torch.pow(
            2.0, (torch.arange(shape[-1], dtype=torch.float32) % 17) - 8.0
        )
        if rows.shape[0] > 1:
            rows[1, :_MX_BLOCK] = float("inf")
        if rows.shape[0] > 2:
            rows[2, :_MX_BLOCK] = float("nan")
        if rows.shape[0] > 3:
            rows[3] = 0.0
    return x.to(dtype).to("npu")


def dequantize(payload, scales):
    """Apply the per-32-element E8M0 scales back to the E4M3FN payload."""
    scale = (
        scales.contiguous().view(torch.uint8).reshape(*payload.shape[:-1], -1).float()
    )
    scale = torch.pow(2.0, scale - 127.0)
    return (
        payload.float()
        * scale.repeat_interleave(_MX_BLOCK, dim=-1)[..., : payload.shape[-1]]
    )


@pytest.mark.parametrize("shape,dtype,mode", CASES)
def test_dynamic_mx_quant(shape, dtype, mode):
    from sgl_kernel_npu.operators.activation import dynamic_mx_quant

    x = generate_test_data(shape, dtype, mode)
    x_copy = x.clone()
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(dynamic_mx_quant, impl)
    else:
        require_distinct_impls(dynamic_mx_quant, impl, impl_ref)
        ref_payload, ref_scales = dynamic_mx_quant(
            x, dst_type=torch.float8_e4m3fn, impl=impl_ref
        )

    payload, scales = dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, impl=impl)

    assert payload.shape == x.shape
    assert payload.dtype == torch.float8_e4m3fn
    assert scales.shape == (*x.shape[:-1], (x.shape[-1] + 63) // 64, 2)
    assert scales.dtype == torch.uint8
    # Byte-wise comparison so NaN inputs do not break the equality check.
    assert torch.equal(
        x.view(torch.uint8), x_copy.view(torch.uint8)
    ), "x must not be modified"

    if not skip_accuracy:
        actual = dequantize(payload, scales)
        expected = dequantize(ref_payload, ref_scales)
        torch.testing.assert_close(torch.isnan(actual), torch.isnan(expected))
        torch.testing.assert_close(
            actual, expected, rtol=0.08, atol=0.08, equal_nan=True
        )

    executor = make_executor(
        dynamic_mx_quant,
        args=(x,),
        kwargs={"dst_type": torch.float8_e4m3fn},
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

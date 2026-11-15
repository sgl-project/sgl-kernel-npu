"""A5-only fused SiTU + MXFP8 quantization through the public API.

The E4M3FN payload cannot be compared bitwise across independent rounding
paths, so accuracy is checked after dequantizing with the shared E8M0 scales,
as in the legacy operator test.
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

_COUNTS = (0, 7, 13, 0, 9, 3, 0, 5)
_GATE_COLS = 3072

CASES = [
    pytest.param(128, _COUNTS, 1, torch.int64, id="count-int64"),
    pytest.param(128, _COUNTS, 1, torch.int32, id="count-int32"),
    pytest.param(128, _COUNTS, 0, torch.int64, id="cumulative-int64"),
    pytest.param(128, _COUNTS, 0, torch.int32, id="cumulative-int32"),
    pytest.param(37, (37,), 1, torch.int64, id="single-expert"),
    pytest.param(16, (0, 0, 0), 1, torch.int64, id="zero-valid-rows"),
    pytest.param(8, (5, 6, 7), 1, torch.int64, id="sum-exceeds-capacity"),
    pytest.param(32768, (4, 8, 3, 0, 6, 2, 5, 4), 1, torch.int64, id="capacity-32768"),
    pytest.param(8192, tuple([512] * 8), 1, torch.int64, id="moe-dispatch-half-filled"),
    # Boundary shapes stay out of the default run and only execute under
    # ``-m extra_checks``.
    pytest.param(
        1, (1,), 1, torch.int64, id="single-row", marks=pytest.mark.extra_checks
    ),
    pytest.param(
        64,
        tuple([1] * 32),
        1,
        torch.int32,
        id="many-experts",
        marks=pytest.mark.extra_checks,
    ),
]


def generate_test_data(capacity, counts, group_list_type, dtype, seed=7):
    """BF16 GMM1-style input with both group_list layouts."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(
        (capacity, 2 * _GATE_COLS),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    x = (x * 2).to(torch.bfloat16).to("npu")
    counts_tensor = torch.tensor(counts, dtype=dtype)
    group_list = counts_tensor.cumsum(0) if group_list_type == 0 else counts_tensor
    return x, group_list.to("npu")


def valid_rows(group_list, group_list_type):
    if group_list_type == 0:
        value = int(group_list[-1].item())
    else:
        value = int(group_list.sum().item())
    return max(value, 0)


def dequantize(payload, scales):
    """Apply the per-32-element E8M0 scales back to the E4M3FN payload."""
    scale = scales.contiguous().view(torch.uint8).reshape(scales.shape[0], -1).float()
    scale = torch.pow(2.0, scale - 127.0)
    return payload.float() * scale.repeat_interleave(32, dim=-1)


@pytest.mark.parametrize("capacity,counts,group_list_type,dtype", CASES)
def test_situ_mxfp8_quant(capacity, counts, group_list_type, dtype):
    from sgl_kernel_npu.operators.activation import situ_mxfp8_quant

    x, group_list = generate_test_data(capacity, counts, group_list_type, dtype)
    x_copy = x.clone()
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(situ_mxfp8_quant, impl)
    else:
        require_distinct_impls(situ_mxfp8_quant, impl, impl_ref)
        ref_payload, ref_scales = situ_mxfp8_quant(
            x, group_list, group_list_type, impl=impl_ref
        )

    payload, scales = situ_mxfp8_quant(x, group_list, group_list_type, impl=impl)

    assert payload.shape == (capacity, _GATE_COLS)
    assert payload.dtype == torch.float8_e4m3fn
    assert scales.shape == (capacity, _GATE_COLS // 64, 2)
    assert scales.dtype == torch.float8_e8m0fnu
    assert torch.equal(x, x_copy), "hidden_states must not be modified"

    valid = min(valid_rows(group_list, group_list_type), capacity)
    if not skip_accuracy and valid > 0:
        actual = dequantize(payload[:valid], scales[:valid])
        expected = dequantize(ref_payload[:valid], ref_scales[:valid])
        torch.testing.assert_close(torch.isnan(actual), torch.isnan(expected))
        torch.testing.assert_close(
            actual, expected, rtol=0.08, atol=0.08, equal_nan=True
        )

    executor = make_executor(
        situ_mxfp8_quant,
        args=(x, group_list),
        kwargs={"group_list_type": group_list_type},
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy and valid > 0,
    )
    check_performance(executor, enable_perf=capacity >= 8192)


@pytest.mark.extra_checks
def test_rejects_invalid_contract():
    from sgl_kernel_npu.operators.activation import situ_mxfp8_quant

    x = torch.zeros((4, 2 * _GATE_COLS), dtype=torch.bfloat16, device="npu")
    group_list = torch.ones(2, dtype=torch.int64, device="npu")

    with pytest.raises(ValueError):
        situ_mxfp8_quant(
            torch.zeros((4, _GATE_COLS), dtype=torch.bfloat16, device="npu"),
            group_list,
        )
    with pytest.raises(ValueError):
        situ_mxfp8_quant(x.float(), group_list)
    with pytest.raises(ValueError):
        situ_mxfp8_quant(x, torch.ones(2, dtype=torch.float32, device="npu"))
    with pytest.raises(ValueError):
        situ_mxfp8_quant(x, group_list, group_list_type=2)
    with pytest.raises(ValueError):
        situ_mxfp8_quant(x, group_list, beta=0.0)
    with pytest.raises(ValueError):
        situ_mxfp8_quant(x, group_list, linear_beta=-1.0)
    with pytest.raises(RuntimeError):
        non_contiguous = torch.zeros(
            (4, 4 * _GATE_COLS), dtype=torch.bfloat16, device="npu"
        )[:, ::2]
        situ_mxfp8_quant(non_contiguous, group_list)

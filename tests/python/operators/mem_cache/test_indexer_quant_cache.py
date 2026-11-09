"""Indexer FP8 cache-write accuracy, including slot zero and the -1 sentinel."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

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

perf_shape = [(8, 128)]


def generate_data(tokens, input_kind, round_scale):
    torch.manual_seed(16)
    x = torch.randn(tokens, 128, dtype=torch.bfloat16)
    if input_kind == "zero":
        x.zero_()
    elif input_kind == "scale-policy":
        # amax/448 is not a power of two. This distinguishes cann_9_1 9.1
        # unrounded scales from the released 9.2 round_scale=True behavior.
        x.fill_(3.0)
        x[0, 1:4] = torch.tensor([-3.0, 1.0, -1.0], dtype=x.dtype, device=x.device)
    elif input_kind == "boundaries":
        # With amax=448, scale=1: exercise E4M3FN ties-to-even, signed zero,
        # subnormals, the minimum normal, exponent transitions and endpoints.
        edge = torch.tensor(
            [
                -448,
                -432,
                -272,
                -248,
                -1.1875,
                -1.0625,
                -0.0,
                0.0,
                2**-10,
                3 * 2**-10,
                2**-6,
                1.0625,
                1.1875,
                248,
                272,
                432,
                448,
            ],
            dtype=x.dtype,
            device=x.device,
        )
        x.zero_()
        x[0, : edge.numel()] = edge
    slots = torch.arange(tokens, dtype=torch.int32)
    if tokens > 1:
        slots[-1] = -1
    # CANN 9.1 may not support generic FP8 initialization/copy/conversion.
    keys = torch.zeros(1, 128, 1, 128, dtype=torch.uint8).view(torch.float8_e4m3fn)
    scales = torch.zeros(1, 128, 1, 1, dtype=torch.float32)
    # Exercise both settings: cann_9_1 9.1 ignores this normal-FP8 attribute;
    # released 9.2 honors it. The reference uses the same SDK input contract.
    kwargs = dict(quant_mode="fp8", round_scale=round_scale)
    return keys, scales, x, slots, kwargs


@pytest.mark.parametrize(
    "tokens,input_kind",
    [
        pytest.param(1, "random", id="deepseek-v4-flash-indexer-decode-t1-d128"),
        pytest.param(8, "random", id="deepseek-v4-flash-indexer-batch-t8-d128"),
        pytest.param(
            1,
            "scale-policy",
            id="synthetic-normal-fp8-scale-policy",
            marks=pytest.mark.extra_checks,
        ),
        pytest.param(
            1,
            "boundaries",
            id="synthetic-fp8-rne-boundaries",
            marks=pytest.mark.extra_checks,
        ),
        pytest.param(
            1, "zero", id="synthetic-zero-row", marks=pytest.mark.extra_checks
        ),
    ],
)
@pytest.mark.parametrize(
    "round_scale",
    [
        pytest.param(True, id="round-on"),
        pytest.param(False, id="synthetic-round-off", marks=pytest.mark.extra_checks),
    ],
)
def test_indexer_cache(tokens, input_kind, round_scale):
    from sgl_kernel_npu.operators.mem_cache import indexer_quant_cache

    impl, impl_ref = get_test_impls()
    # Validate hardware before allocating FP8 buffers on unsupported A3 devices.
    if get_skip_acc_check() is True:
        require_test_impl(indexer_quant_cache, impl)
    else:
        require_distinct_impls(indexer_quant_cache, impl, impl_ref)
    keys, scales, x, slots, kwargs = generate_data(tokens, input_kind, round_scale)
    executor = make_executor(
        indexer_quant_cache, args=(keys, scales, x, slots), kwargs=kwargs
    )
    if get_skip_acc_check() is not True:
        # Run independent reference math on CPU, where FP8 casts are available.
        # Copy cache bytes rather than requesting an NPU FP8 copy/conversion.
        with torch.device("cpu"):
            expected = indexer_quant_cache(
                keys.view(torch.uint8).cpu().clone().view(torch.float8_e4m3fn),
                scales.cpu().clone(),
                x.cpu(),
                slots.cpu(),
                impl=executor.impl_ref,
                **kwargs,
            )
        actual = executor()
        assert actual[0].dtype == expected[0].dtype == torch.float8_e4m3fn
        actual_scales = actual[1].cpu()
        # Check scales first: an incorrect scale must not be hidden behind a
        # later FP8 payload mismatch. Zero and boundary rows are exact cases.
        scale_standard = AccuracyStandard(atol=1e-5, rtol=0.03)
        if input_kind != "random":
            scale_standard = AccuracyStandard(atol=0, rtol=0)
        assert_accuracy(
            actual_scales,
            expected[1],
            {"torch.float32": scale_standard},
        )
        actual_bytes = actual[0].view(torch.uint8).cpu()
        if input_kind != "random":
            # Exactly representable edge inputs require exact FP8 bytes.
            assert_accuracy(actual_bytes, expected[0].view(torch.uint8))
        # Widen on CPU; retain the existing random-data and scale tolerances.
        assert_accuracy(
            actual[0].view(torch.uint8).cpu().view(torch.float8_e4m3fn).float(),
            expected[0].float(),
            {"torch.float32": AccuracyStandard(atol=16, rtol=0.06)},
        )
        # The skipped token and every unassigned slot must remain untouched.
        unused = torch.ones(128, dtype=torch.bool, device="cpu")
        used = slots.cpu()
        unused[used[used >= 0].long()] = False
        assert_accuracy(
            actual_bytes.reshape(128, 128)[unused],
            expected[0].view(torch.uint8).reshape(128, 128)[unused],
        )
        assert_accuracy(
            actual[1].cpu().reshape(128)[unused], expected[1].reshape(128)[unused]
        )
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=(tokens, 128) in perf_shape)

"""Exercise the weight preparation and cache hints used by SGLang on Ascend."""

import pytest

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("shape", [pytest.param((640, 5120), id="wan2.2-a14b-q-tp8")])
def test_dtype_cast(shape):
    import torch_npu
    from sgl_kernel_npu.operators.data_move import dtype_cast
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation is not Generation.ARCH35:
        pytest.skip("Wan2.2 dual-level MXFP4 weight packing requires Ascend 950")
    # Offline ModelSlim stores one FP4 weight per FP8 element before packing.
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cpu")
    values = torch.cat((magnitudes, -magnitudes))
    rng = torch.Generator(device="cpu").manual_seed(42)
    codes = torch.randint(16, shape, generator=rng, device="cpu")
    weight = values[codes].to(torch.float8_e4m3fn)
    weight = weight.view(torch.uint8).npu().view(torch.float8_e4m3fn)
    executor = check_accuracy(dtype_cast, (weight, torch_npu.float4_e2m1fn_x2))
    check_performance(executor)


@pytest.mark.parametrize(
    "shape", [pytest.param((1536, 1536), id="qwen2.5-1.5b-w8a8-o-proj")]
)
def test_format_cast_to_nz(shape):
    import torch_npu
    from sgl_kernel_npu.operators.data_move import format_cast
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    torch.npu.manual_seed(42)
    weight = torch.randint(-128, 128, shape, dtype=torch.int8).T.contiguous()
    executor = make_executor(format_cast, (weight, 29))
    if executor.impl == "torch_native":
        pytest.skip("The PyTorch reference exposes logical values in ND layout only")
    require_test_impl(format_cast, executor.impl)
    arch22 = get_hardware().generation is Generation.ARCH22
    was_enabled = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
    try:
        if arch22:
            torch_npu.npu.config.allow_internal_format = True
        if get_skip_acc_check() is not True:
            require_distinct_impls(format_cast, executor.impl, executor.impl_ref)
            result = executor()
            # INT8 weights use FRACTAL_NZ or the device's NZ C0_16/C0_32 format.
            assert torch_npu.get_npu_format(result) in (29, 50, 51)
            expected = format_cast(weight, 2, impl=executor.impl_ref)
            assert_accuracy(result.cpu(), expected.cpu())
            executor.accuracy_checked = True
        check_performance(executor)
    finally:
        if arch22:
            torch_npu.npu.config.allow_internal_format = was_enabled


@pytest.mark.parametrize(
    "weight_shape, hidden_shape",
    [pytest.param((3072, 4096), (8, 4096), id="qwen3-8b-prefill-gate-up-tp8")],
)
def test_prefetch_arch22(weight_shape, hidden_shape):
    from sgl_kernel_npu.operators.data_move import prefetch

    torch.npu.manual_seed(42)
    weight = torch.randn(weight_shape, dtype=torch.bfloat16)
    hidden_states = torch.randn(hidden_shape, dtype=torch.bfloat16)
    # Qwen3's FFN boundary submits one CMO per local weight, with this byte limit.
    executor = make_executor(prefetch, (weight, hidden_states, 1_000_000_000))
    require_test_impl(prefetch, executor.impl)
    if get_skip_acc_check() is not True:
        before = weight.clone()
        dependency_before = hidden_states.clone()
        assert executor() is None
        torch.npu.synchronize()
        exact = {"torch.bfloat16": AccuracyStandard(atol=0, rtol=0)}
        assert_accuracy(weight, before, exact)
        assert_accuracy(hidden_states, dependency_before, exact)
        executor.accuracy_checked = True
    check_performance(executor)

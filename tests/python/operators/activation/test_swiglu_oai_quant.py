"""Accuracy and selected performance checks through the public interface.

Production configurations: MiniMaxAI/MiniMax-M3, BF16, alpha=1.702, limit=7.0.
The official config sets intermediate_size=3072, n_shared_experts=1 and
dense_intermediate_size=12288:
https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/98e20d0a1c04395d0aec808f70cc4a9f18141d5a/config.json

SGLang MiniMaxM3MoE replicates its shared MLP under DeepEP (TP=1), giving
2*3072=6144 concatenated gate/up columns. MiniMaxM3MLP's dense path has
2*12288/8=3072 columns at TP=8, with dense FFN fully-DP disabled. Its NPU
activation calls swiglu_oai_quant with need_quant=False. Call sites and TP
partitioning are pinned here:
https://github.com/sgl-project/sglang/blob/53a24c448ecd6c26bf6461b80ffb8d55397d57cf/python/sglang/srt/models/minimax_m3.py
https://github.com/sgl-project/sglang/blob/53a24c448ecd6c26bf6461b80ffb8d55397d57cf/python/sglang/srt/layers/linear.py

The BF16 model cases use these actual call shapes and activation options;
token counts are representative decode/prefill workloads, not captured traces.
Dense INT8/FP16 cases and grouped buffers are additional regression scenarios.
The NPU DeepEP activation also uses this operator with optional quantization.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(1, 6144), (17, 3072), (128, 6144), (1024, 6144)]

# Use selected shape/dtype pairs instead of a Cartesian product. The default
# cases cover both output modes, decode/prefill scheduling, a second quantization
# tile, zero scales and clamping. FP32 and layout/empty edges are opt-in.
dense_cases = [
    pytest.param(
        (1, 6144), "random", torch.bfloat16, id="minimax-m3-shared-tp1-decode"
    ),
    pytest.param(
        (17, 3072), "random", torch.bfloat16, id="minimax-m3-dense-tp8-batched-decode"
    ),
    pytest.param((128, 6144), "random", torch.float16, id="prefill-two-tiles-fp16"),
    pytest.param(
        (1024, 6144), "random", torch.bfloat16, id="minimax-m3-shared-tp1-prefill"
    ),
    pytest.param((3, 3072), "zero", torch.bfloat16, id="zero-row-scale-bf16"),
    pytest.param((3, 3072), "clamp", torch.float16, id="gate-up-clamping-fp16"),
    # Supplementary dtype, layout and empty-input coverage.
    pytest.param(
        (1, 3072),
        "random",
        torch.float16,
        marks=pytest.mark.extra_checks,
        id="decode-fp16",
    ),
    pytest.param(
        (17, 3072),
        "random",
        torch.float32,
        marks=pytest.mark.extra_checks,
        id="fp32-reference",
    ),
    pytest.param(
        (2, 3, 3072),
        "strided",
        torch.bfloat16,
        marks=pytest.mark.extra_checks,
        id="3d-noncontiguous",
    ),
    pytest.param(
        (3072,),
        "random",
        torch.float16,
        marks=pytest.mark.extra_checks,
        id="single-vector",
    ),
    pytest.param(
        (0, 3072),
        "random",
        torch.float32,
        marks=pytest.mark.extra_checks,
        id="empty-tokens",
    ),
]

# Cover both group encodings and both output modes without multiplying every
# dtype/index-type combination. Partial buffers also include empty experts.
grouped_cases = [
    pytest.param(
        torch.bfloat16, True, 1, torch.int64, [0, 2, 0, 3, 0], id="count-quant-partial"
    ),
    pytest.param(
        torch.float16, True, 0, torch.int32, [0, 2, 0, 3, 0], id="cumsum-quant-partial"
    ),
    pytest.param(
        torch.bfloat16,
        False,
        1,
        torch.int64,
        [0, 2, 0, 3, 0],
        id="count-activation-partial",
    ),
    pytest.param(
        torch.float16,
        False,
        0,
        torch.int32,
        [0, 5, 0, 12, 0],
        id="cumsum-activation-full",
    ),
    pytest.param(
        torch.bfloat16, True, 1, torch.int64, [0, 5, 0, 12, 0], id="count-quant-full"
    ),
    pytest.param(
        torch.bfloat16, True, 1, torch.int64, [0, 0, 0, 0, 0], id="no-routed-tokens"
    ),
    # Supplementary index alignment and zero-token paths.
    pytest.param(
        torch.bfloat16,
        True,
        1,
        torch.int32,
        [0, 2, 0, 3, 0],
        marks=pytest.mark.extra_checks,
        id="count-int32",
    ),
    pytest.param(
        torch.float16,
        True,
        0,
        torch.int64,
        [0, 2, 0, 3, 0],
        marks=pytest.mark.extra_checks,
        id="cumsum-int64",
    ),
    pytest.param(
        torch.bfloat16,
        True,
        0,
        torch.int64,
        [0, 0, 0, 0, 0],
        marks=pytest.mark.extra_checks,
        id="cumsum-no-routed-tokens",
    ),
    pytest.param(
        torch.float16,
        False,
        1,
        torch.int32,
        [0, 0, 0, 0, 0],
        marks=pytest.mark.extra_checks,
        id="activation-no-routed-tokens",
    ),
]


# Borrow the INT8 max-error/MAE and scale budgets from
# tests/python/sgl_kernel_npu/test_swiglu_quant.py. That legacy test covers the
# related swiglu_quant operator; swiglu_oai_quant had no dedicated legacy test.
INT8_MAE_LIMIT = 2e-2
quant_standard = {
    "torch.int8": AccuracyStandard(atol=1, rtol=0),
    "torch.float32": AccuracyStandard(atol=0, rtol=5e-3),
}

# Analytic known values must remain exact, including floating-point outputs.
exact_standard = {
    str(dtype): AccuracyStandard(atol=0, rtol=0)
    for dtype in (torch.int8, torch.float16, torch.bfloat16, torch.float32)
}


def generate_test_data(shape, dtype, mode="random"):
    """Cover clamping, all-zero quantization, reshape and input layout."""
    torch.npu.manual_seed(42)
    if mode == "strided":
        return torch.randn(*shape[:-1], shape[-1] * 2, dtype=dtype)[..., ::2]
    x = torch.randn(shape, dtype=dtype)
    if mode == "zero":
        x.zero_()
    elif mode == "clamp":
        x.mul_(20)
    return x


@pytest.mark.parametrize("shape,mode,dtype", dense_cases)
def test_swiglu_oai_quant_dense(shape, mode, dtype):
    """Compare the complete INT8 output and scale tuple, then profile."""
    from sgl_kernel_npu.operators.activation import swiglu_oai_quant

    x = generate_test_data(shape, dtype, mode)
    args = (x, 1.702, 7.0)
    executor = check_accuracy(swiglu_oai_quant, args=args, standard=quant_standard)
    if get_skip_acc_check() is not True and x.numel():
        # check_accuracy checks elementwise bounds. Also retain the mean-error
        # budget; subtract in FP32 so signed INT8 differences cannot wrap.
        output, _ = executor()
        reference, _ = swiglu_oai_quant(*args, impl=executor.impl_ref)
        mae = (output.float() - reference.float()).abs().mean().item()
        assert mae < INT8_MAE_LIMIT, (
            f"INT8 mean absolute error {mae:.6f} " f"must be below {INT8_MAE_LIMIT}"
        )
    if x.numel():
        check_performance(
            executor,
            enable_perf=shape in perf_shape
            and mode == "random"
            and dtype != torch.float32,
        )


@pytest.mark.parametrize("shape,mode,dtype", dense_cases)
def test_swiglu_oai_quant_without_quantization(shape, mode, dtype):
    """Compare activation only: scale is undefined when need_quant=False."""
    from sgl_kernel_npu.operators.activation import swiglu_oai_quant

    x = generate_test_data(shape, dtype, mode)
    args = (x, 1.702, 7.0, False)
    executor = make_executor(swiglu_oai_quant, args=args)
    if get_skip_acc_check() is True:
        require_test_impl(swiglu_oai_quant, executor.impl)
    else:
        require_distinct_impls(swiglu_oai_quant, executor.impl, executor.impl_ref)
        output, _ = executor()
        reference, _ = swiglu_oai_quant(*args, impl=executor.impl_ref)
        assert_accuracy(output, reference)
        executor.accuracy_checked = True
    if x.numel():
        check_performance(
            executor,
            enable_perf=shape in perf_shape
            and mode == "random"
            and dtype != torch.float32,
        )


@pytest.mark.parametrize(
    "dtype,need_quant,group_list_type,group_dtype,counts", grouped_cases
)
def test_swiglu_oai_quant_grouped(
    dtype, need_quant, group_list_type, group_dtype, counts
):
    """Compare only routed rows, covering count/cumsum and empty experts."""
    from sgl_kernel_npu.operators.activation import swiglu_oai_quant

    x = generate_test_data((17, 3072), dtype)
    valid_rows = sum(counts)
    # Unused buffer rows must never participate in the reference calculation.
    x[valid_rows:] = float("nan")
    groups = torch.tensor(counts, dtype=group_dtype)
    if group_list_type == 0:
        groups = groups.cumsum(0, dtype=group_dtype)
    args = (x, 1.702, 7.0, need_quant, groups, group_list_type)
    executor = make_executor(swiglu_oai_quant, args=args)
    if get_skip_acc_check() is True:
        require_test_impl(swiglu_oai_quant, executor.impl)
    else:
        require_distinct_impls(swiglu_oai_quant, executor.impl, executor.impl_ref)
        output, scale = executor()
        reference, scale_ref = swiglu_oai_quant(*args, impl=executor.impl_ref)
        # Both buffers leave rows after valid_rows undefined. Without
        # quantization the entire scale buffer is undefined as well.
        output, reference = output[:valid_rows], reference[:valid_rows]
        if need_quant:
            assert_accuracy(
                (output, scale[:valid_rows]),
                (reference, scale_ref[:valid_rows]),
                standard=quant_standard,
            )
            if valid_rows:
                mae = (output.float() - reference.float()).abs().mean().item()
                assert mae < INT8_MAE_LIMIT, (
                    f"INT8 mean absolute error {mae:.6f} "
                    f"must be below {INT8_MAE_LIMIT}"
                )
        else:
            assert_accuracy(output, reference)
        executor.accuracy_checked = True
    if valid_rows:
        check_performance(executor, enable_perf=valid_rows == 17)


@pytest.mark.parametrize(
    "dtype,need_quant",
    [
        pytest.param(torch.float16, True, id="signed-truncation-fp16"),
        pytest.param(torch.bfloat16, True, id="signed-truncation-bf16"),
        pytest.param(
            torch.float32,
            True,
            marks=pytest.mark.extra_checks,
            id="signed-truncation-fp32",
        ),
        pytest.param(
            torch.float32,
            False,
            marks=pytest.mark.extra_checks,
            id="split-layout-activation",
        ),
    ],
)
def test_swiglu_oai_quant_known_values(dtype, need_quant):
    """Use an exact alpha=0 oracle for split layout and signed truncation."""
    if get_skip_acc_check() is True:
        pytest.skip("Known-value accuracy check disabled by --skip-acc-check=true")

    from sgl_kernel_npu.operators.activation import swiglu_oai_quant

    impl, _ = get_test_impls()
    require_test_impl(swiglu_oai_quant, impl)
    x = torch.zeros((1, 3072), dtype=dtype)
    # sigmoid(0)=1/2, scale=1. Exact signed fractional values detect a
    # round-to-nearest cast or an accidental adjacent gate/up layout.
    x[0, :8] = torch.tensor([254, -254, 1, -1, 3, -3, 5, -5], dtype=dtype)
    output, scale = swiglu_oai_quant(x, 0.0, 1000.0, need_quant, impl=impl)
    expected = torch.zeros_like(output)
    expected[0, :8] = torch.tensor(
        (
            [127, -127, 0, 0, 1, -1, 2, -2]
            if need_quant
            else [127, -127, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5]
        ),
        dtype=output.dtype,
    )
    assert_accuracy(output, expected, standard=exact_standard)
    if need_quant:
        assert_accuracy(scale, torch.ones_like(scale), standard=exact_standard)

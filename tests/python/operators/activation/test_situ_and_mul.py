"""SiTU accuracy and selected NPU performance checks through the public API.

Production dimensions: moonshotai/Kimi-K3 config.json at c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721
sets intermediate_size=33792, moe_intermediate_size=3072, num_shared_experts=2,
and SiTU bounds 4/25. SGLang KimiK3MLP packs gate/up projection halves before
SituAndMul (53a24c448ecd6c26bf6461b80ffb8d55397d57cf):
https://huggingface.co/moonshotai/Kimi-K3/blob/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/config.json
https://github.com/sgl-project/sglang/blob/53a24c448ecd6c26bf6461b80ffb8d55397d57cf/python/sglang/srt/models/kimi_k3.py#L290-L344
At TP1 the first dense MLP has d=33792 and shared experts have d=2*3072=6144;
shared-expert TP2 gives d=3072. Inputs have width 2*d. Token counts, group counts
and local expert counts below are synthetic, not captured production traces.
The default suite has 28 accuracy cases and five BF16 performance selections.
Supplementary combinations and the row-limit check use extra_checks.
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

# Input shapes [..., gate | up]; output width is half the final dimension.
perf_shape = [(1, 6144), (128, 12288), (128, 67584)]
grouped_perf_shape = [(128, 6144), (512, 12288)]


def generate_data(shape, dtype, mode="random", layout="contiguous"):
    """Create signed gates/ups, saturation inputs, or exact-zero activations."""
    torch.npu.manual_seed(42)
    if layout == "padded":
        x = torch.randn((*shape[:-1], shape[-1] + 16), dtype=dtype)[..., : shape[-1]]
    elif layout == "strided":
        x = torch.randn((*shape[:-1], shape[-1] * 2), dtype=dtype)[..., ::2]
    else:
        x = torch.randn(shape, dtype=dtype)
    if mode == "saturation":
        # Cover both signs, near-zero gates and saturated gate/up paths.
        values = torch.tensor(
            [-1000, -100, -25, -4, -1e-3, 0, 1e-3, 4, 25, 100, 1000],
            dtype=dtype,
        )
        indices = torch.arange(x.numel()) % values.numel()
        x.copy_(values[indices].reshape(shape))
    elif mode == "zero":
        x.zero_()
    elif mode != "random":
        raise ValueError(f"Unknown mode: {mode}")
    return x


# Main workloads cover all three dtypes; other rows cover one distinct branch.
@pytest.mark.parametrize(
    "shape,dtype,beta,linear_beta,mode,layout",
    [
        ((1, 6144), torch.float32, 4.0, 25.0, "random", "contiguous"),
        ((1, 6144), torch.float16, 4.0, 25.0, "random", "contiguous"),
        ((1, 6144), torch.bfloat16, 4.0, 25.0, "random", "contiguous"),
        ((128, 12288), torch.float32, 4.0, 25.0, "random", "contiguous"),
        ((128, 12288), torch.float16, 4.0, 25.0, "random", "contiguous"),
        ((128, 12288), torch.bfloat16, 4.0, 25.0, "random", "contiguous"),
        ((128, 67584), torch.float32, 4.0, 25.0, "random", "contiguous"),
        ((128, 67584), torch.float16, 4.0, 25.0, "random", "contiguous"),
        ((128, 67584), torch.bfloat16, 4.0, 25.0, "random", "contiguous"),
        ((2,), torch.float32, 4.0, 25.0, "random", "contiguous"),
        ((7, 66), torch.float32, 2.5, 7.0, "random", "contiguous"),
        ((7, 66), torch.bfloat16, 1.0, None, "random", "contiguous"),
        ((5, 2050), torch.float32, 4.0, 25.0, "random", "contiguous"),
        ((3, 16386), torch.bfloat16, 4.0, 25.0, "random", "contiguous"),
        ((2, 3, 128), torch.bfloat16, 4.0, 25.0, "random", "strided"),
        ((7, 6144), torch.bfloat16, 4.0, 25.0, "random", "padded"),
        ((5, 130), torch.float32, 4.0, 25.0, "saturation", "contiguous"),
        ((3, 62), torch.bfloat16, 4.0, 25.0, "zero", "contiguous"),
        ((0, 128), torch.float32, 4.0, 25.0, "random", "contiguous"),
        # Supplementary combinations; excluded by default.
        pytest.param(
            (128, 67584),
            torch.bfloat16,
            1.0,
            None,
            "random",
            "contiguous",
            marks=pytest.mark.extra_checks,
        ),
        pytest.param(
            (3, 2048),
            torch.float32,
            4.0,
            25.0,
            "random",
            "contiguous",
            marks=pytest.mark.extra_checks,
        ),
        pytest.param(
            (5, 130),
            torch.bfloat16,
            4.0,
            25.0,
            "saturation",
            "contiguous",
            marks=pytest.mark.extra_checks,
        ),
    ],
    ids=[
        "kimi-k3-shared-tp2-decode-fp32",
        "kimi-k3-shared-tp2-decode-fp16",
        "kimi-k3-shared-tp2-decode-bf16",
        "kimi-k3-shared-tp1-batch-fp32",
        "kimi-k3-shared-tp1-batch-fp16",
        "kimi-k3-shared-tp1-batch-bf16",
        "kimi-k3-dense-tp1-batch-fp32",
        "kimi-k3-dense-tp1-batch-fp16",
        "kimi-k3-dense-tp1-batch-bf16",
        "minimum-gate-up-pair",
        "nondefault-bounds-tail33",
        "unbounded-up-tail33",
        "tile1024-plus-one",
        "all-tile-configs-tail8193",
        "nd-strided-features",
        "padded-input-rows",
        "saturation-near-zero-fp32",
        "zero-activation",
        "empty-tokens",
        "large-width-unbounded-up",
        "exact-tile1024",
        "saturation-bf16",
    ],
)
def test_situ_and_mul(shape, dtype, beta, linear_beta, mode, layout):
    from sgl_kernel_npu.operators.activation import situ_and_mul

    x = generate_data(shape, dtype, mode, layout)
    executor = check_accuracy(
        situ_and_mul, args=(x,), kwargs={"beta": beta, "linear_beta": linear_beta}
    )
    if x.numel() == 0:
        return  # Empty outputs have no device task to profile.
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape
            and dtype == torch.bfloat16
            and (beta, linear_beta) == (4.0, 25.0)
            and mode == "random"
            and layout == "contiguous"
        ),
    )


def generate_grouped_data(shape, dtype, group_dtype, group_type, experts, valid_rows):
    """Distribute the prefix over experts, including zero-token experts."""
    x = generate_data(shape, dtype)
    counts = [0] * experts
    # The distribution includes empty experts in some scenarios, exercising
    # masked reduction and repeated cumulative counts.
    for row in range(valid_rows):
        counts[(row * 2) % experts] += 1
    if group_type == 0:
        cumulative = 0
        for index, count in enumerate(counts):
            cumulative += count
            counts[index] = cumulative
    group_list = torch.tensor(counts, dtype=group_dtype, device=x.device)
    return x, group_list


def check_grouped_accuracy(operator, args, valid_rows):
    """Compare only initialized rows; profile the original public call.

    Padding is unspecified in both implementations, so the shared full-output
    check_accuracy would incorrectly compare uninitialized allocation contents.
    Honor the same CLI implementation and accuracy switches as check_accuracy.
    """
    executor = make_executor(operator, args=args)
    if get_skip_acc_check():
        require_test_impl(operator, executor.impl)
        return executor
    require_distinct_impls(operator, executor.impl, executor.impl_ref)
    reference = operator(*args, impl=executor.impl_ref)
    result = executor()
    width = args[0].shape[-1] // 2
    assert result.shape == reference.shape
    assert_accuracy(
        result.reshape(-1, width)[:valid_rows],
        reference.reshape(-1, width)[:valid_rows],
    )
    executor.accuracy_checked = True
    return executor


# Cover both encodings and integer widths, reduction tails and valid prefixes.
@pytest.mark.parametrize(
    "shape,dtype,group_dtype,group_type,beta,linear_beta,experts,valid_rows",
    [
        ((128, 6144), torch.float16, torch.int32, 1, 4.0, 25.0, 8, 97),
        ((128, 6144), torch.bfloat16, torch.int64, 1, 4.0, 25.0, 8, 97),
        ((512, 12288), torch.float32, torch.int64, 0, 4.0, 25.0, 17, 385),
        ((512, 12288), torch.bfloat16, torch.int64, 1, 4.0, 25.0, 17, 385),
        ((17, 67584), torch.bfloat16, torch.int64, 0, 4.0, 25.0, 33, 13),
        ((2, 3, 66), torch.float32, torch.int32, 0, 4.0, 25.0, 5, 4),
        ((2, 3, 66), torch.bfloat16, torch.int32, 1, 1.0, None, 5, 4),
        ((3, 2), torch.float16, torch.int64, 0, 4.0, 25.0, 1, 3),
        ((7, 130), torch.bfloat16, torch.int32, 1, 4.0, 25.0, 5, 0),
        # Supplementary combinations; excluded by default.
        pytest.param(
            (512, 12288),
            torch.float32,
            torch.int32,
            1,
            4.0,
            25.0,
            17,
            385,
            marks=pytest.mark.extra_checks,
        ),
        pytest.param(
            (17, 67584),
            torch.bfloat16,
            torch.int64,
            1,
            1.0,
            None,
            33,
            13,
            marks=pytest.mark.extra_checks,
        ),
    ],
    ids=[
        "routed-counts-i32-fp16",
        "routed-counts-i64-bf16",
        "cumulative-i64-experts17",
        "counts-i64-experts17",
        "cumulative-final-entry-experts33",
        "nd-cumulative-i32",
        "nd-counts-unbounded-up",
        "single-expert-full-prefix",
        "zero-valid-tokens",
        "counts-i32-experts17",
        "large-counts-unbounded-up",
    ],
)
def test_situ_and_mul_grouped(
    shape, dtype, group_dtype, group_type, beta, linear_beta, experts, valid_rows
):
    from sgl_kernel_npu.operators.activation import situ_and_mul

    x, group_list = generate_grouped_data(
        shape, dtype, group_dtype, group_type, experts, valid_rows
    )
    # Invalid/padded rows must never influence the valid activation prefix.
    x.reshape(-1, shape[-1])[valid_rows:] = float("nan")
    executor = check_grouped_accuracy(
        situ_and_mul, (x, group_list, group_type, beta, linear_beta), valid_rows
    )
    if valid_rows == 0:
        return
    check_performance(
        executor,
        enable_perf=(
            shape in grouped_perf_shape
            and dtype == torch.bfloat16
            and group_dtype == torch.int64
            and group_type == 1
            and (beta, linear_beta) == (4.0, 25.0)
        ),
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "shape", [pytest.param((1 << 31, 2), id="int32-row-overflow-before-allocation")]
)
def test_situ_and_mul_rejects_row_overflow(shape):
    """Use shape-only storage to exercise rejection without allocating GiBs."""
    from sgl_kernel_npu.operators.activation import situ_and_mul

    impl, _ = get_test_impls()
    require_test_impl(situ_and_mul, impl)
    if impl not in ("auto", "triton"):
        pytest.skip("The row scheduling limit belongs to the Triton backend")
    x = torch.empty(shape, dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match="int32 scheduling limit"):
        situ_and_mul(x, impl=impl)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "group_type",
    [pytest.param(0, id="empty-cumulative"), pytest.param(1, id="empty-counts")],
)
def test_situ_and_mul_rejects_empty_group(group_type):
    """Reject invalid expert metadata before either backend can read it."""
    from sgl_kernel_npu.operators.activation import situ_and_mul

    impl, _ = get_test_impls()
    require_test_impl(situ_and_mul, impl)
    x = torch.empty((1, 2), dtype=torch.bfloat16)
    group_list = torch.empty((0,), dtype=torch.int64)
    with pytest.raises(ValueError, match="group_list must not be empty"):
        situ_and_mul(x, group_list, group_type, impl=impl)

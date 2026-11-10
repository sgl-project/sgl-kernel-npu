"""Public-interface tests for LoRA sgemmc_expand.

Unlike sgemmv_expand the activation dtype follows y, so x, weight and y all
share float16/bfloat16. The legacy Kernel feeds the rank directly into the cube
matmul (no rank-8 padding), so cases use ranks of at least 16. The operator
accumulates the B-projection into the caller's y in place; accuracy is checked
on separate y copies. Cases cover the legacy single-slice layout, fused
multi-slice layouts, rank-0 / negative-index skips, zero-length segments and a
trailing gap that must stay untouched.
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

DTYPES = [torch.float16, torch.bfloat16]
STANDARD = {
    "torch.float16": AccuracyStandard(atol=1e-2, rtol=1e-2),
    "torch.bfloat16": AccuracyStandard(atol=5e-2, rtol=5e-2),
}
INPUT_NAMES = (
    "x",
    "weight",
    "lora_indices",
    "seq_len",
    "lora_ranks",
    "slice_offsets",
)

MAIN_CASES = [
    pytest.param(
        {
            "name": "single-slice-all-ranks",
            "seq_lengths": [2, 3, 1, 4],
            "lora_indices": [0, 1, 2, 0],
            "ranks": [16, 32, 64],
            "out_widths": [1024],
            "max_rank": 64,
        },
        id="single-slice-all-ranks",
    ),
    pytest.param(
        {
            "name": "qkv-three-slices",
            "seq_lengths": [2, 3],
            "lora_indices": [0, 1],
            "ranks": [16, 64],
            "out_widths": [512, 1024, 2048],
            "max_rank": 64,
        },
        id="qkv-three-slices",
    ),
    pytest.param(
        {
            "name": "skips-and-rank0",
            "seq_lengths": [1, 2, 1, 3, 2],
            "lora_indices": [0, 1, 2, -1, 3],
            "ranks": [16, 64, 0, 32],
            "out_widths": [1024],
            "max_rank": 64,
        },
        id="skips-and-rank0",
    ),
    pytest.param(
        {
            "name": "model-scale-all-ranks",
            "seq_lengths": [16] * 8,
            "lora_indices": [0, 1, 2, 0, 1, 2, 0, 1],
            "ranks": [16, 32, 64],
            "out_widths": [4096],
            "max_rank": 64,
        },
        id="model-scale-all-ranks",
    ),
    pytest.param(
        {
            "name": "zero-length-segments",
            "seq_lengths": [0, 2, 0, 1],
            "lora_indices": [0, 1, 2, 0],
            "ranks": [16, 32, 64],
            "out_widths": [1024],
            "max_rank": 64,
        },
        id="zero-length-segments",
    ),
    pytest.param(
        {
            "name": "trailing-gap-untouched",
            "seq_lengths": [2, 2],
            "lora_indices": [0, 1],
            "ranks": [16, 32],
            "out_widths": [1024, 1024],
            "max_rank": 64,
            "trailing_gap": 2048,
        },
        id="trailing-gap-untouched",
    ),
]

EXTRA_CASES = [
    pytest.param(
        {
            "name": "narrow-out-width",
            "seq_lengths": [3],
            "lora_indices": [0],
            "ranks": [16],
            "out_widths": [16],
            "max_rank": 16,
        },
        id="narrow-out-width",
    ),
    pytest.param(
        {
            "name": "rank16-packed-in-rank64",
            "seq_lengths": [4, 2],
            "lora_indices": [0, 0],
            "ranks": [16],
            "out_widths": [2048, 2048],
            "max_rank": 64,
        },
        id="rank16-packed-in-rank64",
    ),
]


def generate_case(case, dtype, seed=20261106):
    """Build one CPU dataset, then move every tensor to the current NPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = sum(case["seq_lengths"])
    slice_count = len(case["out_widths"])
    max_rank = case["max_rank"]
    covered_width = sum(case["out_widths"])
    out_width = covered_width + case.get("trailing_gap", 0)
    num_loras = len(case["ranks"])

    x = torch.randn(
        tokens,
        slice_count * max_rank,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(dtype)
    weight = (
        torch.randn(
            num_loras,
            out_width,
            max_rank,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        * 0.02
    )
    # Columns past each adapter's rank must never be read.
    for slot, rank in enumerate(case["ranks"]):
        if rank < max_rank:
            weight[slot, :, rank:] = 17.0
    weight = weight.to(dtype)
    y = (
        torch.randn(
            tokens, out_width, generator=generator, dtype=torch.float32, device="cpu"
        )
        * 0.1
    ).to(dtype)

    offsets = [0]
    for width in case["out_widths"]:
        offsets.append(offsets[-1] + width)

    args = (
        x.npu(),
        weight.npu(),
        torch.tensor(case["lora_indices"], dtype=torch.int32, device="cpu").npu(),
        torch.tensor(case["seq_lengths"], dtype=torch.int32, device="cpu").npu(),
        torch.tensor(case["ranks"], dtype=torch.int32, device="cpu").npu(),
        torch.tensor(offsets, dtype=torch.int32, device="cpu").npu(),
    )
    return args, y.npu()


def written_mask(case):
    """CPU mask of the y entries the Kernel accumulates into for this case."""
    covered_width = sum(case["out_widths"])
    out_width = covered_width + case.get("trailing_gap", 0)
    mask = torch.zeros(
        sum(case["seq_lengths"]), out_width, dtype=torch.bool, device="cpu"
    )
    token_offset = 0
    for length, lora_index in zip(case["seq_lengths"], case["lora_indices"]):
        if length > 0 and lora_index >= 0 and case["ranks"][lora_index] > 0:
            mask[token_offset : token_offset + length, :covered_width] = True
        token_offset += length
    return mask


def run_case(case, dtype, *, enable_perf=False):
    from sgl_kernel_npu.operators.lora import sgemmc_expand

    args, y = generate_case(case, dtype)
    originals = tuple(tensor.clone() for tensor in args)

    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(sgemmc_expand, impl)
    else:
        require_distinct_impls(sgemmc_expand, impl, impl_ref)
        result_ref = sgemmc_expand(*args, y.clone(), impl=impl_ref)

    y_dut = y.clone()
    result = sgemmc_expand(*args, y_dut, impl=impl)
    if not skip_accuracy:
        assert_accuracy(result, result_ref, standard=STANDARD)

    # In-place contract: the returned tensor aliases the caller's y.
    assert result.data_ptr() == y_dut.data_ptr()
    for actual, expected, name in zip(args, originals, INPUT_NAMES):
        assert torch.equal(actual, expected), "{} was modified".format(name)
    # Skipped blocks and a trailing gap keep their original values.
    mask = written_mask(case).to(result.device)
    assert torch.equal(result[~mask], y[~mask]), "untouched y entries changed"

    executor = make_executor(
        sgemmc_expand,
        args=(*args, y.clone()),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    check_performance(executor, enable_perf=enable_perf)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", MAIN_CASES)
def test_sgemmc_expand(case, dtype):
    run_case(
        case,
        dtype,
        enable_perf=(
            dtype == torch.float16 and case["name"] == "model-scale-all-ranks"
        ),
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", EXTRA_CASES)
def test_sgemmc_expand_extra_shapes(case, dtype):
    run_case(case, dtype)


def test_sgemmc_expand_rejects_zero_batch():
    from sgl_kernel_npu.operators.lora import sgemmc_expand

    impl, _ = get_test_impls()
    require_test_impl(sgemmc_expand, impl)

    with pytest.raises(RuntimeError):
        sgemmc_expand(
            torch.empty((0, 8), dtype=torch.float16, device="npu"),
            torch.empty((1, 4, 8), dtype=torch.float16, device="npu"),
            torch.empty((0,), dtype=torch.int32, device="npu"),
            torch.empty((0,), dtype=torch.int32, device="npu"),
            torch.tensor([8], dtype=torch.int32, device="npu"),
            torch.tensor([0, 4], dtype=torch.int32, device="npu"),
            torch.empty((0, 4), dtype=torch.float16, device="npu"),
            impl=impl,
        )

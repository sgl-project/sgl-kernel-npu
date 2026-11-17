"""Public-interface tests for LoRA sgemmv_expand.

The operator accumulates LoRA expand output into y in place. Accuracy is
checked by running the selected implementation and the reference on separate
y copies, then comparing the two mutated outputs. The shapes follow the
contract used by SGLang's ascend_backend: x is float32, weight and y share
float16/bfloat16, and each sequence block selects a LoRA adapter through
lora_indices. Model-scale widths come from SGLang LoRA projection shapes;
token counts and sequence splits are synthetic scenarios, not traced data.
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
    "torch.float16": AccuracyStandard(atol=1e-3, rtol=1e-3),
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

# Ranks cover every Kernel branch (8/16/32/64) and the rank-0 skip path.
# Widths 1024/4096 and the multi-slice splits match shapes exercised by the
# legacy LoRA regression tests, so the migrated Kernel is compared on the same
# branch coverage rather than only on new shapes.
MAIN_CASES = [
    pytest.param(
        {
            "name": "single-slice-all-ranks",
            "seq_lengths": [2, 3, 1, 4],
            "lora_indices": [0, 1, 2, 3],
            "ranks": [8, 16, 32, 64],
            "out_widths": [4096],
            "max_rank": 64,
        },
        id="single-slice-all-ranks",
    ),
    pytest.param(
        {
            "name": "multi-slice-with-skips",
            "seq_lengths": [1, 2, 1, 3, 2],
            "lora_indices": [0, 1, 2, 3, -1],
            "ranks": [16, 64, 0, 8],
            "out_widths": [2048, 1024, 1024],
            "max_rank": 64,
        },
        id="multi-slice-with-skips",
    ),
    pytest.param(
        {
            "name": "model-scale-all-ranks",
            "seq_lengths": [16] * 8,
            "lora_indices": [0, 1, 2, 3, 0, 1, 2, 3],
            "ranks": [8, 16, 32, 64],
            "out_widths": [4096],
            "max_rank": 64,
        },
        id="model-scale-all-ranks",
    ),
    pytest.param(
        {
            "name": "rank8-padded-to-64",
            "seq_lengths": [4, 4],
            "lora_indices": [0, 0],
            "ranks": [8],
            "out_widths": [1024],
            "max_rank": 64,
        },
        id="rank8-padded-to-64",
    ),
    pytest.param(
        {
            "name": "wide-two-slice",
            "seq_lengths": [3, 3],
            "lora_indices": [0, 1],
            "ranks": [8, 64],
            "out_widths": [3072, 3072],
            "max_rank": 64,
        },
        id="wide-two-slice",
    ),
    pytest.param(
        {
            "name": "zero-length-segments",
            "seq_lengths": [0, 2, 0, 1],
            "lora_indices": [0, 1, 2, 3],
            "ranks": [8, 16, 32, 16],
            "out_widths": [1024],
            "max_rank": 32,
        },
        id="zero-length-segments",
    ),
]

# Tail widths are supplementary and stay inside the legacy Kernel's 32-byte
# DataCopy alignment contract: for half/bfloat16 each slice's output width must
# be a multiple of 16 elements. Widths 1 / 4095 / 4097 are outside that
# contract because DataCopy silently drops the unaligned tail.
EXTRA_CASES = [
    pytest.param(
        {
            "name": "width-16",
            "seq_lengths": [2],
            "lora_indices": [0],
            "ranks": [8],
            "out_widths": [16],
            "max_rank": 64,
        },
        id="width-16",
    ),
    pytest.param(
        {
            "name": "width-4080",
            "seq_lengths": [2],
            "lora_indices": [0],
            "ranks": [16],
            "out_widths": [4080],
            "max_rank": 16,
        },
        id="width-4080",
    ),
    pytest.param(
        {
            "name": "width-4112",
            "seq_lengths": [2],
            "lora_indices": [0],
            "ranks": [64],
            "out_widths": [4112],
            "max_rank": 64,
        },
        id="width-4112",
    ),
]


def generate_case(case, dtype, seed=20261102):
    """Build one CPU dataset, then move every tensor to the current NPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = sum(case["seq_lengths"])
    slice_count = len(case["out_widths"])
    max_rank = case["max_rank"]
    output_dim = sum(case["out_widths"])
    num_loras = len(case["ranks"])

    # The public test fixture sets the default device to NPU; pin CPU here so the
    # CPU generator and the tensor it fills always agree.
    x = torch.randn(
        tokens,
        slice_count * max_rank,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    weight = (
        torch.randn(
            num_loras,
            output_dim,
            max_rank,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        * 0.02
    )
    # Dirty columns past each adapter's rank must never affect the result.
    for slot, rank in enumerate(case["ranks"]):
        if rank < max_rank:
            weight[slot, :, rank:] = 17.0
    weight = weight.to(dtype)
    y = (
        torch.randn(
            tokens,
            output_dim,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
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


def run_case(case, dtype, *, enable_perf=False):
    from sgl_kernel_npu.operators.lora import sgemmv_expand

    args, y = generate_case(case, dtype)
    originals = tuple(tensor.clone() for tensor in args)

    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(sgemmv_expand, impl)
    else:
        require_distinct_impls(sgemmv_expand, impl, impl_ref)
        result_ref = sgemmv_expand(*args, y.clone(), impl=impl_ref)

    y_dut = y.clone()
    result = sgemmv_expand(*args, y_dut, impl=impl)
    if not skip_accuracy:
        assert_accuracy(result, result_ref, standard=STANDARD)

    # In-place contract: the returned tensor aliases the caller's y.
    assert result.data_ptr() == y_dut.data_ptr()
    # Every non-output input must be byte-identical after the call.
    for actual, expected, name in zip(args, originals, INPUT_NAMES):
        assert torch.equal(actual, expected), "{} was modified".format(name)

    # Reuse the already-populated arguments for profiling; values are not
    # re-checked, but input mutation and aliasing were verified above.
    executor = make_executor(
        sgemmv_expand,
        args=(*args, y.clone()),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    check_performance(executor, enable_perf=enable_perf)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", MAIN_CASES)
def test_sgemmv_expand(case, dtype):
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
def test_sgemmv_expand_tail_widths(case, dtype):
    run_case(case, dtype)


def test_sgemmv_expand_rejects_zero_batch():
    from sgl_kernel_npu.operators.lora import sgemmv_expand

    impl, _ = get_test_impls()
    require_test_impl(sgemmv_expand, impl)

    with pytest.raises(RuntimeError):
        sgemmv_expand(
            torch.empty((0, 8), dtype=torch.float32, device="npu"),
            torch.empty((1, 4, 8), dtype=torch.float16, device="npu"),
            torch.tensor([0], dtype=torch.int32, device="npu"),
            torch.empty((0,), dtype=torch.int32, device="npu"),
            torch.tensor([8], dtype=torch.int32, device="npu"),
            torch.tensor([0, 4], dtype=torch.int32, device="npu"),
            torch.empty((0, 4), dtype=torch.float16, device="npu"),
            impl=impl,
        )

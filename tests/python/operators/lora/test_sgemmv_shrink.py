"""Public-interface tests for LoRA sgemmv_shrink.

The operator writes the LoRA A-projection into the caller's float32 y buffer in
place, so accuracy is checked by running the selected implementation and the
reference on separate y copies and comparing the two mutated buffers. The
first case mirrors the legacy tests/python/sgl_kernel_npu/test_lora_kernels.py
sgemmv_shrink shape; the remaining cases add sequence splits, multi-slice
projections, rank-0 / negative-index skips and the model-scale widths used by
SGLang's ascend_backend.
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
    "torch.float32": AccuracyStandard(atol=1e-3, rtol=1e-3),
}
INPUT_NAMES = (
    "x",
    "weight",
    "lora_indices",
    "seq_len",
    "lora_ranks",
    "lora_scales",
)

MAIN_CASES = [
    pytest.param(
        {
            "name": "legacy-shape",
            "seq_lengths": [1, 1],
            "lora_indices": [0, 2],
            "ranks": [8, 16, 32],
            "scales": [0.25, 1.0, 4.0],
            "hidden_in": 2048,
            "max_rank": 32,
            "slices": 1,
        },
        id="legacy-shape",
    ),
    pytest.param(
        {
            "name": "all-ranks-sequence-splits",
            "seq_lengths": [2, 3, 1, 4],
            "lora_indices": [0, 1, 2, 3],
            "ranks": [8, 16, 32, 64],
            "scales": [0.5, 1.0, 2.0, 4.0],
            "hidden_in": 1024,
            "max_rank": 64,
            "slices": 1,
        },
        id="all-ranks-sequence-splits",
    ),
    pytest.param(
        {
            "name": "skips-and-rank0",
            "seq_lengths": [1, 2, 1, 3, 2],
            "lora_indices": [0, 1, 2, -1, 3],
            "ranks": [16, 64, 0, 8],
            "scales": [1.0, 0.5, 4.0, 2.0],
            "hidden_in": 2048,
            "max_rank": 64,
            "slices": 1,
        },
        id="skips-and-rank0",
    ),
    pytest.param(
        {
            "name": "qkv-three-slices",
            "seq_lengths": [2, 3],
            "lora_indices": [0, 1],
            "ranks": [8, 64],
            "scales": [1.0, 0.5],
            "hidden_in": 4096,
            "max_rank": 64,
            "slices": 3,
        },
        id="qkv-three-slices",
    ),
    pytest.param(
        {
            "name": "model-scale-all-ranks",
            "seq_lengths": [16] * 8,
            "lora_indices": [0, 1, 2, 3, 0, 1, 2, 3],
            "ranks": [8, 16, 32, 64],
            "scales": [0.25, 0.5, 2.0, 4.0],
            "hidden_in": 4096,
            "max_rank": 64,
            "slices": 1,
        },
        id="model-scale-all-ranks",
    ),
    pytest.param(
        {
            "name": "zero-length-segments",
            "seq_lengths": [0, 2, 0, 1],
            "lora_indices": [0, 1, 2, 3],
            "ranks": [8, 16, 32, 64],
            "scales": [1.0, 2.0, 0.5, 0.25],
            "hidden_in": 2048,
            "max_rank": 64,
            "slices": 1,
        },
        id="zero-length-segments",
    ),
]

# The Kernel walks the hidden dimension in TILE_LENGTH (11776) chunks and has a
# dedicated tail path, so both sides of that boundary are covered here.
EXTRA_CASES = [
    pytest.param(
        {
            "name": "hidden-below-tile",
            "seq_lengths": [3],
            "lora_indices": [0],
            "ranks": [16],
            "scales": [1.0],
            "hidden_in": 11712,
            "max_rank": 16,
            "slices": 1,
        },
        id="hidden-below-tile",
    ),
    pytest.param(
        {
            "name": "hidden-above-tile",
            "seq_lengths": [2],
            "lora_indices": [0],
            "ranks": [8],
            "scales": [2.0],
            "hidden_in": 12032,
            "max_rank": 16,
            "slices": 1,
        },
        id="hidden-above-tile",
    ),
    pytest.param(
        {
            "name": "narrow-hidden",
            "seq_lengths": [4, 2],
            "lora_indices": [0, 1],
            "ranks": [8, 64],
            "scales": [1.0, 0.5],
            "hidden_in": 512,
            "max_rank": 64,
            "slices": 2,
        },
        id="narrow-hidden",
    ),
]


def generate_case(case, dtype, seed=20261106):
    """Build one CPU dataset, then move every tensor to the current NPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = sum(case["seq_lengths"])
    max_rank = case["max_rank"]
    slices = case["slices"]
    hidden_in = case["hidden_in"]
    packed_width = slices * max_rank
    num_loras = len(case["ranks"])

    x = torch.randn(
        tokens, hidden_in, generator=generator, dtype=torch.float32, device="cpu"
    ).to(dtype)
    weight = (
        torch.randn(
            num_loras,
            packed_width,
            hidden_in,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        * 0.02
    )
    # Rows past an adapter's packed rank segments must never be read.
    for slot, rank in enumerate(case["ranks"]):
        if rank < max_rank:
            weight[slot, slices * rank :, :] = 17.0
    weight = weight.to(dtype)
    # y starts dirty so untouched columns can be checked byte for byte.
    y = torch.randn(
        tokens, packed_width, generator=generator, dtype=torch.float32, device="cpu"
    )
    scales = torch.tensor(case["scales"], dtype=torch.float16, device="cpu")

    args = (
        x.npu(),
        weight.npu(),
        torch.tensor(case["lora_indices"], dtype=torch.int32, device="cpu").npu(),
        torch.tensor(case["seq_lengths"], dtype=torch.int32, device="cpu").npu(),
        torch.tensor(case["ranks"], dtype=torch.int32, device="cpu").npu(),
        scales.npu(),
    )
    return args, y.npu()


def written_mask(case):
    """CPU mask of the y entries the Kernel writes for this case."""
    max_rank = case["max_rank"]
    slices = case["slices"]
    mask = torch.zeros(
        sum(case["seq_lengths"]), slices * max_rank, dtype=torch.bool, device="cpu"
    )
    token_offset = 0
    for length, lora_index in zip(case["seq_lengths"], case["lora_indices"]):
        if length > 0 and lora_index >= 0:
            rank = case["ranks"][lora_index]
            if rank > 0:
                for index in range(slices):
                    mask[
                        token_offset : token_offset + length,
                        index * rank : (index + 1) * rank,
                    ] = True
        token_offset += length
    return mask


def run_case(case, dtype, *, enable_perf=False):
    from sgl_kernel_npu.operators.lora import sgemmv_shrink

    args, y = generate_case(case, dtype)
    slices = case["slices"]
    originals = tuple(tensor.clone() for tensor in args)

    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(sgemmv_shrink, impl)
    else:
        require_distinct_impls(sgemmv_shrink, impl, impl_ref)
        result_ref = sgemmv_shrink(*args, y.clone(), slices, impl=impl_ref)

    y_dut = y.clone()
    result = sgemmv_shrink(*args, y_dut, slices, impl=impl)
    if not skip_accuracy:
        assert_accuracy(result, result_ref, standard=STANDARD)

    # In-place contract: the returned tensor aliases the caller's y.
    assert result.data_ptr() == y_dut.data_ptr()
    # Every non-output input must be byte-identical after the call.
    for actual, expected, name in zip(args, originals, INPUT_NAMES):
        assert torch.equal(actual, expected), "{} was modified".format(name)
    # Columns outside every adapter's rank segments keep their original values.
    mask = written_mask(case).to(result.device)
    assert torch.equal(result[~mask], y[~mask]), "untouched y entries changed"

    executor = make_executor(
        sgemmv_shrink,
        args=(*args, y.clone(), slices),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    check_performance(executor, enable_perf=enable_perf)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", MAIN_CASES)
def test_sgemmv_shrink(case, dtype):
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
def test_sgemmv_shrink_hidden_tails(case, dtype):
    run_case(case, dtype)


def test_sgemmv_shrink_rejects_zero_batch():
    from sgl_kernel_npu.operators.lora import sgemmv_shrink

    impl, _ = get_test_impls()
    require_test_impl(sgemmv_shrink, impl)

    with pytest.raises(RuntimeError):
        sgemmv_shrink(
            torch.empty((0, 32), dtype=torch.float16, device="npu"),
            torch.empty((1, 8, 32), dtype=torch.float16, device="npu"),
            torch.empty((0,), dtype=torch.int32, device="npu"),
            torch.empty((0,), dtype=torch.int32, device="npu"),
            torch.tensor([8], dtype=torch.int32, device="npu"),
            torch.tensor([1.0], dtype=torch.float16, device="npu"),
            torch.empty((0, 8), dtype=torch.float32, device="npu"),
            impl=impl,
        )

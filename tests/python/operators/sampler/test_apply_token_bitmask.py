"""Model-vocabulary and boundary coverage through the public sampler interface."""

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

APPLY_TOKEN_BITMASK_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# SGLang's XGrammar NPU path passes logits [requests, config.vocab_size].
# Request counts are representative decode batches, not model constants.
APPLY_TOKEN_BITMASK_SHAPES = [
    # https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
    pytest.param((1, 151936), id="qwen3-8b-1x151936"),
    # https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
    pytest.param((16, 129280), id="deepseek-v3-16x129280"),
]
APPLY_TOKEN_BITMASK_PERF_SHAPES = [(1, 151936), (16, 129280)]
APPLY_TOKEN_BITMASK_EDGE_CASES = [
    pytest.param((2, 1), "none", id="minimum-vocab"),
    pytest.param((2, 31), "none", id="below-mask-word"),
    pytest.param((2, 32), "none", id="full-mask-word"),
    pytest.param((2, 33), "none", id="mask-word-tail"),
    pytest.param((8, 320), "subset", id="selected-rows"),
    pytest.param((4, 320), "all", id="all-rows"),
    pytest.param((4, 320), "empty", id="empty-indices"),
]
APPLY_TOKEN_BITMASK_STANDARD = {
    str(dtype): AccuracyStandard(atol=1e-2, rtol=1e-2)
    for dtype in APPLY_TOKEN_BITMASK_DTYPES
}


def generate_test_data(shape, dtype, mode, selection="none"):
    """Generate logits, bitmask, and optional row indices for bitmask tests."""
    generator = torch.Generator(device="cpu").manual_seed(20261022)
    logits = torch.randn(shape, generator=generator, device="cpu").to(dtype).npu()
    width = (shape[1] + 31) // 32
    if mode == "random":
        bitmask = torch.randint(
            -(2**31),
            2**31,
            (shape[0], width),
            generator=generator,
            dtype=torch.int32,
            device="cpu",
        ).npu()
    else:
        value = {"all_masked": 0, "all_unmasked": -1, "half_masked": 0x55555555}[mode]
        bitmask = torch.full(
            (shape[0], width), value, dtype=torch.int32, device=logits.device
        )
    choices = {
        "none": None,
        "subset": [0, 2, 5, 7],
        "all": list(range(shape[0])),
        "empty": [],
    }
    indices = (
        None
        if selection == "none"
        else torch.tensor(choices[selection], dtype=torch.int32, device=logits.device)
    )
    return logits, bitmask, indices


@pytest.mark.parametrize("dtype", APPLY_TOKEN_BITMASK_DTYPES)
@pytest.mark.parametrize("shape", APPLY_TOKEN_BITMASK_SHAPES)
@pytest.mark.parametrize(
    "mode", ["all_masked", "all_unmasked", "half_masked", "random"]
)
def test_apply_token_bitmask(shape, dtype, mode):
    from sgl_kernel_npu.operators.sampler import apply_token_bitmask

    logits, bitmask, _ = generate_test_data(shape, dtype, mode)

    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(apply_token_bitmask, impl)
    else:
        require_distinct_impls(apply_token_bitmask, impl, impl_ref)
        result_ref = apply_token_bitmask(logits.clone(), bitmask.clone(), impl=impl_ref)
    values, bits = logits.clone(), bitmask.clone()
    result = apply_token_bitmask(values, bits, impl=impl)
    if not skip_accuracy:
        assert_accuracy(result, result_ref, standard=APPLY_TOKEN_BITMASK_STANDARD)
    assert result.data_ptr() == values.data_ptr()
    torch.testing.assert_close(
        result[~torch.isneginf(result)], logits[~torch.isneginf(result)], atol=0, rtol=0
    )
    assert torch.equal(bits, bitmask)

    # Reapplying the same mask is idempotent; no input copies enter profiling.
    executor = make_executor(
        apply_token_bitmask,
        args=(values, bits),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    # SGLang's LogitsProcessor converts sampling logits to FP32.
    enable_perf = (
        shape in APPLY_TOKEN_BITMASK_PERF_SHAPES
        and dtype == torch.float32
        and mode == "random"
    )
    check_performance(executor, enable_perf=enable_perf)


@pytest.mark.parametrize("dtype", APPLY_TOKEN_BITMASK_DTYPES)
@pytest.mark.parametrize("shape,selection", APPLY_TOKEN_BITMASK_EDGE_CASES)
def test_apply_token_bitmask_boundaries(shape, dtype, selection):
    from sgl_kernel_npu.operators.sampler import apply_token_bitmask

    logits, bitmask, indices = generate_test_data(
        shape, dtype, mode="random", selection=selection
    )

    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(apply_token_bitmask, impl)
    else:
        require_distinct_impls(apply_token_bitmask, impl, impl_ref)
        result_ref = apply_token_bitmask(
            logits.clone(),
            bitmask.clone(),
            None if indices is None else indices.clone(),
            impl=impl_ref,
        )
    values, bits = logits.clone(), bitmask.clone()
    rows = None if indices is None else indices.clone()
    result = apply_token_bitmask(values, bits, rows, impl=impl)
    if not skip_accuracy:
        assert_accuracy(result, result_ref, standard=APPLY_TOKEN_BITMASK_STANDARD)
    assert result.data_ptr() == values.data_ptr()
    torch.testing.assert_close(
        result[~torch.isneginf(result)], logits[~torch.isneginf(result)], atol=0, rtol=0
    )
    assert torch.equal(bits, bitmask)
    if rows is not None:
        assert torch.equal(rows, indices)

    executor = make_executor(
        apply_token_bitmask,
        args=(values, bits, rows),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    check_performance(executor, enable_perf=False)

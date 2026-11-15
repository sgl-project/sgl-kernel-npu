"""Batch-invariance tests for batch-invariant fused linear + online max/sum.

Each token's logits and cross-entropy statistics must be bitwise identical
alone or inside a larger batch. The packed target mask is compared only for
aligned prefixes; only the logits are checked against torch.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.operators.batch_invariant.batch_invariance import (
    assert_batch_invariant,
    assert_unchanged,
    require_batch_invariant_ops,
    snapshot,
)
from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_skip_acc_check, get_test_impls, make_executor
from tests.python.utils.performance import check_performance

# Index of the bit-packed target mask in the six outputs.
MASK_INDEX = 3

# Decode-sized vocab-parallel loss; half the vocabulary is local to this rank.
perf_shape = [
    (257, 7168, 16384),
]

# Tokens, hidden, global vocabulary.
shape_cases = [
    pytest.param((257, 7168, 16384), id="decode-257x7168x16384"),
    pytest.param((17, 128, 256), id="edge-17x128x256"),
    pytest.param((7, 100, 33), id="edge-unaligned-7x100x33"),
    pytest.param((1, 1, 8), id="edge-1x1x8"),
]


def generate_test_data(shape, dtype, seed=42):
    """Create NPU inputs; the middle half of the vocabulary is local."""
    torch.npu.manual_seed(seed)
    tokens, hidden, vocab = shape
    vocab_start = vocab // 4
    vocab_end = 3 * vocab // 4
    input_tensor = torch.randn(tokens, hidden, dtype=dtype)
    weight = torch.randn(vocab_end - vocab_start, hidden, dtype=dtype)
    target = torch.randint(0, vocab, (tokens,), dtype=torch.int64)
    return input_tensor, weight, target, vocab_start, vocab_end


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", shape_cases)
def test_fused_linear_online_max_sum_batch_invariant(shape, dtype):
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import (
        fused_linear_online_max_sum_batch_invariant,
    )

    impl, impl_ref = get_test_impls()
    checked = get_skip_acc_check() is not True
    tokens = shape[0]
    input_tensor, weight, target, vocab_start, vocab_end = generate_test_data(
        shape, dtype
    )
    bounds = (vocab_start, vocab_end)
    if checked:
        snapshots = snapshot((input_tensor, weight, target))
        full = fused_linear_online_max_sum_batch_invariant(
            input_tensor, weight, target, *bounds, True, impl=impl
        )
        for take in sorted({1, 2, 7, 8, 16, tokens // 2}):
            if not 0 < take <= tokens:
                continue
            for offset in sorted({0, tokens // 2, tokens - take}):
                result = fused_linear_online_max_sum_batch_invariant(
                    input_tensor[offset : offset + take],
                    weight,
                    target[offset : offset + take],
                    *bounds,
                    True,
                    impl=impl,
                )
                aligned = offset % 8 == 0 and take % 8 == 0
                for index in range(len(full)):
                    if index == MASK_INDEX:
                        if not aligned:
                            continue
                        expected = full[index][offset // 8 : (offset + take) // 8]
                    else:
                        expected = full[index][offset : offset + take]
                    assert_batch_invariant(
                        expected,
                        result[index],
                        path=f"output[{index}][{offset}:{offset + take}]",
                    )
        assert_unchanged((input_tensor, weight, target), snapshots)
    executor = make_executor(
        fused_linear_online_max_sum_batch_invariant,
        args=(input_tensor, weight, target, *bounds, True),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=checked,
    )
    check_performance(
        executor, enable_perf=shape in perf_shape and dtype == torch.float16
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", shape_cases)
def test_fused_linear_online_max_sum_accuracy(shape, dtype):
    """Compare the local logits with torch and check output contracts."""
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import (
        fused_linear_online_max_sum_batch_invariant,
    )

    if get_skip_acc_check() is True:
        pytest.skip("Accuracy checks are disabled by --skip-acc-check=true")

    impl, _ = get_test_impls()
    tokens = shape[0]
    input_tensor, weight, target, vocab_start, vocab_end = generate_test_data(
        shape, dtype
    )
    result = fused_linear_online_max_sum_batch_invariant(
        input_tensor, weight, target, vocab_start, vocab_end, True, impl=impl
    )
    assert result[0].dtype == torch.float32  # Row-wise logits max.
    assert result[1].dtype == torch.float32  # Row-wise sum of exp.
    assert result[2].dtype == torch.float32  # Target logits.
    assert result[3].dtype == torch.uint8  # Packed target mask.
    assert tuple(result[3].shape) == ((tokens + 7) // 8,)
    assert result[4].dtype == target.dtype  # Masked target.
    expected = torch.matmul(input_tensor.float(), weight.float().t()).to(dtype)
    assert_accuracy(result[5], expected, path="logits")

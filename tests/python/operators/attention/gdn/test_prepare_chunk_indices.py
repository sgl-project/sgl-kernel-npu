from itertools import accumulate

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_skip_acc_check, get_test_impls


def generate_data(lengths, dtype, chunk_indices):
    cu_seqlens = torch.tensor([0, *accumulate(lengths)], dtype=dtype)
    expected = cu_seqlens.new_tensor(chunk_indices).reshape(-1, 2)
    return cu_seqlens, expected


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize(
    "lengths,chunk_size,chunk_indices",
    [
        pytest.param([1], 64, [(0, 0)], id="minimum"),
        # Packed GDN prefill (Qwen3-Next/Qwen3.5) uses chunk_size=64.
        pytest.param(
            [128, 65, 64],
            64,
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
            id="qwen3-next-packed-prefill",
        ),
        pytest.param([4, 0, 3], 3, [(0, 0), (0, 1), (1, 0)], id="empty-sequence"),
        pytest.param([0, 0], 64, [], id="all-empty"),
    ],
)
def test_prepare_chunk_indices(lengths, chunk_size, chunk_indices, dtype):
    from sgl_kernel_npu.operators.attention.gdn import prepare_chunk_indices

    cu_seqlens, expected = generate_data(lengths, dtype, chunk_indices)
    impl, _ = get_test_impls()
    original = cu_seqlens.clone()
    result = prepare_chunk_indices(cu_seqlens, chunk_size, impl=impl)
    if get_skip_acc_check() is not True:
        assert_accuracy(result, expected)
        assert_accuracy(cu_seqlens, original)

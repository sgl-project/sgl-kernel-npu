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

# Packed LongCat2 decode and prefill workloads, with empty ignore_tokens as
# used by the model. Sizes are representative, not captured model tensors.
perf_shape = [(16, 1), (1, 2048)]


def generate_data(batch_size, request_length, prefix_length, mode):
    """Build independent starting state and a mask of the written region."""
    if mode == "legacy_empty_ignore":
        table = torch.full((4, 8), -1, dtype=torch.int32)
        tokens = torch.tensor([11, 12, 13, 21, 22], dtype=torch.int32)
        lengths = [3, 0, 2]
        rows = [2, 1, 0]
        columns = [1, 4, 3]
        ignore = []
    elif mode == "legacy_ignore":
        table = torch.full((3, 7), -7, dtype=torch.int32)
        tokens = torch.tensor([5, 9, 6, 7, 9, 8], dtype=torch.int32)
        lengths = [2, 4]
        rows = [1, 2]
        columns = [0, 2]
        ignore = [9, 10]
    else:
        lengths = [request_length] * batch_size
        rows = list(reversed(range(batch_size)))
        columns = [prefix_length] * batch_size
        table = torch.full(
            (batch_size + 1, prefix_length + request_length + 3),
            9,
            dtype=torch.int32,
        )
        tokens = torch.arange(batch_size * request_length, dtype=torch.int32) % 17
        ignore = [9, 10] if mode == "ignore" else []

    written = torch.zeros_like(table, dtype=torch.bool)
    for row, column, length in zip(rows, columns, lengths):
        written[row, column : column + length] = True
    return (
        dict(
            tokens=tokens,
            req_lens=torch.tensor(lengths, dtype=torch.int32),
            row_indices=torch.tensor(rows, dtype=torch.int64),
            column_starts=torch.tensor(columns, dtype=torch.int32),
            ignore_tokens=torch.tensor(ignore, dtype=torch.int32),
            batch_size=batch_size,
            max_context_len=table.shape[1],
        ),
        table,
        written,
    )


@pytest.mark.parametrize(
    "batch_size,request_length,prefix_length,mode",
    [
        pytest.param(
            3, 3, 1, "legacy_empty_ignore", id="existing-empty-ignore-zero-request"
        ),
        pytest.param(2, 4, 2, "legacy_ignore", id="existing-ignore-tokens"),
        pytest.param(16, 1, 1024, "empty_ignore", id="longcat2-batched-decode"),
        pytest.param(1, 2048, 16, "empty_ignore", id="longcat2-chunked-prefill"),
        pytest.param(1, 1, 0, "empty_ignore", id="edge-single-token"),
        pytest.param(3, 7, 3, "ignore", id="edge-odd-length-preserve-unwritten-ignore"),
    ],
)
def test_update_oe_token_table(batch_size, request_length, prefix_length, mode):
    from sgl_kernel_npu.operators.embedding import update_oe_token_table

    impl, impl_ref = get_test_impls()
    require_test_impl(update_oe_token_table, impl)
    inputs, initial_table, written = generate_data(
        batch_size, request_length, prefix_length, mode
    )
    actual_table = initial_table.clone()
    executor = make_executor(
        update_oe_token_table,
        kwargs={**inputs, "oe_token_table": actual_table},
        impl=impl,
        impl_ref=impl_ref,
    )
    if get_skip_acc_check() is not True:
        require_distinct_impls(update_oe_token_table, impl, impl_ref)
        # The two implementations must not share the mutable table. Compare
        # both the return value and the table, including unwritten entries.
        reference_table = initial_table.clone()
        expected = update_oe_token_table(
            **inputs, oe_token_table=reference_table, impl=impl_ref
        )
        actual = executor()
        assert actual.data_ptr() == actual_table.data_ptr()
        assert expected.data_ptr() == reference_table.data_ptr()
        assert_accuracy(actual, expected)
        assert_accuracy(actual_table, reference_table)
        assert_accuracy(actual_table[~written], initial_table[~written])
        executor.accuracy_checked = True

    # Rewriting the same slices with the same tokens is idempotent, so the
    # public operator can be profiled without cloning inside the timed call.
    check_performance(
        executor,
        enable_perf=(batch_size, request_length) in perf_shape
        and mode == "empty_ignore",
    )

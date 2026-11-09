"""Causal KPool packing with isolated in-place inputs and representative profiles."""

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

# Same GLM workloads as the preparation operator: extend=4096, MTP=64*5.
perf_shape = [(4096, 1), (320, 64)]
TOPK, POOL_SIZE = 2048, 4


def generate_test_data(shape):
    tokens, requests = shape
    width = TOPK + POOL_SIZE - 1
    query_ends = torch.arange(1, requests + 1, dtype=torch.int64) * tokens // requests
    starts = torch.cat((query_ends.new_zeros(1), query_ends[:-1]))
    query_lens = query_ends - starts
    positions = torch.arange(tokens, dtype=torch.int64) - torch.repeat_interleave(
        starts, query_lens, output_size=tokens
    )
    positions += 8192 if shape in perf_shape else 0
    tail_start = (positions + 1) // POOL_SIZE * POOL_SIZE
    # BF16 CANN expands selected pools to token indices; keep valid history IDs.
    indices = (
        torch.arange(width, dtype=torch.int32).expand(tokens, -1)
        % tail_start.clamp_min(1).unsqueeze(1)
    ).to(torch.int32)
    return indices, positions, query_ends


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((4096, 1), id="glm53-extend-4096"),
        pytest.param((320, 64), id="glm53-mtp-b64-n5"),
        pytest.param((1, 1), id="short-history"),
        pytest.param((7, 2), id="odd-rows"),
    ],
)
@pytest.mark.parametrize(
    "reuse_output", [False, True], ids=["in-place", "output-buffer"]
)
def test_pack_pool_key_indexer_indices(shape, reuse_output):
    from sgl_kernel_npu.operators.indexer import pack_pool_key_indexer_indices

    indices, positions, query_ends = generate_test_data(shape)
    tokens, width = indices.shape

    def isolated_pack(impl):
        source = indices.clone()
        buffer = (
            torch.full((tokens + 1, width + 5), 19, dtype=torch.int32)
            if reuse_output
            else None
        )
        result = pack_pool_key_indexer_indices(
            source,
            positions,
            query_ends,
            TOPK,
            POOL_SIZE,
            output_buffer=buffer,
            impl=impl,
        )
        if reuse_output:
            assert result.data_ptr() == buffer.data_ptr()
            assert_accuracy(source, indices)
            assert_accuracy(
                buffer[:tokens, width:], torch.full_like(buffer[:tokens, width:], -1)
            )
            assert_accuracy(buffer[tokens:], torch.full_like(buffer[tokens:], 19))
        elif tokens:
            assert result.data_ptr() == source.data_ptr()
        return result

    impl, impl_ref = get_test_impls()
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        require_distinct_impls(pack_pool_key_indexer_indices, impl, impl_ref)
        assert_accuracy(isolated_pack(impl), isolated_pack(impl_ref))
    else:
        require_test_impl(pack_pool_key_indexer_indices, impl)
    buffer = torch.empty_like(indices)
    # All valid history columns survive unchanged: reusing inputs is steady-state.
    executor = make_executor(
        pack_pool_key_indexer_indices,
        args=(indices, positions, query_ends, TOPK, POOL_SIZE),
        kwargs={"output_buffer": buffer} if reuse_output else {},
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=accuracy_checked,
    )
    check_performance(executor, enable_perf=shape in perf_shape and not reuse_output)


@pytest.mark.parametrize(
    "real_tokens", [0, 3, 5], ids=["all-padding", "prefix", "full"]
)
def test_pack_empty_history_and_padding(real_tokens):
    """Synthetic pool candidates must never leak into short-history or padding rows."""
    from sgl_kernel_npu.operators.indexer import pack_pool_key_indexer_indices

    positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    ends = torch.tensor([real_tokens // 2, real_tokens], dtype=torch.int64)
    expected = torch.full((5, TOPK + POOL_SIZE - 1), -1, dtype=torch.int32)
    for row in range(real_tokens):
        expected[row, : row + 1] = torch.arange(row + 1, dtype=torch.int32)
    impl, impl_ref = get_test_impls()
    require_distinct_impls(pack_pool_key_indexer_indices, impl, impl_ref)
    outputs = []
    for backend in (impl, impl_ref):
        # Each artificial pool reports [0,1,2,3]; short requests keep only their tail.
        source = (
            torch.arange(TOPK + POOL_SIZE - 1, dtype=torch.int32).expand(5, -1).clone()
        )
        outputs.append(
            pack_pool_key_indexer_indices(
                source, positions, ends, TOPK, POOL_SIZE, impl=backend
            )
        )
    assert_accuracy(outputs[0], outputs[1])
    assert_accuracy(outputs[0], expected)

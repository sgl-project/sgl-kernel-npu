import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# SGLang AscendDSAAttnBackend expands packed extend/target-verify queries.
# B64/Q4 and B1/Q4096 are representative workloads, not captured runtime shapes.
perf_shape = [(64, 256), (1, 4096)]


def generate_test_data(cache_seqlens, query_lens, max_query_len):
    cache = torch.tensor(cache_seqlens, dtype=torch.int32)
    return dict(
        cache_seqlens=cache,
        query_lens=torch.tensor(query_lens, dtype=torch.int32),
        device=cache.device,
        total_q=sum(query_lens),
        max_query_len=max_query_len,
    )


@pytest.mark.parametrize(
    "cache_seqlens,query_lens,max_query_len",
    [
        # Case 01: Minimum nonempty expansion.
        pytest.param([1], [1], 1, id="single-token"),
        # Case 02: Ragged requests, zero-query request and clamped causal start.
        pytest.param([10, 8, 1], [2, 0, 3], 3, id="ragged-zero-query-clamp"),
        # Case 03: Cross 256-token tiles; launch bound may exceed actual maximum.
        pytest.param([600, 0, 1024], [257, 0, 513], 1024, id="query-tile-tail"),
        # Case 04: Empty batch returns two typed empty outputs.
        pytest.param([], [], 0, id="empty-batch"),
        # Case 05: Nonempty batch with no packed query tokens.
        pytest.param([0, 8, 16], [0, 0, 0], 0, id="all-zero-queries"),
        # Case 06: Typical target verification, B64/Q4; also profile.
        pytest.param(list(range(100, 164)), [4] * 64, 4, id="verify-b64-q4"),
        # Case 07: Typical extend, one 4096-token query; also profile.
        pytest.param([8192], [4096], 4096, id="extend-b1-q4096"),
    ],
)
def test_expand_causal_seqlens(cache_seqlens, query_lens, max_query_len):
    from sgl_kernel_npu.operators.indexer import expand_causal_seqlens

    inputs = generate_test_data(cache_seqlens, query_lens, max_query_len)
    executor = check_accuracy(expand_causal_seqlens, kwargs=inputs)
    shape = (len(cache_seqlens), inputs["total_q"])
    check_performance(executor, enable_perf=shape in perf_shape)


# Case 08: Graph replay reuses output storage while request lengths change.
# In-place Operator
@pytest.mark.extra_checks
def test_expand_causal_seqlens_reuses_outputs():
    """In-place: overwrite only the two supplied outputs; keep their identity."""
    from sgl_kernel_npu.operators.indexer import expand_causal_seqlens

    impl, impl_ref = get_test_impls()
    if get_skip_acc_check() is True:
        require_test_impl(expand_causal_seqlens, impl)
        return  # This case checks storage reuse only; no profiling workload.
    require_distinct_impls(expand_causal_seqlens, impl, impl_ref)
    outputs = (
        torch.full((5,), -1, dtype=torch.int32),
        torch.full((5,), -1, dtype=torch.int64),
    )
    reference_outputs = tuple(tensor.clone() for tensor in outputs)
    for query_lens in ([2, 0, 3], [0, 3, 2]):
        inputs = generate_test_data([10, 8, 20], query_lens, 3)
        reference = expand_causal_seqlens(
            **inputs,
            seqlens_out=reference_outputs[0],
            token_to_batch_out=reference_outputs[1],
            impl=impl_ref,
        )
        actual = expand_causal_seqlens(
            **inputs,
            seqlens_out=outputs[0],
            token_to_batch_out=outputs[1],
            impl=impl,
        )
        assert_accuracy(actual, reference)
        for result, output, ref_result, ref_output in zip(
            actual, outputs, reference, reference_outputs
        ):
            assert result is output
            assert ref_result is ref_output

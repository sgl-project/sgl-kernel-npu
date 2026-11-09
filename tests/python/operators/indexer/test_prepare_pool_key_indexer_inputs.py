"""GLM-5.3-Flash sentinel preparation and empty-history regression."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import get_test_impls
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Representative workloads: chunk=4096; MTP B=64, N=5. Not captured tensors.
# GLM indexer dimensions: 32 heads, 128 channels, pool=4.
perf_shape = [(4096, 1, 64), (320, 64, 64)]
EXACT = {"torch.bfloat16": AccuracyStandard(atol=0, rtol=0)}


def generate_test_data(shape, *, strided=False):
    tokens, requests, columns = shape
    torch.manual_seed(908)
    step = 2 if strided else 1
    query = torch.randn(tokens, 32, 128 * step, dtype=torch.bfloat16)[..., ::step]
    weights = torch.randn(tokens, 32 * step, dtype=torch.bfloat16)[:, ::step]
    ends = torch.arange(1, requests + 1, dtype=torch.int64) * tokens // requests
    query_lens = ends - torch.cat((ends.new_zeros(1), ends[:-1]))
    visible = query_lens + (8192 if shape in perf_shape else 0)
    lengths, tails = visible // 4, visible % 4
    table = torch.arange(requests * columns * step, dtype=torch.int32).reshape(
        requests, columns * step
    )[:, ::step]
    return query, weights, ends, lengths, tails, table


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((4096, 1, 64), id="glm53-extend-4096"),
        pytest.param((320, 64, 64), id="glm53-mtp-b64-n5"),
        pytest.param((1, 1, 1), id="minimum"),
        pytest.param((65, 4, 17), id="copy-tile-tail"),
    ],
)
def test_prepare_pool_key_indexer_inputs(shape):
    from sgl_kernel_npu.operators.indexer import prepare_pool_key_indexer_inputs

    inputs = generate_test_data(shape)
    executor = check_accuracy(
        prepare_pool_key_indexer_inputs, args=inputs, standard=EXACT
    )
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.extra_checks
def test_prepare_strided_inputs_remain_read_only():
    from sgl_kernel_npu.operators.indexer import prepare_pool_key_indexer_inputs

    inputs = generate_test_data((5, 2, 17), strided=True)
    originals = tuple(value.clone() for value in inputs)
    executor = check_accuracy(
        prepare_pool_key_indexer_inputs, args=inputs, standard=EXACT
    )
    outputs = executor()
    assert all(value.is_contiguous() for value in outputs)
    assert_accuracy(inputs, originals, EXACT)


def generate_empty_history_data():
    # Two real requests and five static rows: later rows can be graph padding.
    query = torch.zeros(5, 32, 128, dtype=torch.bfloat16)
    weights = torch.ones(5, 32, dtype=torch.bfloat16)
    ends = torch.tensor([3, 3], dtype=torch.int64)
    lengths = torch.zeros(2, dtype=torch.int64)
    tails = torch.tensor([3, 0], dtype=torch.int64)
    table = torch.zeros(2, 1, dtype=torch.int32)
    cache = torch.zeros(1, 128, 1, 128, dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 2, 0, 0], dtype=torch.int64)
    return (query, weights, ends, lengths, tails, table), cache, positions


@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "replay"])
def test_empty_history_and_graph_padding_regression(graph_mode):
    """Zero-pool/all-padding requests must not enter CANN's unsafe no-work path."""
    from sgl_kernel_npu.operators.indexer import (
        pack_pool_key_indexer_indices,
        pool_key_indexer,
        prepare_pool_key_indexer_inputs,
    )

    inputs, cache, positions = generate_empty_history_data()
    query, weights, ends, lengths, tails, table = inputs
    impl, _ = get_test_impls()
    prepared = check_accuracy(
        prepare_pool_key_indexer_inputs, args=inputs, standard=EXACT
    )()
    # Fail before CANN if a future rewrite removes the nonempty sentinel.
    assert prepared[0].shape[0] == query.shape[0] + 1
    assert_accuracy(prepared[0][-1], torch.zeros_like(query[0]))
    assert_accuracy(prepared[1][-1], torch.zeros_like(weights[0]))
    assert_accuracy(prepared[2][-1:], ends.new_tensor([6]))
    assert_accuracy(prepared[3], lengths.new_ones(3))
    assert_accuracy(prepared[4][-1:], tails.new_zeros(1))
    assert_accuracy(prepared[5][-1], torch.zeros_like(table[0]))

    def run():
        safe = prepare_pool_key_indexer_inputs(*inputs, impl=impl)
        selected = pool_key_indexer(
            safe[0],
            cache,
            safe[1],
            safe[4],
            actual_seq_q=safe[2],
            actual_seq_k=safe[3],
            block_table=safe[5],
            topk=2048,
            pool_size=4,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            impl="aclnn",
        )
        return pack_pool_key_indexer_indices(
            selected[: query.shape[0]],
            positions,
            ends,
            2048,
            4,
            impl=impl,
        )

    run()
    torch.npu.synchronize()
    graph = None
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run()
    # Replay one graph through live, empty-local and fully padded iterations.
    for query_ends, pool_lens, tail_lens, token_positions in (
        ([3, 3], [0, 0], [3, 0], [0, 1, 2, 0, 0]),
        ([0, 3], [0, 0], [0, 3], [0, 1, 2, 0, 0]),
        ([0, 0], [0, 0], [0, 0], [0, 0, 0, 0, 0]),
        ([2, 4], [1, 2], [1, 1], [3, 4, 7, 8, 0]),
    ):
        for target, values in zip(
            (ends, lengths, tails, positions),
            (query_ends, pool_lens, tail_lens, token_positions),
        ):
            target.copy_(target.new_tensor(values))
        if graph is None:
            output = run()
        else:
            graph.replay()
        # Every visible token fits in top-k; pool ordering is intentionally free.
        expected = torch.full((5, 2051), -1, dtype=torch.int32)
        for row in range(query_ends[-1]):
            expected[row, : token_positions[row] + 1] = torch.arange(
                token_positions[row] + 1, dtype=torch.int32
            )
        assert_accuracy(output.sort(1).values, expected.sort(1).values)

"""GLM-5.3-Flash sentinel preparation."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
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

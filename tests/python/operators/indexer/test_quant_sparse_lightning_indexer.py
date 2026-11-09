"""DSV4.1 decode/verify and empty-key regression, with selected device timing."""

import pytest

from tests.python.utils.performance import check_performance

from .indexer_utils import check_indexer, generate_data

perf_shape = [(1, 4096), (6, 4096)]


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4096), id="dsv41-decode"),
        pytest.param((6, 4096), id="dsv41-static-verify6"),
        pytest.param((1, 0), id="edge-empty-keys"),
    ],
)
def test_quant_sparse_lightning_indexer(shape):
    from sgl_kernel_npu.operators.indexer import (
        quant_sparse_lightning_indexer,
        quant_sparse_lightning_indexer_metadata,
    )

    args, kwargs, meta = generate_data(*shape, sparse=True)
    kwargs["metadata"] = quant_sparse_lightning_indexer_metadata(**meta)
    executor = check_indexer(quant_sparse_lightning_indexer, args, kwargs, shape[1])
    check_performance(executor, enable_perf=shape in perf_shape)

"""sparse_flash_mla_metadata accuracy and isolated operator timing.

Metadata correctness uses paired compute with an independent eager reference.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

from tests.python.operators.attention.test_sparse_flash_mla import (
    check_sparse_flash_mla_accuracy,
    generate_data,
)
from tests.python.utils.performance import check_performance

perf_shape = [(1, 1), (1, 128)]


@pytest.mark.parametrize(
    "batch,query_len",
    [
        pytest.param(1, 1, id="deepseek-v4-flash-decode-b1-q1-n64-d512-page128"),
        pytest.param(1, 128, id="deepseek-v4-flash-prefill-b1-q128-n64-d512-page128"),
        pytest.param(
            2, 2, id="synthetic-ragged-extend", marks=pytest.mark.extra_checks
        ),
    ],
)
def test_sparse_flash_mla_metadata(batch, query_len):
    data = generate_data(batch, query_len)
    executor = check_sparse_flash_mla_accuracy(data, target="metadata")
    check_performance(executor, enable_perf=(batch, query_len) in perf_shape)

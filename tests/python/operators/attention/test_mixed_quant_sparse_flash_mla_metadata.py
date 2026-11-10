"""mixed_quant_sparse_flash_mla_metadata accuracy and isolated operator timing.

Metadata correctness uses paired compute with an independent eager reference.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

from tests.python.operators.attention.test_mixed_quant_sparse_flash_mla import (
    check_mixed_quant_sparse_flash_mla_accuracy,
    generate_data,
)
from tests.python.utils.performance import check_performance

perf_shape = [1]


@pytest.mark.parametrize(
    "query_len",
    [1, 128],
    ids=[
        "deepseek-v4-flash-decode-q1-n64-d512-page128",
        "deepseek-v4-flash-prefill-q128-n64-d512-page128",
    ],
)
def test_mixed_quant_sparse_flash_mla_metadata(query_len):
    data = generate_data(query_len)
    executor = check_mixed_quant_sparse_flash_mla_accuracy(data, target="metadata")
    check_performance(executor, enable_perf=query_len in perf_shape)

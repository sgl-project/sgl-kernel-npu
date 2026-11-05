"""sparse_flash_mla accuracy and isolated operator timing.

Metadata correctness uses paired compute with an independent eager reference.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.operators.attention.sparse_flash_mla_inputs import (
    run_sparse_flash_mla_case,
)
from tests.python.utils.performance import check_performance


@pytest.mark.parametrize(
    "batch,query_len",
    [(1, 1), (2, 1), (2, 2)],
    ids=[
        "synthetic-single-decode",
        "synthetic-batched-decode",
        "synthetic-two-token-extend",
    ],
)
def test_sparse_flash_mla(batch, query_len):
    executor = run_sparse_flash_mla_case(batch, query_len, target="compute")
    check_performance(executor, enable_perf=(batch, query_len) in [(2, 1)])

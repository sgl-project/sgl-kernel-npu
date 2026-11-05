"""mixed_quant_sparse_flash_mla accuracy and isolated operator timing.

Metadata correctness uses paired compute with an independent eager reference.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.operators.attention.mixed_quant_sparse_flash_mla_inputs import (
    run_mixed_quant_sparse_flash_mla_case,
)
from tests.python.utils.performance import check_performance


@pytest.mark.parametrize(
    "query_len", [1, 2], ids=["synthetic-decode", "synthetic-extend"]
)
def test_mixed_quant_sparse_flash_mla(query_len):
    executor = run_mixed_quant_sparse_flash_mla_case(query_len, target="compute")
    check_performance(executor, enable_perf=query_len == 1)

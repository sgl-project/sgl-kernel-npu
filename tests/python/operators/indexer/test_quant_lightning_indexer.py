"""Top-K set comparison against independent dequantized PyTorch math."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.operators.indexer.indexer_inputs import (
    CASE_IDS,
    CASES,
    generate_data,
    hardware_mode,
    paired_inputs,
    reference_inputs,
)
from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_skip_acc_check, make_executor
from tests.python.utils.performance import check_performance

perf_shape = [CASES[1]]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_quant_lightning_indexer(case):
    from sgl_kernel_npu.operators.indexer import quant_lightning_indexer

    args, kwargs, kwargs_ref, _, _ = paired_inputs(generate_data(case, hardware_mode()))
    executor = make_executor(quant_lightning_indexer, args=args, kwargs=kwargs)
    if get_skip_acc_check() is not True:
        reference_args, reference_kwargs = reference_inputs(
            args, kwargs, kwargs_ref, executor.impl_ref
        )
        expected = quant_lightning_indexer(
            *reference_args, **reference_kwargs, impl=executor.impl_ref
        )
        # The selected Top-K set is the contract; accelerator kernels may
        # return the same indices in a different order.
        actual_indices, actual_values = executor()
        expected_indices, expected_values = expected
        assert_accuracy(
            (
                actual_indices.cpu().sort(dim=-1).values,
                actual_values.cpu(),
            ),
            (
                expected_indices.cpu().sort(dim=-1).values,
                expected_values.cpu(),
            ),
        )
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=case in perf_shape)

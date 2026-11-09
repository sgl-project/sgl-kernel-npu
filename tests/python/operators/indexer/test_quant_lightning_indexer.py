"""Top-K set comparison against independent dequantized PyTorch math."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

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


def assert_indexer_accuracy(actual, expected):
    # Compare the selected set; accelerator kernels can return a different order.
    actual_indices, actual_values = actual
    expected_indices, expected_values = expected
    assert_accuracy(
        (actual_indices.cpu().sort(dim=-1).values, actual_values.cpu()),
        (expected_indices.cpu().sort(dim=-1).values, expected_values.cpu()),
    )


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
        assert_indexer_accuracy(executor(), expected)
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=case in perf_shape)


@pytest.mark.extra_checks
@pytest.mark.parametrize("case", CASES[:2], ids=CASE_IDS[:2])
def test_quant_lightning_indexer_queued_lifetimes(case):
    from sgl_kernel_npu.operators.indexer import (
        quant_lightning_indexer,
        quant_lightning_indexer_metadata,
    )

    if get_skip_acc_check() is True:
        pytest.skip("Queued lifetime regression requires accuracy comparison")
    args, kwargs, kwargs_ref, metadata_args, common = paired_inputs(
        generate_data(case, hardware_mode())
    )
    executor = make_executor(quant_lightning_indexer, args=args, kwargs=kwargs)
    reference_args, reference_kwargs = reference_inputs(
        args, kwargs, kwargs_ref, executor.impl_ref
    )
    expected = quant_lightning_indexer(
        *reference_args, **reference_kwargs, impl=executor.impl_ref
    )
    results = []
    for _ in range(4):
        # Each iteration replaces the previous metadata Tensor. The queued
        # bridge must own its inputs and workspace until the ACL call runs.
        metadata = quant_lightning_indexer_metadata(
            *metadata_args, impl=executor.impl, **common
        )
        results.append(
            quant_lightning_indexer(
                *args, **{**kwargs, "metadata": metadata}, impl=executor.impl
            )
        )
        # Encourage allocator reuse without a host read or synchronization.
        torch.empty(1 << 20, dtype=torch.uint8, device=args[0].device).fill_(0xA5)
    for result in results:
        assert_indexer_accuracy(result, expected)

"""Validate opaque tiling via paired compute; profile the metadata op alone."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

from tests.python.operators.indexer.test_quant_lightning_indexer import (
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
def test_quant_lightning_indexer_metadata(case):
    from sgl_kernel_npu.operators.indexer import (
        quant_lightning_indexer,
        quant_lightning_indexer_metadata,
    )

    args, kwargs, kwargs_ref, metadata_args, common = paired_inputs(
        generate_data(case, hardware_mode())
    )
    metadata = kwargs["metadata"]
    assert isinstance(metadata, torch.Tensor)
    assert metadata.dtype == torch.int32
    from sgl_kernel_npu.operators.indexer import (
        get_quant_lightning_indexer_metadata_size,
    )

    assert metadata.shape == (get_quant_lightning_indexer_metadata_size(),)
    assert metadata.is_contiguous()
    # Binary tiling is not comparable to the eager reference descriptor.
    compute = make_executor(quant_lightning_indexer, args=args, kwargs=kwargs)
    if get_skip_acc_check() is not True:
        reference_args, reference_kwargs = reference_inputs(
            args, kwargs, kwargs_ref, compute.impl_ref
        )
        expected = quant_lightning_indexer(
            *reference_args, **reference_kwargs, impl=compute.impl_ref
        )
        actual_indices, actual_values = compute()
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
        compute.accuracy_checked = True
    executor = make_executor(
        quant_lightning_indexer_metadata,
        args=metadata_args,
        kwargs=common,
        accuracy_checked=compute.accuracy_checked,
    )
    check_performance(executor, enable_perf=case in perf_shape)

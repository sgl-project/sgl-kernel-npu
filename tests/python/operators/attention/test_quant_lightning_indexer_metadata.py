"""Validate opaque tiling via paired compute; profile the metadata op alone."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.operators.attention._indexer_inputs import (
    CASE_IDS,
    CASES,
    paired_inputs,
)
from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_skip_acc_check, make_executor
from tests.python.utils.performance import check_performance


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_quant_lightning_indexer_metadata(case):
    from sgl_kernel_npu.operators.attention import (
        quant_lightning_indexer,
        quant_lightning_indexer_metadata,
    )

    args, kwargs, kwargs_ref, metadata_args, common = paired_inputs(case)
    metadata = kwargs["metadata"]
    assert isinstance(metadata, torch.Tensor)
    assert metadata.dtype == torch.int32
    assert metadata.ndim == 1 and metadata.numel() > 0
    assert metadata.is_contiguous()
    # Binary tiling is not comparable to the eager reference descriptor.
    compute = make_executor(quant_lightning_indexer, args=args, kwargs=kwargs)
    if get_skip_acc_check() is not True:
        expected = quant_lightning_indexer(
            *args, **{**kwargs, **kwargs_ref}, impl=compute.impl_ref
        )
        actual_indices, actual_values = compute()
        expected_indices, expected_values = expected
        assert_accuracy(
            (
                actual_indices.sort(dim=-1).values,
                actual_values,
            ),
            (
                expected_indices.sort(dim=-1).values,
                expected_values,
            ),
        )
        compute.accuracy_checked = True
    executor = make_executor(
        quant_lightning_indexer_metadata,
        args=metadata_args,
        kwargs=common,
        accuracy_checked=compute.accuracy_checked,
    )
    check_performance(executor, enable_perf=case == CASES[1])

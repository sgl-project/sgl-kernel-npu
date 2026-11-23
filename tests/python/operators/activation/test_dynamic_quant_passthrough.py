"""Check that public quantization wrappers preserve the vendor call."""

from unittest.mock import Mock

import pytest
import torch_npu

from tests.python.utils.common import require_test_impl


@pytest.mark.parametrize("name", ["dynamic_quant", "dynamic_mx_quant"])
def test_vendor_arguments_and_results_are_forwarded(name, monkeypatch):
    from sgl_kernel_npu.operators import activation

    operator = getattr(activation, name)
    require_test_impl(operator, "auto")
    vendor_name = "npu_" + name
    result = (object(), object())
    vendor = Mock(return_value=result)
    monkeypatch.setattr(torch_npu, vendor_name, vendor)
    x = object()
    if name == "dynamic_mx_quant":
        options = {"axis": 1, "block_size": 64}
    else:
        options = {"smooth_scales": object(), "quant_mode": "pertensor"}

    assert operator(x, **options) is result
    vendor.assert_called_once_with(x, **options)

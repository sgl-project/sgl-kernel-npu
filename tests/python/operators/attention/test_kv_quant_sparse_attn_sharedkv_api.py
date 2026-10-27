"""Check the refactored public API alongside the legacy Torch schema.

The numerical paged-SWA comparison runs in the existing A5 test module.
"""

import pytest
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import (
    kv_quant_sparse_attn_sharedkv,
    kv_quant_sparse_attn_sharedkv_metadata_host,
)
from sgl_kernel_npu.operators.utils.registry import get_op_name, list_impls


@pytest.mark.parametrize(
    ("operator", "name", "expected_impls", "native_name"),
    [
        (
            kv_quant_sparse_attn_sharedkv,
            "attention.kv_quant_sparse_attn_sharedkv",
            {"auto", "ascendc", "torch_native"},
            "kv_quant_sparse_attn_sharedkv",
        ),
        (
            kv_quant_sparse_attn_sharedkv_metadata_host,
            "attention.kv_quant_sparse_attn_sharedkv_metadata_host",
            {"auto", "host"},
            "kv_quant_sparse_attn_sharedkv_metadata_host",
        ),
    ],
)
def test_public_registration(operator, name, expected_impls, native_name):
    if not torch_npu.npu.is_available() or "950" not in torch_npu.npu.get_device_name(
        0
    ).lower():
        pytest.skip("quantized shared-KV attention requires Ascend 950")
    assert get_op_name(operator) == name
    assert set(list_impls(operator)) == expected_impls
    assert hasattr(torch.ops.npu, native_name)

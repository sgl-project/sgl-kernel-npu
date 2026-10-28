"""Check the refactored public API alongside the legacy Torch schema.

The numerical paged-SWA comparison runs in the existing A5 test module.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


@pytest.mark.parametrize(
    ("public_name", "name", "expected_impls", "native_name"),
    [
        (
            "kv_quant_sparse_attn_sharedkv",
            "attention.kv_quant_sparse_attn_sharedkv",
            {"auto", "ascendc", "torch_native"},
            "kv_quant_sparse_attn_sharedkv",
        ),
        (
            "kv_quant_sparse_attn_sharedkv_metadata_host",
            "attention.kv_quant_sparse_attn_sharedkv_metadata_host",
            {"auto", "host"},
            "kv_quant_sparse_attn_sharedkv_metadata_host",
        ),
    ],
)
def test_public_registration(
    public_name, name, expected_impls, native_name, configure_npu
):
    if "950" not in torch.npu.get_device_name(torch.npu.current_device()).lower():
        pytest.skip("quantized shared-KV attention requires Ascend 950")
    from sgl_kernel_npu.operators import attention
    from sgl_kernel_npu.operators.utils.registry import get_op_name, list_impls

    operator = getattr(attention, public_name)
    assert get_op_name(operator) == name
    assert set(list_impls(operator)) == expected_impls
    assert hasattr(torch.ops.npu, native_name)

"""Public-API registration, accuracy and performance checks for A5 metadata."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance


@pytest.fixture(autouse=True)
def require_ascend950(configure_npu):
    device_index = torch.npu.current_device()
    if "950" not in torch.npu.get_device_name(device_index).lower():
        pytest.skip("quantized shared-KV metadata requires Ascend 950")


def test_public_registration():
    from sgl_kernel_npu.operators import attention
    from sgl_kernel_npu.operators.utils.registry import get_op_name, list_impls

    operator = attention.kv_quant_sparse_attn_sharedkv_metadata_host
    assert (
        get_op_name(operator)
        == "attention.kv_quant_sparse_attn_sharedkv_metadata_host"
    )
    assert set(list_impls(operator)) == {"auto", "host", "torch_native"}
    assert hasattr(torch.ops.npu, "kv_quant_sparse_attn_sharedkv_metadata_host")


@pytest.mark.parametrize(
    ("layout", "heads", "has_cmp", "cmp_ratio", "kv_len"),
    [
        pytest.param("BSND", 64, False, 1, 128, id="decode-64-heads"),
        pytest.param("TND", 64, True, 4, 128, id="compressed-64-heads"),
        pytest.param("TND", 128, False, 1, 256, id="decode-128-heads"),
    ],
)
def test_host_metadata_accuracy(layout, heads, has_cmp, cmp_ratio, kv_len):
    from sgl_kernel_npu.operators.attention import (
        kv_quant_sparse_attn_sharedkv_metadata_host,
    )

    kwargs = dict(
        num_heads_q=heads,
        num_heads_kv=1,
        head_dim=512,
        layout_q=layout,
        layout_kv="PA_ND",
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        seqused_kv=torch.tensor([kv_len], dtype=torch.int32),
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_kv=kv_len,
        cmp_ratio=cmp_ratio,
        has_ori_kv=True,
        has_cmp_kv=has_cmp,
    )
    executor = check_accuracy(
        kv_quant_sparse_attn_sharedkv_metadata_host,
        kwargs=kwargs,
    )
    check_performance(
        executor,
        enable_perf=layout == "BSND" and heads == 64,
    )

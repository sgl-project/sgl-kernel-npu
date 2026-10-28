"""Public-API registration, accuracy and performance checks for A5 paged SWA."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Production-style decode plus a multi-request and prefill scheduling case.
perf_shape = [(1, 1, 64, 256)]


@pytest.fixture(autouse=True)
def require_ascend950(configure_npu):
    device_index = torch.npu.current_device()
    if "950" not in torch.npu.get_device_name(device_index).lower():
        pytest.skip("quantized shared-KV attention requires Ascend 950")


def test_public_registration():
    from sgl_kernel_npu.operators import attention
    from sgl_kernel_npu.operators.utils.registry import get_op_name, list_impls

    operator = attention.kv_quant_sparse_attn_sharedkv
    assert get_op_name(operator) == "attention.kv_quant_sparse_attn_sharedkv"
    assert set(list_impls(operator)) == {"auto", "ascendc", "torch_native"}
    assert hasattr(torch.ops.npu, "kv_quant_sparse_attn_sharedkv")


def generate_test_data(batch, query_len, heads, kv_len, seed=42):
    """Create packed FP8 KV with RoPE BF16, NoPE FP8 and seven E8M0 scales."""
    torch.manual_seed(seed)
    block_size = 16
    block_count = max(20, math.ceil(kv_len / block_size))
    rope = torch.randn(block_count, block_size, 64).to(torch.bfloat16)
    nope = torch.randn(block_count, block_size, 448).to(torch.float8_e4m3fn)
    packed = torch.zeros((block_count, block_size, 1, 640), dtype=torch.uint8)
    packed[..., :128] = rope.contiguous().view(torch.uint8).unsqueeze(-2)
    packed[..., 128:576] = nope.contiguous().view(torch.uint8).unsqueeze(-2)
    packed[..., 576:583] = 127  # E8M0 scale 1.0 for each NoPE group.
    table = (
        torch.arange(math.ceil(kv_len / block_size), dtype=torch.int32)
        .roll(3)
        .expand(batch, -1)
        .contiguous()
    )
    lengths_cpu = torch.full((batch,), kv_len, dtype=torch.int32)
    q = (torch.randn(batch, query_len, heads, 512) * 0.1).to(torch.bfloat16)
    return (
        q.to("npu"),
        packed.view(torch.float8_e4m3fn).to("npu"),
        table.to("npu"),
        lengths_cpu,
    )


@pytest.mark.parametrize(
    ("batch", "query_len", "heads", "kv_len", "window"),
    [
        pytest.param(1, 1, 64, 256, 127, id="decode-64-heads"),
        pytest.param(2, 1, 128, 256, 127, id="two-requests-128-heads"),
        pytest.param(1, 2, 64, 256, 0, id="prefill-zero-left-window"),
    ],
)
def test_paged_swa(batch, query_len, heads, kv_len, window):
    from sgl_kernel_npu.operators.attention import (
        kv_quant_sparse_attn_sharedkv,
        kv_quant_sparse_attn_sharedkv_metadata_host,
    )

    q, packed, table, lengths_cpu = generate_test_data(batch, query_len, heads, kv_len)
    metadata = kv_quant_sparse_attn_sharedkv_metadata_host(
        num_heads_q=heads,
        num_heads_kv=1,
        head_dim=512,
        layout_q="BSND",
        layout_kv="PA_ND",
        seqused_kv=lengths_cpu,
        batch_size=batch,
        max_seqlen_q=query_len,
        max_seqlen_kv=kv_len,
        ori_win_left=window,
        has_ori_kv=True,
        has_cmp_kv=False,
    )
    kwargs = dict(
        ori_kv=packed,
        ori_block_table=table,
        seqused_kv=lengths_cpu.to("npu"),
        sinks=torch.full((heads,), -1e4, dtype=torch.float32, device="npu"),
        metadata=metadata,
        softmax_scale=1 / math.sqrt(512),
        ori_win_left=window,
    )
    executor = check_accuracy(
        kv_quant_sparse_attn_sharedkv,
        args=(q,),
        kwargs=kwargs,
    )
    check_performance(
        executor,
        enable_perf=(batch, query_len, heads, kv_len) in perf_shape,
    )

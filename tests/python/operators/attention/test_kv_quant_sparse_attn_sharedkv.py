"""DeepSeek-V4-Flash DSpArk paged attention through the public API."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# DeepSeek-V4-Flash uses D=512, one KV head and 64-head padding on
# sharded DSpArk queries. 128 heads exercises the other supported kernel shape.
# These are synthetic decode/prefill fixtures, not captured production batches.
# Page size 128 matches the end-to-end launch configuration; 16 is an edge case.
perf_shape = [(1, 1, 64, 256, 128)]


def generate_test_data(batch, query_len, heads, kv_len, block_size, seed=42):
    """Create packed FP8 KV with RoPE BF16, NoPE FP8 and seven E8M0 scales."""
    torch.manual_seed(seed)
    block_count = max(20, math.ceil(kv_len / block_size))
    rope = torch.randn(block_count, block_size, 64, device="cpu").to(torch.bfloat16)
    nope = torch.randn(block_count, block_size, 448, device="cpu").to(
        torch.float8_e4m3fn
    )
    packed = torch.zeros(
        (block_count, block_size, 1, 640), dtype=torch.uint8, device="cpu"
    )
    packed[..., :128] = rope.contiguous().view(torch.uint8).unsqueeze(-2)
    packed[..., 128:576] = nope.contiguous().view(torch.uint8).unsqueeze(-2)
    packed[..., 576:583] = 127  # E8M0 scale 1.0 for each NoPE group.
    table = (
        torch.arange(math.ceil(kv_len / block_size), dtype=torch.int32, device="cpu")
        .roll(3)
        .expand(batch, -1)
        .contiguous()
    )
    lengths_cpu = torch.full((batch,), kv_len, dtype=torch.int32, device="cpu")
    q = (torch.randn(batch, query_len, heads, 512, device="cpu") * 0.1).to(
        torch.bfloat16
    )
    return (
        q.to("npu"),
        packed.view(torch.float8_e4m3fn).to("npu"),
        table.to("npu"),
        lengths_cpu,
    )


@pytest.mark.parametrize(
    ("batch", "query_len", "heads", "kv_len", "window", "block_size"),
    [
        pytest.param(1, 1, 64, 256, 127, 128, id="dsv4-flash-dspark-decode-page128"),
        pytest.param(1, 1, 64, 256, 127, 16, id="edge-page16"),
        pytest.param(2, 1, 128, 256, 127, 128, id="dsv4-two-requests-128-heads"),
        pytest.param(1, 2, 64, 256, 0, 128, id="edge-two-token-zero-window"),
    ],
)
def test_paged_swa(batch, query_len, heads, kv_len, window, block_size):
    from sgl_kernel_npu.operators.attention import (
        kv_quant_sparse_attn_sharedkv,
        kv_quant_sparse_attn_sharedkv_metadata_host,
    )

    q, packed, table, lengths_cpu = generate_test_data(
        batch, query_len, heads, kv_len, block_size
    )
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
        enable_perf=(batch, query_len, heads, kv_len, block_size) in perf_shape,
    )

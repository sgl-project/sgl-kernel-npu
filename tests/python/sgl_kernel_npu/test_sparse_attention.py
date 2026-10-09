import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

import sgl_kernel_npu  # noqa: F401

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu")
    or not torch.npu.is_available()
    or not hasattr(torch.ops.npu, "ada_block_sparse_attention"),
    reason="Sparse attention requires a build for Ascend A2/A3 and an NPU",
)


def as_layout(x, layout):
    if layout == "BNSD":
        return x
    x = x.transpose(1, 2).contiguous()
    return x if layout == "BSND" else x.flatten(2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["BNSD", "BSND", "BSH"])
@pytest.mark.parametrize("causal,kv_heads", [(False, 2), (True, 1)])
def test_ada_matches_masked_reference(dtype, layout, causal, kv_heads):
    torch.manual_seed(42)
    batch, heads, seq, dim, block = 2, 2, 2048, 128, 128
    q = torch.randn(batch, heads, seq, dim, dtype=dtype)
    k = torch.randn(batch, kv_heads, seq, dim, dtype=dtype)
    v = torch.randn_like(k)
    blocks = seq // block
    selected = torch.rand(batch, heads, blocks, blocks) > 0.5
    selected |= torch.eye(blocks, dtype=torch.bool)
    if causal:
        selected &= torch.ones(blocks, blocks, dtype=torch.bool).tril()
    mask = torch.zeros(batch, heads, blocks, 32, dtype=torch.int8)
    mask[..., :blocks] = selected
    count = selected.sum(-1).to(torch.int32)
    scale = 1 / math.sqrt(dim)
    result = torch.ops.npu.ada_block_sparse_attention(
        as_layout(q, layout).npu(),
        as_layout(k, layout).npu(),
        as_layout(v, layout).npu(),
        mask.npu(),
        count.npu(),
        input_layout=layout,
        num_heads=heads,
        num_key_value_heads=kv_heads,
        scale_value=scale,
        causal=causal,
        actual_seq_lengths=[seq, seq],
        actual_seq_lengths_kv=[seq],
    ).cpu()
    allowed = selected.repeat_interleave(block, -2).repeat_interleave(block, -1)
    if causal:
        allowed &= torch.ones(seq, seq, dtype=torch.bool).tril()
    scores = q.float() @ k.float().repeat_interleave(heads // kv_heads, 1).transpose(
        -1, -2
    )
    scores = (scores * scale).masked_fill(~allowed, -torch.inf)
    expected = scores.softmax(-1) @ v.float().repeat_interleave(heads // kv_heads, 1)
    assert result.dtype == dtype
    torch.testing.assert_close(
        result.float(), as_layout(expected, layout), atol=3e-2, rtol=3e-2
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["BNSD", "BSND", "BSH"])
def test_estimator_mask_and_call_order(dtype, layout):
    torch.manual_seed(123)
    heads, dim, seq = 2, 128, 4096
    q = torch.randn(1, heads, seq, dim, dtype=dtype).npu()
    k = torch.randn_like(q)

    def estimate(q, k):
        return torch.ops.npu.sparse_block_estimate(
            as_layout(q, layout),
            as_layout(k, layout),
            input_layout=layout,
            num_heads=heads,
            num_key_value_heads=heads,
            scale_value=1 / math.sqrt(dim),
            causal=True,
        )

    mask, counts = estimate(q, k)
    estimate(q[:, :, :1024], k[:, :, :1024])
    repeated_mask, repeated_counts = estimate(q, k)
    torch.testing.assert_close(mask, repeated_mask)
    torch.testing.assert_close(counts, repeated_counts)
    assert mask.dtype == torch.int8
    assert counts.dtype == torch.int32
    assert tuple(mask.shape) == (1, heads, seq // 128, seq // 128)
    assert torch.all((mask == 0) | (mask == 1))
    torch.testing.assert_close(mask.sum(-1).to(torch.int32), counts)
    assert torch.all(mask.diagonal(dim1=-2, dim2=-1) == 1)
    assert torch.count_nonzero(mask.triu(1)) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["BNSD", "BSND", "BSH"])
def test_estimator_and_ada_dense_pipeline(dtype, layout):
    """Exercise both launches and compare their combined output to dense attention."""
    torch.manual_seed(321)
    batch, heads, seq, dim, block, stride = 1, 2, 2048, 128, 128, 8
    q, k, v = [torch.randn(batch, heads, seq, dim, dtype=dtype) for _ in range(3)]
    q_npu, k_npu, v_npu = [as_layout(x, layout).npu() for x in (q, k, v)]
    scale = 1 / math.sqrt(dim)
    mask, counts = torch.ops.npu.sparse_block_estimate(
        q_npu,
        k_npu,
        input_layout=layout,
        num_heads=heads,
        num_key_value_heads=heads,
        sparse_size=block,
        stride=stride,
        scale_value=scale / stride,
        threshold=1.0,
        row_sparse=1.0,
        causal=False,
    )
    # Synchronize each launch so an estimator failure cannot be attributed to Ada.
    torch.npu.synchronize()
    blocks = seq // block
    mask_cpu, counts_cpu = mask.cpu(), counts.cpu()
    assert torch.all(mask_cpu[..., :blocks] == 1)
    assert torch.count_nonzero(mask_cpu[..., blocks:]) == 0
    assert torch.all(counts_cpu == blocks)
    result = torch.ops.npu.ada_block_sparse_attention(
        q_npu,
        k_npu,
        v_npu,
        mask,
        counts,
        input_layout=layout,
        num_heads=heads,
        num_key_value_heads=heads,
        sparse_size=block,
        scale_value=scale,
        causal=False,
    )
    torch.npu.synchronize()
    result = result.cpu()
    assert torch.isfinite(result).all()
    expected = ((q.float() @ k.float().transpose(-2, -1)) * scale).softmax(-1)
    expected = expected @ v.float()
    torch.testing.assert_close(
        result.float(), as_layout(expected, layout), atol=3e-2, rtol=3e-2
    )

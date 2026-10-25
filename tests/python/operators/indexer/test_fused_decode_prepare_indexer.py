import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.standards import AccuracyStandard

BF16_GEMM_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=1e-2, rtol=1e-2),
}


def _fused_decode_prepare_indexer_torch(
    x,
    weights_proj_weight,
    seqlens_32,
    block_tables,
    pool_size,
    n_heads_pow,
    softmax_scale,
):
    """Independent Torch reference, including right-side page-table padding."""
    w = ((x.float() @ weights_proj_weight.T) * (n_heads_pow * softmax_scale)).to(
        torch.bfloat16
    )
    pool_seqlens = torch.div(seqlens_32, pool_size, rounding_mode="floor").to(
        torch.int32
    )

    batch_size, num_blocks = block_tables.shape
    padded_blocks = -(-num_blocks // pool_size) * pool_size or pool_size
    dummy = torch.zeros(
        (batch_size, padded_blocks - num_blocks), dtype=block_tables.dtype
    )
    padded = torch.cat((block_tables, dummy), dim=-1)
    pool_block_tables = padded.view(batch_size, padded_blocks // pool_size, pool_size)[
        :, :, 0
    ].contiguous()
    return w, pool_seqlens, pool_block_tables


@pytest.mark.parametrize(
    "num_blocks", [5, 129], ids=["single-vv-tile", "cross-vv-tile"]
)
def test_fused_decode_prepare_indexer(num_blocks):
    from sgl_kernel_npu.operators.indexer import fused_decode_prepare_indexer

    torch.manual_seed(1234)
    batch_size = 33
    hidden_size = 256
    n_heads = 32
    pool_size = 4
    n_heads_pow = n_heads**-0.5
    softmax_scale = 128**-0.5

    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)
    weights_proj_weight = torch.randn(n_heads, hidden_size, dtype=torch.float32)
    seqlens_32 = torch.arange(batch_size, dtype=torch.int32) % 8 + 3
    block_tables = torch.arange(batch_size * num_blocks, dtype=torch.int32).view(
        batch_size, num_blocks
    )

    actual = fused_decode_prepare_indexer(
        x,
        weights_proj_weight,
        seqlens_32,
        block_tables,
        pool_size=pool_size,
        n_heads_pow=n_heads_pow,
        softmax_scale=softmax_scale,
    )
    expected = _fused_decode_prepare_indexer_torch(
        x,
        weights_proj_weight,
        seqlens_32,
        block_tables,
        pool_size,
        n_heads_pow,
        softmax_scale,
    )

    # BF16 GEMM uses an explicit tolerance; both integer outputs are exact.
    assert_accuracy(actual, expected, BF16_GEMM_STANDARD)

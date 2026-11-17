import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# P/C: hidden=4096 and heads=32; W/T: batch and page-table widths.
perf_shape = [(16, 4096, 32, 64)]

BF16_GEMM_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=1e-2, rtol=1e-2),
}


def generate_test_data(batch_size, hidden_size, n_heads, num_blocks):
    torch.manual_seed(1234)
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)
    weights_proj_weight = torch.randn(n_heads, hidden_size, dtype=torch.float32)
    seqlens_32 = torch.arange(batch_size, dtype=torch.int32) % 8 + 3
    block_tables = torch.arange(batch_size * num_blocks, dtype=torch.int32).view(
        batch_size, num_blocks
    )
    return x, weights_proj_weight, seqlens_32, block_tables


@pytest.mark.parametrize(
    "batch_size,hidden_size,n_heads,num_blocks",
    [
        # Case 01: Cross the vector tile and retain a page-table grouping tail.
        pytest.param(33, 256, 32, 129, id="cross-vv-tile"),
        # Case 02: Minimum batch, single head and single page-table element.
        pytest.param(1, 256, 1, 1, id="single-block-single-head"),
        # Case 03: Hidden512 and an exact page-table group.
        pytest.param(2, 512, 16, 4, id="exact-pool-hidden512"),
        # Case 04: Multiple batch tiles and a partial page-table group.
        pytest.param(65, 512, 8, 7, id="multi-batch-nonmultiple-pool"),
        # Case 05: Typical B16/H4096/32 heads/M64; also profile.
        pytest.param(16, 4096, 32, 64, id="T-glm53-flash-b16-m64"),
        # Case 06: 64 heads use B32/K256; cover batch, K and page-table tails.
        pytest.param(33, 384, 64, 7, id="heads64-batch-and-k-tail"),
        # Case 07: 128 heads use B16/K128; cover batch, K and page-table tails.
        pytest.param(17, 192, 128, 9, id="heads128-batch-and-k-tail"),
    ],
)
def test_kpool_indexer_prepare_decode_input(
    batch_size, hidden_size, n_heads, num_blocks
):
    from sgl_kernel_npu.operators.indexer import kpool_indexer_prepare_decode_input

    pool_size = 4
    n_heads_pow = n_heads**-0.5
    softmax_scale = 128**-0.5
    x, weights_proj_weight, seqlens_32, block_tables = generate_test_data(
        batch_size, hidden_size, n_heads, num_blocks
    )

    executor = check_accuracy(
        kpool_indexer_prepare_decode_input,
        args=(x, weights_proj_weight, seqlens_32, block_tables),
        kwargs={
            "pool_size": pool_size,
            "n_heads_pow": n_heads_pow,
            "softmax_scale": softmax_scale,
        },
        standard=BF16_GEMM_STANDARD,
    )
    shape = (batch_size, hidden_size, n_heads, num_blocks)
    check_performance(executor, enable_perf=shape in perf_shape)


# Case 08: Reject noncontiguous block tables.
@pytest.mark.extra_checks
def test_fused_decode_prepare_rejects_noncontiguous_block_tables():
    from sgl_kernel_npu.operators.indexer import kpool_indexer_prepare_decode_input

    x = torch.randn(1, 256, dtype=torch.bfloat16)
    weights = torch.randn(32, 256, dtype=torch.float32)
    seqlens = torch.tensor([4], dtype=torch.int32)
    base = torch.arange(4, dtype=torch.int32).view(1, 4)
    block_tables = torch.stack((base, base), dim=-1)[..., 0]
    assert not block_tables.is_contiguous()
    with pytest.raises(AssertionError, match="block_tables must be contiguous"):
        kpool_indexer_prepare_decode_input(x, weights, seqlens, block_tables)

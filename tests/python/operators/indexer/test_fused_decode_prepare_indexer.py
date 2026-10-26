import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# P/C: hidden=4096 and heads=32; W/T: batch and page-table widths.
perf_shape = [(16, 4096, 32, 64), (64, 4096, 32, 129)]

BF16_GEMM_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=1e-2, rtol=1e-2),
}


@pytest.mark.parametrize(
    "batch_size,hidden_size,n_heads,num_blocks",
    [
        pytest.param(33, 256, 32, 5, id="single-vv-tile"),
        pytest.param(33, 256, 32, 129, id="cross-vv-tile"),
        pytest.param(1, 256, 1, 1, id="single-block-single-head"),
        pytest.param(2, 512, 16, 4, id="exact-pool-hidden512"),
        pytest.param(65, 512, 8, 7, id="multi-batch-nonmultiple-pool"),
        pytest.param(16, 4096, 32, 64, id="T-glm53-flash-b16-m64"),
        pytest.param(64, 4096, 32, 129, id="W-glm53-flash-b64-m129"),
    ],
)
def test_fused_decode_prepare_indexer(batch_size, hidden_size, n_heads, num_blocks):
    from sgl_kernel_npu.operators.indexer import fused_decode_prepare_indexer

    torch.manual_seed(1234)
    pool_size = 4
    n_heads_pow = n_heads**-0.5
    softmax_scale = 128**-0.5

    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)
    weights_proj_weight = torch.randn(n_heads, hidden_size, dtype=torch.float32)
    seqlens_32 = torch.arange(batch_size, dtype=torch.int32) % 8 + 3
    block_tables = torch.arange(batch_size * num_blocks, dtype=torch.int32).view(
        batch_size, num_blocks
    )

    executor = check_accuracy(
        fused_decode_prepare_indexer,
        args=(x, weights_proj_weight, seqlens_32, block_tables),
        kwargs={
            "pool_size": pool_size,
            "n_heads_pow": n_heads_pow,
            "softmax_scale": softmax_scale,
        },
        impl="triton",
        impl_ref="torch_native",
        standard=BF16_GEMM_STANDARD,
    )
    shape = (batch_size, hidden_size, n_heads, num_blocks)
    check_performance(executor, enable_perf=shape in perf_shape)


def test_fused_decode_prepare_rejects_noncontiguous_block_tables():
    from sgl_kernel_npu.operators.indexer import fused_decode_prepare_indexer

    x = torch.randn(1, 256, dtype=torch.bfloat16)
    weights = torch.randn(32, 256, dtype=torch.float32)
    seqlens = torch.tensor([4], dtype=torch.int32)
    base = torch.arange(4, dtype=torch.int32).view(1, 4)
    block_tables = torch.stack((base, base), dim=-1)[..., 0]
    assert not block_tables.is_contiguous()
    with pytest.raises(AssertionError, match="block_tables must be contiguous"):
        fused_decode_prepare_indexer(x, weights, seqlens, block_tables)

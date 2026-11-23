# In-place Operator
"""mxFP8 pooled-key writes: dequantized cache must track the BF16 reference."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_test_impls
from tests.python.utils.standards import AccuracyStandard

HEAD_DIM = 128
MX_BLOCK = 32
E8M0_BIAS = 127
# E4M3FN keeps only three mantissa bits per 32-channel block, so the dequantized
# key is coarser than INT8; allow a looser tolerance than the INT8 reference.
MXFP8_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=1e-1, rtol=1e-1),
}


def _seed(seed):
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed(seed)


def dequantize(cache, scale_bytes):
    """Expand an mxFP8 cache with its per-32-block E8M0 scales for comparison."""
    payload = cache.float().reshape(-1, HEAD_DIM // MX_BLOCK, MX_BLOCK)
    scales = torch.pow(
        2.0, scale_bytes.reshape(-1, HEAD_DIM // MX_BLOCK).float() - E8M0_BIAS
    )
    return (payload * scales.unsqueeze(-1)).reshape(cache.shape).to(torch.bfloat16)


def test_assemble_mxfp8_cache_tracks_bf16():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_assemble_softmax_rotate_write_cache,
    )

    _seed(0)
    impl, impl_ref = get_test_impls()
    pool_size, n_pools, tail_size = 4, 3, 8
    chunk_k = torch.randn(n_pools * pool_size, HEAD_DIM).bfloat16()
    chunk_score = torch.randn(n_pools * pool_size, HEAD_DIM)
    tail_k = torch.randn(2, tail_size, HEAD_DIM).bfloat16()
    tail_score = torch.randn(2, tail_size, HEAD_DIM)
    ape = torch.randn(pool_size, HEAD_DIM) * 2
    loc = torch.tensor([0, 1, 5], dtype=torch.int64)
    plan = dict(
        req_pool_idx=torch.tensor([0, 1, 0], dtype=torch.int64),
        n_from_tail=torch.tensor([1, 2, 3], dtype=torch.int32),
        chunk_src_start=torch.tensor([0, 4, 8], dtype=torch.int64),
        tail_logical_base=torch.tensor([0, 3, 5], dtype=torch.int32),
        loc=loc,
    )

    bf16_cache = torch.zeros(8, HEAD_DIM, dtype=torch.bfloat16)
    kpool_assemble_softmax_rotate_write_cache(
        bf16_cache,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        ape=ape,
        impl=impl_ref,
        **plan,
    )

    mx_cache = torch.zeros(8, HEAD_DIM, dtype=torch.float8_e4m3fn)
    scale = torch.zeros(8, HEAD_DIM // 64, 2, dtype=torch.uint8)
    kpool_assemble_softmax_rotate_write_cache(
        mx_cache,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        ape=ape,
        scale_cache=scale,
        impl=impl,
        **plan,
    )

    assert_accuracy(
        dequantize(mx_cache, scale),
        bf16_cache,
        MXFP8_STANDARD,
        path="index_k_cache",
    )
    assert torch.all(scale[loc] > 0)


def test_decode_mxfp8_cache_tracks_bf16():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_decode_update_and_maybe_write_cache,
    )

    _seed(1)
    impl, impl_ref = get_test_impls()
    pool_size, slots_per_page, tail_size = 4, 2, 8
    tail_k = torch.randn(1, tail_size, HEAD_DIM).bfloat16()
    tail_score = torch.randn(1, tail_size, HEAD_DIM)
    key = torch.randn(1, HEAD_DIM).bfloat16()
    slot_score = torch.randn(1, HEAD_DIM)
    ape = torch.randn(pool_size, HEAD_DIM) * 2
    block_tables = torch.zeros(1, 8, dtype=torch.int32)
    # The completed pool must be found in the first table block.
    block_tables[0, 0] = pool_size
    page_rows = slots_per_page * 4
    cache_rows = page_rows * slots_per_page
    common = dict(
        tail_k=tail_k,
        tail_score=tail_score,
        key=key,
        slot_score=slot_score,
        ape=ape,
        block_tables=block_tables,
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        positions=torch.tensor([pool_size - 1], dtype=torch.int64),
        seq_lens=torch.tensor([pool_size], dtype=torch.int64),
        out_cache_loc=torch.tensor([1], dtype=torch.int64),
        pool_size=pool_size,
        slots_per_page=slots_per_page,
    )

    bf16_cache = torch.zeros(page_rows, slots_per_page * HEAD_DIM, dtype=torch.bfloat16)
    kpool_decode_update_and_maybe_write_cache(buf=bf16_cache, impl=impl_ref, **common)

    mx_cache = torch.zeros(
        page_rows, slots_per_page * HEAD_DIM, dtype=torch.float8_e4m3fn
    )
    scale = torch.zeros(cache_rows, HEAD_DIM // 64, 2, dtype=torch.uint8)
    kpool_decode_update_and_maybe_write_cache(
        buf=mx_cache, scale_cache=scale, impl=impl, **common
    )

    assert_accuracy(
        dequantize(mx_cache, scale),
        bf16_cache,
        MXFP8_STANDARD,
        path="index_k_cache",
    )
    # The pool that completes in this step stores nonzero E8M0 scale bytes.
    assert torch.all(scale[slots_per_page] > 0)


def test_mtp_mxfp8_cache_tracks_bf16():
    from sgl_kernel_npu.operators.mem_cache import kpool_write_tail_and_maybe_compress

    _seed(2)
    impl, impl_ref = get_test_impls()
    pool_size, num_draft_tokens = 4, 4
    tail_size = pool_size + num_draft_tokens
    key = torch.randn(num_draft_tokens, HEAD_DIM).bfloat16()
    score = torch.randn(num_draft_tokens, HEAD_DIM)
    ape = torch.randn(pool_size, HEAD_DIM) * 2
    common = dict(
        key=key,
        score=score,
        tail_k=torch.zeros(1, tail_size, HEAD_DIM, dtype=torch.bfloat16),
        tail_score=torch.zeros(1, tail_size, HEAD_DIM),
        ape=ape,
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        write_start=torch.tensor([0], dtype=torch.int32),
        tail_logical_start=torch.tensor([0], dtype=torch.int32),
        write_loc=torch.tensor([[0]], dtype=torch.int64),
        out_cache_loc=torch.arange(1, num_draft_tokens + 1),
        num_draft_tokens=num_draft_tokens,
    )

    bf16_cache = torch.zeros(8, HEAD_DIM, dtype=torch.bfloat16)
    kpool_write_tail_and_maybe_compress(bf16_cache, impl=impl_ref, **common)

    mx_cache = torch.zeros(8, HEAD_DIM, dtype=torch.float8_e4m3fn)
    scale = torch.zeros(8, HEAD_DIM // 64, 2, dtype=torch.uint8)
    kpool_write_tail_and_maybe_compress(
        mx_cache, scale_cache=scale, impl=impl, **common
    )

    assert_accuracy(
        dequantize(mx_cache, scale),
        bf16_cache,
        MXFP8_STANDARD,
        path="index_k_cache",
    )
    # Draft tokens close the pool and compress it into cache row 0.
    assert torch.all(scale[0] > 0)

# In-place Operator
"""INT8 pooled-key writes: dequantized cache must track the BF16 reference."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_test_impls
from tests.python.utils.standards import AccuracyStandard

HEAD_DIM = 128
# One INT8 step is amax / 127 over the pooled row, so the dequantized key stays
# within a few quantization steps of the BF16 pooled key for these inputs.
INT8_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=5e-2, rtol=5e-2),
}


def dequantize(cache, scale):
    """Expand an INT8 cache with its per-row FP16 scales for comparison."""
    return (cache.float() * scale.float().unsqueeze(-1)).to(torch.bfloat16)


def test_assemble_int8_cache_tracks_bf16():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_assemble_softmax_rotate_write_cache,
    )

    torch.manual_seed(0)
    torch.npu.manual_seed(0)
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

    int8_cache = torch.zeros(8, HEAD_DIM, dtype=torch.int8)
    scale = torch.zeros(8, dtype=torch.float16)
    kpool_assemble_softmax_rotate_write_cache(
        int8_cache,
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
        dequantize(int8_cache, scale),
        bf16_cache,
        INT8_STANDARD,
        path="index_k_cache",
    )
    assert torch.all(scale[loc] > 0)


def test_decode_int8_cache_tracks_bf16():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_decode_update_and_maybe_write_cache,
    )

    torch.manual_seed(1)
    torch.npu.manual_seed(1)
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

    int8_cache = torch.zeros(page_rows, slots_per_page * HEAD_DIM, dtype=torch.int8)
    scale = torch.zeros(page_rows, dtype=torch.float16)
    kpool_decode_update_and_maybe_write_cache(
        buf=int8_cache, scale_cache=scale, impl=impl, **common
    )

    assert_accuracy(
        dequantize(int8_cache, scale),
        bf16_cache,
        INT8_STANDARD,
        path="index_k_cache",
    )
    # The pool that completes in this step stores a positive scale.
    assert scale[slots_per_page] > 0


def test_mtp_int8_cache_tracks_bf16():
    from sgl_kernel_npu.operators.mem_cache import kpool_write_tail_and_maybe_compress

    torch.manual_seed(2)
    torch.npu.manual_seed(2)
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

    int8_cache = torch.zeros(8, HEAD_DIM, dtype=torch.int8)
    scale = torch.zeros(8, dtype=torch.float16)
    kpool_write_tail_and_maybe_compress(
        int8_cache, scale_cache=scale, impl=impl, **common
    )

    assert_accuracy(
        dequantize(int8_cache, scale),
        bf16_cache,
        INT8_STANDARD,
        path="index_k_cache",
    )
    # Draft tokens close the pool and compress it into cache row 0.
    assert scale[0] > 0

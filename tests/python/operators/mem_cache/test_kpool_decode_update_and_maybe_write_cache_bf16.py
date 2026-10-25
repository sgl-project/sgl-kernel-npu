import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.standards import AccuracyStandard

BUF_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=2e-2, rtol=1e-5),
}
TAIL_K_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=1e-3, rtol=1e-5),
}
TAIL_SCORE_STANDARD = {
    "torch.float32": AccuracyStandard(atol=1e-6, rtol=1e-5),
}
EXACT_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0, rtol=0),
    "torch.float32": AccuracyStandard(atol=0, rtol=0),
}


@torch.no_grad()
def kpool_decode_update_and_maybe_write_cache_bf16_torch(
    buf,
    tail_k,
    tail_score,
    key,
    slot_score,
    ape,
    block_tables,
    req_pool_indices,
    positions,
    seq_lens,
    out_cache_loc,
    *,
    pool_size,
    slots_per_page,
    head_dim=128,
):
    """Pure PyTorch reference with the original BF16 kernel semantics."""
    batch = key.shape[0]
    if batch == 0:
        return

    tail_size = tail_k.shape[1]
    for row in range(batch):
        req_raw = req_pool_indices[row].item()
        pos = positions[row].item()
        seq_len = seq_lens[row].item()
        cache_loc = out_cache_loc[row].item()

        req_valid = (req_raw >= 0) and (req_raw < tail_k.shape[0])
        pos_valid = req_valid and (cache_loc != 0) and (pos >= 0) and (pos < seq_len)
        req = max(0, min(req_raw, tail_k.shape[0] - 1))
        slot = pos % pool_size
        phys_slot = pos % tail_size
        cur_key = key[row].float()
        cur_score = slot_score[row].float()

        if pos_valid and slot == pool_size - 1:
            pool_logical_start = pos - slot
            scores_list = []
            keys_list = []
            for pool_slot in range(pool_size):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % tail_size
                if is_current:
                    score = cur_score
                    current_key = cur_key
                else:
                    score = tail_score[req, phys].float()
                    current_key = tail_k[req, phys].float()
                scores_list.append(score + ape[pool_slot].float())
                keys_list.append(current_key)

            all_scores = torch.stack(scores_list)
            all_keys = torch.stack(keys_list)
            max_score = all_scores.max(dim=0).values
            probs = torch.exp(all_scores - max_score.unsqueeze(0))
            denom = probs.sum(dim=0)
            acc = (all_keys * probs).sum(dim=0)
            compressed = (acc / denom).to(torch.bfloat16)

            pool_id = pos // pool_size
            pool_page_group = pool_id // slots_per_page
            token_page_row = pool_page_group * pool_size
            token_page_row = max(0, min(token_page_row, block_tables.shape[1] - 1))
            packed_page = block_tables[row, token_page_row].item()
            loc_token_offset = pool_id % slots_per_page
            out_offset = packed_page * buf.shape[1] + loc_token_offset * head_dim
            buf.view(-1)[out_offset : out_offset + head_dim] = compressed

        if pos_valid:
            tail_k[req, phys_slot] = key[row]
            tail_score[req, phys_slot] = slot_score[row]


def _assert_decode_outputs(actual, expected):
    actual_buf, actual_tail_k, actual_tail_score = actual
    expected_buf, expected_tail_k, expected_tail_score = expected
    assert_accuracy(actual_tail_k, expected_tail_k, TAIL_K_STANDARD, path="tail_k")
    assert_accuracy(
        actual_tail_score,
        expected_tail_score,
        TAIL_SCORE_STANDARD,
        path="tail_score",
    )
    assert_accuracy(actual_buf, expected_buf, BUF_STANDARD, path="buf")


def test_decode_cache_write():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_decode_update_and_maybe_write_cache_bf16,
    )

    torch.manual_seed(42)
    pool_size = 4
    tail_size = pool_size + 1
    head_dim = 128
    slots_per_page = 64
    batch = 16
    req_pool_size = batch
    num_pages = 32
    block_table_cols = 64

    buf_numel_per_page = slots_per_page * head_dim
    initial_buf = torch.zeros(num_pages, buf_numel_per_page, dtype=torch.bfloat16)
    buf = initial_buf.clone()
    buf_ref = initial_buf.clone()
    tail_k = torch.randn(req_pool_size, tail_size, head_dim, dtype=torch.bfloat16) * 0.1
    tail_k_ref = tail_k.clone()
    tail_score = (
        torch.randn(req_pool_size, tail_size, head_dim, dtype=torch.float32) * 0.1
    )
    tail_score_ref = tail_score.clone()
    key = torch.randn(batch, head_dim, dtype=torch.bfloat16) * 0.1
    slot_score = torch.randn(batch, head_dim, dtype=torch.float32) * 0.1
    ape = torch.randn(pool_size, head_dim, dtype=torch.float32) * 0.01
    block_tables = torch.randint(
        0, num_pages, (batch, block_table_cols), dtype=torch.int32
    )
    req_pool_indices = torch.arange(batch, dtype=torch.int32)
    positions = torch.tensor(
        [3, 7, 10, 15, 20, 23, 25, 27, 30, 31, 35, 38, 43, 44, 47, 50],
        dtype=torch.int32,
    )
    seq_lens = positions + 10
    out_cache_loc = torch.ones(batch, dtype=torch.int32)

    kpool_decode_update_and_maybe_write_cache_bf16(
        buf,
        tail_k,
        tail_score,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        pool_size=pool_size,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
    )
    kpool_decode_update_and_maybe_write_cache_bf16_torch(
        buf_ref,
        tail_k_ref,
        tail_score_ref,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        pool_size=pool_size,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
    )

    _assert_decode_outputs(
        (buf, tail_k, tail_score), (buf_ref, tail_k_ref, tail_score_ref)
    )


def test_invalid_cache_loc():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_decode_update_and_maybe_write_cache_bf16,
    )

    torch.manual_seed(99)
    pool_size = 4
    tail_size = 5
    head_dim = 128
    slots_per_page = 64
    batch = 4

    initial_buf = torch.zeros(8, slots_per_page * head_dim, dtype=torch.bfloat16)
    buf = initial_buf.clone()
    buf_ref = initial_buf.clone()
    tail_k = torch.randn(batch, tail_size, head_dim, dtype=torch.bfloat16) * 0.1
    tail_k_ref = tail_k.clone()
    tail_score = torch.randn(batch, tail_size, head_dim, dtype=torch.float32) * 0.1
    tail_score_ref = tail_score.clone()
    original_invalid_tail = tail_k[1].clone()
    original_invalid_score = tail_score[1].clone()

    key = torch.randn(batch, head_dim, dtype=torch.bfloat16) * 0.1
    slot_score = torch.randn(batch, head_dim, dtype=torch.float32) * 0.1
    ape = torch.randn(pool_size, head_dim, dtype=torch.float32) * 0.01
    block_tables = torch.zeros(batch, 64, dtype=torch.int32)
    block_tables[:, 0] = torch.arange(batch, dtype=torch.int32)
    req_pool_indices = torch.arange(batch, dtype=torch.int32)
    positions = torch.tensor([3, 5, 3, 7], dtype=torch.int32)
    seq_lens = torch.tensor([100, 100, 100, 100], dtype=torch.int32)
    out_cache_loc = torch.tensor([1, 0, 1, 1], dtype=torch.int32)

    kpool_decode_update_and_maybe_write_cache_bf16(
        buf,
        tail_k,
        tail_score,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        pool_size=pool_size,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
    )
    kpool_decode_update_and_maybe_write_cache_bf16_torch(
        buf_ref,
        tail_k_ref,
        tail_score_ref,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        pool_size=pool_size,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
    )

    _assert_decode_outputs(
        (buf, tail_k, tail_score), (buf_ref, tail_k_ref, tail_score_ref)
    )
    assert_accuracy(
        tail_k[1], original_invalid_tail, EXACT_STANDARD, path="invalid_tail_k"
    )
    assert_accuracy(
        tail_score[1],
        original_invalid_score,
        EXACT_STANDARD,
        path="invalid_tail_score",
    )

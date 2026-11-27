"""Independent PyTorch reference for Copy-SFA metadata."""

import torch


def prepare_copy_sfa_metadata(
    *,
    source_ids,
    valid_rows,
    sparse_rows,
    request_slots,
    seq_lens,
    actual_seq_lengths_kv,
    hbm_block_table,
    capacity,
    query_width=1,
):
    batch = valid_rows.numel()
    valid = valid_rows.bool()
    source_ids.view(batch, query_width, -1).masked_fill_(~valid[:, None, None], -1)
    pages = hbm_block_table.shape[1]
    base = torch.where(valid, request_slots, 0)
    hbm_block_table.copy_(
        base[:, None] * pages + torch.arange(pages, device=base.device)
    )
    actual_seq_lengths_kv.copy_(
        torch.where(
            valid,
            torch.where(sparse_rows.bool(), capacity + query_width - 1, seq_lens),
            0,
        )
    )

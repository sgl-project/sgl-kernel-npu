"""Independent torch reference for the LPLB LP-input builder."""

import torch


def lp_prep(
    A_full,
    b,
    t1,
    global_counts,
    log_single,
    log_replicated,
    B1,
    A_base_row_sum,
):
    """Build the IPM inputs in place and return ``(A_full, b, t1)``.

    Mirrors ``sglang/kernels/jit/csrc/lplb/lp_prep.cuh``: normalise the global
    counts, take the single/replicated shares, form the GPU row objective, and
    rewrite the last ``A_full`` column.
    """
    inv_total = 1.0 / torch.clamp(global_counts.sum(), min=1.0)
    t1_values = global_counts.index_select(0, log_single.long()) * inv_total
    b1_values = global_counts.index_select(0, log_replicated.long()) * inv_total
    b2_values = -(B1 @ t1_values)
    b_values = torch.cat((b1_values, b2_values))

    t1.copy_(t1_values)
    b.copy_(b_values)
    A_full[:, -1] = b_values - A_base_row_sum
    return A_full, b, t1

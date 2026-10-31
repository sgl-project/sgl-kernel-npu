"""Expand NPU pooled attention indices and append uncompressed tails."""

import torch
import triton
import triton.language as tl

from ..utils.triton_utils import get_device_properties

UB_SPECIFICATION = 192  # KiB of unified buffer per vector core.
# The live tiles may claim two thirds of it; the rest goes to pipeline stages and
# alignment padding that do not scale with the tile.
UB_TILE_BUDGET = UB_SPECIFICATION * 2 // 3  # KiB


def _choose_block_b(n_pool_topk: int, pool_size: int) -> int:
    # Once BLOCK_B > 1 the expanded tile is materialised four times per row (tmp,
    # transpose, reshape and the pipeline copy); BLOCK_B == 1 fuses the transpose
    # into the reshape and only keeps two.
    tile_bytes_per_row = 4 * n_pool_topk * pool_size * 4
    max_rows = max(1, UB_TILE_BUDGET * 1024 // tile_bytes_per_row)
    return min(8, 1 << (max_rows.bit_length() - 1))


@triton.jit
def _expand_kpool_group_and_tail_to_topk_kernel(
    pool_indices_ptr,
    seq_lens_per_token_ptr,
    out_ptr,
    stride_pi0: tl.constexpr,
    stride_pi1: tl.constexpr,
    stride_o0: tl.constexpr,
    stride_o1: tl.constexpr,
    n_real,
    N_POOL_TOPK: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    TAIL_LEN: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Expand pool-level topk indices to token-level and append tail tokens.

    Each program processes BLOCK_B rows at a time with a stride of
    NUM_CORES * BLOCK_B. Phase 1 expands a row block in parallel; phase 2
    preserves the per-row layout while unrolling with static_range.
    """
    pid = tl.program_id(0)

    for r0 in range(pid * BLOCK_B, n_real, NUM_CORES * BLOCK_B):
        row_offs = r0 + tl.arange(0, BLOCK_B)  # [BLOCK_B]
        row_mask = row_offs < n_real

        # Phase 1: expand a BLOCK_B batch from pool indices to token indices.
        p_offs = tl.arange(0, N_POOL_TOPK)
        pid_val = tl.load(
            pool_indices_ptr
            + row_offs[:, None] * stride_pi0
            + p_offs[None, :] * stride_pi1,
            mask=row_mask[:, None],
            other=0,
        ).to(
            tl.int32
        )  # [BLOCK_B, N_POOL_TOPK]

        base = pid_val * POOL_SIZE  # [BLOCK_B, N_POOL_TOPK]
        o_offs = tl.arange(0, POOL_SIZE)  # [POOL_SIZE]

        # [BLOCK_B, P, N] -> [BLOCK_B, N, P] -> [BLOCK_B, N * P]
        tmp = o_offs[None, :, None] + base[:, None, :]  # [BLOCK_B, P, N]
        tmp = tl.trans(tmp, [0, 2, 1])  # [BLOCK_B, N, P]
        result = tl.reshape(tmp, (BLOCK_B, N_POOL_TOPK * POOL_SIZE))

        col_offs = tl.arange(0, N_POOL_TOPK * POOL_SIZE)
        tl.store(
            out_ptr + row_offs[:, None] * stride_o0 + col_offs[None, :] * stride_o1,
            result,
            mask=row_mask[:, None],
        )

        # Phase 2: append each row's uncompressed tail.
        for m in tl.static_range(BLOCK_B):
            r = r0 + m
            valid = r < n_real
            seq_len = tl.load(
                seq_lens_per_token_ptr + r,
                mask=valid,
                other=0,
            ).to(tl.int32)
            pool_len = seq_len // POOL_SIZE
            tail_start = pool_len * POOL_SIZE
            tail_count = seq_len - tail_start

            t_offs = tl.arange(0, TAIL_LEN)
            tail_vals = tl.where(t_offs < tail_count, tail_start + t_offs, -1)
            tl.store(
                out_ptr + r * stride_o0 + TOPK * stride_o1 + t_offs * stride_o1,
                tail_vals,
                mask=valid,
            )


def _expand_kpool_group_and_tail_to_topk(
    pool_indices: torch.Tensor,
    seq_lens_per_token: torch.Tensor,
    n_real: int,
    num_q_padded: int,
    pool_size: int,
    index_topk: int,
) -> torch.Tensor:
    """Expand NPU pool-level top-k indices and append uncompressed tails.

    Args:
        pool_indices: Selected pool positions, shaped [>=n_real, n_pool_topk].
        seq_lens_per_token: Request length for each query, shaped [>=n_real].
        n_real: Number of valid query rows; later rows are masked.
        num_q_padded: Output row count; padded rows contain -1.
        pool_size: Number of tokens represented by one pool.
        index_topk: Token-level top-k, equal to n_pool_topk * pool_size.

    Returns:
        Token indices shaped [num_q_padded, index_topk + pool_size - 1].
    """
    n_pool_topk = pool_indices.shape[1]
    # compiler will pad the short axis in transpose
    assert pool_size >= 4, "pool_size must be >= 4"
    tail_len = pool_size - 1
    out_cols = index_topk + tail_len
    # Align rows to eight int32 values (32 bytes).
    out_cols_pad = (out_cols + 7) // 8 * 8

    # Let the kernel mask rows beyond n_real so callers need not slice inputs.
    n_real = min(n_real, pool_indices.shape[0], seq_lens_per_token.shape[0])

    assert pool_indices.ndim == 2
    assert pool_indices.shape[0] >= n_real
    assert seq_lens_per_token.ndim == 1
    assert seq_lens_per_token.shape[0] >= n_real
    assert num_q_padded >= n_real
    assert pool_indices.dtype == torch.int32
    assert seq_lens_per_token.dtype == torch.int32
    assert seq_lens_per_token.is_contiguous(), "seq_lens_per_token must be contiguous"
    assert n_pool_topk * pool_size == index_topk

    # Prefill padding rows with -1 and align the physical column count.
    out = torch.full(
        (num_q_padded, out_cols_pad), -1, dtype=torch.int32, device=pool_indices.device
    )

    assert out.stride(1) == 1

    if n_real > 0:
        block_b = _choose_block_b(n_pool_topk, pool_size)
        num_cores = min(triton.cdiv(n_real, block_b), get_device_properties()[1])
        _expand_kpool_group_and_tail_to_topk_kernel[(num_cores,)](
            pool_indices,
            seq_lens_per_token,
            out,
            pool_indices.stride(0),
            pool_indices.stride(1),
            out.stride(0),
            out.stride(1),
            n_real,
            N_POOL_TOPK=n_pool_topk,
            POOL_SIZE=pool_size,
            TOPK=index_topk,
            TAIL_LEN=tail_len,
            NUM_CORES=num_cores,
            BLOCK_B=block_b,
        )

    return out[:, :out_cols]

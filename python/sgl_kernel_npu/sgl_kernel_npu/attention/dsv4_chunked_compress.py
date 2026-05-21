import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _dsv4_chunked_prefill_compress_kernel(
    chunk_kv,
    chunk_score,
    state_kv_score,
    page_table,
    ape,
    out,
    PREFIX_LEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COFF_D: tl.constexpr,
    BLOCK_SRC: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_OUT: tl.constexpr,
    NUM_COL_BLOCKS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    n_src = 2 * RATIO if OVERLAP else RATIO
    first_k = PREFIX_LEN // RATIO
    num_tasks = N_OUT * NUM_COL_BLOCKS

    for task in tl.range(pid, num_tasks, NUM_PROGRAMS):
        out_i = task // NUM_COL_BLOCKS
        col_block = task - out_i * NUM_COL_BLOCKS
        k = first_k + out_i
        head_cols = col_block * BLOCK_D + cols
        col_mask = head_cols < HEAD_DIM

        acc = tl.full((BLOCK_D,), 0.0, tl.float32)
        denom = tl.full((BLOCK_D,), 0.0, tl.float32)
        score_max = tl.full((BLOCK_D,), -float("inf"), tl.float32)

        for src_i in tl.range(0, BLOCK_SRC):
            src_mask = src_i < n_src
            if OVERLAP:
                is_prev = src_i >= RATIO
            else:
                is_prev = src_i < 0
            ape_j = tl.where(is_prev, src_i - RATIO, src_i)
            src_global = tl.where(
                is_prev,
                (k - 1) * RATIO + ape_j,
                k * RATIO + ape_j,
            )
            zero_src = is_prev & (k == 0) if OVERLAP else src_i < 0
            from_state = src_global < PREFIX_LEN
            chunk_local = src_global - PREFIX_LEN
            safe_src_global = tl.maximum(src_global, 0)
            safe_chunk_local = tl.maximum(chunk_local, 0)

            if OVERLAP:
                data_col = tl.where(is_prev, HEAD_DIM + head_cols, head_cols)
            else:
                data_col = head_cols
            mask = src_mask & col_mask & ~zero_src

            chunk_offsets = safe_chunk_local * COFF_D + data_col
            chunk_kv_vals = tl.load(
                chunk_kv + chunk_offsets,
                mask=mask & ~from_state,
                other=0.0,
            )
            chunk_score_vals = tl.load(
                chunk_score + chunk_offsets,
                mask=mask & ~from_state,
                other=-float("inf"),
            )
            ape_vals = tl.load(
                ape + ape_j * COFF_D + data_col,
                mask=mask & ~from_state,
                other=0.0,
            )
            chunk_score_vals += ape_vals

            state_page_idx = tl.where(
                from_state & ~zero_src, safe_src_global // PAGE_SIZE, 0
            )
            state_page = tl.load(
                page_table + state_page_idx,
                mask=src_mask & from_state & ~zero_src,
                other=0,
            )
            state_slot = state_page * PAGE_SIZE + (safe_src_global % PAGE_SIZE)
            state_base = state_slot * (2 * COFF_D)
            state_kv_vals = tl.load(
                state_kv_score + state_base + data_col,
                mask=mask & from_state,
                other=0.0,
            )
            state_score_vals = tl.load(
                state_kv_score + state_base + COFF_D + data_col,
                mask=mask & from_state,
                other=-float("inf"),
            )

            kv_vals = tl.where(from_state, state_kv_vals, chunk_kv_vals).to(
                tl.float32
            )
            score_vals = tl.where(
                from_state, state_score_vals, chunk_score_vals
            ).to(tl.float32)
            score_vals = tl.where(mask, score_vals, -float("inf"))
            kv_vals = tl.where(mask, kv_vals, 0.0)

            new_score_max = tl.maximum(score_max, score_vals)
            old_delta = tl.where(
                score_max == -float("inf"), 0.0, score_max - new_score_max
            )
            new_delta = tl.where(
                score_vals == -float("inf"), 0.0, score_vals - new_score_max
            )
            old_scale = tl.where(
                new_score_max == -float("inf"),
                0.0,
                tl.exp(old_delta),
            )
            new_scale = tl.where(
                score_vals == -float("inf"),
                0.0,
                tl.exp(new_delta),
            )
            acc = acc * old_scale + kv_vals * new_scale
            denom = denom * old_scale + new_scale
            score_max = new_score_max

        acc = acc / denom
        tl.store(
            out + out_i * HEAD_DIM + head_cols,
            acc.to(out.dtype.element_ty),
            mask=col_mask,
        )


def dsv4_chunked_prefill_compress(
    chunk_kv: torch.Tensor,
    chunk_score: torch.Tensor,
    state_kv_score: torch.Tensor,
    page_table: torch.Tensor,
    ape: torch.Tensor,
    *,
    prefix_len: int,
    chunk_len: int,
    ratio: int,
    overlap: bool,
    page_size: int,
) -> torch.Tensor:
    assert ratio in (4, 128), f"unsupported DeepSeek-V4 compress ratio: {ratio}"
    assert overlap == (ratio == 4), "DeepSeek-V4 overlap is only valid for ratio=4"
    assert chunk_kv.is_contiguous()
    assert chunk_score.is_contiguous()
    assert state_kv_score.is_contiguous()
    assert ape.is_contiguous()

    n_out = (prefix_len + chunk_len) // ratio - prefix_len // ratio
    coff = 2 if overlap else 1
    head_dim = ape.shape[1] // coff
    out = torch.empty((n_out, head_dim), dtype=chunk_kv.dtype, device=chunk_kv.device)
    if n_out == 0:
        return out

    _, num_vectorcore = get_device_properties()
    block_d = min(triton.next_power_of_2(head_dim), 64)
    num_col_blocks = triton.cdiv(head_dim, block_d)
    num_programs = min(n_out * num_col_blocks, num_vectorcore)
    block_src = triton.next_power_of_2(2 * ratio if overlap else ratio)
    _dsv4_chunked_prefill_compress_kernel[(num_programs,)](
        chunk_kv,
        chunk_score,
        state_kv_score,
        page_table,
        ape,
        out,
        PREFIX_LEN=prefix_len,
        PAGE_SIZE=page_size,
        RATIO=ratio,
        HEAD_DIM=head_dim,
        COFF_D=ape.shape[1],
        BLOCK_SRC=block_src,
        BLOCK_D=block_d,
        N_OUT=n_out,
        NUM_COL_BLOCKS=num_col_blocks,
        NUM_PROGRAMS=num_programs,
        OVERLAP=overlap,
        num_warps=1,
    )
    return out

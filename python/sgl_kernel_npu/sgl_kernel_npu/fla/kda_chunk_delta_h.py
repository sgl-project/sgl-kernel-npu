# Adapted from the Kimi K3 NPU implementation on the 0728_dspark branch.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.fla.utils import (
    exp,
    exp2,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    safe_exp,
)

CHUNK_SIZE = 64


@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_npu(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    initial_state,
    initial_state_indices,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    """Build chunk states in KxV layout without Triton transposes.

    Ascend's Triton compiler can compile the GPU VxK implementation, but its
    transpose-heavy recurrence does not complete on A3.  This is the compact
    K<=128 form of the original 0728 NPU kernel.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # The NPU implementation keeps each state tile as [K, V].
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K

    index = tl.load(initial_state_indices + i_n).to(tl.int32)
    state = initial_state + index * stride_h + i_h * K * V

    if USE_INITIAL_STATE:
        p_initial_state1 = tl.make_block_ptr(
            state,
            (K, V),
            (V, 1),
            (0, i_v * BV),
            (64, BV),
            (1, 0),
        )
        b_h1 += tl.load(p_initial_state1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_initial_state2 = tl.make_block_ptr(
                state,
                (K, V),
                (V, 1),
                (64, i_v * BV),
                (64, BV),
                (1, 0),
            )
            b_h2 += tl.load(p_initial_state2, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h,
            (K, V),
            (V, 1),
            (0, i_v * BV),
            (64, BV),
            (1, 0),
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h,
                (K, V),
                (V, 1),
                (64, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w,
                (T, K),
                (stride_w, 1),
                (i_t * BT, 64),
                (BT, 64),
                (1, 0),
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v,
            (T, V),
            (stride_v, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(
                v_new,
                (T, V),
                (stride_v, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h,
                (T,),
                (H,),
                (i_t * BT,),
                (BT,),
                (0,),
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last

        if USE_GK:
            o_k = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k,
                mask=o_k < K,
                other=0.0,
            )
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[:, None]
            else:
                b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=o_k2 < K,
                    other=0.0,
                )
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[:, None]
                else:
                    b_h2 *= exp(b_gk_last2)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)

    p_state1 = tl.make_block_ptr(state, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
    tl.store(p_state1, b_h1.to(p_state1.dtype.element_ty), boundary_check=(0, 1))
    if K > 64:
        p_state2 = tl.make_block_ptr(
            state, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
        )
        tl.store(p_state2, b_h2.to(p_state2.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h_npu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    use_exp2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert not (
        use_exp2 and g is not None
    ), "use_exp2 covers only the per-channel gk path"
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    if K > 128:
        raise ValueError("The Kimi K3 NPU chunk-state kernel supports K <= 128")

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, CHUNK_SIZE), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE),
        )

    h = k.new_empty(B, NT, H, K, V)
    v_new = torch.empty_like(u) if save_new_value else None

    # The unified cache remains [..., H, V, K] for the NPU verify/decode
    # kernels. Select only this request's slots, convert them to the original
    # KxV prefill layout, then scatter the updated states back after launch.
    if initial_state is not None:
        if initial_state_indices is None:
            raise ValueError("initial_state_indices are required with initial_state")
        source_indices = initial_state_indices[:N].to(torch.long)
        kernel_state = (
            initial_state.index_select(0, source_indices).transpose(-1, -2).contiguous()
        )
        kernel_indices = torch.arange(N, dtype=torch.long, device=initial_state.device)
    else:
        source_indices = None
        # The kernel always materializes its final tile. Keep that write in a
        # private scratch buffer when the caller does not own a state cache.
        kernel_state = torch.empty(
            max(N, 1), H, K, V, dtype=torch.float32, device=k.device
        )
        kernel_indices = torch.arange(max(N, 1), dtype=torch.long, device=k.device)

    grid = (triton.cdiv(V, 32), N * H)
    chunk_gated_delta_rule_fwd_kernel_h_npu[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=kernel_state,
        initial_state_indices=kernel_indices,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=CHUNK_SIZE,
        BV=32,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_INITIAL_STATE=initial_state is not None,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        USE_EXP2=use_exp2,
        num_warps=4,
        num_stages=2,
    )

    if initial_state is not None:
        updated_state = kernel_state.transpose(-1, -2).contiguous()
        initial_state.index_copy_(0, source_indices, updated_state)
    return h, v_new


_KDA_CP_AFFINE_STREAMS: dict[int, object] = {}


def _get_kda_cp_affine_stream(device_index: int):
    stream = _KDA_CP_AFFINE_STREAMS.get(device_index)
    if stream is None:
        stream = torch.npu.Stream(device=device_index)
        _KDA_CP_AFFINE_STREAMS[device_index] = stream
    return stream


@triton.jit
def _load_kda_cp_chunk_row_tile(
    tensor,
    segment_len,
    stride,
    chunk_id,
    row_offset: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
):
    ptr = tl.make_block_ptr(
        tensor,
        (segment_len, K),
        (stride, 1),
        (chunk_id * BT, row_offset),
        (BT, 64),
        (1, 0),
    )
    return tl.load(ptr, boundary_check=(0, 1))


@triton.jit
def _load_kda_cp_chunk_column_tile(
    tensor,
    segment_len,
    stride,
    chunk_id,
    row_offset: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
):
    ptr = tl.make_block_ptr(
        tensor,
        (K, segment_len),
        (1, stride),
        (row_offset, chunk_id * BT),
        (64, BT),
        (0, 1),
    )
    return tl.load(ptr, boundary_check=(0, 1))


@triton.jit
def _load_kda_cp_gate_tile(
    gk,
    gate_offset,
    row_offset: tl.constexpr,
    K: tl.constexpr,
):
    rows = row_offset + tl.arange(0, 64)
    gate = tl.load(gk + gate_offset + rows, mask=rows < K, other=0.0)
    return exp2(gate)[:, None]


@triton.jit
def _load_kda_cp_row_tile(
    tensor,
    col_offset,
    row_offset: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BC: tl.constexpr,
):
    ptr = tl.make_block_ptr(
        tensor,
        (K, C),
        (ROW_STRIDE, 1),
        (row_offset, col_offset),
        (64, BC),
        (1, 0),
    )
    return tl.load(ptr, boundary_check=(0, 1))


@triton.jit
def _store_kda_cp_row_tile(
    tensor,
    tile,
    col_offset,
    row_offset: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BC: tl.constexpr,
):
    ptr = tl.make_block_ptr(
        tensor,
        (K, C),
        (ROW_STRIDE, 1),
        (row_offset, col_offset),
        (64, BC),
        (1, 0),
    )
    tl.store(ptr, tile.to(ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_affine_h_kernel(
    k,
    v,
    w,
    gk,
    affine,
    cu_seqlens,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    """Compute the additive half of a segment's recurrent affine map.

    This is the multi-segment Ascend adaptation of FLA PR 691's
    ``pre_process_fwd_kernel_stage1``.  Unlike the ordinary state kernel it
    keeps only the final state of every segment and therefore does not write a
    ``[num_chunks, H, K, V]`` intermediate tensor.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    segment_len = eos - bos
    num_chunks = tl.cdiv(segment_len, BT)

    affine += ((i_n * H + i_h) * K * (V + K)).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    h0 = tl.zeros([64, BV], dtype=tl.float32)
    h1 = tl.zeros([64, BV], dtype=tl.float32)

    for chunk_id in range(num_chunks):
        w0 = _load_kda_cp_chunk_row_tile(
            w, segment_len, stride_w, chunk_id, 0, K=K, BT=BT
        )
        value = tl.dot(w0, h0.to(w0.dtype))
        w1 = _load_kda_cp_chunk_row_tile(
            w, segment_len, stride_w, chunk_id, 64, K=K, BT=BT
        )
        value += tl.dot(w1, h1.to(w1.dtype))

        value_ptr = tl.make_block_ptr(
            v,
            (segment_len, V),
            (stride_v, 1),
            (chunk_id * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        value = tl.load(value_ptr, boundary_check=(0, 1)) - value

        last_token = min((chunk_id + 1) * BT, segment_len) - 1
        gate_offset = (bos + last_token) * H * K + i_h * K
        h0 *= _load_kda_cp_gate_tile(gk, gate_offset, 0, K=K)
        h1 *= _load_kda_cp_gate_tile(gk, gate_offset, 64, K=K)

        value = value.to(k.dtype.element_ty)
        key0 = _load_kda_cp_chunk_column_tile(
            k, segment_len, stride_k, chunk_id, 0, K=K, BT=BT
        )
        h0 += tl.dot(key0, value)
        key1 = _load_kda_cp_chunk_column_tile(
            k, segment_len, stride_k, chunk_id, 64, K=K, BT=BT
        )
        h1 += tl.dot(key1, value)

    _store_kda_cp_row_tile(
        affine, h0, i_v * BV, 0, K=K, C=V, ROW_STRIDE=V + K, BC=BV
    )
    _store_kda_cp_row_tile(
        affine, h1, i_v * BV, 64, K=K, C=V, ROW_STRIDE=V + K, BC=BV
    )


@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_affine_m_kernel(
    k,
    w,
    gk,
    affine,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    """Compute the transition half of a segment affine map.

    This is the Ascend-friendly counterpart of FLA PR 691's stage-2
    preprocessing kernel.  PR 691 forms a full KxK chunk transition before
    multiplying it into the running transform.  Keeping that full accumulator
    across a dynamic loop is not currently stable in BiShengIR.  Instead, this
    kernel advances one 64-column slab of M at a time using

        M <- D @ M - K.T @ (W @ M).

    The recurrence is identical, but every live dot accumulator is at most
    64x64 -- the same shape already proven by the regular Ascend KDA state
    kernel.  Unlike that generic kernel, this path creates the identity in
    registers and has no state-index, value, or chunk-state branches.
    """
    i_c, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    segment_len = eos - bos
    num_chunks = tl.cdiv(segment_len, BT)

    affine += ((i_n * H + i_h) * K * (V + K) + V).to(tl.int64)
    k += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    stride_kw = H * K

    cols = i_c * BC + tl.arange(0, BC)
    rows0 = tl.arange(0, 64)
    m0 = (rows0[:, None] == cols[None, :]).to(tl.float32)
    rows1 = 64 + rows0
    m1 = (rows1[:, None] == cols[None, :]).to(tl.float32)

    for chunk_id in range(num_chunks):
        w0 = _load_kda_cp_chunk_row_tile(
            w, segment_len, stride_kw, chunk_id, 0, K=K, BT=BT
        )
        tmp = tl.dot(w0, m0.to(w0.dtype))
        w1 = _load_kda_cp_chunk_row_tile(
            w, segment_len, stride_kw, chunk_id, 64, K=K, BT=BT
        )
        tmp += tl.dot(w1, m1.to(w1.dtype))

        last_token = min((chunk_id + 1) * BT, segment_len) - 1
        gate_offset = (bos + last_token) * H * K + i_h * K
        m0 *= _load_kda_cp_gate_tile(gk, gate_offset, 0, K=K)
        m1 *= _load_kda_cp_gate_tile(gk, gate_offset, 64, K=K)

        tmp = tmp.to(k.dtype.element_ty)
        k0 = _load_kda_cp_chunk_column_tile(
            k, segment_len, stride_kw, chunk_id, 0, K=K, BT=BT
        )
        m0 -= tl.dot(k0, tmp)
        k1 = _load_kda_cp_chunk_column_tile(
            k, segment_len, stride_kw, chunk_id, 64, K=K, BT=BT
        )
        m1 -= tl.dot(k1, tmp)

    _store_kda_cp_row_tile(
        affine, m0, i_c * BC, 0, K=K, C=K, ROW_STRIDE=V + K, BC=BC
    )
    _store_kda_cp_row_tile(
        affine, m1, i_c * BC, 64, K=K, C=K, ROW_STRIDE=V + K, BC=BC
    )


@triton.jit
def _apply_kda_cp_affine_block(
    gathered,
    h0,
    h1,
    owner_rank,
    source_segment,
    i_h,
    i_v,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    BV: tl.constexpr,
):
    """Apply one execution-plan affine transform to a state tile."""
    affine = gathered + (
        ((owner_rank * MAX_SEGMENTS + source_segment) * H + i_h) * K * (V + K)
    ).to(tl.int64)

    add0 = _load_kda_cp_row_tile(
        affine, i_v * BV, 0, K=K, C=V, ROW_STRIDE=V + K, BC=BV
    ).to(tl.float32)
    m00 = _load_kda_cp_row_tile(
        affine + V, 0, 0, K=K, C=K, ROW_STRIDE=V + K, BC=64
    )
    next0 = tl.dot(m00, h0.to(m00.dtype)) + add0

    m01 = _load_kda_cp_row_tile(
        affine + V, 64, 0, K=K, C=K, ROW_STRIDE=V + K, BC=64
    )
    next0 += tl.dot(m01, h1.to(m01.dtype))

    add1 = _load_kda_cp_row_tile(
        affine, i_v * BV, 64, K=K, C=V, ROW_STRIDE=V + K, BC=BV
    ).to(tl.float32)
    m10 = _load_kda_cp_row_tile(
        affine + V, 0, 64, K=K, C=K, ROW_STRIDE=V + K, BC=64
    )
    m11 = _load_kda_cp_row_tile(
        affine + V, 64, 64, K=K, C=K, ROW_STRIDE=V + K, BC=64
    )
    next1 = tl.dot(m10, h0.to(m10.dtype)) + tl.dot(m11, h1.to(m11.dtype)) + add1
    return next0, next1


@triton.jit
def _store_kda_cp_state_tile(
    target,
    h0,
    h1,
    local_index: tl.constexpr,
    i_h,
    i_v,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    target += ((local_index * H + i_h) * K * V).to(tl.int64)
    _store_kda_cp_row_tile(
        target, h0, i_v * BV, 0, K=K, C=V, ROW_STRIDE=V, BC=BV
    )
    _store_kda_cp_row_tile(
        target, h1, i_v * BV, 64, K=K, C=V, ROW_STRIDE=V, BC=BV
    )


@triton.jit
def merge_kda_cp_affine_states_kernel(
    gathered,
    initial_state,
    local_initial,
    final_state,
    tracked_state,
    owner_ranks,
    source_segments,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    NUM_STEPS: tl.constexpr,
    TRACK_STEP: tl.constexpr,
    LOCAL_STEP_0: tl.constexpr,
    LOCAL_STEP_1: tl.constexpr,
    LOCAL_STEP_2: tl.constexpr,
    BV: tl.constexpr,
):
    """Fuse plan-driven affine composition for the batch-one hot path."""
    i_v, i_h = tl.program_id(0), tl.program_id(1)
    initial_state += (i_h * K * V).to(tl.int64)

    h0 = _load_kda_cp_row_tile(
        initial_state, i_v * BV, 0, K=K, C=V, ROW_STRIDE=V, BC=BV
    ).to(tl.float32)
    h1 = _load_kda_cp_row_tile(
        initial_state, i_v * BV, 64, K=K, C=V, ROW_STRIDE=V, BC=BV
    ).to(tl.float32)

    for step_id in range(NUM_STEPS):
        owner_rank = tl.load(owner_ranks + step_id)
        source_segment = tl.load(source_segments + step_id)

        local_index = -1
        if step_id == LOCAL_STEP_0:
            local_index = 0
        if step_id == LOCAL_STEP_1:
            local_index = 1
        if step_id == LOCAL_STEP_2:
            local_index = 2
        if local_index >= 0:
            _store_kda_cp_state_tile(
                local_initial,
                h0,
                h1,
                local_index,
                i_h,
                i_v,
                H=H,
                K=K,
                V=V,
                BV=BV,
            )

        h0, h1 = _apply_kda_cp_affine_block(
            gathered,
            h0,
            h1,
            owner_rank,
            source_segment,
            i_h,
            i_v,
            H=H,
            K=K,
            V=V,
            MAX_SEGMENTS=MAX_SEGMENTS,
            BV=BV,
        )
        if step_id == TRACK_STEP:
            _store_kda_cp_state_tile(
                tracked_state,
                h0,
                h1,
                0,
                i_h,
                i_v,
                H=H,
                K=K,
                V=V,
                BV=BV,
            )
    _store_kda_cp_state_tile(
        final_state,
        h0,
        h1,
        0,
        i_h,
        i_v,
        H=H,
        K=K,
        V=V,
        BV=BV,
    )


def merge_kda_cp_affine_states(
    gathered: torch.Tensor,
    initial_state: torch.Tensor,
    local_initial: torch.Tensor,
    final_state: torch.Tensor,
    *,
    cp_rank: int,
    owner_ranks: torch.Tensor | None = None,
    source_segments: torch.Tensor | None = None,
    local_indices: torch.Tensor | None = None,
    local_steps: tuple[int, ...] | None = None,
    tracked_state: torch.Tensor | None = None,
    track_step: int = -1,
) -> None:
    """Launch the fused batch-one affine merge used by Kimi-K3 PCP.

    ``gathered`` is ``[cp, max_segments, H, K, V+K]`` in rank-owned
    segment order.  The optional device plan supports the extra segment
    created when a radix checkpoint splits a natural zigzag block.
    """
    cp_size, max_segments, num_heads, key_dim, affine_dim = gathered.shape
    value_dim = affine_dim - key_dim
    if initial_state.shape[0] != 1 or key_dim != 128:
        raise ValueError(
            "fused KDA CP merge requires batch one and K = 128; "
            f"got batch={initial_state.shape[0]}, K={key_dim}"
        )
    if owner_ranks is None or source_segments is None or local_indices is None:
        owners = []
        sources = []
        locals_ = []
        local_id = 0
        for block_id in range(2 * cp_size):
            owner = block_id if block_id < cp_size else 2 * cp_size - block_id - 1
            source = int(block_id >= cp_size)
            owners.append(owner)
            sources.append(source)
            if owner == cp_rank:
                locals_.append(local_id)
                local_id += 1
            else:
                locals_.append(-1)
        owner_ranks = torch.tensor(owners, dtype=torch.int32, device=gathered.device)
        source_segments = torch.tensor(
            sources, dtype=torch.int32, device=gathered.device
        )
        local_indices = torch.tensor(locals_, dtype=torch.int32, device=gathered.device)
        local_steps = tuple(
            step_id for step_id, local_id in enumerate(locals_) if local_id >= 0
        )
    num_steps = owner_ranks.numel()
    if tracked_state is None:
        tracked_state = final_state
    valid_plan = (
        num_steps == source_segments.numel() == local_indices.numel()
        and 0 < local_initial.shape[0] <= max_segments
        and track_step < num_steps
        and local_steps is not None
        and len(local_steps) == local_initial.shape[0]
        and len(local_steps) <= 3
    )
    if not valid_plan:
        raise ValueError(
            "invalid fused KDA CP merge plan; "
            f"steps=({num_steps},{source_segments.numel()},{local_indices.numel()}), "
            f"local_states={local_initial.shape[0]}/{max_segments}, "
            f"local_steps={local_steps}, track_step={track_step}"
        )
    padded_local_steps = (*local_steps, -1, -1, -1)[:3]
    merge_kda_cp_affine_states_kernel[(triton.cdiv(value_dim, 64), num_heads)](
        gathered=gathered,
        initial_state=initial_state,
        local_initial=local_initial,
        final_state=final_state,
        tracked_state=tracked_state,
        owner_ranks=owner_ranks,
        source_segments=source_segments,
        H=num_heads,
        K=key_dim,
        V=value_dim,
        MAX_SEGMENTS=max_segments,
        NUM_STEPS=num_steps,
        TRACK_STEP=track_step,
        LOCAL_STEP_0=padded_local_steps[0],
        LOCAL_STEP_1=padded_local_steps[1],
        LOCAL_STEP_2=padded_local_steps[2],
        BV=64,
        num_warps=4,
        num_stages=2,
    )


def chunk_gated_delta_rule_fwd_affine_npu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    cu_seqlens: torch.LongTensor,
) -> torch.Tensor:
    """Return per-segment ``[H | M]`` without materializing chunk states.

    The output satisfies ``state_out = M @ state_in + H`` and is laid out as
    ``[segments, heads, K, V + K]``.  It is intentionally FP32 because the
    cross-rank affine composition is numerically sensitive.
    """
    if cu_seqlens is None or gk is None:
        raise ValueError("KDA CP affine preprocessing requires cu_seqlens and gk")
    _, token_count, key_heads, key_dim = k.shape
    num_heads = u.shape[-2]
    value_dim = u.shape[-1]
    num_segments = len(cu_seqlens) - 1
    if key_dim != 128 or num_heads % key_heads != 0:
        raise ValueError(
            "KDA CP affine preprocessing requires K = 128 and H divisible "
            f"by Hg; got K={key_dim}, H={num_heads}, Hg={key_heads}"
        )

    affine = torch.empty(
        num_segments,
        num_heads,
        key_dim,
        value_dim + key_dim,
        dtype=torch.float32,
        device=k.device,
    )

    def launch_additive() -> None:
        chunk_gated_delta_rule_fwd_affine_h_kernel[
            (triton.cdiv(value_dim, 64), num_segments * num_heads)
        ](
            k=k,
            v=u,
            w=w,
            gk=gk,
            affine=affine,
            cu_seqlens=cu_seqlens,
            T=token_count,
            H=num_heads,
            Hg=key_heads,
            K=key_dim,
            V=value_dim,
            BT=CHUNK_SIZE,
            BV=64,
            num_warps=4,
            num_stages=2,
        )

    def launch_transition() -> None:
        chunk_gated_delta_rule_fwd_affine_m_kernel[
            (triton.cdiv(key_dim, 64), num_segments * num_heads)
        ](
            k=k,
            w=w,
            gk=gk,
            affine=affine,
            cu_seqlens=cu_seqlens,
            T=token_count,
            H=num_heads,
            K=key_dim,
            V=value_dim,
            BT=CHUNK_SIZE,
            BC=64,
            num_warps=4,
            num_stages=2,
        )

    # H and M write disjoint affine columns and have no data dependency. Run
    # them on separate streams, then join before the caller starts the
    # collective. Streams are cached per process/device.
    device_index = k.device.index or 0
    main_stream = torch.npu.current_stream(device_index)
    affine_stream = _get_kda_cp_affine_stream(device_index)
    affine_stream.wait_stream(main_stream)
    with torch.npu.stream(affine_stream):
        launch_additive()
    launch_transition()
    main_stream.wait_stream(affine_stream)
    return affine

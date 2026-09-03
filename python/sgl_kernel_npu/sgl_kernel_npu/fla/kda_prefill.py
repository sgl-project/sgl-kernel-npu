"""Framework-ready WY recompute with wrapper-owned beta layout conversion.

The public ``recompute_w_u_fwd`` retains
the original contiguous ``beta[B,T,H]`` interface and materializes a contiguous
head-major copy before launch. The kernel body is adapted from
``sglang.srt.layers.attention.fla.kda.recompute_w_u_fwd_kernel``.

Framework contract:

* ``k``: ``[B, T, H, K]``, contiguous.
* ``v``: ``[B, T, H, V]``, contiguous.
* public-wrapper ``beta``: ``[B, T, H]``, contiguous.
* kernel/direct-wrapper ``beta``: ``[B, H, T]``, contiguous.
* ``A``: ``[B, T, H, BT]``, contiguous.
* ``gk``: ``[B, T, H, K]``, contiguous FP32 chunk-local cumulative gate.
* varlen input is packed with physical ``B == 1`` and sequence boundaries in
  ``cu_seqlens``; ``beta`` covers the complete packed ``T`` axis.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.fla.utils import exp, exp2, prepare_chunk_indices


@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_head_major_kernel(
    k,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    # T_TOTAL remains the physical packed-token extent even when the varlen
    # branch replaces T with the current logical sequence length.
    T_TOTAL = T
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # beta is physical [B,H,T_TOTAL].  Fixed mode selects the physical batch
    # through i_b.  Packed varlen requires B=1, and bos selects the logical
    # sequence inside the shared T_TOTAL axis.
    beta_bos = bos if IS_VARLEN else 0
    p_b = tl.make_block_ptr(
        beta + (i_b * H + i_h) * T_TOTAL + beta_bos,
        (T,),
        (1,),
        (i_t * BT,),
        (BT,),
        (0,),
    )
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kb *= exp(b_gk)
        last_idx = min(i_t * BT + BT, T) - 1

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0)
        b_kg = b_k * exp(b_gn - b_gk)

        p_kg = tl.make_block_ptr(
            kg + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def _validate_inputs(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
) -> tuple[int, int, int, int, int, int]:
    """Validate pointer-layout invariants without synchronizing the device."""
    if k.ndim != 4:
        raise ValueError(f"k must be [B,T,H,K], got shape={tuple(k.shape)}")
    B, T, H, K = k.shape
    valid_shapes = (
        min(B, T, H, K) > 0
        and v.ndim == 4
        and tuple(v.shape[:3]) == (B, T, H)
        and v.shape[-1] > 0
        and tuple(beta.shape) == (B, H, T)
        and A.ndim == 4
        and tuple(A.shape[:3]) == (B, T, H)
        and A.shape[-1] == 64
        and gk is not None
        and tuple(gk.shape) == (B, T, H, K)
    )
    if not valid_shapes:
        raise ValueError(
            "invalid KDA recompute shapes; expected v/A to match k[B,T,H], "
            "beta[B,H,T], gk==k, and A chunk size 64; "
            f"got k={tuple(k.shape)}, v={tuple(v.shape)}, beta={tuple(beta.shape)}, "
            f"A={tuple(A.shape)}, gk={None if gk is None else tuple(gk.shape)}"
        )
    V, BT = v.shape[-1], A.shape[-1]

    tensors = {"k": k, "v": v, "beta": beta, "A": A, "gk": gk}
    invalid_layouts = [
        name
        for name, tensor in tensors.items()
        if not tensor.is_contiguous() or tensor.device != k.device
    ]
    if invalid_layouts:
        raise ValueError(
            "KDA recompute inputs must be contiguous and on "
            f"{k.device}; invalid={invalid_layouts}"
        )
    if not (
        k.dtype == v.dtype == A.dtype
        and gk.dtype == torch.float32
        and beta.dtype in (k.dtype, torch.float32)
    ):
        raise ValueError(
            "invalid KDA recompute dtypes; expected k/v/A to match, gk FP32, "
            f"and beta data dtype or FP32; got k={k.dtype}, v={v.dtype}, "
            f"A={A.dtype}, gk={gk.dtype}, beta={beta.dtype}"
        )

    if cu_seqlens is None:
        if chunk_indices is not None:
            raise ValueError("chunk_indices requires cu_seqlens")
    else:
        valid_cu_seqlens = (
            B == 1
            and cu_seqlens.ndim == 1
            and cu_seqlens.numel() >= 2
            and cu_seqlens.dtype in (torch.int32, torch.int64)
            and cu_seqlens.device == k.device
            and cu_seqlens.is_contiguous()
        )
        if not valid_cu_seqlens:
            raise ValueError(
                "packed varlen requires B=1 and contiguous int32/int64 "
                f"cu_seqlens on {k.device}; got B={B}, "
                f"shape={tuple(cu_seqlens.shape)}, dtype={cu_seqlens.dtype}, "
                f"device={cu_seqlens.device}, stride={cu_seqlens.stride()}"
            )
        valid_chunk_indices = chunk_indices is None or (
            chunk_indices.ndim == 2
            and chunk_indices.shape[1] == 2
            and chunk_indices.dtype == cu_seqlens.dtype
            and chunk_indices.device == k.device
            and chunk_indices.is_contiguous()
        )
        if not valid_chunk_indices:
            raise ValueError(
                "chunk_indices must be contiguous [num_chunks,2] and match "
                f"cu_seqlens; got shape={tuple(chunk_indices.shape)}, "
                f"device={chunk_indices.device}, dtype={chunk_indices.dtype}, "
                f"stride={chunk_indices.stride()}"
            )
    return B, T, H, K, V, BT


def recompute_w_u_fwd_head_major(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run WY recompute using caller-owned contiguous ``beta[B,H,T]``.

    No beta view, transpose, allocation, or copy occurs in this wrapper.
    """
    B, T, H, K, V, BT = _validate_inputs(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)
    recompute_w_u_fwd_head_major_kernel[(NT, B * H)](
        k=k,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=64,
        IS_VARLEN=cu_seqlens is not None,
        DOT_PRECISION="ieee",
    )
    return w, u, kg


def prepare_recompute_beta_head_major(
    beta: torch.Tensor,
    *,
    expected_shape: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Materialize contiguous ``beta[B,T,H]`` as physical ``[B,H,T]``."""
    valid_beta = (
        beta.ndim == 3
        and (expected_shape is None or tuple(beta.shape) == expected_shape)
        and beta.is_contiguous()
    )
    if not valid_beta:
        raise ValueError(
            "beta must be contiguous [B,T,H] matching k; "
            f"expected={expected_shape}, shape={tuple(beta.shape)}, "
            f"stride={beta.stride()}"
        )
    return beta.permute(0, 2, 1).contiguous()


@triton.jit(do_not_specialize=["T"])
def _chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    H_LAYOUT_VK: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        if H_LAYOUT_VK:
            p_h = tl.make_block_ptr(
                h + (i_tg * H + i_h) * V * K,
                (V, K),
                (K, 1),
                (i_v * BV, i_k * BK),
                (BV, BK),
                (1, 0),
            )
            # CUDA keeps h as [V, K].
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
        else:
            p_h = tl.make_block_ptr(
                h + (i_tg * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
            # The 0728 NPU producer stores h directly as [K, V].
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the existing public API while using the head-major kernel."""
    if k.ndim != 4:
        raise ValueError(f"k must be [B,T,H,K], got shape={tuple(k.shape)}")
    beta_head_major = prepare_recompute_beta_head_major(
        beta,
        expected_shape=tuple(k.shape[:3]),
    )
    w, u, kg = recompute_w_u_fwd_head_major(
        k=k,
        v=v,
        beta=beta_head_major,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, kg


def chunk_gla_fwd_o_gk_npu(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    chunk_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """Consume Ascend's KxV chunk-state layout without a transpose."""
    B, T, H, K, V = *q.shape, v.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    num_chunks = (
        triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    )
    grid = (triton.cdiv(V, 64), num_chunks, B * H)
    _chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=out,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=chunk_size,
        BK=64,
        BV=64,
        IS_VARLEN=cu_seqlens is not None,
        H_LAYOUT_VK=False,
    )
    return out

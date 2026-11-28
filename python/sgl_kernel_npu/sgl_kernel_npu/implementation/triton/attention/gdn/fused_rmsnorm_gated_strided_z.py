import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _fused_rmsnorm_gated_strided_z_kernel(
    X,  # [T, NV, D] contiguous
    QKVZ,  # [T, QKVZ_DIM] contiguous, Z starts at QKV_DIM
    W,  # [D]
    OUT,  # [T, V_DIM] contiguous
    T,
    QKVZ_DIM: tl.constexpr,
    QKV_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    NV: tl.constexpr,
    D: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    n_h_blocks = NV // HEADS_PER_PROG
    total_blocks = T * n_h_blocks
    n_progs = tl.num_programs(0)
    w = tl.load(W + tl.arange(0, D)).to(tl.float32)[None, :]
    offs_hd = (tl.arange(0, HEADS_PER_PROG)[:, None] * D + tl.arange(0, D)[None, :]).to(
        tl.int64
    )
    for blk in range(pid, total_blocks, n_progs):
        row = (blk // n_h_blocks).to(tl.int64)
        hb = (blk % n_h_blocks).to(tl.int64)
        off = hb * (HEADS_PER_PROG * D) + offs_hd
        x = tl.load(X + row * V_DIM + off).to(tl.float32)
        z = tl.load(QKVZ + row * QKVZ_DIM + QKV_DIM + off).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) * (1.0 / D) + EPS)
        y = (x * rstd) * w * (z * tl.sigmoid(z))
        tl.store(OUT + row * V_DIM + off, y.to(OUT.dtype.element_ty))


def fused_rmsnorm_gated_strided_z(
    core_attn_out: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    norm_weight: torch.Tensor,
    z_offset: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute RMSNorm(core_attn_out) * SiLU(z) reading strided z directly from qkvz."""
    if core_attn_out.ndim != 3 or not core_attn_out.is_contiguous():
        raise ValueError("core_attn_out must be contiguous 3D [T, NV, D]")
    if projected_states_qkvz.ndim != 2 or not projected_states_qkvz.is_contiguous():
        raise ValueError("projected_states_qkvz must be contiguous 2D [T, QKVZ_DIM]")
    if norm_weight.ndim != 1 or not norm_weight.is_contiguous():
        raise ValueError("norm_weight must be contiguous 1D [D]")

    T, num_v_heads, head_v_dim = (
        int(core_attn_out.shape[0]),
        int(core_attn_out.shape[1]),
        int(core_attn_out.shape[2]),
    )
    v_dim = num_v_heads * head_v_dim
    qkvz_dim = int(projected_states_qkvz.shape[1])
    if projected_states_qkvz.shape[0] != T:
        raise ValueError(
            f"Token count mismatch: core_attn_out has {T}, qkvz has {projected_states_qkvz.shape[0]}"
        )
    if norm_weight.shape[0] != head_v_dim:
        raise ValueError(
            f"norm_weight length {norm_weight.shape[0]} != head_v_dim {head_v_dim}"
        )
    if z_offset < 0 or z_offset + v_dim > qkvz_dim:
        raise ValueError(
            f"z slice [{z_offset}:{z_offset + v_dim}] exceeds qkvz width {qkvz_dim}"
        )

    out = torch.empty(
        (T, v_dim), dtype=core_attn_out.dtype, device=core_attn_out.device
    )
    if T == 0 or v_dim == 0:
        return out

    if T >= 56 and num_v_heads <= 64:
        hpp = num_v_heads
    elif T <= 4 and num_v_heads % 4 == 0:
        hpp = 4
    elif T <= 16 and num_v_heads % 8 == 0:
        hpp = 8
    elif num_v_heads % 16 == 0:
        hpp = 16
    else:
        hpp = 1
    grid = min(56, max(1, T * (num_v_heads // hpp)))
    _fused_rmsnorm_gated_strided_z_kernel[(grid,)](
        core_attn_out,
        projected_states_qkvz,
        norm_weight,
        out,
        T,
        qkvz_dim,
        z_offset,
        v_dim,
        num_v_heads,
        head_v_dim,
        hpp,
        eps,
    )
    return out

from functools import lru_cache
from typing import Optional

import torch

GLOBAL_VALUE_HEADS = (16, 32, 48, 64)
GLOBAL_KEY_HEADS = 16
SUPPORTED_TP_DEGREES = (1, 2, 4, 8)
HEAD_DIM = 128
CHUNK_SIZE = 128


def _device_key(device: torch.device) -> tuple[str, int]:
    return device.type, 0 if device.index is None else device.index


def _device_from_key(device_type: str, device_index: int) -> torch.device:
    return torch.device(device_type, device_index)


@lru_cache(maxsize=16)
def _masks(device_type: str, device_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = _device_from_key(device_type, device_index)
    mask_lower = torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=device), diagonal=-1
    ).float()
    mask_full = torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=device), diagonal=0
    ).float()
    return mask_lower, mask_full


@lru_cache(maxsize=16)
def _minus_identity(device_type: str, device_index: int) -> torch.Tensor:
    device = _device_from_key(device_type, device_index)
    minus_identity = torch.zeros(
        CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=torch.float16
    )
    minus_identity.fill_diagonal_(-1)
    return minus_identity


def _total_chunks(cu_seqlens: torch.Tensor) -> int:
    cu = cu_seqlens.cpu().tolist()
    total = 0
    for start, end in zip(cu, cu[1:]):
        total += (end - start + CHUNK_SIZE - 1) // CHUNK_SIZE
    return total


def _block_dim(device: torch.device) -> int:
    try:
        props = torch.npu.get_device_properties(device)
        return max(1, int(getattr(props, "cube_core_num", 24)))
    except (RuntimeError, AttributeError, AssertionError):
        return 24


def _head_pair_supported(num_value_heads: int, num_key_heads: int) -> bool:
    if num_key_heads <= 0 or GLOBAL_KEY_HEADS % num_key_heads != 0:
        return False
    tp_degree = GLOBAL_KEY_HEADS // num_key_heads
    if tp_degree not in SUPPORTED_TP_DEGREES:
        return False
    return num_value_heads * tp_degree in GLOBAL_VALUE_HEADS


def mega_gdn_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor] = None,
) -> bool:
    if q.dim() != 4 or v.dim() != 4:
        return False
    if k.shape != q.shape:
        return False
    if q.shape[0] != 1 or v.shape[0] != 1 or q.shape[1] != v.shape[1]:
        return False

    if not _head_pair_supported(v.shape[2], q.shape[2]):
        return False
    if q.shape[3] != HEAD_DIM or v.shape[3] != HEAD_DIM:
        return False

    if g.shape != beta.shape:
        return False
    if g.shape != (1, q.shape[1], v.shape[2]):
        return False

    if initial_state is not None:
        if initial_state.dim() != 4:
            return False
        num_sequences = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
        if initial_state.shape != (
            num_sequences,
            v.shape[2],
            HEAD_DIM,
            HEAD_DIM,
        ):
            return False
    return True


def run_mega_chunk_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if scale is None:
        scale = k.shape[-1] ** -0.5

    q_dtype, k_dtype, v_dtype = q.dtype, k.dtype, v.dtype
    q, k, v, beta = (t.half() for t in (q, k, v, beta))

    _, total_tokens, _, head_dim = q.shape
    num_value_heads = v.shape[-2]
    device_type, device_index = _device_key(q.device)
    if cu_seqlens is None:
        cu32 = torch.tensor([0, total_tokens], dtype=torch.int32, device=q.device)
    else:
        cu32 = cu_seqlens.to(torch.int32)

    num_sequences = cu32.numel() - 1
    num_chunks = _total_chunks(cu32)
    num_matrices = num_chunks * num_value_heads

    mask_lower, mask_full = _masks(device_type, device_index)
    minus_identity = _minus_identity(device_type, device_index)

    g_sum = torch.empty_like(g, dtype=torch.float32)
    g_t = torch.empty(
        num_value_heads, total_tokens, device=q.device, dtype=torch.float32
    )
    beta_t = torch.empty(
        num_value_heads, total_tokens, device=q.device, dtype=torch.float16
    )

    A = torch.zeros(
        1,
        total_tokens,
        num_value_heads,
        CHUNK_SIZE,
        device=q.device,
        dtype=torch.float16,
    )
    A_inv_f32 = torch.zeros_like(A, dtype=torch.float32)
    A_inv = torch.zeros_like(A)

    w = torch.empty_like(v)
    u = torch.empty_like(v)
    h = torch.zeros(
        num_chunks * num_value_heads,
        head_dim,
        head_dim,
        device=q.device,
        dtype=torch.float16,
    )
    v_new = torch.empty_like(v)
    final_state = torch.zeros(
        num_sequences * num_value_heads,
        head_dim,
        head_dim,
        device=q.device,
        dtype=torch.float16,
    )
    has_initial_state = initial_state is not None
    initial_state = initial_state.half() if has_initial_state else final_state

    block_dim = _block_dim(q.device)
    kkt_workspace = torch.zeros(
        block_dim * 2, CHUNK_SIZE, CHUNK_SIZE, device=q.device, dtype=torch.float16
    )
    wy_workspace_a1 = torch.zeros(
        block_dim, CHUNK_SIZE, CHUNK_SIZE, device=q.device, dtype=torch.float16
    )
    wy_workspace_a2 = torch.zeros_like(wy_workspace_a1)
    h_workspace = torch.zeros(
        block_dim * 4, head_dim, head_dim, device=q.device, dtype=torch.float16
    )
    o_workspace_qk = torch.zeros(
        block_dim, CHUNK_SIZE, CHUNK_SIZE, device=q.device, dtype=torch.float16
    )
    o_workspace_qs = torch.zeros(
        block_dim, CHUNK_SIZE, head_dim, device=q.device, dtype=torch.float16
    )
    o_workspace_gated = torch.zeros_like(o_workspace_qk)
    out = torch.empty_like(v)

    torch.ops.npu.mega_chunk_gdn(
        q,
        k,
        v,
        g,
        beta,
        mask_lower,
        mask_full,
        minus_identity,
        cu32,
        out,
        g_sum,
        g_t,
        beta_t,
        A,
        A_inv_f32,
        A_inv,
        w,
        u,
        h,
        v_new,
        final_state,
        initial_state,
        has_initial_state,
        kkt_workspace,
        wy_workspace_a1,
        wy_workspace_a2,
        h_workspace,
        o_workspace_qk,
        o_workspace_qs,
        o_workspace_gated,
        block_dim,
        num_sequences,
        total_tokens,
        total_tokens,
        num_matrices,
    )

    h = h.view(1, num_chunks, num_value_heads, head_dim, head_dim)
    if output_final_state:
        final_state_out = final_state.view(
            num_sequences, num_value_heads, head_dim, head_dim
        ).to(torch.float32)
    else:
        final_state_out = None
    return (
        g_sum,
        (out * scale).to(q_dtype),
        A_inv.to(k_dtype),
        final_state_out,
        w.to(k_dtype),
        h.to(k_dtype),
        v_new.to(v_dtype),
    )

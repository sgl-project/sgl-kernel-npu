import torch
import torch_npu


def chunk_gated_delta_rule_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    beta: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    actual_seq_lengths: torch.Tensor | None = None,
    scale: float | None = None,
    g: torch.Tensor | None = None,
    chunk_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Fused chunk gated delta rule forward kernel (single-op NPU implementation).

    Args:
        query: (T, Nk, Dk) bfloat16, L2-normalized.
        key:   (T, Nk, Dk) bfloat16, L2-normalized.
        value: (T, Nv, Dv) bfloat16.
        beta:  (T, Nv) bfloat16.
        initial_state: (B, Nv, Dv, Dk) bfloat16 (or float32 on Ascend950).
        actual_seq_lengths: (B,) int32, sum must equal T.
        scale: scalar, defaults to 1/sqrt(Dk).
        g: (T, Nv) float32, cumulative log decay gate. None means no gating.
        chunk_state: optional pre-allocated output tensor of shape
            (totalChunks, Nv, Dv, Dk) where
            totalChunks = sum_b ceil(actual_seq_lengths[b] / 64).
            When provided, the kernel writes each chunk's entering (pre-decay)
            state in-place. dtype must match initial_state. Default None (disabled).

    Returns:
        out: (T, Nv, Dv) bfloat16.
        final_state: (B, Nv, Dv, Dk) bfloat16 (or float32).
    """
    return torch.ops.npu.chunk_gated_delta_rule(
        query,
        key,
        value,
        beta=beta,
        initial_state=initial_state,
        actual_seq_lengths=actual_seq_lengths,
        scale=scale,
        g=g,
        chunk_state=chunk_state,
    )


def chunk_gated_delta_rule_pool_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state_pool: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    *,
    src_pool: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    actual_seq_lengths: torch.Tensor | None = None,
    scale: float | None = None,
    g: torch.Tensor | None = None,
    chunk_state: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run GDN prefill with indexed, read-only sources and in-place destinations.

    Pools have shape (P, Nv, Dv, Dk). Slot and source-head strides may describe
    the real pool layout, but each [Dv, Dk] matrix must be row-contiguous and
    each destination slot must be one contiguous [Nv, Dv, Dk] block with a
    64-byte-aligned base and slot stride. Index tensors are contiguous int32
    arrays of length B. A negative source slot or a false
    has_initial_state entry starts that row from zero; a negative destination
    slot skips that row without writing state. Destination slots must be unique
    and must not alias a source slot that another row still needs.
    """
    return torch.ops.npu.chunk_gated_delta_rule_pool(
        query,
        key,
        value,
        state_pool,
        src_slots,
        dst_slots,
        src_pool=src_pool,
        beta=beta,
        actual_seq_lengths=actual_seq_lengths,
        scale=scale,
        g=g,
        chunk_state=chunk_state,
        has_initial_state=has_initial_state,
    )

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

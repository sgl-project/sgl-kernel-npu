"""Independent PyTorch algebra for the attention CGDR interface.

States use Dv x Dk orientation. FP32 chunk algebra expresses the token recurrence;
it does not reproduce custom Kernel intermediate BF16 storage or rounding.
No custom operator, ACLNN interface or other implementation backend is called.
"""

import torch


def chunk_gated_delta_rule(
    query,
    key,
    value,
    *,
    beta=None,
    initial_state=None,
    actual_seq_lengths=None,
    scale=None,
    g=None,
    chunk_state=None,
):
    """Return BF16 output and final state in initial_state's dtype.

    Required-in-practice optional arguments and shape/dtype constraints follow
    the Ascend C contract. Direct source loading also permits CPU algebra checks;
    it does not establish NPU support. Contiguous chunk_state receives entering
    states; a noncontiguous chunk buffer gets a temporary with no copyback.
    """
    # Validate the public tensor contract without importing NPU infrastructure.
    for name, tensor in (
        ("beta", beta),
        ("initial_state", initial_state),
        ("actual_seq_lengths", actual_seq_lengths),
    ):
        if tensor is None:
            raise ValueError(f"{name} must be provided")
    tensors = (query, key, value, beta, initial_state, actual_seq_lengths)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError(
            "query/key/value/beta/initial_state/actual_seq_lengths must be tensors"
        )
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query/key/value must be 3D")
    if beta.ndim != 2 or initial_state.ndim != 4 or actual_seq_lengths.ndim != 1:
        raise ValueError(
            "beta, initial_state, actual_seq_lengths must have ranks 2, 4, 1"
        )
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value, beta)):
        raise ValueError("query/key/value/beta must be bfloat16")
    if initial_state.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("initial_state must be bfloat16 or float32")
    if actual_seq_lengths.dtype != torch.int32:
        raise ValueError("actual_seq_lengths must be int32")
    t, nk, dk = query.shape
    _, nv, dv = value.shape
    b = initial_state.shape[0]
    if min(t, nk, dk, nv, dv, b) <= 0 or nk > 64 or nv > 64 or dk > 128 or dv > 128:
        raise ValueError("positive dimensions required; Nk/Nv <= 64 and Dk/Dv <= 128")
    if nv % nk:
        raise ValueError("nv must be an integer multiple of nk")
    if key.shape != query.shape or value.shape[0] != t or beta.shape != (t, nv):
        raise ValueError("query/key/value/beta shapes do not match")
    if initial_state.shape != (b, nv, dv, dk) or actual_seq_lengths.shape != (b,):
        raise ValueError("initial_state/actual_seq_lengths shapes do not match")
    if g is not None and (g.dtype != torch.float32 or g.shape != (t, nv)):
        raise ValueError("g must be float32 (T, Nv)")
    if chunk_state is not None and (
        chunk_state.ndim != 4
        or chunk_state.shape[1:] != (nv, dv, dk)
        or chunk_state.dtype != initial_state.dtype
    ):
        raise ValueError(
            "chunk_state must match (totalChunks, Nv, Dv, Dk) and initial_state dtype"
        )
    for tensor in (*tensors, g, chunk_state):
        if tensor is not None and tensor.device != query.device:
            raise ValueError("all CGDR inputs must be on the same device")
    lengths = actual_seq_lengths.tolist()
    if any(length < 0 for length in lengths) or sum(lengths) != t:
        raise ValueError("actual_seq_lengths must be nonnegative and sum to T")

    effective_scale = 1.0 if scale is None else scale
    out = torch.empty(value.shape, dtype=value.dtype, device=value.device)
    final_state = torch.empty(
        initial_state.shape, dtype=initial_state.dtype, device=query.device
    )
    chunk_work = None if chunk_state is None else chunk_state.contiguous()
    group = nv // nk
    offset, chunk_index = 0, 0
    for sequence, length in enumerate(lengths):
        # Own the working state even when float() returns an FP32 input alias.
        state = initial_state[sequence].float().clone()
        for start in range(offset, offset + length, 64):
            end = min(start + 64, offset + length)
            size = end - start
            if chunk_work is not None:
                chunk_work[chunk_index].copy_(state.to(initial_state.dtype))
            chunk_index += 1

            # Map grouped heads, then place heads first for batched matrix products.
            q = query[start:end].float().repeat_interleave(group, dim=1).transpose(0, 1)
            k = key[start:end].float().repeat_interleave(group, dim=1).transpose(0, 1)
            v = value[start:end].float().transpose(0, 1)
            beta_chunk = beta[start:end].float().transpose(0, 1)
            cumulative = (
                torch.zeros((nv, size), dtype=torch.float32, device=query.device)
                if g is None
                else g[start:end].transpose(0, 1).cumsum(-1)
            )
            decay = cumulative.exp()
            causal = torch.ones(
                (size, size), dtype=torch.bool, device=query.device
            ).tril()
            differences = cumulative[:, :, None] - cumulative[:, None, :]
            # Mask before exp: discarded upper entries must not form inf * 0.
            relative_decay = differences.masked_fill(~causal, -torch.inf).exp()
            state_transposed = state.transpose(-1, -2)

            # Unit lower-triangular solve A U = beta * (V - exp(G) K S^T).
            lower = (
                (k @ k.transpose(-1, -2)) * relative_decay * beta_chunk[:, :, None]
            ).tril(-1)
            rhs = beta_chunk[:, :, None] * (
                v - decay[:, :, None] * (k @ state_transposed)
            )
            rows = []
            for row in range(size):
                solved = rhs[:, row]
                if row:
                    solved = solved - (
                        lower[:, row : row + 1, :row] @ torch.stack(rows, dim=1)
                    ).squeeze(1)
                rows.append(solved)
            updates = torch.stack(rows, dim=1)

            # Causal output uses the entering state; update the exiting state next.
            attention = (q @ k.transpose(-1, -2)) * relative_decay
            chunk_out = effective_scale * (
                decay[:, :, None] * (q @ state_transposed) + attention @ updates
            )
            out[start:end].copy_(chunk_out.transpose(0, 1).to(value.dtype))
            last = cumulative[:, -1]
            weighted_keys = (last[:, None] - cumulative).exp()[:, :, None] * k
            state = (
                last.exp()[:, None, None] * state
                + updates.transpose(-1, -2) @ weighted_keys
            )
        final_state[sequence].copy_(state.to(initial_state.dtype))
        offset += length
    return out, final_state

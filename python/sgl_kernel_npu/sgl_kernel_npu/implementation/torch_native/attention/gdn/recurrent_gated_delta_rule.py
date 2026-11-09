"""Independent packed RGDR equations and state protocol; no custom backend calls.

Protocol source: csrc/recurrent_gated_delta_rule at upstream dc112d9ef68c.
The old test oracle uses a different L2 epsilon/sequence-offset expression;
this reference follows the actual kernel, including FP32 state across tokens.
"""

import torch


def _present(tensor):
    return tensor is not None


def _raw_dense(tensor):
    stride = 1
    strides = []
    for size in reversed(tensor.shape):
        strides.append(stride)
        stride *= size
    return tensor.as_strided(tensor.shape, tuple(reversed(strides)))


@torch.no_grad()
def recurrent_gated_delta_rule(
    mix_qkv,
    recurrent_state,
    beta,
    scale,
    actual_seq_lengths,
    ssm_state_indices,
    nk,
    nv,
    intermediate_state=None,
    cache_indices=None,
    num_accepted_tokens=None,
    g=None,
    gk=None,
):
    if mix_qkv.ndim != 3 or recurrent_state.ndim != 4 or beta.ndim != 3:
        raise ValueError("mix_qkv/beta must be 3D and recurrent_state must be 4D")
    if any(t.dtype != torch.bfloat16 for t in (mix_qkv, recurrent_state, beta)):
        raise ValueError("RGDR mix_qkv, beta and recurrent_state must be bfloat16")
    b, s, packed = mix_qkv.shape
    n, state_nv, dv, dk = recurrent_state.shape
    if min(b, s, nk, nv, dv, dk, n) <= 0 or s > 8 or nv % nk:
        raise ValueError(
            "RGDR requires positive dimensions, S <= 8 and nv divisible by nk"
        )
    if packed != 2 * nk * dk + nv * dv or state_nv != nv or beta.shape != (b, s, nv):
        raise ValueError("packed QKV, beta or state shape mismatch")
    if (
        actual_seq_lengths.dtype != torch.int32
        or ssm_state_indices.dtype != torch.int32
    ):
        raise ValueError("actual_seq_lengths and ssm_state_indices must be int32")
    if actual_seq_lengths.shape != (b,) or ssm_state_indices.shape != (b, s):
        raise ValueError("actual_seq_lengths must be (B,) and indices must be (B,S)")
    device = mix_qkv.device
    for tensor in (
        recurrent_state,
        beta,
        actual_seq_lengths,
        ssm_state_indices,
        intermediate_state,
        cache_indices,
        num_accepted_tokens,
        g,
        gk,
    ):
        if tensor is not None and tensor.device != device:
            raise ValueError("all RGDR inputs must be on the same device")
    recurrent = _raw_dense(recurrent_state)
    work = recurrent
    if _present(intermediate_state):
        if intermediate_state.dtype != torch.bfloat16:
            raise ValueError("intermediate_state must be bfloat16")
        if intermediate_state.numel() == 0 or intermediate_state.numel() % (
            nv * dv * dk
        ):
            raise ValueError(
                "intermediate_state must contain complete (nv,dv,dk) slots"
            )
        if cache_indices is None or cache_indices.numel() < b:
            raise ValueError("cache_indices must provide B recurrent seed slots")
        # contiguous() aliases an already contiguous caller buffer; otherwise it
        # creates the same intentionally unreturned/no-copyback work buffer as old Host.
        work = intermediate_state.contiguous().view(-1, nv, dv, dk)
    cache = (
        cache_indices.to(torch.int32).contiguous().view(-1)
        if cache_indices is not None
        else None
    )
    if num_accepted_tokens is not None and num_accepted_tokens.numel() < b:
        raise ValueError("num_accepted_tokens must contain B values")
    accepted = (
        num_accepted_tokens.to(torch.int32).contiguous().view(-1)
        if num_accepted_tokens is not None
        else None
    )
    if g is not None and g.numel() < b * s * nv:
        raise ValueError("g must provide B*S*nv log-decay values")
    gamma = (
        g.float().contiguous().view(-1)[: b * s * nv].view(b, s, nv).exp()
        if g is not None
        else None
    )
    if gk is not None:
        gk = (
            gk.float().contiguous()
        )  # Preserve conversion; the legacy recurrence ignores it.

    x = _raw_dense(mix_qkv).float()
    q, k, v = torch.split(x, [nk * dk, nk * dk, nv * dv], dim=-1)
    q, k, v = q.reshape(b, s, nk, dk), k.reshape(b, s, nk, dk), v.reshape(b, s, nv, dv)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    # The old launch narrows the Python double to a float scalar.
    scale_value = torch.tensor(scale, dtype=torch.float32, device="cpu").item()
    q = q * scale_value
    betas = _raw_dense(beta).float().sigmoid()
    lengths, indices = _raw_dense(actual_seq_lengths), _raw_dense(ssm_state_indices)
    initial_work = work.clone()
    out = mix_qkv.new_empty((b, s, nv, dv))
    heads = torch.arange(nv, dtype=torch.int64, device=device) // (nv // nk)
    for seq in range(b):
        length = int(lengths[seq].item())
        if not 1 <= length <= s:
            raise ValueError("reference requires 1 <= actual_seq_lengths[b] <= S")
        column = int(accepted[seq].item()) - 1 if accepted is not None else 0
        if not 0 <= column < s:
            raise ValueError("reference requires accepted counts in [1,S]")
        slot = int(indices[seq, column].item())
        if not 0 <= slot < work.shape[0]:
            raise ValueError("state index outside working pool")
        if intermediate_state is not None and slot % s == 0:
            seed_slot = int(cache[seq].item())
            if not 0 <= seed_slot < recurrent.shape[0]:
                raise ValueError("recurrent seed slot outside pool")
            h = recurrent[seed_slot].float().clone()
        else:
            h = initial_work[slot].float().clone()
        for token in range(length):
            qt = q[seq, token].index_select(0, heads)
            kt = k[seq, token].index_select(0, heads)
            if gamma is not None:
                h = h * gamma[seq, token, :, None, None]
            delta = (v[seq, token] - (h * kt[:, None, :]).sum(-1)) * betas[
                seq, token, :, None
            ]
            h = h + delta[:, :, None] * kt[:, None, :]
            target = int(indices[seq, token].item())
            if not 0 <= target < work.shape[0]:
                raise ValueError("state output index outside working pool")
            work[target] = h.to(work.dtype)
            out[seq, token] = (h * qt[:, None, :]).sum(-1).to(out.dtype)
    return out

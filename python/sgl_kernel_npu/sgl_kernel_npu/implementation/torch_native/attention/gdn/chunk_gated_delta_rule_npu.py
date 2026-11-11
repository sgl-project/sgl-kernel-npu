"""Independent CPU FP64 recurrence for the composed GDN interface."""

import os

import torch


def chunk_gated_delta_rule_npu(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=True,
    cu_seqlens=None,
    head_first=False,
    use_qk_l2norm_in_kernel=False,
):
    """Return output, optional final state and entering chunk states."""
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    dtype, device = q.dtype, q.device
    q, k, v, g, beta = (x.detach().cpu() for x in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        q, k = (
            (
                x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)
            ).to(dtype)
            for x in (q, k)
        )
    if os.getenv("GDN_USE_MEGA_GDN", "0") == "1":
        from .mega_chunk_gdn import run_mega_chunk_gdn

        result = run_mega_chunk_gdn(
            q,
            k,
            v,
            g,
            beta,
            scale,
            None if initial_state is None else initial_state.detach().cpu(),
            output_final_state,
            None if cu_seqlens is None else cu_seqlens.detach().cpu(),
        )
        return tuple(
            None if value is None else value.to(device)
            for value in (result[1], result[3], result[5])
        )
    chunk_size = 64
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    batch, tokens, groups, key_dim = k.shape
    heads, value_dim = v.shape[-2:]
    q, k = (x.repeat_interleave(heads // groups, dim=2) for x in (q, k))
    scale = key_dim**-0.5 if scale is None else scale
    if cu_seqlens is None:
        sequences = [(b, 0, tokens) for b in range(batch)]
    else:
        bounds = cu_seqlens.cpu().tolist()
        sequences = [(0, start, end) for start, end in zip(bounds, bounds[1:])]
    output = torch.empty_like(v)
    final_states, chunks = [], []
    for sequence, (b, start, end) in enumerate(sequences):
        state = (
            torch.zeros(heads, key_dim, value_dim, dtype=torch.float64, device="cpu")
            if initial_state is None
            else initial_state[sequence].detach().cpu().double().clone()
        )
        for t in range(start, end):
            if (t - start) % chunk_size == 0:
                chunks.append(state.to(dtype))
            state = state * g[b, t].exp()[:, None, None]
            residual = v[b, t] - torch.einsum("hk,hkv->hv", k[b, t], state)
            state = state + torch.einsum(
                "hk,hv->hkv", k[b, t], residual * beta[b, t, :, None]
            )
            output[b, t] = torch.einsum("hk,hkv->hv", q[b, t], state) * scale
        final_states.append(state.float())
    h = torch.stack(chunks).reshape(batch, -1, heads, key_dim, value_dim)
    return (
        output.to(device=device, dtype=dtype),
        torch.stack(final_states).to(device) if output_final_state else None,
        h.to(device),
    )

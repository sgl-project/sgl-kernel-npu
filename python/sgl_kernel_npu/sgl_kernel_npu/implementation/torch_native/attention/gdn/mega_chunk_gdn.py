"""Independent CPU FP64 chunk algebra for the fused 128-token GDN pipeline."""

import torch


def run_mega_chunk_gdn(
    q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens
):
    device = q.device
    q_dtype, k_dtype, v_dtype = q.dtype, k.dtype, v.dtype
    # Match the helper's entry conversion, then compute independently on CPU.
    q, k, v, beta = (x.detach().cpu().half().double() for x in (q, k, v, beta))
    g = g.detach().cpu().double()
    _, tokens, groups, dim = q.shape
    heads = v.shape[2]
    scale = dim**-0.5 if scale is None else scale
    bounds = [0, tokens] if cu_seqlens is None else cu_seqlens.detach().cpu().tolist()
    state0 = (
        None if initial_state is None else initial_state.detach().cpu().half().double()
    )
    g_sum = torch.empty_like(g)
    output = torch.empty_like(v)
    inverse = torch.zeros(1, tokens, heads, 128, dtype=torch.float64, device="cpu")
    w = torch.empty_like(v)
    v_new = torch.empty_like(v)
    states, final = [], []
    for sequence, (start, end) in enumerate(zip(bounds, bounds[1:])):
        state = (
            torch.zeros(heads, dim, dim, dtype=torch.float64, device="cpu")
            if state0 is None
            else state0[sequence].clone()
        )
        for pos in range(start, end, 128):
            stop = min(pos + 128, end)
            n = stop - pos
            cumulative = g[0, pos:stop].transpose(0, 1).cumsum(-1)
            g_sum[0, pos:stop] = cumulative.transpose(0, 1)
            keys = (
                k[0, pos:stop].repeat_interleave(heads // groups, dim=1).transpose(0, 1)
            )
            queries = (
                q[0, pos:stop].repeat_interleave(heads // groups, dim=1).transpose(0, 1)
            )
            values = v[0, pos:stop].transpose(0, 1)
            weights = beta[0, pos:stop].transpose(0, 1)
            relative = (
                (cumulative[:, :, None] - cumulative[:, None, :]).tril().exp().tril()
            )
            lower = (
                weights[:, :, None] * (keys @ keys.transpose(-1, -2)) * relative
            ).tril(-1)
            identity = torch.eye(n, dtype=torch.float64, device="cpu").expand(
                heads, n, n
            )
            a = torch.linalg.solve_triangular(identity + lower, identity, upper=False)
            inverse[0, pos:stop, :, :n] = a.transpose(0, 1)
            transform = a * (weights * cumulative.exp())[:, None, :]
            wc = transform @ keys
            uc = (a * weights[:, None, :]) @ values
            w[0, pos:stop] = wc.transpose(0, 1)
            states.append(state.clone())
            residual = uc - wc @ state
            v_new[0, pos:stop] = residual.transpose(0, 1)
            attention = (queries @ keys.transpose(-1, -2)) * relative
            output[0, pos:stop] = (
                (queries @ state) * cumulative.exp()[:, :, None] + attention @ residual
            ).transpose(0, 1)
            last = cumulative[:, -1]
            state = (
                state * last.exp()[:, None, None]
                + (keys * (last[:, None] - cumulative).exp()[:, :, None]).transpose(
                    -1, -2
                )
                @ residual
            )
        final.append(state)

    def half_cast(x, dtype):
        return x.half().to(device=device, dtype=dtype)

    # Native scale multiplication happens on its FP16 output before q.dtype cast.
    out = (output.half() * scale).to(device=device, dtype=q_dtype)
    final_state = (
        half_cast(torch.stack(final), torch.float32) if output_final_state else None
    )
    h = torch.stack(states).unsqueeze(0)
    return (
        g_sum.float().to(device),
        out,
        half_cast(inverse, k_dtype),
        final_state,
        half_cast(w, k_dtype),
        half_cast(h, k_dtype),
        half_cast(v_new, v_dtype),
    )

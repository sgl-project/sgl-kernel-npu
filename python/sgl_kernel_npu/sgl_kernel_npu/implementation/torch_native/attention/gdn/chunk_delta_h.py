"""CPU FP64 reference retaining the kernel's two low-precision rounding points."""

import torch


def chunk_delta_h(
    k,
    w,
    u,
    g=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    cu_seqlens=None,
):
    if g is None:
        raise ValueError("g is required by the current Triton wrapper")
    output_device = k.device
    k, w, u, g = (x.detach().cpu() for x in (k, w, u, g))
    if initial_state is not None:
        initial_state = initial_state.detach().cpu()
    batch, length, groups, key_dim = k.shape
    heads, value_dim = u.shape[-2:]
    if cu_seqlens is None:
        sequences = [(b, 0, length) for b in range(batch)]
        num_chunks = (length + chunk_size - 1) // chunk_size
    else:
        if batch != 1:
            raise ValueError("variable-length input requires batch size 1")
        offsets = cu_seqlens.tolist()
        sequences = [(0, start, end) for start, end in zip(offsets, offsets[1:])]
        num_chunks = sum(
            (end - start + chunk_size - 1) // chunk_size for _, start, end in sequences
        )
    h = k.new_empty((batch, num_chunks, heads, key_dim, value_dim))
    v_new = torch.empty_like(u) if save_new_value else None
    final = (
        k.new_empty((len(sequences), heads, key_dim, value_dim), dtype=torch.float32)
        if output_final_state
        else None
    )
    packed_chunk = 0
    for sequence, (b, start, end) in enumerate(sequences):
        states = (
            torch.zeros((heads, value_dim, key_dim), dtype=torch.float64, device="cpu")
            if initial_state is None
            else initial_state[sequence].transpose(-1, -2).double().clone()
        )
        for local_chunk, pos in enumerate(range(start, end, chunk_size)):
            stop = min(pos + chunk_size, end)
            out_index = packed_chunk if cu_seqlens is not None else local_chunk
            for head in range(heads):
                current = states[head]
                h[b, out_index, head] = current.T.to(k.dtype)
                # SGLang computes u - w @ state.T before applying gate decay.
                # Round the state to the input dtype before the FP64 matmul.
                delta = u[b, pos:stop, head].double() - (
                    w[b, pos:stop, head].double() @ current.T.to(w.dtype).double()
                )
                if save_new_value:
                    v_new[b, pos:stop, head] = delta.to(u.dtype)
                gate = g[b, pos:stop, head].double()
                diff = gate[-1] - gate
                decay = torch.where(diff <= 0, diff.exp(), 0.0)
                # Retain the second rounding point after applying gate decay.
                delta = (delta * decay[:, None]).to(k.dtype).double()
                keys = k[b, pos:stop, head // (heads // groups)].double()
                states[head] = current * gate[-1].exp() + delta.T @ keys
            packed_chunk += 1
        if final is not None:
            final[sequence] = states.transpose(-1, -2)
    return (
        h.to(output_device),
        v_new.to(output_device) if v_new is not None else None,
        final.to(output_device) if final is not None else None,
    )

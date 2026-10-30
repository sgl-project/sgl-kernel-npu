"""Reference state recurrence for chunked gated delta rule."""

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
            torch.zeros(
                (heads, key_dim, value_dim), dtype=torch.float32, device=k.device
            )
            if initial_state is None
            else initial_state[sequence].float().clone()
        )
        for local_chunk, pos in enumerate(range(start, end, chunk_size)):
            stop = min(pos + chunk_size, end)
            out_index = packed_chunk if cu_seqlens is not None else local_chunk
            for head in range(heads):
                current = states[head]
                h[b, out_index, head] = current.to(k.dtype)
                delta = u[b, pos:stop, head].float() - (
                    w[b, pos:stop, head].float() @ current.to(w.dtype).float()
                )
                if save_new_value:
                    v_new[b, pos:stop, head] = delta.to(u.dtype)
                gate = g[b, pos:stop, head].float()
                diff = gate[-1] - gate
                decay = torch.where(diff <= 0, diff.exp(), 0.0)
                delta = (delta * decay[:, None]).to(k.dtype).float()
                keys = k[b, pos:stop, head // (heads // groups)].float()
                states[head] = current * gate[-1].exp() + keys.T @ delta
            packed_chunk += 1
        if final is not None:
            final[sequence] = states
    return h, v_new, final

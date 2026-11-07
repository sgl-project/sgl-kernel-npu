import math
from typing import Optional

import torch
import torch.nn.functional as F


def _check_inputs(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    initial_state,
    cu_seqlens,
    chunk_indices,
    layout,
    chunk_size,
    safe_gate,
    lower_bound,
    use_gate_in_kernel,
    state_v_first,
):
    def check(condition, message):
        if not condition:
            raise RuntimeError("chunk_kda_fwd: " + message)

    check(chunk_size in (64, 128), "chunkSize must be 64 or 128")
    check(
        layout in ("BSND", "BNSD", "TND", "NTD"),
        "layout must be BSND, BNSD, TND or NTD",
    )
    check(
        all(x.dtype == torch.bfloat16 for x in (q, k, v)),
        "q, k and v must be bfloat16 (direct-launch currently supports bf16)",
    )
    check(g.dtype in (torch.float32, torch.bfloat16), "g must be float32 or bfloat16")
    check(
        beta.dtype in (torch.float32, torch.bfloat16),
        "beta must be float32 or bfloat16",
    )
    for name, value in (
        ("a_log", a_log),
        ("dt_bias", dt_bias),
        ("initial_state", initial_state),
    ):
        if value is not None:
            check(value.dtype == torch.float32, name + " must be float32")
    if use_gate_in_kernel:
        check(a_log is not None, "a_log is required when use_gate_in_kernel is true")
    rank3 = layout in ("TND", "NTD")
    rank = 3 if rank3 else 4
    check(
        all(x.ndim == rank for x in (q, k, v, g)) and beta.ndim == rank - 1,
        "q/k/v/g and beta ranks must match layout: rank3/rank2 for TND/NTD, rank4/rank3 for BSND/BNSD.",
    )
    if rank3:
        heads, tokens = q.shape[:2] if layout == "NTD" else (q.shape[1], q.shape[0])
        values = v.shape[0] if layout == "NTD" else v.shape[1]
        batch = 1
    else:
        batch = q.shape[0]
        tokens, heads = q.shape[1:3] if layout == "BSND" else (q.shape[2], q.shape[1])
        values = v.shape[2] if layout == "BSND" else v.shape[1]
    kd, vd = q.shape[-1], v.shape[-1]
    leading = (
        (tokens, values)
        if layout == "TND"
        else (
            (values, tokens)
            if layout == "NTD"
            else (
                (batch, tokens, values) if layout == "BSND" else (batch, values, tokens)
            )
        )
    )
    check(
        tuple(v.shape) == (*leading, vd)
        and tuple(g.shape) == (*leading, kd)
        and tuple(beta.shape) == leading,
        {
            "BSND": "BSND expects v/g/beta as [B,T,HV,V], [B,T,HV,K], [B,T,HV].",
            "BNSD": "BNSD expects v/g/beta as [B,HV,T,V], [B,HV,T,K], [B,HV,T].",
            "TND": "TND expects v/g/beta as [T,HV,V], [T,HV,K], [T,HV].",
            "NTD": "NTD expects v/g/beta as [HV,T,V], [HV,T,K], [HV,T].",
        }[layout],
    )
    check(
        heads > 0 and values >= heads and values % heads == 0,
        "H and HV must be positive, HV must be a multiple of H",
    )
    check(heads <= 128 and values <= 128, "H and HV must be <= 128")
    check(
        16 <= kd <= 256 and kd % 16 == 0 and 16 <= vd <= 256 and vd % 16 == 0,
        "K/V must be multiples of 16, K <= 256 and V <= 256",
    )
    check(q.shape == k.shape, "q and k must have identical shape")
    check(tokens > 0, "invalid sequence length")
    sequences = batch
    if cu_seqlens is not None:
        check(cu_seqlens.dtype == torch.int64, "cu_seqlens must be int64")
        check(cu_seqlens.numel() >= 2, "cu_seqlens must contain at least [0, T]")
        sequences = cu_seqlens.numel() - 1
        check(sequences <= 1024, "varlen supports at most 1024 sequences")
        bounds = cu_seqlens.reshape(-1).tolist()
        check(bounds[0] == 0, "cu_seqlens[0] must be 0")
        check(
            bounds[-1] == tokens,
            "cu_seqlens last element must equal the sequence length",
        )
        check(
            all(a <= b for a, b in zip(bounds[:-1], bounds[1:])),
            "cu_seqlens must be nondecreasing",
        )
    else:
        bounds = [0, tokens]
    chunks = sum(
        (b - a + chunk_size - 1) // chunk_size for a, b in zip(bounds[:-1], bounds[1:])
    )
    if chunk_indices is not None:
        check(cu_seqlens is not None, "chunk_indices requires cu_seqlens")
        check(chunk_indices.dtype == torch.int64, "chunk_indices must be int64")
        check(
            chunk_indices.numel() == chunks * 2,
            "chunk_indices must contain one (seq_id, chunk_id) pair per chunk",
        )
    if use_gate_in_kernel and safe_gate:
        check(
            -5.0 <= lower_bound < 0.0,
            "lower_bound must be in [-5, 0) when safe_gate is true",
        )
    if initial_state is not None:
        shape = (
            (sequences, values, vd, kd)
            if state_v_first
            else (sequences, values, kd, vd)
        )
        check(
            tuple(initial_state.shape) == shape,
            "initial_state must be [N,HV,K,V] (state_v_first=false) or [N,HV,V,K] (state_v_first=true)",
        )
    return batch, tokens, heads, values, kd, vd, bounds, chunks


def _sequence_first(value, layout):
    if layout == "BSND":
        return value
    if layout == "BNSD":
        return value.transpose(1, 2)
    if layout == "TND":
        return value.unsqueeze(0)
    return value.transpose(0, 1).unsqueeze(0)


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    layout: str = "BSND",
    scale: float = 1.0,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    use_gate_in_kernel: bool = False,
    state_v_first: bool = False,
    output_final_state: bool = True,
    output_gk: bool = False,
    output_w: bool = False,
    output_u: bool = False,
    output_qg: bool = False,
    output_kg: bool = False,
    output_v_new: bool = False,
    output_h: bool = False
):
    batch, tokens, heads, values, kd, vd, bounds, chunks = _check_inputs(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        initial_state,
        cu_seqlens,
        chunk_indices,
        layout,
        chunk_size,
        safe_gate,
        lower_bound,
        use_gate_in_kernel,
        state_v_first,
    )
    rank3 = layout in ("TND", "NTD")
    query, key, value, gate, betas = [
        _sequence_first(t, layout).float() for t in (q, k, v, g, beta)
    ]
    query = query.repeat_interleave(values // heads, dim=2)
    key = key.repeat_interleave(values // heads, dim=2)
    if use_gate_in_kernel:
        if dt_bias is not None:
            gate = gate + dt_bias.reshape(1, 1, values, kd)
        decay = a_log.reshape(1, 1, values, 1).exp()
        gate = (
            lower_bound * torch.sigmoid(decay * gate)
            if safe_gate
            else -decay * F.softplus(gate)
        )
    out = torch.empty((batch, tokens, values, vd), dtype=q.dtype, device=q.device)
    aqk = torch.zeros(
        (batch, values, tokens, chunk_size), dtype=q.dtype, device=q.device
    )
    akk = torch.zeros_like(aqk)
    gk = torch.empty((batch, values, tokens, kd), dtype=torch.float32, device=q.device)
    w = torch.empty((batch, values, tokens, kd), dtype=q.dtype, device=q.device)
    u = torch.empty((batch, values, tokens, vd), dtype=q.dtype, device=q.device)
    qg = torch.empty_like(w)
    kg = torch.empty_like(w)
    v_new = torch.empty_like(u)
    h = torch.empty((batch, chunks, values, kd, vd), dtype=q.dtype, device=q.device)
    sequence_count = batch if cu_seqlens is None else len(bounds) - 1
    state = (
        torch.zeros(
            (sequence_count, values, kd, vd), dtype=torch.float32, device=q.device
        )
        if initial_state is None
        else (
            initial_state.transpose(-1, -2) if state_v_first else initial_state
        ).clone()
    )
    for b in range(batch):
        chunk_id = 0
        sequences = (
            [(0, tokens)] if cu_seqlens is None else list(zip(bounds[:-1], bounds[1:]))
        )
        for request, (bos, eos) in enumerate(sequences):
            state_id = b if cu_seqlens is None else request
            current = state[state_id]
            for start in range(bos, eos, chunk_size):
                end = min(start + chunk_size, eos)
                size = end - start
                qc = query[b, start:end].transpose(0, 1)
                kc = key[b, start:end].transpose(0, 1)
                vc = value[b, start:end].transpose(0, 1)
                bc = betas[b, start:end].transpose(0, 1)
                gc = (gate[b, start:end].transpose(0, 1) * 1.4426950408889634).cumsum(1)
                gk[b, :, start:end] = gc
                gn = gc * 0.6931471805599453
                difference = gn[:, :, None, :] - gn[:, None, :, :]
                decay_matrix = difference.clamp(
                    -80 * math.log(2), 80 * math.log(2)
                ).exp()
                scores = torch.einsum("hik,hjk,hijk->hij", qc, kc, decay_matrix)
                gram = torch.einsum("hik,hjk,hijk->hij", kc, kc, decay_matrix)
                causal = (scores * scale).tril().to(q.dtype)
                coefficients = (gram * bc[:, :, None]).tril(-1)
                inverse = (
                    torch.eye(size, device=q.device, dtype=torch.float32)
                    .expand(values, -1, -1)
                    .clone()
                )
                for row in range(1, size):
                    inverse[:, row, :row] = -(
                        coefficients[:, row, :row, None] * inverse[:, :row, :row]
                    ).sum(1)
                inverse = inverse.to(q.dtype)
                aqk[b, :, start:end, :size] = causal
                akk[b, :, start:end, :size] = inverse
                eg = gn.clamp(-80 * math.log(2), 80 * math.log(2)).exp()
                wc = (
                    inverse.float() @ (kc * eg * bc[:, :, None]).to(q.dtype).float()
                ).to(q.dtype)
                uc = (inverse.float() @ (vc * bc[:, :, None]).to(q.dtype).float()).to(
                    q.dtype
                )
                qgc = (qc * eg).to(q.dtype)
                kgc = (
                    kc
                    * (gn[:, -1:, :] - gn)
                    .clamp(-80 * math.log(2), 80 * math.log(2))
                    .exp()
                ).to(q.dtype)
                entering = current.to(q.dtype)
                new_value = (uc.float() - wc.float() @ entering.float()).to(q.dtype)
                h[b, chunk_id] = entering
                w[b, :, start:end] = wc
                u[b, :, start:end] = uc
                qg[b, :, start:end] = qgc
                kg[b, :, start:end] = kgc
                v_new[b, :, start:end] = new_value
                scaled = (qc * scale * eg).to(q.dtype)
                result = (
                    scaled.float() @ entering.float()
                    + causal.float() @ new_value.float()
                )
                out[b, start:end] = result.transpose(0, 1).to(q.dtype)
                current = (
                    current * gn[:, -1].exp()[:, :, None]
                    + kgc.float().transpose(-1, -2) @ new_value.float()
                )
                chunk_id += 1
            state[state_id] = current
    if state_v_first:
        state = state.transpose(-1, -2).contiguous()
        h = h.transpose(-1, -2).contiguous()
    if rank3:
        out = out.squeeze(0)
        gk, aqk, akk, w, u, qg, kg, v_new, h = [
            t.squeeze(0) for t in (gk, aqk, akk, w, u, qg, kg, v_new, h)
        ]
    return (
        out,
        state if output_final_state else None,
        gk if output_gk else None,
        aqk,
        akk,
        w if output_w else None,
        u if output_u else None,
        qg if output_qg else None,
        kg if output_kg else None,
        v_new if output_v_new else None,
        h if output_h else None,
    )

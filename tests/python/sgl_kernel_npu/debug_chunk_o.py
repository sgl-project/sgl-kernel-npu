import torch
import torch.nn.functional as F

import sgl_kernel_npu  # noqa: F401
from debug_chunk_h import ref_chunk_h, ref_cumsum, stats_ok, total_chunks, transpose_gates


def ref_chunk_o(q, k, v_new, h_states, g_cumsum, chunk_size, cu_seqlens):
    _, _, hg, d = q.shape
    h = v_new.shape[2]
    group = h // hg
    qf, kf, vf, gf = q.float(), k.float(), v_new.float(), g_cumsum.float()
    cu = cu_seqlens.cpu().tolist()
    out = torch.zeros_like(vf)

    chunk_base = 0
    for bos, eos in zip(cu, cu[1:]):
        num_chunks = (eos - bos + chunk_size - 1) // chunk_size
        for head in range(h):
            key_head = head // group
            for chunk_idx in range(num_chunks):
                start = bos + chunk_idx * chunk_size
                end = min(start + chunk_size, eos)
                length = end - start
                qc = qf[0, start:end, key_head]
                kc = kf[0, start:end, key_head]
                vc = vf[0, start:end, head]
                gc = gf[0, start:end, head]
                state = h_states[chunk_base + chunk_idx, head]

                inter = (qc @ state) * torch.exp(gc)[:, None]
                qk = qc @ kc.T
                causal = torch.arange(length)[:, None] >= torch.arange(length)[None, :]
                gate = torch.exp(torch.minimum(gc[:, None] - gc[None, :], torch.zeros(length, length)))
                out[0, start:end, head] = inter + (qk * gate * causal.float()) @ vc
        chunk_base += num_chunks
    return out


def block_dim():
    try:
        return int(getattr(torch.npu.get_device_properties("npu:0"), "cube_core_num", 20))
    except (RuntimeError, AssertionError):
        return 24


def main():
    device = torch.device("npu")
    total_tokens = 256
    chunk_size = 128
    bd = block_dim()
    h = 16
    hg = 16
    d = 128
    cu_cpu = torch.tensor([0, 64, total_tokens], dtype=torch.int32)
    batch_size = cu_cpu.numel() - 1

    torch.manual_seed(42)
    k_cpu = F.normalize(torch.randn(1, total_tokens, hg, d, dtype=torch.float16), dim=-1, p=2)
    q_cpu = F.normalize(torch.randn(1, total_tokens, hg, d, dtype=torch.float16), dim=-1, p=2)
    w_cpu = torch.randn(1, total_tokens, h, d, dtype=torch.float16)
    u_cpu = torch.randn(1, total_tokens, h, d, dtype=torch.float16)
    g_in_cpu = F.logsigmoid(torch.randn(1, total_tokens, h, dtype=torch.float32))
    g_sum_cpu = ref_cumsum(g_in_cpu, chunk_size, cu_cpu)

    num_chunks = total_chunks(cu_cpu, chunk_size)
    s_ref, v_ref, _ = ref_chunk_h(k_cpu, w_cpu, u_cpu, g_sum_cpu, chunk_size, cu_cpu)
    s_cpu = s_ref.reshape(num_chunks * h, d, d).to(torch.float16).contiguous()
    v_new_cpu = v_ref.to(torch.float16).contiguous()

    q = q_cpu.to(device).contiguous()
    k = k_cpu.to(device).contiguous()
    v_new = v_new_cpu.to(device).contiguous()
    s = s_cpu.to(device).contiguous()
    g_t = transpose_gates(g_sum_cpu).to(device).contiguous()
    cu = cu_cpu.to(device).contiguous()
    mask = torch.tril(torch.ones(chunk_size, chunk_size), diagonal=0).float().to(device).contiguous()
    ws_qk = torch.zeros(bd, chunk_size, chunk_size, device=device, dtype=torch.float16)
    ws_qs = torch.zeros(bd, chunk_size, d, device=device, dtype=torch.float16)
    ws_gated = torch.zeros_like(ws_qk)
    out = torch.empty(1, total_tokens, h, d, device=device, dtype=torch.float16)
    for name, tensor in {
        "q": q, "k": k, "v_new": v_new, "s": s, "g_t": g_t, "mask": mask,
        "ws_qk": ws_qk, "ws_qs": ws_qs, "ws_gated": ws_gated,
        "out": out, "cu": cu,
    }.items():
        assert tensor.is_contiguous(), f"{name} must be contiguous"

    print("launching chunk_o_debug only")
    torch.ops.npu.chunk_o_debug(
        q, k, v_new, s, g_t, mask, ws_qk, ws_qs, ws_gated, out, cu,
        bd, batch_size, total_tokens, total_tokens
    )
    torch.npu.synchronize()

    s_cpu = s.float().cpu().view(num_chunks, h, d, d)
    expected = ref_chunk_o(q_cpu, k_cpu, v_new_cpu, s_cpu, g_sum_cpu, chunk_size, cu_cpu)
    actual = out.float().cpu()
    print("o max_abs:", (actual - expected).abs().max().item())
    print("o ok:", stats_ok(actual, expected.float()))


if __name__ == "__main__":
    main()

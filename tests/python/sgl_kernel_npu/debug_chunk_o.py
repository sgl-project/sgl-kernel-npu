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
    cu = torch.tensor([0, 64, total_tokens], device=device, dtype=torch.int32)
    batch_size = cu.numel() - 1

    torch.manual_seed(42)
    q = F.normalize(torch.randn(1, total_tokens, hg, d, device=device, dtype=torch.float16), dim=-1, p=2)
    k = F.normalize(torch.randn(1, total_tokens, hg, d, device=device, dtype=torch.float16), dim=-1, p=2)
    w = torch.randn(1, total_tokens, h, d, device=device, dtype=torch.float16)
    u = torch.randn(1, total_tokens, h, d, device=device, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, total_tokens, h, device=device, dtype=torch.float32))
    g_sum = ref_cumsum(g_in.cpu(), chunk_size, cu.cpu()).to(device)
    g_t = transpose_gates(g_sum)

    num_chunks = total_chunks(cu, chunk_size)
    s = torch.zeros(num_chunks * h, d, d, device=device, dtype=torch.float16)
    v_new = torch.empty(1, total_tokens, h, d, device=device, dtype=torch.float16)
    final_state = torch.zeros(batch_size * h, d, d, device=device, dtype=torch.float16)
    h_workspace = torch.zeros(bd * 4, d, d, device=device, dtype=torch.float16)
    torch.ops.npu.chunk_h_debug(
        k, w, u, g_t, s, v_new, final_state, h_workspace, cu,
        bd, batch_size, total_tokens, total_tokens
    )
    torch.npu.synchronize()

    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=device), diagonal=0).float()
    ws_qk = torch.zeros(bd, chunk_size, chunk_size, device=device, dtype=torch.float16)
    ws_qs = torch.zeros(bd, chunk_size, d, device=device, dtype=torch.float16)
    ws_gated = torch.zeros_like(ws_qk)
    out = torch.empty(1, total_tokens, h, d, device=device, dtype=torch.float16)
    torch.ops.npu.chunk_o_debug(
        q, k, v_new, s, g_t, mask, ws_qk, ws_qs, ws_gated, out, cu,
        bd, batch_size, total_tokens, total_tokens
    )
    torch.npu.synchronize()

    s_cpu = s.float().cpu().view(num_chunks, h, d, d)
    expected = ref_chunk_o(q.cpu(), k.cpu(), v_new.cpu(), s_cpu, g_sum.cpu(), chunk_size, cu.cpu())
    actual = out.float().cpu()
    print("o max_abs:", (actual - expected).abs().max().item())
    print("o ok:", stats_ok(actual, expected.float()))


if __name__ == "__main__":
    main()

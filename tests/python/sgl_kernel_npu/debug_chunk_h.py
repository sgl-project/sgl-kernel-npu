import numpy as np
import torch
import torch.nn.functional as F

import sgl_kernel_npu  # noqa: F401

RTOL = 1e-2
ATOL = 1e-5
MAX_RMSE_RATIO = 0.05
MIN_R2 = 0.99
HARD_FAIL_MAX = 1.0


def _r2(y_ref, y_pred):
    ref = y_ref.detach().cpu().numpy().ravel().astype(np.float64)
    pred = y_pred.detach().cpu().numpy().ravel().astype(np.float64)
    ss_res = np.sum((ref - pred) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    return float("nan") if ss_tot < 1e-30 else 1.0 - ss_res / ss_tot


def stats_ok(actual, expected):
    diff = (actual - expected).abs()
    if diff.max().item() > HARD_FAIL_MAX:
        return False
    if (diff <= ATOL + RTOL * expected.abs()).all():
        return True
    mean_abs = float(expected.float().flatten().abs().mean())
    rmse = float(torch.sqrt((diff.float().flatten() ** 2).mean()))
    ratio = rmse / max(mean_abs, 1e-15)
    r2 = _r2(expected, actual)
    if mean_abs < 1e-9:
        return rmse < 5e-4
    return ratio <= MAX_RMSE_RATIO and np.isfinite(r2) and r2 >= MIN_R2


def ref_cumsum(g, chunk_size, cu_seqlens):
    out = torch.zeros_like(g, dtype=torch.float32)
    cu = cu_seqlens.cpu().tolist()
    for bos, eos in zip(cu, cu[1:]):
        for offset in range(0, eos - bos, chunk_size):
            start = bos + offset
            end = min(start + chunk_size, eos)
            out[:, start:end, :] = g.float()[:, start:end, :].cumsum(dim=1)
    return out


def transpose_gates(g_sum):
    return g_sum.squeeze(0).t().contiguous()


def total_chunks(cu_seqlens, chunk_size):
    cu = cu_seqlens.cpu().tolist()
    return sum((e - s + chunk_size - 1) // chunk_size for s, e in zip(cu, cu[1:]))


def ref_chunk_h(k, w, u, g_cumsum, chunk_size, cu_seqlens):
    _, _, hg, d = k.shape
    h = w.shape[2]
    group = h // hg
    kf, wf, uf, gf = k.float(), w.float(), u.float(), g_cumsum.float()
    cu = cu_seqlens.cpu().tolist()

    s_out = torch.zeros(total_chunks(cu_seqlens, chunk_size), h, d, d)
    v_new = torch.zeros_like(uf)
    final = torch.zeros(len(cu) - 1, h, d, d)

    chunk_base = 0
    for seq_idx, (bos, eos) in enumerate(zip(cu, cu[1:])):
        num_chunks = (eos - bos + chunk_size - 1) // chunk_size
        for head in range(h):
            key_head = head // group
            state = torch.zeros(d, d)
            for chunk_idx in range(num_chunks):
                start = bos + chunk_idx * chunk_size
                end = min(start + chunk_size, eos)
                gates = gf[0, start:end, head]
                last_gate = gates[end - start - 1]

                s_out[chunk_base + chunk_idx, head] = state
                vc = uf[0, start:end, head] - wf[0, start:end, head] @ state
                v_new[0, start:end, head] = vc

                decay = torch.exp(last_gate - gates)[:, None]
                kv = kf[0, start:end, key_head].T @ (vc * decay)
                state = torch.exp(last_gate) * state + kv
            final[seq_idx, head] = state
        chunk_base += num_chunks

    return s_out, v_new, final


def main():
    device = torch.device("npu")
    total_tokens = 256
    chunk_size = 128
    try:
        block_dim = int(getattr(torch.npu.get_device_properties("npu:0"), "cube_core_num", 20))
    except (RuntimeError, AssertionError):
        block_dim = 24
    h = 16
    hg = 16
    d = 128
    cu = torch.tensor([0, 64, total_tokens], device=device, dtype=torch.int32)
    batch_size = cu.numel() - 1

    torch.manual_seed(42)
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
    workspace = torch.zeros(block_dim * 4, d, d, device=device, dtype=torch.float16)

    torch.ops.npu.chunk_h_debug(
        k, w, u, g_t, s, v_new, final_state, workspace, cu,
        block_dim, batch_size, total_tokens, total_tokens
    )
    torch.npu.synchronize()

    s_ref, v_ref, _ = ref_chunk_h(k.cpu(), w.cpu(), u.cpu(), g_sum.cpu(), chunk_size, cu.cpu())
    s_actual = s.float().cpu().view(num_chunks, h, d, d)
    v_actual = v_new.float().cpu()

    print("s max_abs:", (s_actual - s_ref).abs().max().item())
    print("v max_abs:", (v_actual - v_ref).abs().max().item())
    print("s ok:", stats_ok(s_actual, s_ref.float()))
    print("v ok:", stats_ok(v_actual, v_ref.float()))


if __name__ == "__main__":
    main()

"""Decisive test: which pool layout does the NPU prefill
(chunk_gated_delta_rule_fwd_h_npu, used by _AscendKDAExtendKernel.extend)
actually consume/produce?

The wrapper does:
    kernel_state = pool.index_select(0, idx).transpose(-1, -2).contiguous()
    <kernel runs on kernel_state as [N, H, K, V] (dk, dv) tiles>
    pool.index_copy_(0, idx, kernel_state.transpose(-1, -2))

Test both pool presentations against a torch reference:
  A. pool = contiguous V-major [H, V, K] with S_true          (memory_pool fix)
  B. pool = transposed view presenting [H, K, V]              (current code)

Run on NPU:
    python tests/python/sgl_kernel_npu/test_kda_prefill_pool_layout.py
"""

import argparse

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.fla.kda_chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h_npu,
)


def ref_chunk_h(k, w, u, gk, initial_state_vm, chunk_size=64):
    """Torch reference of the prefill h-update.

    initial_state_vm: [H, V, K] V-major (the logical state S_true).
    Returns (h [NT, H, V, K] chunk-start states, final [H, V, K]).
    """
    T = k.shape[0]
    h_out = []
    state_vm = initial_state_vm
    n_chunks = (T + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        s0, s1 = c * chunk_size, min((c + 1) * chunk_size, T)
        last = s1 - 1
        state_km = state_vm.transpose(-1, -2)  # [H, K, V] kernel convention
        b_v = u[s0:s1] - torch.einsum("thk,hkv->thv", w[s0:s1], state_km)
        h_out.append(state_vm.clone())
        state_km = state_km * torch.exp2(gk[last])[:, :, None]
        b_v = b_v.to(k.dtype)
        state_km = state_km + torch.einsum("thk,thv->hkv", k[s0:s1], b_v.float())
        state_vm = state_km.transpose(-1, -2)  # back to [H, V, K]
    return torch.stack(h_out), state_vm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not hasattr(torch, "npu") or torch.npu.device_count() <= 0:
        raise SystemExit("NPU device is not available")

    device = torch.device("npu")
    torch.manual_seed(args.seed)

    B, T, H, K, V = 2, 8, 2, 64, 64
    CS = 64

    k = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device)
    u = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=device)
    gk = torch.randn(1, T, H, K, dtype=torch.float32, device=device) * 0.1
    cu_seqlens = torch.tensor([0, T // 2, T], dtype=torch.int32, device=device)
    chunk_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    indices = torch.arange(B, dtype=torch.int32, device=device)

    # Logical V-major state per slot
    S_true = torch.randn(B, H, V, K, dtype=torch.bfloat16, device=device) * 0.1

    for name, pool in [
        ("A: contiguous V-major [H,V,K] (fix)", S_true.clone()),
        (
            "B: transposed view [H,K,V] (current)",
            S_true.transpose(-1, -2).contiguous().transpose(-1, -2),
        ),
    ]:
        print(f"\n=== pool scenario {name} ===")
        print(
            f"    shape={tuple(pool.shape)} contiguous={pool.is_contiguous()} "
            f"stride={pool.stride()}"
        )

        h_npu, _ = chunk_gated_delta_rule_fwd_h_npu(
            k=k,
            w=w,
            u=u,
            g=None,
            gk=gk,
            initial_state=pool,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )
        torch.npu.synchronize()

        # Reference per sequence
        max_h_diff = 0.0
        max_pool_diff = 0.0
        for b in range(B):
            s0, s1 = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
            kb = k[0, s0:s1]  # [Tb, H, K]
            wb = w[0, s0:s1]
            ub = u[0, s0:s1]
            gkb = gk[0, s0:s1]
            h_ref, final_ref = ref_chunk_h(kb, wb, ub, gkb, S_true[b])
            # kernel h: [1, NT, H, K, V] per global chunk (K-major)
            h_km = h_npu[0, b, :, :, :]  # [NT, H, K, V] (b-th global chunk)
            h_diff = (h_km.float() - h_ref.transpose(-1, -2).float()).abs().max().item()
            max_h_diff = max(max_h_diff, h_diff)
            pool_after = pool[b].float()
            pool_diff = (pool_after - final_ref.float()).abs().max().item()
            rel = pool_diff / (final_ref.float().abs().max().item() + 1e-8)
            max_pool_diff = max(max_pool_diff, pool_diff)
            print(
                f"    seq {b}: h diff={h_diff:.6f} pool write-back diff={pool_diff:.6f} "
                f"(ref max |state|={final_ref.float().abs().max().item():.4f}, "
                f"rel={rel:.2e})"
            )

        ok = max_h_diff < args.atol and max_pool_diff < args.atol
        print(
            f"  [{'PASS' if ok else 'FAIL'}] h max={max_h_diff:.6f} "
            f"pool max={max_pool_diff:.6f} atol={args.atol}"
        )

    print("\ndone")


if __name__ == "__main__":
    main()

"""Reproduce the NPU spec-branch transposed temporal pool scenario in a single-op test.

SGLang memory_pool.py (spec branch, NPU) does::

    temporal_state = temporal_state.transpose(-1, -2)   # [.., H, V, K] -> [.., H, K, V] view

and that *view* is what later reaches ``kda_target_verify_npu`` as
``initial_state_source``, where it is gathered and ``index_copy_``-ed into the
intermediate ``state_pool`` before the AscendC recurrent_kda kernel runs.

The physical memory content then depends on the writer convention:
  * ``kda_chunk_delta_h_npu`` (NPU prefill) keeps its tiles in K-major [K, V]
    and writes the pool through the same transposed view, so physically the
    pool stores S^T and the view presents S ([H, V, K], what the recurrent
    kernel with ``state_v_first=True`` wants).      -> consistent
  * If any writer stores V-major [V, K] directly into the raw buffer, the
    view presents S^T and the verify kernel seeds TRANSPOSED states. -> bug

This test reproduces both conventions against a pure-torch reference:
  1. contiguous pool (= logical S)                      -- baseline
  2. phys = S^T (K-major), view = phys.transpose        -- model-consistent case
  3. phys = S   (V-major), view = phys.transpose        -- suspected bug case

Run on NPU:
    python tests/python/sgl_kernel_npu/test_recurrent_kda_transposed_pool.py
"""

import argparse

import sgl_kernel_npu  # noqa: F401  registers npu ops
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.fla.kda_target_verify import kda_target_verify_npu

EPS = 1e-6


def reference_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound,
) -> torch.Tensor:
    """Pure-torch reference (per-sequence loop), state layout [pool, HV, V, K]."""
    _, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    batch = cu_seqlens.numel() - 1

    q_f = q.float().squeeze(0)
    k_f = k.float().squeeze(0)
    v_f = v.float().squeeze(0)
    g_f = gate_raw.float().squeeze(0)
    b_f = beta_raw.float().squeeze(0)
    out = torch.zeros(T, HV, V, dtype=v.dtype, device=v.device)

    for b in range(batch):
        seq0 = int(cu_seqlens[b].item())
        seq1 = int(cu_seqlens[b + 1].item())
        for hv in range(HV):
            h_head = hv // (HV // H)
            init_slot = int(ssm_state_indices[b, 0].item())
            state = initial_state[init_slot, hv].float().clone()
            for t in range(seq0, seq1):
                q_t = q_f[t, h_head, :]
                k_t = k_f[t, h_head, :]
                v_t = v_f[t, hv, :]
                q_norm = q_t / (torch.sqrt(torch.sum(q_t * q_t)) + EPS)
                k_norm = k_t / (torch.sqrt(torch.sum(k_t * k_t)) + EPS)
                q_scaled = q_norm * scale
                if lower_bound is not None:
                    gate_decay = torch.exp(
                        lower_bound
                        * torch.sigmoid(
                            torch.exp(A_log[hv].float())
                            * (g_f[t, hv, :] + dt_bias[hv].float())
                        )
                    )
                else:
                    gate_decay = torch.exp(
                        -torch.exp(A_log[hv].float())
                        * torch.nn.functional.softplus(
                            g_f[t, hv, :] + dt_bias[hv].float()
                        )
                    )
                beta_val = torch.sigmoid(b_f[t, hv])
                state = state * gate_decay
                delta = v_t - state @ k_norm
                delta = delta * beta_val
                state = state + delta.unsqueeze(-1) * k_norm.unsqueeze(0)
                out[t, hv, :] = state @ q_scaled
    return out.unsqueeze(0)


def seed_check(
    name,
    source,
    initial_state_indices,
    init_indices_flat,
    S_true,
    batch,
    hv,
    v_dim,
    k_dim,
    device,
):
    """Replicate the wrapper's seed copy into a scratch pool, pre-op."""
    pool = torch.zeros(
        (int(init_indices_flat.max().item()) + 1, hv, v_dim, k_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    src = source[initial_state_indices[:batch].to(torch.int64)]
    pool.index_copy_(0, init_indices_flat, src)
    got = pool[init_indices_flat[:batch]]
    diff = (got.float() - S_true.float()).abs()
    print(
        f"  seed check [{name}]: max diff vs logical S_true = {diff.max().item():.6f}"
    )
    return diff.max().item()


def run_case(
    device,
    *,
    batch,
    steps,
    h,
    hv,
    k_dim,
    v_dim,
    lower_bound,
    seed,
    atol,
):
    print(
        f"\n=== B={batch} steps={steps} H={h} HV={hv} K={k_dim} V={v_dim} "
        f"lower_bound={lower_bound} ==="
    )
    torch.manual_seed(seed)
    T = batch * steps
    scale = k_dim**-0.5

    q = torch.randn(1, T, h, k_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, T, h, k_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, T, hv, v_dim, dtype=torch.bfloat16, device=device)
    a_raw = torch.randn(1, T, hv, k_dim, dtype=torch.bfloat16, device=device)
    b_raw = torch.randn(1, T, hv, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(hv, dtype=torch.float32, device=device) * 0.5
    dt_bias = torch.randn(hv, k_dim, dtype=torch.float32, device=device) * 0.1

    # Logical V-major state per slot: [batch, HV, V, K]
    S_true = (
        torch.randn(batch, hv, v_dim, k_dim, dtype=torch.bfloat16, device=device) * 0.1
    )
    initial_state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    intermediate_state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(0, T + 1, steps, dtype=torch.int32, device=device)
    ssm_state_indices = torch.arange(batch, dtype=torch.int32, device=device).unsqueeze(
        1
    ) * steps + torch.arange(steps, dtype=torch.int32, device=device).unsqueeze(0)

    def make_buffer():
        return torch.zeros(
            batch, steps, hv, v_dim, k_dim, dtype=torch.bfloat16, device=device
        )

    # ---- reference: seeded directly with the logical S_true ----
    buf_ref = make_buffer()
    buf_ref.view(-1, hv, v_dim, k_dim).index_copy_(
        0,
        initial_state_indices.to(torch.int64) * steps,
        S_true.clone(),
    )
    out_ref = reference_recurrent_kda(
        q,
        k,
        v,
        a_raw,
        b_raw,
        buf_ref.view(-1, hv, v_dim, k_dim),
        cu_seqlens,
        ssm_state_indices,
        A_log,
        dt_bias,
        scale,
        lower_bound,
    )

    scenarios = {}

    # 1. contiguous pool holding S_true (baseline)
    scenarios["contiguous (logical S)"] = S_true.clone()

    # 2. model-consistent: phys = S^T (K-major, what the NPU prefill writes),
    #    source = phys.transpose(-1, -2) -> presents S
    phys_k = S_true.transpose(-1, -2).contiguous()
    scenarios["view of phys=S^T (consistent)"] = phys_k.transpose(-1, -2)

    # 3. suspected bug: phys = S (V-major writer bypassing the transpose),
    #    source = phys.transpose(-1, -2) -> presents S^T
    phys_v = S_true.clone()
    scenarios["view of phys=S (suspected)"] = phys_v.transpose(-1, -2)

    for name, source in scenarios.items():
        print(
            f"  source layout: shape={tuple(source.shape)} "
            f"contiguous={source.is_contiguous()} stride={source.stride()}"
        )
        init_flat = initial_state_indices.to(torch.int64) * steps
        seed_check(
            name,
            source,
            initial_state_indices,
            init_flat,
            S_true,
            batch,
            hv,
            v_dim,
            k_dim,
            device,
        )
        buf = make_buffer()
        out_asc = kda_target_verify_npu(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a_raw,
            b=b_raw,
            initial_state_source=source,
            initial_state_indices=initial_state_indices,
            intermediate_states_buffer=buf,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=steps,
            safe_gate=lower_bound is not None,
            lower_bound=lower_bound if lower_bound is not None else -5.0,
        )
        torch.npu.synchronize()

        diff = (out_asc.float() - out_ref.float()).abs()
        print(f"  [{name}]")
        print(
            f"    out vs reference: max={diff.max().item():.6f} "
            f"mean={diff.mean().item():.6f}"
        )
        for b in range(batch):
            b_slice = diff[0, b * steps : (b + 1) * steps]
            print(
                f"    batch {b}: max={b_slice.max().item():.6f} "
                f"mean={b_slice.mean().item():.6f}"
            )
        worst = diff.max().item()
        ok = worst < atol
        print(f"  [{'PASS' if ok else 'FAIL'}] worst={worst:.6f} atol={atol}")
    return


def main():
    parser = argparse.ArgumentParser(
        description="Transposed temporal pool seed reproduction for recurrent_kda"
    )
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not hasattr(torch.ops.npu, "recurrent_kda"):
        raise SystemExit(
            "torch.ops.npu.recurrent_kda is not registered. Build sgl-kernel-npu first."
        )
    if not hasattr(torch, "npu") or torch.npu.device_count() <= 0:
        raise SystemExit("NPU device is not available")

    device = torch.device("npu")

    # K3 TP8-local square state (V=K=128): the transpose is shape-invisible.
    run_case(
        device,
        batch=4,
        steps=8,
        h=12,
        hv=12,
        k_dim=128,
        v_dim=128,
        lower_bound=-5.0,
        seed=args.seed,
        atol=args.atol,
    )


if __name__ == "__main__":
    raise SystemExit(main())

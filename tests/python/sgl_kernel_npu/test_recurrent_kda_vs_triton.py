"""Multi-batch accuracy comparison: AscendC recurrent_kda vs original Triton.

Compares the migrated AscendC path (``kda_target_verify_npu`` -> the AscendC
``recurrent_kda`` kernel) against the original Triton implementation
(vendored from sgl-kernel-npu commit 34c6ad7, "verify ops optimization"),
on identical multi-batch inputs.

Three execution paths are exercised:

  1. ``triton_preactivated``: the original model-faithful call -- gate
     pre-activated in fp32 by ``fused_kda_gate_npu`` (safe-gate aware) and
     beta pre-sigmoided, kernel runs with ``GATES_ARE_PREACTIVATED=True``.
  2. ``triton_raw``: the same Triton kernel with RAW bf16 gate/beta and
     in-kernel softplus/sigmoid (``GATES_ARE_PREACTIVATED=False``). This is
     the apples-to-apples comparison for the AscendC kernel's gate contract.
  3. ``ascendc``: the current ``kda_target_verify_npu`` wrapper.

Per-batch max/mean diffs are printed so a batch-indexed bug (batch 0 clean,
batch k diverging) is visible directly.

Run on NPU:
    python tests/python/sgl_kernel_npu/test_recurrent_kda_vs_triton.py
"""

import argparse

import sgl_kernel_npu  # noqa: F401  registers npu ops
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from sgl_kernel_npu.fla.kda_gate import fused_kda_gate_npu
from sgl_kernel_npu.fla.kda_target_verify import kda_target_verify_npu


# ---------------------------------------------------------------------------
# Original Triton kernel, vendored verbatim from commit 34c6ad7.
# ---------------------------------------------------------------------------
@triton.jit
def _kda_target_verify_kernel(
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    initial_state_ptr,
    initial_indices_ptr,
    snapshot_ptr,
    snapshot_indices_ptr,
    out_ptr,
    scale,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_v_dim: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_a_head: tl.constexpr,
    stride_a_dim: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_b_head: tl.constexpr,
    initial_stride_0,
    initial_stride_1,
    initial_stride_2,
    initial_stride_3,
    snapshot_stride_0,
    snapshot_stride_1,
    snapshot_stride_2,
    snapshot_stride_3,
    snapshot_stride_4,
    H_Q: tl.constexpr,
    H_K: tl.constexpr,
    H_V: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STEPS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    GATES_ARE_PREACTIVATED: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_v = tl.program_id(2)

    offset_k = tl.arange(0, BK)
    offset_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offset_k < K
    mask_v = offset_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    q_ratio = H_V // H_Q
    k_ratio = H_V // H_K
    q_head = pid_hv // q_ratio
    k_head = pid_hv // k_ratio
    initial_idx = tl.load(initial_indices_ptr + pid_batch).to(tl.int64)
    snapshot_idx = tl.load(snapshot_indices_ptr + pid_batch).to(tl.int64)

    initial_offsets = (
        initial_idx * initial_stride_0
        + pid_hv * initial_stride_1
        + offset_v[:, None] * initial_stride_2
        + offset_k[None, :] * initial_stride_3
    )
    state = tl.load(
        initial_state_ptr + initial_offsets,
        mask=(initial_idx >= 0) & mask_state,
        other=0.0,
    ).to(tl.float32)

    A_log = tl.zeros((), dtype=tl.float32)
    dt_bias = tl.zeros((BK,), dtype=tl.float32)
    if not GATES_ARE_PREACTIVATED:
        A_log = tl.load(A_log_ptr + k_head).to(tl.float32)
        dt_bias = tl.load(
            dt_bias_ptr + k_head * K + offset_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

    for step in tl.static_range(0, STEPS):
        token = pid_batch * STEPS + step
        q = tl.load(
            q_ptr
            + token * stride_q_token
            + q_head * stride_q_head
            + offset_k * stride_q_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr
            + token * stride_k_token
            + k_head * stride_k_head
            + offset_k * stride_k_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            v_ptr
            + token * stride_v_token
            + pid_hv * stride_v_head
            + offset_v * stride_v_dim,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        a = tl.load(
            a_ptr
            + token * stride_a_token
            + k_head * stride_a_head
            + offset_k * stride_a_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        beta_input = tl.load(
            b_ptr + token * stride_b_token + pid_hv * stride_b_head
        ).to(tl.float32)

        q = q / (tl.sqrt(tl.sum(q * q, axis=0)) + 1e-6)
        k = k / (tl.sqrt(tl.sum(k * k, axis=0)) + 1e-6)
        q *= scale

        if GATES_ARE_PREACTIVATED:
            gate = tl.exp(a)
            beta = beta_input
        else:
            gate_input = a + dt_bias
            softplus = tl.where(
                gate_input <= 20.0,
                tl.log(1.0 + tl.exp(gate_input)),
                gate_input,
            )
            gate = tl.exp(-tl.exp(A_log) * softplus)
            beta = 1.0 / (1.0 + tl.exp(-beta_input))

        state *= gate[None, :]
        value -= tl.sum(state * k[None, :], axis=1)
        value *= beta
        state += value[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], axis=1)

        tl.store(
            out_ptr + (token * H_V + pid_hv) * V + offset_v,
            output,
            mask=mask_v,
        )
        snapshot_offsets = (
            snapshot_idx * snapshot_stride_0
            + step * snapshot_stride_1
            + pid_hv * snapshot_stride_2
            + offset_v[:, None] * snapshot_stride_3
            + offset_k[None, :] * snapshot_stride_4
        )
        tl.store(
            snapshot_ptr + snapshot_offsets,
            state,
            mask=(snapshot_idx >= 0) & mask_state,
        )


def triton_kda_target_verify(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    cache_steps: int,
    scale: float,
    gates_are_preactivated: bool,
) -> torch.Tensor:
    """Faithful replica of the original wrapper's kernel launch (34c6ad7)."""
    batch = q.shape[1] // cache_steps
    h_q, key_dim = q.shape[2:]
    h_k = k.shape[2]
    h_v, value_dim = v.shape[2:]
    a = a.squeeze(0) if a.ndim == 4 else a
    b = b.squeeze(0) if b.ndim == 3 else b
    out = torch.empty((1, q.shape[1], h_v, value_dim), dtype=v.dtype, device=v.device)
    bk = triton.next_power_of_2(key_dim)
    bv = min(64, triton.next_power_of_2(value_dim))
    grid = (batch, h_v, triton.cdiv(value_dim, bv))
    _kda_target_verify_kernel[grid](
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        initial_state_indices,
        intermediate_states_buffer,
        intermediate_state_indices,
        out,
        scale,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        initial_state_source.stride(0),
        initial_state_source.stride(1),
        initial_state_source.stride(2),
        initial_state_source.stride(3),
        intermediate_states_buffer.stride(0),
        intermediate_states_buffer.stride(1),
        intermediate_states_buffer.stride(2),
        intermediate_states_buffer.stride(3),
        intermediate_states_buffer.stride(4),
        H_Q=h_q,
        H_K=h_k,
        H_V=h_v,
        K=key_dim,
        V=value_dim,
        STEPS=cache_steps,
        BK=bk,
        BV=bv,
        GATES_ARE_PREACTIVATED=gates_are_preactivated,
        num_warps=1,
        num_stages=3,
        multibuffer=False,
    )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_inputs(
    batch: int,
    steps: int,
    h_q: int,
    h_k: int,
    h_v: int,
    k_dim: int,
    v_dim: int,
    device: torch.device,
    seed: int = 42,
):
    torch.manual_seed(seed)
    T = batch * steps
    q = torch.randn(1, T, h_q, k_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, T, h_k, k_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, T, h_v, v_dim, dtype=torch.bfloat16, device=device)
    a_raw = torch.randn(1, T, h_k, k_dim, dtype=torch.bfloat16, device=device)
    b_raw = torch.randn(1, T, h_v, dtype=torch.bfloat16, device=device)

    A_log = torch.randn(h_k, dtype=torch.float32, device=device) * 0.5
    dt_bias = torch.randn(h_k, k_dim, dtype=torch.float32, device=device) * 0.1

    initial_state_source = (
        torch.randn(batch, h_v, v_dim, k_dim, dtype=torch.bfloat16, device=device) * 0.1
    )
    initial_state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    intermediate_state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    return (
        q,
        k,
        v,
        a_raw,
        b_raw,
        A_log,
        dt_bias,
        initial_state_source,
        initial_state_indices,
        intermediate_state_indices,
    )


def report(
    name: str,
    diff: torch.Tensor,
    batch: int,
    steps: int,
    ref_abs_mean: float,
    snapshot_diff=None,
):
    diff_f = diff.float().abs()
    print(f"  [{name}]")
    print(f"    max  abs diff: {diff_f.max().item():.6f}")
    print(f"    mean abs diff: {diff_f.mean().item():.6f}")
    print(f"    ref  mean abs: {ref_abs_mean:.6f}")
    for b in range(batch):
        b_slice = diff_f[0, b * steps : (b + 1) * steps]
        print(
            f"    batch {b}: max={b_slice.max().item():.6f} "
            f"mean={b_slice.mean().item():.6f}"
        )
    if snapshot_diff is not None:
        s = snapshot_diff.float().abs()
        print(f"    snapshot max abs diff: {s.max().item():.6f}")
        for b in range(batch):
            print(f"    snapshot batch {b}: max={s[b].max().item():.6f}")
    return diff_f.max().item()


def run_case(
    device,
    *,
    batch,
    steps,
    h_q,
    h_k,
    h_v,
    k_dim,
    v_dim,
    lower_bound,
    seed,
    atol,
):
    print(
        f"\n=== B={batch} steps={steps} H_Q={h_q} H_K={h_k} H_V={h_v} "
        f"K={k_dim} V={v_dim} lower_bound={lower_bound} ==="
    )
    (
        q,
        k,
        v,
        a_raw,
        b_raw,
        A_log,
        dt_bias,
        initial_state_source,
        initial_state_indices,
        intermediate_state_indices,
    ) = make_inputs(batch, steps, h_q, h_k, h_v, k_dim, v_dim, device, seed)
    scale = k_dim**-0.5
    T = batch * steps

    def make_buffer():
        return torch.zeros(
            batch, steps, h_v, v_dim, k_dim, dtype=torch.bfloat16, device=device
        )

    # ---- 1. original Triton, model-faithful preactivated fp32 gates ----
    pre_a = fused_kda_gate_npu(
        a_raw.flatten(-2),
        A_log,
        k_dim,
        gate_bias=dt_bias,
        lower_bound=lower_bound,
    )
    pre_b = b_raw.float().sigmoid()
    buf_tri_pre = make_buffer()
    out_tri_pre = triton_kda_target_verify(
        A_log,
        dt_bias,
        q,
        k,
        v,
        pre_a,
        pre_b,
        initial_state_source,
        initial_state_indices,
        buf_tri_pre,
        intermediate_state_indices,
        cache_steps=steps,
        scale=scale,
        gates_are_preactivated=True,
    )

    # ---- 2. original Triton, raw bf16 gates in-kernel (apples-to-apples) ----
    out_tri_raw = None
    buf_tri_raw = None
    if lower_bound is None:  # in-kernel Triton gate has no safe-gate branch
        buf_tri_raw = make_buffer()
        out_tri_raw = triton_kda_target_verify(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a_raw,
            b_raw,
            initial_state_source,
            initial_state_indices,
            buf_tri_raw,
            intermediate_state_indices,
            cache_steps=steps,
            scale=scale,
            gates_are_preactivated=False,
        )

    # ---- 3. current AscendC wrapper (raw gates, safe_gate aware) ----
    buf_asc = make_buffer()
    out_asc = kda_target_verify_npu(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a_raw,
        b=b_raw,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        intermediate_states_buffer=buf_asc,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=steps,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound if lower_bound is not None else -5.0,
    )
    torch.npu.synchronize()

    ref_abs = out_tri_pre.float().abs().mean().item()
    diffs = {}
    diffs["ascendc vs triton(preactivated)"] = (
        out_asc.float() - out_tri_pre.float(),
        buf_asc - buf_tri_pre,
    )
    if out_tri_raw is not None:
        diffs["ascendc vs triton(raw in-kernel)"] = (
            out_asc.float() - out_tri_raw.float(),
            buf_asc - buf_tri_raw,
        )
        diffs["triton(raw) vs triton(preactivated)"] = (
            out_tri_raw.float() - out_tri_pre.float(),
            buf_tri_raw - buf_tri_pre,
        )

    worst = 0.0
    for name, (out_diff, snap_diff) in diffs.items():
        worst = max(
            worst,
            report(
                name,
                out_diff,
                batch,
                steps,
                ref_abs,
                snapshot_diff=snap_diff,
            ),
        )
    ok = worst < atol
    print(f"  [{'PASS' if ok else 'FAIL'}] worst={worst:.6f} atol={atol}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Multi-batch AscendC recurrent_kda vs original Triton accuracy"
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

    # K3 TP8-local shapes: 96/8 = 12 heads, K=V=128, gamma=7 -> 8 steps,
    # gate_lower_bound=-5.0 (safe gate, as in the K3 checkpoint).
    cases = [
        dict(
            batch=1,
            steps=8,
            h_q=12,
            h_k=12,
            h_v=12,
            k_dim=128,
            v_dim=128,
            lower_bound=-5.0,
        ),
        dict(
            batch=4,
            steps=8,
            h_q=12,
            h_k=12,
            h_v=12,
            k_dim=128,
            v_dim=128,
            lower_bound=-5.0,
        ),
        dict(
            batch=8,
            steps=8,
            h_q=12,
            h_k=12,
            h_v=12,
            k_dim=128,
            v_dim=128,
            lower_bound=-5.0,
        ),
        # non-safe gate, three-way comparison incl. raw in-kernel Triton
        dict(
            batch=4,
            steps=8,
            h_q=12,
            h_k=12,
            h_v=12,
            k_dim=128,
            v_dim=128,
            lower_bound=None,
        ),
        # h_v != h_k expansion path (exercises the fca4601 repeat_interleave)
        dict(
            batch=4,
            steps=8,
            h_q=4,
            h_k=4,
            h_v=16,
            k_dim=128,
            v_dim=128,
            lower_bound=-5.0,
        ),
    ]

    all_ok = True
    for i, cfg in enumerate(cases):
        all_ok &= run_case(
            device,
            seed=args.seed + i,
            atol=args.atol,
            **cfg,
        )

    print("\n=== All comparisons %s ===" % ("PASSED" if all_ok else "FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
from dataclasses import dataclass

import torch
from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_kernel_without_sigmoid


@dataclass
class CaseConfig:
    name: str
    dtype: torch.dtype
    batch: int
    num_heads: int
    beta: float
    threshold: float


def summarize_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs.float() - rhs.float()).abs()
    return diff.max().item(), diff.mean().item()


def reference_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
):
    x = a.to(torch.float32) + dt_bias.to(torch.float32)
    beta_x = beta * x
    softplus_x = torch.where(
        beta_x <= threshold, (1.0 / beta) * torch.log1p(torch.exp(beta_x)), x
    )
    g = -torch.exp(A_log.to(torch.float32)) * softplus_x
    g = g.unsqueeze(0)
    return g, b


def make_case_tensors(case: CaseConfig, device: torch.device):
    A_log = torch.randn((case.num_heads,), device=device, dtype=case.dtype)
    a = torch.randn((case.batch, case.num_heads), device=device, dtype=case.dtype)
    b = torch.randn((case.batch, case.num_heads), device=device, dtype=case.dtype)
    dt_bias = torch.randn((case.num_heads,), device=device, dtype=case.dtype)
    return A_log, a, b, dt_bias


def run_positive_case(case: CaseConfig, device: torch.device, atol: float, rtol: float):
    host_device = torch.device("cpu")
    A_log_cpu, a_cpu, b_cpu, dt_bias_cpu = make_case_tensors(case, host_device)

    A_log = A_log_cpu.to(device)
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    dt_bias = dt_bias_cpu.to(device)

    g_ref, b_ref = reference_fused_gdn_gating(
        A_log_cpu, a_cpu, b_cpu, dt_bias_cpu, case.beta, case.threshold
    )
    g_kernel, b_kernel = fused_gdn_gating_kernel_without_sigmoid(
        A_log, a, b, dt_bias, case.beta, case.threshold
    )
    torch.npu.synchronize()

    g_ref_cpu = g_ref.float().cpu()
    g_kernel_cpu = g_kernel.float().cpu()
    b_ref_cpu = b_ref.float().cpu()
    b_kernel_cpu = b_kernel.float().cpu()

    torch.testing.assert_close(g_kernel_cpu, g_ref_cpu, atol=atol, rtol=rtol)
    torch.testing.assert_close(b_kernel_cpu, b_ref_cpu, atol=0.0, rtol=0.0)

    g_max_diff, g_mean_diff = summarize_diff(g_kernel_cpu, g_ref_cpu)
    b_max_diff, b_mean_diff = summarize_diff(b_kernel_cpu, b_ref_cpu)
    print(
        f"[PASS] {case.name}: "
        f"g_output(max={g_max_diff:.6g}, mean={g_mean_diff:.6g}) "
        f"b_forward(max={b_max_diff:.6g}, mean={b_mean_diff:.6g})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260326)
    args = parser.parse_args()

    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit(f"Import failed: {exc}") from exc

    if not torch.npu.is_available():
        raise SystemExit("NPU device not available")

    torch.manual_seed(args.seed)
    device = torch.device("npu")

    positive_cases = [
        CaseConfig("minimal_fp32", torch.float32, 1, 8, 1.0, 20.0),
        CaseConfig("normal_bf16", torch.bfloat16, 16, 8, 1.0, 20.0),
        CaseConfig("large_batch_fp16", torch.float16, 64, 16, 0.5, 15.0),
        CaseConfig("multi_heads_bf16", torch.bfloat16, 32, 32, 2.0, 25.0),
    ]

    for case in positive_cases:
        run_positive_case(case, device, args.atol, args.rtol)

    print("All fused_gdn_gating_kernel_without_sigmoid tests passed.")


if __name__ == "__main__":
    main()

import argparse
import math
import statistics
import time

import torch
import torch_npu
from sgl_kernel_npu.norm._gemma_rmsnorm_triton import launch_gemma_rms_norm

ROWS = (1, 16, 128, 512)
HIDDEN_SIZES = (256, 2048, 4096, 5120)
DTYPES = (torch.float16, torch.bfloat16)
EPS = 1e-6


def benchmark(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def run_case(rows, hidden_size, dtype, has_residual, warmup, iterations):
    x = torch.randn(rows, hidden_size, device="npu", dtype=dtype)
    weight = torch.randn(hidden_size, device="npu", dtype=dtype)
    residual = torch.randn_like(x) if has_residual else None

    def triton_fn():
        return launch_gemma_rms_norm(x, weight, residual, EPS)

    if has_residual:

        def aclnn_fn():
            output, _, residual_sum = torch_npu.npu_add_rms_norm(
                residual, x, 1.0 + weight, EPS
            )
            return output, residual_sum

    else:

        def aclnn_fn():
            output, _ = torch_npu.npu_rms_norm(x, 1.0 + weight, EPS)
            return output, x

    triton_output, triton_sum = triton_fn()
    aclnn_output, aclnn_sum = aclnn_fn()
    torch.testing.assert_close(triton_output, aclnn_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(triton_sum, aclnn_sum, atol=2e-2, rtol=2e-2)

    triton_ms = benchmark(triton_fn, warmup, iterations)
    aclnn_ms = benchmark(aclnn_fn, warmup, iterations)
    return triton_ms, aclnn_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    ratios = []
    critical_regressions = []
    print("dtype,rows,hidden,residual,triton_ms,aclnn_ms,speedup_pct")
    for dtype in DTYPES:
        for rows in ROWS:
            for hidden_size in HIDDEN_SIZES:
                for has_residual in (False, True):
                    triton_ms, aclnn_ms = run_case(
                        rows,
                        hidden_size,
                        dtype,
                        has_residual,
                        args.warmup,
                        args.iterations,
                    )
                    ratio = aclnn_ms / triton_ms
                    ratios.append(ratio)
                    regression_pct = (triton_ms / aclnn_ms - 1.0) * 100.0
                    if rows == 1 or has_residual:
                        critical_regressions.append(regression_pct)
                    print(
                        f"{dtype},{rows},{hidden_size},{has_residual},"
                        f"{triton_ms:.6f},{aclnn_ms:.6f},{(ratio - 1.0) * 100.0:.2f}"
                    )

    geometric_speedup_pct = (
        math.exp(statistics.mean(map(math.log, ratios))) - 1.0
    ) * 100.0
    worst_critical_regression_pct = max(critical_regressions)
    passed = geometric_speedup_pct >= 5.0 and worst_critical_regression_pct <= 3.0
    print(f"geometric_speedup_pct={geometric_speedup_pct:.2f}")
    print(f"worst_critical_regression_pct={worst_critical_regression_pct:.2f}")
    print(f"gate={'PASS' if passed else 'FAIL'}")
    if args.enforce_gate and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

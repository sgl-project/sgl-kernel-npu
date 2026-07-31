import argparse
import importlib
import math
import statistics
import time

import torch
import torch_npu

EPS = 1e-6
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def synchronized_p50(function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.npu.synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def run_case(add_gemma_rms_norm, rows, hidden_size, dtype, warmup, iterations):
    x = torch.randn(rows, hidden_size, device="npu", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="npu", dtype=dtype)
    x_before = x.clone()
    residual_before = residual.clone()

    def triton_function():
        return add_gemma_rms_norm(x, weight, residual, EPS)

    def aclnn_function():
        output, _, residual_sum = torch_npu.npu_add_rms_norm(
            residual, x, 1.0 + weight, EPS
        )
        return output, residual_sum

    triton_output, triton_sum = triton_function()
    aclnn_output, aclnn_sum = aclnn_function()
    torch.npu.synchronize()

    torch.testing.assert_close(triton_output, aclnn_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(triton_sum, aclnn_sum, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x, x_before, atol=0, rtol=0)
    torch.testing.assert_close(residual, residual_before, atol=0, rtol=0)

    norm_max_abs = (triton_output.float() - aclnn_output.float()).abs().max().item()
    sum_max_abs = (triton_sum.float() - aclnn_sum.float()).abs().max().item()
    triton_ms = synchronized_p50(triton_function, warmup, iterations)
    aclnn_ms = synchronized_p50(aclnn_function, warmup, iterations)
    return triton_ms, aclnn_ms, norm_max_abs, sum_max_abs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the existing Triton add_gemma_rms_norm with "
            "torch_npu.npu_add_rms_norm using 1 + weight on every invocation."
        )
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128, 512])
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[256, 2048, 4096, 5120]
    )
    parser.add_argument(
        "--dtypes",
        choices=DTYPES,
        nargs="+",
        default=["float16", "bfloat16"],
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    module = importlib.import_module("sgl_kernel_npu.norm.add_rmsnorm_bias")
    add_gemma_rms_norm = module.add_gemma_rms_norm
    torch.manual_seed(0)

    print("===== Environment =====")
    print(f"torch={torch.__version__}")
    print(f"torch_npu={torch_npu.__version__}")
    print(f"device={torch.npu.get_device_name()}")
    print(f"module={module.__file__}")
    print(f"warmup={args.warmup}, iterations={args.iterations}")
    print("===== Correctness / Performance =====")
    print(
        "dtype,rows,hidden,triton_ms,aclnn_ms,aclnn_faster_pct,"
        "norm_max_abs,sum_max_abs,status"
    )

    ratios = []
    errors = []
    for dtype_name in args.dtypes:
        dtype = DTYPES[dtype_name]
        for rows in args.rows:
            for hidden_size in args.hidden_sizes:
                try:
                    (
                        triton_ms,
                        aclnn_ms,
                        norm_max_abs,
                        sum_max_abs,
                    ) = run_case(
                        add_gemma_rms_norm,
                        rows,
                        hidden_size,
                        dtype,
                        args.warmup,
                        args.iterations,
                    )
                    ratio = triton_ms / aclnn_ms
                    ratios.append(ratio)
                    print(
                        f"{dtype_name},{rows},{hidden_size},{triton_ms:.6f},"
                        f"{aclnn_ms:.6f},{(ratio - 1.0) * 100.0:.2f},"
                        f"{norm_max_abs:.8f},{sum_max_abs:.8f},PASS"
                    )
                except Exception as error:
                    errors.append((dtype_name, rows, hidden_size, error))
                    print(
                        f"{dtype_name},{rows},{hidden_size},,,,,,,"
                        f"ERROR:{type(error).__name__}:{error}"
                    )

    print("===== Summary =====")
    if ratios:
        geometric_aclnn_faster_pct = (
            math.exp(statistics.mean(map(math.log, ratios))) - 1.0
        ) * 100.0
        print(f"geometric_aclnn_faster_pct={geometric_aclnn_faster_pct:.2f}")
    print(f"passed_cases={len(ratios)}")
    print(f"error_cases={len(errors)}")
    print(f"FINAL_RESULT={'PASS' if ratios and not errors else 'FAIL'}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

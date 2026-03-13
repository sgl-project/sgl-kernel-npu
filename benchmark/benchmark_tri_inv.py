"""
Docstring for benchmark.bench_solve_tril

Profiling script that compares various solve_tril methods. Currently, profiles:
1. `torch_eager`: This is the vanilla PyTorch eager mode forward substitution method.
2. `triton`: Triton-ascend method
3. `ascendc_aiv`: Vector-only column sweep method written in AscendC.
4. `pto`: Vector-only column sweep method written in pto-isa.
"""

import argparse
import logging
import sys
import typing

import torch
import torch.nn.functional as F
from sgl_kernel_npu.fla.solve_tril import solve_tril_npu as solve_tril

file_handler = logging.FileHandler(filename="benchmark_tri_inv.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CACHE_SIZE = 256 * 1024 * 1024

DEVICE_ID = 0
NPU_DEVICE = f"npu:{DEVICE_ID}"


# Map "triangular inverse method name" to Python function with signature fn(A) -> A^{-1}.
TRIANGULAR_INVERSE_METHODS_ = {
    #    "torch_eager": inv_tril_inplace, # from sgl_kernel_npu.fla.chunk import inv_tril_inplace
    "triton": solve_tril,
    "ascendc_aiv": torch.ops.npu.triangular_inverse,
    "pto": torch.ops.npu.pto_triangular_inverse,
}


def run_benchmark(
    fn: typing.Callable,
    warmup_iters: int = 1,
    benchmark_iters: int = 5,
):
    """
    Benchmark a given function with warmup.

    Args:
        device: Device to run benchmark on.
        fn: Function to benchmark.
        warmup_iters: Number of warmup runs.
        benchmark_iters: Number of benchmark runs.

    Returns:
        Average time in microseconds.
    """
    torch.npu.set_device(DEVICE_ID)

    start_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]

    torch.npu.synchronize()
    for i in range(warmup_iters):
        logger.info(f"Warmup iteration: {i}")
        fn()

    torch.npu.synchronize()

    cache_size = _DEFAULT_MAX_CACHE_SIZE
    cache = torch.ones(cache_size, dtype=torch.int8, device=NPU_DEVICE)

    for i in range(benchmark_iters):
        logger.info(f"Benchmarking iteration {i}")
        cache.zero_()
        torch.npu.synchronize()
        start_events[i].record()
        fn()
        end_events[i].record()
        torch.npu.synchronize()
        elapsed_time_ms = start_events[i].elapsed_time(end_events[i])
        logger.info(f"Elapsed time: {elapsed_time_ms} ms")
        yield elapsed_time_ms


@torch.inference_mode()
def profile_solve_tril(
    B: int,
    T: int,
    H: int,
    D: int,
    chunk_size: int = 64,
    dtype: torch.dtype = torch.float16,
    inverse_method: str = "triton",
):
    torch.manual_seed(42)

    inv_fn = TRIANGULAR_INVERSE_METHODS_[inverse_method]

    # do not randomly initialize A otherwise the inverse is not stable
    k = F.normalize(
        torch.randn((B, H, T, chunk_size), dtype=dtype, device=NPU_DEVICE), dim=-1
    )
    # Pad the second-to-last dimension (T) to be a multiple of chunk_size
    padding_size = (chunk_size - T % chunk_size) % chunk_size
    k_padded = F.pad(k, (0, 0, 0, padding_size, 0, 0, 0, 0))
    k_padded = k_padded.reshape(B, H, -1, chunk_size, chunk_size)
    A = (k_padded @ k_padded.transpose(-1, -2)).tril(-1)
    torch.npu.synchronize()

    assert (
        A.ndim >= 4
    ), f"Input tensor must be at least 4-dimensional. Got {A.ndim} dimensions."
    A = A.reshape(-1, A.shape[-3], A.shape[-2], A.shape[-1])

    def run_solve_tril():
        _ = inv_fn(A)

    times_ms = list(
        run_benchmark(
            run_solve_tril,
            args.warmup,
            args.repeats,
        )
    )
    avg_time_ms = sum(times_ms) / len(times_ms)
    avg_time_us = int(avg_time_ms * 1000)
    with open(filename, "a", encoding="UTF-8") as fd:
        line = f"{inverse_method},{B},{T},{H},{D},{chunk_size},{avg_time_us}"
        fd.write(f"{line}\n")


if __name__ == "__main__":  # noqa
    parser = argparse.ArgumentParser(description="Triangular inverse benchmarking")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--inverse-type", type=str, default="ascendc_aiv")
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--D", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    filename = f"bench_results_solve_tril.csv"
    with open(filename, "w", encoding="UTF-8") as fd:
        fd.write("inverse_type,B,T,H,D,chunk_size,elapsed_time_us\n")

    for B in [10, 20, 30]:
        for inverse_method in TRIANGULAR_INVERSE_METHODS_.keys():
            for chunk_size in [16, 32, 64, 128]:
                T, H, D = args.T, args.H, args.D
                logger.info(
                    f"Profiling case: {inverse_method},{B},{T},{H},{D},{chunk_size}"
                )

                # Triton does not support chunk_size = 128
                if inverse_method == "triton" and chunk_size == 128:
                    continue

                profile_solve_tril(
                    B=B,
                    T=T,
                    H=H,
                    D=D,
                    chunk_size=chunk_size,
                    inverse_method=inverse_method,
                )

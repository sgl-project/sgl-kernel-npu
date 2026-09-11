import argparse
import time

import torch

import sgl_kernel_npu  # noqa: F401


HC_MULT = 4
HIDDEN_SIZE = 3584
MIX_SIZE = HC_MULT * (HC_MULT + 2)


def _measure(call, warmup, repeat):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        call()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / repeat


def run_case(num_tokens, warmup, repeat):
    residual = torch.randn(
        num_tokens,
        HC_MULT,
        HIDDEN_SIZE,
        device="npu",
        dtype=torch.bfloat16,
    )
    fn = (
        torch.randn(
            MIX_SIZE,
            HC_MULT * HIDDEN_SIZE,
            device="npu",
            dtype=torch.float32,
        )
        / (HC_MULT * HIDDEN_SIZE) ** 0.5
    ).contiguous()
    scale = torch.tensor([0.8, 1.1, 0.7], device="npu", dtype=torch.float32)
    base = torch.randn(MIX_SIZE, device="npu", dtype=torch.float32) * 0.1
    x = torch.randn(
        num_tokens, HIDDEN_SIZE, device="npu", dtype=torch.bfloat16
    )

    def pre():
        return torch.ops.npu.hc_pre(
            residual,
            fn,
            scale,
            base,
            hc_mult=HC_MULT,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )

    layer_input, post, comb = pre()

    def post_op():
        return torch.ops.npu.hc_post(x, residual, post, comb)

    def pre_post():
        current_input, current_post, current_comb = pre()
        return torch.ops.npu.hc_post(
            current_input, residual, current_post, current_comb
        )

    return (
        _measure(pre, warmup, repeat),
        _measure(post_op, warmup, repeat),
        _measure(pre_post, warmup, repeat),
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark TeleChat4 AscendC mHC")
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 16, 128, 512, 1024]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(20260807)
    print(
        f"{'tokens':>8} {'hc_pre ms':>12} {'hc_post ms':>12} "
        f"{'combined ms':>12}"
    )
    for num_tokens in args.tokens:
        pre_ms, post_ms, combined_ms = run_case(
            num_tokens, args.warmup, args.repeat
        )
        print(
            f"{num_tokens:>8} {pre_ms:>12.4f} {post_ms:>12.4f} "
            f"{combined_ms:>12.4f}"
        )


if __name__ == "__main__":
    main()

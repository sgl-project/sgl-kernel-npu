"""Compare direct CANN MegaMoe with DeepEP FUSED_DEEP_MOE on shared A5 inputs.

Run from this directory:
    python3 test_mega_moe_vs_fused_deep_moe_a5.py --print-per-rank-profile

Defaults: 8 ranks, 1/16/32/64 tokens per rank, 64 experts, top-k=8,
H=7168, I=3072, SwiGLU, MXFP8 E4M3, stacked ND weights, FP32 routing.
Each token count runs in fresh workers with capacity equal to its token count.
Both paths use the same values, 50 warmups, and 30 counted NPU Event samples.
Weight-layout conversion, buffer initialization, and validation precede timing.
"""

import argparse
import math
import os
from functools import partial
from statistics import mean


def make_shared_inputs(args, rank, world_size, num_tokens):
    import torch
    from test_fused_deep_moe_a5 import make_umdk_static_inputs

    # Always generate the largest batch so weights and input prefixes stay
    # identical across the sweep despite running each size in fresh workers.
    torch.manual_seed(args.seed + rank)
    source_args = argparse.Namespace(
        **vars(args),
        quant=args.quant_mode.removeprefix("mx_"),
        weight_format="ND",
        single_active_rank=None,
    )
    max_tokens = max(args.num_tokens)
    inputs = make_umdk_static_inputs(
        rank, world_size, max_tokens, rank * max_tokens, source_args
    )
    inputs["x"] = inputs["x"][:num_tokens].contiguous()
    inputs["expert_scales"] = inputs["expert_scales"][:num_tokens].contiguous()
    # Preserve the existing FusedDeepMoe test's globally balanced routing.
    start = rank * num_tokens * args.num_topk
    inputs["expert_ids"] = (
        torch.arange(
            start,
            start + num_tokens * args.num_topk,
            dtype=torch.int32,
            device="npu",
        )
        .reshape(num_tokens, args.num_topk)
        .remainder(args.num_experts)
    )
    return inputs


def make_mega_weights(inputs):
    import torch

    weights = {}
    for layer in (1, 2):
        # DeepEP uses [E,K,N], MegaMoe [E,N,K]; preserve the quantized bytes.
        weight = inputs[f"gmm{layer}_weight_q"]
        scale = inputs[f"gmm{layer}_weight_scale"]
        weights[f"l{layer}_weights"] = [
            weight.view(torch.uint8).transpose(-1, -2).contiguous().view(weight.dtype)
        ]
        weights[f"l{layer}_weights_sf"] = [
            scale.view(torch.uint8).transpose(-2, -3).contiguous().view(scale.dtype)
        ]
    return weights


def check_outputs(args, rank, world_size, num_tokens, inputs, mega, fused):
    import torch
    import torch.distributed as dist
    from utils import calc_diff

    error, diff = None, 0.0
    try:
        assignments = num_tokens * world_size * args.num_topk
        expected = torch.full(
            (args.num_experts,),
            assignments // args.num_experts,
            dtype=torch.int32,
            device="npu",
        )
        expected[: assignments % args.num_experts] += 1
        local_experts = args.num_experts // world_size
        expected = expected[rank * local_experts : (rank + 1) * local_experts]
        for name, (output, counts) in (("MegaMoe", mega), ("FusedDeepMoe", fused)):
            if output.shape != inputs["x"].shape or output.dtype != torch.bfloat16:
                raise AssertionError(f"{name}: unexpected output shape/dtype")
            if not bool(torch.isfinite(output).all()):
                raise AssertionError(f"{name}: non-finite output")
            torch.testing.assert_close(counts.int(), expected, rtol=0, atol=0)
        # Match the existing A5 FP8 accuracy metric; CPU keeps FP64 arithmetic.
        diff = calc_diff(mega[0].float().cpu(), fused[0].float().cpu())
        if not math.isfinite(diff) or diff >= args.diff_threshold:
            raise AssertionError(f"calc_diff={diff}, limit={args.diff_threshold}")
    except AssertionError as exc:
        error = str(exc)
    failed = torch.tensor([int(error is not None)], dtype=torch.int32, device="npu")
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        if error:
            print(f"[Rank {rank}] validation failed: {error}", flush=True)
        raise AssertionError("MegaMoe/FusedDeepMoe validation failed; timing aborted")
    worst = torch.tensor([diff], dtype=torch.float32, device="npu")
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            f"tokens/rank={num_tokens}: output/count checks passed, "
            f"max_rank_calc_diff={worst.item():.6e}",
            flush=True,
        )


def run_rank(local_rank, num_processes, num_tokens, args, results):
    import deep_ep
    import torch
    import torch.distributed as dist
    from deep_ep.buffer import FuseMode
    from deep_ep.device_info import DEVICE_VERSION_TABLE, get_device_version
    from utils import bench, init_dist, make_hccl_pg_options

    sym_buffer = None
    try:
        rank, world_size, mega_group = init_dist(
            local_rank, num_processes, hccl_buffer_size=200
        )
        if DEVICE_VERSION_TABLE.get(get_device_version()) != "A5":
            raise RuntimeError("This comparison requires Ascend 950 (A5)")
        if args.num_experts % world_size:
            raise ValueError("num_experts must be divisible by the actual world size")
        # CANN looks up an existing HCCL handle when creating its SymmBuffer.
        # Initialize the lazy communicator before CANN queries its group name.
        mega_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe

        fused_group = dist.new_group(
            ranks=list(range(world_size)),
            backend="hccl",
            pg_options=make_hccl_pg_options(args.hccl_buffsize),
        )
        buffer = deep_ep.Buffer(fused_group, low_latency_mode=True)
        with torch.inference_mode():
            inputs = make_shared_inputs(args, rank, world_size, num_tokens)
            mega_weights = make_mega_weights(inputs)
            sym_buffer = get_symm_buffer_for_mega_moe(
                mega_group,
                num_experts=args.num_experts,
                num_max_tokens_per_rank=num_tokens,
                num_topk=args.num_topk,
                hidden=args.hidden,
                intermediate_hidden=args.moe_intermediate_size,
                dispatch_quant_mode=4,
                dispatch_quant_out_dtype=inputs["gmm1_weight_q"].dtype,
            )
            activation_params = (
                {"beta": args.beta, "linear_beta": args.linear_beta}
                if args.activation == "situ"
                else None
            )
            calls = {
                "MegaMoe": partial(
                    mega_moe,
                    x=inputs["x"],
                    topk_ids=inputs["expert_ids"],
                    topk_weights=inputs["expert_scales"],
                    **mega_weights,
                    sym_buffer=sym_buffer,
                    activation="situglu" if args.activation == "situ" else "swiglu",
                    activation_params=activation_params,
                ),
                "FusedDeepMoe": partial(
                    buffer.fused_deep_moe,
                    x=inputs["x"],
                    topk_idx=inputs["expert_ids"],
                    topk_weights=inputs["expert_scales"],
                    gmm1_permuted_weight=inputs["gmm1_weight_q"],
                    gmm1_permuted_weight_scale=inputs["gmm1_weight_scale"],
                    gmm2_weight=inputs["gmm2_weight_q"],
                    gmm2_weight_scale=inputs["gmm2_weight_scale"],
                    num_max_dispatch_tokens_per_rank=num_tokens,
                    num_experts=args.num_experts,
                    quant_mode=0,  # A5 infers MXFP8 from the weight dtype.
                    fuse_mode=FuseMode.FUSED_DEEP_MOE,
                    activation=args.activation,
                    beta=args.beta,
                    linear_beta=args.linear_beta,
                ),
            }
            mega_out, fused_out = calls["MegaMoe"](), calls["FusedDeepMoe"]()
            torch.npu.synchronize()
            check_outputs(
                args, rank, world_size, num_tokens, inputs, mega_out, fused_out
            )
            del mega_out, fused_out
            torch.npu.empty_cache()
            # Alternate which path is timed first across token counts.
            order = ("FusedDeepMoe", "MegaMoe")
            if args.num_tokens.index(num_tokens) % 2:
                order = order[::-1]
            stats = {}
            for name in order:
                stats[name] = bench(
                    calls[name],
                    num_warmups=args.num_warmups,
                    num_tests=args.num_tests + 1,  # bench drops its first sample.
                    sync_fn=dist.barrier,
                )
            local = torch.tensor(
                [stats["FusedDeepMoe"], stats["MegaMoe"]],
                dtype=torch.float32,
                device="npu",
            )
            gathered = [torch.empty_like(local) for _ in range(world_size)]
            dist.all_gather(gathered, local)
            if rank == 0:
                results.put((num_tokens, [value.cpu().tolist() for value in gathered]))
            dist.barrier()
    finally:
        try:
            if sym_buffer is not None:
                torch.npu.synchronize()
                sym_buffer.destroy()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def print_results(results, per_rank=False):
    print("\nNPU Event latency (us); Fused/Mega > 1 means MegaMoe is faster.")
    header = (
        f"{'Tokens':>6}  {'Fused mean':>12}  {'Mega mean':>12}  "
        f"{'Fused/Mega':>10}  {'Fused max':>12}  {'Mega max':>12}  "
        f"{'Ratio(max)':>10}"
    )
    print(header)
    print("-" * len(header), flush=True)
    for tokens, ranks in results:
        fused = [rank[0][0] * 1e6 for rank in ranks]
        mega = [rank[1][0] * 1e6 for rank in ranks]
        print(
            f"{tokens:>6}  {mean(fused):>12.2f}  {mean(mega):>12.2f}  "
            f"{mean(fused) / mean(mega):>10.3f}  {max(fused):>12.2f}  "
            f"{max(mega):>12.2f}  {max(fused) / max(mega):>10.3f}",
            flush=True,
        )
    if per_rank:
        print("\nPer-rank NPU Event latency (us), avg/min/max:")
        header = f"{'Tokens':>6}  {'Rank':>4}  {'Fused':>24}  {'Mega':>24}"
        print(header)
        print("-" * len(header), flush=True)
        for tokens, ranks in results:
            for rank, (fused, mega) in enumerate(ranks):
                f = "/".join(f"{value * 1e6:.2f}" for value in fused)
                m = "/".join(f"{value * 1e6:.2f}" for value in mega)
                print(f"{tokens:>6}  {rank:>4}  {f:>24}  {m:>24}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[1, 16, 32, 64])
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--moe-intermediate-size", type=int, default=3072)
    parser.add_argument(
        "--quant-mode", choices=("mx_fp8_e4m3", "mx_fp8_e5m2"), default="mx_fp8_e4m3"
    )
    parser.add_argument("--activation", choices=("swiglu", "situ"), default="swiglu")
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--linear-beta", type=float, default=25.0)
    parser.add_argument("--num-warmups", type=int, default=50)
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument("--hccl-buffsize", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--diff-threshold", type=float, default=2e-3)
    parser.add_argument("--print-per-rank-profile", action="store_true")
    args = parser.parse_args(argv)
    if not 2 <= args.num_processes <= args.num_experts <= 2048:
        parser.error("require 2 <= num_processes <= num_experts <= 2048")
    if args.num_experts % args.num_processes:
        parser.error("--num-experts must be divisible by --num-processes")
    if not 1 <= args.num_topk <= min(16, args.num_experts):
        parser.error("--num-topk must be in [1, min(16, num_experts)]")
    if any(not 1 <= n <= 256 for n in args.num_tokens):
        parser.error("each --num-tokens value must be in [1, 256]")
    if len(set(args.num_tokens)) != len(args.num_tokens):
        parser.error("--num-tokens values must be unique")
    if not 1024 <= args.hidden <= 8192 or args.hidden % 64:
        parser.error("--hidden must be 64-aligned and in [1024, 8192]")
    if (
        not 512 <= args.moe_intermediate_size <= 3584
        or args.moe_intermediate_size % 512
    ):
        parser.error("--moe-intermediate-size must be 512-aligned and in [512, 3584]")
    if args.num_warmups < 0 or args.num_tests <= 0 or args.hccl_buffsize <= 0:
        parser.error("require num_warmups >= 0, num_tests > 0 and hccl_buffsize > 0")
    if not math.isfinite(args.diff_threshold) or args.diff_threshold <= 0:
        parser.error("--diff-threshold must be finite and positive")
    if args.activation == "situ" and not (args.beta > 0 and args.linear_beta > 0):
        parser.error("SiTU requires positive --beta and --linear-beta")
    return args


def main():
    args = parse_args()
    os.environ["HCCL_BUFFSIZE"] = str(args.hccl_buffsize)
    os.environ["MOE_EXPERT_TOKEN_NUMS_TYPE"] = "1"
    import torch.multiprocessing as mp

    print(
        f"Shared config: {vars(args)}, weight_layout=stacked, weight_format=ND, "
        "x_dtype=bf16, routing_dtype=fp32, expert_ids_dtype=int32, token_jitter=0",
        flush=True,
    )
    queue = mp.get_context("spawn").SimpleQueue()
    results = []
    try:
        for tokens in args.num_tokens:
            mp.spawn(
                run_rank,
                args=(args.num_processes, tokens, args, queue),
                nprocs=args.num_processes,
            )
            results.append(queue.get())
    finally:
        queue.close()
    print_results(results, args.print_per_rank_profile)


if __name__ == "__main__":
    main()

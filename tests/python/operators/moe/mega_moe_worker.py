"""Worker for the pytest-managed four-rank MegaMoE tests.

Run on one host with four visible A3 NPUs and CANN 9.2/provider installed:
    torchrun --standalone --nnodes=1 --nproc-per-node=4 \
        tests/python/operators/moe/mega_moe_worker.py

This is a standalone distributed test, not a pytest test: each worker owns one
NPU and all four workers must execute every collective in the same order.
"""

import argparse
import os
from datetime import timedelta
from time import perf_counter


def make_weights(torch, expert, hidden, intermediate):
    """Recreate each global expert independently, without gathering NPU weights."""
    generator = torch.Generator(device="cpu").manual_seed(7300 + expert)
    w1 = torch.randn(hidden, 2 * intermediate, generator=generator) / hidden**0.5
    w2 = torch.randn(intermediate, hidden, generator=generator) / intermediate**0.5
    return w1.to(torch.bfloat16), w2.to(torch.bfloat16)


def reference_moe(torch, x, ids, routing_weights, hidden, intermediate):
    """Independent dense FP32 FFN per expert; use the same BF16 input weights."""
    result = torch.zeros(x.shape, dtype=torch.float32)
    for expert in ids.unique().tolist():
        tokens, slots = torch.where(ids == expert)
        w1, w2 = make_weights(torch, expert, hidden, intermediate)
        gate, up = (x[tokens].float() @ w1.float()).chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        down = activated @ w2.float()
        result.index_add_(0, tokens, down * routing_weights[tokens, slots, None])
    return result


def generate_data(torch, args, rank, tokens):
    generator = torch.Generator(device="cpu").manual_seed(4200 + rank)
    x = torch.randn(tokens, args.hidden, generator=generator).to(torch.bfloat16)
    if args.input_mode == "zero":
        x.zero_()
    rows = torch.arange(tokens)[:, None]
    if args.routing == "skewed":
        # All ranks target the same experts; some experts receive no tokens.
        ids = torch.arange(args.topk)[None, :].expand(tokens, -1).clone()
    else:
        ids = (rows + torch.arange(args.topk)[None, :] + rank) % args.num_experts
    routing_weights = torch.softmax(
        torch.randn(tokens, args.topk, generator=generator), dim=-1
    )
    return x, ids, routing_weights


def measure_latency(torch, dist, group, execute, device, repeats):
    for _ in range(3):
        execute()
    torch.npu.synchronize()
    dist.barrier(group=group)
    start = perf_counter()
    for _ in range(repeats):
        execute()
    torch.npu.synchronize()
    elapsed = torch.tensor([(perf_counter() - start) * 1e6 / repeats], device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
    if dist.get_rank(group) == 0:
        print(f"PERF max_rank_mean_latency_us={elapsed.item():.3f}", flush=True)


def run_case(torch, dist, torch_npu, group, args, tokens, capacity, impl):
    from sgl_kernel_npu.operators.moe import get_symm_buffer_for_mega_moe, mega_moe

    rank = dist.get_rank(group)
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    num_experts, topk = args.num_experts, args.topk
    local_experts = num_experts // dist.get_world_size(group)
    x, ids, routing_weights = generate_data(torch, args, rank, tokens)
    expected = reference_moe(
        torch, x, ids, routing_weights, args.hidden, args.intermediate
    )

    gathered_ids = [torch.empty_like(ids, device=device) for _ in range(4)]
    dist.all_gather(gathered_ids, ids.to(device), group=group)
    global_counts = torch.bincount(
        torch.cat([value.cpu().reshape(-1) for value in gathered_ids]),
        minlength=num_experts,
    )
    expected_counts = global_counts[rank * local_experts : (rank + 1) * local_experts]

    # A3 uses per-expert [H,2I] / [I,H] NZ tensors, unlike A5 MXFP layouts.
    l1, l2 = [], []
    for expert in range(rank * local_experts, (rank + 1) * local_experts):
        w1, w2 = make_weights(torch, expert, args.hidden, args.intermediate)
        l1.append(torch_npu.npu_format_cast(w1.to(device), 29))
        l2.append(torch_npu.npu_format_cast(w2.to(device), 29))
    padded_x = torch.zeros(capacity, args.hidden, dtype=torch.bfloat16, device=device)
    padded_ids = torch.zeros(capacity, topk, dtype=torch.int32, device=device)
    padded_weights = torch.zeros(capacity, topk, dtype=torch.float32, device=device)
    active = torch.zeros(capacity, dtype=torch.int8, device=device)
    padded_x[:tokens] = x.to(device)
    padded_ids[:tokens] = ids.to(device=device, dtype=torch.int32)
    padded_weights[:tokens] = routing_weights.to(device)
    active[:tokens] = 1

    buffer = get_symm_buffer_for_mega_moe(
        group,
        num_experts=num_experts,
        num_max_tokens_per_rank=capacity,
        num_topk=topk,
        hidden=args.hidden,
        intermediate_hidden=args.intermediate,
        max_recv_token_num=0,
        dispatch_quant_mode=0,
        dispatch_quant_out_dtype=None,
        impl=impl,
    )

    def execute():
        return mega_moe(
            x=padded_x,
            topk_ids=padded_ids,
            topk_weights=padded_weights,
            l1_weights=l1,
            l2_weights=l2,
            sym_buffer=buffer,
            l1_weights_sf=None,
            l2_weights_sf=None,
            l1_bias=None,
            l2_bias=None,
            x_active_mask=active,
            activation="swiglu",
            impl=impl,
        )

    try:
        output, counts = execute()
        torch.npu.synchronize()
        actual = output[:tokens].float().cpu()
        actual_counts = counts.reshape(-1).to(device="cpu", dtype=torch.int64)
        assert output.shape == padded_x.shape, (output.shape, padded_x.shape)
        assert output.dtype == padded_x.dtype
        assert torch.isfinite(actual).all(), "MegaMoE output contains NaN/Inf"
        torch.testing.assert_close(actual_counts, expected_counts, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.02)
        relative_rms = (actual - expected).square().mean().sqrt() / (
            expected.square().mean().sqrt().clamp_min(1e-12)
        )
        assert relative_rms.item() < 0.015, relative_rms.item()
        print(
            f"rank={rank} impl={impl} tokens={tokens}/{capacity} "
            f"relative_rms={relative_rms.item():.6f} PASS",
            flush=True,
        )
        # All ranks must pass accuracy before any rank starts the timed phase.
        dist.barrier(group=group)
        if args.perf_iters:
            measure_latency(torch, dist, group, execute, device, args.perf_iters)
    finally:
        torch.npu.synchronize()
        buffer.destroy()


def main():
    from mega_moe_quant_worker import QUANT_MODES, run_quantized_case

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--impl", choices=("auto", "cann"), default="auto")
    parser.add_argument(
        "--quant-mode", choices=QUANT_MODES["arch22"] + QUANT_MODES["arch35"]
    )
    parser.add_argument("--routing", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--input-mode", choices=("normal", "zero"), default="normal")
    parser.add_argument("--perf-iters", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.tokens <= args.capacity:
        parser.error("require 0 < tokens <= capacity")
    if args.quant_mode in QUANT_MODES["arch35"] and args.tokens + 3 > args.capacity:
        parser.error("MXFP cases require capacity for rank-varying token counts")
    if args.num_experts <= 0 or args.num_experts % 4:
        parser.error("num-experts must be positive and divisible by four")
    if not 0 < args.topk <= args.num_experts or args.perf_iters < 0:
        parser.error("invalid topk or perf-iters")
    os.environ["MOE_EXPERT_TOKEN_NUMS_TYPE"] = "1"
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        parser.error(
            "launch with torchrun --nproc-per-node=4; exactly four ranks required"
        )
    if any(value <= 0 or value % 512 for value in (args.hidden, args.intermediate)):
        parser.error("hidden and intermediate must be positive multiples of 512")

    import torch
    import torch.distributed as dist
    import torch_npu

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    torch_npu.npu.config.allow_internal_format = True
    torch.set_num_threads(1)
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    try:
        from sgl_kernel_npu.operators.utils.generation import get_hardware

        generation = "arch35" if args.quant_mode in QUANT_MODES["arch35"] else "arch22"
        if get_hardware().generation.value != generation:
            raise RuntimeError(f"This MegaMoE case requires {generation}")
        group = dist.group.WORLD
        if args.quant_mode:
            run_quantized_case(
                torch,
                dist,
                torch_npu,
                group,
                args,
                torch.device("npu", local_rank),
                generate_data,
                measure_latency,
            )
        else:
            run_case(
                torch,
                dist,
                torch_npu,
                group,
                args,
                args.tokens,
                args.capacity,
                args.impl,
            )
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Compare dispatch/combine accuracy between Default (deep_ep_cpp) and AlltoAll
(torch.distributed) strategies via the buffer interface.

Workflow:
  1. Generate N rounds of random inputs (identical for both strategies).
  2. Run Default strategy over all N rounds, collect results.
  3. Run AlltoAll strategy over all N rounds, collect results.
  4. Compare per-round results and display a summary table.
"""

import argparse
import os
import random

import deep_ep
import torch
import torch.distributed as dist
import torch_npu
from deep_ep.strategies import AlltoAllNormalCommStrategy, DefaultNormalCommStrategy
from utils import calc_diff, get_diff_threshold, init_dist, per_token_cast_back


def run_with_buffer(buffer, x, topk_idx, topk_weights, num_experts, config, quant_type):
    """Run dispatch + combine through the buffer interface.

    The strategy is determined by `buffer.normal_strategy` (swapped by the caller).
    Returns: (recv_x, recv_num_tokens_per_expert_list, combined_x)
    """
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    (
        recv_x,
        _,
        _,
        recv_num_tokens_per_expert_list,
        handle,
        _,
    ) = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        quant_mode=None if quant_type == "bf16" else quant_type,
    )
    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

    combine_topk_weights = (
        handle["topk_weights"] if isinstance(handle, dict) else handle[7]
    )
    combined_x, _, _ = buffer.combine(
        x=recv_x,
        handle=handle,
        config=config,
        async_finish=False,
        topk_weights=combine_topk_weights,
    )

    return recv_x, recv_num_tokens_per_expert_list, combined_x


def generate_round_inputs(args, rank, round_idx):
    """Generate deterministic random inputs for one round.

    Seeding is deterministic per (round, rank) so both strategies see the
    identical input for the same round.
    """
    seed = args.seed + round_idx
    torch.manual_seed(seed + rank)
    torch.npu.manual_seed(seed + rank)
    random.seed(seed + rank)

    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts
    base_num_tokens = args.num_tokens

    if args.enable_dynamic_tokens:
        if base_num_tokens < 10:
            num_tokens = max(1, base_num_tokens + random.randint(-2, 2))
        else:
            num_tokens = max(1, int(base_num_tokens * random.uniform(0.9, 1.1)))
    else:
        num_tokens = base_num_tokens

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device="npu")
    return x, topk_idx, topk_weights, num_tokens


def run_strategy_all_rounds(
    buffer, strategy, inputs, num_experts, config, strategy_name, rank, quant_type
):
    """Run one strategy over all rounds. Returns list of result tuples."""
    buffer.normal_strategy = strategy
    results = []
    for round_idx, (x, topk_idx, topk_weights, num_tokens) in enumerate(inputs):
        if rank == 0:
            print(
                f"  [{strategy_name}] round {round_idx + 1}/{len(inputs)}...",
                flush=True,
            )

        recv_x, recv_list, combined_x = run_with_buffer(
            buffer, x, topk_idx, topk_weights, num_experts, config, quant_type
        )
        results.append((recv_x, recv_list, combined_x, num_tokens))
    return results


def test_compare(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts

    assert (
        num_experts % num_ranks == 0
    ), f"num_experts ({num_experts}) must be divisible by num_ranks ({num_ranks})"

    if args.quant_type == "int8":
        os.environ["DEEP_NORMAL_MODE_USE_INT8_QUANT"] = "1"

    if local_rank == 0:
        print(
            f"[config] hidden={hidden}, num_topk={num_topk}, "
            f"num_experts={num_experts}, num_ranks={num_ranks}, seed={args.seed}, "
            f"dynamic_tokens={args.enable_dynamic_tokens}, "
            f"num_rounds={args.num_rounds}, quant_type={args.quant_type}",
            flush=True,
        )

    # ==========================================
    # Phase 1: Generate all rounds' inputs (collective, deterministic)
    # ==========================================
    if local_rank == 0:
        print("\n>>> Phase 1: Generating inputs for all rounds...", flush=True)
    inputs = []
    for round_idx in range(args.num_rounds):
        x, topk_idx, topk_weights, num_tokens = generate_round_inputs(
            args, rank, round_idx
        )
        inputs.append((x, topk_idx, topk_weights, num_tokens))
    dist.barrier()
    if local_rank == 0:
        token_summary = ", ".join(f"r{i + 1}={inp[3]}" for i, inp in enumerate(inputs))
        print(f"  num_tokens per round: {token_summary}", flush=True)

    # ==========================================
    # Create buffer + strategies
    # ==========================================
    print(f"[Rank {rank}] Initializing buffer...", flush=True)
    buffer = deep_ep.Buffer(
        group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1
    )
    print(f"[Rank {rank}] Buffer created OK.", flush=True)

    default_strategy = DefaultNormalCommStrategy(runtime=buffer.runtime, group=group)
    alltoall_strategy = AlltoAllNormalCommStrategy(runtime=buffer.runtime, group=group)
    config = deep_ep.Config(24, 8, 256)

    # ==========================================
    # Phase 2: Run Default strategy over all rounds
    # ==========================================
    if local_rank == 0:
        print("\n>>> Phase 2: Running Default strategy for all rounds...", flush=True)
    results_d = run_strategy_all_rounds(
        buffer,
        default_strategy,
        inputs,
        num_experts,
        config,
        "Default",
        rank,
        args.quant_type,
    )

    # ==========================================
    # Phase 3: Run AlltoAll strategy over all rounds
    # ==========================================
    if local_rank == 0:
        print("\n>>> Phase 3: Running AlltoAll strategy for all rounds...", flush=True)
    results_a = run_strategy_all_rounds(
        buffer,
        alltoall_strategy,
        inputs,
        num_experts,
        config,
        "AlltoAll",
        rank,
        args.quant_type,
    )

    # ==========================================
    # Phase 4: Compare and display table
    # ==========================================
    dist.barrier()
    if local_rank == 0:
        print("\n" + "=" * 90, flush=True)
        print("NORMAL COMPARISON RESULTS", flush=True)
        print("=" * 90, flush=True)

    all_passed = True
    quant_threshold = get_diff_threshold(args.quant_type)
    # Collect per-round stats for table
    table_rows = []
    for round_idx in range(args.num_rounds):
        recv_x_d, recv_list_d, combined_x_d, num_tokens = results_d[round_idx]
        recv_x_a, recv_list_a, combined_x_a, _ = results_a[round_idx]

        # Dispatch comparison
        list_match = recv_list_d == recv_list_a
        shape_match = recv_x_d.shape == recv_x_a.shape
        dispatch_pass = list_match and shape_match
        dispatch_max_diff = 0.0
        if dispatch_pass:
            recv_x_d_f = recv_x_d.float()
            recv_x_a_f = recv_x_a.float()
            dispatch_max_diff = torch.max(torch.abs(recv_x_d_f - recv_x_a_f)).item()
            if args.quant_type == "bf16":
                dispatch_pass = dispatch_max_diff == 0.0
            else:
                dispatch_pass = calc_diff(recv_x_d_f, recv_x_a_f) < quant_threshold

        # Combine comparison
        combined_x_d_f = combined_x_d.float()
        combined_x_a_f = combined_x_a.float()
        combine_max_diff = torch.max(torch.abs(combined_x_d_f - combined_x_a_f)).item()
        if args.quant_type == "bf16":
            combine_pass = combine_max_diff == 0.0
        else:
            combine_pass = calc_diff(combined_x_d_f, combined_x_a_f) < quant_threshold

        round_pass = dispatch_pass and combine_pass
        if not round_pass:
            all_passed = False

        table_rows.append(
            {
                "round": round_idx + 1,
                "tokens": num_tokens,
                "dispatch_pass": dispatch_pass,
                "dispatch_max_diff": dispatch_max_diff,
                "combine_pass": combine_pass,
                "combine_max_diff": combine_max_diff,
                "pass": round_pass,
            }
        )

        if args.debug and rank == 0:
            print(
                f"  [Round {round_idx + 1}] tokens={num_tokens}, "
                f"list_match={list_match}, shape_match={shape_match}, "
                f"dispatch_max_diff={dispatch_max_diff:.8f}, "
                f"combine_max_diff={combine_max_diff:.8f}",
                flush=True,
            )

    # Print table (rank 0 only)
    if local_rank == 0:
        print_table(table_rows)
        print("=" * 90, flush=True)
        if all_passed:
            print(f"Status: ALL {args.num_rounds} ROUND(S) PASSED", flush=True)
        else:
            failed = sum(1 for r in table_rows if not r["pass"])
            print(
                f"Status: {failed}/{args.num_rounds} ROUND(S) FAILED",
                flush=True,
            )
        print("=" * 90 + "\n", flush=True)

    assert all_passed, f"[rank {rank}] Some rounds failed"
    dist.barrier()
    dist.destroy_process_group()


def print_table(rows):
    """Print comparison results as a formatted table."""
    if not rows:
        return
    headers = [
        "Round",
        "Tokens",
        "Dispatch",
        "Disp MaxDiff",
        "Combine",
        "Comb MaxDiff",
        "Status",
    ]
    # Compute column widths
    widths = []
    for i, h in enumerate(headers):
        w = len(h)
        for r in rows:
            val = format_cell(r, i)
            if len(val) > w:
                w = len(val)
        widths.append(w)

    # Build separator and header
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_line = "|"
    for h, w in zip(headers, widths):
        header_line += f" {h:<{w}} |"
    print(sep, flush=True)
    print(header_line, flush=True)
    print(sep, flush=True)
    for r in rows:
        line = "|"
        for i in range(len(headers)):
            val = format_cell(r, i)
            line += f" {val:<{widths[i]}} |"
        print(line, flush=True)
    print(sep, flush=True)


def format_cell(row, col_idx):
    """Format a cell value for the table."""
    if col_idx == 0:  # Round
        return str(row["round"])
    if col_idx == 1:  # Tokens
        return str(row["tokens"])
    if col_idx == 2:  # Dispatch
        return "PASS" if row["dispatch_pass"] else "FAIL"
    if col_idx == 3:  # Disp MaxDiff
        return f"{row['dispatch_max_diff']:.6f}"
    if col_idx == 4:  # Combine
        return "PASS" if row["combine_pass"] else "FAIL"
    if col_idx == 5:  # Comb MaxDiff
        return f"{row['combine_max_diff']:.6f}"
    if col_idx == 6:  # Status
        return "PASS" if row["pass"] else "FAIL"
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare dispatch/combine accuracy between Default (deep_ep_cpp) "
        "and AlltoAll (torch.distributed) strategies via the buffer interface"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--enable-dynamic-tokens",
        action="store_true",
        help="Enable dynamic and inconsistent num_tokens across different ranks",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Number of rounds to run with fresh random input each round (default: 1)",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="bf16",
        choices=["bf16", "int8"],
        help="Quantization type for dispatch/combine: bf16 (no quantization) or int8 (default: bf16)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-rank dispatch comparison details",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_compare, args=(num_processes, args), nprocs=num_processes
    )

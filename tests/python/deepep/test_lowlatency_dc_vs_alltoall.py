"""Compare low_latency dispatch & combine accuracy between Default (deep_ep_cpp)
and AlltoAll (torch.distributed all_to_all) strategies via the buffer interface.

Workflow:
  1. Generate N rounds of random inputs (identical for both strategies).
  2. Run Default strategy over all N rounds, collect results.
  3. Run AlltoAll strategy over all N rounds, collect results.
  4. Compare per-round results (dispatch per-expert + combine bitwise) and
     display a summary table.
"""

import argparse
import random

import torch
import torch.distributed as dist
import torch_npu
from deep_ep import Buffer
from deep_ep.ep_strategy import get_low_latency_strategy
from utils import calc_diff, get_diff_threshold, init_dist, per_token_cast_back


def run_with_buffer(
    buffer, x, topk_idx, topk_weights, aligned_num_tokens, num_experts, quant_type
):
    """Run low_latency dispatch + combine through the buffer interface.

    The strategy is determined by `buffer.low_latency_strategy` (swapped by the caller).
    Returns: (recv_x, recv_count, handle, combined_x)
    """
    cumulative_local_expert_recv_stats = torch.zeros(
        num_experts // buffer.group_size, dtype=torch.int, device="npu"
    )

    recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x=x,
        topk_idx=topk_idx,
        num_max_dispatch_tokens_per_rank=aligned_num_tokens,
        num_experts=num_experts,
        use_fp8=(quant_type != "bf16"),
        round_scale=False,
        use_ue8m0=False,
        use_mxfp4=False,
        async_finish=False,
        return_recv_hook=False,
        topk_weights=topk_weights,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        quant_mode=None if quant_type == "bf16" else quant_type,
    )
    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

    combined_x, event, hook = buffer.low_latency_combine(
        x=recv_x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        handle=handle,
        async_finish=False,
        zero_copy=False,
        return_recv_hook=False,
    )

    return recv_x, recv_count, handle, combined_x


def extract_expert_tokens_default(recv_x, recv_count, num_local_experts):
    """Extract per-expert valid tokens from Default strategy's dispatch output.

    Layout: tokens are compactly packed by expert with no inter-expert padding.
    Expert i's tokens occupy `recv_count[i]` rows starting at the cumulative
    offset of all preceding experts.
    """
    expert_tokens = []
    start = 0
    for i in range(num_local_experts):
        count = recv_count[i].item()
        tokens = recv_x[start : start + count]
        expert_tokens.append(tokens)
        start += count
    return expert_tokens


def extract_expert_tokens_alltoall(
    recv_x, num_local_experts, num_ranks, aligned_num_tokens, all_topk_idx, rank
):
    """Extract per-expert valid tokens from AlltoAll strategy's dispatch output.

    Layout: recv_x has `num_local_experts * num_ranks * aligned_num_tokens` rows.
    Expert i's tokens occupy a slice of `num_ranks * aligned_num_tokens` rows,
    split into `num_ranks` sub-blocks of `aligned_num_tokens` rows each. We use
    `all_topk_idx` to compute the real token count per expert per rank.
    """
    expert_capacity = aligned_num_tokens
    expert_tokens = []
    for i in range(num_local_experts):
        expert_id = rank * num_local_experts + i
        chunk_start = i * num_ranks * expert_capacity
        valid_tokens = []
        for r in range(num_ranks):
            count_from_r = (all_topk_idx[r] == expert_id).sum().item()
            sub_block_start = chunk_start + r * expert_capacity
            valid_tokens.append(
                recv_x[sub_block_start : sub_block_start + count_from_r]
            )
        expert_tokens.append(
            torch.cat(valid_tokens, dim=0) if valid_tokens else recv_x[:0]
        )
    return expert_tokens


def generate_round_inputs(args, rank, num_ranks, group, round_idx, buffer_capacity):
    """Generate deterministic random inputs for one round.

    Returns (x, topk_idx, topk_weights, all_topk_idx, num_tokens, aligned_num_tokens).
    `all_topk_idx` is gathered across ranks for computing expected expert counts.
    """
    seed = args.seed + round_idx
    torch.manual_seed(seed + rank)
    torch.npu.manual_seed(seed + rank)
    random.seed(seed + rank)

    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts

    if args.enable_dynamic_tokens:
        if buffer_capacity < 10:
            lower = max(1, buffer_capacity - 2)
            num_tokens = random.randint(lower, buffer_capacity)
        else:
            lower = max(1, int(buffer_capacity * 0.9))
            num_tokens = random.randint(lower, buffer_capacity)
    else:
        num_tokens = buffer_capacity

    # Align num_tokens across ranks for this round
    local_tokens_tensor = torch.tensor([num_tokens], dtype=torch.int32, device="npu")
    dist.all_reduce(local_tokens_tensor, op=dist.ReduceOp.MAX)
    aligned_num_tokens = local_tokens_tensor.item()

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    ).abs()

    # Gather all_topk_idx (with -1 for padding) for computing expected counts
    padding_size = aligned_num_tokens - num_tokens
    if padding_size > 0:
        padding_tensor = torch.full(
            (padding_size, num_topk),
            fill_value=-1,
            dtype=topk_idx.dtype,
            device="npu",
        )
        topk_idx_padded = torch.cat([topk_idx, padding_tensor], dim=0)
    else:
        topk_idx_padded = topk_idx
    all_topk_idx = torch.empty(
        (num_ranks, aligned_num_tokens, num_topk),
        dtype=topk_idx.dtype,
        device="npu",
    )
    dist.all_gather_into_tensor(all_topk_idx, topk_idx_padded, group=group)

    return x, topk_idx, topk_weights, all_topk_idx, num_tokens, aligned_num_tokens


def run_strategy_all_rounds(
    buffer, strategy, inputs, num_experts, strategy_name, rank, quant_type
):
    """Run one strategy over all rounds. Returns list of result tuples."""
    buffer.low_latency_strategy = strategy
    results = []
    for round_idx, (
        x,
        topk_idx,
        topk_weights,
        all_topk_idx,
        num_tokens,
        aligned_num_tokens,
    ) in enumerate(inputs):
        if rank == 0:
            print(
                f"  [{strategy_name}] round {round_idx + 1}/{len(inputs)}...",
                flush=True,
            )

        recv_x, recv_count, handle, combined_x = run_with_buffer(
            buffer,
            x,
            topk_idx,
            topk_weights,
            aligned_num_tokens,
            num_experts,
            quant_type,
        )
        results.append(
            (
                recv_x,
                recv_count,
                combined_x,
                all_topk_idx,
                num_tokens,
                aligned_num_tokens,
            )
        )
    return results


def compare_round(
    results_d, results_a, round_idx, rank, num_ranks, num_local_experts, args
):
    """Compare one round's results between Default and AlltoAll.

    Returns a dict of per-round stats for the summary table.
    """
    (
        recv_x_d,
        recv_count_d,
        combined_x_d,
        all_topk_idx,
        num_tokens,
        aligned_num_tokens,
    ) = results_d
    recv_x_a, recv_count_a, combined_x_a, _, _, _ = results_a

    # --- Dispatch comparison ---
    tokens_d = extract_expert_tokens_default(recv_x_d, recv_count_d, num_local_experts)
    tokens_a = extract_expert_tokens_alltoall(
        recv_x_a, num_local_experts, num_ranks, aligned_num_tokens, all_topk_idx, rank
    )

    dispatch_pass = True
    dispatch_max_diff = 0.0
    quant_threshold = get_diff_threshold(args.quant_type)
    for i in range(num_local_experts):
        expert_id = rank * num_local_experts + i
        expected_count = (all_topk_idx == expert_id).sum().item()

        td = tokens_d[i]
        ta = tokens_a[i]
        count_d = td.shape[0]
        count_a = ta.shape[0]
        count_ok = (count_d == expected_count) and (count_a == expected_count)
        if not count_ok:
            dispatch_pass = False
            if args.debug:
                print(
                    f"  [Round {round_idx + 1} Dispatch] rank={rank} expert={expert_id} "
                    f"count mismatch: expected={expected_count}, default={count_d}, "
                    f"alltoall={count_a}",
                    flush=True,
                )
            continue

        if count_d > 0:
            td_f = td.float()
            ta_f = ta.float()
            expert_max_diff = torch.max(torch.abs(td_f - ta_f)).item()
            if args.quant_type == "bf16":
                expert_ok = expert_max_diff == 0
            else:
                expert_ok = calc_diff(td_f, ta_f) < quant_threshold
            if not expert_ok:
                dispatch_pass = False
                if args.debug:
                    sorted_d, _ = td_f.sort(dim=0)
                    sorted_a, _ = ta_f.sort(dim=0)
                    sorted_max_diff = torch.max(torch.abs(sorted_d - sorted_a)).item()
                    order_note = (
                        " (order differs, set matches)"
                        if sorted_max_diff == 0
                        else " (real mismatch)"
                    )
                    print(
                        f"  [Round {round_idx + 1} Dispatch] rank={rank} "
                        f"expert={expert_id} max_diff={expert_max_diff:.8f}{order_note}",
                        flush=True,
                    )
                if expert_max_diff > dispatch_max_diff:
                    dispatch_max_diff = expert_max_diff

    # --- Combine comparison ---
    combined_x_d_f = combined_x_d.float()
    combined_x_a_f = combined_x_a.float()
    combine_max_diff = torch.max(torch.abs(combined_x_d_f - combined_x_a_f)).item()
    if args.quant_type == "bf16":
        combine_pass = combine_max_diff == 0.0
    else:
        combine_pass = calc_diff(combined_x_d_f, combined_x_a_f) < quant_threshold

    round_pass = dispatch_pass and combine_pass
    return {
        "round": round_idx + 1,
        "tokens": num_tokens,
        "aligned": aligned_num_tokens,
        "dispatch_pass": dispatch_pass,
        "dispatch_max_diff": dispatch_max_diff,
        "combine_pass": combine_pass,
        "combine_max_diff": combine_max_diff,
        "pass": round_pass,
    }


def test_compare(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts

    assert (
        num_experts % num_ranks == 0
    ), f"num_experts ({num_experts}) must be divisible by num_ranks ({num_ranks})"
    num_local_experts = num_experts // num_ranks

    buffer_capacity = args.num_tokens

    if local_rank == 0:
        print(
            f"[config] hidden={hidden}, num_topk={num_topk}, "
            f"num_experts={num_experts}, num_ranks={num_ranks}, seed={args.seed}, "
            f"dynamic_tokens={args.enable_dynamic_tokens}, "
            f"num_rounds={args.num_rounds}, buffer_capacity={buffer_capacity}, "
            f"quant_type={args.quant_type}",
            flush=True,
        )

    # ==========================================
    # Phase 1: Generate all rounds' inputs (collective, deterministic)
    # ==========================================
    if local_rank == 0:
        print("\n>>> Phase 1: Generating inputs for all rounds...", flush=True)
    inputs = []
    for round_idx in range(args.num_rounds):
        inp = generate_round_inputs(
            args, rank, num_ranks, group, round_idx, buffer_capacity
        )
        inputs.append(inp)
    dist.barrier()
    if local_rank == 0:
        token_summary = ", ".join(f"r{i + 1}={inp[4]}" for i, inp in enumerate(inputs))
        print(f"  num_tokens per round: {token_summary}", flush=True)

    # ==========================================
    # Create buffer + strategies
    # ==========================================
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        buffer_capacity, hidden, num_ranks, num_experts
    )
    print(f"[Rank {rank}] Initializing low_latency buffer...", flush=True)
    buffer = Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // num_ranks,
        low_latency_strategy="default",
    )
    print(f"[Rank {rank}] Buffer created OK.", flush=True)

    DefaultStrategy = get_low_latency_strategy("default")
    AlltoAllStrategy = get_low_latency_strategy("alltoall")
    default_strategy = DefaultStrategy(runtime=buffer.runtime, group=group)
    alltoall_strategy = AlltoAllStrategy(runtime=buffer.runtime, group=group)

    # ==========================================
    # Phase 2: Run Default strategy over all rounds
    # ==========================================
    if local_rank == 0:
        print("\n>>> Phase 2: Running Default strategy for all rounds...", flush=True)
    results_d = run_strategy_all_rounds(
        buffer, default_strategy, inputs, num_experts, "Default", rank, args.quant_type
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
        "AlltoAll",
        rank,
        args.quant_type,
    )

    # ==========================================
    # Phase 4: Compare and display table
    # ==========================================
    dist.barrier()
    if local_rank == 0:
        print("\n" + "=" * 100, flush=True)
        print("LOW_LATENCY COMPARISON RESULTS", flush=True)
        print("=" * 100, flush=True)

    all_passed = True
    table_rows = []
    for round_idx in range(args.num_rounds):
        row = compare_round(
            results_d[round_idx],
            results_a[round_idx],
            round_idx,
            rank,
            num_ranks,
            num_local_experts,
            args,
        )
        table_rows.append(row)
        if not row["pass"]:
            all_passed = False

    # Print table (rank 0 only)
    if local_rank == 0:
        print_table(table_rows)
        print("=" * 100, flush=True)
        if all_passed:
            print(f"Status: ALL {args.num_rounds} ROUND(S) PASSED", flush=True)
        else:
            failed = sum(1 for r in table_rows if not r["pass"])
            print(
                f"Status: {failed}/{args.num_rounds} ROUND(S) FAILED",
                flush=True,
            )
        print("=" * 100 + "\n", flush=True)

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
        "Aligned",
        "Dispatch",
        "Disp MaxDiff",
        "Combine",
        "Comb MaxDiff",
        "Status",
    ]
    widths = []
    for i, h in enumerate(headers):
        w = len(h)
        for r in rows:
            val = format_cell(r, i)
            if len(val) > w:
                w = len(val)
        widths.append(w)

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
    if col_idx == 2:  # Aligned
        return str(row["aligned"])
    if col_idx == 3:  # Dispatch
        return "PASS" if row["dispatch_pass"] else "FAIL"
    if col_idx == 4:  # Disp MaxDiff
        return f"{row['dispatch_max_diff']:.6f}"
    if col_idx == 5:  # Combine
        return "PASS" if row["combine_pass"] else "FAIL"
    if col_idx == 6:  # Comb MaxDiff
        return f"{row['combine_max_diff']:.6f}"
    if col_idx == 7:  # Status
        return "PASS" if row["pass"] else "FAIL"
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare low_latency dispatch/combine accuracy between Default "
        "(deep_ep_cpp) and AlltoAll (torch.distributed) strategies via the buffer interface"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=256, help="Number of tokens (default: 256)"
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
        help="Quantization type for low_latency dispatch: no (no quantization) or int8 (default: bf16)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-expert dispatch comparison details",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_compare, args=(num_processes, args), nprocs=num_processes
    )

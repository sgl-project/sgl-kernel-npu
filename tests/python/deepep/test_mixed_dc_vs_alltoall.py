"""Compare mixed-scenario (normal + low_latency) accuracy between Default
(deep_ep_cpp) and AlltoAll (torch.distributed) strategies.

Each strategy runs the full pipeline per round:
  dispatch -> combine -> low_latency_dispatch -> low_latency_combine

As in the real mixed-deployment scenario, a single buffer (low_latency_mode=True)
is shared by both the normal and low_latency stages. Both strategies share the
same inputs per round, and results are compared bitwise (normal) / per-expert
(ll dispatch) + bitwise (ll combine). A summary table is printed at the end.
"""

import argparse
import os
import random

import deep_ep
import torch
import torch.distributed as dist
import torch_npu
from deep_ep import Buffer
from deep_ep.ep_strategy import get_low_latency_strategy
from deep_ep.strategies import AlltoAllNormalCommStrategy, DefaultNormalCommStrategy
from utils import calc_diff, get_diff_threshold, init_dist, per_token_cast_back


# ==========================================
# Normal mode helpers
# ==========================================
def run_normal(buffer, x, topk_idx, topk_weights, num_experts, config, quant_type):
    """Run normal dispatch + combine. Returns (recv_x, recv_list, combined_x)."""
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


def generate_normal_inputs(args, rank, round_idx):
    """Generate deterministic random inputs for one normal round."""
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


# ==========================================
# Low-latency mode helpers
# ==========================================
def run_ll(
    buffer, x, topk_idx, topk_weights, aligned_num_tokens, num_experts, quant_type
):
    """Run low_latency dispatch + combine. Returns (recv_x, recv_count, combined_x)."""
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
    return recv_x, recv_count, combined_x


def extract_expert_tokens_default(recv_x, recv_count, num_local_experts):
    """Extract per-expert tokens from Default ll dispatch output (compact layout)."""
    expert_tokens = []
    start = 0
    for i in range(num_local_experts):
        count = recv_count[i].item()
        expert_tokens.append(recv_x[start : start + count])
        start += count
    return expert_tokens


def extract_expert_tokens_alltoall(
    recv_x, num_local_experts, num_ranks, aligned_num_tokens, all_topk_idx, rank
):
    """Extract per-expert tokens from AlltoAll ll dispatch output (capacity layout)."""
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


def generate_ll_inputs(args, rank, num_ranks, group, round_idx, buffer_capacity):
    """Generate deterministic random inputs for one ll round.

    Returns (x, topk_idx, topk_weights, all_topk_idx, num_tokens, aligned_num_tokens).
    """
    seed = args.seed + round_idx + 10000  # offset to differ from normal seed
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

    # Align num_tokens across ranks
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


# ==========================================
# Pipeline: run one strategy over all 4 stages
# ==========================================
def run_strategy_pipeline(
    buffer,
    normal_strategy,
    ll_strategy,
    normal_input,
    ll_input,
    num_experts,
    config,
    strategy_name,
    rank,
    quant_type,
):
    """Run dispatch->combine->ll_dispatch->ll_combine for one strategy.

    All stages run on the same shared buffer (mixed deployment), switching
    only the active strategy. Returns a dict of all stage outputs.
    """
    # Stage 1+2: normal dispatch + combine
    buffer.normal_strategy = normal_strategy
    x_n, topk_idx_n, topk_weights_n, num_tokens_n = normal_input

    n_recv_x, n_recv_list, n_combined_x = run_normal(
        buffer, x_n, topk_idx_n, topk_weights_n, num_experts, config, quant_type
    )

    # Stage 3+4: low_latency dispatch + combine
    buffer.low_latency_strategy = ll_strategy
    x_l, topk_idx_l, topk_weights_l, all_topk_idx, num_tokens_l, aligned_num_tokens = (
        ll_input
    )

    ll_recv_x, ll_recv_count, ll_combined_x = run_ll(
        buffer,
        x_l,
        topk_idx_l,
        topk_weights_l,
        aligned_num_tokens,
        num_experts,
        quant_type,
    )

    return {
        "n_recv_x": n_recv_x,
        "n_recv_list": n_recv_list,
        "n_combined_x": n_combined_x,
        "n_num_tokens": num_tokens_n,
        "ll_recv_x": ll_recv_x,
        "ll_recv_count": ll_recv_count,
        "ll_combined_x": ll_combined_x,
        "ll_all_topk_idx": all_topk_idx,
        "ll_num_tokens": num_tokens_l,
        "ll_aligned_num_tokens": aligned_num_tokens,
    }


# ==========================================
# Comparison
# ==========================================
def compare_round(
    results_d, results_a, round_idx, rank, num_ranks, num_local_experts, args
):
    """Compare one round's results between Default and AlltoAll.

    Returns a dict of per-round stats for the summary table.
    """
    # --- Normal dispatch comparison (bitwise) ---
    n_list_match = results_d["n_recv_list"] == results_a["n_recv_list"]
    n_shape_match = results_d["n_recv_x"].shape == results_a["n_recv_x"].shape
    n_dispatch_pass = n_list_match and n_shape_match
    n_dispatch_max_diff = 0.0
    quant_threshold = get_diff_threshold(args.quant_type)
    if n_dispatch_pass:
        n_dispatch_max_diff = torch.max(
            torch.abs(results_d["n_recv_x"].float() - results_a["n_recv_x"].float())
        ).item()
        if args.quant_type == "bf16":
            n_dispatch_pass = n_dispatch_max_diff == 0.0
        else:
            n_dispatch_pass = (
                calc_diff(results_d["n_recv_x"].float(), results_a["n_recv_x"].float())
                < quant_threshold
            )

    # --- Normal combine comparison (bitwise) ---
    n_combine_max_diff = torch.max(
        torch.abs(results_d["n_combined_x"].float() - results_a["n_combined_x"].float())
    ).item()
    if args.quant_type == "bf16":
        n_combine_pass = n_combine_max_diff == 0.0
    else:
        n_combine_pass = (
            calc_diff(
                results_d["n_combined_x"].float(), results_a["n_combined_x"].float()
            )
            < quant_threshold
        )

    # --- LL dispatch comparison (per-expert) ---
    aligned_num_tokens = results_d["ll_aligned_num_tokens"]
    all_topk_idx = results_d["ll_all_topk_idx"]
    tokens_d = extract_expert_tokens_default(
        results_d["ll_recv_x"], results_d["ll_recv_count"], num_local_experts
    )
    tokens_a = extract_expert_tokens_alltoall(
        results_a["ll_recv_x"],
        num_local_experts,
        num_ranks,
        aligned_num_tokens,
        all_topk_idx,
        rank,
    )

    ll_dispatch_pass = True
    ll_dispatch_max_diff = 0.0
    for i in range(num_local_experts):
        expert_id = rank * num_local_experts + i
        expected_count = (all_topk_idx == expert_id).sum().item()
        td = tokens_d[i]
        ta = tokens_a[i]
        count_d = td.shape[0]
        count_a = ta.shape[0]
        count_ok = (count_d == expected_count) and (count_a == expected_count)
        if not count_ok:
            ll_dispatch_pass = False
            if args.debug:
                print(
                    f"  [Round {round_idx + 1} LL Dispatch] rank={rank} "
                    f"expert={expert_id} count mismatch: expected={expected_count}, "
                    f"default={count_d}, alltoall={count_a}",
                    flush=True,
                )
            continue
        if count_d > 0:
            expert_max_diff = torch.max(torch.abs(td.float() - ta.float())).item()
            if args.quant_type == "bf16":
                expert_ok = expert_max_diff == 0
            else:
                expert_ok = calc_diff(td.float(), ta.float()) < quant_threshold
            if not expert_ok:
                ll_dispatch_pass = False
                if expert_max_diff > ll_dispatch_max_diff:
                    ll_dispatch_max_diff = expert_max_diff
                if args.debug:
                    sorted_d, _ = td.float().sort(dim=0)
                    sorted_a, _ = ta.float().sort(dim=0)
                    sorted_max_diff = torch.max(torch.abs(sorted_d - sorted_a)).item()
                    order_note = (
                        " (order differs, set matches)"
                        if sorted_max_diff == 0
                        else " (real mismatch)"
                    )
                    print(
                        f"  [Round {round_idx + 1} LL Dispatch] rank={rank} "
                        f"expert={expert_id} max_diff={expert_max_diff:.8f}{order_note}",
                        flush=True,
                    )

    # --- LL combine comparison (bitwise) ---
    ll_combine_max_diff = torch.max(
        torch.abs(
            results_d["ll_combined_x"].float() - results_a["ll_combined_x"].float()
        )
    ).item()
    if args.quant_type == "bf16":
        ll_combine_pass = ll_combine_max_diff == 0.0
    else:
        ll_combine_pass = (
            calc_diff(
                results_d["ll_combined_x"].float(), results_a["ll_combined_x"].float()
            )
            < quant_threshold
        )

    round_pass = (
        n_dispatch_pass and n_combine_pass and ll_dispatch_pass and ll_combine_pass
    )
    return {
        "round": round_idx + 1,
        "n_tokens": results_d["n_num_tokens"],
        "n_dispatch_pass": n_dispatch_pass,
        "n_dispatch_max_diff": n_dispatch_max_diff,
        "n_combine_pass": n_combine_pass,
        "n_combine_max_diff": n_combine_max_diff,
        "ll_aligned": aligned_num_tokens,
        "ll_dispatch_pass": ll_dispatch_pass,
        "ll_dispatch_max_diff": ll_dispatch_max_diff,
        "ll_combine_pass": ll_combine_pass,
        "ll_combine_max_diff": ll_combine_max_diff,
        "pass": round_pass,
    }


# ==========================================
# Main test flow
# ==========================================
def test_compare(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts

    assert (
        num_experts % num_ranks == 0
    ), f"num_experts ({num_experts}) must be divisible by num_ranks ({num_ranks})"
    num_local_experts = num_experts // num_ranks
    ll_buffer_capacity = args.ll_num_tokens

    if local_rank == 0:
        print(
            f"[config] hidden={hidden}, num_topk={num_topk}, "
            f"num_experts={num_experts}, num_ranks={num_ranks}, seed={args.seed}, "
            f"dynamic_tokens={args.enable_dynamic_tokens}, "
            f"num_rounds={args.num_rounds}, "
            f"normal_num_tokens={args.num_tokens}, "
            f"ll_buffer_capacity={ll_buffer_capacity}, "
            f"quant_type={args.quant_type}",
            flush=True,
        )

    # The AlltoAll normal strategy reads this env var to enable INT8 quantization.
    if args.quant_type == "int8":
        os.environ["DEEP_NORMAL_MODE_USE_INT8_QUANT"] = "1"

    # ==========================================
    # Phase 1: Generate all rounds' inputs
    # ==========================================
    if local_rank == 0:
        print("\n>>> Phase 1: Generating inputs for all rounds...", flush=True)
    normal_inputs = []
    ll_inputs = []
    for round_idx in range(args.num_rounds):
        normal_inputs.append(generate_normal_inputs(args, rank, round_idx))
        ll_inputs.append(
            generate_ll_inputs(
                args, rank, num_ranks, group, round_idx, ll_buffer_capacity
            )
        )
    dist.barrier()
    if local_rank == 0:
        n_summary = ", ".join(
            f"r{i + 1}={inp[3]}" for i, inp in enumerate(normal_inputs)
        )
        ll_summary = ", ".join(f"r{i + 1}={inp[5]}" for i, inp in enumerate(ll_inputs))
        print(f"  normal num_tokens per round: {n_summary}", flush=True)
        print(f"  ll aligned_num_tokens per round: {ll_summary}", flush=True)

    # ==========================================
    # Create a single mixed buffer shared by both the normal and low_latency
    # stages (matches the real mixed-deployment scenario)
    # ==========================================
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        ll_buffer_capacity, hidden, num_ranks, num_experts
    )
    print(
        f"[Rank {rank}] Initializing mixed (normal + low_latency) buffer...",
        flush=True,
    )
    buffer = Buffer(
        group,
        num_nvl_bytes=int(2e9),
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // num_ranks,
    )
    print(f"[Rank {rank}] Mixed buffer created OK.", flush=True)

    # Create strategy instances (all bound to the same buffer runtime)
    normal_default = DefaultNormalCommStrategy(runtime=buffer.runtime, group=group)
    normal_alltoall = AlltoAllNormalCommStrategy(runtime=buffer.runtime, group=group)
    DefaultLLStrategy = get_low_latency_strategy("default")
    AlltoAllLLStrategy = get_low_latency_strategy("alltoall")
    ll_default = DefaultLLStrategy(runtime=buffer.runtime, group=group)
    ll_alltoall = AlltoAllLLStrategy(runtime=buffer.runtime, group=group)

    config = deep_ep.Config(24, 8, 256)

    # ==========================================
    # Phase 2: Run Default strategy over all rounds
    # ==========================================
    if local_rank == 0:
        print(
            "\n>>> Phase 2: Running Default strategy (all 4 stages) for all rounds...",
            flush=True,
        )
    results_d = []
    for round_idx in range(args.num_rounds):
        if rank == 0:
            print(f"  [Default] round {round_idx + 1}/{args.num_rounds}...", flush=True)
        res = run_strategy_pipeline(
            buffer,
            normal_default,
            ll_default,
            normal_inputs[round_idx],
            ll_inputs[round_idx],
            num_experts,
            config,
            "Default",
            rank,
            args.quant_type,
        )
        results_d.append(res)

    # ==========================================
    # Phase 3: Run AlltoAll strategy over all rounds
    # ==========================================
    if local_rank == 0:
        print(
            "\n>>> Phase 3: Running AlltoAll strategy (all 4 stages) for all rounds...",
            flush=True,
        )
    results_a = []
    for round_idx in range(args.num_rounds):
        if rank == 0:
            print(
                f"  [AlltoAll] round {round_idx + 1}/{args.num_rounds}...", flush=True
            )
        res = run_strategy_pipeline(
            buffer,
            normal_alltoall,
            ll_alltoall,
            normal_inputs[round_idx],
            ll_inputs[round_idx],
            num_experts,
            config,
            "AlltoAll",
            rank,
            args.quant_type,
        )
        results_a.append(res)

    # ==========================================
    # Phase 4: Compare and display table
    # ==========================================
    dist.barrier()
    if local_rank == 0:
        print("\n" + "=" * 110, flush=True)
        print("MIXED SCENARIO COMPARISON RESULTS", flush=True)
        print("=" * 110, flush=True)

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

    if local_rank == 0:
        print_table(table_rows)
        print("=" * 110, flush=True)
        if all_passed:
            print(f"Status: ALL {args.num_rounds} ROUND(S) PASSED", flush=True)
        else:
            failed = sum(1 for r in table_rows if not r["pass"])
            print(
                f"Status: {failed}/{args.num_rounds} ROUND(S) FAILED",
                flush=True,
            )
        print("=" * 110 + "\n", flush=True)

    assert all_passed, f"[rank {rank}] Some rounds failed"
    dist.barrier()
    dist.destroy_process_group()


def print_table(rows):
    """Print comparison results as a formatted table."""
    if not rows:
        return
    headers = [
        "Round",
        "N Tok",
        "N Disp",
        "N Comb",
        "LL Tok",
        "LL Disp",
        "LL Comb",
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
    if col_idx == 1:  # N Tok
        return str(row["n_tokens"])
    if col_idx == 2:  # N Disp
        return "PASS" if row["n_dispatch_pass"] else "FAIL"
    if col_idx == 3:  # N Comb
        return "PASS" if row["n_combine_pass"] else "FAIL"
    if col_idx == 4:  # LL Tok
        return str(row["ll_aligned"])
    if col_idx == 5:  # LL Disp
        return "PASS" if row["ll_dispatch_pass"] else "FAIL"
    if col_idx == 6:  # LL Comb
        return "PASS" if row["ll_combine_pass"] else "FAIL"
    if col_idx == 7:  # Status
        return "PASS" if row["pass"] else "FAIL"
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare mixed-scenario (normal + low_latency) accuracy between "
        "Default (deep_ep_cpp) and AlltoAll (torch.distributed) strategies. "
        "Each strategy runs dispatch->combine->ll_dispatch->ll_combine per round."
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of processes to spawn (default: 16)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=4096,
        help="Number of tokens for normal mode (default: 4096)",
    )
    parser.add_argument(
        "--ll-num-tokens",
        type=int,
        default=256,
        help="Buffer capacity (num_tokens) for low_latency mode (default: 256)",
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
        help="Quantization type for normal + low_latency dispatch: bf16 (no quantization) or int8 (default: bf16)",
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

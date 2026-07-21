"""A5 FusedDeepMoe correctness and performance comparison script."""

import argparse
import os
from typing import Dict, List, Tuple

import deep_ep
import torch
import torch.distributed as dist
import torch_npu
from utils import calc_diff, init_dist, profile_npu_event_sequences

torch_npu.npu.config.allow_internal_format = True

FUSED_EVENT_PATTERNS = (("FusedDeepMoe", "aclnnFusedDeepMoe"),)
SMALL_OP_EVENT_PATTERNS = (
    ("MoeDistributeDispatchV2", "MoeDistributeDispatchV3"),
    (
        "GroupedMatmul",
        "GroupedMatMul",
        "aclnnGroupedMatmulV4_GroupedMatmul_GroupedMatmul",
    ),
    ("Swiglu", "SwiGlu"),
    ("DynamicMxQuant", "DynamicMXQuant"),
    (
        "GroupedMatmul",
        "GroupedMatMul",
        "aclnnGroupedMatmulV4_GroupedMatmul_GroupedMatmul",
    ),
    ("MoeDistributeCombineV2", "MoeDistributeCombineV3"),
)
SMALL_OP_EVENT_LABELS = ("dispatch", "gmm1", "swiglu", "requant", "gmm2", "combine")
# Only the very first profiler warmup iteration gets this heavier burn-in.
# The extra work is intentionally excluded from final statistics and only
# exists to keep the device busy a bit longer before the profiled small-op
# iterations start.
SMALL_FIRST_WARMUP_GMM_BURN_IN_REPEATS = 100
# The burn-in matmul uses larger tensors than the routed path on purpose.
# Increasing dimensions tends to create a more stable device-side delay than
# simply increasing the repeat count of very small kernels.
SMALL_FIRST_WARMUP_MATMUL_DIM_SCALE = 4
ACCURACY_ATOL = 2.0
ACCURACY_RTOL = 0.02


def make_umdk_static_inputs(
    rank: int, world_size: int, args: argparse.Namespace
) -> Dict[str, torch.Tensor]:
    assert args.num_experts % world_size == 0
    local_experts = args.num_experts // world_size

    x = torch.rand((args.num_tokens, args.hidden)).bfloat16().npu() * 2 - 1
    expert_ids = torch.arange(
        rank * args.num_tokens * args.num_topk,
        (rank + 1) * args.num_tokens * args.num_topk,
        dtype=torch.int32,
    ).view(args.num_tokens, args.num_topk)
    expert_ids = expert_ids.remainder(args.num_experts).npu()
    expert_scales = torch.rand(
        (args.num_tokens, args.num_topk), dtype=torch.float32
    ).npu()

    gmm1_fp = (
        torch.rand((local_experts, args.hidden, args.moe_intermediate_size * 2))
        .bfloat16()
        .npu()
        * 2
        - 1
    )
    gmm1_weight, gmm1_scale_raw = torch_npu.npu_dynamic_mx_quant(
        gmm1_fp, dst_type=torch.float8_e4m3fn, axis=1
    )
    gmm1_scale = gmm1_scale_raw.view(torch.float8_e8m0fnu)

    gmm2_fp = (
        torch.rand((local_experts, args.moe_intermediate_size, args.hidden))
        .bfloat16()
        .npu()
        * 2
        - 1
    )
    gmm2_weight, gmm2_scale_raw = torch_npu.npu_dynamic_mx_quant(
        gmm2_fp, dst_type=torch.float8_e4m3fn, axis=1
    )
    gmm2_scale = gmm2_scale_raw.view(torch.float8_e8m0fnu)

    return {
        "x": x,
        "expert_ids": expert_ids,
        "expert_scales": expert_scales,
        "gmm1_weight_fp8": gmm1_weight,
        "gmm1_scale_fp8": gmm1_scale,
        "gmm2_weight_fp8": gmm2_weight,
        "gmm2_scale_fp8": gmm2_scale,
    }


def make_small_warmup_burn_in_buffers(
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # These tensors are dedicated to the first warmup burn-in path and are not
    # consumed by the routed MoE computation. Keeping them separate makes the
    # functional path easier to reason about.
    dim_scale = SMALL_FIRST_WARMUP_MATMUL_DIM_SCALE
    lhs_rows = args.num_tokens * dim_scale
    shared_dim = args.hidden * dim_scale
    rhs_cols = args.moe_intermediate_size * dim_scale
    lhs = torch.rand((lhs_rows, shared_dim)).bfloat16().npu() * 2 - 1
    rhs = torch.rand((shared_dim, rhs_cols)).bfloat16().npu() * 2 - 1
    return lhs, rhs


def run_small_op_baseline(
    inputs: Dict[str, torch.Tensor],
    hcomm_name: str,
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    gmm_burn_in_repeats: int = 1,
    warmup_burn_in_buffers: Tuple[torch.Tensor, torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output_dtype = inputs["x"].dtype

    outputs = torch_npu.npu_moe_distribute_dispatch_v2(
        x=inputs["x"],
        expert_ids=inputs["expert_ids"],
        expert_scales=inputs["expert_scales"],
        scales=None,
        x_active_mask=None,
        group_ep=hcomm_name,
        ep_world_size=world_size,
        ep_rank_id=rank,
        moe_expert_num=args.num_experts,
        group_tp="",
        tp_world_size=1,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        quant_mode=4,
        global_bs=args.num_tokens * world_size,
        expert_token_nums_type=1,
        y_dtype=torch.float8_e4m3fn,
    )
    (
        expand_x,
        dynamic_scales,
        assist_info_for_combine,
        expert_token_nums,
        ep_send_counts,
        tp_send_counts,
        expand_scales,
    ) = outputs

    if gmm_burn_in_repeats > 1 and warmup_burn_in_buffers is not None:
        # This burn-in only runs on the first profiler warmup iteration.
        # It does not change the routed output; it only inserts extra device
        # work before the normal small-op chain of that warmup iteration.
        burn_in_lhs, burn_in_rhs = warmup_burn_in_buffers
        for _ in range(gmm_burn_in_repeats):
            _ = torch.matmul(burn_in_lhs, burn_in_rhs)

    dynamic_scales = dynamic_scales.view(*(dynamic_scales.shape[:-1]), -1, 2).view(
        torch.float8_e8m0fnu
    )

    y1_fp = torch_npu.npu_grouped_matmul(
        x=[expand_x],
        weight=[inputs["gmm1_weight_fp8"]],
        scale=[inputs["gmm1_scale_fp8"]],
        per_token_scale=[dynamic_scales],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_token_nums,
        output_dtype=output_dtype,
    )[0]
    swiglu_out = torch_npu.npu_swiglu(y1_fp)

    x2, x2_scale = torch_npu.npu_dynamic_mx_quant(
        swiglu_out, dst_type=torch.float8_e4m3fn
    )

    x2_scale = x2_scale.view(torch.float8_e8m0fnu)
    y2_fp = torch_npu.npu_grouped_matmul(
        x=[x2],
        weight=[inputs["gmm2_weight_fp8"]],
        scale=[inputs["gmm2_scale_fp8"]],
        per_token_scale=[x2_scale],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_token_nums,
        output_dtype=output_dtype,
    )[0]
    output = torch_npu.npu_moe_distribute_combine_v2(
        expand_x=y2_fp,
        expert_ids=inputs["expert_ids"],
        assist_info_for_combine=assist_info_for_combine,
        ep_send_counts=ep_send_counts,
        expert_scales=inputs["expert_scales"],
        x_active_mask=None,
        group_ep=hcomm_name,
        ep_world_size=world_size,
        ep_rank_id=rank,
        moe_expert_num=args.num_experts,
        tp_send_counts=tp_send_counts,
        expand_scales=expand_scales,
        group_tp="",
        tp_world_size=1,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        global_bs=args.num_tokens * world_size,
    )
    return output, expert_token_nums.to(torch.int32)


def run_buffer_fused(
    buffer: deep_ep.Buffer,
    inputs: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output, ep_recv_count = buffer.fused_deep_moe(
        inputs["x"],
        inputs["expert_ids"],
        inputs["expert_scales"],
        inputs["gmm1_weight_fp8"],
        inputs["gmm1_scale_fp8"],
        inputs["gmm2_weight_fp8"],
        inputs["gmm2_scale_fp8"],
        args.num_tokens,
        args.num_experts,
        args.quant_mode,
    )
    return output, ep_recv_count


def run_buffer_fused_with_burn_in(
    buffer: deep_ep.Buffer,
    inputs: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    fused_burn_in_repeats: int = 1,
    warmup_burn_in_buffers: Tuple[torch.Tensor, torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # The extra burn-in is only used for the first profiler warmup iteration.
    # It intentionally stays on the same execution path as the fused op so the
    # host has more time to enqueue the remaining fused iterations while this
    # warmup is still occupying device time.
    if fused_burn_in_repeats > 1 and warmup_burn_in_buffers is not None:
        burn_in_lhs, burn_in_rhs = warmup_burn_in_buffers
        for _ in range(fused_burn_in_repeats):
            _ = torch.matmul(burn_in_lhs, burn_in_rhs)

    return run_buffer_fused(buffer, inputs, args)


def format_triplet(name: str, values_us: Tuple[float, float, float]) -> str:
    avg_us, min_us, max_us = values_us
    return f"{name}: avg={avg_us:.2f} us, min={min_us:.2f} us, max={max_us:.2f} us"


def summarize_profile_durations(durations_s) -> Tuple[float, float, float]:
    return (
        float(durations_s.mean()),
        float(durations_s.min()),
        float(durations_s.max()),
    )


def summarize_profile_breakdown(step_durations_s) -> List[Tuple[float, float, float]]:
    return [
        (
            float(step_durations_s[:, idx].mean()),
            float(step_durations_s[:, idx].min()),
            float(step_durations_s[:, idx].max()),
        )
        for idx in range(step_durations_s.shape[1])
    ]


def print_profile_match_debug(
    rank: int, title: str, discovered_names, matched_sequences
):
    if rank != 0:
        return
    if matched_sequences:
        print(f"{title} matched profiler event sequences:", flush=True)
        for idx, sequence in enumerate(matched_sequences, start=1):
            print(f"  [{idx}] {' -> '.join(sequence)}", flush=True)
    else:
        print(f"{title} discovered profiler events:", flush=True)
        for name in discovered_names:
            print(f"  - {name}", flush=True)


def print_profile_iteration_debug(rank: int, title: str, debug_info):
    if rank != 0:
        return
    print(
        f"{title} profiler iterations: "
        f"matched_total_iterations={debug_info['matched_total_iterations']}, "
        f"dropped_warmup_iterations={debug_info['dropped_warmup_iterations']}, "
        f"counted_iterations={debug_info['counted_iterations']}",
        flush=True,
    )


def print_rank_skew_summary(gathered_small, gathered_small_breakdown, gathered_fused):
    # This table is primarily a maintenance/debugging view. It keeps the
    # per-rank stage averages together so rank-to-rank skew can be judged
    # without manually correlating multiple profiler tables or traces.
    small_tensor = torch.stack(gathered_small).cpu()
    fused_tensor = (
        torch.stack(gathered_fused).cpu() if gathered_fused is not None else None
    )
    breakdown_tensor = torch.stack(gathered_small_breakdown).cpu()

    print("rank skew summary:", flush=True)
    print(
        "| Rank | Dispatch (us) | GMM1 (us) | SwiGLU (us) | Requant (us) | GMM2 (us) | Combine (us) | Small Total (us) | Fused (us) |",
        flush=True,
    )
    print(
        "|-----:|--------------:|----------:|------------:|-------------:|----------:|-------------:|-----------------:|-----------:|",
        flush=True,
    )

    for rank_idx in range(len(gathered_small)):
        breakdown_avg = breakdown_tensor[rank_idx, :, 0]
        dispatch_avg = float(breakdown_avg[0].item())
        gmm1_avg = float(breakdown_avg[1].item())
        swiglu_avg = float(breakdown_avg[2].item())
        requant_avg = float(breakdown_avg[3].item())
        gmm2_avg = float(breakdown_avg[4].item())
        combine_avg = float(breakdown_avg[5].item())
        small_avg = float(small_tensor[rank_idx, 0].item())
        fused_avg = (
            float(fused_tensor[rank_idx, 0].item())
            if fused_tensor is not None
            else float("nan")
        )
        fused_text = f"{fused_avg * 1e6:.2f}" if fused_tensor is not None else "-"
        print(
            f"| {rank_idx:>4} | {dispatch_avg * 1e6:>13.2f} | {gmm1_avg * 1e6:>9.2f} | "
            f"{swiglu_avg * 1e6:>11.2f} | {requant_avg * 1e6:>12.2f} | {gmm2_avg * 1e6:>9.2f} | "
            f"{combine_avg * 1e6:>12.2f} | {small_avg * 1e6:>16.2f} | {fused_text:>10} |",
            flush=True,
        )

    mean_breakdown_avg = breakdown_tensor[:, :, 0].mean(dim=0)
    mean_small_avg = float(small_tensor[:, 0].mean().item())
    mean_fused_avg = (
        float(fused_tensor[:, 0].mean().item())
        if fused_tensor is not None
        else float("nan")
    )
    mean_fused_text = f"{mean_fused_avg * 1e6:.2f}" if fused_tensor is not None else "-"
    print(
        f"| {'mean':>4} | {float(mean_breakdown_avg[0].item()) * 1e6:>13.2f} | "
        f"{float(mean_breakdown_avg[1].item()) * 1e6:>9.2f} | "
        f"{float(mean_breakdown_avg[2].item()) * 1e6:>11.2f} | "
        f"{float(mean_breakdown_avg[3].item()) * 1e6:>12.2f} | "
        f"{float(mean_breakdown_avg[4].item()) * 1e6:>9.2f} | "
        f"{float(mean_breakdown_avg[5].item()) * 1e6:>12.2f} | "
        f"{mean_small_avg * 1e6:>16.2f} | {mean_fused_text:>10} |",
        flush=True,
    )


def print_comm_overlap_rate(total_small_stats, total_fused_stats, mean_breakdown_stats):
    # Communication overlap rate is reported as a coarse derived metric using
    # mean-over-ranks averages:
    #   (small_total - fused_total) / (dispatch + combine)
    # It is intended for cross-run trend comparison rather than as a strict
    # micro-kernel efficiency metric.
    dispatch_avg = mean_breakdown_stats[0][0]
    combine_avg = mean_breakdown_stats[5][0]
    denom = dispatch_avg + combine_avg
    if denom <= 0:
        print("communication overlap rate: N/A (dispatch + combine <= 0)", flush=True)
        return
    overlap_rate = (total_small_stats[0] - total_fused_stats[0]) / denom
    print(
        "communication overlap rate: "
        f"(({total_small_stats[0] * 1e6:.2f} - {total_fused_stats[0] * 1e6:.2f}) / "
        f"({dispatch_avg * 1e6:.2f} + {combine_avg * 1e6:.2f})) = {overlap_rate:.4f}",
        flush=True,
    )


def build_trace_path(trace_dir: str, rank: int, tag: str) -> str:
    os.makedirs(trace_dir, exist_ok=True)
    return os.path.join(trace_dir, f"{tag}_rank{rank}.json")


def format_stats_row(
    name: str, stats_s: Tuple[float, float, float], ratio_pct: float = None
) -> str:
    avg_us, min_us, max_us = (value * 1e6 for value in stats_s)
    ratio_str = "-" if ratio_pct is None else f"{ratio_pct:6.2f}%"
    return f"| {name:<12} | {avg_us:>12.2f} | {min_us:>12.2f} | {max_us:>12.2f} | {ratio_str:>8} |"


def print_stats_table(
    title: str, rows: List[Tuple[str, Tuple[float, float, float], float]]
):
    print(title, flush=True)
    print(
        "| Stage        |    Avg (us) |    Min (us) |    Max (us) |  Share % |",
        flush=True,
    )
    print(
        "|--------------|-------------:|-------------:|-------------:|---------:|",
        flush=True,
    )
    for name, stats_s, ratio_pct in rows:
        print(format_stats_row(name, stats_s, ratio_pct), flush=True)


def build_small_rows(
    total_stats: Tuple[float, float, float],
    breakdown_stats: List[Tuple[float, float, float]],
) -> List[Tuple[str, Tuple[float, float, float], float]]:
    rows = []
    for label, stats_tuple in zip(SMALL_OP_EVENT_LABELS, breakdown_stats):
        ratio_pct = (
            stats_tuple[0] / total_stats[0] * 100.0 if total_stats[0] > 0 else 0.0
        )
        rows.append((label, stats_tuple, ratio_pct))
    rows.append(("total", total_stats, 100.0))
    return rows


def print_per_rank_profile_tables(
    gathered_small,
    gathered_small_breakdown,
    gathered_fused,
):
    world_size = 0
    if gathered_small is not None:
        world_size = len(gathered_small)
    elif gathered_fused is not None:
        world_size = len(gathered_fused)

    for rank_idx in range(world_size):
        small_stats = None
        fused_stats = None

        if gathered_small is not None:
            small_stats = tuple(
                float(v) for v in gathered_small[rank_idx].cpu().tolist()
            )
            small_breakdown_stats = [
                tuple(float(v) for v in stats_values)
                for stats_values in gathered_small_breakdown[rank_idx].cpu().tolist()
            ]
            print_stats_table(
                f"small-op breakdown (rank {rank_idx}):",
                build_small_rows(small_stats, small_breakdown_stats),
            )

        if gathered_fused is not None:
            fused_stats = tuple(
                float(v) for v in gathered_fused[rank_idx].cpu().tolist()
            )
            print_stats_table(
                f"fused buffer path (rank {rank_idx}):",
                [("fused", fused_stats, 100.0)],
            )

        if small_stats is not None and fused_stats is not None:
            speedup = small_stats[0] / fused_stats[0]
            delta_pct = (fused_stats[0] - small_stats[0]) / small_stats[0] * 100.0
            print(
                f"[rank {rank_idx}] small_total_avg_us={small_stats[0] * 1e6:.2f}, "
                f"fused_avg_us={fused_stats[0] * 1e6:.2f}, "
                f"speedup={speedup:.4f}x, delta_pct={delta_pct:.2f}%",
                flush=True,
            )


def print_fused_counts_table(gathered_fused_counts: List[torch.Tensor]):
    print("fused counts (per rank):", flush=True)
    print("| Rank | Local Experts | Fused Counts         | Sum |", flush=True)
    print("|-----:|--------------:|----------------------|----:|", flush=True)
    for rank_idx, counts in enumerate(gathered_fused_counts):
        counts_list = [int(v) for v in counts.cpu().tolist()]
        print(
            f"| {rank_idx:>4} | {len(counts_list):>13} | {str(counts_list):<20} | {sum(counts_list):>3} |",
            flush=True,
        )


def summarize_output_diff(
    reference: torch.Tensor, actual: torch.Tensor
) -> Tuple[float, float, float]:
    reference_f = reference.float()
    actual_f = actual.float()
    eps = 1e-8
    reference_nozero = torch.where(reference_f == 0, eps, reference_f)
    rel_diff = torch.abs(actual_f - reference_f) / torch.abs(reference_nozero)
    avg_diff = torch.mean(rel_diff).item()
    max_diff = torch.max(rel_diff).item()
    cosine_diff = calc_diff(reference_f, actual_f)
    return avg_diff, max_diff, cosine_diff


def summarize_tensor_stats(tensor: torch.Tensor) -> Tuple[float, float]:
    tensor_f = tensor.float()
    return tensor_f.abs().max().item(), tensor_f.mean().item()


def get_uniform_expected_counts(
    num_tokens: int,
    world_size: int,
    num_topk: int,
    num_experts: int,
    local_experts: int,
):
    # The synthetic expert_ids in this test are built with
    # arange(...).remainder(num_experts), so exact per-expert expected counts
    # only exist when the total routed assignments divide evenly by num_experts.
    # When that is not true, we intentionally skip strict expected_counts
    # assertions and only enforce small_counts == fused_counts.
    total_assignments = num_tokens * world_size * num_topk
    if total_assignments % num_experts != 0:
        return None, total_assignments
    expected_per_expert = total_assignments // num_experts
    expected_counts = torch.full(
        (local_experts,),
        expected_per_expert,
        dtype=torch.int32,
        device="npu",
    )
    return expected_counts, total_assignments


@torch.inference_mode()
def run_rank(local_rank: int, num_processes: int, args: argparse.Namespace):
    group = None
    group_small = None
    group_fused = None
    try:
        rank, world_size, group = init_dist(local_rank, num_processes)
        torch.manual_seed(2026 + rank)
        ranks = list(range(world_size))
        group_fused = dist.new_group(ranks)
        group_small = dist.new_group(ranks)
        buffer = deep_ep.Buffer(group_fused, low_latency_mode=True)
        assert (
            buffer.runtime.is_a5_build()
        ), "The installed DeepEP wheel is not an A5 build"

        inputs = make_umdk_static_inputs(rank, world_size, args)
        warmup_burn_in_buffers = make_small_warmup_burn_in_buffers(args)
        hcomm_name_small = group_small._get_backend(
            torch.device("npu")
        ).get_hccl_comm_name(rank)
        local_experts = args.num_experts // world_size
        expected_counts, total_assignments = get_uniform_expected_counts(
            args.num_tokens,
            world_size,
            args.num_topk,
            args.num_experts,
            local_experts,
        )
        if rank == 0:
            print(
                "Config: "
                f"num_processes={num_processes}, "
                f"num_tokens={args.num_tokens}, "
                f"hidden={args.hidden}, "
                f"moe_intermediate_size={args.moe_intermediate_size}, "
                f"num_experts={args.num_experts}, "
                f"num_topk={args.num_topk}, "
                f"quant_mode={args.quant_mode}, "
                f"num_warmups={args.num_warmups}, "
                f"num_tests={args.num_tests}, "
                f", small_first_warmup_gmm_burn_in_repeats={SMALL_FIRST_WARMUP_GMM_BURN_IN_REPEATS}",
                flush=True,
            )
            if expected_counts is None:
                print(
                    "Warning: num_tokens * num_processes * num_topk is not divisible by num_experts. "
                    "This synthetic expert_ids construction will produce non-uniform expert token counts, "
                    "so strict per-expert expected_counts checks are skipped; only small_counts == fused_counts "
                    "is enforced.",
                    flush=True,
                )

        small_output, small_counts = run_small_op_baseline(
            inputs, hcomm_name_small, rank, world_size, args
        )
        fused_output, fused_counts = run_buffer_fused(buffer, inputs, args)
        torch.npu.synchronize()

        assert small_output.shape == (args.num_tokens, args.hidden)
        assert fused_output.shape == (args.num_tokens, args.hidden)
        assert small_output.dtype == torch.bfloat16
        assert fused_output.dtype == torch.bfloat16
        assert small_counts.shape == (local_experts,)
        assert fused_counts.shape == (local_experts,)
        assert torch.isnan(small_output).sum().item() == 0
        assert torch.isnan(fused_output).sum().item() == 0

        torch.testing.assert_close(small_counts, fused_counts)
        if expected_counts is not None:
            torch.testing.assert_close(small_counts, expected_counts)
            torch.testing.assert_close(fused_counts, expected_counts)
        valid_token_num = args.num_tokens

        avg_diff, max_diff, cosine_diff = summarize_output_diff(
            small_output[:valid_token_num], fused_output[:valid_token_num]
        )
        small_absmax, small_mean = summarize_tensor_stats(
            small_output[:valid_token_num]
        )
        fused_absmax, fused_mean = summarize_tensor_stats(
            fused_output[:valid_token_num]
        )
        diag_tensor = torch.tensor(
            [
                avg_diff,
                max_diff,
                cosine_diff,
                small_absmax,
                small_mean,
                fused_absmax,
                fused_mean,
            ],
            dtype=torch.float32,
            device="npu",
        )
        gathered_diag = [torch.empty_like(diag_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_diag, diag_tensor)
        gathered_fused_counts = [
            torch.empty_like(fused_counts) for _ in range(world_size)
        ]
        dist.all_gather(gathered_fused_counts, fused_counts)
        dist.barrier()
        torch.testing.assert_close(
            small_output[:valid_token_num].float(),
            fused_output[:valid_token_num].float(),
            atol=ACCURACY_ATOL,
            rtol=ACCURACY_RTOL,
        )
        if rank == 0:
            print_fused_counts_table(gathered_fused_counts)
            gathered_diag_cpu = torch.stack(gathered_diag).cpu()
            print(
                "Accuracy check passed. "
                f"avg_diff={gathered_diag_cpu[:, 0].max().item():.6f}, "
                f"max_diff={gathered_diag_cpu[:, 1].max().item():.6f}, "
                f"calc_diff={gathered_diag_cpu[:, 2].max().item():.6f}",
                flush=True,
            )

        dist.barrier()
        small_stats = None
        small_breakdown_stats = None
        fused_stats = None

        if True:
            small_trace_path = (
                build_trace_path(args.trace_dir, rank, "small_op")
                if args.trace_dir is not None
                else None
            )
            small_profile_call_idx = 0

            def small_profile_fn():
                nonlocal small_profile_call_idx
                # Only the first warmup iteration gets the extra burn-in.
                # All later warmups and all counted iterations run the normal
                # small-op path so the reported profiler statistics remain
                # comparable to real steady-state execution.
                gmm_burn_in_repeats = (
                    SMALL_FIRST_WARMUP_GMM_BURN_IN_REPEATS
                    if small_profile_call_idx == 0 and args.num_warmups > 0
                    else 1
                )
                small_profile_call_idx += 1
                return run_small_op_baseline(
                    inputs,
                    hcomm_name_small,
                    rank,
                    world_size,
                    args,
                    gmm_burn_in_repeats=gmm_burn_in_repeats,
                    warmup_burn_in_buffers=warmup_burn_in_buffers,
                )

            (
                small_durations,
                small_event_names,
                small_matched_sequences,
                small_step_durations,
                small_debug_info,
            ) = profile_npu_event_sequences(
                small_profile_fn,
                SMALL_OP_EVENT_PATTERNS,
                num_warmups=args.num_warmups,
                num_tests=args.num_tests,
                suppress_kineto_output=True,
                trace_path=small_trace_path,
                allow_no_match=args.dump_profile_events,
            )
            if args.dump_profile_events:
                print_profile_iteration_debug(rank, "small-op", small_debug_info)
                print_profile_match_debug(
                    rank, "small-op", small_event_names, small_matched_sequences
                )
            if len(small_durations) == 0:
                raise AssertionError(
                    "No matched NPU event sequence found for small-op. "
                    f"Patterns={SMALL_OP_EVENT_PATTERNS}. "
                    f"Discovered events={small_event_names}"
                )
            small_stats = summarize_profile_durations(small_durations)
            small_breakdown_stats = summarize_profile_breakdown(small_step_durations)
        dist.barrier()

        if True:
            fused_trace_path = (
                build_trace_path(args.trace_dir, rank, "fused")
                if args.trace_dir is not None
                else None
            )
            fused_profile_call_idx = 0

            def fused_profile_fn():
                nonlocal fused_profile_call_idx
                # Only the first profiler warmup iteration gets the extra
                # burn-in. The extra work is excluded from the profiled
                # iterations, so it only helps keep the device occupied longer
                # during launch without changing the measured fused KPI.
                fused_burn_in_repeats = (
                    SMALL_FIRST_WARMUP_GMM_BURN_IN_REPEATS
                    if fused_profile_call_idx == 0 and args.num_warmups > 0
                    else 1
                )
                fused_profile_call_idx += 1
                return run_buffer_fused_with_burn_in(
                    buffer,
                    inputs,
                    args,
                    fused_burn_in_repeats=fused_burn_in_repeats,
                    warmup_burn_in_buffers=warmup_burn_in_buffers,
                )

            (
                fused_durations,
                fused_event_names,
                fused_matched_sequences,
                _,
                fused_debug_info,
            ) = profile_npu_event_sequences(
                fused_profile_fn,
                FUSED_EVENT_PATTERNS,
                num_warmups=args.num_warmups,
                num_tests=args.num_tests,
                suppress_kineto_output=True,
                trace_path=fused_trace_path,
                allow_no_match=args.dump_profile_events,
            )
            if args.dump_profile_events:
                print_profile_iteration_debug(rank, "fused", fused_debug_info)
                print_profile_match_debug(
                    rank, "fused", fused_event_names, fused_matched_sequences
                )
            if len(fused_durations) == 0:
                raise AssertionError(
                    "No matched NPU event sequence found for fused. "
                    f"Patterns={FUSED_EVENT_PATTERNS}. "
                    f"Discovered events={fused_event_names}"
                )
            fused_stats = summarize_profile_durations(fused_durations)
        dist.barrier()

        if small_stats is not None:
            small_stats_tensor = torch.tensor(
                small_stats, dtype=torch.float32, device="npu"
            )
            gathered_small = [
                torch.empty_like(small_stats_tensor) for _ in range(world_size)
            ]
            dist.all_gather(gathered_small, small_stats_tensor)
            small_breakdown_tensor = torch.tensor(
                small_breakdown_stats, dtype=torch.float32, device="npu"
            )
            gathered_small_breakdown = [
                torch.empty_like(small_breakdown_tensor) for _ in range(world_size)
            ]
            dist.all_gather(gathered_small_breakdown, small_breakdown_tensor)
        else:
            gathered_small = None
            gathered_small_breakdown = None

        if fused_stats is not None:
            fused_stats_tensor = torch.tensor(
                fused_stats, dtype=torch.float32, device="npu"
            )
            gathered_fused = [
                torch.empty_like(fused_stats_tensor) for _ in range(world_size)
            ]
            dist.all_gather(gathered_fused, fused_stats_tensor)
        else:
            gathered_fused = None

        if rank == 0:
            print("Profiled NPU op time:", flush=True)
            small_mean = None
            fused_mean = None
            total_small_stats = None
            mean_breakdown_stats = None
            total_fused_stats = None
            if gathered_small is not None:
                small_mean = torch.stack(gathered_small).mean(dim=0).cpu().tolist()
                total_small_stats = tuple(small_mean)
                small_breakdown_mean = (
                    torch.stack(gathered_small_breakdown).mean(dim=0).cpu().tolist()
                )
                mean_breakdown_stats = [
                    tuple(float(v) for v in stats_values)
                    for stats_values in small_breakdown_mean
                ]
                print_stats_table(
                    "small-op breakdown (mean over ranks):",
                    build_small_rows(
                        total_small_stats,
                        mean_breakdown_stats,
                    ),
                )
            if gathered_fused is not None:
                fused_mean = torch.stack(gathered_fused).mean(dim=0).cpu().tolist()
                total_fused_stats = tuple(fused_mean)
                fused_rows = [("fused", total_fused_stats, 100.0)]
                print_stats_table("fused buffer path (mean over ranks):", fused_rows)
            if small_mean is not None and fused_mean is not None:
                speedup = small_mean[0] / fused_mean[0]
                delta_pct = (fused_mean[0] - small_mean[0]) / small_mean[0] * 100.0
                print(f"speedup={speedup:.4f}x, delta_pct={delta_pct:.2f}%", flush=True)
                print_rank_skew_summary(
                    gathered_small,
                    gathered_small_breakdown,
                    gathered_fused,
                )
                print_comm_overlap_rate(
                    total_small_stats,
                    total_fused_stats,
                    mean_breakdown_stats,
                )
            if args.print_per_rank_profile:
                print_per_rank_profile_tables(
                    gathered_small,
                    gathered_small_breakdown,
                    gathered_fused,
                )
    finally:
        if dist.is_initialized():
            dist.barrier()
            if group_small is not None:
                dist.destroy_process_group(group_small)
            if group_fused is not None:
                dist.destroy_process_group(group_fused)
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="A5 fused vs small-op correctness and profiler performance comparison"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of spawned ranks/devices to use for the test.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=32,
        help="Per-rank token count for the routed tokens.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=7168,
        help="Hidden size of the MoE input/output tensor.",
    )
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=3072,
        help="Per-expert intermediate size; 2x this value must satisfy the A5 GMM1 constraint.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=64,
        help="Global number of routed experts across all ranks.",
    )
    parser.add_argument(
        "--num-topk",
        type=int,
        default=6,
        help="Top-k experts selected for each token.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=5,
        help="Number of profiler-only warmup iterations to exclude from performance statistics.",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=30,
        help="Number of counted performance iterations when --profile-num-tests is not set.",
    )
    parser.add_argument(
        "--quant-mode",
        type=int,
        default=0,
        help="Fused operator quant mode; currently supports 0 and 1.",
    )
    parser.add_argument(
        "--trace-dir",
        help="Optional directory to export profiler chrome traces.",
    )
    parser.add_argument(
        "--dump-profile-events",
        "--debug",
        dest="dump_profile_events",
        action="store_true",
        help="Print matched profiler iterations and discovered event names for debugging.",
    )
    parser.add_argument(
        "--print-per-rank-profile",
        action="store_true",
        help="Print per-rank small-op/fused profiler tables in addition to the mean-over-ranks summary.",
    )
    args = parser.parse_args()

    gmm1_hidden = 2 * args.moe_intermediate_size
    if args.num_processes <= 0:
        parser.error("--num-processes must be positive")
    if args.num_experts % args.num_processes != 0:
        parser.error("--num-experts must be divisible by --num-processes")
    if not 1 <= args.num_topk <= min(args.num_experts, 12):
        parser.error("--num-topk must be in [1, min(--num-experts, 12)]")
    if not 1 <= args.num_tokens <= 256:
        parser.error("--num-tokens must be in [1, 256]")
    if not 512 <= args.hidden <= 7168:
        parser.error("--hidden must be in [512, 7168]")
    if not 1024 <= gmm1_hidden <= 6144 or gmm1_hidden % 1024 != 0:
        parser.error(
            "2 * --moe-intermediate-size must be in [1024, 6144] and divisible by 1024"
        )
    if args.num_warmups < 0:
        parser.error("--num-warmups must be non-negative")
    if args.num_tests <= 0:
        parser.error("--num-tests must be positive")
    if args.quant_mode not in (0, 1):
        parser.error("--quant-mode currently only supports 0 or 1")

    torch.multiprocessing.spawn(
        run_rank, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()

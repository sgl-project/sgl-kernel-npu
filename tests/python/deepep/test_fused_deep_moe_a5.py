"""A5 FusedDeepMoe correctness and performance comparison script."""

import argparse
import os
from typing import Dict, Tuple

import deep_ep
import torch
import torch.distributed as dist
import torch_npu

from utils import bench, calc_diff, init_dist

torch_npu.npu.config.allow_internal_format = True


def make_umdk_static_inputs(
    rank: int, world_size: int, args: argparse.Namespace
) -> Dict[str, torch.Tensor]:
    assert args.num_experts % world_size == 0
    local_experts = args.num_experts // world_size

    x = torch.rand((args.batch_size, args.hidden)).bfloat16().npu() * 2 - 1
    expert_ids = torch.arange(
        rank * args.batch_size * args.topk,
        (rank + 1) * args.batch_size * args.topk,
        dtype=torch.int32,
    ).view(args.batch_size, args.topk)
    expert_ids = expert_ids.remainder(args.num_experts).npu()
    expert_scales = torch.rand((args.batch_size, args.topk), dtype=torch.float32).npu()

    gmm1_fp = torch.rand(
        (local_experts, args.hidden, args.moe_intermediate_size * 2)
    ).bfloat16().npu() * 2 - 1
    gmm1_weight, gmm1_scale_raw = torch_npu.npu_dynamic_mx_quant(
        gmm1_fp, dst_type=torch.float8_e4m3fn, axis=1
    )
    gmm1_scale = gmm1_scale_raw.view(torch.float8_e8m0fnu)

    gmm2_fp = torch.rand(
        (local_experts, args.moe_intermediate_size, args.hidden)
    ).bfloat16().npu() * 2 - 1
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


def run_small_op_baseline(
    inputs: Dict[str, torch.Tensor],
    hcomm_name: str,
    rank: int,
    world_size: int,
    args: argparse.Namespace,
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
        global_bs=args.batch_size * world_size,
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
        global_bs=args.batch_size * world_size,
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
        args.batch_size,
        args.num_experts,
        args.quant_mode,
    )
    return output, ep_recv_count


def format_triplet(name: str, values_us: Tuple[float, float, float]) -> str:
    avg_us, min_us, max_us = values_us
    return f"{name}: avg={avg_us:.2f} us, min={min_us:.2f} us, max={max_us:.2f} us"


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
        assert buffer.runtime.is_a5_build(), "The installed DeepEP wheel is not an A5 build"

        inputs = make_umdk_static_inputs(rank, world_size, args)
        hcomm_name_small = group_small._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        local_experts = args.num_experts // world_size
        expected_counts = torch.full(
            (local_experts,),
            args.batch_size * world_size * args.topk // args.num_experts,
            dtype=torch.int32,
            device="npu",
        )
        if rank == 0:
            print(
                "Config: "
                f"num_processes={num_processes}, "
                f"batch_size={args.batch_size}, "
                f"hidden={args.hidden}, "
                f"moe_intermediate_size={args.moe_intermediate_size}, "
                f"num_experts={args.num_experts}, "
                f"topk={args.topk}, "
                f"quant_mode={args.quant_mode}, "
                f"num_warmups={args.num_warmups}, "
                f"num_tests={args.num_tests}",
                flush=True,
            )
        run_small_op_baseline(inputs, hcomm_name_small, rank, world_size, args)
        run_buffer_fused(buffer, inputs, args)
        torch.npu.synchronize()
        dist.barrier()
        if rank == 0:
            print("Warmup completed.", flush=True)

        small_output, small_counts = run_small_op_baseline(
            inputs, hcomm_name_small, rank, world_size, args
        )
        fused_output, fused_counts = run_buffer_fused(buffer, inputs, args)
        torch.npu.synchronize()

        assert small_output.shape == (args.batch_size, args.hidden)
        assert fused_output.shape == (args.batch_size, args.hidden)
        assert small_output.dtype == torch.bfloat16
        assert fused_output.dtype == torch.bfloat16
        assert small_counts.shape == (local_experts,)
        assert fused_counts.shape == (local_experts,)
        assert torch.isnan(small_output).sum().item() == 0
        assert torch.isnan(fused_output).sum().item() == 0

        torch.testing.assert_close(small_counts, expected_counts)
        torch.testing.assert_close(fused_counts, expected_counts)
        torch.testing.assert_close(small_counts, fused_counts)
        valid_token_num = args.batch_size

        avg_diff, max_diff, cosine_diff = summarize_output_diff(
            small_output[:valid_token_num], fused_output[:valid_token_num]
        )
        small_absmax, small_mean = summarize_tensor_stats(small_output[:valid_token_num])
        fused_absmax, fused_mean = summarize_tensor_stats(fused_output[:valid_token_num])
        diag_tensor = torch.tensor(
            [avg_diff, max_diff, cosine_diff, small_absmax, small_mean, fused_absmax, fused_mean],
            dtype=torch.float64,
            device="npu",
        )
        gathered_diag = [torch.empty_like(diag_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_diag, diag_tensor)
        dist.barrier()
        torch.testing.assert_close(
            small_output[:valid_token_num].float(),
            fused_output[:valid_token_num].float(),
            atol=args.atol,
            rtol=args.rtol,
        )
        if rank == 0:
            gathered_diag_cpu = torch.stack(gathered_diag).cpu()
            print(
                "Accuracy check passed. "
                f"avg_diff={gathered_diag_cpu[:, 0].max().item():.6f}, "
                f"max_diff={gathered_diag_cpu[:, 1].max().item():.6f}, "
                f"calc_diff={gathered_diag_cpu[:, 2].max().item():.6f}",
                flush=True,
            )

        dist.barrier()
        small_stats = bench(
            lambda: run_small_op_baseline(inputs, hcomm_name_small, rank, world_size, args),
            num_warmups=args.num_warmups,
            num_tests=args.num_tests,
        )
        dist.barrier()
        fused_stats = bench(
            lambda: run_buffer_fused(buffer, inputs, args),
            num_warmups=args.num_warmups,
            num_tests=args.num_tests,
        )
        dist.barrier()

        small_stats_tensor = torch.tensor(small_stats, dtype=torch.float64, device="npu")
        fused_stats_tensor = torch.tensor(fused_stats, dtype=torch.float64, device="npu")
        gathered_small = [torch.empty_like(small_stats_tensor) for _ in range(world_size)]
        gathered_fused = [torch.empty_like(fused_stats_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_small, small_stats_tensor)
        dist.all_gather(gathered_fused, fused_stats_tensor)

        if rank == 0:
            small_mean = torch.stack(gathered_small).mean(dim=0).cpu().tolist()
            fused_mean = torch.stack(gathered_fused).mean(dim=0).cpu().tolist()
            small_us = tuple(v * 1e6 for v in small_mean)
            fused_us = tuple(v * 1e6 for v in fused_mean)
            speedup = small_mean[0] / fused_mean[0]
            delta_pct = (fused_mean[0] - small_mean[0]) / small_mean[0] * 100.0
            print(format_triplet("small-op baseline (mean over ranks)", small_us), flush=True)
            print(format_triplet("fused buffer path (mean over ranks)", fused_us), flush=True)
            print(f"speedup={speedup:.4f}x, delta_pct={delta_pct:.2f}%", flush=True)
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
        description="A5 FusedDeepMoe correctness and performance comparison"
    )
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--moe-intermediate-size", type=int, default=3072)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--num-tests", type=int, default=10)
    parser.add_argument("--quant-mode", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2.0)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument(
        "--master-addr",
        help="Override MASTER_ADDR for the HCCL process-group rendezvous.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        help="Override MASTER_PORT for the HCCL process-group rendezvous.",
    )
    args = parser.parse_args()

    gmm1_hidden = 2 * args.moe_intermediate_size
    if args.num_processes <= 0:
        parser.error("--num-processes must be positive")
    if args.num_experts % args.num_processes != 0:
        parser.error("--num-experts must be divisible by --num-processes")
    if not 1 <= args.topk <= min(args.num_experts, 12):
        parser.error("--topk must be in [1, min(--num-experts, 12)]")
    if not 1 <= args.batch_size <= 256:
        parser.error("--batch-size must be in [1, 256]")
    if not 512 <= args.hidden <= 7168:
        parser.error("--hidden must be in [512, 7168]")
    if not 1024 <= gmm1_hidden <= 6144 or gmm1_hidden % 1024 != 0:
        parser.error(
            "2 * --moe-intermediate-size must be in [1024, 6144] and divisible by 1024"
        )
    if (args.batch_size * args.num_processes * args.topk) % args.num_experts:
        parser.error(
            "--batch-size * --num-processes * --topk must be divisible by --num-experts"
        )
    if args.num_warmups < 0:
        parser.error("--num-warmups must be non-negative")
    if args.num_tests <= 0:
        parser.error("--num-tests must be positive")
    if args.quant_mode != 0:
        parser.error("--quant-mode currently only supports 0 to match the UMDK A5 test path")
    if args.master_addr is not None:
        os.environ["MASTER_ADDR"] = args.master_addr
    if args.master_port is not None:
        os.environ["MASTER_PORT"] = str(args.master_port)

    torch.multiprocessing.spawn(
        run_rank, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()


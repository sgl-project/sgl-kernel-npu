"""A5 static FusedDeepMoe smoke test through the public DeepEP Buffer API."""

import argparse
import os

import deep_ep
import torch
import torch.distributed as dist
import torch_npu

from utils import init_dist

torch_npu.npu.config.allow_internal_format = True


def debug_log(args: argparse.Namespace, rank: int, stage: str, **kwargs):
    if not getattr(args, "debug", False):
        return
    suffix = ""
    if kwargs:
        suffix = " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[rank {rank}] {stage}{suffix}", flush=True)


def tensor_format(tensor: torch.Tensor):
    try:
        return torch_npu.get_npu_format(tensor)
    except Exception:
        return "unavailable"


def tensor_meta(tensor: torch.Tensor):
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "format": tensor_format(tensor),
        "contiguous": bool(tensor.is_contiguous()),
    }


def make_umdk_static_inputs(rank: int, world_size: int, args: argparse.Namespace):
    """Build the static MXFP8 payload used by umdk/test_fp8_fused.py."""
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

    return (
        x,
        expert_ids,
        expert_scales,
        gmm1_weight,
        gmm1_scale,
        gmm2_weight,
        gmm2_scale,
    )


@torch.inference_mode()
def run_rank(local_rank: int, num_processes: int, args: argparse.Namespace):
    group = None
    try:
        rank, world_size, group = init_dist(local_rank, num_processes)
        torch.manual_seed(2026 + rank)
        buffer = deep_ep.Buffer(group, low_latency_mode=True)
        assert buffer.runtime.is_a5_build(), "The installed DeepEP wheel is not an A5 build"

        (
            x,
            expert_ids,
            expert_scales,
            gmm1_weight,
            gmm1_scale,
            gmm2_weight,
            gmm2_scale,
        ) = make_umdk_static_inputs(rank, world_size, args)
        debug_log(
            args,
            rank,
            "inputs.ready",
            x_shape=tuple(x.shape),
            gmm1_shape=tuple(gmm1_weight.shape),
            gmm2_shape=tuple(gmm2_weight.shape),
        )
        debug_log(
            args,
            rank,
            "inputs.meta",
            expected_count=args.batch_size * world_size * args.topk // args.num_experts,
            expert_ids_min=int(expert_ids.min().item()),
            expert_ids_max=int(expert_ids.max().item()),
            x_dtype=x.dtype,
            gmm1_weight_dtype=gmm1_weight.dtype,
            gmm1_weight_format=tensor_format(gmm1_weight),
            gmm1_scale_dtype=gmm1_scale.dtype,
            gmm1_scale_format=tensor_format(gmm1_scale),
            gmm2_weight_dtype=gmm2_weight.dtype,
            gmm2_weight_format=tensor_format(gmm2_weight),
            gmm2_scale_dtype=gmm2_scale.dtype,
            gmm2_scale_format=tensor_format(gmm2_scale),
        )

        dist.barrier()
        debug_log(args, rank, "warmup.begin")

        debug_log(
            args,
            rank,
            "warmup.fused_inputs",
            x_meta=tensor_meta(x),
            expert_ids_meta=tensor_meta(expert_ids),
            expert_scales_meta=tensor_meta(expert_scales),
            gmm1_weight_meta=tensor_meta(gmm1_weight),
            gmm1_scale_meta=tensor_meta(gmm1_scale),
            gmm2_weight_meta=tensor_meta(gmm2_weight),
            gmm2_scale_meta=tensor_meta(gmm2_scale),
        )
        debug_log(args, rank, "warmup.fused.begin")
        buffer.fused_deep_moe(
            x,
            expert_ids,
            expert_scales,
            gmm1_weight,
            gmm1_scale,
            gmm2_weight,
            gmm2_scale,
            args.batch_size,
            args.num_experts,
            args.quant_mode,
        )
        torch.npu.synchronize()
        debug_log(args, rank, "warmup.fused.end")
        dist.barrier()
        if rank == 0 and args.debug:
            print("Warmup completed.", flush=True)

        debug_log(args, rank, "verify.begin")
        debug_log(args, rank, "verify.fused.begin")
        output, ep_recv_count = buffer.fused_deep_moe(
            x,
            expert_ids,
            expert_scales,
            gmm1_weight,
            gmm1_scale,
            gmm2_weight,
            gmm2_scale,
            args.batch_size,
            args.num_experts,
            args.quant_mode,
        )
        torch.npu.synchronize()
        debug_log(
            args,
            rank,
            "verify.fused.end",
            output_shape=tuple(output.shape),
            ep_recv_count_shape=tuple(ep_recv_count.shape),
            output_format=tensor_format(output),
        )

        local_experts = args.num_experts // world_size
        expected_count = args.batch_size * world_size * args.topk // args.num_experts
        assert output.shape == (args.batch_size, args.hidden)
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output.float()).all().item()
        assert ep_recv_count.shape == (local_experts,)
        assert ep_recv_count.dtype == torch.int32
        torch.testing.assert_close(
            ep_recv_count,
            torch.full((local_experts,), expected_count, dtype=torch.int32, device="npu"),
        )
        debug_log(
            args,
            rank,
            "verify.done",
            ep_recv_count=ep_recv_count.cpu().tolist(),
        )

        if rank == 0:
            print(
                "A5 Buffer FusedDeepMoe smoke test passed: "
                f"output={tuple(output.shape)}, ep_recv_count={expected_count}",
                flush=True,
            )
        dist.barrier()
    finally:
        if group is not None and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="A5 Buffer FusedDeepMoe static smoke test")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--moe-intermediate-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--quant-mode", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
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
    if args.master_addr is not None:
        os.environ["MASTER_ADDR"] = args.master_addr
    if args.master_port is not None:
        os.environ["MASTER_PORT"] = str(args.master_port)

    torch.multiprocessing.spawn(
        run_rank, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()

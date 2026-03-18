import argparse
import os
import random
import sys
import time
from functools import partial

import torch
import torch.distributed as dist
import torch_npu
from deep_ep import Buffer
from utils import bench_kineto, calc_diff, hash_tensor, init_dist

torch_npu.npu.config.allow_internal_format = True

GMM_TILE_N_DIM = 64


# ======================== Weight Initialization ========================
def init_base_weights(
    num_local_experts,
    hidden_in=7168,
    moe_intermediate_size=4096,
):
    """
    Initialize the weights for each local expert.
    `num_local_experts`: Number of experts per rank = `num_experts` // `num_ranks`
    `hidden_in`: Input dimension (default 7168)
    `moe_intermediate_size`: Intermediate moe layer dimension (default 4096)
    """
    hidden_out = moe_intermediate_size // 2
    w13_weight = torch.randint(
        -16, 16, [num_local_experts, moe_intermediate_size, hidden_in], dtype=torch.int8
    )
    w2_weight = torch.randint(
        -16, 16, [num_local_experts, hidden_in, hidden_out], dtype=torch.int8
    )

    w13_weight_scale = (
        torch.rand([num_local_experts, moe_intermediate_size, 1]) * 0.0004 + 0.0015
    ).float()
    w2_weight_scale = (
        torch.rand([num_local_experts, hidden_in, 1]) * 0.0004 + 0.0015
    ).float()

    return w13_weight, w13_weight_scale, w2_weight, w2_weight_scale


def init_baseline_weights(w13_weight, w13_weight_scale, w2_weight, w2_weight_scale):
    w13_weight = w13_weight.data.transpose(1, 2).contiguous()
    w2_weight = w2_weight.data.transpose(1, 2).contiguous()
    w13_weight_scale = w13_weight_scale.data.squeeze(-1).contiguous()
    w2_weight_scale = w2_weight_scale.data.squeeze(-1).contiguous()
    # baseline store as nz
    w13_weight = torch_npu.npu_format_cast(w13_weight, 29)
    w2_weight = torch_npu.npu_format_cast(w2_weight, 29)

    return w13_weight, w13_weight_scale, w2_weight, w2_weight_scale


def scale_from_float_to_int64(scale):
    """Convert float32 scale to int64 representation."""
    import numpy as np

    scale = torch.from_numpy(
        np.frombuffer(
            scale.cpu().to(torch.float32).numpy().tobytes(), dtype=np.int32
        ).astype(np.int64)
    ).to(scale.device)
    return torch.nn.Parameter(scale, requires_grad=False)


def init_fused2_weights_int8(w13_weight, w13_weight_scale, w2_weight, w2_weight_scale):
    w13_weight = w13_weight.data.transpose(1, 2).contiguous()
    w2_weight = w2_weight.data.transpose(1, 2).contiguous()
    # baseline store as nz
    w13_weight = torch_npu.npu_format_cast(w13_weight, 29)
    w2_weight = torch_npu.npu_format_cast(w2_weight, 29)

    w13_int8_nz = torch.nn.Parameter(w13_weight, requires_grad=False)
    w2_int8_nz = torch.nn.Parameter(w2_weight, requires_grad=False)
    w13_scale_o = scale_from_float_to_int64(
        w13_weight_scale.data.squeeze(-1).contiguous()
    )
    w2_scale_o = scale_from_float_to_int64(
        w2_weight_scale.data.squeeze(-1).contiguous()
    )

    return w13_int8_nz, w13_scale_o, w2_int8_nz, w2_scale_o


# ======================== Utility Functions ========================
def make_uniform_topk_idx(
    num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, device="npu"
):
    assert num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks"
    experts_per_rank = num_experts // num_ranks

    topk_idx = torch.full((num_tokens, num_topk), -1, dtype=torch.int64, device=device)

    for t in range(num_tokens):
        for k in range(num_topk):
            rank_id = (t * num_topk + k) % num_ranks
            expert_base = rank_id * experts_per_rank
            expert_id = expert_base + ((t + k) % experts_per_rank)
            topk_idx[t, k] = expert_id
    return topk_idx


def from_inclusive_prefix_sum(pref):
    if isinstance(pref, torch.Tensor):
        if pref.numel() == 0:
            return pref
        return torch.cat([pref[:1], pref[1:] - pref[:-1]])

    if not pref:
        return []
    out = [pref[0]]
    for i in range(1, len(pref)):
        out.append(pref[i] - pref[i - 1])
    return out


# ======================== Baseline Reference ========================
def baseline_test(
    buffer,
    x,
    topk_idx,
    num_max_dispatch_tokens_per_rank,
    num_experts,
    cumulative_local_expert_recv_stats,
    return_recv_hook,
    w13,
    w13_scale,
    w2,
    w2_scale,
    topk_weights,
):
    hidden_states, packed_recv_count, handle, _, _ = buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        use_fp8=True,
        async_finish=not return_recv_hook,
        return_recv_hook=return_recv_hook,
    )
    output_dtype = torch.bfloat16
    group_list_type = 1

    per_token_scale = hidden_states[1]
    hidden_states = hidden_states[0]

    group_list = packed_recv_count.to(torch.int64)

    # gmm1: gate_up_proj
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.int32,
    )[0]

    # act_fn: swiglu
    hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
        x=hidden_states,
        weight_scale=w13_scale.to(torch.float32),
        activation_scale=per_token_scale,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=group_list,
        activate_left=True,
        quant_mode=1,
    )

    # gmm2: down_proj
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale.to(output_dtype)],
        per_token_scale=[swiglu_out_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=output_dtype,
    )[0]

    hidden_states, event, hook = buffer.low_latency_combine(
        hidden_states,
        topk_idx,
        topk_weights,
        handle,
        async_finish=not return_recv_hook,
        return_recv_hook=return_recv_hook,
    )
    # packed_recv_count(expertTokenNumsOut)
    return hidden_states, packed_recv_count


# ======================== Main Test ========================
def test(
    num_tokens: int,
    hidden: int,
    moe_intermediate_size: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: Buffer,
    buffer2: Buffer,
    args: argparse.Namespace,
    aligned_num_tokens: int,
    seed: int = 0,
):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert (
        num_ranks - rank_offset < 257
    ), "Too many ranks (exceeding test precision limit)"

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")

    # ----- Routing(topk_idx) -----
    if args.active_ranks:
        try:
            active_ranks = [
                int(r.strip()) for r in args.active_ranks.split(",") if r.strip()
            ]
        except ValueError:
            raise ValueError(
                f"Invalid value in --active-ranks: {args.active_ranks}. "
                f"Must be a comma-separated list of integers, e.g., '0,1,3'."
            )

        if any(r < 0 or r >= num_ranks for r in active_ranks):
            raise ValueError(
                f"Invalid rank in --active-ranks: {active_ranks}. "
                f"Ranks must be in range [0, {num_ranks - 1}]."
            )

        if not active_ranks:
            raise ValueError(
                "Parsed --active-ranks is empty. Provide at least one valid rank."
            )

        experts_per_rank = num_experts // num_ranks
        valid_experts = torch.cat(
            [
                torch.arange(
                    r * experts_per_rank, (r + 1) * experts_per_rank, device="npu"
                )
                for r in active_ranks
            ]
        )
        # Randomly sample experts from active ranks only
        topk_idx = valid_experts[
            torch.randint(0, len(valid_experts), (num_tokens, num_topk), device="npu")
        ]
        if rank == 0:
            print(
                f"[config] active_ranks={active_ranks}, valid_experts={len(valid_experts)}",
                flush=True,
            )
    else:
        scores = (
            torch.randn(
                (num_tokens, num_experts), dtype=torch.float32, device="npu"
            ).abs()
            + 1
        )
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]

    # ----- Weights -----
    w13_weight, w13_weight_scale, w2_weight, w2_weight_scale = init_base_weights(
        num_local_experts=num_local_experts,
        hidden_in=hidden,
        moe_intermediate_size=moe_intermediate_size,
    )
    # for gmm1、swiglu、gmm2
    w13, w13_scale, w2, w2_scale = init_baseline_weights(
        w13_weight.clone().detach(),
        w13_weight_scale.clone().detach(),
        w2_weight.clone().detach(),
        w2_weight_scale.clone().detach(),
    )
    # for dispatch_ffn_combine
    w13_f2, w13s_f2, w2_f2, w2s_f2 = init_fused2_weights_int8(
        w13_weight.clone().detach(),
        w13_weight_scale.clone().detach(),
        w2_weight.clone().detach(),
        w2_weight_scale.clone().detach(),
    )

    if args.debug and rank == 0:
        print("=== Check base weights ===")
        print(
            "w13:", w13.shape, w13.dtype, w13.device, torch_npu.get_npu_format(w13)
        )  # FRACTAL_NZ
        print("w13_scale:", w13_scale.shape, w13_scale.dtype, w13_scale.device)
        print(
            "w2:", w2.shape, w2.dtype, w2.device, torch_npu.get_npu_format(w2)
        )  # FRACTAL_NZ
        print("w2_scale:", w2_scale.shape, w2_scale.dtype, w2_scale.device)
        print("=== Check fused2 weights ===")
        print(
            "w13_f2:",
            w13_f2.shape,
            w13_f2.dtype,
            w13_f2.device,
            torch_npu.get_npu_format(w13_f2),
        )
        print("w13s_f2:", w13s_f2.shape, w13s_f2.dtype, w13s_f2.device)
        print(
            "w2_f2:",
            w2_f2.shape,
            w2_f2.dtype,
            w2_f2.device,
            torch_npu.get_npu_format(w2_f2),
        )
        print("w2s_f2:", w2s_f2.shape, w2s_f2.dtype, w2s_f2.device)

    # ----- Tokens per rank -----
    tokens_per_rank = torch.zeros(num_ranks, dtype=torch.int64, device="npu")
    experts_per_rank = num_experts // num_ranks
    for r in range(num_ranks):
        start, end = r * experts_per_rank, (r + 1) * experts_per_rank
        tokens_per_rank[r] = ((topk_idx >= start) & (topk_idx < end)).sum()

    if args.debug:
        print(f"[DEBUG] Tokens per rank: {tokens_per_rank}", flush=True)

    # ====== ensure topk_weights is defined (fix missing var) ======
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    ).abs()

    # ====== cumulative stats and flags ======
    cumulative_local_expert_recv_stats = torch.zeros(
        (num_local_experts,), dtype=torch.int32, device="npu"
    )
    return_recv_hook = False

    # ----- Random or fixed drop: Currently, topk=-1 is not supported.
    topk_idx_dropped = topk_idx
    topk_weights_dropped = topk_weights

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="npu")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx_dropped == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    if args.debug:
        print(f"[Rank {rank}] num_tokens_per_expert: {num_tokens_per_expert.tolist()}")
        if rank == 0:
            print(
                f"[Rank {rank}] gbl_num_tokens_per_expert: {gbl_num_tokens_per_expert.tolist()}"
            )

    # ----- Baseline -----
    baseline_output, base_ep_recv_count = baseline_test(
        buffer2,
        x,
        topk_idx,
        aligned_num_tokens,  # num_max_dispatch_tokens_per_rank
        num_experts,
        cumulative_local_expert_recv_stats,
        return_recv_hook,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights_dropped,
    )

    # ----- Fused2: dispatch_ffn_combine -----
    fused2_output, fused2_ep_recv_count = buffer.fused_deep_moe(
        x,
        topk_idx_dropped,
        topk_weights_dropped,
        w13_f2,
        w13s_f2,
        w2_f2,
        w2s_f2,
        (
            aligned_num_tokens * topk_idx_dropped.size(1) * num_ranks
        ),  # max_output_size can be set to a larger value.
        num_experts,
        1,  # quant_mode: 1
        2,  # fuse_mode: DISPATCH_FFN_COMBINE
    )

    # ----- Compare Outputs -----
    baseline_output_avg = torch.mean(torch.abs(baseline_output)).item()
    max_diff2 = torch.max(torch.abs(fused2_output - baseline_output)).item()
    avg_diff2 = torch.mean(torch.abs(fused2_output - baseline_output)).item()
    fused_output_avg2 = torch.mean(torch.abs(fused2_output)).item()
    diff2 = calc_diff(fused2_output, baseline_output)
    print(
        f"[Precision Compare] {rank=}, baseline_avg={baseline_output_avg:.6e}, diff2={diff2:.6e} "
        f"fused2_avg={fused_output_avg2:.6e}, max_diff2={max_diff2:.6e}, avg_diff2={avg_diff2:.6e}",
        flush=True,
    )
    # The difference between the results is closely related to the input x value range.
    assert avg_diff2 < 1e-2, f"[fused2] {rank=} Mismatch detected! diff={avg_diff2}"

    # ----- Compare Recv Count -----
    experts_per_rank = num_experts // dist.get_world_size()
    start_expert = rank * experts_per_rank
    end_expert = start_expert + experts_per_rank

    expected_recv = gbl_num_tokens_per_expert[start_expert:end_expert]
    base_recv = base_ep_recv_count.to(torch.int32)
    fuse2_recv = fused2_ep_recv_count.to(torch.int32)
    if args.debug:
        print(f"[Rank {rank}] expected_recv: {expected_recv}")
        print(f"[Rank {rank}] base_recv: {base_recv}")
        print(f"[Rank {rank}] fuse2_recv: {fuse2_recv}")

    assert torch.allclose(
        expected_recv, base_recv
    ), f"Assertion base recv_count failed on rank {rank}: Expected {expected_recv}, Actual {base_recv}"
    assert torch.allclose(
        expected_recv, fuse2_recv
    ), f"Assertion fuse2 recv_count failed on rank {rank}: Expected {expected_recv}, Actual {fuse2_recv}"

    # ----- performance test -----
    dist.barrier()
    baseline_args = {
        "buffer": buffer2,
        "x": x,
        "topk_idx": topk_idx,
        "num_max_dispatch_tokens_per_rank": aligned_num_tokens,
        "num_experts": num_experts,
        "cumulative_local_expert_recv_stats": cumulative_local_expert_recv_stats,
        "return_recv_hook": return_recv_hook,
        "w13": w13,
        "w13_scale": w13_scale,
        "w2": w2,
        "w2_scale": w2_scale,
        "topk_weights": topk_weights_dropped,
    }

    fused2_moe_args = {
        "x": x,
        "topk_idx": topk_idx_dropped,
        "topk_weights": topk_weights,
        "gmm1_permuted_weight": w13_f2,
        "gmm1_permuted_weight_scale": w13s_f2,
        "gmm2_weight": w2_f2,
        "gmm2_weight_scale": w2s_f2,
        "num_max_dispatch_tokens_per_rank": (
            aligned_num_tokens * topk_idx_dropped.size(1) * num_ranks
        ),
        "num_experts": num_experts,
        "quant_mode": 0,
        "fuse_mode": 2,
    }

    baseline_time = bench_kineto(
        lambda: baseline_test(**baseline_args),
        (
            "aclnnInplaceOne_OnesLikeAiCore_OnesLike",
            "MoeDistributeDispatchV2",
            "aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul",
            "DequantSwigluQuant",
            "MoeDistributeCombineV2",
        ),
        barrier_comm_profiling=True,
    )
    dist.barrier()

    fused2_moe_time = bench_kineto(
        lambda: buffer.fused_deep_moe(**fused2_moe_args),
        "DispatchFFNCombine",
        barrier_comm_profiling=True,
    )

    # aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul was calculated twice
    baseline_time_ = sum(baseline_time) + baseline_time[2]
    print(
        f"[Rank {rank}] baseline_time= {baseline_time_ * 1e6:.2f} us, fused2_moe_time= {fused2_moe_time * 1e6:.2f} us",
        flush=True,
    )


# ======================== Distributed Entry ========================
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    group2 = dist.new_group(list(range(num_ranks)))

    shared_expert_rank_num = 0
    num_tokens, hidden, moe_intermediate_size = (
        args.num_tokens,
        args.hidden,
        args.moe_intermediate_size,
    )
    num_topk, num_experts = args.num_topk, args.num_experts
    use_experts = num_experts if shared_expert_rank_num == 0 else (num_experts - 1)
    use_ranks = num_ranks - shared_expert_rank_num
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )
    buffer = Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=use_experts // use_ranks if use_ranks > 0 else 1,
    )
    buffer2 = Buffer(
        group2,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=use_experts // use_ranks if use_ranks > 0 else 1,
    )

    local_tokens_tensor = torch.tensor([num_tokens], dtype=torch.int32, device="npu")
    dist.all_reduce(local_tokens_tensor, op=dist.ReduceOp.MAX)
    aligned_num_tokens = local_tokens_tensor.item()

    test(
        num_tokens,
        hidden,
        moe_intermediate_size,
        use_experts,
        num_topk,
        rank,
        use_ranks,
        group,
        buffer,
        buffer2,
        args,
        aligned_num_tokens,
        seed=1,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dispatch_ffn_combine kernels")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of processes to spawn (default: 16)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=256, help="Number of tokens (default: 256)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=4096,
        help="Moe intermediate size (default: 4096)",
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--use-fp8", type=int, default=1, help="Number of use-fp8 (default: 1)"
    )
    parser.add_argument(
        "--active-ranks",
        type=str,
        default="",
        help='Comma-separated list of ranks that will receive tokens. Example: "0,1,3". If empty, all ranks may receive tokens.',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )

    args = parser.parse_args()
    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )

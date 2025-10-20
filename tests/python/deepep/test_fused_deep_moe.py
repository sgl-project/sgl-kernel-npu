import argparse
import os
import random
import time
from functools import partial

import torch
import torch.distributed as dist
import torch_npu
from deep_ep import Buffer
from utils import bench, calc_diff, hash_tensor, init_dist

torch_npu.npu.config.allow_internal_format = True
test_topk_minus1 = True


# ======================== Weight Initialization ========================
def init_base_weights():
    w13_weight = torch.randint(-16, 16, [16, 4096, 7168]).to(torch.int8)
    w2_weight = torch.randint(-16, 16, [16, 7168, 2048]).to(torch.int8)
    w13_weight_scale = (torch.rand([16, 4096, 1]) * 0.0004 + 0.0015).bfloat16()
    w2_weight_scale = (torch.rand([16, 7168, 1]) * 0.0004 + 0.0015).bfloat16()

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


# ======================== Weight Permutation & Fusion ========================
def permute_weight(w: torch.Tensor, tile_n):

    *dims, n = w.shape
    order = list(range(len(dims))) + [-2, -3, -1]
    return (
        w.reshape(*dims, 2, n // tile_n, tile_n // 2)
        .permute(order)
        .reshape(*dims, n)
        .contiguous()
    )


def reshape_fusion_gmm_weight(weight, dim):
    original_shape = weight.shape
    if dim < 0:
        dim += len(original_shape)

    weight = weight.view(*original_shape[:dim], 2, 32, 64, *original_shape[dim + 1 :])
    weight = weight.transpose(dim, dim + 1).contiguous()
    weight = weight.view(*original_shape[:dim], -1, *original_shape[dim + 1 :])

    return weight.contiguous()


def init_fused_weights_int8(
    w13_weight,
    w13_weight_scale,
    w2_weight,
    w2_weight_scale,
    device="npu",
    block_m: int = 16,
    block_n: int = 16,
):

    # -------- w13_weight --------
    w13 = w13_weight.transpose(1, 2).contiguous()
    torch_npu.npu_format_cast_(w13, 2)
    cpu_w13 = w13.cpu()
    w13 = reshape_fusion_gmm_weight(cpu_w13, -1).npu()
    torch_npu.npu_format_cast_(w13, 29)
    w13_int8_nz = torch.nn.Parameter(w13, requires_grad=False)

    # -------- w2_weight --------
    w2 = torch_npu.npu_format_cast(w2_weight, 29)
    w2_int8_nz = torch.nn.Parameter(w2, requires_grad=False)

    # -------- w13_weight_scale --------
    w13_scale = permute_weight(w13_weight_scale.squeeze(-1).contiguous(), 128)
    w13_scale_o = torch.nn.Parameter(w13_scale, requires_grad=False)

    # -------- w2_weight_scale --------
    w2_scale = w2_weight_scale.squeeze(-1).contiguous()
    w2_scale_o = torch.nn.Parameter(w2_scale, requires_grad=False)

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
    num_tokens,
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
        num_tokens,
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
        scale=[w2_scale],
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
    # handle[1] -> ep_recv_count
    return hidden_states, handle[1]


# ======================== Main Test ========================
def test(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: Buffer,
    buffer2: Buffer,
    args: argparse.Namespace,
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

    x = torch.rand((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * 10 - 5
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
    w13_weight, w13_weight_scale, w2_weight, w2_weight_scale = init_base_weights()
    w13, w13_scale, w2, w2_scale = init_baseline_weights(
        w13_weight.clone().detach(),
        w13_weight_scale.clone().detach(),
        w2_weight.clone().detach(),
        w2_weight_scale.clone().detach(),
    )
    w13_f, w13s_f, w2_f, w2s_f = init_fused_weights_int8(
        w13_weight.clone().detach(),
        w13_weight_scale.clone().detach(),
        w2_weight.clone().detach(),
        w2_weight_scale.clone().detach(),
    )

    if rank == 0:
        print("=== Check fused weights ===")
        print("w13_f:", w13_f.shape, w13_f.dtype, w13_f.device)
        print("w13s_f:", w13s_f.shape, w13s_f.dtype, w13s_f.device)
        print("w2_f:", w2_f.shape, w2_f.dtype, w2_f.device)
        print("w2s_f:", w2s_f.shape, w2s_f.dtype, w2s_f.device)

    # ----- Tokens per rank -----
    tokens_per_rank = torch.zeros(num_ranks, dtype=torch.int64, device="npu")
    experts_per_rank = num_experts // num_ranks
    for r in range(num_ranks):
        start, end = r * experts_per_rank, (r + 1) * experts_per_rank
        tokens_per_rank[r] = ((topk_idx >= start) & (topk_idx < end)).sum()
    print(f"Tokens per rank: {tokens_per_rank}")

    # ----- Random drop -----
    if args.drop_prob > 0:
        drop_mask = torch.rand_like(topk_idx, dtype=torch.float32) < args.drop_prob
        topk_idx = topk_idx.masked_fill(drop_mask, -1)
        for i in range(num_tokens):
            if (topk_idx[i] == -1).all():
                topk_idx[i, 0] = torch.topk(scores[i], 1, largest=True)[1].item()

    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    ).abs()
    cumulative_local_expert_recv_stats = torch.zeros(
        (num_local_experts,), dtype=torch.int, device="npu"
    )
    return_recv_hook = False

    # ----- Baseline -----
    hidden_states = x

    if test_topk_minus1:
        topk_weights_minus1 = topk_weights
        topk_weights_minus1[:, -2:-1] = 0
        topk_idx_minus1 = topk_idx
        topk_idx_minus1[:, -2:-1] = -1
        baseline_output = baseline_test(
            buffer2,
            x,
            topk_idx,
            num_tokens,
            num_experts,
            cumulative_local_expert_recv_stats,
            return_recv_hook,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights_minus1,
        )
        hidden_states, event, hook = buffer.fused_deep_moe(
            x,
            topk_idx_minus1,
            topk_weights,
            w13_f,
            w13s_f,
            w2_f,
            w2s_f,
            num_tokens,
            num_experts,
            0,
        )

    else:
        baseline_output = baseline_test(
            buffer2,
            x,
            topk_idx,
            num_tokens,
            num_experts,
            cumulative_local_expert_recv_stats,
            return_recv_hook,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights,
        )

        hidden_states, event, hook = buffer.fused_deep_moe(
            x,
            topk_idx,
            topk_weights,
            w13_f,
            w13s_f,
            w2_f,
            w2s_f,
            num_tokens,
            num_experts,
            0,
        )
    # fused_output = hidden_states.to(dtype=torch.bfloat16)
    fused_output = hidden_states
    print(f"[Rank {rank}] fused finish.")
    baseline_output, base_ep_recv_count = baseline_test(
        buffer2,
        x,
        topk_idx,
        num_tokens,
        num_experts,
        cumulative_local_expert_recv_stats,
        return_recv_hook,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
    )

    # ----- Fused -----
    fused_output, fused_ep_recv_count = buffer.fused_deep_moe(
        x,
        topk_idx,
        topk_weights,
        w13_f,
        w13s_f,
        w2_f,
        w2s_f,
        num_tokens,
        num_experts,
        0,
    )

    # ----- Compare Outputs -----
    max_diff = torch.max(torch.abs(fused_output - baseline_output)).item()
    avg_diff = torch.mean(torch.abs(fused_output - baseline_output)).item()
    baseline_output_avg = torch.mean(torch.abs(baseline_output)).item()
    fused_output_avg = torch.mean(torch.abs(fused_output)).item()

    print(
        f"[Rank {rank}] baseline_avg={baseline_output_avg:.6e}, fused_avg={fused_output_avg:.6e}, "
        f"max_diff={max_diff:.6e}, avg_diff={avg_diff:.6e}"
    )
    assert avg_diff < 1e-4, f"[Rank {rank}] Mismatch detected! diff={avg_diff}"

    # ----- Compare RecvCount -----
    recv_count_diff = (
        from_inclusive_prefix_sum(base_ep_recv_count) - fused_ep_recv_count
    ).abs()
    max_recv_count_diff = recv_count_diff.max().item()
    mean_recv_count_diff = recv_count_diff.mean().item()
    print(
        f"[Rank {rank}] Difference between base and fused recv_count -> max: {max_recv_count_diff}, mean: {mean_recv_count_diff}"
    )
    assert (
        max_recv_count_diff < 1e-4
    ), f"[Rank {rank}] Mismatch detected! diff={max_recv_count_diff}"


# ======================== Distributed Entry ========================
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    group2 = dist.new_group(list(range(16)))
    shared_expert_rank_num = int(os.getenv("MOE_SHARED_EXPERT_RANK_NUM", 0))
    num_tokens, hidden = args.num_tokens, args.hidden
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

    test(
        num_tokens,
        hidden,
        use_experts,
        num_topk,
        rank,
        use_ranks,
        group,
        buffer,
        buffer2,
        args,
        seed=1,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test intranode EP kernels")
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
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--active-ranks",
        type=str,
        default="",
        help="Comma-separated list of ranks that will receive tokens. "
        'Example: "0,1,3". If empty, all ranks may receive tokens.',
    )
    parser.add_argument(
        "--drop-prob",
        type=float,
        default=0.0,
        help="Probability of dropping an individual top-k index (set to -1). "
        "Guaranteed that each token keeps at least one valid expert.",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
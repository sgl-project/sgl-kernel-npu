import argparse
import os

import deep_ep
import torch
import torch.distributed as dist
import torch_npu
from utils import calc_diff, init_dist, inplace_unique, per_token_cast_back


def intranode_test(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    local_rank: int,
    num_local_ranks: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    experts_per_rank = num_experts // num_ranks
    assert num_experts % num_ranks == 0

    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]

    rank_idx = topk_idx // experts_per_rank
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="npu")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="npu")
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device="npu"
    )
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(
            count, dtype=torch.long, device="npu"
        )
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = (token_idx_in_rank >= 0).to(torch.int)
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    try:
        return_values = buffer.get_dispatch_layout(topk_idx, num_experts)
        (
            ref_num_tokens_per_rank,
            _,
            ref_num_tokens_per_expert,
            ref_is_token_in_rank,
            _,
        ) = return_values
        try:
            assert torch.allclose(
                ref_num_tokens_per_rank, num_tokens_per_rank
            ), f"Assertion num_tokens_per_rank failed on rank {rank}: Expected {num_tokens_per_rank}, Actual {ref_num_tokens_per_rank}"
            assert torch.allclose(
                ref_num_tokens_per_expert, num_tokens_per_expert
            ), f"Assertion num_tokens_per_expert failed on rank {rank}: Expected {num_tokens_per_expert}, Actual {ref_num_tokens_per_expert}"
            assert torch.allclose(
                ref_is_token_in_rank, is_token_in_rank
            ), f"Assertion is_token_in_rank failed on rank {rank}: Expected {is_token_in_rank}, Actual {ref_is_token_in_rank}"
        except AssertionError as e:
            print(e)
            raise
    except Exception as e:
        print(f"An error occurred: {e}")

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    )

    buffer_size = 256
    config = deep_ep.Config(24, 8, buffer_size)

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": config,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
    }

    (
        recv_x,
        _,
        _,
        _,
        handle,
        _,
    ) = buffer.dispatch(**dispatch_args)

    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
    combine_args = {
        "x": recv_x,
        "handle": handle,
        "config": config,
        "async_finish": False,
        "topk_weights": handle[7],
    }
    (
        combined_x,
        _,
        _,
    ) = buffer.combine(**combine_args)

    assert (
        calc_diff(
            combined_x.float(),
            x * handle[7].masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1),
        )
        < 5e-5
    )


def low_latency_test(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    assert num_experts % num_ranks == 0
    experts_per_rank = num_experts // num_ranks

    rank_offset = 128
    assert (
        num_ranks - rank_offset < 257
    ), "Too many ranks (exceeding test precision limit)"

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * (
        rank - rank_offset
    )
    x[:, -128:] = torch.arange(num_tokens, device="npu").to(torch.bfloat16).view(-1, 1)
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    ).abs()

    return_recv_hook = False
    cumulative_local_expert_recv_stats = torch.zeros(
        (experts_per_rank,), dtype=torch.int, device="npu"
    )
    dispatch_use_fp8 = True
    packed_recv_x, packed_recv_count, handle2, event, hook = (
        buffer.low_latency_dispatch(
            x,
            topk_idx,
            num_tokens,
            num_experts,
            use_fp8=dispatch_use_fp8,
            round_scale=False,
            use_ue8m0=False,
            topk_weights=topk_weights,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            async_finish=not return_recv_hook,
            return_recv_hook=return_recv_hook,
        )
    )
    simulated_gemm_x = (
        per_token_cast_back(*packed_recv_x) if dispatch_use_fp8 else packed_recv_x
    )

    all_topk_idx = torch.empty(
        (num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device="npu"
    )
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

    (
        src_info,
        layout_range,
        num_max_dispatch_tokens_per_rank,
        hidden,
        num_experts,
        packed_recv_count,
        expand_scales,
    ) = handle2

    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    combined_x, event, hook = buffer.low_latency_combine(
        simulated_gemm_x,
        topk_idx,
        topk_weights,
        handle2,
        async_finish=not return_recv_hook,
        zero_copy=False,
        return_recv_hook=return_recv_hook,
        out=out,
    )

    diff = calc_diff(
        x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1),
        combined_x,
    )
    assert torch.isnan(combined_x).sum().item() == 0
    if dispatch_use_fp8:
        assert diff < 1e-4, f"Error: {diff=}"
    else:
        assert diff < 1e-5, f"Error: {diff=}"


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_topk, num_experts, hidden = args.num_topk, args.num_experts, args.hidden

    num_tokens_i = args.num_tokens_i
    buffer_i = deep_ep.Buffer(
        group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1
    )

    shared_expert_rank_num = int(os.getenv("MOE_SHARED_EXPERT_RANK_NUM", 0))
    num_tokens_l = args.num_tokens_l
    use_experts = num_experts if shared_expert_rank_num == 0 else (num_experts - 1)
    use_ranks = num_ranks - shared_expert_rank_num
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        num_tokens_l, hidden, num_ranks, num_experts
    )
    buffer_l = deep_ep.Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=use_experts // use_ranks if use_ranks > 0 else 1,
    )

    torch.manual_seed(rank)

    print("Start executing intranode test...", flush=True)
    intranode_test(
        num_tokens_i,
        hidden,
        num_experts,
        num_topk,
        rank,
        num_ranks,
        local_rank,
        num_local_ranks,
        buffer_i,
        group,
    )

    dist.barrier()

    print("Start executing low latency test...", flush=True)
    low_latency_test(
        num_tokens_l,
        hidden,
        num_experts,
        num_topk,
        rank,
        num_ranks,
        buffer_l,
        group,
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
        "--num-tokens-i",
        type=int,
        default=256,
        help="Number of intranode tokens (default: 4096)",
    )
    parser.add_argument(
        "--num-tokens-l",
        type=int,
        default=256,
        help="Number of low latency tokens (default: 256)",
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

    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )

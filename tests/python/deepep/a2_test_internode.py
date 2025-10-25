import argparse
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, inplace_unique, bench, per_token_cast_back, calc_diff

# noinspection PyShadowingNames
def test_main(args: argparse.Namespace, local_rank: int, num_ranks: int, rank: int,
              buffer: deep_ep.Buffer, group: dist.ProcessGroup):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}', flush=True)

    # Random data
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='npu').abs() + 1
    # topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = torch.arange(8, device='npu').repeat(num_tokens, 1)

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='npu')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='npu')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='npu')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='npu')
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    try:
        try:
            return_values = buffer.get_dispatch_layout(topk_idx, num_experts)
        except Exception as e:
            print(f"Error occurred while calling get_dispatch_layout: {e}")
            raise

        ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = return_values
        # try:
        #     assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank), \
        #         f"Assertion num_tokens_per_rank failed on rank {rank}: Expected {num_tokens_per_rank}, Actual {ref_num_tokens_per_rank}"
        #     assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert), \
        #         f"Assertion num_tokens_per_expert failed on rank {rank}: Expected {num_tokens_per_expert}, Actual {ref_num_tokens_per_expert}"
        #     assert torch.allclose(ref_is_token_in_rank, is_token_in_rank), \
        #         f"Assertion is_token_in_rank failed on rank {rank}: Expected {is_token_in_rank}, Actual {ref_is_token_in_rank}"
        # except AssertionError as e:
        #     print(e)
        #     raise
    except Exception as e:
        print(f"An error occurred: {e}")

    # t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    # print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
    print('', flush=True)
    # 同步
    # torch.npu.synchronize()
    dist.barrier()
    time.sleep(1)

    # Config
    buffer_size = 256
    config = deep_ep.Config(24, 8, buffer_size)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='npu') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='npu')
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='npu')
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='npu')

    torch.set_printoptions(threshold=float('inf'))
    # print(f'{rank=}, {topk_idx=}\n')
    # print(f'{rank=}, {topk_weights=}\n')
    print('', flush=True)

    if local_rank == 0:
        print(f'[testing] Running with {"FP8" if isinstance(x, tuple) else "BF16"}, with top-k ...', flush=True)
    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": config,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
    }

    expandx_out, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)
    torch.npu.synchronize()
    dist.barrier()

    (
        src_idx,
        is_recv_token_in_rank,
        ep_rank_token_cnt,       # send_head
        topk_idx,
        topk_weights_ori,
        offset_inner,
        token_server_idx,        # offset_outer
        count_outer,
        expand_scales,
    ) = handle

    if local_rank == 0:
        filename = f"notify_data_{rank}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f'{rank=}, {topk_idx=}\n')
            f.write(f'{rank=}, {num_tokens_per_expert=}\n')
            f.write(f'{rank=}, {token_server_idx=}\n')
            f.write(f'{rank=}, {count_outer=}\n')
            f.write(f'{rank=}, {ep_rank_token_cnt=}\n')
            f.write(f'{rank=}, {offset_inner=}\n')
            # f.write(f'{rank=}, {expand_scales=}\n')

    print(f'{rank=}, {expand_scales=}\n')
    print(f'{rank=}, expandx_out: {expandx_out.shape}, {expandx_out[:,0]}\n')

    return

    # Test combine
    expert_idx = torch.zeros((num_tokens, num_experts), dtype=torch.int, device='npu')

    combine_args = {'x': expandx_out, 'handle': handle, 'config': None, 'async_finish': False, 'topk_weights': topk_weights}
    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
    torch.npu.synchronize()
    dist.barrier()
    filename = f"output_data_{rank}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f'{rank=}, {topk_idx=}\n')
        f.write(f'{rank=}, {expand_scales=}\n')
        f.write(f'{rank=}, expandx_out: {expandx_out.shape}, {expandx_out[:,0]}\n')
        f.write(f'{rank=}, combined_x: {combined_x.shape}, {combined_x[:,0]}\n')
    print('combined_x', rank)
    print('', flush=True)
    time.sleep(1)

    check_x = combined_x.float()

    filename = f"output_data_{rank}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f'{rank=}, {topk_idx=}\n')
        f.write(f'{rank=}, {expand_scales=}\n')
        f.write(f'{rank=}, expandx_out: {expandx_out.shape}, {expandx_out[:,0]}\n')
        f.write(f'{rank=}, combined_x: {combined_x.shape}, {combined_x[:,0]}\n')

    # assert (calc_diff(
    #     check_x,
    #     x * num_topk
    # ) < 5e-5)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    print(f"[Rank {rank} | Local rank {local_rank}] Initializing buffer...", flush=True)
    buffer = deep_ep.Buffer(group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1)
    print(f"[Rank {rank}] Buffer created OK.", flush=True)
    torch.manual_seed(rank)

    test_main(args, local_rank, num_ranks, rank, buffer, group)
    if local_rank == 0:
        print('', flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=16,
                       help='Number of tokens (default: 4096)')
    parser.add_argument('--hidden', type=int, default=7168,
                       help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8,
                       help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=16,
                       help='Number of experts (default: 256)')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)

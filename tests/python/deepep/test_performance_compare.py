import argparse
import os
import time
from typing import List, Optional, Tuple

import deep_ep
import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from utils import bench, calc_diff, init_dist


def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None):
    if output_split_sizes is None:
        a2a_out = torch.empty_like(input_)
    else:
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.npu.current_device(),
        )

    if event:
        global COMM_STREAM
        if "COMM_STREAM" not in globals() or COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
    else:
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
    return input_, a2a_out, handle


def _gather_along_first_dim(input_, group, output_split_sizes=None):
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size
        output = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.npu.current_device()
        )
        torch.distributed.all_gather_into_tensor(
            output, input_.contiguous(), group=group
        )
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.npu.current_device()
        )
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def gather_from_sequence_parallel_region(input_, group, output_split_sizes=None):
    return _gather_along_first_dim(input_, group, output_split_sizes)


class HCCLDispatcher:
    def __init__(self, ep_group, num_experts, num_local_experts):
        self.ep_group = ep_group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)

        local_expert_indices_offset = self.ep_rank * self.num_local_experts
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]

        self.expert_ids_per_ep_rank = torch.tensor(
            [i % self.num_local_experts for i in range(self.num_experts)],
            dtype=torch.int32,
            device="npu",
        )

    def dispatch(self, hidden_states, topk_ids, topk_weights):
        self.hidden_shape = hidden_states.shape
        self.topk_weights = topk_weights
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # 1. Preprocess: Count tokens per expert
        num_local_tokens_per_expert = torch.histc(
            topk_ids.float(), bins=self.num_experts, min=0, max=self.num_experts
        )

        # Calculate splits
        self.input_splits = (
            num_local_tokens_per_expert.reshape(self.ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.int64)
            .cpu()
            .numpy()
            .tolist()
        )

        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert, self.ep_group
        ).reshape(self.ep_size, self.num_experts)
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ]

        self.output_splits = (
            self.num_global_tokens_per_local_expert.sum(axis=-1)
            .to(torch.int64)
            .cpu()
            .numpy()
            .tolist()
        )

        # 2. Permute tokens locally
        permutated_tokens, self.reversed_local_mapping = (
            torch_npu.npu_moe_token_permute(
                hidden_states, topk_ids.to(torch.int32), num_out_tokens=topk_ids.numel()
            )
        )

        # 3. AllToAllV
        _, global_input_tokens, handle = async_all_to_all(
            permutated_tokens, self.output_splits, self.input_splits, self.ep_group
        )
        handle.wait()

        # 4. Post-process (Re-permute for local experts)
        self.global_tokens_indices = torch.repeat_interleave(
            self.expert_ids_per_ep_rank,
            self.num_global_tokens_per_local_expert.ravel().to(torch.int32),
        )

        dispatch_out, self.reversed_global_mapping = torch_npu.npu_moe_token_permute(
            global_input_tokens, self.global_tokens_indices
        )
        return dispatch_out

    def combine(self, hidden_states):
        # 1. Unpermute locally
        hidden_states = torch_npu.npu_moe_token_unpermute(
            hidden_states, self.reversed_global_mapping
        )

        # 2. AllToAllV back
        _, local_tokens, handle = async_all_to_all(
            hidden_states, self.input_splits, self.output_splits, self.ep_group
        )
        handle.wait()

        # 3. Final unpermute and weighted sum
        output = torch_npu.npu_moe_token_unpermute(
            local_tokens,
            self.reversed_local_mapping.to(torch.int32),
            probs=self.topk_weights,
            restore_shape=self.hidden_shape,
        )
        return output


def test_compare(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank)

    x = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device="npu")

    scores = torch.randn((args.num_tokens, args.num_experts), device="npu")
    topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1).to(torch.float32)

    num_local_experts = args.num_experts // world_size

    if rank == 0:
        print(f"[{rank}] Initializing DeepEP...", flush=True)

    dep_conf = deep_ep.Config(24, 8, int(2e9))
    dep_buffer = deep_ep.Buffer(group, int(2e9), 0)

    def deepep_dispatch_func():

        layout = dep_buffer.get_dispatch_layout(topk_idx, args.num_experts)

        dep_buffer.dispatch(
            x,
            num_tokens_per_rank=layout[0],
            is_token_in_rank=layout[3],
            num_tokens_per_expert=layout[2],
            config=dep_conf,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )

    layout_cache = dep_buffer.get_dispatch_layout(topk_idx, args.num_experts)
    x_expert_de, _, _, _, de_handle, _ = dep_buffer.dispatch(
        x,
        num_tokens_per_rank=layout_cache[0],
        is_token_in_rank=layout_cache[3],
        num_tokens_per_expert=layout_cache[2],
        config=dep_conf,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )

    if isinstance(x_expert_de, tuple):
        x_expert_de = x_expert_de[0]

    def deepep_combine_func():
        dep_buffer.combine(x_expert_de, de_handle, config=dep_conf)

    if rank == 0:
        print(f"[{rank}] Initializing HCCL Dispatcher...", flush=True)

    hccl_dispatcher = HCCLDispatcher(group, args.num_experts, num_local_experts)

    def hccl_dispatch_func():
        return hccl_dispatcher.dispatch(x, topk_idx, topk_weights)

    x_expert_hccl = hccl_dispatcher.dispatch(x, topk_idx, topk_weights)

    def hccl_combine_func():
        return hccl_dispatcher.combine(x_expert_hccl)

    # ==========================================
    # Benchmarking
    # ==========================================
    dist.barrier()
    if rank == 0:
        print(">>> Start Benchmarking...", flush=True)

    t_de_disp_avg, _, _ = bench(deepep_dispatch_func, num_warmups=10, num_tests=20)
    t_hccl_disp_avg, _, _ = bench(hccl_dispatch_func, num_warmups=10, num_tests=20)

    t_de_comb_avg, _, _ = bench(deepep_combine_func, num_warmups=10, num_tests=20)
    t_hccl_comb_avg, _, _ = bench(hccl_combine_func, num_warmups=10, num_tests=20)

    # ==========================================
    # Correctness Check
    # ==========================================
    out_de, _, _ = dep_buffer.combine(x_expert_de, de_handle, config=dep_conf)
    out_hccl = hccl_dispatcher.combine(x_expert_hccl)

    diff_val = calc_diff(out_de, out_hccl)

    # ==========================================
    # Report
    # ==========================================
    if rank == 0:
        print("\n" + "=" * 90)
        print(f"BENCHMARK REPORT (World Size: {world_size})")
        print(
            f"Params: Tokens={args.num_tokens}, Hidden={args.hidden}, TopK={args.num_topk}, Experts={args.num_experts}"
        )
        print("Note: Dispatch times INCLUDE layout/split calculation overhead.")
        print("-" * 90)

        def to_ms(t_s):
            return t_s * 1000.0

        header = (
            f"{'Operation':<12} | "
            f"{'DeepEP (ms)':<12} | "
            f"{'HCCL (ms)':<12} | "
            f"{'Speedup':<10} | "
            f"{'Saved (ms)':<12} | "
            f"{'Reduction':<10}"
        )
        print(header)
        print("-" * 90)

        def print_row(name, t_de, t_hccl):
            ms_de = to_ms(t_de)
            ms_hccl = to_ms(t_hccl)

            # 1. Speedup
            speedup_val = t_hccl / t_de if t_de > 1e-9 else 0.0
            speedup_str = f"{speedup_val:.2f}x"

            # 2. Saved Time
            saved = ms_hccl - ms_de

            # 3. Reduction Rate
            reduction_val = (t_hccl - t_de) / t_hccl * 100 if t_hccl > 1e-9 else 0.0
            reduction_str = f"{reduction_val:.1f}%"

            print(
                f"{name:<12} | "
                f"{ms_de:<12.3f} | "
                f"{ms_hccl:<12.3f} | "
                f"{speedup_str:<10} | "
                f"{saved:<12.3f} | "
                f"{reduction_str:<10}"
            )

        print_row("Dispatch", t_de_disp_avg, t_hccl_disp_avg)

        print_row("Combine", t_de_comb_avg, t_hccl_comb_avg)

        print("-" * 90)
        print(f"Correctness (1 - CosineSim): {diff_val:.6e}")
        if diff_val < 1e-4:
            print("Status: PASS")
        else:
            print("Status: CHECK FAILED")
        print("=" * 90 + "\n")
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)

    args = parser.parse_args()

    torch.multiprocessing.spawn(
        test_compare, args=(args.num_processes, args), nprocs=args.num_processes
    )

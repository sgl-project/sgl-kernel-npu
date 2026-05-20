import os

import torch
import torch.distributed as dist
import torch_npu

from .utils import EventOverlap

COMM_STREAM = None


def async_all_to_all(
    input_, output_split_sizes, input_split_sizes, group, event=None
):  #! 已检查
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.npu.current_device(),
        )

    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
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


def _gather_along_first_dim(input_, group):  #! 已检查
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.npu.current_device()
    )
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)
    return output


def alltoall_get_dispatch_layout(buffer, topk_idx, num_experts):  #! 已检查
    group = buffer.group
    group_size = buffer.group_size
    num_local_experts = num_experts // group_size  # todo 检查这个
    ep_rank = buffer.rank

    num_local_tokens_per_expert = torch.histc(
        topk_idx, bins=num_experts, min=0, max=num_experts
    )

    input_splits = (
        num_local_tokens_per_expert.reshape(group_size, num_local_experts)
        .sum(axis=1)
        .to(device="cpu", non_blocking=True)
        .numpy()
        .tolist()
    )

    num_global_tokens_per_expert = _gather_along_first_dim(
        num_local_tokens_per_expert, group
    ).reshape(group_size, num_experts)

    local_expert_indices_offset = ep_rank * num_local_experts
    local_expert_indices = [
        local_expert_indices_offset + i for i in range(num_local_experts)
    ]

    num_global_tokens_per_local_expert = num_global_tokens_per_expert[
        :, local_expert_indices[0] : local_expert_indices[-1] + 1
    ]

    output_splits = (
        num_global_tokens_per_local_expert.sum(axis=-1)
        .to(device="cpu", non_blocking=True)
        .numpy()
        .tolist()
    )

    num_tokens_per_expert = num_global_tokens_per_local_expert.sum(axis=0)

    expert_ids_per_ep_rank = torch.tensor(
        [i % num_local_experts for i in range(num_experts)],
        dtype=torch.int32,
        device=topk_idx.device,
    )

    num_global_tokens_per_local_expert_ravel = (
        num_global_tokens_per_local_expert.ravel()
    )
    if num_local_experts > 1:
        global_tokens_indices = torch.repeat_interleave(
            expert_ids_per_ep_rank,
            num_global_tokens_per_local_expert_ravel,
        )
    else:
        torch.npu.synchronize()
        global_tokens_indices = None

    layout = {
        "num_local_experts": num_local_experts,
        "input_splits": input_splits,
        "output_splits": output_splits,
        "num_global_tokens_per_local_expert": num_global_tokens_per_local_expert,
        "global_tokens_indices": global_tokens_indices,
    }
    buffer._alltoall_layout = layout

    num_tokens_per_rank = torch.tensor(input_splits, dtype=torch.int32, device="npu")
    is_token_in_rank = torch.zeros(
        (topk_idx.size(0), group_size), dtype=torch.bool, device="npu"
    )

    return (
        num_tokens_per_rank,
        None,
        num_tokens_per_expert,
        is_token_in_rank,
        EventOverlap(),
    )


def alltoall_dispatch(buffer, x, topk_idx, topk_weights):
    layout = buffer._alltoall_layout
    num_local_experts = layout["num_local_experts"]
    input_splits = layout["input_splits"]
    output_splits = layout["output_splits"]
    num_global_tokens_per_local_expert = layout["num_global_tokens_per_local_expert"]
    global_tokens_indices = layout["global_tokens_indices"]

    hidden_shape = x.shape
    x = x.view(-1, hidden_shape[-1])

    permutated_tokens, reversed_local_mapping = torch_npu.npu_moe_token_permute(
        tokens=x,
        indices=topk_idx,
        num_out_tokens=topk_idx.numel(),
    )

    input_quant = os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT") == "1"
    if input_quant:
        permutated_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
            permutated_tokens
        )
        _, dynamic_scale_after_all2all, scale_handle = async_all_to_all(
            dynamic_scale, output_splits, input_splits, buffer.group
        )
        scale_handle.wait()
        dynamic_scale.untyped_storage().resize_(0)
    else:
        dynamic_scale_after_all2all = None

    _, global_input_tokens, handle = async_all_to_all(
        permutated_tokens,
        output_splits,
        input_splits,
        buffer.group,
    )
    handle.wait()
    permutated_tokens.untyped_storage().resize_(0)

    if num_local_experts > 1:
        if input_quant:
            dynamic_scale_after_all2all, _ = torch_npu.npu_moe_token_permute(
                dynamic_scale_after_all2all.unsqueeze(-1), global_tokens_indices
            )
            dynamic_scale_after_all2all = dynamic_scale_after_all2all.squeeze(-1)

        dispatch_out, reversed_global_mapping = torch_npu.npu_moe_token_permute(
            global_input_tokens, global_tokens_indices
        )
    else:
        dispatch_out = global_input_tokens
        reversed_global_mapping = None

    num_recv_tokens_per_expert_list = (
        num_global_tokens_per_local_expert.sum(axis=0)
        .to(device="cpu", non_blocking=True)
        .numpy()
        .tolist()
    )

    combine_handle = {
        "input_splits": input_splits,
        "output_splits": output_splits,
        "topk_weights": topk_weights,
        "reversed_local_mapping": reversed_local_mapping,
        "reversed_global_mapping": reversed_global_mapping,
        "hidden_shape": hidden_shape,
        "hidden_shape_before_permute": x.shape,
        "num_local_experts": num_local_experts,
    }

    return (
        dispatch_out,
        dynamic_scale_after_all2all,
        None,
        num_recv_tokens_per_expert_list,
        combine_handle,
        EventOverlap(),
    )


def alltoall_combine(buffer, x, handle):
    input_splits = handle["input_splits"]
    output_splits = handle["output_splits"]
    topk_weights = handle["topk_weights"]
    reversed_local_mapping = handle["reversed_local_mapping"]
    reversed_global_mapping = handle["reversed_global_mapping"]
    hidden_shape = handle["hidden_shape"]
    hidden_shape_before_permute = handle["hidden_shape_before_permute"]
    num_local_experts = handle["num_local_experts"]

    if x.shape[0] > 0 and num_local_experts > 1 and reversed_global_mapping is not None:
        x = torch_npu.npu_moe_token_unpermute(x, reversed_global_mapping)

    _, local_tokens, a2a_handle = async_all_to_all(
        x,
        input_splits,
        output_splits,
        buffer.group,
    )
    a2a_handle.wait()
    x.untyped_storage().resize_(0)

    output = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=local_tokens,
        sorted_indices=reversed_local_mapping.to(torch.int32),
        probs=topk_weights,
        restore_shape=hidden_shape_before_permute,
    )
    output = output.view(hidden_shape)

    return output, None, EventOverlap()

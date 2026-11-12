'''
    Optimize the internal logic of the operator: support dividing the grid into the number of NPU cores; 
    add a new input parameter "total_token" to the kernel; implement the maximum even distribution of 
    "total_token" across different cores for computation inside the kernel.
'''

import torch
import triton
import triton.language as tl

# python/sgl_kernel_npu/sgl_kernel_npu/moe/deepep_post_reorder.py
@triton.jit
def deepep_post_reorder_triton_kernel(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = down_output_ptr.dtype.element_ty
    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    topk_weights_ptr = topk_weights_ptr + src_idx * topk
    store_ptr = output_ptr + src_idx * hidden_size

    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)
        for idx in range(0, topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                weigh_scale = tl.load(topk_weights_ptr + idx).to(InDtype)
                load_ptr = down_output_ptr + dst_idx * hidden_size
                in_data = tl.load(load_ptr + offset, mask=mask)
                sum_vec += in_data * weigh_scale
        tl.store(store_ptr + offset, sum_vec, mask=mask)


@triton.jit
def deepep_post_reorder_triton_kernel_npu(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    total_tokens,
):
    large_block_id = tl.program_id(0)
    num_blocks = tl.num_programs(0)

    base_tokens_per_block = total_tokens // num_blocks
    extra = total_tokens % num_blocks
    block_token_size = base_tokens_per_block + (large_block_id < extra)
    start_token = large_block_id * base_tokens_per_block + tl.minimum(large_block_id, extra)
    end_token = start_token + block_token_size
    InDtype = down_output_ptr.dtype.element_ty

    for src_idx in range(start_token, end_token):
        topk_offset = src_idx * topk
        src2dst_base = src2dst_ptr + topk_offset
        topk_ids_row = topk_ids_ptr + topk_offset
        topk_weights_row = topk_weights_ptr + topk_offset
        store_ptr = output_ptr + src_idx * hidden_size

        for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < hidden_size
            sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)

            for k in range(0, topk):
                dst_idx = tl.load(src2dst_base + k)
                if dst_idx >= 0:
                    weigh_scale = tl.load(topk_weights_row + k).to(InDtype)
                    load_ptr = down_output_ptr + dst_idx * hidden_size
                    in_data = tl.load(load_ptr + offset, mask=mask)
                    sum_vec += in_data * weigh_scale

            tl.store(store_ptr + offset, sum_vec, mask=mask)

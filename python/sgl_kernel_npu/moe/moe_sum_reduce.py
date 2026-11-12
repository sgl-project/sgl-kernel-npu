'''
    Optimize the data loading and accumulation logic of this Triton operator: 
    remove the internal for-loop-based loading and implement batch loading and 
    batch computation for 2D matrices.
'''

import torch
import triton
import triton.language as tl


# python/sgl_kernel_npu/sgl_kernel_npu/moe/moe_sum_reduce.py
@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: tl.constexpr,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)
    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, loop_unroll_factor=4):
            tmp = tl.load(
                input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0
            )
            accumulator += tmp
        accumulator = accumulator * routed_scaling_factor
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


@triton.jit
def _moe_sum_reduce_kernel_npu(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,  
    output_ptr,
    output_stride_0,
    output_stride_1, 
    token_num: int,
    topk_num: tl.constexpr,     
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    total_tokens: int,  # new parameter for balanced workload distribution
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    large_block_id = tl.program_id(0)
    num_blocks = tl.num_programs(0)

    # Balanced token distribution across NPU cores
    base_tokens_per_block = total_tokens // num_blocks
    extra = total_tokens % num_blocks
    block_size = base_tokens_per_block + (large_block_id < extra)
    start_token = large_block_id * base_tokens_per_block + tl.minimum(large_block_id, extra)
    end_token = start_token + block_size

    offs_dim = tl.arange(0, BLOCK_DIM)
    offs_expert = tl.arange(0, topk_num)

    for token_index in range(start_token, end_token):
        dim_start = 0
        while dim_start < hidden_dim:
            dim_end = min(dim_start + BLOCK_DIM, hidden_dim)
            current_offs_dim = dim_start + offs_dim
            mask_dim = current_offs_dim < dim_end

            # Vectorized 2D load: [topk_num, BLOCK_DIM]
            input_t_base = input_ptr + token_index * input_stride_0
            ptrs_2d = (
                input_t_base[None, :] +
                (offs_expert[:, None] * input_stride_1) +
                current_offs_dim[None, :]
            )
            mask_2d = mask_dim[None, :]

            tmps = tl.load(ptrs_2d, mask=mask_2d, other=0.0)  # [topk_num, BLOCK_DIM]
            accumulator = tl.sum(tmps, axis=0) * routed_scaling_factor  # [BLOCK_DIM]

            store_ptr = output_ptr + token_index * output_stride_0 + current_offs_dim
            tl.store(
                store_ptr,
                accumulator.to(input_ptr.dtype.element_ty),
                mask=mask_dim,
            )

            dim_start += BLOCK_DIM
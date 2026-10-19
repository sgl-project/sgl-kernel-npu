import torch
import triton
import triton.language as tl
import os
from typing import Any, Dict, List, Optional

"""
0. Discrete memory access: Declare input parameter N as a compile-time constant.
1. Discrete memory access: Load a_ptr row-wise and use insert_slice to improve performance.
2. Discrete memory access: Transpose b_ptr first, then apply insert_slice.
3. Loop unrolling: Unroll the for-loop to increase parallelism in data movement.
4. Discrete memory access: The store offsets for c_ptr are non-contiguous; use extract_slice to enhance performance.
5. Tiling optimization
6. Change type conversion from to(tl.int64) to to(tl.int32).
"""


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    for i in range(0,BLOCK_SIZE_M,1):
        offs_token_i = tl.get_element(offs_token, (i,))
        c_ptrs_i = c_ptr + stride_cm * offs_token_i + stride_cn * offs_cn[None, :]
        accumulator_i = tl.extract_slice(accumulator, (i,0), (1,BLOCK_SIZE_N), (1,1))
        token_mask_i = tl.get_element(token_mask, (i,))
        c_mask_i = token_mask_i & (offs_cn[None, :] < N)

        tl.store(c_ptrs_i, accumulator_i, mask=c_mask_i)
    # tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bias_e,
    stride_bias_n,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,#4
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
    mutibuffer = 1,
    ):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.
    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)# 0-19
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M) 
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_in_group = GROUP_SIZE_M * num_pid_n  
    group_id = pid // num_pid_in_group 
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int32)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id) 
    offs_token = offs_token.to(tl.int32)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int32)#size 4

    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak 
    ) 

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        # + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        + (offs_k[ None,:] * stride_bk + offs_bn[ :, None] * stride_bn)
    )
    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offset_by_cycle_a = 0 #tl.zeros([1], dtype=compute_type)
    offset_by_cycle_b = 0
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):#2048 32
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if even_Ks:#TRUE
            a_buffer = tl.zeros([BLOCK_SIZE_M , BLOCK_SIZE_K], dtype=compute_type)
            b_buffer = tl.zeros([ BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=compute_type)
            # b_buffer = tl.zeros([BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=compute_type)
            for i in range(0,BLOCK_SIZE_M,4):

                offs_token_i = tl.get_element(offs_token, (i,))
                a_ptrs_i = a_ptr + ( offs_token_i // top_k * stride_am + offs_k[None, :] * stride_ak ) + offset_by_cycle_a
                offs_token_i_1 = tl.get_element(offs_token, (i+1,))
                a_ptrs_i_1 = a_ptr + ( offs_token_i_1 // top_k * stride_am + offs_k[None, :] * stride_ak ) + offset_by_cycle_a
                offs_token_i_2 = tl.get_element(offs_token, (i+2,))
                a_ptrs_i_2 = a_ptr + ( offs_token_i_2 // top_k * stride_am + offs_k[None, :] * stride_ak ) + offset_by_cycle_a
                offs_token_i_3 = tl.get_element(offs_token, (i+3,))
                a_ptrs_i_3 = a_ptr + ( offs_token_i_3 // top_k * stride_am + offs_k[None, :] * stride_ak ) + offset_by_cycle_a

                token_mask_i = tl.get_element(token_mask, (i,))
                a_i = tl.load(a_ptrs_i, mask=token_mask_i, other=0.0,)
                token_mask_i_1 = tl.get_element(token_mask, (i+1,))
                a_i_1 = tl.load(a_ptrs_i_1, mask=token_mask_i_1, other=0.0,)
                token_mask_i_2 = tl.get_element(token_mask, (i+2,))
                a_i_2 = tl.load(a_ptrs_i_2, mask=token_mask_i_2, other=0.0,)
                token_mask_i_3 = tl.get_element(token_mask, (i+3,))
                a_i_3 = tl.load(a_ptrs_i_3, mask=token_mask_i_3, other=0.0,)

                a_buffer = tl.insert_slice(a_buffer, a_i, offsets=(i, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))
                a_buffer = tl.insert_slice(a_buffer, a_i_1, offsets=(i+1, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))
                a_buffer = tl.insert_slice(a_buffer, a_i_2, offsets=(i+2, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))
                a_buffer = tl.insert_slice(a_buffer, a_i_3, offsets=(i+3, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))

            for j in range(0,BLOCK_SIZE_N,4):
                offs_bn_j = tl.get_element(offs_bn, (j,))
                b_ptrs_j = ( b_ptr + off_experts * stride_be + (offs_k[ None,:] * stride_bk + offs_bn_j * stride_bn)) + offset_by_cycle_b
                offs_bn_j_1 = tl.get_element(offs_bn, (j+1,))
                b_ptrs_j_1 = ( b_ptr + off_experts * stride_be + (offs_k[ None,:] * stride_bk + offs_bn_j_1 * stride_bn)) + offset_by_cycle_b
                offs_bn_j_2 = tl.get_element(offs_bn, (j+2,))
                b_ptrs_j_2 = ( b_ptr + off_experts * stride_be + (offs_k[ None,:] * stride_bk + offs_bn_j_2 * stride_bn)) + offset_by_cycle_b
                offs_bn_j_3 = tl.get_element(offs_bn, (j+3,))
                b_ptrs_j_3 = ( b_ptr + off_experts * stride_be + (offs_k[ None,:] * stride_bk + offs_bn_j_3 * stride_bn)) + offset_by_cycle_b

                b_j = tl.load(b_ptrs_j)
                b_j_1 = tl.load(b_ptrs_j_1)
                b_j_2 = tl.load(b_ptrs_j_2)
                b_j_3 = tl.load(b_ptrs_j_3)

                b_buffer = tl.insert_slice(b_buffer, b_j, offsets=(j, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))
                b_buffer = tl.insert_slice(b_buffer, b_j_1, offsets=(j+1, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))
                b_buffer = tl.insert_slice(b_buffer, b_j_2, offsets=(j+2, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))
                b_buffer = tl.insert_slice(b_buffer, b_j_3, offsets=(j+3, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))

        else:
            # a = tl.load(
            #     a_ptrs,
            #     mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            #     other=0.0,
            # )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

            a_buffer = tl.zeros([BLOCK_SIZE_M , BLOCK_SIZE_K], dtype=compute_type)
            b_buffer = tl.zeros([ BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=compute_type)
            # b_buffer = tl.zeros([BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=compute_type)
            for i in range(0,BLOCK_SIZE_M,4):

                offs_token_i = tl.get_element(offs_token, (i,))
                a_ptrs_i = a_ptr + ( offs_token_i // top_k * stride_am + offs_k[None, :] * stride_ak ) + offset_by_cycle_a
                offs_token_i_1 = tl.get_element(offs_token, (i+1,))
                a_ptrs_i_1 = a_ptr + ( offs_token_i_1 // top_k * stride_am + offs_k[None, :] * stride_ak ) + offset_by_cycle_a
                offs_token_i_2 = tl.get_element(offs_token, (i+2,))
                a_ptrs_i_2 = a_ptr + ( offs_token_i_2 // top_k * stride_am + offs_k[None, :] * stride_ak ) + offset_by_cycle_a
                offs_token_i_3 = tl.get_element(offs_token, (i+3,))
                a_ptrs_i_3 = a_ptr + ( offs_token_i_3 // top_k * stride_am + offs_k[None, :] * stride_ak ) + offset_by_cycle_a

                token_mask_i = tl.get_element(token_mask, (i,)) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
                a_i = tl.load(a_ptrs_i, mask=token_mask_i, other=0.0,)
                token_mask_i_1 = tl.get_element(token_mask, (i+1,)) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
                a_i_1 = tl.load(a_ptrs_i_1, mask=token_mask_i_1, other=0.0,)
                token_mask_i_2 = tl.get_element(token_mask, (i+2,))& (offs_k[None, :] < K - k * BLOCK_SIZE_K)
                a_i_2 = tl.load(a_ptrs_i_2, mask=token_mask_i_2, other=0.0,)
                token_mask_i_3 = tl.get_element(token_mask, (i+3,))& (offs_k[None, :] < K - k * BLOCK_SIZE_K)
                a_i_3 = tl.load(a_ptrs_i_3, mask=token_mask_i_3, other=0.0,)

                a_buffer = tl.insert_slice(a_buffer, a_i, offsets=(i, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))
                a_buffer = tl.insert_slice(a_buffer, a_i_1, offsets=(i+1, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))
                a_buffer = tl.insert_slice(a_buffer, a_i_2, offsets=(i+2, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))
                a_buffer = tl.insert_slice(a_buffer, a_i_3, offsets=(i+3, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))

            for j in range(0,BLOCK_SIZE_N,4):
                offs_bn_j = tl.get_element(offs_bn, (j,))
                b_ptrs_j = ( b_ptr + off_experts * stride_be + (offs_k[ None,:] * stride_bk + offs_bn_j * stride_bn)) + offset_by_cycle_b
                b_mask_j = tl.get_element(offs_k, (j,)) < K - k * BLOCK_SIZE_K
                offs_bn_j_1 = tl.get_element(offs_bn, (j+1,))
                b_ptrs_j_1 = ( b_ptr + off_experts * stride_be + (offs_k[ None,:] * stride_bk + offs_bn_j_1 * stride_bn)) + offset_by_cycle_b
                b_mask_j_1 = tl.get_element(offs_k, (j+1,)) < K - k * BLOCK_SIZE_K
                offs_bn_j_2 = tl.get_element(offs_bn, (j+2,))
                b_ptrs_j_2 = ( b_ptr + off_experts * stride_be + (offs_k[ None,:] * stride_bk + offs_bn_j_2 * stride_bn)) + offset_by_cycle_b
                b_mask_j_2 = tl.get_element(offs_k, (j+3,)) < K - k * BLOCK_SIZE_K
                offs_bn_j_3 = tl.get_element(offs_bn, (j+3,))
                b_ptrs_j_3 = ( b_ptr + off_experts * stride_be + (offs_k[ None,:] * stride_bk + offs_bn_j_3 * stride_bn)) + offset_by_cycle_b
                b_mask_j_3 = tl.get_element(offs_k, (j+4,)) < K - k * BLOCK_SIZE_K

                b_j = tl.load(b_ptrs_j, mask=b_mask_j, other=0.0)
                b_j_1 = tl.load(b_ptrs_j_1, mask=b_mask_j_1, other=0.0)
                b_j_2 = tl.load(b_ptrs_j_2, mask=b_mask_j_2, other=0.0)
                b_j_3 = tl.load(b_ptrs_j_3, mask=b_mask_j_3, other=0.0)

                b_buffer = tl.insert_slice(b_buffer, b_j, offsets=(j, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))
                b_buffer = tl.insert_slice(b_buffer, b_j_1, offsets=(j+1, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))
                b_buffer = tl.insert_slice(b_buffer, b_j_2, offsets=(j+2, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))
                b_buffer = tl.insert_slice(b_buffer, b_j_3, offsets=(j+3, 0), sizes=(1, BLOCK_SIZE_K), strides=(1, 1))



        # We accumulate along the K dimension.
        if use_int8_w8a16:
            # accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
            a_buffer = a_buffer + 1e-6
            b_buffer = b_buffer + 1e-6
            accumulator += tl.dot(a_buffer, tl.trans(b_buffer.to(compute_type)), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                # accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                a_buffer = a_buffer + 1e-6
                b_buffer = b_buffer + 1e-6
                accumulator += tl.dot(a_buffer, tl.trans(b_buffer))* a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # accumulator = tl.dot(a, b, acc=accumulator)
                    a_buffer = a_buffer + 1e-6
                    b_buffer = b_buffer + 1e-6
                    accumulator += tl.dot(a_buffer, tl.trans(b_buffer), acc=accumulator)
                else:
                    a_buffer = a_buffer + 1e-6
                    b_buffer = b_buffer + 1e-6
                    accumulator += tl.dot(a_buffer, tl.trans(b_buffer))
                    # accumulator += tl.dot(a, b)
        else:
            # accumulator += tl.dot(a, b)
            a_buffer = a_buffer + 1e-6
            b_buffer = b_buffer + 1e-6
            accumulator += tl.dot(a_buffer, tl.trans(b_buffer))

        # Advance the ptrs to the next K block.
        # a_ptrs += BLOCK_SIZE_K * stride_ak
        # b_ptrs += BLOCK_SIZE_K * stride_bk
        offset_by_cycle_a += BLOCK_SIZE_K * stride_ak
        offset_by_cycle_b += BLOCK_SIZE_K * stride_bk


    if use_int8_w8a16:
        accumulator *= b_scale
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k == 0 or group_n == 0:
            accumulator *= a_scale * b_scale

    if bias_ptr is not None:
        accumulator += bias

    if MUL_ROUTED_WEIGHT: 
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type) 
    # accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    for i in range(0,BLOCK_SIZE_M,1):
        offs_token_i = tl.get_element(offs_token, (i,))
        c_ptrs_i = c_ptr + stride_cm * offs_token_i + stride_cn * offs_cn[None, :]
        accumulator_i = tl.extract_slice(accumulator, (i,0), (1,BLOCK_SIZE_N), (1,1))
        token_mask_i = tl.get_element(token_mask, (i,))
        c_mask_i = token_mask_i & (offs_cn[None, :] < N)

        tl.store(c_ptrs_i, accumulator_i, mask=c_mask_i)

    # tl.store(c_ptrs, accumulator, mask=c_mask)

def generate_test_data(
    M=32,
    top_k=2,
    E=4,
    N=64,
    K=128,
    group_n=32,
    group_k=64,
    use_bias=True,
    MUL_ROUTED_WEIGHT=True,
    use_fp8_w8a8=False,
    use_int8_w8a8=False,
    use_int8_w8a16=False,
    per_channel_quant=False,
    BLOCK_SIZE_M=16,
    BLOCK_SIZE_N=32,
    BLOCK_SIZE_K=32,
    GROUP_SIZE_M=8,
    dtype=torch.float16,
    device="npu",
):
    assert sum([use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16]) <= 1, "Only one quant mode allowed"

    total_tokens = M * top_k
    EM = ((total_tokens + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * BLOCK_SIZE_M

    expert_ids_ptr = torch.randint(0, E, (EM // BLOCK_SIZE_M,), dtype=torch.int64, device=device)

    sorted_token_ids_ptr = torch.full((EM,), -1, dtype=torch.int64, device=device)
    token_counter = 0
    for i in range(EM // BLOCK_SIZE_M):
        for j in range(BLOCK_SIZE_M):
            idx = i * BLOCK_SIZE_M + j
            if token_counter < total_tokens:
                sorted_token_ids_ptr[idx] = token_counter
                token_counter += 1
            else:
                sorted_token_ids_ptr[idx] = total_tokens

    num_valid_tokens = torch.tensor(total_tokens, dtype=torch.int64, device=device)
    num_tokens_post_padded_ptr = torch.tensor(EM, dtype=torch.int64, device=device)

    a_ptr = torch.randn(M, K, dtype=dtype, device=device)
    b_ptr = torch.randn(E, N, K, dtype=dtype, device=device)
    bias_ptr = torch.randn(E, N, dtype=dtype, device=device) if use_bias else None
    topk_weights_ptr = torch.rand(total_tokens, dtype=torch.float32, device=device) if MUL_ROUTED_WEIGHT else None

    a_scale_ptr = None
    b_scale_ptr = None

    if use_int8_w8a16:
        if per_channel_quant:
            b_scale_ptr = torch.rand(E, N, dtype=torch.float32, device=device)
        else:
            b_scale_ptr = torch.rand(E, dtype=torch.float32, device=device)
        b_ptr = b_ptr.to(torch.float32)
        if per_channel_quant:
            b_ptr = (b_ptr / b_scale_ptr[:, :, None]).round().clamp(-128, 127).to(torch.int8)
        else:
            b_ptr = (b_ptr / b_scale_ptr[:, None, None]).round().clamp(-128, 127).to(torch.int8)
        a_ptr = a_ptr.to(torch.float16)

    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_n > 0 and group_k > 0:
            num_groups_k = (K + group_k - 1) // group_k
            num_groups_n = (N + group_n - 1) // group_n
            a_scale_ptr = torch.rand(M, num_groups_k, dtype=torch.float32, device=device)
            b_scale_ptr = torch.rand(E, num_groups_n, dtype=torch.float32, device=device)
        elif per_channel_quant:
            a_scale_ptr = torch.rand(M, dtype=torch.float32, device=device)
            b_scale_ptr = torch.rand(E, N, dtype=torch.float32, device=device)
        else:
            a_scale_ptr = torch.tensor(1.0, dtype=torch.float32, device=device)
            b_scale_ptr = torch.tensor(1.0, dtype=torch.float32, device=device)

        a_fp32 = a_ptr.to(torch.float32)
        b_fp32 = b_ptr.to(torch.float32)

        if group_n > 0 and group_k > 0:
            a_quant = torch.zeros_like(a_fp32)
            num_groups_k = a_scale_ptr.shape[1]
            for i in range(num_groups_k):
                k_start = i * group_k
                k_end = min(k_start + group_k, K)
                scale = a_scale_ptr[:, i:i+1]
                a_quant[:, k_start:k_end] = (a_fp32[:, k_start:k_end] / scale).round().clamp(-128, 127)
            a_ptr = a_quant.to(torch.int8)

            b_quant = torch.zeros_like(b_fp32)
            num_groups_n = b_scale_ptr.shape[1]
            for e in range(E):
                for j in range(num_groups_n):
                    n_start = j * group_n
                    n_end = min(n_start + group_n, N)
                    scale = b_scale_ptr[e, j]
                    b_quant[e, n_start:n_end, :] = (b_fp32[e, n_start:n_end, :] / scale).round().clamp(-128, 127)
            b_ptr = b_quant.to(torch.int8)
        else:
            if per_channel_quant:
                a_ptr = (a_fp32 / a_scale_ptr[:, None]).round().clamp(-128, 127).to(torch.int8)
                b_ptr = (b_fp32 / b_scale_ptr[:, :, None]).round().clamp(-128, 127).to(torch.int8)
            else:
                a_ptr = (a_fp32 / a_scale_ptr).round().clamp(-128, 127).to(torch.int8)
                b_ptr = (b_fp32 / b_scale_ptr).round().clamp(-128, 127).to(torch.int8)

    c_ptr = torch.zeros(EM, N, dtype=dtype, device=device)

    def get_stride(tensor, dim):
        return tensor.stride(dim) if tensor is not None else 0

    return {
        # Pointers
        "a_ptr": a_ptr,
        "b_ptr": b_ptr,
        "bias_ptr": bias_ptr,
        "c_ptr": c_ptr,
        "a_scale_ptr": a_scale_ptr,
        "b_scale_ptr": b_scale_ptr,
        "topk_weights_ptr": topk_weights_ptr,
        "sorted_token_ids_ptr": sorted_token_ids_ptr,
        "expert_ids_ptr": expert_ids_ptr,
        "num_tokens_post_padded_ptr": num_tokens_post_padded_ptr,

        # Dimensions
        "N": N,
        "K": K,
        "EM": EM,
        "num_valid_tokens": num_valid_tokens.item(),  # scalar

        # Strides
        "stride_am": get_stride(a_ptr, 0),
        "stride_ak": get_stride(a_ptr, 1),
        "stride_be": get_stride(b_ptr, 0),
        "stride_bk": get_stride(b_ptr, 2),
        "stride_bn": get_stride(b_ptr, 1),
        "stride_bias_e": get_stride(bias_ptr, 0) if bias_ptr is not None else 0,
        "stride_bias_n": get_stride(bias_ptr, 1) if bias_ptr is not None else 0,
        "stride_cm": get_stride(c_ptr, 0),
        "stride_cn": get_stride(c_ptr, 1),
        "stride_asm": get_stride(a_scale_ptr, 0) if a_scale_ptr is not None and a_scale_ptr.ndim > 0 else 0,
        "stride_ask": get_stride(a_scale_ptr, 1) if a_scale_ptr is not None and a_scale_ptr.ndim > 1 else 0,
        "stride_bse": get_stride(b_scale_ptr, 0) if b_scale_ptr is not None and b_scale_ptr.ndim > 0 else 0,
        "stride_bsk": get_stride(b_scale_ptr, 1) if b_scale_ptr is not None and b_scale_ptr.ndim > 1 else 0,
        "stride_bsn": get_stride(b_scale_ptr, 1) if b_scale_ptr is not None and b_scale_ptr.ndim == 2 else 0,

        # Constexpr (will be passed separately or via **kwargs if using wrapper)
        "group_n": group_n,
        "group_k": group_k,
        "BLOCK_SIZE_M": BLOCK_SIZE_M,
        "BLOCK_SIZE_N": BLOCK_SIZE_N,
        "BLOCK_SIZE_K": BLOCK_SIZE_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
        "MUL_ROUTED_WEIGHT": MUL_ROUTED_WEIGHT,
        "top_k": top_k,
        "compute_type" : (
            "float16" if dtype == torch.float16 else
            "bfloat16" if dtype == torch.bfloat16 else
            "float32"
        ),
        "use_fp8_w8a8": use_fp8_w8a8,
        "use_int8_w8a8": use_int8_w8a8,
        "use_int8_w8a16": use_int8_w8a16,
        "per_channel_quant": per_channel_quant,
        "even_Ks": (K % BLOCK_SIZE_K == 0),
    }


def generate(path = "./fused_moe_kernel_test_data.pt"):

    param = {
        'M' : 160,
        'top_k' : 8,
        'E' : 128,
        'N' : 192,
        'K' : 2048,
        'group_n' : 0,
        'group_k' : 0,
        'use_bias' : True,
        'MUL_ROUTED_WEIGHT' : True,
        'use_fp8_w8a8' : False,
        'use_int8_w8a8' : False,
        'use_int8_w8a16' : False,
        'per_channel_quant' : False,
        'BLOCK_SIZE_M' : 64,
        'BLOCK_SIZE_N' : 64,
        'BLOCK_SIZE_K' : 32,
        'GROUP_SIZE_M' : 8,
        'dtype' : torch.float16,
        }

    data = generate_test_data(**param)
    grid =  (
        triton.cdiv(data["EM"], data["BLOCK_SIZE_M"])
        * triton.cdiv(data["N"], data["BLOCK_SIZE_N"]),
    )
    ret = {'data':data, "grid":grid}

    torch.save(ret,path)

if __name__ == "__main__":
    # 设置设备为 NPU
    device = 'npu:0'
    torch.npu.set_device(device)

    print("Using device:", torch.npu.current_device())

    # =============== 1. 生成测试数据 ===============
    path = "./fused_moe_kernel_gpu_full.pt"
    generate(path)
    data = torch.load(path,map_location="npu")
    kernel_args = data["data"]

    kernel_args["BLOCK_SIZE_N"] = 128
    # kernel_args["BLOCK_SIZE_M"] = 64 #fixed
    kernel_args["BLOCK_SIZE_K"] = 128

    grid =  (
        triton.cdiv(kernel_args["EM"], kernel_args["BLOCK_SIZE_M"])
        * triton.cdiv(kernel_args["N"], kernel_args["BLOCK_SIZE_N"]),
    )

    str_to_tl_dtype = {
    "float16": tl.float16,
    "bfloat16":tl.bfloat16,
    "float32": tl.float32,
    }

    kernel_args["compute_type"] = str_to_tl_dtype[kernel_args["compute_type"]]
    fused_moe_kernel[grid](**kernel_args)

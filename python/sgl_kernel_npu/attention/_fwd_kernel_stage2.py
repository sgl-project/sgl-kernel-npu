'''
    Optimize the internal logic of the operator: support dividing the grid into the number of NPU cores; 
    add new input parameters "total_batch" and "num_heads" to the kernel; "total_batch" is used to control the number of batches processed per block,
    "num_heads" is used to control the number of iterations of the for-loop inside the operator, reducing the division of blocks.
'''

import torch
import triton
import triton.language as tl


# python/sgl_kernel_npu/attention/decode_softmax_reducev.py
@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    kv_indptr,
    num_kv_splits,
    sink_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    kv_splits = tl.load(num_kv_splits + cur_batch)

    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * (stride_mid_os // Lv))

            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        e_sum += tl.exp(cur_sink - e_max)

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


@triton.jit
def _fwd_kernel_stage2_npu(
    Mid_O,
    Mid_O_1,
    O,
    kv_indptr,
    num_kv_splits,
    sink_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    HAS_SINK: tl.constexpr,
    total_batch: tl.constexpr,
    num_heads: tl.constexpr,
):
    cur_large_batch_id = tl.program_id(0)
    num_blocks = tl.num_programs(0)

    base_batch_per_block = total_batch // num_blocks
    extra = total_batch % num_blocks

    block_batch_size = base_batch_per_block + (cur_large_batch_id < extra)
    start_batch = cur_large_batch_id * base_batch_per_block + tl.minimum(cur_large_batch_id, extra)
    end_batch = start_batch + block_batch_size

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    for cur_batch in range(start_batch, end_batch):
        # Early exit if out of real batch range (handled by caller via grid size)
        for cur_head in range(num_heads):
            cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
            kv_splits = tl.load(num_kv_splits + cur_batch)

            e_sum = 0.0
            e_max = -float("inf")
            acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

            offs_v_base = cur_batch * stride_mid_ob + cur_head * stride_mid_oh
            offs_v = offs_v_base + offs_d
            offs_logic = offs_v_base // Lv

            kv_len_per_split = (
                tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
            )

            for split_kv_id in range(0, MAX_KV_SPLITS):
                split_kv_start = kv_len_per_split * split_kv_id
                split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

                if split_kv_end > split_kv_start:
                    tv = tl.load(
                        Mid_O + offs_v + split_kv_id * stride_mid_os,
                        mask=mask_d,
                        other=0.0
                    )
                    tlogic = tl.load(
                        Mid_O_1 + offs_logic + split_kv_id * (stride_mid_os // Lv)
                    )
                    n_e_max = tl.maximum(tlogic, e_max)

                    old_scale = tl.exp(e_max - n_e_max)
                    acc *= old_scale
                    exp_logic = tl.exp(tlogic - n_e_max)
                    acc += exp_logic * tv

                    e_sum = e_sum * old_scale + exp_logic
                    e_max = n_e_max

            if HAS_SINK:
                cur_sink = tl.load(sink_ptr + cur_head)
                e_sum += tl.exp(cur_sink - e_max)

            output_ptrs = O + cur_batch * stride_obs + cur_head * stride_oh + offs_d
            tl.store(output_ptrs, acc / e_sum, mask=mask_d)
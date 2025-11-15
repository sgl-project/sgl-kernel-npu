# Optimize the internal logic of the operator: support dividing the grid into the number of NPU cores; 
# add new input parameters "BATCH_GROUP_SIZE" and "num_heads" to the kernel; BATCH_GROUP_SIZE is used to control the number of batches processed per block,
# num_heads is used to control the number of iterations of the for-loop inside the operator, reducing the division of blocks.

# source: python/sglang/srt/layers/attention/triton_ops/decode_attention.py
import sys
import pytest
import triton
import torch
import triton.language as tl
import os
import torch_npu
import triton.runtime.driver as driver

def convert_tensor_with_device_type(indata: dict, device_type: str):
    target_device = torch.device(device_type)
    outdata = {}
    for key, value in indata.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != target_device.type:
                outdata[key] = value.to(target_device)
            else:
                outdata[key] = value
        else:
            outdata[key] = value
    return outdata

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

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
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

            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)


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
    BATCH_GROUP_SIZE: tl.constexpr,  
    num_heads,
):
    cur_large_batch_id = tl.program_id(0)
    start_batch = cur_large_batch_id * BATCH_GROUP_SIZE  
    end_batch = start_batch + BATCH_GROUP_SIZE          
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    for cur_batch in range(start_batch, end_batch):
        for cur_head in range(0, num_heads): 

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

# Calling scenario of this Triton operator (sglang v0.4.8 python/sglang/srt/layers/attention/triton_ops/decode_attention.py)
def _decode_softmax_reducev_fwd(
    logits,
    lse,
    q,
    o,
    v_buffer,
    kv_indptr,
    num_kv_splits,
    max_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits

    extra_kargs = {}
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    # grid = (batch, head_num)
    
    # Add a new input parameter BATCH_GROUP_SIZE to control the amount of data processed per block; meanwhile, remove the division of head_num and integrate it into the operator, processing it within the same block using a for-loop.
    small_batch = batch // 5
    grid = (small_batch, )
    BATCH_GROUP_SIZE = (batch + small_batch - 1)// small_batch
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        kv_indptr,
        num_kv_splits,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        BATCH_GROUP_SIZE=BATCH_GROUP_SIZE, # new input parameter
        num_heads=head_num, # new input parameter
        **extra_kargs,
    )

def profiling_test(fn_triton, args=(), name="46", shape=(), tiling=()):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
        )
    with torch_npu.profiler.profile(
            activities=[  # torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False,  # Switch to collect the function call stack of PyTorch operators. This parameter is optional, default is disabled.
            record_shapes=False,  # Switch to collect the input shape and input type of PyTorch operators. This parameter is optional, default is disabled.
            profile_memory=False,  # Switch to collect memory-related data. This parameter is optional, default is disabled.
            schedule=torch_npu.profiler.schedule(wait=1,
                                                warmup=1,
                                                active=30,
                                                repeat=1,
                                                skip_first=1),
            # warmup defaults to 0; this parameter is mandatory in older versions of the torch_npu package
            experimental_config=experimental_config,  # This parameter is optional, default is Level0
            # Location of the generated profiling files
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
    ) as prof:
        # prof.start()
        for i in range(30):
            fn_triton(*args)
            torch.npu.synchronize()  # Ensure the kernel is actually executed to completion
            prof.step()

def fn_triton(grid, input_data):
    # _fwd_kernel_stage2_npu[grid](**input_data)
    
    # The following code is used for comparison with the original operator.
    _fwd_kernel_stage2[grid](**input_data)
    return 

def test_fwd_kernel_stage2():
    batch = 160
    num_heads = 8
    MAX_KV_SPLITS = 16
    MIN_BLOCK_KV = 32
    BLOCK_DV = 128
    Lv = 128

    Mid_O = torch.randn(size=(batch, num_heads, MAX_KV_SPLITS, Lv), dtype=torch.float32, device='cpu')
    Mid_O_1 = torch.randn(size=(batch, num_heads, MAX_KV_SPLITS), dtype=torch.float32, device='cpu')
    O = torch.randn(size=(batch, num_heads, Lv), dtype=torch.bfloat16, device='cpu')
    kv_indptr = torch.arange(0, batch+1, dtype=torch.int32, device='cpu')
    num_kv_splits = torch.full(size=(batch,), fill_value=8, dtype=torch.int32, device='cpu')
    sink_ptr = torch.zeros(size=(8,), dtype=torch.float32, device='cpu')
    stride_mid_ob = Mid_O.stride(0)
    stride_mid_oh = Mid_O.stride(1)
    stride_mid_os = Mid_O.stride(2)
    stride_obs = O.stride(1)
    stride_oh = O.stride(2)
    HAS_SINK = 0

    device = torch.npu.current_device()
    device_properties = driver.active.utils.get_device_properties(device)
    npu_num_core = device_properties["num_vectorcore"]
    BATCH_GROUP_SIZE = (batch + npu_num_core - 1) // npu_num_core
    
    grid = (npu_num_core, 1,)
    # input_data = {
    #     'Mid_O': Mid_O,
    #     'Mid_O_1': Mid_O_1,
    #     'O': O,
    #     'kv_indptr': kv_indptr,
    #     'num_kv_splits': num_kv_splits,
    #     'stride_mid_ob': stride_mid_ob,
    #     'stride_mid_oh': stride_mid_oh,
    #     'stride_mid_os': stride_mid_os,
    #     'stride_obs': stride_obs,
    #     'stride_oh': stride_oh,
    #     'MAX_KV_SPLITS': MAX_KV_SPLITS,
    #     'MIN_BLOCK_KV': MIN_BLOCK_KV,
    #     'BLOCK_DV': BLOCK_DV,
    #     'Lv': Lv,
    #     'sink_ptr': sink_ptr,
    #     'HAS_SINK': HAS_SINK,
    #     'BATCH_GROUP_SIZE': BATCH_GROUP_SIZE,
    #     'num_heads': num_heads
    # }

    # The following code is used to compare the performance of the original operator. Please modify the operator call in fn_triton accordingly.
    grid = (batch, num_heads,)
    input_data = {
        'Mid_O': Mid_O,
        'Mid_O_1': Mid_O_1,
        'O': O,
        'kv_indptr': kv_indptr,
        'num_kv_splits': num_kv_splits,
        'stride_mid_ob': stride_mid_ob,
        'stride_mid_oh': stride_mid_oh,
        'stride_mid_os': stride_mid_os,
        'stride_obs': stride_obs,
        'stride_oh': stride_oh,
        'MAX_KV_SPLITS': MAX_KV_SPLITS,
        'MIN_BLOCK_KV': MIN_BLOCK_KV,
        'BLOCK_DV': BLOCK_DV,
        'Lv': Lv,
        'sink_ptr': sink_ptr,
        'HAS_SINK': HAS_SINK,
    }

    input_data = convert_tensor_with_device_type(input_data, device_type='npu')

    profiling_test(fn_triton, (grid, input_data))

if __name__ == '__main__':
    test_fwd_kernel_stage2()
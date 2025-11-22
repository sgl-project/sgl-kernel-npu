# Optimize the internal logic of the operator: support dividing the grid into the number of NPU cores; 
# add a new input parameter "total_token" to the kernel; implement the maximum even distribution of 
# total_token across different cores for computation inside the kernel.

import torch
import triton
import triton.language as tl


import torch_npu 
import copy
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

    # Calculate the number of tokens per group (the first 'total_tokens % number_of_NPU_cores' groups have one extra token).
    # Maximize the even distribution of data across each core as much as possible.
    base_tokens_per_block = total_tokens // num_blocks
    extra = total_tokens % num_blocks
    block_size = base_tokens_per_block + (large_block_id < extra)
    start_token = large_block_id * base_tokens_per_block + tl.minimum(large_block_id, extra) 
    end_token = start_token + block_size
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

# Calling scenario of this Triton operator (sglang v0.4.8 /python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py)
def combine_a(
    self,
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
):
    if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        output = hidden_states
    else:
        if hidden_states.shape[0] > 0:
            num_tokens = self.src2dst.shape[0] // self.router_topk
            output = torch.empty(
                (num_tokens, hidden_states.shape[1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            deepep_post_reorder_triton_kernel[(num_npu_vec_kernel,)](
                hidden_states,
                output,
                self.src2dst,
                topk_idx,
                topk_weights,
                self.router_topk,
                hidden_states.shape[1],
                BLOCK_SIZE=512,
                total_tokens=num_tokens # new input parameter
            )
        else:
            output = torch.zeros(
                (0, hidden_states.shape[1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
    previous_event = Buffer.capture() if self.async_finish else None
    return output, previous_event

def profiling_test(fn_triton, args=(), name="46", shape=(), tiling=()):
    grid, input_data = args
    experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
        )
    with torch_npu.profiler.profile(
            activities=[  # torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False,  # 采集torch 算子的函数调用栈的开关，该参数选填，默认关闭
            record_shapes=False,  # 采集torch 算子的input shape和input type的开关，该参数选填，默认关闭
            profile_memory=False,  # 采集memory相关数据的开关，该参数选填，默认关闭
            schedule=torch_npu.profiler.schedule(wait=1,
                                                warmup=1,
                                                active=30,
                                                repeat=1,
                                                skip_first=1),
            # warmup默认为0，老版本torch_npu包该参数为必填项
            experimental_config=experimental_config,  # 该参数选填，默认为Level0
            # 产生的profling文件的位置
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
    ) as prof:
        # prof.start()
        for i in range(30):
            fn_triton(grid, input_data)
            torch.npu.synchronize()  # 确保 kernel 真正执行完
            prof.step()

def fn_triton(grid, input_data):

    deepep_post_reorder_triton_kernel_npu[grid](**input_data)

    # The following code is used for comparison with the original operator.
    # deepep_post_reorder_triton_kernel[grid](**input_data)
    return 

def test_deepep_post_reorder_triton_kernel_npu():
    total_routed_tokens = 17390  # down_output_ptr
    num_tokens = 6920            # output_ptr
    hidden_size = 1024
    BLOCK_SIZE = hidden_size     # Maximize BLOCK_SIZE
    topk = 8
    dtype_bf16 = torch.bfloat16
    dtype_fp32 = torch.float32
    dtype_int64 = torch.int64

    # 1. down_output_ptr: (17393, 7168), bfloat16
    down_output_ptr = torch.randn(total_routed_tokens, hidden_size, dtype=dtype_bf16)

    # 2. output_ptr: (6923, 7168), bfloat16 
    output_ptr = torch.zeros(num_tokens, hidden_size, dtype=dtype_bf16)

    # 3. src2dst_ptr: (6923 * 8,) = (55384,), int64
    # src2dst_ptr[i] ∈ [0, 17393)
    src2dst_ptr = torch.randint(
        low=0,
        high=total_routed_tokens,
        size=(num_tokens * topk,),
        dtype=dtype_int64
    )

    # 4. topk_ids_ptr: (6923, 8), int64
    topk_ids_ptr = torch.randint(
        low=0,
        high=1000,  # ID ∈ [0, 999]
        size=(num_tokens, topk),
        dtype=dtype_int64
    )
    # simulator padding
    mask_invalid = torch.rand(num_tokens, topk) < 0.1  # 10% invalid
    topk_ids_ptr[mask_invalid] = -1

    # 5. topk_weights_ptr: (6923, 8), float32
    topk_weights_ptr = torch.rand(num_tokens, topk, dtype=dtype_fp32)
    topk_weights_ptr = torch.nn.functional.softmax(topk_weights_ptr, dim=-1)

    # 7. npu_num_core as grid[0]
    device = torch.npu.current_device()
    device_properties = driver.active.utils.get_device_properties(device)
    npu_num_core = device_properties["num_vectorcore"]

    grid = (npu_num_core, 1, 1)
    input_data = {
        'down_output_ptr': down_output_ptr,
        'output_ptr': output_ptr,
        'src2dst_ptr': src2dst_ptr,
        'topk_ids_ptr': topk_ids_ptr,
        'topk_weights_ptr': topk_weights_ptr,
        'topk': topk,
        'hidden_size': hidden_size,
        'BLOCK_SIZE': hidden_size,
        'total_tokens': num_tokens
    }

    # The following code is used to compare the performance of the original operator. Please modify the operator call in fn_triton accordingly.
    # grid = (num_tokens, 1, 1)
    # input_data = {
    #     'down_output_ptr': down_output_ptr,
    #     'output_ptr': output_ptr,
    #     'src2dst_ptr': src2dst_ptr,
    #     'topk_ids_ptr': topk_ids_ptr,
    #     'topk_weights_ptr': topk_weights_ptr,
    #     'topk': topk,
    #     'hidden_size': hidden_size,
    #     'BLOCK_SIZE': hidden_size,
    # }
    

    # pdb.set_trace()
    input_data = convert_tensor_with_device_type(input_data, device_type='npu')

    profiling_test(fn_triton, (grid, input_data))




if __name__ == '__main__':
    test_deepep_post_reorder_triton_kernel_npu()
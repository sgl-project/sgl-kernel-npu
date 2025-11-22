# Optimize the data loading and accumulation logic of this Triton operator: 
# remove the internal for-loop-based loading and implement batch loading and 
# batch computation for 2D matrices.

#source: python\sglang\srt\layers\moe\fused_moe_triton\fused_moe.py
import triton
import torch
import triton.language as tl
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

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)  # [BLOCK_DIM]

    offs_expert = tl.arange(0, topk_num)  # [topk_num]

    for token_index in range(token_start, token_end):
        # base pointer for this token: [hidden_dim]
        input_t_ptr = input_ptr + token_index * input_stride_0  # base of token

        # 2D pointers: [topk_num, BLOCK_DIM]
        # ptrs[i, d] = input_t_ptr + i * input_stride_1 + d
        ptrs_2d = (
            input_t_ptr[None, :] +                    # [1, BLOCK_DIM]
            (offs_expert[:, None] * input_stride_1) + # [topk_num, 1]
            offs_dim[None, :]                         # [1, BLOCK_DIM]
        )

        # [topk_num, BLOCK_DIM]
        mask_2d = (offs_dim < dim_end)[None, :]  

        tmps = tl.load(ptrs_2d, mask=mask_2d, other=0.0)  # [topk_num, BLOCK_DIM]

        accumulator = tl.sum(tmps, axis=0)  # float32 默认

        accumulator = accumulator * routed_scaling_factor

        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )

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
    _moe_sum_reduce_kernel_npu[grid](**input_data)

    # The following code is used for comparison with the original operator.
    #  _moe_sum_reduce_kernel[grid](**input_data)
    return 

# The calling scenario of this Triton operator remains unchanged. (sglang v0.4.8 /python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py)
def moe_sum_reduce_triton(
    input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 8

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )
    return


def test_moe_sum_reduce_kernel_npu():
    token_num = 128
    topk_num = 16 
    hidden_dim = 1024
    input_ptr = torch.rand(size=(token_num, topk_num, hidden_dim), dtype=torch.float32, device='cpu')

    input_stride_0 = input_ptr.stride(0)
    input_stride_1 = input_ptr.stride(1)
    input_stride_2 = input_ptr.stride(2)

    output_ptr = torch.zeros(size=(token_num, token_num), dtype=torch.float32, device='cpu')

    output_stride_0 = output_ptr.stride(0)
    output_stride_1 = output_ptr.stride(1)

    routed_scaling_factor = 0.5
    token_num, topk_num, hidden_dim = input_ptr.shape
    device = torch.npu.current_device()
    device_properties = driver.active.utils.get_device_properties(device)
    npu_num_core = device_properties["num_vectorcore"]
    BLOCK_M = triton.cdiv(token_num, npu_num_core)
    BLOCK_DIM = hidden_dim // 2
    NUM_STAGE = 1

    grid = (npu_num_core, 1, 1)
    input_data = {
        'input_ptr': input_ptr,
        'input_stride_0': input_stride_0,
        'input_stride_1': input_stride_1,
        'input_stride_2': input_stride_2,
        'output_ptr': output_ptr,
        'output_stride_0': output_stride_0,
        'output_stride_1': output_stride_1,
        'token_num': token_num,
        'topk_num': topk_num,
        'hidden_dim': hidden_dim,
        'routed_scaling_factor': routed_scaling_factor,
        'BLOCK_M': BLOCK_M,
        'BLOCK_DIM': BLOCK_DIM,
        'NUM_STAGE': NUM_STAGE

    }
    input_data = convert_tensor_with_device_type(input_data, device_type='npu')
    profiling_test(fn_triton, (grid, input_data))

if __name__ == '__main__':
    test_moe_sum_reduce_kernel_npu()

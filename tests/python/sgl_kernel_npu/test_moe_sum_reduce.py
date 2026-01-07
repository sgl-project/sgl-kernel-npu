# tests/test_moe_sum_reduce.py

import torch
import pytest
import triton
import triton.language as tl

from python.sgl_kernel_npu.moe.moe_sum_reduce import _moe_sum_reduce_kernel, _moe_sum_reduce_kernel_npu


def run_moe_sum_reduce(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    routed_scaling_factor: float,
    use_optimized_kernel: bool = True,
) -> torch.Tensor:
    """
    Unified launcher for moe sum-reduce kernels.

    Args:
        input_tensor: [token_num, topk_num, hidden_dim]
        output_tensor: [token_num, hidden_dim] (pre-allocated)
        routed_scaling_factor: float
        use_optimized_kernel: if True, use _moe_sum_reduce_kernel_npu;
                              else, use _moe_sum_reduce_kernel.

    Returns:
        output_tensor filled with result
    """
    token_num, topk_num, hidden_dim = input_tensor.shape
    assert output_tensor.shape == (token_num, hidden_dim)
    assert input_tensor.is_contiguous()
    assert output_tensor.is_contiguous()

    device = input_tensor.device
    dtype = input_tensor.dtype

    if use_optimized_kernel:
        import triton.runtime.driver as driver
        device_id = torch.npu.current_device()
        npu_num_core = driver.active.utils.get_device_properties(device_id)["num_vectorcore"]
        
        BLOCK_M = triton.cdiv(token_num, npu_num_core)
        BLOCK_DIM = min(triton.next_power_of_2(hidden_dim), 2048)
        grid = (npu_num_core,)

        _moe_sum_reduce_kernel_npu[grid](
            input_tensor,
            *input_tensor.stride(),
            output_tensor,
            *output_tensor.stride(),
            token_num=token_num,
            topk_num=topk_num,
            hidden_dim=hidden_dim,
            routed_scaling_factor=routed_scaling_factor,
            BLOCK_M=BLOCK_M,
            BLOCK_DIM=BLOCK_DIM,
            NUM_STAGE=1,
            total_tokens=token_num,
        )
    else:
        BLOCK_M = 1
        BLOCK_DIM = min(triton.next_power_of_2(hidden_dim), 2048)
        grid = (
            triton.cdiv(token_num, BLOCK_M),
            triton.cdiv(hidden_dim, BLOCK_DIM),
        )
        _moe_sum_reduce_kernel[grid](
            input_tensor,
            *input_tensor.stride(),
            output_tensor,
            *output_tensor.stride(),
            token_num=token_num,
            topk_num=topk_num,
            hidden_dim=hidden_dim,
            routed_scaling_factor=routed_scaling_factor,
            BLOCK_M=BLOCK_M,
            BLOCK_DIM=BLOCK_DIM,
            NUM_STAGE=1,
            num_warps=8,
        )

    return output_tensor


def _run_and_compare_kernels(
    input_tensor: torch.Tensor,
    routed_scaling_factor: float,
):
    token_num, topk_num, hidden_dim = input_tensor.shape
    device = input_tensor.device

    # Clone inputs to avoid in-place modification interference
    input_orig = input_tensor.clone()
    input_opt = input_tensor.clone()

    output_orig = torch.empty(token_num, hidden_dim, device=device, dtype=input_tensor.dtype)
    output_opt = torch.empty(token_num, hidden_dim, device=device, dtype=input_tensor.dtype)

    with torch.no_grad():
        run_moe_sum_reduce(input_orig, output_orig, routed_scaling_factor, use_optimized_kernel=False)
        run_moe_sum_reduce(input_opt, output_opt, routed_scaling_factor, use_optimized_kernel=True)

    torch.testing.assert_close(
        output_opt,
        output_orig,
        atol=1e-5 if input_tensor.dtype == torch.float32 else 1e-3,
        rtol=1e-3 if input_tensor.dtype == torch.float32 else 1e-2,
        msg=f"Kernel outputs differ! Shape={input_tensor.shape}, scale={routed_scaling_factor}"
    )


@pytest.mark.parametrize("token_num", [64, 256, 1024])
@pytest.mark.parametrize("topk_num", [4, 8, 16])
@pytest.mark.parametrize("hidden_dim", [256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_moe_sum_reduce_correctness(token_num, topk_num, hidden_dim, dtype):
    device = "npu"
    routed_scaling_factor = 0.5

    input_tensor = torch.randn(token_num, topk_num, hidden_dim, dtype=dtype, device=device)

    _run_and_compare_kernels(input_tensor, routed_scaling_factor)


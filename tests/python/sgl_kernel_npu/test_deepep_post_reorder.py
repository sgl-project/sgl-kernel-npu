import torch
import pytest
import triton.language as tl
import triton

from python.sgl_kernel_npu.moe.deepep_post_reorder import deepep_post_reorder_triton_kernel_npu, deepep_post_reorder_triton_kernel

def run_deepep_post_reorder(
    down_output: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    router_topk: int,
    use_optimized_kernel: bool = True,
) -> torch.Tensor:
    """
    Unified entry to call either the original or NPU-optimized kernel.

    Args:
        down_output: [total_routed_tokens, hidden_size]
        src2dst: [num_tokens * router_topk]
        topk_ids: [num_tokens, router_topk]
        topk_weights: [num_tokens, router_topk]
        router_topk: int
        use_optimized_kernel: if True, use deepep_post_reorder_triton_kernel_npu;
                              otherwise, use deepep_post_reorder_triton_kernel.

    Returns:
        output: [num_tokens, hidden_size]
    """
    num_tokens = src2dst.shape[0] // router_topk
    hidden_size = down_output.shape[1]
    device = down_output.device

    if num_tokens == 0:
        return torch.zeros((0, hidden_size), device=device, dtype=down_output.dtype)

    output = torch.empty((num_tokens, hidden_size), device=device, dtype=down_output.dtype)

    if use_optimized_kernel:
        # Use NPU-aware grid based on vector cores
        import triton.runtime.driver as driver
        device_id = torch.npu.current_device()
        npu_num_core = driver.active.utils.get_device_properties(device_id)["num_vectorcore"]
        grid = (npu_num_core,)
        deepep_post_reorder_triton_kernel_npu[grid](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            router_topk,
            hidden_size,
            BLOCK_SIZE=triton.next_power_of_2(hidden_size),
            total_tokens=num_tokens,
        )
    else:
        # Original per-token launch
        grid = (num_tokens,)
        deepep_post_reorder_triton_kernel[grid](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            router_topk,
            hidden_size,
            BLOCK_SIZE=triton.next_power_of_2(hidden_size),
        )

    return output

def _run_and_compare_kernels(
    down_output: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    router_topk: int,
):

    with torch.no_grad():
        output_orig = run_deepep_post_reorder(
            down_output=down_output.clone(),
            src2dst=src2dst.clone(),
            topk_ids=topk_ids.clone(),
            topk_weights=topk_weights.clone(),
            router_topk=router_topk,
            use_optimized_kernel=False,
        )

    with torch.no_grad():
        output_opt = run_deepep_post_reorder(
            down_output=down_output.clone(),
            src2dst=src2dst.clone(),
            topk_ids=topk_ids.clone(),
            topk_weights=topk_weights.clone(),
            router_topk=router_topk,
            use_optimized_kernel=True,
        )

    torch.testing.assert_close(
        output_opt,
        output_orig,
        atol=1e-5 if down_output.dtype == torch.float32 else 1e-3,
        rtol=1e-3 if down_output.dtype == torch.float32 else 1e-2,
        msg=f"Optimized kernel output differs from original!\n"
            f"Shape: {down_output.shape}, router_topk={router_topk}"
    )


@pytest.mark.parametrize("num_tokens", [64, 256, 1024, 2048])
@pytest.mark.parametrize("router_topk", [2, 4])
@pytest.mark.parametrize("hidden_size", [256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_deepep_post_reorder_correctness(num_tokens, router_topk, hidden_size, dtype):
    device = "npu"

    total_routed_tokens = num_tokens * router_topk  # worst-case: all valid

    # Input tensors
    down_output = torch.randn(total_routed_tokens, hidden_size, dtype=dtype, device=device)
    src2dst = torch.randint(-1, total_routed_tokens, (num_tokens * router_topk,), device=device)
    # Ensure first expert per token is valid to avoid all-zero outputs
    for i in range(num_tokens):
        src2dst[i * router_topk] = i  # valid index

    topk_ids = torch.randint(0, 64, (num_tokens, router_topk), device=device)
    topk_weights = torch.rand(num_tokens, router_topk, dtype=torch.float32, device=device)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # normalize

    _run_and_compare_kernels(
        down_output=down_output,
        src2dst=src2dst,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        router_topk=router_topk,
    )


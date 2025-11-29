# tests/test_decode_softmax_reducev.py

import torch
import pytest
import triton

from python.sgl_kernel_npu.attention._fwd_kernel_stage2 import _fwd_kernel_stage2, _fwd_kernel_stage2_npu

def run_decode_softmax_reducev(
    Mid_O: torch.Tensor,
    Mid_O_1: torch.Tensor,
    O: torch.Tensor,
    kv_indptr: torch.Tensor,
    num_kv_splits: torch.Tensor,
    sink_ptr: torch.Tensor,
    min_block_kv: int,
    has_sink: bool,
    use_optimized_kernel: bool = True,
) -> torch.Tensor:
    """
    Unified launcher for stage2 softmax-reduce kernels.

    Args:
        Mid_O: [batch, num_heads, max_kv_splits, Lv]
        Mid_O_1: [batch, num_heads, max_kv_splits]
        O: [batch, num_heads, Lv] (output, pre-allocated)
        kv_indptr: [batch + 1]
        num_kv_splits: [batch]
        sink_ptr: [num_heads] or None
        min_block_kv: int
        has_sink: bool
        use_optimized_kernel: bool

    Returns:
        Filled O tensor
    """
    batch, num_heads, max_kv_splits, Lv = Mid_O.shape
    assert O.shape == (batch, num_heads, Lv)
    assert Mid_O_1.shape == (batch, num_heads, max_kv_splits)
    assert kv_indptr.shape[0] == batch + 1
    assert num_kv_splits.shape[0] == batch
    if has_sink:
        assert sink_ptr.shape[0] == num_heads

    device = Mid_O.device
    dtype = Mid_O.dtype

    stride_mid_ob, stride_mid_oh, stride_mid_os, _ = Mid_O.stride()
    stride_obs, stride_oh, _ = O.stride()

    if use_optimized_kernel:
        import triton.runtime.driver as driver
        device_id = torch.npu.current_device()
        npu_num_core = driver.active.utils.get_device_properties(device_id)["num_vectorcore"]

        # BATCH_GROUP_SIZE = (batch + npu_num_core - 1) // npu_num_core
        grid = (npu_num_core,)

        _fwd_kernel_stage2_npu[grid](
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
            MAX_KV_SPLITS=max_kv_splits,
            MIN_BLOCK_KV=min_block_kv,
            BLOCK_DV=triton.next_power_of_2(Lv),
            Lv=Lv,
            HAS_SINK=has_sink,
            total_batch=batch,
            num_heads=num_heads,
        )
    else:
        grid = (batch, num_heads)
        _fwd_kernel_stage2[grid](
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
            MAX_KV_SPLITS=max_kv_splits,
            MIN_BLOCK_KV=min_block_kv,
            BLOCK_DV=triton.next_power_of_2(Lv),
            Lv=Lv,
            HAS_SINK=has_sink,
        )

    return O

def _run_and_compare_kernels(
    batch: int,
    num_heads: int,
    max_kv_splits: int,
    Lv: int,
    min_block_kv: int = 32,
    has_sink: bool = False,
    dtype: torch.dtype = torch.float32,
):
    device = "npu"

    # Generate inputs following the original logic
    Mid_O = torch.randn(batch, num_heads, max_kv_splits, Lv, dtype=dtype, device=device)
    Mid_O_1 = torch.randn(batch, num_heads, max_kv_splits, dtype=dtype, device=device)

    output_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
    O_orig = torch.empty(batch, num_heads, Lv, dtype=output_dtype, device=device)
    O_opt = torch.empty_like(O_orig)

    # Original logic: kv_indptr is sequential [0, 1, 2, ..., batch]
    kv_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=device)
    
    # Original logic: num_kv_splits is fixed for all batches (e.g., 8 in the original)
    fixed_splits = min(8, max_kv_splits)  # Use 8 or max_kv_splits, whichever is smaller
    num_kv_splits = torch.full((batch,), fixed_splits, dtype=torch.int32, device=device)

    sink_ptr = torch.randn(num_heads, dtype=dtype, device=device) if has_sink else torch.zeros(num_heads, device=device)

    with torch.no_grad():
        run_decode_softmax_reducev(
            Mid_O.clone(), Mid_O_1.clone(), O_orig, kv_indptr, num_kv_splits,
            sink_ptr, min_block_kv, has_sink, use_optimized_kernel=False
        )
        run_decode_softmax_reducev(
            Mid_O.clone(), Mid_O_1.clone(), O_opt, kv_indptr, num_kv_splits,
            sink_ptr, min_block_kv, has_sink, use_optimized_kernel=True
        )

    # Convert to same dtype for comparison
    O_orig_f32 = O_orig.to(torch.float32)
    O_opt_f32 = O_opt.to(torch.float32)

    torch.testing.assert_close(
        O_opt_f32,
        O_orig_f32,
        atol=1e-4 if dtype == torch.float32 else 5e-3,
        rtol=1e-3 if dtype == torch.float32 else 1e-2,
        msg=f"Kernel mismatch! batch={batch}, heads={num_heads}, Lv={Lv}, splits={max_kv_splits}, dtype={dtype}"
    )

@pytest.mark.parametrize("batch", [32, 128, 240])
@pytest.mark.parametrize("num_heads", [8,16])
@pytest.mark.parametrize("Lv", [128, 256])
@pytest.mark.parametrize("max_kv_splits", [8, 16])
@pytest.mark.parametrize("has_sink", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_decode_softmax_reducev_correctness(batch, num_heads, Lv, max_kv_splits, has_sink, dtype):
    _run_and_compare_kernels(
        batch=batch,
        num_heads=num_heads,
        max_kv_splits=max_kv_splits,
        Lv=Lv,
        min_block_kv=32,
        has_sink=has_sink,
        dtype=dtype,
    )

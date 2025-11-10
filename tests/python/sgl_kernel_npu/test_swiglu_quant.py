import torch
import torch_npu
import numpy as np
import triton
import triton.language as tl
# from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _swiglu_quant_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALGIN: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    SCALE: tl.constexpr,
):
    # calc real total_rows
    if GROUP_LIST_TYPE == 0:  # cusum
        total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
    else:
        gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
        gl_mask = gl_offsets < NUM_EXPERTS
        group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(tl.int32)
        total_rows = tl.sum(group_list)

    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    for row_idx in range(row_begin, row_end):
        # swiglu
        x_offsets = row_idx * TOTAL_COLS + tl.arange(0, TOTAL_COLS)
        cur_x = tl.load(x_ptr + x_offsets)
        x1 = tl.extract_slice(cur_x, offsets=(0,), sizes=(HALF_COLS,), strides=(1,))
        x2 = tl.extract_slice(
            cur_x, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,)
        )
        out = x1 * tl.sigmoid(x1) * x2

        # quant
        if SCALE:
            scale = tl.max(tl.abs(out)).to(tl.float32) / DTYPE_MAX
            # store scale
            tl.store(scale_ptr + row_idx, scale.to(scale_ptr.dtype.element_ty))
            # out = tl.math.rint(out / scale) # scale.reshape(SUB_BLOCK_SIZE, 1))   # ub overflow
            # tl.store(out_ptr + row_idx * TOTAL_COLS, out)
            for col_blk_idx in range(0, HALF_COLS, COL_BLOCK_SIZE):
                col_indices = col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)
                valid_mask = col_indices < HALF_COLS
                tmp_out = tl.extract_slice(
                    out, offsets=(col_blk_idx,), sizes=(COL_BLOCK_SIZE,), strides=(1,)
                )
                tmp_out = (tmp_out.to(tl.float32) / scale).to(x_ptr.dtype.element_ty)
                # tmp_out = tl.clamp(tmp_out, -128, 127)
                tmp_out = tl.math.rint(tmp_out)

                o_offsets = (
                    row_idx * HALF_COLS + col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)
                )
                tl.store(out_ptr + o_offsets, tmp_out.to(out_ptr.dtype.element_ty), mask=valid_mask)
        else:
            # store out
            o_offsets = row_idx * HALF_COLS + tl.arange(0, HALF_COLS)
            tl.store(out_ptr + o_offsets, out.to(out_ptr.dtype.element_ty))

def get_device_properties():
    device = torch.npu.current_device()
    device_properties = (
        triton.runtime.driver.active.utils.get_device_properties(device)
    )

    num_aicore = device_properties.get("num_aicore", -1)
    num_vectorcore = device_properties.get("num_vectorcore", -1)

    return num_aicore, num_vectorcore


_, num_vectorcore = get_device_properties()


def swiglu_quant(x, group_list, group_list_type, need_quant=True):
    # group_list_type must be 0 cusum or 1 count
    if group_list_type not in [0, 1]:
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    s, h = x.shape
    out_dtype = torch.int8 if need_quant else x.dtype
    out = torch.empty((s, h // 2), dtype=out_dtype, device=x.device)
    scale = torch.empty((s,), dtype=torch.float32, device=x.device)
    num_experts = group_list.shape[0]
    # ub must be 32-byte aligned on npu
    if group_list.dtype == torch.int64:
        num_experts_algin = (num_experts + 7) // 8 * 8
    elif group_list.dtype == torch.int32:
        num_experts_algin = (num_experts + 15) // 16 * 16
    else:
        raise ValueError(
            f"group_list dtype must be torch.int32 or torch.int64, but got {group_list.dtype}"
        )

    # _, num_vectorcore = get_device_properties()
    _swiglu_quant_kernel[(num_vectorcore,)](
        x,
        group_list,
        out,
        scale,
        TOTAL_COLS=h,
        HALF_COLS=h // 2,
        COL_BLOCK_SIZE=1536,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_ALGIN=num_experts_algin,
        GROUP_LIST_TYPE=group_list_type,
        NUM_CORES=num_vectorcore,
        DTYPE_MAX=127,
        SCALE=need_quant,
        multibuffer=True,
    )
    return out, scale

def test_swiglu_quant():
    def to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    # create inputs
    s, h = 4096, 3072
    x = torch.randn((s, h), dtype=torch.bfloat16).npu()
    group_list = (
        torch.Tensor([0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0])
        .npu()
        .to(torch.int64)
    )
    # torch native
    swglu_out = torch_npu.npu_swiglu(x)
    ans1, ans2 = torch_npu.npu_dynamic_quant(swglu_out)
    # fused_triton_kernel
    res1, res2 = swiglu_quant(x, group_list, group_list_type=1)

    print(f"swglu_out: {swglu_out.shape}, ans1 : {ans1.shape}, res1: {res1.shape}")

    real_tokens = torch.sum(group_list)
    diff = res1[:real_tokens, :] - ans1[:real_tokens, :]
    print(f"trition output: {res1[:real_tokens, :]}")
    print(f"output_golden:{ans1[:real_tokens, :]}")
    print(f"diff= {diff.shape}")
    print(f"diff= {diff}")

    num_different = (res1[:real_tokens, :] != ans1[:real_tokens, :]).sum().item()
    print(f"num_different: {num_different}")
    max_diff = torch.max(torch.abs(diff))
    print(f"max_diff: {max_diff}")
    assert max_diff <= 1

    diff_rate = torch.sum(torch.abs(diff)) / (real_tokens * h // 2)
    assert diff_rate < 2e-2

    assert (
        np.testing.assert_allclose(
            to_numpy(res2[:real_tokens]),
            to_numpy(ans2[:real_tokens]),
            rtol=5e-3,
        )
        is None
    )


if __name__ == "__main__":
    test_swiglu_quant()
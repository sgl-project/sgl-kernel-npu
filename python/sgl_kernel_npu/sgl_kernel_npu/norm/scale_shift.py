import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.autotune(
    configs=[
        triton.Config(
            {"block_l": 128, "block_c": 128},
        ),
        triton.Config(
            {"block_l": 112, "block_c": 128},
        ),
    ],
    key=["num_tokens", "hidden_size"],
)
@triton.jit
def fused_scale_shift_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    output_ptr,
    num_tokens,
    hidden_size,
    scale_numel: tl.constexpr,
    shift_numel: tl.constexpr,
    block_l: tl.constexpr,
    block_c: tl.constexpr,
    kernel_num: tl.constexpr,
):
    pid = tl.program_id(0)
    row_tasks = tl.cdiv(num_tokens, block_l)
    col_tasks = tl.cdiv(hidden_size, block_c)
    total_tasks = row_tasks * col_tasks

    for task_id in range(pid, total_tasks, kernel_num):
        row_pid = task_id // col_tasks
        col_pid = task_id % col_tasks

        token_offsets = row_pid * block_l + tl.arange(0, block_l)
        dim_offsets = col_pid * block_c + tl.arange(0, block_c)

        mask = (token_offsets[:, None] < num_tokens) & (dim_offsets[None, :] < hidden_size)
        offset = token_offsets[:, None] * hidden_size + dim_offsets[None, :]

        x = tl.load(x_ptr + offset, mask=mask, other=0.0)

        if scale_numel == 1:
            scale = tl.load(scale_ptr)
        else:
            scale_offsets = dim_offsets[None, :]
            scale_mask = dim_offsets[None, :] < hidden_size
            scale = tl.load(scale_ptr + scale_offsets, mask=scale_mask, other=0.0)

        if shift_numel == 1:
            shift = tl.load(shift_ptr)
        else:
            shift_offsets = dim_offsets[None, :]
            shift_mask = dim_offsets[None, :] < hidden_size
            shift = tl.load(shift_ptr + shift_offsets, mask=shift_mask, other=0.0).to(
                tl.float32
            )

        output = x * (1.0 + scale) + shift

        tl.store(output_ptr + offset, output.to(output_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"block_l": 96, "block_c": 128},
        ),
        triton.Config(
            {"block_l": 80, "block_c": 128},
        ),
        triton.Config(
            {"block_l": 64, "block_c": 128},
        ),
    ],
    key=["num_tokens", "hidden_size"],
)
@triton.jit
def fused_scale_shift_kernel_2(
    x_ptr,
    scale_ptr,
    shift_ptr,
    output_ptr,
    num_tokens,
    hidden_size,
    scale_constant: tl.constexpr,
    block_l: tl.constexpr,
    block_c: tl.constexpr,
    kernel_num: tl.constexpr,
):
    pid = tl.program_id(0)
    row_tasks = tl.cdiv(num_tokens, block_l)
    col_tasks = tl.cdiv(hidden_size, block_c)
    total_tasks = row_tasks * col_tasks

    for task_id in range(pid, total_tasks, kernel_num):
        row_pid = task_id // col_tasks
        col_pid = task_id % col_tasks

        token_offsets = row_pid * block_l + tl.arange(0, block_l)
        dim_offsets = col_pid * block_c + tl.arange(0, block_c)

        mask = (token_offsets[:, None] < num_tokens) & (
            dim_offsets[None, :] < hidden_size
        )
        offset = token_offsets[:, None] * hidden_size + dim_offsets[None, :]

        x = tl.load(x_ptr + offset, mask=mask, other=0.0)

        scale_offsets = dim_offsets[None, :]
        scale_mask = dim_offsets[None, :] < hidden_size
        scale = tl.load(scale_ptr + scale_offsets, mask=scale_mask, other=0.0)

        shift = tl.load(shift_ptr + offset, mask=mask, other=0.0).to(tl.float32)

        output = x * (scale_constant + scale) + shift

        tl.store(output_ptr + offset, output.to(output_ptr.dtype.element_ty), mask=mask)


def fused_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_constant: float = 1.0,
):
    orig_shape = x.shape
    num_tokens = orig_shape[0] * orig_shape[1]
    hidden_size = orig_shape[2]
    x_numel = num_tokens * hidden_size

    scale = scale.view(-1)
    shift = shift.view(-1)

    scale_numel = scale.numel()
    shift_numel = shift.numel()

    assert (
        scale_numel == 1 or scale_numel == hidden_size
    ), f"Scale size must be 1 or {hidden_size}, got {scale_numel}"
    assert (
        shift_numel == 1 or shift_numel == hidden_size or shift_numel == x_numel
    ), f"Scale size must be 1, {hidden_size} or {x_numel}, got {shift_numel}"

    output = torch.empty_like(x)

    kernel_num = get_device_properties()[1]
    grid = (kernel_num,)

    if shift_numel == x_numel:
        fused_scale_shift_kernel_2[grid](
            x,
            scale,
            shift,
            output,
            num_tokens,
            hidden_size,
            scale_constant,
            kernel_num=kernel_num,
        )

    else:
        fused_scale_shift_kernel[grid](
            x,
            scale,
            shift,
            output,
            num_tokens,
            hidden_size,
            scale_numel=scale_numel,
            shift_numel=shift_numel,
            kernel_num=kernel_num,
        )

    return output

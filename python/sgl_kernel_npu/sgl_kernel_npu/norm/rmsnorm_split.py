import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.autotune(
    configs=[
        triton.Config(
            {"block_l": 256, "block_c": 256},
        ),
        triton.Config(
            {"block_l": 128, "block_c": 256},
        ),
        triton.Config(
            {"block_l": 128, "block_c": 128},
        ),
        triton.Config(
            {"block_l": 64, "block_c": 128},
        ),
        triton.Config(
            {"block_l": 64, "block_c": 64},
        ),
        triton.Config(
            {"block_l": 64, "block_c": 32},
        ),
        triton.Config(
            {"block_l": 32, "block_c": 32},
        ),
    ],
    key=["num_tokens", "hidden_size"],
)
@triton.jit
def fused_rsqrt_mul_kernel(
    x_ptr,
    variance_ptr,
    weight_ptr,
    eps,
    output_ptr,
    num_tokens,
    hidden_size,
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
        offset = token_offsets[:, None] * hidden_size + dim_offsets[None, :]

        mask_token = token_offsets < num_tokens
        mask_dim = dim_offsets < hidden_size
        mask = mask_token[:, None] & mask_dim[None, :]

        x = tl.load(x_ptr + offset, mask=mask, other=0.0)
        variance = tl.load(
            variance_ptr + token_offsets[:, None], mask=mask_token[:, None], other=0.0
        )
        weight = tl.load(weight_ptr + dim_offsets, mask=mask_dim, other=0.0)

        rsqrt = tl.math.rsqrt(variance + eps)
        output = x * rsqrt * weight
        tl.store(output_ptr + offset, output, mask=mask)


def fused_rsqrt_mul(x, variance, weight, eps=1e-6):
    _, kernel_num = get_device_properties()
    B, L, C = x.shape[0], x.shape[1], x.shape[2]
    grid = (kernel_num,)

    output = torch.empty_like(x)

    fused_rsqrt_mul_kernel[grid](
        x,
        variance,
        weight,
        eps,
        output,
        B * L,
        C,
        kernel_num=kernel_num,
    )

    return output


@triton.autotune(
    configs=[
        triton.Config(
            {"block_l": 96},
        ),
        triton.Config(
            {"block_l": 64},
        ),
        triton.Config(
            {"block_l": 32},
        ),
        triton.Config(
            {"block_l": 16},
        ),
        triton.Config(
            {"block_l": 8},
        ),
        triton.Config(
            {"block_l": 4},
        ),
        triton.Config(
            {"block_l": 2},
        ),
        triton.Config(
            {"block_l": 1},
        ),
    ],
    key=["num_tokens"],
)
@triton.jit
def fused_variance_kernel(
    x_ptr,
    output_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    block_l: tl.constexpr,
    kernel_num: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tasks = tl.cdiv(num_tokens, block_l)

    for task_id in range(pid, total_tasks, kernel_num):
        token_offsets = task_id * block_l + tl.arange(0, block_l)
        dim_offsets = tl.arange(0, hidden_size)

        offset = token_offsets[:, None] * hidden_size + dim_offsets[None, :]
        mask_out = token_offsets[:, None] < num_tokens
        mask = mask_out & (dim_offsets[None, :] < hidden_size)

        x = tl.load(x_ptr + offset, mask=mask, other=0.0)

        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=1)
        variance = sum_sq / hidden_size

        tl.store(output_ptr + token_offsets[:, None], variance[:, None], mask=mask_out)


def fused_variance(x: torch.Tensor):
    _, kernel_num = get_device_properties()
    B, L, C = x.shape[0], x.shape[1], x.shape[2]
    grid = (kernel_num,)

    output = torch.empty((B, L, 1), device=x.device, dtype=x.dtype)

    fused_variance_kernel[grid](x, output, B * L, C, kernel_num=kernel_num)
    return output

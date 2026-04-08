import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.autotune(
    configs=[
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
    ],
    key=["num_tokens"],
)
@triton.jit
def fused_rmsnorm_without_weight_kernel(
    x_ptr,
    eps,
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

        offsets = token_offsets[:, None] * hidden_size + dim_offsets[None, :]
        mask = (token_offsets[:, None] < num_tokens) & (dim_offsets[None, :] < hidden_size)

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        x_sq = x * x
        var = tl.sum(x_sq, axis=1) * (1 / hidden_size)
        rstd = tl.math.rsqrt(var + eps)

        y = x * rstd[:, None]
        tl.store(output_ptr + offsets, y, mask=mask)


def fused_rmsnorm_without_weight(x, eps):
    _, num_vectorcore = get_device_properties()
    B, L, C = x.shape[0], x.shape[1], x.shape[2]
    grid = (num_vectorcore,)
    output = torch.empty_like(x)

    fused_rmsnorm_without_weight_kernel[grid](
        x,
        eps,
        output,
        B * L,
        C,
        kernel_num=num_vectorcore,
    )

import triton
import triton.language as tl


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

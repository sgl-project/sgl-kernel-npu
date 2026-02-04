import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties
from torch.library import triton_op, wrap_triton


_, num_vectorcore = get_device_properties()

@triton.jit
def l1_norm_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    hidden_size: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    block_size = (batch_size - 1) // NUM_CORES + 1
    row_begin = pid * block_size
    row_end = tl.minimum((pid + 1) * block_size, batch_size)
    for row_idx in range(row_begin, row_end):
        cols = tl.arange(0, hidden_size)
        buffered_values = tl.load(input_ptr + row_idx * hidden_size + cols)
        buffered_values /= tl.sum(buffered_values)
        tl.store(
            output_ptr + row_idx * hidden_size + cols, buffered_values.to(tl.float32)
        )

@triton_op("sgl_kernel_npu::l1_norm",mutates_args={})
def l1_norm(input: torch.Tensor) -> torch.Tensor:
    batch_size = input.shape[0]
    hidden_size = input.shape[1]
    output = torch.empty(
        batch_size, hidden_size, device=input.device, dtype=torch.float32
    )

    wrap_triton(l1_norm_kernel)[(num_vectorcore,)](
        input, output, batch_size, hidden_size, num_vectorcore
    )
    return output

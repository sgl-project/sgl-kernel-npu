import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def mul_add_kernel(
    input1_ptr,
    input2_ptr,
    output_ptr,
    factor: tl.constexpr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    valid_mask = cols < hidden_size
    input_offsets = row_start * hidden_size + cols
    for row_idx in tl.range(row_start, batch_size, row_step):
        routed_values = tl.load(input1_ptr + input_offsets, mask=valid_mask, other=0.0)
        shared_values = tl.load(input2_ptr + input_offsets, mask=valid_mask, other=0.0)
        buffered_values = routed_values * factor + shared_values
        tl.store(
            output_ptr + row_idx * hidden_size + cols,
            buffered_values,
            mask=valid_mask,
        )

        input_offsets += row_step * hidden_size


def mul_add(
    routed_input,
    shared_input,
    scaling_factor,
):
    _, num_vectorcore = get_device_properties()

    batch_size = routed_input.shape[0]
    hidden_size = routed_input.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    n_rows = min(batch_size, num_vectorcore)

    output = torch.empty(
        batch_size, hidden_size, device=routed_input.device, dtype=routed_input.dtype
    )

    mul_add_kernel[(n_rows, 1, 1)](
        routed_input,
        shared_input,
        output,
        scaling_factor,
        batch_size,
        hidden_size,
        BLOCK_SIZE,
    )
    return output

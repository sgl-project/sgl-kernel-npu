import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def add_rmsnorm_bias_kernel(
    input_ptr,
    residual_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    quant_scale_ptr,
    quant_offset_ptr,
    output_ptr,
    output2_ptr,
    batch_size,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    valid_mask = cols < hidden_size
    norm_weight_values = tl.load(norm_weight_ptr + cols, mask=valid_mask, other=0.0)
    input_offsets = row_start * hidden_size + cols
    for i in tl.range(row_start, batch_size, row_step):
        # add
        buffered_values = tl.load(input_ptr + input_offsets, mask=valid_mask, other=0.0)
        buffered_values += tl.load(residual_ptr + input_offsets, mask=valid_mask, other=0.0)
        tl.store(output2_ptr + input_offsets, buffered_values, mask=valid_mask)
        buffered_values = buffered_values.to(tl.float32)
        # norm
        squares = (buffered_values * buffered_values)  # .to(tl.float32)
        variance = tl.sum(squares) / hidden_size
        reciprocal_std = (1 / tl.sqrt(variance + eps))  # .to(tl.bfloat16)
        buffered_values = buffered_values * reciprocal_std
        buffered_values = buffered_values * norm_weight_values
        # bias
        norm_bias_values = tl.load(norm_bias_ptr + cols, mask=valid_mask, other=0.0)
        buffered_values = buffered_values + norm_bias_values
        if SCALE:
            block_cols = tl.arange(0, COL_BLOCK_SIZE)
            for block_offset in range(0, hidden_size, COL_BLOCK_SIZE):
                col_indices = block_offset + block_cols
                valid_mask2 = col_indices < hidden_size
                block_buffered_values = tl.extract_slice(buffered_values, (block_offset,), (COL_BLOCK_SIZE,), (1,))
                # quant
                quant_scale_values = tl.load(quant_scale_ptr + col_indices, mask=valid_mask2, other=0.0)
                quant_offset_values = tl.load(quant_offset_ptr + col_indices, mask=valid_mask2, other=0.0)
                block_buffered_values = (block_buffered_values.to(tl.float32) * quant_scale_values) + quant_offset_values
                block_buffered_values = tl.math.rint(block_buffered_values)
                tl.store(output_ptr + i * hidden_size + col_indices, block_buffered_values, mask=valid_mask2)
        else:
            tl.store(output_ptr + input_offsets, buffered_values, mask=valid_mask)

        input_offsets += row_step * hidden_size


kernels = {}


def add_rmsnorm_bias(
    input,
    residual,
    norm_weight,
    norm_bias,
    eps,
    quant_scale=None,
    quant_offset=None,
):
    _, num_vectorcore = get_device_properties()

    batch_size = input.shape[0]
    hidden_size = input.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    COL_BLOCK_SIZE = 2048
    n_rows = min(batch_size, num_vectorcore)

    SCALE = quant_scale is not None
    if SCALE:
        output = torch.empty(batch_size, hidden_size, device=input.device, dtype=torch.int8)
    else:
        output = torch.empty(batch_size, hidden_size, device=input.device, dtype=input.dtype)
    output2 = torch.empty(batch_size, hidden_size, device=input.device, dtype=input.dtype)
    kernel = kernels.get((n_rows, BLOCK_SIZE, COL_BLOCK_SIZE, SCALE), None)
    if kernel is None:
        kernel = add_rmsnorm_bias_kernel.warmup(input, residual, norm_weight,
                    norm_bias, quant_scale, quant_offset, output, output2, batch_size, hidden_size, eps, BLOCK_SIZE,
                    COL_BLOCK_SIZE, SCALE, grid=(n_rows,))
        kernel._init_handles()
        kernels[(n_rows, BLOCK_SIZE, COL_BLOCK_SIZE, SCALE)] = kernel

    kernel[(n_rows, 1, 1)](
        input, residual, norm_weight, norm_bias, quant_scale, quant_offset, output, output2,
        batch_size, hidden_size, eps,
    )
    return output, output2
import torch
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(input_ptr, output_ptr, hidden_size: tl.constexpr):
    pid = tl.program_id(0)
    input_ptr += pid * hidden_size
    output_ptr += pid * hidden_size
    cols = tl.arange(0, hidden_size)
    buffered_values = tl.load(input_ptr + cols)
    buffered_values /= tl.sum(buffered_values)
    tl.store(output_ptr + cols, buffered_values.to(tl.bfloat16))


def l1_norm(input):
    batch_size = input.shape[0]
    hidden_size = input.shape[1]
    output = torch.empty(batch_size, hidden_size, device=input.device, dtype=input.dtype)

    l1_norm_kernel[(batch_size,)](
        input, output, hidden_size
    )
    return output
import torch
from sgl_kernel_npu.implementation.triton.norm.l1_norm import l1_norm_kernel
from sgl_kernel_npu.utils.triton_utils import get_device_properties


def l1_norm(input, valid_rows=None):
    batch_size = input.shape[0]
    hidden_size = input.shape[1]
    output = torch.empty(
        batch_size, hidden_size, device=input.device, dtype=torch.float32
    )
    _, num_vectorcore = get_device_properties()
    l1_norm_kernel[(num_vectorcore,)](
        input, output, batch_size, hidden_size, num_vectorcore, valid_rows
    )
    return output

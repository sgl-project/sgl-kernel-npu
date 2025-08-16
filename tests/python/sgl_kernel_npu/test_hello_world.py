import torch
import torch_npu

import sgl_kernel_npu

torch.ops.npu.sgl_kernel_npu_print_version()

x = torch.tensor([1], device='npu')
y = torch.tensor([2], device='npu')
print("hellword: ", torch.ops.npu.helloworld(x, y))
## Introduction
Matrix matmul enable einsum, [m, b, n] = [m, b, k] * [b, k, n]

## Sheet 1: Parameters
| Parameter    | Dimension                                     | Data Type     | Format | Description                                      |
|--------------|----------------------------------------------|---------------|--------|--------------------------------------------------|
| tensor_a     | [m, batch, k]                                | float16/bf16  | ND     | Matrix A for matrix multiplication.             |
| tensor_b     | ND: [batch, k, n]<br>NZ: [batch, n/16, k, 16] | float16/bf16  | ND/NZ  | Matrix B for matrix multiplication, weights.<br>When dimension is 4D, both k and n values must be multiples of 16. |
| format_mode  | /                                            | string        | /      | ND/NZ, default ND                               |
| tensor_c     | [m, batch, n]                                | float16/bf16  | ND     | Result of matrix multiplication, assigned back by reference. |


## Restrictions
1. m <= 1024.
2. Only support Ascend A2/A3.
3. The dtype of tensor_a and tensor_b must be the same and is float16 or bfloat16.
4. The dim1 of tensor_a must equal to dim0 of tensor_b.


## Sample Code
```python
import torch
import torch_npu
import sgl_kernel_npu

dtype = torch.float16
b = 128
m = 8
k = 128
n = 1024

tensor_a = torch.randn(m, b, k, dtype=dtype, device="npu")
tensor_b = torch.randn(b, k, n, dtype=dtype, device="npu")
res = torch.empty((m, b, n), dtype=dtype, device="npu")

torch.ops.npu.batch_matmul_transpose(tensor_a, b_tensor, res)
```

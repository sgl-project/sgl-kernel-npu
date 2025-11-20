# torch.ops.catlass_matmul_basic


## Function Description | 功能描述

### English:
This is the catlass version matmul kernel  `catlass_matmul_basic` kernel function

### 中文:
这是调用catlass模板库实现的矩阵乘法运算 `catlass_matmul_basic`内核方法

参考/Refs: [CATLSS](https://gitcode.com/cann/catlass)


## Interface Prototype | 接口原型

### Python Binding Definition
```python
import sgl_kernel_npu

torch.ops.npu.catlass_matmul_basic(
    input_a: torch.Tensor,        # bf16/fp16, [m, k]
    input_b: torch.Tensor,        # bf16/fp16, [k, n]
    output_c: torch.Tensor        # bf16/fp16, [m, n]
) -> None
```

### Kernel Definition | 核函数定义
```C++
extern "C" __global__ __aicore__ void catlass_matmul_basic(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC,
                                                           GM_ADDR gmWorkspace, GM_ADDR gmTiling)
```

## Parameter Description | 参数说明

| Parameter Name (参数名称) | DataType (数据类型) | Description                          | 说明            |
|:----------------------|:----------------|:-------------------------------------|:--------------|
| `input_a`         | `torch.Tensor`  | input left tensor with shape (m, k)  | 左输入矩阵，（m,k)大小 |
| `input_b`      | `torch.Tensor`  | input right tensor with shape (k, n) | 右输入矩阵，（k,n)大小 |


## Output Description | 输出说明

| Parameter Name (参数名称)  | DataType (数据类型) | Description                     | 说明            |
|:-----------------------|:----------------|:--------------------------------|:--------------|
| `output_c`    | `torch.Tensor`  | output tensor with shape (m, n) | 输出矩阵，（m, n）大小 |


## Constraints | 约束说明

### English:
`WeightFormatMode.WEIGHT_NZ` is not implemented

### 中文:
`WeightFormatMode.WEIGHT_NZ` 暂未实现

## Example | 调用示例

```python
import math
import sgl_kernel_npu
import torch
import torch_npu

device = torch.device('npu:0')

dtypes = [torch.float16, torch.bfloat16]
m, k, n = 128, 256, 256

for dtype in dtypes:
    a = torch.rand(m, k, dtype=dtype, device="npu")
    b = torch.rand(k, n, dtype=dtype, device="npu")
    res = torch.empty((m, n), dtype=dtype, device="npu")

    torch.ops.npu.catlass_matmul_basic(a, b, res)
```

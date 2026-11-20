# transpose_quant_batchmatmul

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | Unsupported | Supported |
| `torch_native` | Unsupported | Supported |

## Signature

`auto` selects `ops-nn` wherever it is registered.

```python
from sgl_kernel_npu.operators.matmul import transpose_quant_batchmatmul


def transpose_quant_batchmatmul(x1, x2, dtype, *, impl="auto", **kwargs): ...
```

## Description

Permutes the axes requested by the caller and computes a scaled quantized matrix product independently for each batch. With permutations (1,0,2), (0,1,2), (1,0,2) the output is token-major [M,B,N].

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `x1`, `x2` | Tensor | [M, B, K], [B, K, N] for the example, FP8 | Required | Quantized operands. The contracting and batch dimensions must match after the requested permutations. |
| `dtype` | torch.dtype | - | Required | Output dtype, such as torch.float16. |
| `x1_scale`, `x2_scale` | Tensor or None | [M], [N], FP32 in the per-row and per-column case | `None` | Optional input scales. Grouped recipes have different required shapes. |
| `perm_x1`, `perm_x2`, `perm_y` | tuple[int, int, int] | - | `None` | Optional permutations for the left operand, right operand and result. |
| `group_sizes` | list[int] or None | - | `None` | Optional SDK quantization groups. |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | [M, B, N] for the example, dtype | Batched product after the output permutation. |

## ops-nn backend

Calls `torch_npu.npu_transpose_quant_batchmatmul`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_transpose_quant_batchmatmul.md) [CANN ops-nn source](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/matmul/transpose_quant_batch_mat_mul)

## torch_native backend

Computes MXFP8 batched multiplication with E8M0 scales for groups of 32, including the grouped WoA layout used by DeepSeek V4.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.matmul import transpose_quant_batchmatmul

m, batch, k, n = 32, 32, 512, 128


def fp8(shape):
    return torch.randint(-4, 5, shape, dtype=torch.int8).to(torch.float8_e4m3fn).npu()


a = fp8((m, batch, k))
b = fp8((batch, k, n))
s1 = torch.ones((m,), dtype=torch.float32, device=a.device)
s2 = torch.ones((n,), dtype=torch.float32, device=a.device)
y = transpose_quant_batchmatmul(
    a,
    b,
    torch.float16,
    x1_scale=s1,
    x2_scale=s2,
    perm_x1=(1, 0, 2),
    perm_x2=(0, 1, 2),
    perm_y=(1, 0, 2),
    impl="ops-nn",
)
```

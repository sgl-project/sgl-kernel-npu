# dual_level_quant_matmul

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Unsupported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.matmul import dual_level_quant_matmul


def dual_level_quant_matmul(
    x1,
    x2,
    x1_level0_scale,
    x2_level0_scale,
    x1_level1_scale,
    x2_level1_scale,
    *,
    impl="auto",
    **kwargs
): ...
```

## Description

Multiplies FP4 matrices after applying each input's level-0 and level-1 scales. With the transposed weight representation, $Y=A_{\mathrm{dequant}}B_{\mathrm{dequant}}^\top+b$.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `x1`, `x2` | Tensor | [M, K/2], [N, K/2], packed FP4 | Required | Left operand is ND. Right operand is in transposed FRACTAL_NZ format. |
| `x1_level0_scale`, `x2_level0_scale` | Tensor | [M, ceil(K/512)], [ceil(K/512), N], FP32 | Required | Level-0 scales for 512-element groups. |
| `x1_level1_scale`, `x2_level1_scale` | Tensor | [M, ceil(K/64), 2], [N, ceil(K/64), 2], E8M0 bytes | Required | Paired level-1 scales for 32-element groups, in the SDK's packed format. |
| `bias`, `output_dtype` | Tensor or None, torch.dtype | [N], FP32 bias | `None` / `torch.bfloat16` | Optional keywords. The result dtype is FP16 or BF16. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | [M, N], output_dtype | Matrix product on the input NPU. |

## torch_npu backend

Calls `torch_npu.npu_dual_level_quant_matmul`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_dual_level_quant_matmul.md)

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.matmul import dual_level_quant_matmul

m, n, k = 256, 2048, 1024
a = torch.full((m, k // 2), 0x22, dtype=torch.int8, device="npu")
b_nd = torch.full((n, k // 2), 0x22, dtype=torch.int8, device="npu")
b_nz = torch_npu.npu_format_cast(b_nd, 29, customize_dtype=b_nd.dtype)
a0 = torch.ones((m, k // 512), dtype=torch.float32, device=a.device)
b0 = torch.ones((k // 512, n), dtype=torch.float32, device=a.device)
a1 = torch.full((m, k // 64, 2), 127, dtype=torch.uint8, device=a.device)
b1 = torch.full((n, k // 64, 2), 127, dtype=torch.uint8, device=a.device)
y = dual_level_quant_matmul(
    a, b_nz, a0, b0, a1, b1, output_dtype=torch.bfloat16, impl="torch_npu"
)
```

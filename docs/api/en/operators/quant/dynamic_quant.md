# dynamic_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.quant import dynamic_quant


def dynamic_quant(input, *, impl="auto", **kwargs): ...
```

## Description

For symmetric INT8 quantization, computes $s=\max|x|/127$ per token and quantizes $x/s$ with the SDK rounding and range. An optional smooth scale is applied before finding the maximum.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `input` | Tensor | [..., H], FP16/BF16/FP32 as supported | Required | ND floating activations with H features. |
| `dst_type` | torch.dtype | - | `None` | Optional keyword. INT8 is the standard per-token destination. |
| `smooth_scales`, `group_index` | Tensor or None | [H] or [groups, H], and [groups] | `None` | Optional smoothing factors and expert group boundaries. |
| `quant_mode` | str | - | `"pertoken"` | Optional SDK quantization granularity. The default is pertoken. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `values`, `scales` | tuple[Tensor, Tensor] | destination dtype, FP32 | INT8 values have input shape in ordinary per-token mode, with one scale for each row. |

## torch_npu backend

Calls `torch_npu.npu_dynamic_quant`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_dynamic_quant.md)

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.quant import dynamic_quant

x = torch.randn((1, 4096), dtype=torch.float16, device="npu")
values, scales = dynamic_quant(x, dst_type=torch.int8, impl="torch_npu")
```

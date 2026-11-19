# dynamic_block_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.quant import dynamic_block_quant


def dynamic_block_quant(x, *, impl="auto", **kwargs): ...
```

## Description

Quantizes blocks along the last two input axes. Each block gets its own maximum-absolute-value scale, optionally limited by min_scale, before quantization and rounding. The DSA FP8 configuration applies on arch35.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `x` | Tensor | rank 2 or 3, FP16/BF16/FP32 as supported | Required | ND activations. |
| `row_block_size`, `col_block_size` | int | - | `1` / `128` | Optional block sizes. The SDK defaults are 1 and 128. |
| `dst_type` | torch.dtype | - | `torch.int8` | Optional quantized dtype. FP8 requires a compatible arch35 SDK. |
| `min_scale`, `round_mode` | float, str | - | `0.0` / `"rint"` | Optional scale floor and rounding mode. The typical rounding mode is rint. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `values`, `scales` | tuple[Tensor, Tensor] | input shape, FP32 per block | Quantized input and its blockwise scales. |

## torch_npu backend

Calls `torch_npu.npu_dynamic_block_quant`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_dynamic_block_quant.md)

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.quant import dynamic_block_quant

x = torch.randn((16, 256), dtype=torch.float16, device="npu")
y, s = dynamic_block_quant(x, row_block_size=1, col_block_size=128, impl="torch_npu")
```

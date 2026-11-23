# dtype_cast

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-math` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-math` wherever it is registered.

```python
from sgl_kernel_npu.operators.data_move import dtype_cast


def dtype_cast(input, dtype, input_dtype=None, *, impl="auto"): ...
```

## Description

Converts values to a target dtype. The installed SDK defines the packed dtype encoding and packed output shape.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `input` | Tensor | any supported shape and input dtype | Required | Source NPU tensor. Packed types require the SDK's expected representation. |
| `dtype` | torch.dtype | - | Required | Target dtype. |
| `input_dtype` | torch.dtype or None | - | `None` | Optional logical input dtype for a packed representation. |
| `impl` | str | - | `"auto"` | `auto`, `ops-math`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | dtype | Converted tensor on the same NPU. Ordinary casts preserve the logical shape. |

## ops-math backend

Calls `torch_npu.npu_dtype_cast`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/op_plugin/config/op_plugin_functions.yaml#L6168) [CANN ops-math source](https://gitcode.com/cann/ops-math/tree/a6032a3fbb02592142169b05a2157bea08b47ac5/math/cast)

## torch_native backend

Uses PyTorch casting for ordinary tensor dtypes. For FP8 E4M3 weights and an E2M1 destination, packs two FP4 values per output byte.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.data_move import dtype_cast

x = torch.randn((4, 32), dtype=torch.float32, device="npu")
y = dtype_cast(x, torch.float16, impl="ops-math")
```

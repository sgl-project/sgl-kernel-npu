# swiglu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-nn` wherever it is registered.

```python
from sgl_kernel_npu.operators.activation import swiglu


def swiglu(input, dim=-1, *, impl="auto"): ...
```

## Description

Splits the packed gate and up projection into equal halves $g$ and $u$, then computes $y=g\,\sigma(g)\odot u$.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `input` | Tensor | FP16/BF16/FP32 | Required | ND input with gate values followed by up values along the even-length split dimension. |
| `dim` | int | - | `-1` | Split dimension. |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | input dtype | Half of the input size along dim. All other dimensions are preserved. |

## ops-nn backend

Calls `torch_npu.npu_swiglu`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/op_plugin/config/op_plugin_functions.yaml#L7011) [CANN ops-nn source](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/activation/swi_glu)

## torch_native backend

Uses PyTorch SiLU and multiplication with FP32 intermediates before the final cast.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import swiglu

x = torch.randn((16, 4096), dtype=torch.float16, device="npu")
y = swiglu(x, impl="ops-nn")
```

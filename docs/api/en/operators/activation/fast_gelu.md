# fast_gelu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-nn` wherever it is registered.

```python
from sgl_kernel_npu.operators.activation import fast_gelu


def fast_gelu(input, *, impl="auto"): ...
```

## Description

Applies the fast GELU approximation elementwise, $y=x\,\sigma(1.702x)$.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `input` | Tensor | any shape, FP16/BF16/FP32 | Required | ND input. |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | same as input | Activation with the input shape and dtype. |

## ops-nn backend

Calls `torch_npu.npu_fast_gelu`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_fast_gelu.md) [CANN ops-nn source](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/activation/fast_gelu)

## torch_native backend

Evaluates the sigmoid and product with FP32 intermediates and casts the result back to the input dtype.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import fast_gelu

x = torch.randn((16, 3072), dtype=torch.float16, device="npu")
y = fast_gelu(x, impl="ops-nn")
```

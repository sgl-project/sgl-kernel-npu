# rms_norm

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-nn` wherever it is registered.

```python
from sgl_kernel_npu.operators.norm import rms_norm


def rms_norm(input, gamma, epsilon=1e-06, *, impl="auto"): ...
```

## Description

Computes $r=1/\sqrt{\mathrm{mean}(x^2)+\epsilon}$ over the trailing dimensions covered by gamma, then returns $(x\odot r\odot\gamma,r)$.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `input` | Tensor | [B, ..., H], FP16/BF16/FP32 | Required | Floating-point ND activations. |
| `gamma` | Tensor | trailing dimensions of input, input dtype | Required | Learned normalization weight. |
| `epsilon` | float | - | `1e-06` | Nonnegative constant inside the square root. |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `output`, `rstd` | tuple[Tensor, Tensor] | input dtype, FP32 | Output matches input. rstd keeps reduced axes as length-one dimensions. |

## ops-nn backend

Calls `torch_npu.npu_rms_norm`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/op_plugin/config/op_plugin_functions.yaml#L6860) [CANN ops-nn source](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/norm/rms_norm)

## torch_native backend

Computes root-mean-square statistics and multiplication in FP32 before casting the output back.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import rms_norm

x = torch.randn((16, 7168), dtype=torch.float32, device="npu")
gamma = torch.ones((7168,), dtype=x.dtype, device=x.device)
y, rstd = rms_norm(x, gamma, impl="ops-nn")
```

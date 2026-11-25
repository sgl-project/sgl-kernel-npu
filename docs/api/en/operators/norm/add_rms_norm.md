# add_rms_norm

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-nn` wherever it is registered.

```python
from sgl_kernel_npu.operators.norm import add_rms_norm


def add_rms_norm(x1, x2, gamma, epsilon=1e-06, *, impl="auto"): ...
```

## Description

Forms the residual sum $z=x_1+x_2$ and normalizes it with $r=1/\sqrt{\mathrm{mean}(z^2)+\epsilon}$. Returns $(z\odot r\odot\gamma,r,z)$.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `x1`, `x2` | Tensor | same ND shape, FP16/BF16/FP32 | Required | Activation and residual on the same NPU and with matching dtype. |
| `gamma` | Tensor | trailing dimensions of x1, input dtype | Required | Learned normalization weight. |
| `epsilon` | float | - | `1e-06` | Nonnegative constant inside the square root. |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `output`, `rstd`, `residual` | tuple[Tensor, Tensor, Tensor] | input dtype, FP32, input dtype | Output and residual sum match x1. rstd keeps reduced dimensions. |

## ops-nn backend

Calls `torch_npu.npu_add_rms_norm`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm.md) [CANN ops-nn source](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/norm/add_rms_norm)

## torch_native backend

Adds the residual and computes the RMS statistic in FP32. For BF16 inputs on arch22, it rounds the residual sum to BF16 before computing `rstd` and rounds the normalized values to BF16 before multiplying by `gamma`. The first and third outputs use the input dtype.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import add_rms_norm

x = torch.randn((16, 7168), dtype=torch.float32, device="npu")
residual = torch.randn_like(x)
gamma = torch.ones((7168,), dtype=x.dtype, device=x.device)
y, rstd, updated_residual = add_rms_norm(x, residual, gamma, impl="ops-nn")
```

# geglu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-nn` wherever it is registered.

```python
from sgl_kernel_npu.operators.activation import geglu


def geglu(input, dim=-1, approximate=1, activate_left=False, *, impl="auto"): ...
```

## Description

Splits the input into two equal halves. For the selected gate $g$ and the other half $u$, returns $(\mathrm{GELU}(g)\odot u,\mathrm{GELU}(g))$.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `input` | Tensor | FP16/BF16/FP32 | Required | ND input with an even split dimension. |
| `dim` | int | - | `-1` | Split dimension. |
| `approximate` | int | - | `1` | 0 selects exact erf GELU; 1 selects the tanh approximation. |
| `activate_left` | bool | - | `False` | Selects the first half when true and the second half otherwise. |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `gated_output`, `gelu_output` | tuple[Tensor, Tensor] | FP16/BF16/FP32 | The product and GELU intermediate, each with half the original length along dim. |

## ops-nn backend

Calls `torch_npu.npu_geglu`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/op_plugin/config/op_plugin_functions.yaml#L6371) [CANN ops-nn source](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/activation/ge_glu_v2)

## torch_native backend

Splits the halves in PyTorch, applies GELU in FP32 and casts both outputs back to the input dtype.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import geglu

x = torch.randn((16, 6144), dtype=torch.float16, device="npu")
y, gate = geglu(x, activate_left=True, impl="ops-nn")
```

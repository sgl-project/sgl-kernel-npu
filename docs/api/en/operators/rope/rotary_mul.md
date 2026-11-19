# rotary_mul

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.rope import rotary_mul


def rotary_mul(input, r1, r2, rotary_mode="half", rotate=None, *, impl="auto"): ...
```

## Description

Applies $y=x\odot r_1+\mathrm{rotate}(x)\odot r_2$. The half mode pairs corresponding elements of the two halves. The interleave mode pairs adjacent elements.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `input` | Tensor | [B, S, H, D], FP16/BF16/FP32 | Required | ND input with even rotary width D. |
| `r1`, `r2` | Tensor | broadcastable to input | Required | Cosine and sine factors for the chosen layout. |
| `rotary_mode` | str | - | `'half'` | Pairing mode, such as half or interleave. |
| `rotate` | Tensor or None | SDK-specific | `None` | Optional rotation matrix accepted by the installed TorchNPU. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | same as input | Rotated activations. |

## torch_npu backend

Calls `torch_npu.npu_rotary_mul`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_rotary_mul.md)

## torch_native backend

Implements half and adjacent-pair rotations with FP32 arithmetic. It requires rotate=None.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.rope import rotary_mul

x = torch.randn((2, 17, 4, 64), dtype=torch.float16, device="npu")
r1 = torch.ones((1, 17, 1, 64), dtype=x.dtype, device=x.device)
r2 = torch.zeros_like(r1)
y = rotary_mul(x, r1, r2, impl="torch_npu")
```

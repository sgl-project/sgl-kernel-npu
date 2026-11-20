# interleave_rope

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-transformer` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-transformer` wherever it is registered.

```python
from sgl_kernel_npu.operators.rope import interleave_rope


def interleave_rope(x, cos, sin, *, impl="auto"): ...
```

## Description

Moves even and odd input coordinates into separate halves $z$ and applies $y=z\odot\cos+\mathrm{rotateHalf}(z)\odot\sin$ to the resulting order.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `x` | Tensor | [T, H, S, D], FP16/BF16/FP32 | Required | ND input. D is even. |
| `cos`, `sin` | Tensor | broadcastable to x, x.dtype | Required | Rotary coefficients in the deinterleaved order. |
| `impl` | str | - | `"auto"` | `auto`, `ops-transformer`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | same as x | Rotated result in the deinterleaved order. |

## ops-transformer backend

Calls `torch_npu.npu_interleave_rope`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_interleave_rope.md) [CANN ops-transformer source](https://gitcode.com/cann/ops-transformer/tree/7326dcc4f1e2dde5d2a802458877f927570267d1/posembedding/interleave_rope)

## torch_native backend

Concatenates the even and odd columns and evaluates half-rotation RoPE using PyTorch.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.rope import interleave_rope

x = torch.randn((16, 16, 1, 64), dtype=torch.float16, device="npu")
cos = torch.ones((16, 1, 1, 64), dtype=x.dtype, device=x.device)
sin = torch.zeros_like(cos)
y = interleave_rope(x, cos, sin, impl="ops-transformer")
```

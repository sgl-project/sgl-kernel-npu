# scatter_nd_update_

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | Supported | Supported |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-nn` wherever it is registered.

```python
from sgl_kernel_npu.operators.mem_cache import scatter_nd_update_


def scatter_nd_update_(input, indices, updates, *, impl="auto"): ...
```

## Description

Writes $\mathrm{input}[\mathrm{indices}[i],\ldots]=\mathrm{updates}[i,\ldots]$ in place. Each coordinate indexes a prefix of the input dimensions, so an update can replace a slice.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `input` | Tensor | ND, dtype supported by TorchNPU | Required | Destination tensor on the NPU. |
| `indices` | Tensor | [..., R], INT32/INT64 | Required | Last axis holds R coordinates. Use distinct valid indices for determinism. |
| `updates` | Tensor | indices.shape[:-1] + input.shape[R:], input dtype | Required | Values to write at those coordinates on the same device. |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `ascendc`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | same as input | Aliases the updated destination tensor. |

## ops-nn backend

Calls `torch_npu.npu_scatter_nd_update_`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_scatter_nd_update_.md) [CANN ops-nn source](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/index/scatter_nd_update)

## ascendc backend

Selects the high-performance cache update kernel with `impl="ascendc"`, updates `input` in place, and returns the same tensor. For row updates, index `-1` skips that update. Repeated writes can be nondeterministic, and positive out-of-range indices are unsafe. Supported dtypes follow this backend's implementation.

## torch_native backend

Uses PyTorch indexed assignment to update the caller-owned tensor and returns that tensor. Negative indices raise an error.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import scatter_nd_update_

dst = torch.zeros((7, 32), dtype=torch.float16, device="npu")
slots = torch.tensor([[6], [0], [3]], dtype=torch.int32, device=dst.device)
values = torch.ones((3, 32), dtype=dst.dtype, device=dst.device)
scatter_nd_update_(dst, slots, values, impl="ops-nn")
```

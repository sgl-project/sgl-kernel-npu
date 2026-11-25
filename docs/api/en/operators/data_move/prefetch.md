# prefetch

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `runtime` | Supported | Unsupported |

## Signature

`auto` selects `runtime` wherever it is registered.

```python
from sgl_kernel_npu.operators.data_move import prefetch


def prefetch(input, dependency, max_size, offset=0, *, impl="auto"): ...
```

## Description

Requests asynchronous L2 prefetch for a region of the input storage. The dependency tensor can express execution order relative to another operation.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `input` | Tensor | contiguous NPU tensor | Required | Input storage to prefetch. |
| `dependency` | Tensor or None | same NPU | Required | Optional dependency tensor. |
| `max_size` | int | bytes | Required | Maximum number of bytes to prefetch. |
| `offset` | int | bytes | `0` | Offset from the start of the input storage. |
| `impl` | str | - | `"auto"` | `auto`, `runtime`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `result` | None | - | This is a cache hint. It returns no tensor. |

## runtime backend

Calls `torch_npu.npu_prefetch`, which issues the CANN Runtime request `aclrtCmoAsync`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_prefetch.md) [CANN Runtime API](https://gitcode.com/cann/runtime/blob/be5219e84b0af2b95165b44d9231871fa921d5fa/include/external/acl/acl_rt.h)

Prefetch is a hardware cache hint. Its observable contract is that it returns `None` and preserves the input values.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.data_move import prefetch

x = torch.ones(2048, dtype=torch.float16, device="npu")
prefetch(x, x, x.numel() * x.element_size(), impl="runtime")
```

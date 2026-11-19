# reshape_and_cache

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.mem_cache import reshape_and_cache


def reshape_and_cache(
    key, value, key_cache, value_cache, slot_indices, *, impl="auto"
): ...
```

## Description

Writes the input key/value rows into physical paged slots. For page size L, slot s identifies page s//L and row s%L within that page.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `key`, `value` | Tensor | [T, Hkv, D] for the example, FP16 | Required | T cache rows on the same NPU as their destination. |
| `key_cache`, `value_cache` | Tensor | [P, L, Hkv, D] for ND, input dtype | Required | Caller-owned paged destination caches, modified in place. |
| `slot_indices` | Tensor | [T], INT32 in the example | Required | Physical slot for each input row. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `result` | None | - | Only the supplied caches are updated. |

## torch_npu backend

Calls `torch_npu._npu_reshape_and_cache`. Shape, dtype and layout combinations follow the installed SDK. Requires an SDK exposing the private ATB reshape-and-cache binding. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/op_plugin/config/atb_ops.yaml)

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import reshape_and_cache

key = torch.randn((3, 2, 128), dtype=torch.float16, device="npu")
value = torch.randn_like(key)
key_cache = torch.zeros((2, 16, 2, 128), dtype=key.dtype, device=key.device)
value_cache = torch.zeros_like(key_cache)
slots = torch.tensor([31, 15, 16], dtype=torch.int32, device=key.device)
reshape_and_cache(key, value, key_cache, value_cache, slots, impl="torch_npu")
```

# scatter_pa_kv_cache

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-transformer` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-transformer` wherever it is registered.

```python
from sgl_kernel_npu.operators.mem_cache import scatter_pa_kv_cache


def scatter_pa_kv_cache(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    *,
    compress_lens=None,
    compress_seq_offsets=None,
    seq_lens=None,
    cache_mode="PA_NZ",
    impl="auto"
): ...
```

## Description

Copies key and value rows into the supplied paged caches using slot_mapping. With an uncompressed ND cache, each mapped slot receives the corresponding K/V row. The SDK supports additional compressed-cache modes.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `key`, `value` | Tensor | [T, Hkv, D] for Norm, FP16 in example | Required | Cache rows to write on NPU. |
| `key_cache`, `value_cache` | Tensor | [P, L, Hkv, D] for Norm, input dtype | Required | Caller-owned paged destination caches, modified in place. |
| `slot_mapping` | Tensor | [T], INT32 in example | Required | Physical slot for each token row. |
| `compress_lens`, `compress_seq_offsets`, `seq_lens` | Tensor or None | SDK metadata tensors | `None` | Optional metadata for compressed-cache placement. |
| `cache_mode` | str | - | `'PA_NZ'` | Cache layout mode. Use Norm for uncompressed ND cache in the example, PA_NZ for the default vendor format. |
| `impl` | str | - | `"auto"` | `auto`, `ops-transformer`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `result` | None | - | Only the supplied caches are updated. |

## ops-transformer backend

Calls `torch_npu.npu_scatter_pa_kv_cache`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_scatter_pa_kv_cache.md) [CANN ops-transformer source](https://gitcode.com/cann/ops-transformer/tree/7326dcc4f1e2dde5d2a802458877f927570267d1/attention/scatter_pa_kv_cache)

## torch_native backend

Writes K/V into uncompressed ND pages in Norm mode.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import scatter_pa_kv_cache

key = torch.randn((3, 2, 128), dtype=torch.float16, device="npu")
value = torch.randn_like(key)
key_cache = torch.zeros((2, 16, 2, 128), dtype=key.dtype, device=key.device)
value_cache = torch.zeros_like(key_cache)
slots = torch.tensor([31, 15, 16], dtype=torch.int32, device=key.device)
scatter_pa_kv_cache(
    key, value, key_cache, value_cache, slots, cache_mode="Norm", impl="ops-transformer"
)
```

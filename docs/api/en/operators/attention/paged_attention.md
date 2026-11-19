# paged_attention

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.attention import paged_attention


def paged_attention(
    query,
    key_cache,
    value_cache,
    num_heads,
    num_kv_heads,
    scale_value,
    block_table,
    context_lens,
    out,
    *,
    impl="auto",
    **kwargs
): ...
```

## Description

Gathers cached keys and values through the block table and performs grouped-query attention for one decode token per request. For each head, $Y=\mathrm{softmax}(sQK^\top)V$ over the valid cached prefix.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `query` | Tensor | [T, H, D], FP16/BF16 | Required | One row per query token on NPU. |
| `key_cache`, `value_cache` | Tensor | [P, L, Hkv, Dk] and [P, L, Hkv, Dv] | Required | Paged key and value storage. P is the number of physical pages and L is the page size. |
| `num_heads`, `num_kv_heads` | int | - | Required | Query and KV head counts. H must be divisible by Hkv. |
| `scale_value` | float | - | Required | Attention logit scale, commonly Dk**-0.5. |
| `block_table` | Tensor | [B, max_pages], INT32, NPU | Required | Maps logical pages of each request to physical pages. |
| `context_lens` | Tensor | [B], INT32 CPU in the example | Required | Number of visible cached tokens per request. |
| `out` | Tensor | [T, H, Dv], same dtype and NPU as query | Required | Caller-owned output buffer, written in place. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `result` | None | - | The output is written into out. |

## torch_npu backend

Calls `torch_npu._npu_paged_attention`. Shape, dtype and layout combinations follow the installed SDK. Requires an SDK exposing the private ATB paged-attention binding. The example uses host sequence lengths and a device block table. Optional backend keywords such as workspace are SDK-specific. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/test/test_custom_ops/test_atb_paged_attention.py)

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import paged_attention

heads, kv_heads, dk, dv = 32, 16, 288, 96
q = torch.randn((2, heads, dk), dtype=torch.float16, device="npu")
k = torch.randn((64, 128, kv_heads, dk), dtype=q.dtype, device=q.device)
v = torch.randn((64, 128, kv_heads, dv), dtype=q.dtype, device=q.device)
table = torch.arange(16, dtype=torch.int32, device=q.device).reshape(2, 8)
lengths = torch.full((2,), 1024, dtype=torch.int32, device="cpu")
out = torch.empty((2, heads, dv), dtype=q.dtype, device=q.device)
paged_attention(
    q, k, v, heads, kv_heads, dk**-0.5, table, lengths, out, impl="torch_npu"
)
```

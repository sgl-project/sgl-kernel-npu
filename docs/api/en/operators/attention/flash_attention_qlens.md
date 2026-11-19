# flash_attention_qlens

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.attention import flash_attention_qlens


def flash_attention_qlens(
    query,
    key_cache,
    value_cache,
    mask,
    block_table,
    seq_len,
    context_lens,
    scale_value,
    num_heads,
    num_kv_heads,
    out,
    *,
    impl="auto",
    **kwargs
): ...
```

## Description

Performs variable-length causal prefill, using the block table to gather K/V and the query/context lengths to find each request's visible key prefix. The per-head output is $\mathrm{softmax}(sQK^\top+M)V$ with the supplied mask in the SDK's format.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `query` | Tensor | [T, H, D], FP16/BF16 | Required | Queries packed across requests on NPU. T is the sum of seq_len. |
| `key_cache`, `value_cache` | Tensor | [P, L, Hkv, Dk] and [P, L, Hkv, Dv] | Required | Paged key and value storage. P is the number of physical pages and L is the page size. |
| `num_heads`, `num_kv_heads` | int | - | Required | Query and KV head counts. H must be divisible by Hkv. |
| `scale_value` | float | - | Required | Attention logit scale, commonly Dk**-0.5. |
| `block_table` | Tensor | [B, max_pages], INT32, NPU | Required | Maps logical pages of each request to physical pages. |
| `context_lens` | Tensor | [B], INT32 CPU in the example | Required | Number of visible cached tokens per request. |
| `out` | Tensor | [T, H, Dv], same dtype and NPU as query | Required | Caller-owned output buffer, written in place. |
| `mask` | Tensor | [128, 128], BF16 in the example | Required | Compressed upper-triangular mask for the vendor causal path. |
| `seq_len` | Tensor | [B], INT32 CPU in the example | Required | Number of packed query tokens for each request. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `result` | None | - | The output is written into out. |

## torch_npu backend

Calls `torch_npu._npu_flash_attention_qlens`. Shape, dtype and layout combinations follow the installed SDK. Requires the private ATB Flash Attention binding. The example follows an ATB BF16 test configuration and assumes the cache already contains the current query suffix. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/test/test_custom_ops/test_atb_flash_attention_qlens.py)

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import flash_attention_qlens

heads, kv_heads, d = 32, 16, 128
q = torch.randn((384, heads, d), dtype=torch.bfloat16, device="npu")
k = torch.randn((64, 128, kv_heads, d), dtype=q.dtype, device=q.device)
v = torch.randn_like(k)
mask = torch.ones((128, 128), dtype=q.dtype, device=q.device).triu(1)
table = torch.arange(16, dtype=torch.int32, device=q.device).reshape(2, 8)
qlens = torch.tensor([128, 256], dtype=torch.int32, device="cpu")
lengths = torch.tensor([1024, 1024], dtype=torch.int32, device="cpu")
out = torch.empty_like(q)
flash_attention_qlens(
    q,
    k,
    v,
    mask,
    table,
    qlens,
    lengths,
    d**-0.5,
    heads,
    kv_heads,
    out,
    impl="torch_npu",
)
```

# attention_sinks

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.attention import attention_sinks


def attention_sinks(
    query,
    k_cache,
    v_cache,
    sinks,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
    *,
    impl="auto"
): ...
```

## Description

Computes paged grouped-query decode attention with a learned sink logit per query head. The sink participates in the softmax denominator with zero value contribution. Optional sliding windows limit visible KV tokens.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | Tensor | `Required` | Contiguous floating [tokens,QH*D] queries. |
| `k_cache` | Tensor | `Required` | Contiguous floating [pages,page_size,KVH,D] key cache. |
| `v_cache` | Tensor | `Required` | Value cache with the same shape and dtype as k_cache. |
| `sinks` | Tensor | `Required` | Floating [QH] sink logits. |
| `block_tables` | Tensor | `Required` | INT32/INT64 [batch,max_pages] physical page IDs. |
| `context_lens` | Tensor | `Required` | INT32/INT64 [batch] total KV lengths including appended queries. |
| `scale` | float or None | `Required` | QK scaling; KDA defaults to key_dim**-0.5 and requires positive values. |
| `sliding_window_size` | int | `Required` | Visible token window; -1 disables the window. |
| `q_head_num` | int | `Required` | Query heads; divisible by k_head_num. |
| `k_head_num` | int | `Required` | KV head count. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | [tokens,QH*D] attention output, query dtype/device. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import attention_sinks

q = torch.randn(1, 4096, device="npu", dtype=torch.bfloat16)
k = torch.randn(1, 128, 8, 64, device="npu", dtype=q.dtype)
v = torch.randn_like(k)
sinks = torch.zeros(64, device="npu")
table = torch.zeros(1, 1, device="npu", dtype=torch.int32)
length = torch.tensor([1], device="npu", dtype=torch.int32)
y = attention_sinks(q, k, v, sinks, table, length, 0.125, -1, 64, 8, impl="triton")
```

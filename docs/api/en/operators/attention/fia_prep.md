# fia_prep

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
import torch
from sgl_kernel_npu.operators.attention import fia_prep

def fia_prep(ti, seq_lens, per_query_req, req_to_token, block_size, block_table_out=None, actual_kvlen_out=None, *, impl='auto'):
    ...
```

`auto` aliases `triton`; it does not automatically fall back to the reference.

## Description

Keep non-own selected blocks in order, append the own block, map token slots to physical pages by integer division, and compute the causal length.

## Parameters and return values

Triton tensor inputs should share one NPU and use contiguous layouts; outputs reside on the input device.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `ti` | Tensor | `[T,S]; int32` | Required | Logical page IDs with -1 padding; reserve capacity for the own page. |
| `seq_lens` | Tensor | `[T] or [B]; int32` | Required | Causal KV length per query/request. |
| `per_query_req` | Tensor | `[T]; int32/int64` | Required | Request index for each query. |
| `req_to_token` | Tensor | `[requests,max_ctx]; int32/int64` | Required | Token-slot mapping; same NPU as ti. |
| `block_size` | int | `—` | Required | Number of tokens per KV page; positive. |
| `block_table_out` | Tensor or None | `[T,S]; int32` | `None` | Optional output buffer, overwritten and returned. |
| `actual_kvlen_out` | Tensor or None | `[T]; int32` | `None` | Optional causal-length output buffer. |
| `impl` | str | `—` | `'auto'` | Keyword-only: auto, triton or torch_native. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `block_table` | Tensor or None | `[T,S]; int32` | Reordered physical pages; reuses block_table_out if provided. |
| `actual_kvlen` | Tensor or None | `[T]; int32` | Full non-own pages plus the causal own-page prefix when selected. |

## triton backend

S must be a power of two for the current launch. Keep one free output slot if the own page is absent; request/page indices must be valid.

## torch_native backend

Independent PyTorch operations provide the reference result. Host metadata transfers may synchronize the device; this backend is intended for accuracy comparison.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention import fia_prep

ti = torch.tensor([[0, 1, -1, -1]], device="npu", dtype=torch.int32)
lengths = torch.tensor([17], device="npu", dtype=torch.int32)
req = torch.tensor([0], device="npu", dtype=torch.int32)
slots = torch.arange(64, device="npu", dtype=torch.int32).reshape(1, 64)
table, kvlen = fia_prep(ti, lengths, req, slots, 16, impl="triton")
```

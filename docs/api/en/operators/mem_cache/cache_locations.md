# cache_loc_assign / cache_loc_update

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import cache_loc_assign, cache_loc_update

def cache_loc_assign(req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc, *, impl="auto"): ...
def cache_loc_update(req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc, *, impl="auto"): ...
```

Both functions default to `ascendc` when `impl="auto"`. Select `torch_native` explicitly to use PyTorch.

## Description

These functions copy cache-slot indices between request rows and a packed buffer. They operate on the indices only. For request `i`, the selected interval is `token_pool[req_pool_indices[i], start_offset[i]:end_offset[i]]`. Intervals occupy consecutive positions in the buffer in request order. An empty interval consumes no positions.

### cache_loc_assign

Copies each packed interval into its table row. Positions outside the destination intervals retain their values.

### cache_loc_update

Copies the selected table intervals into the packed buffer. Elements after the last packed interval retain their values.

## Parameters and return value

All tensors must reside on the same NPU. Supply dense table and index tensors, valid row and column indices, and nonoverlapping storage for the table and packed buffer.

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `req_pool_indices` | `torch.Tensor` | `[B]` | `int32 / int64` | Required | One table row per request. Use distinct rows. |
| `token_pool` | `torch.Tensor` | `[pool_rows, row_width]` | `int32` | Required | Cache-slot table. Destination for assign, source for update. |
| `start_offset` | `torch.Tensor` | `[B]` | `int64` | Required | Inclusive start column in each row. |
| `end_offset` | `torch.Tensor` | `[B]` | `int64` | Required | Exclusive end column, at least the start column. |
| `out_cache_loc` | `torch.Tensor` | Any shape | `int32` | Required | Contiguous buffer with at least `sum(end_offset - start_offset)` elements. A multidimensional buffer is treated as flat. Source for assign, destination for update. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only. `auto`, `ascendc`, `torch_native` |

| Function | Return type | Shape and dtype | Description |
| --- | --- | --- | --- |
| `cache_loc_assign` | `torch.Tensor` | Same as `token_pool` | Shares storage with `token_pool`, which is updated in place. |
| `cache_loc_update` | `torch.Tensor` | Same as `out_cache_loc` | Shares storage with `out_cache_loc`, which is updated in place. |

## ascendc backend

Each interval can contain at most 16 indices. Provide storage for aligned reads and a full 16-element window from each accessed start column. Batch size and packed buffer length must fit the device’s on-chip buffer capacity.

## torch_native backend

Uses the same packed intervals and in-place outputs.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import cache_loc_assign, cache_loc_update

rows = torch.arange(8, device="npu", dtype=torch.int32)
pool = torch.zeros((8, 32), device="npu", dtype=torch.int32)
starts = torch.zeros(8, device="npu", dtype=torch.int64)
ends = starts + 1
slots = rows + 100

cache_loc_assign(rows, pool, starts, ends, slots, impl="ascendc")
result = torch.empty_like(slots)
cache_loc_update(rows, pool, starts, ends, result, impl="ascendc")
# result: [100, 101, 102, 103, 104, 105, 106, 107]
```

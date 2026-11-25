# alloc_extend

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import alloc_extend

def alloc_extend(
    pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values,
    *, impl="auto",
): ...
```

`auto` selects `ascendc`.

## Description

Compute the KV-cache slot indices for tokens appended to a batch of paged requests during prefill or decoding. Each request first fills the unused tail of its current page, then consumes new pages from free_pages in order, including a final partial page when needed. The indices are packed in request order into out_indices. The caller retains responsibility for consuming the free-page list; the operation does not move KV data. It updates the supplied output buffers in place and returns None.

## Parameters and return value

Use tensors on the same NPU.

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `pre_lens` | `torch.Tensor` | `[B]` | `int64` | Required | Existing length of each request. B must be positive. |
| `seq_lens` | `torch.Tensor` | `[B]` | `int64` | Required | Target lengths, each at least its prefix length. |
| `last_loc` | `torch.Tensor` | `[B]` | `int64` | Required | Last occupied slot of each request. Use -1 for an empty prefix. |
| `free_pages` | `torch.Tensor` | `[P]` | `int64` | Required | Available page IDs in allocation order. Supply enough pages for the batch. |
| `page_size` | `int` | — | — | Required | Positive number of slots per page, representable as int32. |
| `out_indices` | `torch.Tensor` | `[sum(seq_lens - pre_lens)]` | `int64` | Required | Caller-allocated output for the new slot indices. |
| `values` | `torch.Tensor` | [N], N ≥ 1 | `int64` | Required | Writes the batch’s new-page count to element 0 only if the final request ends in a newly allocated partial page. Otherwise it retains its input value. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only. Choices are `"auto"`, `"ascendc"`, `"torch_native"`. |

| Return value | Type | Description |
| --- | --- | --- |
| `None` | `NoneType` | Writes out_indices and conditionally values[0] in place. |

## ascendc backend

Use dense one-dimensional buffers.

## torch_native backend

Computes on the input device with the same output buffers.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import alloc_extend

def vector(items):
    return torch.tensor(items, device="npu", dtype=torch.int64)

out = vector([0, 0, 0])
values = vector([-1])
alloc_extend(vector([3]), vector([6]), vector([22]), vector([8]), 4,
             out, values, impl="torch_native")
print(out.tolist(), values.item())  # [23, 32, 33], 1
```

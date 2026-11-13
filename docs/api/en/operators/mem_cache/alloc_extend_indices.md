# alloc_extend_indices

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.mem_cache import alloc_extend_indices


def alloc_extend_indices(
    pre_lens,
    seq_lens,
    last_loc,
    free_pages,
    page_size,
    out_indices,
    max_num_extend_tokens,
    *,
    impl="auto"
): ...
```

## Description

Fills packed slot indices for appended tokens, first using the old partial page and then consuming free page IDs in batch order. This interface returns no page-count buffer and does not mutate free_pages. It preserves the index dtypes of the caller-provided tensors.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `pre_lens` | Tensor | `Required` | Contiguous INT32/INT64 [batch] existing sequence lengths. |
| `seq_lens` | Tensor | `Required` | Contiguous INT32/INT64 [batch] target lengths, at least pre_lens. |
| `last_loc` | Tensor | `Required` | Contiguous integer [batch] last occupied slot; -1 for empty prefixes. |
| `free_pages` | Tensor | `Required` | Contiguous integer [pages] free page IDs, sufficient for all extensions. |
| `page_size` | int | `Required` | Positive power-of-two slots per page. |
| `out_indices` | Tensor | `Required` | Writable integer [sum(seq_lens-pre_lens)] packed slots. |
| `max_num_extend_tokens` | int | `Required` | Upper bound on each request's appended token count. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | None | Writes out_indices in place. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import alloc_extend_indices

pre = torch.tensor([1], device="npu", dtype=torch.int32)
seq = torch.tensor([3], device="npu", dtype=torch.int32)
last = torch.tensor([0], device="npu", dtype=torch.int32)
pages = torch.tensor([4], device="npu", dtype=torch.int32)
out = torch.empty(2, device="npu", dtype=torch.int32)
alloc_extend_indices(pre, seq, last, pages, 16, out, 2, impl="triton")
```

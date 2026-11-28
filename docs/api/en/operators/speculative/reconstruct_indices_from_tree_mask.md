# reconstruct_indices_from_tree_mask

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

`auto` selects `triton`; both backends are registered without architecture restriction. All tensors use the same NPU.

```python
from sgl_kernel_npu.operators.speculative import reconstruct_indices_from_tree_mask

def reconstruct_indices_from_tree_mask(tree_mask, verified_seq_len, positions, retrive_index, retrive_next_token, retrive_next_sibling, batch_size, draft_token_num, *, impl="auto"): ...
```

## Description

Reconstruct depth positions, global indices and local first-child and next-sibling links from a parent-ordered ancestry mask. Missing links are -1.

## Parameters and return value

| Parameter | Type, shape and constraints |
| --- | --- |
| `tree_mask` | Contiguous bool `[B,D,D]` or flat; parent-ordered ancestry matrix. |
| `verified_seq_len` | Contiguous int32/int64 `[B]`; accepted prefix lengths. |
| `positions` | Mutable int64 `[B*D]` or `[B,D]`; prefix length plus strict ancestor count. |
| `retrive_index` | Mutable int64 `[B,D]`; global slot `b*D+j`. |
| `retrive_next_token` | Mutable int64 `[B,D]`; local first-child slot or -1. |
| `retrive_next_sibling` | Mutable int64 `[B,D]`; local next-sibling slot or -1. |
| `batch_size` | int `B`; number of requests. |
| `draft_token_num` | int `D`; candidate width including the root. |
| `impl` | str; `auto`, `triton`, or `torch_native` (default `auto`). |

All output buffers are contiguous and must not overlap.

| Return value | Type and meaning |
| --- | --- |
| `None` | Output buffers are updated in place. |

## triton backend

Reads the compact mask and writes depth and all three links in a single launch.

## torch_native backend

Independent PyTorch reference verified on Ascend A3 (arch22).

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.speculative import reconstruct_indices_from_tree_mask

device = "npu"
mask = torch.tensor([[[1, 0], [1, 1]]], dtype=torch.bool, device=device)
lens = torch.tensor([3], dtype=torch.int32, device=device)
out = [torch.empty((1, 2), dtype=torch.int64, device=device) for _ in range(4)]
reconstruct_indices_from_tree_mask(mask, lens, *out, 1, 2, impl="triton")
```

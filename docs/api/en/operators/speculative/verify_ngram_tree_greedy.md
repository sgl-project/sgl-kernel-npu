# verify_ngram_tree_greedy

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

`auto` selects `triton`; both backends are registered without architecture restriction. All tensors use the same NPU.

```python
from sgl_kernel_npu.operators.speculative import verify_ngram_tree_greedy

def verify_ngram_tree_greedy(predicts, accept_index, accept_token_num, candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict, *, impl="auto"): ...
```

## Description

Follow matching children and siblings until mismatch, leaf, or acceptance width. Write the root and accepted global slots, count accepted drafts, and write the final target prediction. Unwritten output entries remain unchanged.

## Parameters and return value

| Parameter | Type, shape and constraints |
| --- | --- |
| `predicts` | Mutable int32 `[B*D]`; token IDs in global slot order. |
| `accept_index` | Mutable int32 `[B,S]`, `S >= 1`; root and accepted slots in path order. |
| `accept_token_num` | Mutable int32 `[B]`; counts drafts, excluding the final target prediction. |
| `candidates` | Int32/int64 `[B,D]`; candidate token IDs. |
| `retrive_index` | Int32/int64 `[B,D]`; maps local nodes to valid global slots. |
| `retrive_next_token` | Int32/int64 `[B,D]`; first-child indices, local or -1. |
| `retrive_next_sibling` | Int32/int64 `[B,D]`; next-sibling indices, local or -1. |
| `target_predict` | Int32/int64 `[B,D]`; target next-token IDs, addressed by flattened global slot. |
| `impl` | str; `auto`, `triton`, or `torch_native` (default `auto`). |

All buffers are contiguous and tensor storage for writes must not overlap. Child and sibling links must describe a valid tree.

| Return value | Type and meaning |
| --- | --- |
| `None` | Output buffers are updated in place. |

## triton backend

Traverses child/sibling links on device; a width-one accept buffer takes the root-only path.

## torch_native backend

Independent PyTorch reference verified on Ascend A3 (arch22).

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.speculative import verify_ngram_tree_greedy

device = "npu"
x = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
links = torch.tensor([[1, -1]], dtype=torch.int32, device=device)
siblings = torch.full_like(x, -1)
predicts = torch.zeros(2, dtype=torch.int32, device=device)
accepted = torch.full_like(x, -1)
count = torch.zeros(1, dtype=torch.int32, device=device)
verify_ngram_tree_greedy(predicts, accepted, count, x, x, links, siblings, x, impl="triton")
```

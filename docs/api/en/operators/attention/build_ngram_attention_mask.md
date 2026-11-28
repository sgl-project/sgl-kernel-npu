# build_ngram_attention_mask

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

`auto` selects `triton`; both backends are registered without architecture restriction. All tensors use the same NPU.

```python
from sgl_kernel_npu.operators.attention import build_ngram_attention_mask

def build_ngram_attention_mask(tree_mask, prefix_lens, draft_token_num, padded_kv_len, *, impl="auto"): ...
```

## Description

Build a paged FIA mask: True blocks attention. Prefix and tree ancestors are visible; siblings and unused tail slots are blocked.

## Parameters and return value

| Parameter | Type, shape and constraints |
| --- | --- |
| `tree_mask` | Contiguous bool `[B,D,D]` or flat; `True` marks an ancestor, including the node itself. |
| `prefix_lens` | Contiguous int32/int64 `[B]`; visible prefix lengths. |
| `draft_token_num` | int `D`; tree width including the root. |
| `padded_kv_len` | int; padded key width, at least max prefix + `D` for a nonempty batch. |
| `impl` | str; `auto`, `triton`, or `torch_native` (default `auto`). |

| Return value | Type and meaning |
| --- | --- |
| mask | Bool `[B,1,D,padded_kv_len]` on the device of `tree_mask`. |

## triton backend

Expands the compact tree into a blocked mask in a single launch.

## torch_native backend

Independent PyTorch reference verified on Ascend A3 (arch22).

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import build_ngram_attention_mask

device = "npu"
mask = torch.tensor([[[1, 0], [1, 1]]], dtype=torch.bool, device=device)
lens = torch.tensor([3], dtype=torch.int32, device=device)
out = build_ngram_attention_mask(mask, lens, 2, 5, impl="triton")
```

# qsa_topk

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.indexer import qsa_topk

def qsa_topk(score, lengths, topk, row_starts=None, *, impl="auto"):
    ...
```

## Description

Select sequence-relative QSA blocks from FP32 score[R,M]. topk=512/2048; contiguous int32 bounds. Returns contiguous int32[R,topk], ordered short rows and -1 suffixes. Ties may select any valid equal-valued indices. auto selects the Triton/native hybrid.

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output.

For row r, consider `score[r,start:start+length]`. If length is at most topk, return indices `0..length-1` in input order. Otherwise return any topk indices with the highest scores in that interval. Indices are relative to start; unused output slots are -1. Tied scores do not require a specific index order.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `score` | Tensor | [R,M], FP32 | required | NPU; contiguous columns, row gaps allowed; M <= 2**24; finite active scores. |
| `lengths` | Tensor | [R], int32 | required | Contiguous on score.device; valid interval stays inside score. |
| `topk` | int | — | required | 512 or 2048. |
| `row_starts` | Tensor or None | [R], int32 | None | Contiguous on score.device; None means all zero. |
| `impl` | str | — | "auto" | Keyword-only: auto, triton, torch_native. |

| Return | Type | Shape | Dtype | Device / layout |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,topk] | int32 | Input device; contiguous, newly allocated. |

## triton

Default implementation. Checks input metadata before launch. Supports graph capture and replay after in-place input updates.

## torch_native

Independent CPU reference. Tied scores may select different valid indices. Use outside graph capture. Select explicitly with `impl="torch_native"`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import qsa_topk

score = torch.randn(1, 1024, device="npu", dtype=torch.float32)
lengths = torch.full((1,), 1024, device="npu", dtype=torch.int32)
y = qsa_topk(score, lengths, 512, impl="auto")
```

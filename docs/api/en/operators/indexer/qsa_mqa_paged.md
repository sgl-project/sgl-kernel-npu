# qsa_mqa_paged

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.indexer import qsa_mqa_paged

def qsa_mqa_paged(q, cache, table, lengths, width, *, impl="auto"):
    ...
```

## Description

Paged QSA scores: BF16 Q[R,4,128], cache[P,16,1,128], int32 table[R,L] and lengths[R]; width=16*L. Returns FP32[R,width] with -inf padding. auto selects Triton.

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output.

For j in each valid row, read `cache[table[r,j//16],j%16,0]`, then compute `sum_h ReLU(dot(q[r,h],key)) / sqrt(128)` in FP32.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `q` | Tensor | [R,4,128], BF16 | required | Contiguous NPU query. |
| `cache` | Tensor | [P,16,1,128], BF16 | required | Contiguous on q.device. |
| `table` | Tensor | [R,L], int32 | required | Contiguous physical page table; only active entries must be valid. |
| `lengths` | Tensor | [R], int32 | required | Contiguous on q.device; 0 <= lengths <= width. |
| `width` | int | — | required | 16*L. |
| `impl` | str | — | "auto" | Keyword-only: auto, triton, torch_native. |

| Return | Type | Shape | Dtype | Device / layout |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,width] | FP32 | Input device; contiguous, newly allocated. |

## triton

Default implementation. Checks input metadata before launch. Supports graph capture and replay after in-place input updates.

## torch_native

Independent CPU FP64 reference; returns results to the input device. Use outside graph capture. Select explicitly with `impl="torch_native"`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import qsa_mqa_paged

q = torch.randn(1, 4, 128, device="npu", dtype=torch.bfloat16)
cache = torch.randn(1, 16, 1, 128, device="npu", dtype=q.dtype)
table = torch.zeros(1, 1, device="npu", dtype=torch.int32)
lengths = torch.full((1,), 16, device="npu", dtype=torch.int32)
y = qsa_mqa_paged(q, cache, table, lengths, 16, impl="auto")
```

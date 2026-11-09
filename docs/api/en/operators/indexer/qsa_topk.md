# qsa_topk

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.indexer import qsa_topk

def qsa_topk(score, lengths, topk, row_starts=None, *, impl="auto"):
    ...
```

## Description

Select sequence-relative QSA blocks from FP32 score[R,M]. topk=512/2048; contiguous int32 bounds. Returns contiguous int32[R,topk], ordered short rows and -1 suffixes. Ties may select any valid equal-valued indices. auto selects the Triton/native hybrid.

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output. `auto` aliases `triton`, including its native PyTorch substeps. No error-driven fallback is installed. `torch_native` is an independent reference: short convolution uses native NPU convolution; the other references use CPU arithmetic (FP64 for floating-point oracles) and return to the input device. These references synchronize and are intended for accuracy checks outside graph capture.

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

Default inference implementation. Metadata checks reject unsupported inputs before launch; input contents must satisfy the caller constraints above. Supports graph capture and replay with in-place input and weight updates. Native substeps, where present, are part of this implementation.

## torch_native

Independent accuracy reference for the documented input contract. Synchronizing host transfers are permitted here; run this implementation outside graph capture. It is never selected automatically after a failure.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import qsa_topk

score = torch.randn(1, 1024, device="npu", dtype=torch.float32)
lengths = torch.full((1,), 1024, device="npu", dtype=torch.int32)
y = qsa_topk(score, lengths, 512, impl="auto")
```

# qsa_mqa_packed

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.indexer import qsa_mqa_packed

def qsa_mqa_packed(q, k, starts, ends, *, impl="auto"):
    ...
```

## Description

Packed QSA scores: BF16 Q[R,4,128], K[M,1,128], int32 starts/ends[R]. Returns FP32[R,M], sum(ReLU(QK))/sqrt(128), with -inf outside each interval. auto selects Triton.

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output. `auto` aliases `triton`, including its native PyTorch substeps. No error-driven fallback is installed. `torch_native` is an independent reference: short convolution uses native NPU convolution; the other references use CPU arithmetic (FP64 for floating-point oracles) and return to the input device. These references synchronize and are intended for accuracy checks outside graph capture.

`score[r,j] = sum_h max(dot(q[r,h], k[j,0]), 0) / sqrt(128)`. Accumulation and output are FP32.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `q` | Tensor | [R,4,128], BF16 | required | Contiguous NPU query. |
| `k` | Tensor | [M,1,128], BF16 | required | Contiguous NPU compressed keys. |
| `starts / ends` | Tensor | [R], int32 | required | Contiguous on q.device; caller ensures 0 <= starts <= ends <= M. |
| `impl` | str | — | "auto" | Keyword-only: auto, triton, torch_native. |

| Return | Type | Shape | Dtype | Device / layout |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,M] | FP32 | Input device; contiguous, newly allocated. |

## triton

Default inference implementation. Metadata checks reject unsupported inputs before launch; input contents must satisfy the caller constraints above. Supports graph capture and replay with in-place input and weight updates. Native substeps, where present, are part of this implementation.

## torch_native

Independent accuracy reference for the documented input contract. Synchronizing host transfers are permitted here; run this implementation outside graph capture. It is never selected automatically after a failure.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import qsa_mqa_packed

q = torch.randn(1, 4, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn(16, 1, 128, device="npu", dtype=q.dtype)
start = torch.zeros(1, device="npu", dtype=torch.int32)
y = qsa_mqa_packed(q, k, start, start + 16, impl="auto")
```

# expand_qsa_blocks

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.indexer import expand_qsa_blocks

def expand_qsa_blocks(blocks, positions, lengths, ratio, topk, *, impl="auto"):
    ...
```

## Description

Expand int32/int64 blocks[R,K] and positions/lengths[R]. topk=K*ratio, K=512/2048. Returns contiguous int32[R,topk+ratio-1], valid blocks then tail then exact -1 padding. auto selects the Triton/native hybrid.

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output. `auto` aliases `triton`, including its native PyTorch substeps. No error-driven fallback is installed. `torch_native` is an independent reference: short convolution uses native NPU convolution; the other references use CPU arithmetic (FP64 for floating-point oracles) and return to the input device. These references synchronize and are intended for accuracy checks outside graph capture.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `blocks` | Tensor | [R,K], int32/int64 | required | NPU, strided views accepted; K=512/2048; valid prefix then -1. |
| `positions / lengths` | Tensor | [R], int32/int64 | required | Same NPU; query position and sequence length; strided views accepted. |
| `ratio` | int | — | required | At least 2. |
| `topk` | int | — | required | K*ratio. |
| `impl` | str | — | "auto" | Keyword-only: auto, triton, torch_native. |

| Return | Type | Shape | Dtype | Device / layout |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,topk+ratio-1] | int32 | Input device; contiguous, newly allocated. |

## triton

Default inference implementation. Metadata checks reject unsupported inputs before launch; input contents must satisfy the caller constraints above. Supports graph capture and replay with in-place input and weight updates. Native substeps, where present, are part of this implementation.

## torch_native

Independent accuracy reference for the documented input contract. Synchronizing host transfers are permitted here; run this implementation outside graph capture. It is never selected automatically after a failure.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import expand_qsa_blocks

blocks = torch.full((1, 512), -1, device="npu", dtype=torch.int32)
position = torch.zeros(1, device="npu", dtype=torch.int32)
y = expand_qsa_blocks(blocks, position, position + 1, 4, 2048, impl="auto")
```

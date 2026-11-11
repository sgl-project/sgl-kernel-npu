# expand_qsa_blocks

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.indexer import expand_qsa_blocks

def expand_qsa_blocks(blocks, positions, lengths, ratio, topk, *, impl="auto"):
    ...
```

## Description

Expand int32/int64 blocks[R,K] and positions/lengths[R]. topk=K*ratio, K=512/2048. Returns contiguous int32[R,topk+ratio-1], valid blocks then tail then exact -1 padding. auto selects the Triton/native hybrid.

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output.

Each block b expands to `b*ratio + j`, for `j=0..ratio-1`. Keep block order and duplicates, append the incomplete tail block, then pad with -1. For example, with ratio=4, position=12 and length=13, block 2 produces `[8,9,10,11]`, followed by tail token 12. The tail covers `[floor((position+1)/ratio)*ratio, min(position+1,length))`.

Put all valid blocks before the -1 padding. Each block must be complete: `(b+1)*ratio <= min(position+1,length)`. The caller must ensure these values are valid; the operator checks only shapes, dtypes and strides.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `blocks` | Tensor | [R,K], int32/int64 | required | NPU, strided views accepted; K=512/2048; valid prefix then -1. |
| `positions` | Tensor | [R], int32/int64 | required | Same NPU; query position in [-1,2**31-2], below length for active rows; strided views accepted. |
| `lengths` | Tensor | [R], int32/int64 | required | Same NPU; sequence length in [0,2**31-1]; strided views accepted. Empty rows may use position=0, length=0. |
| `ratio` | int | — | required | At least 2. |
| `topk` | int | — | required | K*ratio; output width topk+ratio-1 must be at most 2**31-1024. |
| `impl` | str | — | "auto" | Keyword-only: auto, triton, torch_native. |

| Return | Type | Shape | Dtype | Device / layout |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,topk+ratio-1] | int32 | Input device; contiguous, newly allocated. |

## triton

Default implementation. Checks input metadata before launch. Supports graph capture and replay after in-place input updates.

## torch_native

Independent CPU reference that expands each block and appends the tail tokens. Use outside graph capture. Select explicitly with `impl="torch_native"`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import expand_qsa_blocks

blocks = torch.full((1, 512), -1, device="npu", dtype=torch.int32)
position = torch.zeros(1, device="npu", dtype=torch.int32)
y = expand_qsa_blocks(blocks, position, position + 1, 4, 2048, impl="auto")
```

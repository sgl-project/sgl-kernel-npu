# prepare_pool_key_indexer_inputs

## Hardware support

| Implementation | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.indexer import prepare_pool_key_indexer_inputs

def prepare_pool_key_indexer_inputs(
    query, weights, ends, lengths, tails, table, *, impl="auto"
):
    ...
```

`auto` selects Triton.

## Description

Keep causal CANN pooled-key indexing safe when a request has no complete pool
or a graph replay contains only padding. Append a zero query and weight row,
give the dummy request one pooled key, and raise empty real pool lengths to one.
The six outputs are contiguous copies; inputs stay unchanged.

After CANN, remove the extra output row and call `pack_pool_key_indexer_indices`
with the original positions and query ends. This removes synthetic candidates.

## Parameters and return value

All tensors must share an NPU. `T` is the static token count, `R` the request
count and `M` the page-table row count.

| Parameter | Type / shape / dtype | Default | Meaning |
| --- | --- | --- | --- |
| `query` | Tensor `[T,H,D]`, BF16 | Required | Packed query rows. |
| `weights` | Tensor `[T,H]`, BF16 | Required | Per-head weights. |
| `ends` | Tensor `[R]`, int64 | Required | Cumulative real-query ends; excludes padding. |
| `lengths` | Tensor `[R]`, int64 | Required | Complete pooled-key counts. |
| `tails` | Tensor `[R]`, int64 | Required | Uncompressed tail counts. |
| `table` | Tensor `[M,C]`, int32 | Required | Physical pooled pages; at least R rows and one column. |
| `impl` | str | `"auto"` | `auto`, `triton`, `torch_native`. |

| Return | Type | Meaning |
| --- | --- | --- |
| Six tensors | tuple | Same order and dtypes as inputs; one extra token/request/table row. The final end is T+1. |

Physical cache block zero and each empty history's first block must be readable.
Extra table rows must already contain valid page IDs. Outputs are allocated per
call; independently captured graphs do not share a mutable workspace.

## triton backend

Copies and prepares inputs in one launch; strided inputs are made contiguous.
The kernel is reused from vllm-ascend under its original license.

## torch_native backend

Uses independent PyTorch concatenation, zero filling and length clamping.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import prepare_pool_key_indexer_inputs

q = torch.zeros(2, 32, 128, dtype=torch.bfloat16, device="npu")
w = torch.ones(2, 32, dtype=q.dtype, device=q.device)
ends = torch.tensor([2], dtype=torch.int64, device=q.device)
lengths = torch.zeros_like(ends)
tails = torch.full_like(ends, 2)
table = torch.zeros(1, 1, dtype=torch.int32, device=q.device)
safe = prepare_pool_key_indexer_inputs(q, w, ends, lengths, tails, table, impl="triton")
```

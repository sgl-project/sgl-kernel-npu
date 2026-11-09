# pack_pool_key_indexer_indices

## Hardware support

| Implementation | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.indexer import pack_pool_key_indexer_indices

def pack_pool_key_indexer_indices(
    indices, positions, query_ends, topk, pool_size, *, output_buffer=None, impl="auto"
):
    ...
```

`auto` selects Triton.

## Description

Keep CANN's complete-history candidates, append each query's uncompressed tail,
and fill unused columns and graph-padding rows with -1. Use the original query
ends and positions, after removing the sentinel output row.

Valid rows must form a prefix. CANN is responsible for causal history selection;
this operator preserves its candidate order. Short histories retain only the
real tail, so their artificial pooled key does not reach attention.

## Parameters and return value

All tensors share an NPU and have contiguous columns. Position arithmetic uses
int64; packed indices use int32.

| Parameter | Type / shape / dtype | Default | Meaning |
| --- | --- | --- | --- |
| `indices` | Tensor `[T,C]`, int32 | Required | CANN candidates; enough columns for topk plus the largest tail. |
| `positions` | Tensor `[T]`, int64 | Required | Original token positions. |
| `query_ends` | Tensor `[R]`, int64 | Required | Original cumulative real-query ends; the last entry excludes padding. |
| `topk` | int | Required | Token-level history budget used by CANN. |
| `pool_size` | int | Required | Number of tokens in one complete pool. |
| `output_buffer` | Tensor or None, int32 | `None` | Optional output with at least T rows and sufficient columns. |
| `impl` | str | `"auto"` | `auto`, `triton`, `torch_native`. |

| Return | Type / shape / dtype | Meaning |
| --- | --- | --- |
| Packed indices | Tensor `[T,topk+pool_size-1]`, int32 | A view of indices by default, or of output_buffer when supplied. |

Default operation overwrites indices. A separate output buffer preserves the
input; unused columns in its first T rows become -1, and later rows stay intact.

## triton backend

Runs one program per token, reusing vllm-ascend's original packing kernel.

## torch_native backend

Uses independent PyTorch selection and tail construction with the same mutations.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import pack_pool_key_indexer_indices

indices = torch.full((2, 2051), -1, dtype=torch.int32, device="npu")
positions = torch.tensor([0, 1], dtype=torch.int64, device=indices.device)
ends = torch.tensor([2], dtype=torch.int64, device=indices.device)
packed = pack_pool_key_indexer_indices(indices, positions, ends, 2048, 4, impl="triton")
```

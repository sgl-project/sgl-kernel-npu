# pool_key_indexer

## Hardware support

| Implementation | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `cann_ops_transformer` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.indexer import pool_key_indexer

def pool_key_indexer(
    query,
    pool_key,
    weights,
    pool_tail_k,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    topk=2048,
    pool_size=16,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=3,
    quant_mode=-1,
    return_value=False,
    *,
    impl="auto",
):
    ...
```

`auto` selects `cann_ops_transformer`, loading CANN's public binding on first execution. There is no backend fallback.

## Description

Score pooled keys and select the highest-ranked pools, expand them into token positions, and append tokens from the incomplete tail pool.

Returns the CANN `(indices, values)` tuple unchanged. Indices use int32, with unused positions filled by `-1`. With `return_value=True`, values contain the selected pool scores in float32; otherwise values are an empty Tensor. Scoring, masking and selected-pool ordering are determined by CANN; stable pool ordering is not guaranteed. Inputs are not modified.

## Parameters and return value

| Parameter | Type | Shape | Dtype / location | Default | Description / constraints |
| --- | --- | --- | --- | --- | --- |
| `query` | `torch.Tensor` | `[B,S,H,D]` / `[T,H,D]` | CANN-supported query dtype | Required | BSND / TND query, forwarded without reshaping. |
| `pool_key` | `torch.Tensor` | `[B,P,1,D]` (BSND); layout-dependent for PA | Same as query | Required | Pooled keys; paged layouts require a compatible page table. |
| `weights` | `torch.Tensor` | `[B,S,H]` (BSND) / `[T,H]` (TND) | CANN-supported weight dtype | Required | Per-query, per-head indexer weights. |
| `pool_tail_k` | `torch.Tensor` | `[B]` | CPU/NPU int64 | Required | Incomplete-tail token counts. |
| `actual_seq_q` | `torch.Tensor` or `None` | `[B]` or — | CPU/NPU int64 | `None` | Cumulative packed-query ends. |
| `actual_seq_k` | `torch.Tensor` or `None` | `[B]` or — | CPU/NPU int64 | `None` | Complete pooled-key counts per request. |
| `block_table` | `torch.Tensor` or `None` | `[B,M]` or — | `int32` | `None` | NPU page IDs for paged keys; None forwards an undefined optional tensor. |
| `topk` | `int` | — | — | `2048` | Token-level selection width; CANN validates its range and divisibility by pool_size. |
| `pool_size` | `int` | — | — | `16` | Positive number of tokens represented by each complete pool. |
| `layout_q` | `str` | — | — | `"BSND"` | Query layout: BSND or TND. |
| `layout_k` | `str` | — | — | `"BSND"` | Key layout: BSND, TND or PA_BBND; CANN validates support. |
| `mask_mode` | `int` | — | — | `3` | Forwarded to CANN. |
| `quant_mode` | `int` | — | — | `-1` | Forwarded unchanged; CANN validates supported parameter combinations. |
| `return_value` | `bool` | — | — | `False` | Forwarded to CANN; controls whether values contain selected pool scores. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only: auto, cann_ops_transformer, or torch_native. |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `indices` | `torch.Tensor` | `[B,S,topk+K-1]` (BSND) or `[T,topk+K-1]` (TND) | `torch.int32` | First tuple element on the query device; K is pool_size. |
| `values` | `torch.Tensor` | `[B,S,topk//K]` (BSND), `[T,topk//K]` (TND), or `[0]` | `torch.float32` | Second tuple element; empty when return_value=False, invalid scores are -inf. |

Compute tensors use the query NPU; length metadata must be int64 Tensors on CPU
or NPU. For graph replay, metadata must be contiguous on the query NPU and is
updated in place; all-CPU metadata is captured as constants, so changing its
values requires recapture. For zero-pool or all-padding TND inputs, use
`prepare_pool_key_indexer_inputs` and pack the cropped output with the original
query boundaries.

## cann_ops_transformer backend

Requires CANN 9.2.0. The facade directly imports `cann_ops_transformer.ops.pool_key_indexer`, forwards arguments, and returns its complete tuple without conversion, validation or output selection. CANN determines supported dtypes, layouts, quantization modes and parameter combinations.

`auto` and `cann_ops_transformer` are aliases.

## torch_native backend

An independent FP32 reference for BSND/BSND and TND/{TND, PA_BBND} inputs, runnable
on CPU or NPU. It returns the same tuple structure and honors return_value.
quant_mode is accepted for call compatibility; supplied tensor values are scored
in FP32 without reproducing vendor-specific quantized rounding.
This is a correctness reference, not a serving backend.

## Example

Requires the CANN 9.2 `cann_ops_transformer` package.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import pool_key_indexer

# BSND, one query/head, 16 complete pools and three tail tokens.
query = torch.zeros(1, 1, 1, 128, device="npu", dtype=torch.float16)
query[..., 0] = 1
keys = torch.zeros(1, 16, 1, 128, device=query.device, dtype=query.dtype)
keys[..., 0] = torch.arange(1, 17, device=query.device, dtype=query.dtype).view(1, 16, 1)
weights = torch.ones(1, 1, 1, device=query.device, dtype=query.dtype)
indices, values = pool_key_indexer(
    query,
    keys,
    weights,
    pool_tail_k=torch.tensor([3], dtype=torch.int64, device=query.device),
    topk=16,
    pool_size=4,
    layout_q="BSND",
    layout_k="BSND",
    impl="cann_ops_transformer",
)
assert indices.shape == (1, 1, 19) and indices.dtype == torch.int32
assert values.shape == (0,)
```

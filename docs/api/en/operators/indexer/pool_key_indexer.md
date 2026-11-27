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
    q_descale=None,
    k_descale=None,
    *,
    impl="auto",
):
    ...
```

`auto` selects `cann_ops_transformer` and loads the bundled binding on first execution.

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
| `actual_seq_k` | `torch.Tensor` or `None` | `[B]` or — | CPU/NPU int64 | `None` | Cumulative pooled-key ends for TND, or pooled-key counts per request for PA_BBND. |
| `block_table` | `torch.Tensor` or `None` | `[B,M]` or — | `int32` | `None` | NPU page IDs for paged keys; None forwards an undefined optional tensor. |
| `topk` | `int` | — | — | `2048` | Token-level selection width; CANN validates its range and divisibility by pool_size. |
| `pool_size` | `int` | — | — | `16` | Positive number of tokens represented by each complete pool. |
| `layout_q` | `str` | — | — | `"BSND"` | Query layout: BSND or TND. |
| `layout_k` | `str` | — | — | `"BSND"` | Key layout: BSND, TND or PA_BBND; CANN validates support. |
| `mask_mode` | `int` | — | — | `3` | Forwarded to CANN. |
| `quant_mode` | `int` | — | — | `-1` | `-1` for unquantized inputs, `0` for ordinary FP8, and `2` for INT8. |
| `q_descale` | `torch.Tensor` or `None` | `query.shape[:-1]` | FP32 for ordinary FP8, FP16 for INT8 | `None` | One dequantization scale per query token and head. Required in ordinary FP8 and INT8 modes. |
| `k_descale` | `torch.Tensor` or `None` | `pool_key.shape[:-1]` | FP32 for ordinary FP8, FP16 for INT8 | `None` | One dequantization scale per pooled-key row and head. Required in ordinary FP8 and INT8 modes. |
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

Requires CANN 9.2.0. The sgl-kernel-npu wheel bundles the ops-transformer binding and vendor. The binding forwards the arguments and returns the complete CANN result tuple.

`auto` and `cann_ops_transformer` are aliases.

INT8 mode takes INT8 query and pooled-key tensors, FP16 weights, and both
FP16 descale tensors. For `TND/PA_BBND`, `q_descale` is shaped `[T, H]`
and `k_descale` is shaped `[block_num, block_size, N2]` with the same page
ordering as `pool_key`. INT8 queries and keys use the original basis.

## torch_native backend

An independent FP32 reference for BSND/BSND and TND/{TND, PA_BBND} inputs, runnable
on CPU or NPU. It returns the same tuple structure and honors return_value.
For ordinary FP8 and INT8, the reference multiplies each query and pooled-key
vector by its corresponding scale in FP32 before computing scores.

## Example

This example uses INT8 with paged keys on A3 and the CANN 9.2.0 wheel.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import pool_key_indexer

query = torch.zeros(4, 32, 128, dtype=torch.int8, device="npu")
query[..., 0] = 1
keys = torch.zeros(4, 16, 1, 128, dtype=torch.int8, device=query.device)
keys[..., 0] = torch.arange(1, 17, dtype=torch.int8, device=query.device).view(1, 16, 1)
weights = torch.ones(4, 32, dtype=torch.float16, device=query.device)
q_descale = torch.full((4, 32), 0.5, dtype=torch.float16, device=query.device)
k_descale = torch.full((4, 16, 1), 0.25, dtype=torch.float16, device=query.device)
indices, values = pool_key_indexer(
    query,
    keys,
    weights,
    pool_tail_k=torch.full((4,), 3, dtype=torch.int64, device=query.device),
    actual_seq_q=torch.arange(1, 5, dtype=torch.int64, device=query.device),
    actual_seq_k=torch.full((4,), 16, dtype=torch.int64, device=query.device),
    block_table=torch.arange(4, dtype=torch.int32, device=query.device).view(4, 1),
    topk=16,
    pool_size=4,
    layout_q="TND",
    layout_k="PA_BBND",
    quant_mode=2,
    q_descale=q_descale,
    k_descale=k_descale,
)
assert indices.shape == (4, 19) and indices.dtype == torch.int32
assert values.shape == (0,)
```

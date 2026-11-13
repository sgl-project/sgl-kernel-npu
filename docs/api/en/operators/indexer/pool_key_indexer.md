# pool_key_indexer

## Hardware support

| Implementation | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `cann_ops_transformer` | Supported | Supported |

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

The output contains int32 indices rather than scores, with unused tail positions filled by `-1`. Scoring, masking and selected-pool ordering are determined by the vendor operator; stable pool ordering is not guaranteed. Inputs are not modified.

## Parameters and return value

| Parameter | Type | Shape | Dtype / location | Default | Description / constraints |
| --- | --- | --- | --- | --- | --- |
| `query` | `torch.Tensor` | `[B,S,H,D]` / `[T,H,D]` / `[T,H*D]` | `float16` / `bfloat16` | Required | BSND / TND query; rank-2 input is forwarded without reshaping. |
| `pool_key` | `torch.Tensor` | `[B,P,1,D]` (BSND); layout-dependent for PA | Same as query | Required | Pooled keys; paged layouts require a compatible page table. |
| `weights` | `torch.Tensor` | `[B,S,H]` (BSND) / `[T,H]` (TND) | Same as query | Required | Per-query, per-head indexer weights. |
| `pool_tail_k` | `Sequence[int]` or Tensor | `[B]` | Host integers or NPU int64 | Required | Incomplete-tail token counts. |
| `actual_seq_q` | `Sequence[int]`, Tensor or None | `[B]` or — | Host integers or NPU int64 | `None` | Cumulative packed-query ends. |
| `actual_seq_k` | `Sequence[int]`, Tensor or None | `[B]` or — | Host integers or NPU int64 | `None` | Complete pooled-key counts per request. |
| `block_table` | `torch.Tensor` or `None` | `[B,M]` or — | `int32` | `None` | NPU page IDs for paged keys; None forwards an undefined optional tensor. |
| `topk` | `int` | — | — | `2048` | Token-level selection width, at least pool_size; use a multiple of pool_size and enough valid pools. |
| `pool_size` | `int` | — | — | `16` | Positive number of tokens represented by each complete pool. |
| `layout_q` | `str` | — | — | `"BSND"` | Query layout; examples cover BSND and TND. |
| `layout_k` | `str` | — | — | `"BSND"` | Key layout; test configurations use BSND or PA_BBND. |
| `mask_mode` | `int` | — | — | `3` | Vendor mask-mode attribute; this integration uses mode 3. |
| `quant_mode` | `int` | — | — | `-1` | Only -1 is supported; no descale tensors are exposed. |
| `return_value` | `bool` | — | — | `False` | Must be False: the schema returns indices only. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only: auto, cann_ops_transformer, or torch_native. |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `indices` | `torch.Tensor` | `[B,S,topk+K-1]` (rank-4 query) or `[T,topk+K-1]` (rank-2/3 query) | `torch.int32` | New output on the query NPU. |

All tensors share an NPU and use contiguous ND storage matching the layout.
Host metadata sequences are converted to CPU int64 tensors. Device metadata can
be updated during graph replay.
For zero-pool or all-padding TND inputs, use `prepare_pool_key_indexer_inputs`
before CANN and pack the cropped output with the original query boundaries.

## cann_ops_transformer backend

Requires CANN 9.2.0. The facade imports and calls the public `cann_ops_transformer.ops.pool_key_indexer` wrapper, returning its sparse-index output. The operator and its upstream Torch binding are provided by CANN 9.2 for arch22 and arch35; they are not rebuilt into `sgl_ops_transformer`. CANN 9.1 is unsupported and the separate `csrc/attentions` extension is not required. Host integer sequences are converted to CPU int64 tensors. Rank-2/3 queries are forwarded without conversion to BSND.

Only BF16/FP16, `quant_mode=-1` and `return_value=False` are supported.
`torch_native` is an independent FP32 reference for BSND/BSND and TND/PA_BBND inputs; it is not a serving backend. `auto` and `cann_ops_transformer` are aliases. Tests compare the selected token set (CANN does not guarantee selected-pool order) and the ordered incomplete tail against this reference.

## Example

Requires the CANN 9.2 `cann_ops_transformer` package. This is a minimal BSND invocation, not evidence of completed NPU acceptance.

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
indices = pool_key_indexer(
    query, keys, weights, pool_tail_k=[3],
    topk=16, pool_size=4, layout_q="BSND", layout_k="BSND", impl="cann_ops_transformer",
)
assert indices.shape == (1, 1, 19) and indices.dtype == torch.int32
```

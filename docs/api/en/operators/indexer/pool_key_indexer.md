# pool_key_indexer

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `aclnn` | Unverified | Verified |

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
    key_stride0=-1,
    *,
    impl="auto",
):
    ...
```

`auto` selects `aclnn`, loading the `attentions` extension on first execution. There is no backend fallback.

## Description

Score pooled keys and select the highest-ranked pools, expand them into token positions, and append tokens from the incomplete tail pool.

The output contains int32 indices rather than scores, with unused tail positions filled by `-1`. Scoring, masking and selected-pool ordering are determined by the vendor operator; stable pool ordering is not guaranteed. Inputs are not modified.

## Parameters and return value

| Parameter | Type | Shape | Dtype / location | Default | Description / constraints |
| --- | --- | --- | --- | --- | --- |
| `query` | `torch.Tensor` | `[B,S,H,D]` / `[T,H,D]` / `[T,H*D]` | `float16` / `bfloat16` | Required | BSND / TND query; rank-2 input is forwarded without reshaping. |
| `pool_key` | `torch.Tensor` | `[B,P,1,D]` (BSND); layout-dependent for PA | Same as query | Required | Pooled keys; paged layouts require a compatible page table. |
| `weights` | `torch.Tensor` | `[B,S,H]` (BSND) / `[T,H]` (TND) | Same as query | Required | Per-query, per-head indexer weights. |
| `pool_tail_k` | `Sequence[int]` | `[B]` | Host integers | Required | Number of incomplete-tail tokens per request, in `[0,pool_size)`; pass a host list/tuple, never an NPU tensor. |
| `actual_seq_q` | `Sequence[int]` or `None` | `[B]` or — | Host integers | `None` | Cumulative query endpoints for packed TND; None forwards an empty aclIntArray. |
| `actual_seq_k` | `Sequence[int]` or `None` | `[B]` or — | Host integers | `None` | Valid pooled-key lengths per request for the paged workload; None forwards an empty aclIntArray. |
| `block_table` | `torch.Tensor` or `None` | `[B,M]` or — | `int32` | `None` | NPU page IDs for paged keys; None forwards an undefined optional tensor. |
| `topk` | `int` | — | — | `2048` | Token-level selection width, at least pool_size; use a multiple of pool_size and enough valid pools. |
| `pool_size` | `int` | — | — | `16` | Positive number of tokens represented by each complete pool. |
| `layout_q` | `str` | — | — | `"BSND"` | Query layout; examples cover BSND and TND. |
| `layout_k` | `str` | — | — | `"BSND"` | Key layout; test configurations use BSND or PA_BBND. |
| `mask_mode` | `int` | — | — | `3` | Vendor mask-mode attribute; this integration uses mode 3. |
| `quant_mode` | `int` | — | — | `-1` | Only -1 is supported; no descale tensors are exposed. |
| `return_value` | `bool` | — | — | `False` | Must be False: the schema returns indices only. |
| `key_stride0` | `int` | — | — | `-1` | Vendor key-stride override; -1 leaves it unspecified. No Python-side inference. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only: auto or aclnn. |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `indices` | `torch.Tensor` | `[B,S,topk+K-1]` (rank-4 query) or `[T,topk+K-1]` (rank-2/3 query) | `torch.int32` | New output on the query NPU. |

All tensors must share an NPU and use contiguous ND storage matching the layout; lengths and `pool_tail_k` remain host integer sequences. The binding accepts `TND*`, `PA_*` and `BSND*` prefixes, which does not establish vendor support for every suffix, shape or stride.

## aclnn backend

The facade calls `torch.ops.attentions.pool_key_indexer`; the C++ binding forwards to `aclnnPoolKeyIndexer`. Build/install the separate `csrc/attentions` extension and provide a CANN/custom-op environment containing the vendor operator; the main kernel package alone is insufficient. Rank-2/3 queries are forwarded without conversion to BSND.

Only BF16/FP16, `quant_mode=-1` and `return_value=False` are supported. No independent `torch_native` backend exists: `auto` and `aclnn` are aliases and cannot serve as mutual accuracy references. Tests use an independent deterministic ranking fixture; hardware accuracy and performance acceptance of this binding remain unverified.

## Example

Requires the extension and vendor dependencies above. This is a minimal BSND invocation, not evidence of completed NPU acceptance.

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
    topk=16, pool_size=4, layout_q="BSND", layout_k="BSND", impl="aclnn",
)
assert indices.shape == (1, 1, 19) and indices.dtype == torch.int32
```

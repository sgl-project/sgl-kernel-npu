# kpool_write_tail_and_maybe_compress_npu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import (
    kpool_write_tail_and_maybe_compress_npu,
)

def kpool_write_tail_and_maybe_compress_npu(
    index_k_cache,
    key,
    score,
    tail_k,
    tail_score,
    ape,
    req_pool_indices,
    write_start,
    tail_logical_start,
    write_loc,
    out_cache_loc,
    num_draft_tokens,
    effective_n_per_batch=None,
    *,
    num_programs=None,
    impl="auto",
):
    ...
```

## Description

Writes draft-token keys and scores to request-local tail rings and compresses newly completed pools. Input rows are grouped as `[B, N, 128]`, where `N = num_draft_tokens`. For each non-skipped batch item, all `N` rows are written to its tail ring.

Pool boundaries crossed by `write_start + effective_n` are reduced with a dimension-wise softmax over `tail_score + ape`, and the resulting BF16 keys are written to the cache rows in `write_loc`. If `effective_n_per_batch` is absent, `effective_n = N`; when provided, it affects pool-completion decisions only, not the number of tail rows written.

## Parameters

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `index_k_cache` | `torch.Tensor` | Contiguous storage flattenable to `[cache_rows, 128]` | `torch.bfloat16` | Required | Cache updated in place for completed pools. |
| `key` | `torch.Tensor` | `[B * N, 128]` | `torch.bfloat16` | Required | Draft keys. |
| `score` | `torch.Tensor` | Same as `key` | `torch.float16`, `torch.bfloat16`, or `torch.float32` | Required | Scores paired with `key`. |
| `tail_k` | `torch.Tensor` | `[R, pool_size + N, 128]` | `torch.bfloat16` | Required | Tail rings updated in place. |
| `tail_score` | `torch.Tensor` | Same as `tail_k` | `torch.float16`, `torch.bfloat16`, or `torch.float32` | Required | Score rings updated in place. |
| `ape` | `torch.Tensor` | `[pool_size, 128]` | `torch.float32` | Required | Additive pooling-position scores; `pool_size > 0`. |
| `req_pool_indices` | `torch.Tensor` | `[>= B]` | `torch.int64` | Required | Request/ring row for each batch item. |
| `write_start` | `torch.Tensor` | `[>= B]` | `torch.int32` or `torch.int64` | Required | Logical tail position at which each batch starts. |
| `tail_logical_start` | `torch.Tensor` | `[>= B]` | `torch.int32` or `torch.int64` | Required | Logical start of the candidate completed pool. |
| `write_loc` | `torch.Tensor` | `[B, ceil(N / pool_size)]` | `torch.int64` | Required | Candidate cache rows; read only and must be contiguous. |
| `out_cache_loc` | `torch.Tensor` | `[>= B * N]` | `torch.int32` or `torch.int64` | Required | Token cache locations; a zero first entry for a batch skips that whole batch. Read only. |
| `num_draft_tokens` | `int` | — | — | Required | Static `N`; must be positive and divide `key.shape[0]`. |
| `effective_n_per_batch` | `torch.Tensor` or `None` | `[>= B]` or — | `torch.int32` or `torch.int64` | `None` | Optional effective draft count used only for pool-completion decisions. |
| `num_programs` | `int` or `None` | — | — | `None` | Keyword-only Triton grid size; positive when supplied and capped at `B`. Defaults to the available Vector Core count capped at `B`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"` or `"triton"`. |

## Return value and in-place state

Returns `None`. `index_k_cache`, `tail_k`, and `tail_score` are mutable; the other arguments are read only. An empty `key` tensor returns without launching the kernel.

All tensors must be on the same NPU, and the final kernel inputs must be contiguous. Mutable state must already be contiguous. The backend makes the read-only forward and plan tensors contiguous internally. Plan values and source/cache locations must be valid upstream.

## triton backend

- Uses a fixed channel width of 128. Pooling accumulates in FP32 and writes BF16 without Hadamard or FP8 processing.
- `auto` aliases `triton`, without dynamic selection or fallback. Registry import still performs ACL architecture detection.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import (
    kpool_write_tail_and_maybe_compress_npu,
)

b, n, pool_size, h = 1, 2, 2, 128
cache = torch.zeros(1, h, device="npu", dtype=torch.bfloat16)
key = torch.randn(b * n, h, device="npu", dtype=torch.bfloat16)
score = torch.zeros(b * n, h, device="npu", dtype=torch.float32)
tail_k = torch.zeros(1, pool_size + n, h, device="npu", dtype=torch.bfloat16)
tail_score = torch.zeros(1, pool_size + n, h, device="npu", dtype=torch.float32)

kpool_write_tail_and_maybe_compress_npu(
    cache,
    key,
    score,
    tail_k,
    tail_score,
    ape=torch.zeros(pool_size, h, device="npu", dtype=torch.float32),
    req_pool_indices=torch.tensor([0], device="npu", dtype=torch.int64),
    write_start=torch.tensor([0], device="npu", dtype=torch.int32),
    tail_logical_start=torch.tensor([0], device="npu", dtype=torch.int32),
    write_loc=torch.tensor([[0]], device="npu", dtype=torch.int64),
    out_cache_loc=torch.ones(b * n, device="npu", dtype=torch.int32),
    num_draft_tokens=n,
)
```

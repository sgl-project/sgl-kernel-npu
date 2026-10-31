# kpool_write_tail_and_maybe_compress

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Verified |
| `torch_native` | Supported | Verified |

## Signature

```python
from typing import Optional
import torch
from sgl_kernel_npu.operators.mem_cache import kpool_write_tail_and_maybe_compress

def kpool_write_tail_and_maybe_compress(
    index_k_cache: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    ape: torch.Tensor,
    req_pool_indices: torch.Tensor,
    write_start: torch.Tensor,
    tail_logical_start: torch.Tensor,
    write_loc: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft_tokens: int,
    effective_n_per_batch: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None,
    impl="auto",
) -> None:
    ...
```

`auto` selects `triton`; omitting impl has the same effect. There is no runtime backend fallback.

## Description

Write a batch of draft keys and scores into request-local tail rings. Compress newly completed pools with channel-wise softmax pooling and write the BF16 results to the cache locations provided in `write_loc`.

Pooling uses FP32 arithmetic and includes positional scores. Effective draft counts determine which pools are complete; all supplied draft rows are still written to the tail for valid batches. Invalid batches are skipped. The cache, tail keys and tail scores are updated in place.

## Parameters and return value

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
| `num_programs` | `int` or `None` | — | — | `None` | Optional execution-parallelism hint; must be positive when supplied. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `auto`, `triton`, `torch_native` |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | Mutation scope is specified above. |

All tensors must be on the same NPU. Callers supply valid indices, sufficient storage, and initialized history, and prevent conflicting state writes; wrappers do not validate every value range.

## triton backend

Channel width is 128. Cache, tails and write_loc must be contiguous; other read-only inputs may be made contiguous internally.

## torch_native backend

Independently writes tails per batch and performs FP32 softmax. An effective count of zero still writes all draft tail rows. num_programs has no effect on this backend.

## Example

Requires sgl-kernel-npu, torch_npu and an NPU runtime. The example explicitly selects `torch_native`. `triton` and `auto` are also available; isolate mutable state when comparing backends.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kpool_write_tail_and_maybe_compress
b, n, pool_size, h = (1, 2, 2, 128)
cache = torch.zeros(1, h, device='npu', dtype=torch.bfloat16)
key = torch.randn(b * n, h, device='npu', dtype=torch.bfloat16)
score = torch.zeros(b * n, h, device='npu', dtype=torch.float32)
tail_k = torch.zeros(1, pool_size + n, h, device='npu', dtype=torch.bfloat16)
tail_score = torch.zeros(1, pool_size + n, h, device='npu', dtype=torch.float32)
kpool_write_tail_and_maybe_compress(
    cache,
    key,
    score,
    tail_k,
    tail_score,
    ape=torch.zeros(pool_size, h, device='npu', dtype=torch.float32),
    req_pool_indices=torch.tensor([0], device='npu', dtype=torch.int64),
    write_start=torch.tensor([0], device='npu', dtype=torch.int32),
    tail_logical_start=torch.tensor([0], device='npu', dtype=torch.int32),
    write_loc=torch.tensor([[0]], device='npu', dtype=torch.int64),
    out_cache_loc=torch.ones(b * n, device='npu', dtype=torch.int32),
    num_draft_tokens=n,
    impl='torch_native',
)
```
